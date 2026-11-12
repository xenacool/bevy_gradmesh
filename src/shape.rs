//! Procedural mesh primitives with configurable detail levels.
//!
//! Each primitive generates watertight geometry with proper normals and UVs.

use bevy::color::{Color, ColorToComponents};
use bevy::math::{Rect, UVec2, Vec2, Vec3};
use bevy::render::mesh::{Mesh, PrimitiveTopology};
use bevy::transform::components::Transform;
use bevy_asset::RenderAssetUsages;

use crate::geometry;
use std::f32::consts::PI;
use std::hash::Hash;
use std::sync::Arc;

/// Mesh detail level controlling segment count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ShapeDetail {
    /// Very low poly (8 segments)
    Minimal,
    /// Low poly (12 segments)
    Low,
    #[default]
    /// Medium poly (16 segments)
    Medium,
    /// High poly (24 segments)
    High,
    /// Very high poly (32 segments)
    Extreme,
}

impl ShapeDetail {
    /// Returns the segment count for this detail level.
    pub fn segments(self) -> u32 {
        match self {
            ShapeDetail::Minimal => 8,
            ShapeDetail::Low => 12,
            ShapeDetail::Medium => 16,
            ShapeDetail::High => 24,
            ShapeDetail::Extreme => 32,
        }
    }
}

/// Descriptor for a procedural primitive shape.
#[derive(Debug, Clone)]
pub struct Shape {
    kind: ShapeKind,
    detail: ShapeDetail,
}

impl PartialEq for Shape {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind && self.detail == other.detail
    }
}

impl Eq for Shape {}

impl std::hash::Hash for Shape {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
        self.detail.hash(state);
    }
}

/// Axis-aligned bounding box of a shape's geometry, in the shape's local space.
///
/// Returned by [`Shape::bounds`] and [`Shape::bounds_2d`]. Empty shapes (for
/// example a polyline with fewer than two points) have a zero-sized box at the
/// origin.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ShapeBounds {
    /// Smallest corner.
    pub min: Vec3,
    /// Largest corner.
    pub max: Vec3,
}

impl From<ShapeBounds> for (Vec3, Vec3) {
    fn from(b: ShapeBounds) -> Self {
        (b.min, b.max)
    }
}

impl ShapeBounds {
    /// Bounds of a set of points; a zero box at the origin when empty.
    pub fn from_points(points: impl IntoIterator<Item = Vec3>) -> Self {
        let mut it = points.into_iter();
        let Some(first) = it.next() else {
            return Self::default();
        };
        let (min, max) = it.fold((first, first), |(lo, hi), p| (lo.min(p), hi.max(p)));
        Self { min, max }
    }

    /// Midpoint of the box.
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Full extent along each axis.
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    /// Half of [`ShapeBounds::size`].
    pub fn half_extents(&self) -> Vec3 {
        self.size() * 0.5
    }

    /// Footprint on the XY plane, for flat shapes and `Mesh2d` layout.
    pub fn rect_xy(&self) -> Rect {
        Rect::from_corners(self.min.truncate(), self.max.truncate())
    }
}

/// One piece of a [`Shape::compound`]: a shape placed by a transform, with an
/// optional vertex tint.
///
/// The tint multiplies the part's gradient colour (via vertex colours), so every
/// part keeps its own gradient while the compound stays a single mesh.
#[derive(Debug, Clone)]
pub struct ShapePart {
    pub shape: Shape,
    pub transform: Transform,
    pub tint: Option<Color>,
}

impl ShapePart {
    /// A part at the origin without tint.
    pub fn new(shape: Shape) -> Self {
        Self {
            shape,
            transform: Transform::IDENTITY,
            tint: None,
        }
    }

    /// Places the part with `transform`.
    #[must_use]
    pub fn with_transform(mut self, transform: Transform) -> Self {
        self.transform = transform;
        self
    }

    /// Multiplies the part's colour by `tint`.
    #[must_use]
    pub fn with_tint(mut self, tint: Color) -> Self {
        self.tint = Some(tint);
        self
    }

    fn tint_bits(&self) -> Option<[u32; 4]> {
        self.tint.map(|c| c.to_srgba().to_f32_array().map(f32::to_bits))
    }

    fn transform_bits(&self) -> [u32; 10] {
        let t = &self.transform;
        let r = t.rotation.to_array();
        [
            t.translation.x.to_bits(),
            t.translation.y.to_bits(),
            t.translation.z.to_bits(),
            r[0].to_bits(),
            r[1].to_bits(),
            r[2].to_bits(),
            r[3].to_bits(),
            t.scale.x.to_bits(),
            t.scale.y.to_bits(),
            t.scale.z.to_bits(),
        ]
    }
}

impl PartialEq for ShapePart {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape
            && self.transform_bits() == other.transform_bits()
            && self.tint_bits() == other.tint_bits()
    }
}

impl Eq for ShapePart {}

impl std::hash::Hash for ShapePart {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.shape.hash(state);
        self.transform_bits().hash(state);
        self.tint_bits().hash(state);
    }
}

/// Heights of the samples just outside a heightfield chunk, used for the
/// normals of its edge vertices and never meshed. See
/// [`HeightfieldOptions::border`].
///
/// For a chunk of `size` samples: `west` is the column left of column 0
/// (`size.y` heights, one per row), `east` the column right of the last one
/// (`size.y` heights), `north` the row before row 0 (`size.x` heights, one per
/// column; rows run along +Z, so this is the -Z side) and `south` the row
/// after the last one (`size.x` heights). The corners are not needed.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct HeightfieldBorder {
    /// Heights at `x = -1`, one per row.
    pub west: Vec<f32>,
    /// Heights at `x = size.x`, one per row.
    pub east: Vec<f32>,
    /// Heights at `y = -1`, one per column.
    pub north: Vec<f32>,
    /// Heights at `y = size.y`, one per column.
    pub south: Vec<f32>,
}

impl HeightfieldBorder {
    /// A border from its four sides; see the type for their lengths.
    pub fn new(west: Vec<f32>, east: Vec<f32>, north: Vec<f32>, south: Vec<f32>) -> Self {
        Self {
            west,
            east,
            north,
            south,
        }
    }

    fn identity_bits(&self) -> Vec<u32> {
        let mut v = Vec::with_capacity(4 + self.west.len() * 2 + self.north.len() * 2);
        for side in [&self.west, &self.east, &self.north, &self.south] {
            v.push(side.len() as u32);
            v.extend(side.iter().map(|h| h.to_bits()));
        }
        v
    }
}

/// How the outline of a heightfield hole is cut, from the cheapest to the
/// smoothest. See [`HeightfieldOptions::hole_edge`].
///
/// The hole mask has one flag per cell, so the outline of a hole can only follow
/// cell borders: a round hole comes out as a staircase. The two other levels
/// cut the cells along the outline instead. No level ever removes a hole or an
/// island of land: a one-cell hole stays a hole (a diamond, or a small round
/// opening) and a one-cell island stays land.
///
/// Only cells that border a hole cost more. Cells away from every hole keep two
/// triangles at every level.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum HoleEdge {
    /// Whole cells: a cell is land or hole. Two triangles per land cell, the
    /// cheapest, and a staircase along every curved outline.
    #[default]
    Cells,
    /// Cells that border a hole are cut through the midpoints of their edges,
    /// so every staircase corner becomes a 45 degree chamfer and a one-cell
    /// hole a diamond. Up to four triangles (and the cut) per border cell.
    Rounded,
    /// The mask is blurred over the neighbouring cells and the outline follows
    /// that, so a curved hole is cut along a curve, not along a chamfered
    /// staircase. Same triangle count as [`HoleEdge::Rounded`], more border
    /// cells, and the outline of a hole can move by up to about a cell.
    Smooth,
}

/// Options for [`Shape::heightfield_with_options`].
///
/// `#[non_exhaustive]`: start from [`HeightfieldOptions::default`] and use the
/// `with_*` methods.
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct HeightfieldOptions {
    /// One flag per cell, `(size.x - 1) * (size.y - 1)` of them, `holes[cx + cy *
    /// (size.x - 1)]` for the cell between columns `cx` and `cx + 1` and rows
    /// `cy` and `cy + 1`. A hole has no triangles; the terrain is see-through
    /// there. Hole rims get no skirt (a skirt only hides seams between chunks),
    /// and vertices that touch only holes are not in the mesh.
    pub holes: Option<Vec<bool>>,
    /// How the outline of the holes is cut: whole cells (the default, a
    /// staircase), rounded corners or a smoothed outline. See [`HoleEdge`].
    /// Without holes this changes nothing.
    pub hole_edge: HoleEdge,
    /// Neighbour heights for the normals of the edge vertices, so that chunks
    /// that share an edge also share its normals exactly. See
    /// [`HeightfieldBorder`].
    pub border: Option<HeightfieldBorder>,
}

impl HeightfieldOptions {
    /// Sets [`HeightfieldOptions::holes`].
    #[must_use]
    pub fn with_holes(mut self, holes: Vec<bool>) -> Self {
        self.holes = Some(holes);
        self
    }

    /// Sets [`HeightfieldOptions::hole_edge`].
    #[must_use]
    pub fn with_hole_edge(mut self, hole_edge: HoleEdge) -> Self {
        self.hole_edge = hole_edge;
        self
    }

    /// Sets [`HeightfieldOptions::border`].
    #[must_use]
    pub fn with_border(mut self, border: HeightfieldBorder) -> Self {
        self.border = Some(border);
        self
    }
}

/// Options for [`Shape::sweep`].
///
/// Everything here is plain data (no closures), so a swept [`Shape`] stays
/// `Clone + Eq + Hash` and can be cached like any other shape. Floats compare
/// and hash by their bits.
///
/// The struct is `#[non_exhaustive]`: start from [`SweepOptions::default`] and
/// use the `with_*` methods, so new options never break existing code.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct SweepOptions {
    /// Joins the last path point back to the first. Closed sweeps have no caps
    /// and no seam: a fractional [`SweepOptions::twist`] is rounded to the
    /// nearest whole turn, and a [`SweepOptions::roll_along`] that does not
    /// return to its starting angle (modulo a whole turn) has the difference
    /// spread evenly over the loop.
    pub closed: bool,
    /// Closes the two open ends of an open sweep with flat caps.
    pub caps: bool,
    /// The up direction the first frame starts from; later frames follow by
    /// parallel transport (the same rule as
    /// [`CatmullRomSpline::bake`](crate::CatmullRomSpline::bake)).
    pub up: Vec3,
    /// Profile scale along the path as `(ratio, scale)` keyframes, with `ratio`
    /// running `0.0..=1.0` over the path length and linear steps in between.
    /// `scale.x` stretches across the path and `scale.y` upwards. See
    /// [`SweepOptions::scale_with`] to fill this from a closure.
    pub scale_along: Option<Vec<(f32, Vec2)>>,
    /// Turns of the profile around the path over its whole length.
    pub twist: f32,
    /// Roll of the profile about the path tangent as `(ratio, radians)`
    /// keyframes, with the same rules as [`SweepOptions::scale_along`]: `ratio`
    /// runs `0.0..=1.0` over the path length, steps are linear and the values
    /// are held beyond the first and last key. Roll turns the (already scaled)
    /// profile on top of the parallel-transported frame, or of the frames given
    /// to [`Shape::sweep_frames`], and adds to [`SweepOptions::twist`]. See
    /// [`SweepOptions::roll_with`] to fill this from a closure.
    pub roll_along: Option<Vec<(f32, f32)>>,
    /// With twist, scale or tint, long path segments are split into
    /// `detail / 4` pieces so the change looks smooth.
    pub detail: ShapeDetail,
    /// Vertex tint along the path as `(ratio, colour)` keyframes (linear steps in
    /// between). It multiplies the style's gradient colour.
    pub tint_by_length: Option<Vec<(f32, Color)>>,
    /// Gates and windows cut through the swept body. See [`SweepOpening`].
    pub openings: Vec<SweepOpening>,
}

/// One cross-section of a sweep given by the caller: see [`Shape::sweep_frames`].
///
/// `right` and `up` are the directions the profile's `x` and `y` run along at
/// `position`; `scale` stretches the profile across (`x`) and upwards (`y`).
/// `right x up` should point along the path. The sweep re-squares the frame
/// against the direction of the path at the joint: a left-handed frame keeps
/// its `up` and gets its `right` flipped, so the surface never turns inside
/// out.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct SweepFrame {
    /// Where the cross-section sits.
    pub position: Vec3,
    /// Direction of the profile's `x` axis.
    pub right: Vec3,
    /// Direction of the profile's `y` axis.
    pub up: Vec3,
    /// Profile scale across (`x`) and upwards (`y`).
    pub scale: Vec2,
}

impl SweepFrame {
    /// A frame with unit scale.
    pub fn new(position: Vec3, right: Vec3, up: Vec3) -> Self {
        Self {
            position,
            right,
            up,
            scale: Vec2::ONE,
        }
    }

    /// Sets the profile scale of the frame.
    #[must_use]
    pub fn with_scale(mut self, scale: Vec2) -> Self {
        self.scale = scale;
        self
    }

    fn identity_bits(&self) -> [u32; 11] {
        let f = |v: f32| v.to_bits();
        [
            f(self.position.x),
            f(self.position.y),
            f(self.position.z),
            f(self.right.x),
            f(self.right.y),
            f(self.right.z),
            f(self.up.x),
            f(self.up.y),
            f(self.up.z),
            f(self.scale.x),
            f(self.scale.y),
        ]
    }
}

/// A hole through a swept wall: a gate or a window.
///
/// `outline` lives in wall-face coordinates: `x` runs `0.0..=1.0` over the
/// opening's `t` range along the path and `y` is the height above the profile
/// origin, in profile units (before any [`SweepOptions::scale_along`]). The hole
/// goes through the whole thickness of the profile, with reveal faces (sill,
/// lintel and jambs) on the inside.
///
/// Rules:
/// * The outline should be convex, or at least cross every vertical line twice
///   at most (rectangles, arches, circles and trapezoids all qualify); the hole
///   spans the full height range of the outline at each position.
/// * A gate is an outline that reaches the profile's lowest height: the wall
///   (and its bottom face) is cut open there. Outlines that reach the top cut
///   the top as well.
/// * Outlines that leave the wall are clipped: `x` is clamped to `0..=1` and
///   heights beyond the profile just open the wall. On an open path, openings
///   are kept 0.1% of the path length away from its ends.
/// * Openings cannot overlap along the path; a later one that does is ignored
///   with a warning, like one with fewer than three outline points.
/// * Openings expect profiles whose horizontal cross-section is one interval
///   (walls, rails, kerbs), and are meant for sweeps without twist.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct SweepOpening {
    /// The stretch of the path the opening's `x` axis maps onto, as ratios of
    /// the path length.
    pub t: std::ops::Range<f32>,
    /// The hole's outline, `x` across the `t` range and `y` as height.
    pub outline: Vec<Vec2>,
}

impl SweepOpening {
    /// An opening with a custom `outline` over the `t` range of the path (see
    /// the type for the coordinates and rules).
    pub fn new(t: std::ops::Range<f32>, outline: Vec<Vec2>) -> Self {
        Self { t, outline }
    }

    /// A rectangle spanning `x` of its range and heights `y_min..y_max`.
    pub fn rect(t: std::ops::Range<f32>, y_min: f32, y_max: f32) -> Self {
        Self {
            t,
            outline: vec![
                Vec2::new(0.0, y_min),
                Vec2::new(1.0, y_min),
                Vec2::new(1.0, y_max),
                Vec2::new(0.0, y_max),
            ],
        }
    }

    /// A gate: a rectangle from `y_min` up to `spring`, topped by a round arch
    /// of `segments` pieces that peaks at `y_max`.
    pub fn arch(t: std::ops::Range<f32>, y_min: f32, spring: f32, y_max: f32, segments: u32) -> Self {
        let segments = segments.max(2);
        let mut outline = vec![Vec2::new(0.0, y_min), Vec2::new(1.0, y_min)];
        for i in 0..=segments {
            let a = i as f32 / segments as f32 * std::f32::consts::PI;
            outline.push(Vec2::new(
                0.5 + 0.5 * a.cos(),
                spring + (y_max - spring) * a.sin(),
            ));
        }
        Self { t, outline }
    }
}

impl Default for SweepOptions {
    fn default() -> Self {
        Self {
            closed: false,
            caps: true,
            up: Vec3::Y,
            scale_along: None,
            twist: 0.0,
            roll_along: None,
            detail: ShapeDetail::default(),
            tint_by_length: None,
            openings: Vec::new(),
        }
    }
}

impl SweepOptions {
    /// Sets [`SweepOptions::closed`].
    #[must_use]
    pub fn with_closed(mut self, closed: bool) -> Self {
        self.closed = closed;
        self
    }

    /// Sets [`SweepOptions::caps`].
    #[must_use]
    pub fn with_caps(mut self, caps: bool) -> Self {
        self.caps = caps;
        self
    }

    /// Sets [`SweepOptions::up`].
    #[must_use]
    pub fn with_up(mut self, up: Vec3) -> Self {
        self.up = up;
        self
    }

    /// Sets [`SweepOptions::scale_along`].
    #[must_use]
    pub fn with_scale_along(mut self, keys: Vec<(f32, Vec2)>) -> Self {
        self.scale_along = Some(keys);
        self
    }

    /// Sets [`SweepOptions::twist`].
    #[must_use]
    pub fn with_twist(mut self, turns: f32) -> Self {
        self.twist = turns;
        self
    }

    /// Sets [`SweepOptions::roll_along`].
    #[must_use]
    pub fn with_roll_along(mut self, keys: Vec<(f32, f32)>) -> Self {
        self.roll_along = Some(keys);
        self
    }

    /// Sets [`SweepOptions::detail`].
    #[must_use]
    pub fn with_detail(mut self, detail: ShapeDetail) -> Self {
        self.detail = detail;
        self
    }

    /// Sets [`SweepOptions::tint_by_length`].
    #[must_use]
    pub fn with_tint_by_length(mut self, keys: Vec<(f32, Color)>) -> Self {
        self.tint_by_length = Some(keys);
        self
    }

    /// Sets [`SweepOptions::openings`].
    #[must_use]
    pub fn with_openings(mut self, openings: Vec<SweepOpening>) -> Self {
        self.openings = openings;
        self
    }

    /// Fills [`SweepOptions::roll_along`] by sampling `f` (path ratio to roll in
    /// radians) at `samples` evenly spaced ratios from 0 to 1 (at least 2).
    #[must_use]
    pub fn roll_with(mut self, samples: usize, f: impl Fn(f32) -> f32) -> Self {
        let samples = samples.max(2);
        self.roll_along = Some(
            (0..samples)
                .map(|i| {
                    let r = i as f32 / (samples - 1) as f32;
                    (r, f(r))
                })
                .collect(),
        );
        self
    }

    /// Fills [`SweepOptions::scale_along`] by sampling `f` (path ratio to
    /// profile scale) at `samples` evenly spaced ratios from 0 to 1 (at least 2).
    ///
    /// A closure cannot be stored in a hashable shape, so it is baked into
    /// keyframes here.
    #[must_use]
    pub fn scale_with(mut self, samples: usize, f: impl Fn(f32) -> Vec2) -> Self {
        let samples = samples.max(2);
        self.scale_along = Some(
            (0..samples)
                .map(|i| {
                    let r = i as f32 / (samples - 1) as f32;
                    (r, f(r))
                })
                .collect(),
        );
        self
    }

    /// Every float and flag that makes two options different, as bits.
    fn identity_bits(&self) -> Vec<u32> {
        let mut v = vec![
            u32::from(self.closed),
            u32::from(self.caps),
            self.up.x.to_bits(),
            self.up.y.to_bits(),
            self.up.z.to_bits(),
            self.twist.to_bits(),
        ];
        match &self.roll_along {
            None => v.push(0),
            Some(keys) => {
                v.extend([1, keys.len() as u32]);
                for (r, a) in keys {
                    v.extend([r.to_bits(), a.to_bits()]);
                }
            }
        }
        match &self.scale_along {
            None => v.push(0),
            Some(keys) => {
                v.extend([1, keys.len() as u32]);
                for (r, s) in keys {
                    v.extend([r.to_bits(), s.x.to_bits(), s.y.to_bits()]);
                }
            }
        }
        match &self.tint_by_length {
            None => v.push(0),
            Some(keys) => {
                v.extend([1, keys.len() as u32]);
                for (r, c) in keys {
                    v.push(r.to_bits());
                    v.extend(c.to_srgba().to_f32_array().map(f32::to_bits));
                }
            }
        }
        v.push(self.openings.len() as u32);
        for op in &self.openings {
            v.extend([op.t.start.to_bits(), op.t.end.to_bits(), op.outline.len() as u32]);
            for p in &op.outline {
                v.extend([p.x.to_bits(), p.y.to_bits()]);
            }
        }
        v
    }
}

fn eq_frames(a: &Option<Vec<SweepFrame>>, b: &Option<Vec<SweepFrame>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.identity_bits() == b.identity_bits())
        }
        _ => false,
    }
}

fn eq_vec2s(a: &[Vec2], b: &[Vec2]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(a, b)| a.x.to_bits() == b.x.to_bits() && a.y.to_bits() == b.y.to_bits())
}

fn hash_vec2s<H: std::hash::Hasher>(v: &[Vec2], state: &mut H) {
    v.len().hash(state);
    for p in v {
        hash_vec2(p, state);
    }
}

fn eq_colors(a: &Option<Vec<[f32; 4]>>, b: &Option<Vec<[f32; 4]>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.len() == b.len()
                && a.iter()
                    .zip(b)
                    .all(|(x, y)| x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()))
        }
        _ => false,
    }
}

/// Pieces per rounded rim of an extruded polygon.
fn bevel_steps(detail: ShapeDetail) -> u32 {
    (detail.segments() / 4).max(2)
}

#[derive(Debug, Clone)]
enum ShapeKind {
    Sphere {
        radius: f32,
    },
    Ellipsoid {
        size: Vec3,
    },
    Cuboid {
        size: Vec3,
        roundness: f32,
    },
    Cylinder {
        radius: f32,
        height: f32,
        roundness: f32,
    },
    Cone {
        radius: f32,
        height: f32,
        roundness: f32,
    },
    Capsule {
        radius: f32,
        height: f32,
    },
    Hemisphere {
        radius: f32,
        capped: bool,
    },
    Torus {
        radius: f32,
        thickness: f32,
        start: f32,
        span: f32,
    },
    RegularPrism {
        sides: u32,
        radius: f32,
        height: f32,
        roundness: f32,
    },
    Pyramid {
        sides: u32,
        radius: f32,
        height: f32,
        roundness: f32,
    },
    // 2D shapes (billboarded flat quads)
    Rectangle {
        size: Vec2,
    },
    RectangleRim {
        size: Vec2,
        rim_width: f32,
    },
    Disc {
        radius: f32,
    },
    DiscRim {
        radius: f32,
        rim_width: f32,
    },
    Polygon {
        sides: u32,
        radius: f32,
    },
    PolygonRim {
        sides: u32,
        radius: f32,
        rim_width: f32,
    },
    Ellipse {
        radius: Vec2,
    },
    EllipseRim {
        radius: Vec2,
        rim_width: f32,
    },
    /// Arbitrary polygon with optional per-vertex colors.
    ArbitraryPolygon {
        vertices: Vec<Vec2>,
        colors: Option<Vec<[f32; 4]>>,
    },
    /// 3D polyline rendered as tube geometry.
    Polyline3D {
        points: Vec<Vec3>,
        thickness: f32,
        closed: bool,
        segments: u32,
    },
    /// Bézier curve rendered as a tube along the curve path.
    BezierCurve {
        control_points: Vec<Vec3>,
        thickness: f32,
        segments: u32,
    },
    /// Catmull-Rom spline rendered as a tube (convenience wrapper around sweep).
    CatmullRom3D {
        points: Vec<Vec3>,
        thickness: f32,
        closed: bool,
        segments: u32,
    },
    /// N-pointed star (2D flat quad, like polygon).
    Star {
        points: u32,
        outer_radius: f32,
        inner_radius: f32,
    },
    /// Ring/annulus (2D flat quad with a hole).
    Ring {
        outer_radius: f32,
        inner_radius: f32,
    },
    /// Wedge/slice of a cylinder (2D flat pie slice).
    Wedge {
        radius: f32,
        angle: f32,
    },
    /// Tapered cylinder (cone-frustum-like, but with flat top and bottom).
    TaperedCylinder {
        top_radius: f32,
        bottom_radius: f32,
        height: f32,
    },
    /// Polygon with optional holes, extruded along Z by `depth` (flat when 0).
    ExtrudedPolygon {
        outer: Vec<Vec2>,
        holes: Vec<Vec<Vec2>>,
        colors: Option<Vec<[f32; 4]>>,
        depth: f32,
        roundness: f32,
    },
    /// Several shapes merged into one mesh.
    Compound { parts: Vec<ShapePart> },
    /// Profile of (radius, height) points revolved around Y.
    Lathe { profile: Vec<Vec2>, segments: u32 },
    /// Flat thick polyline in the XY plane.
    Polyline2D {
        points: Vec<Vec2>,
        thickness: f32,
        closed: bool,
        segments: u32,
    },
    /// Flat thick Bézier curve in the XY plane.
    Bezier2D {
        control_points: Vec<Vec2>,
        thickness: f32,
        segments: u32,
    },
    /// A caller-supplied mesh, identified by `key` (see [`Shape::from_mesh`]).
    Mesh {
        mesh: Arc<Mesh>,
        key: u64,
        bounds: ShapeBounds,
        round: bool,
    },
    /// A profile swept along a path (see [`Shape::sweep`]).
    Sweep {
        profile: Vec<Vec2>,
        path: Vec<Vec3>,
        frames: Option<Vec<SweepFrame>>,
        options: SweepOptions,
    },
    /// A grid of heights over the XZ plane (see [`Shape::heightfield`]).
    Heightfield {
        size: UVec2,
        extent: Vec2,
        heights: Arc<[f32]>,
        holes: Option<Arc<[bool]>>,
        hole_edge: HoleEdge,
        border: Option<Arc<HeightfieldBorder>>,
        /// Hash of `heights`, `holes`, `hole_edge` and `border`, computed once so that
        /// hashing the shape does not walk them again.
        fingerprint: u64,
        /// Bounds of the surface and bounds with skirts.
        bounds: (ShapeBounds, ShapeBounds),
    },
}

impl ShapeKind {
    fn is_flat(&self) -> bool {
        match self {
            Self::Rectangle { .. }
            | Self::RectangleRim { .. }
            | Self::Disc { .. }
            | Self::DiscRim { .. }
            | Self::Polygon { .. }
            | Self::PolygonRim { .. }
            | Self::Ellipse { .. }
            | Self::EllipseRim { .. }
            | Self::ArbitraryPolygon { .. }
            | Self::Star { .. }
            | Self::Ring { .. }
            | Self::Wedge { .. }
            | Self::ExtrudedPolygon { .. }
            | Self::Polyline2D { .. }
            | Self::Bezier2D { .. } => true,
            Self::Compound { parts } => parts.iter().all(|p| p.shape.kind.is_flat()),
            _ => false,
        }
    }
}

/// [`Shape::as_flat`] refused a shape that has depth (a solid, a tube, a lathe,
/// a sweep, a heightfield or a caller-supplied mesh).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NotFlat;

impl std::fmt::Display for NotFlat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("shape is not flat, so it has no 2D form")
    }
}

impl std::error::Error for NotFlat {}

/// Why [`FlatShape::sdf_2d`] could not draw a flat shape with the SDF backend.
///
/// Use [`FlatShape::build_2d`] for any of these shapes: it draws all of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdfConvertError {
    /// A size, radius or rim width is zero, negative or not finite, or a ring's
    /// inner radius is not smaller than its outer radius.
    Degenerate,
    /// The shape has holes; the SDF backend draws one outline.
    HasHoles,
    /// The shape has per-vertex colours; the SDF backend colours by gradient.
    VertexColours,
    /// The shape is flat but has no SDF primitive: a 2D polyline or Bézier
    /// curve, or a compound.
    Unsupported,
    /// A polygon outline has too few or too many vertices.
    Polygon(crate::SdfPolygonError),
}

impl std::fmt::Display for SdfConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Degenerate => f.write_str("a size, radius or rim width is not a positive finite number"),
            Self::HasHoles => f.write_str("shapes with holes need the mesh backend"),
            Self::VertexColours => f.write_str("per-vertex colours need the mesh backend"),
            Self::Unsupported => f.write_str("curves, polylines and compounds need the mesh backend"),
            Self::Polygon(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for SdfConvertError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Polygon(e) => Some(e),
            _ => None,
        }
    }
}

impl From<crate::SdfPolygonError> for SdfConvertError {
    fn from(e: crate::SdfPolygonError) -> Self {
        Self::Polygon(e)
    }
}

/// A [`Shape`] known to be flat (see [`Shape::as_flat`]). The 2D conversions
/// live here, so asking for the flat form of a sphere does not type-check
/// instead of returning something odd.
#[derive(Debug, Clone, Copy)]
pub struct FlatShape<'a> {
    shape: &'a Shape,
}

impl<'a> FlatShape<'a> {
    /// The underlying shape.
    #[must_use]
    pub fn shape(&self) -> &'a Shape {
        self.shape
    }

    /// Flat geometry for `Mesh2d`: in the XY plane, facing +Z, with gradient
    /// UVs. Extruded polygons lose their depth. See [`Shape::build_2d`].
    #[must_use]
    pub fn build_2d(&self) -> Mesh {
        self.shape.build_2d()
    }

    /// Bounds of [`FlatShape::build_2d`]. See [`Shape::bounds_2d`].
    #[must_use]
    pub fn bounds_2d(&self) -> ShapeBounds {
        self.shape.bounds_2d()
    }

    /// The [`SdfMaterial2d`](crate::SdfMaterial2d) that draws this shape on the
    /// shared [`unit_quad`](crate::unit_quad).
    ///
    /// Covered: rectangles, discs, ellipses, regular polygons and their rims,
    /// stars, rings, wedges and arbitrary or extruded polygons of 3 to 16
    /// vertices without holes or per-vertex colours (an extruded polygon is
    /// drawn without depth, like [`FlatShape::build_2d`]).
    ///
    /// The material is centred on the middle of the shape's bounds, so for an
    /// arbitrary polygon place the entity at the middle of
    /// [`FlatShape::bounds_2d`]. Set the entity's `Transform` scale to
    /// [`SdfMaterial2d::quad_scale`](crate::SdfMaterial2d::quad_scale). Rim
    /// widths match the mesh builders: on a regular polygon the mesh rim is
    /// measured along the radius, so the SDF rim is scaled by `cos(pi / sides)`
    /// to the same thickness. [`ShapeDetail`] has no effect, because the field
    /// is exact at any size.
    ///
    /// # Errors
    ///
    /// [`SdfConvertError`] says why the SDF backend cannot draw the shape;
    /// [`FlatShape::build_2d`] still can.
    pub fn sdf_2d(&self) -> Result<crate::SdfMaterial2d, SdfConvertError> {
        use crate::SdfMaterial2d;
        let positive = |v: f32| v.is_finite() && v > 0.0;
        let size_ok = |v: Vec2| positive(v.x) && positive(v.y);
        let rim_ok = |w: f32| w.is_finite();
        let valid = |ok: bool| if ok { Ok(()) } else { Err(SdfConvertError::Degenerate) };
        match &self.shape.kind {
            ShapeKind::Rectangle { size } => {
                valid(size_ok(*size))?;
                Ok(SdfMaterial2d::rect(*size))
            }
            ShapeKind::RectangleRim { size, rim_width } => {
                valid(size_ok(*size) && rim_ok(*rim_width))?;
                Ok(SdfMaterial2d::rect(*size).with_rim(*rim_width))
            }
            ShapeKind::Disc { radius } => {
                valid(positive(*radius))?;
                Ok(SdfMaterial2d::disc(*radius))
            }
            ShapeKind::DiscRim { radius, rim_width } => {
                valid(positive(*radius) && rim_ok(*rim_width))?;
                Ok(SdfMaterial2d::disc(*radius).with_rim(*rim_width))
            }
            ShapeKind::Ellipse { radius } => {
                valid(size_ok(*radius))?;
                Ok(SdfMaterial2d::ellipse(*radius))
            }
            ShapeKind::EllipseRim { radius, rim_width } => {
                valid(size_ok(*radius) && rim_ok(*rim_width))?;
                Ok(SdfMaterial2d::ellipse(*radius).with_rim(*rim_width))
            }
            ShapeKind::Polygon { sides, radius } => {
                valid(positive(*radius))?;
                Ok(SdfMaterial2d::polygon(*sides, *radius))
            }
            ShapeKind::PolygonRim {
                sides,
                radius,
                rim_width,
            } => {
                valid(positive(*radius) && rim_ok(*rim_width))?;
                let sides = (*sides).max(3);
                let apothem_scale = (PI / sides as f32).cos();
                Ok(SdfMaterial2d::polygon(sides, *radius).with_rim(rim_width * apothem_scale))
            }
            ShapeKind::Star {
                points,
                outer_radius,
                inner_radius,
            } => {
                valid(positive(*outer_radius) && inner_radius.is_finite())?;
                Ok(SdfMaterial2d::star(*points, *outer_radius, *inner_radius))
            }
            ShapeKind::Ring {
                outer_radius,
                inner_radius,
            } => {
                valid(positive(*outer_radius) && inner_radius.is_finite() && *inner_radius < *outer_radius)?;
                Ok(SdfMaterial2d::disc(*outer_radius).with_rim(outer_radius - inner_radius.max(0.0)))
            }
            ShapeKind::Wedge { radius, angle } => {
                valid(positive(*radius) && positive(*angle))?;
                Ok(SdfMaterial2d::disc(*radius).with_arc(0.0, *angle))
            }
            ShapeKind::ArbitraryPolygon { vertices, colors } => {
                if colors.is_some() {
                    return Err(SdfConvertError::VertexColours);
                }
                Ok(SdfMaterial2d::polygon_from_points(vertices)?)
            }
            ShapeKind::ExtrudedPolygon {
                outer,
                holes,
                colors,
                ..
            } => {
                if !holes.is_empty() {
                    return Err(SdfConvertError::HasHoles);
                }
                if colors.is_some() {
                    return Err(SdfConvertError::VertexColours);
                }
                Ok(SdfMaterial2d::polygon_from_points(outer)?)
            }
            _ => Err(SdfConvertError::Unsupported),
        }
    }
}

fn hash_vec3<H: std::hash::Hasher>(v: &Vec3, state: &mut H) {
    v.x.to_bits().hash(state);
    v.y.to_bits().hash(state);
    v.z.to_bits().hash(state);
}

fn hash_vec2<H: std::hash::Hasher>(v: &Vec2, state: &mut H) {
    v.x.to_bits().hash(state);
    v.y.to_bits().hash(state);
}

impl PartialEq for ShapeKind {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Sphere { radius: r1 }, Self::Sphere { radius: r2 }) => {
                r1.to_bits() == r2.to_bits()
            }
            (Self::Ellipsoid { size: s1 }, Self::Ellipsoid { size: s2 }) => {
                s1.x.to_bits() == s2.x.to_bits()
                    && s1.y.to_bits() == s2.y.to_bits()
                    && s1.z.to_bits() == s2.z.to_bits()
            }
            (
                Self::Cuboid {
                    size: s1,
                    roundness: r1,
                },
                Self::Cuboid {
                    size: s2,
                    roundness: r2,
                },
            ) => {
                s1.x.to_bits() == s2.x.to_bits()
                    && s1.y.to_bits() == s2.y.to_bits()
                    && s1.z.to_bits() == s2.z.to_bits()
                    && r1.to_bits() == r2.to_bits()
            }
            (
                Self::Cylinder {
                    radius: r1,
                    height: h1,
                    roundness: rn1,
                },
                Self::Cylinder {
                    radius: r2,
                    height: h2,
                    roundness: rn2,
                },
            ) => {
                r1.to_bits() == r2.to_bits()
                    && h1.to_bits() == h2.to_bits()
                    && rn1.to_bits() == rn2.to_bits()
            }
            (
                Self::Cone {
                    radius: r1,
                    height: h1,
                    roundness: rn1,
                },
                Self::Cone {
                    radius: r2,
                    height: h2,
                    roundness: rn2,
                },
            ) => {
                r1.to_bits() == r2.to_bits()
                    && h1.to_bits() == h2.to_bits()
                    && rn1.to_bits() == rn2.to_bits()
            }
            (
                Self::Capsule {
                    radius: r1,
                    height: h1,
                },
                Self::Capsule {
                    radius: r2,
                    height: h2,
                },
            ) => r1.to_bits() == r2.to_bits() && h1.to_bits() == h2.to_bits(),
            (
                Self::Hemisphere {
                    radius: r1,
                    capped: c1,
                },
                Self::Hemisphere {
                    radius: r2,
                    capped: c2,
                },
            ) => r1.to_bits() == r2.to_bits() && c1 == c2,
            (
                Self::Torus {
                    radius: r1,
                    thickness: t1,
                    start: s1,
                    span: sp1,
                },
                Self::Torus {
                    radius: r2,
                    thickness: t2,
                    start: s2,
                    span: sp2,
                },
            ) => {
                r1.to_bits() == r2.to_bits()
                    && t1.to_bits() == t2.to_bits()
                    && s1.to_bits() == s2.to_bits()
                    && sp1.to_bits() == sp2.to_bits()
            }
            (
                Self::RegularPrism {
                    sides: s1,
                    radius: r1,
                    height: h1,
                    roundness: rn1,
                },
                Self::RegularPrism {
                    sides: s2,
                    radius: r2,
                    height: h2,
                    roundness: rn2,
                },
            ) => {
                s1 == s2
                    && r1.to_bits() == r2.to_bits()
                    && h1.to_bits() == h2.to_bits()
                    && rn1.to_bits() == rn2.to_bits()
            }
            (
                Self::Pyramid {
                    sides: s1,
                    radius: r1,
                    height: h1,
                    roundness: rn1,
                },
                Self::Pyramid {
                    sides: s2,
                    radius: r2,
                    height: h2,
                    roundness: rn2,
                },
            ) => {
                s1 == s2
                    && r1.to_bits() == r2.to_bits()
                    && h1.to_bits() == h2.to_bits()
                    && rn1.to_bits() == rn2.to_bits()
            }
            (Self::Rectangle { size: s1 }, Self::Rectangle { size: s2 }) => {
                s1.x.to_bits() == s2.x.to_bits() && s1.y.to_bits() == s2.y.to_bits()
            }
            (
                Self::RectangleRim {
                    size: s1,
                    rim_width: r1,
                },
                Self::RectangleRim {
                    size: s2,
                    rim_width: r2,
                },
            ) => {
                s1.x.to_bits() == s2.x.to_bits()
                    && s1.y.to_bits() == s2.y.to_bits()
                    && r1.to_bits() == r2.to_bits()
            }
            (Self::Disc { radius: r1 }, Self::Disc { radius: r2 }) => r1.to_bits() == r2.to_bits(),
            (
                Self::DiscRim {
                    radius: r1,
                    rim_width: rw1,
                },
                Self::DiscRim {
                    radius: r2,
                    rim_width: rw2,
                },
            ) => r1.to_bits() == r2.to_bits() && rw1.to_bits() == rw2.to_bits(),
            (
                Self::Polygon {
                    sides: s1,
                    radius: r1,
                },
                Self::Polygon {
                    sides: s2,
                    radius: r2,
                },
            ) => s1 == s2 && r1.to_bits() == r2.to_bits(),
            (
                Self::PolygonRim {
                    sides: s1,
                    radius: r1,
                    rim_width: rw1,
                },
                Self::PolygonRim {
                    sides: s2,
                    radius: r2,
                    rim_width: rw2,
                },
            ) => s1 == s2 && r1.to_bits() == r2.to_bits() && rw1.to_bits() == rw2.to_bits(),
            (Self::Ellipse { radius: r1 }, Self::Ellipse { radius: r2 }) => {
                r1.x.to_bits() == r2.x.to_bits() && r1.y.to_bits() == r2.y.to_bits()
            }
            (
                Self::EllipseRim {
                    radius: r1,
                    rim_width: rw1,
                },
                Self::EllipseRim {
                    radius: r2,
                    rim_width: rw2,
                },
            ) => {
                r1.x.to_bits() == r2.x.to_bits()
                    && r1.y.to_bits() == r2.y.to_bits()
                    && rw1.to_bits() == rw2.to_bits()
            }
            (
                Self::ArbitraryPolygon {
                    vertices: v1,
                    colors: c1,
                },
                Self::ArbitraryPolygon {
                    vertices: v2,
                    colors: c2,
                },
            ) => {
                v1.len() == v2.len()
                    && v1.iter().zip(v2.iter()).all(|(a, b)| {
                        a.x.to_bits() == b.x.to_bits() && a.y.to_bits() == b.y.to_bits()
                    })
                    && c1 == c2
            }
            (
                Self::Polyline3D {
                    points: p1,
                    thickness: t1,
                    closed: c1,
                    segments: s1,
                },
                Self::Polyline3D {
                    points: p2,
                    thickness: t2,
                    closed: c2,
                    segments: s2,
                },
            ) => {
                p1.len() == p2.len()
                    && p1.iter().zip(p2.iter()).all(|(a, b)| {
                        // Quantize to 2 decimal places for stable comparison
                        let qa = Vec3::new(
                            (a.x * 100.0).round() / 100.0,
                            (a.y * 100.0).round() / 100.0,
                            (a.z * 100.0).round() / 100.0,
                        );
                        let qb = Vec3::new(
                            (b.x * 100.0).round() / 100.0,
                            (b.y * 100.0).round() / 100.0,
                            (b.z * 100.0).round() / 100.0,
                        );
                        qa.x.to_bits() == qb.x.to_bits()
                            && qa.y.to_bits() == qb.y.to_bits()
                            && qa.z.to_bits() == qb.z.to_bits()
                    })
                    && t1.to_bits() == t2.to_bits()
                    && c1 == c2
                    && s1 == s2
            }
            (
                Self::BezierCurve {
                    control_points: c1,
                    thickness: t1,
                    segments: s1,
                },
                Self::BezierCurve {
                    control_points: c2,
                    thickness: t2,
                    segments: s2,
                },
            ) => {
                c1.len() == c2.len()
                    && c1.iter().zip(c2.iter()).all(|(a, b)| {
                        // Quantize to 2 decimal places for stable comparison
                        let qa = Vec3::new(
                            (a.x * 100.0).round() / 100.0,
                            (a.y * 100.0).round() / 100.0,
                            (a.z * 100.0).round() / 100.0,
                        );
                        let qb = Vec3::new(
                            (b.x * 100.0).round() / 100.0,
                            (b.y * 100.0).round() / 100.0,
                            (b.z * 100.0).round() / 100.0,
                        );
                        qa.x.to_bits() == qb.x.to_bits()
                            && qa.y.to_bits() == qb.y.to_bits()
                            && qa.z.to_bits() == qb.z.to_bits()
                    })
                    && t1.to_bits() == t2.to_bits()
                    && s1 == s2
            }
            (
                Self::CatmullRom3D {
                    points: p1,
                    thickness: t1,
                    closed: c1,
                    segments: s1,
                },
                Self::CatmullRom3D {
                    points: p2,
                    thickness: t2,
                    closed: c2,
                    segments: s2,
                },
            ) => {
                p1.len() == p2.len()
                    && p1.iter().zip(p2.iter()).all(|(a, b)| {
                        let qa = Vec3::new(
                            (a.x * 100.0).round() / 100.0,
                            (a.y * 100.0).round() / 100.0,
                            (a.z * 100.0).round() / 100.0,
                        );
                        let qb = Vec3::new(
                            (b.x * 100.0).round() / 100.0,
                            (b.y * 100.0).round() / 100.0,
                            (b.z * 100.0).round() / 100.0,
                        );
                        qa.x.to_bits() == qb.x.to_bits()
                            && qa.y.to_bits() == qb.y.to_bits()
                            && qa.z.to_bits() == qb.z.to_bits()
                    })
                    && t1.to_bits() == t2.to_bits()
                    && c1 == c2
                    && s1 == s2
            }
            (
                Self::Star {
                    points: p1,
                    outer_radius: o1,
                    inner_radius: i1,
                },
                Self::Star {
                    points: p2,
                    outer_radius: o2,
                    inner_radius: i2,
                },
            ) => p1 == p2 && o1.to_bits() == o2.to_bits() && i1.to_bits() == i2.to_bits(),
            (
                Self::Ring {
                    outer_radius: o1,
                    inner_radius: i1,
                },
                Self::Ring {
                    outer_radius: o2,
                    inner_radius: i2,
                },
            ) => o1.to_bits() == o2.to_bits() && i1.to_bits() == i2.to_bits(),
            (
                Self::Wedge {
                    radius: r1,
                    angle: a1,
                },
                Self::Wedge {
                    radius: r2,
                    angle: a2,
                },
            ) => r1.to_bits() == r2.to_bits() && a1.to_bits() == a2.to_bits(),
            (
                Self::TaperedCylinder {
                    top_radius: t1,
                    bottom_radius: b1,
                    height: h1,
                },
                Self::TaperedCylinder {
                    top_radius: t2,
                    bottom_radius: b2,
                    height: h2,
                },
            ) => {
                t1.to_bits() == t2.to_bits()
                    && b1.to_bits() == b2.to_bits()
                    && h1.to_bits() == h2.to_bits()
            }
            (
                Self::ExtrudedPolygon {
                    outer: o1,
                    holes: h1,
                    colors: c1,
                    depth: d1,
                    roundness: r1,
                },
                Self::ExtrudedPolygon {
                    outer: o2,
                    holes: h2,
                    colors: c2,
                    depth: d2,
                    roundness: r2,
                },
            ) => {
                eq_vec2s(o1, o2)
                    && h1.len() == h2.len()
                    && h1.iter().zip(h2).all(|(a, b)| eq_vec2s(a, b))
                    && eq_colors(c1, c2)
                    && d1.to_bits() == d2.to_bits()
                    && r1.to_bits() == r2.to_bits()
            }
            (Self::Compound { parts: p1 }, Self::Compound { parts: p2 }) => p1 == p2,
            (
                Self::Lathe {
                    profile: p1,
                    segments: s1,
                },
                Self::Lathe {
                    profile: p2,
                    segments: s2,
                },
            ) => eq_vec2s(p1, p2) && s1 == s2,
            (
                Self::Polyline2D {
                    points: p1,
                    thickness: t1,
                    closed: c1,
                    segments: s1,
                },
                Self::Polyline2D {
                    points: p2,
                    thickness: t2,
                    closed: c2,
                    segments: s2,
                },
            ) => eq_vec2s(p1, p2) && t1.to_bits() == t2.to_bits() && c1 == c2 && s1 == s2,
            (
                Self::Bezier2D {
                    control_points: p1,
                    thickness: t1,
                    segments: s1,
                },
                Self::Bezier2D {
                    control_points: p2,
                    thickness: t2,
                    segments: s2,
                },
            ) => eq_vec2s(p1, p2) && t1.to_bits() == t2.to_bits() && s1 == s2,
            (Self::Mesh { key: k1, .. }, Self::Mesh { key: k2, .. }) => k1 == k2,
            (
                Self::Sweep {
                    profile: p1,
                    path: w1,
                    frames: f1,
                    options: o1,
                },
                Self::Sweep {
                    profile: p2,
                    path: w2,
                    frames: f2,
                    options: o2,
                },
            ) => {
                eq_frames(f1, f2)
                    && eq_vec2s(p1, p2)
                    && w1.len() == w2.len()
                    && w1.iter().zip(w2).all(|(a, b)| {
                        a.x.to_bits() == b.x.to_bits()
                            && a.y.to_bits() == b.y.to_bits()
                            && a.z.to_bits() == b.z.to_bits()
                    })
                    && o1.identity_bits() == o2.identity_bits()
            }
            (
                Self::Heightfield {
                    size: s1,
                    extent: e1,
                    holes: h1,
                    hole_edge: he1,
                    border: b1,
                    heights: g1,
                    fingerprint: f1,
                    ..
                },
                Self::Heightfield {
                    size: s2,
                    extent: e2,
                    holes: h2,
                    hole_edge: he2,
                    border: b2,
                    heights: g2,
                    fingerprint: f2,
                    ..
                },
            ) => {
                f1 == f2
                    && s1 == s2
                    && (Arc::ptr_eq(g1, g2)
                        || (g1.len() == g2.len()
                            && g1.iter().zip(g2.iter()).all(|(a, b)| a.to_bits() == b.to_bits())))
                    && e1.x.to_bits() == e2.x.to_bits()
                    && e1.y.to_bits() == e2.y.to_bits()
                    && h1 == h2
                    && he1 == he2
                    && b1.as_deref().map(HeightfieldBorder::identity_bits)
                        == b2.as_deref().map(HeightfieldBorder::identity_bits)
            }
            _ => false,
        }
    }
}

impl Eq for ShapeKind {}

impl std::hash::Hash for ShapeKind {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Sphere { radius } => radius.to_bits().hash(state),
            Self::Ellipsoid { size } => hash_vec3(size, state),
            Self::Cuboid { size, roundness } => {
                hash_vec3(size, state);
                roundness.to_bits().hash(state);
            }
            Self::Cylinder {
                radius,
                height,
                roundness,
            } => {
                radius.to_bits().hash(state);
                height.to_bits().hash(state);
                roundness.to_bits().hash(state);
            }
            Self::Cone {
                radius,
                height,
                roundness,
            } => {
                radius.to_bits().hash(state);
                height.to_bits().hash(state);
                roundness.to_bits().hash(state);
            }
            Self::Capsule { radius, height } => {
                radius.to_bits().hash(state);
                height.to_bits().hash(state);
            }
            Self::Hemisphere { radius, capped } => {
                radius.to_bits().hash(state);
                capped.hash(state);
            }
            Self::Torus {
                radius,
                thickness,
                start,
                span,
            } => {
                radius.to_bits().hash(state);
                thickness.to_bits().hash(state);
                start.to_bits().hash(state);
                span.to_bits().hash(state);
            }
            Self::RegularPrism {
                sides,
                radius,
                height,
                roundness,
            } => {
                sides.hash(state);
                radius.to_bits().hash(state);
                height.to_bits().hash(state);
                roundness.to_bits().hash(state);
            }
            Self::Pyramid {
                sides,
                radius,
                height,
                roundness,
            } => {
                sides.hash(state);
                radius.to_bits().hash(state);
                height.to_bits().hash(state);
                roundness.to_bits().hash(state);
            }
            Self::Rectangle { size } => hash_vec2(size, state),
            Self::RectangleRim { size, rim_width } => {
                hash_vec2(size, state);
                rim_width.to_bits().hash(state);
            }
            Self::Disc { radius } => radius.to_bits().hash(state),
            Self::DiscRim { radius, rim_width } => {
                radius.to_bits().hash(state);
                rim_width.to_bits().hash(state);
            }
            Self::Polygon { sides, radius } => {
                sides.hash(state);
                radius.to_bits().hash(state);
            }
            Self::PolygonRim {
                sides,
                radius,
                rim_width,
            } => {
                sides.hash(state);
                radius.to_bits().hash(state);
                rim_width.to_bits().hash(state);
            }
            Self::Ellipse { radius } => hash_vec2(radius, state),
            Self::EllipseRim { radius, rim_width } => {
                hash_vec2(radius, state);
                rim_width.to_bits().hash(state);
            }
            Self::ArbitraryPolygon { vertices, colors } => {
                vertices.len().hash(state);
                for v in vertices {
                    // Quantize to 2 decimal places for stable hashing
                    let qx = (v.x * 100.0).round() / 100.0;
                    let qy = (v.y * 100.0).round() / 100.0;
                    qx.to_bits().hash(state);
                    qy.to_bits().hash(state);
                }
                if let Some(colors) = colors {
                    colors.len().hash(state);
                    for c in colors {
                        for component in c {
                            component.to_bits().hash(state);
                        }
                    }
                }
            }
            Self::Polyline3D {
                points,
                thickness,
                closed,
                segments,
            } => {
                points.len().hash(state);
                for p in points {
                    // Quantize to 2 decimal places for stable hashing
                    let qx = (p.x * 100.0).round() / 100.0;
                    let qy = (p.y * 100.0).round() / 100.0;
                    let qz = (p.z * 100.0).round() / 100.0;
                    qx.to_bits().hash(state);
                    qy.to_bits().hash(state);
                    qz.to_bits().hash(state);
                }
                thickness.to_bits().hash(state);
                closed.hash(state);
                segments.hash(state);
            }
            Self::BezierCurve {
                control_points,
                thickness,
                segments,
            } => {
                control_points.len().hash(state);
                for p in control_points {
                    // Quantize to 2 decimal places for stable hashing
                    let qx = (p.x * 100.0).round() / 100.0;
                    let qy = (p.y * 100.0).round() / 100.0;
                    let qz = (p.z * 100.0).round() / 100.0;
                    qx.to_bits().hash(state);
                    qy.to_bits().hash(state);
                    qz.to_bits().hash(state);
                }
                thickness.to_bits().hash(state);
                segments.hash(state);
            }
            Self::CatmullRom3D {
                points,
                thickness,
                closed,
                segments,
            } => {
                points.len().hash(state);
                for p in points {
                    let qx = (p.x * 100.0).round() / 100.0;
                    let qy = (p.y * 100.0).round() / 100.0;
                    let qz = (p.z * 100.0).round() / 100.0;
                    qx.to_bits().hash(state);
                    qy.to_bits().hash(state);
                    qz.to_bits().hash(state);
                }
                thickness.to_bits().hash(state);
                closed.hash(state);
                segments.hash(state);
            }
            Self::Star {
                points,
                outer_radius,
                inner_radius,
            } => {
                points.hash(state);
                outer_radius.to_bits().hash(state);
                inner_radius.to_bits().hash(state);
            }
            Self::Ring {
                outer_radius,
                inner_radius,
            } => {
                outer_radius.to_bits().hash(state);
                inner_radius.to_bits().hash(state);
            }
            Self::Wedge { radius, angle } => {
                radius.to_bits().hash(state);
                angle.to_bits().hash(state);
            }
            Self::TaperedCylinder {
                top_radius,
                bottom_radius,
                height,
            } => {
                top_radius.to_bits().hash(state);
                bottom_radius.to_bits().hash(state);
                height.to_bits().hash(state);
            }
            Self::ExtrudedPolygon {
                outer,
                holes,
                colors,
                depth,
                roundness,
            } => {
                hash_vec2s(outer, state);
                holes.len().hash(state);
                for h in holes {
                    hash_vec2s(h, state);
                }
                colors.is_some().hash(state);
                if let Some(colors) = colors {
                    for c in colors {
                        for v in c {
                            v.to_bits().hash(state);
                        }
                    }
                }
                depth.to_bits().hash(state);
                roundness.to_bits().hash(state);
            }
            Self::Compound { parts } => parts.hash(state),
            Self::Lathe { profile, segments } => {
                hash_vec2s(profile, state);
                segments.hash(state);
            }
            Self::Polyline2D {
                points,
                thickness,
                closed,
                segments,
            } => {
                hash_vec2s(points, state);
                thickness.to_bits().hash(state);
                closed.hash(state);
                segments.hash(state);
            }
            Self::Bezier2D {
                control_points,
                thickness,
                segments,
            } => {
                hash_vec2s(control_points, state);
                thickness.to_bits().hash(state);
                segments.hash(state);
            }
            Self::Mesh { key, .. } => key.hash(state),
            Self::Sweep {
                profile,
                path,
                frames,
                options,
            } => {
                frames.as_ref().map(|f| f.iter().map(SweepFrame::identity_bits).collect::<Vec<_>>()).hash(state);
                hash_vec2s(profile, state);
                path.len().hash(state);
                for p in path {
                    hash_vec3(p, state);
                }
                options.identity_bits().hash(state);
            }
            Self::Heightfield {
                size,
                extent,
                fingerprint,
                ..
            } => {
                size.hash(state);
                hash_vec2(extent, state);
                fingerprint.hash(state);
            }
        }
    }
}

impl Shape {
    /// Creates a sphere with the given radius.
    pub fn sphere(radius: f32, detail: ShapeDetail) -> Self {
        Self {
            kind: ShapeKind::Sphere { radius },
            detail,
        }
    }

    /// Creates an ellipsoid with the given size (X, Y, Z dimensions).
    pub fn ellipsoid(size: Vec3, detail: ShapeDetail) -> Self {
        Self {
            kind: ShapeKind::Ellipsoid { size },
            detail,
        }
    }

    /// Creates a cuboid with the given size and optional roundness (0.0-1.0).
    pub fn cuboid(size: Vec3, roundness: f32, detail: ShapeDetail) -> Self {
        Self {
            kind: ShapeKind::Cuboid { size, roundness },
            detail,
        }
    }

    /// Creates a cylinder with the given radius and height.
    pub fn cylinder(radius: f32, height: f32, detail: ShapeDetail) -> Self {
        Self {
            kind: ShapeKind::Cylinder {
                radius,
                height,
                roundness: 0.0,
            },
            detail,
        }
    }

    /// Creates a cone with the given base radius and height.
    pub fn cone(radius: f32, height: f32, detail: ShapeDetail) -> Self {
        Self::rounded_cone(radius, height, 0.0, detail)
    }

    /// Creates a cone whose apex and base rim are rounded.
    ///
    /// `roundness` runs from `0.0` (sharp, same as [`Shape::cone`]) to `1.0`
    /// (as round as the cone allows: an inscribed, ice-cream-drop shape). The
    /// rounded solid stays inside the sharp cone.
    pub fn rounded_cone(radius: f32, height: f32, roundness: f32, detail: ShapeDetail) -> Self {
        Self {
            kind: ShapeKind::Cone {
                radius,
                height,
                roundness,
            },
            detail,
        }
    }

    /// Creates a capsule with the given radius and total height.
    /// Height must be at least 2x the radius.
    pub fn capsule(radius: f32, height: f32, detail: ShapeDetail) -> Self {
        Self {
            kind: ShapeKind::Capsule { radius, height },
            detail,
        }
    }

    /// Creates a hemisphere (half sphere) with the given radius.
    /// If `capped` is true, the flat face is filled with geometry.
    pub fn hemisphere(radius: f32, capped: bool, detail: ShapeDetail) -> Self {
        Self {
            kind: ShapeKind::Hemisphere { radius, capped },
            detail,
        }
    }

    /// Creates a torus with the given major radius and tube thickness.
    /// `start` and `span` define the arc in radians (default: full circle).
    /// `detail` controls the segment count for both the major and minor rings.
    pub fn torus(radius: f32, thickness: f32, start: f32, span: f32, detail: ShapeDetail) -> Self {
        Self {
            kind: ShapeKind::Torus {
                radius,
                thickness,
                start,
                span,
            },
            detail,
        }
    }

    /// Creates a regular prism with the given number of sides, radius, and height.
    pub fn regular_prism(
        sides: u32,
        radius: f32,
        height: f32,
        roundness: f32,
        detail: ShapeDetail,
    ) -> Self {
        Self {
            kind: ShapeKind::RegularPrism {
                sides,
                radius,
                height,
                roundness,
            },
            detail,
        }
    }

    /// Creates a pyramid with the given number of sides, base radius, and height.
    pub fn pyramid(sides: u32, radius: f32, height: f32, detail: ShapeDetail) -> Self {
        Self::rounded_pyramid(sides, radius, height, 0.0, detail)
    }

    /// Creates a pyramid whose apex, base rim and corner edges are rounded.
    ///
    /// `roundness` runs from `0.0` (sharp, same as [`Shape::pyramid`]) to `1.0`
    /// (as round as the pyramid allows). The rounded solid stays inside the
    /// sharp pyramid, so soft roofs and gems keep their footprint.
    pub fn rounded_pyramid(
        sides: u32,
        radius: f32,
        height: f32,
        roundness: f32,
        detail: ShapeDetail,
    ) -> Self {
        Self {
            kind: ShapeKind::Pyramid {
                sides,
                radius,
                height,
                roundness,
            },
            detail,
        }
    }

    /// Creates a 2D rectangle with the given width and height.
    pub fn rectangle(size: Vec2, detail: ShapeDetail) -> Self {
        Self {
            kind: ShapeKind::Rectangle { size },
            detail,
        }
    }

    /// Creates a 2D rectangle with a hollow rim (rim_width from the edge).
    pub fn rectangle_rim(size: Vec2, rim_width: f32, detail: ShapeDetail) -> Self {
        Self {
            kind: ShapeKind::RectangleRim { size, rim_width },
            detail,
        }
    }

    /// Creates a 2D disc (circle) with the given radius.
    pub fn disc(radius: f32, detail: ShapeDetail) -> Self {
        Self {
            kind: ShapeKind::Disc { radius },
            detail,
        }
    }

    /// Creates a 2D disc with a hollow rim (rim_width from the edge).
    pub fn disc_rim(radius: f32, rim_width: f32, detail: ShapeDetail) -> Self {
        Self {
            kind: ShapeKind::DiscRim { radius, rim_width },
            detail,
        }
    }

    /// Creates a 2D regular polygon with the given number of sides and radius.
    pub fn polygon(sides: u32, radius: f32, detail: ShapeDetail) -> Self {
        Self {
            kind: ShapeKind::Polygon { sides, radius },
            detail,
        }
    }

    /// Creates a 2D regular polygon with a hollow rim.
    pub fn polygon_rim(sides: u32, radius: f32, rim_width: f32, detail: ShapeDetail) -> Self {
        Self {
            kind: ShapeKind::PolygonRim {
                sides,
                radius,
                rim_width,
            },
            detail,
        }
    }

    /// Creates a 2D ellipse with the given X and Y radii.
    pub fn ellipse(radius: Vec2, detail: ShapeDetail) -> Self {
        Self {
            kind: ShapeKind::Ellipse { radius },
            detail,
        }
    }

    /// Creates a 2D ellipse with a hollow rim.
    pub fn ellipse_rim(radius: Vec2, rim_width: f32, detail: ShapeDetail) -> Self {
        Self {
            kind: ShapeKind::EllipseRim { radius, rim_width },
            detail,
        }
    }

    /// Creates an arbitrary polygon from a list of 2D vertices.
    ///
    /// The polygon is triangulated using ear-clipping and always faces +Z.
    /// Either winding is accepted, and repeated points, the closing duplicate
    /// and collinear or near-collinear vertices (an edge subdivided into many
    /// almost straight points) are cleaned up, so a concave outline such as a
    /// jigsaw piece with knobs and blanks can be one single polygon. Outlines
    /// that cross themselves have no well-defined fill; they still produce a
    /// mesh, but not a meaningful one. Outlines with no area give an empty mesh.
    ///
    /// # Arguments
    /// * `vertices` - List of 2D vertex positions (minimum 3 vertices).
    /// * `detail` - Ignored; polygon uses exact vertices.
    pub fn arbitrary_polygon(vertices: Vec<Vec2>, detail: ShapeDetail) -> Self {
        Self {
            kind: ShapeKind::ArbitraryPolygon {
                vertices,
                colors: None,
            },
            detail,
        }
    }

    /// Creates an arbitrary polygon with per-vertex colors.
    ///
    /// # Arguments
    /// * `vertices` - List of 2D vertex positions.
    /// * `colors` - Per-vertex RGBA colors (must match vertex count).
    /// * `detail` - Ignored; polygon uses exact vertices.
    pub fn arbitrary_polygon_colored(
        vertices: Vec<Vec2>,
        colors: Vec<[f32; 4]>,
        detail: ShapeDetail,
    ) -> Self {
        let colors = if colors.len() == vertices.len() {
            Some(colors)
        } else {
            None
        };
        Self {
            kind: ShapeKind::ArbitraryPolygon { vertices, colors },
            detail,
        }
    }

    /// Creates a 3D polyline rendered as tube geometry.
    ///
    /// # Arguments
    /// * `points` - Path vertices (minimum 2 points).
    /// * `thickness` - Tube radius.
    /// * `closed` - Whether the polyline forms a closed loop.
    /// * `segments` - Cross-section segments around the tube.
    /// * `detail` - Ignored; uses `segments` for detail.
    pub fn polyline3d(
        points: Vec<Vec3>,
        thickness: f32,
        closed: bool,
        segments: u32,
        detail: ShapeDetail,
    ) -> Self {
        Self {
            kind: ShapeKind::Polyline3D {
                points,
                thickness,
                closed,
                segments,
            },
            detail,
        }
    }

    /// Creates a Bézier curve rendered as a tube along the curve path.
    ///
    /// # Arguments
    /// * `control_points` - Bézier control points (minimum 2).
    /// * `thickness` - Tube radius.
    /// * `segments` - Cross-section segments around the tube.
    /// * `detail` - Ignored; uses `segments` for detail.
    pub fn bezier_curve(
        control_points: Vec<Vec3>,
        thickness: f32,
        segments: u32,
        detail: ShapeDetail,
    ) -> Self {
        Self {
            kind: ShapeKind::BezierCurve {
                control_points,
                thickness,
                segments,
            },
            detail,
        }
    }

    /// Creates an N-pointed star (2D flat quad).
    /// `inner_radius` controls how deep the indentations are (0.0-`outer_radius`).
    pub fn star(points: u32, outer_radius: f32, inner_radius: f32, detail: ShapeDetail) -> Self {
        Self {
            kind: ShapeKind::Star {
                points: points.max(2),
                outer_radius,
                inner_radius,
            },
            detail,
        }
    }

    /// Creates a ring/annulus (2D flat quad with a hole).
    /// If `inner_radius` >= `outer_radius`, produces an empty mesh.
    pub fn ring(outer_radius: f32, inner_radius: f32, detail: ShapeDetail) -> Self {
        Self {
            kind: ShapeKind::Ring {
                outer_radius,
                inner_radius: inner_radius.min(outer_radius),
            },
            detail,
        }
    }

    /// Creates a wedge/pie-slice (2D flat quad).
    /// `angle` is in radians (0 to 2*PI). A full circle at 2*PI.
    pub fn wedge(radius: f32, angle: f32, detail: ShapeDetail) -> Self {
        Self {
            kind: ShapeKind::Wedge {
                radius,
                angle: angle.clamp(0.0, 2.0 * PI),
            },
            detail,
        }
    }

    /// Creates a tapered cylinder (cone-frustum with flat top and bottom).
    /// When `top_radius` == `bottom_radius`, equivalent to a regular cylinder.
    pub fn tapered_cylinder(
        top_radius: f32,
        bottom_radius: f32,
        height: f32,
        detail: ShapeDetail,
    ) -> Self {
        Self {
            kind: ShapeKind::TaperedCylinder {
                top_radius,
                bottom_radius,
                height,
            },
            detail,
        }
    }

    /// Creates a polygon extruded along its normal (+Z) by `depth`, centred on
    /// `z = 0`, with rounded edges.
    ///
    /// * `vertices` - outline in the XY plane; either winding is accepted (see
    ///   [`arch`](crate::outline::arch), [`rounded_rectangle`](crate::outline::rounded_rectangle)
    ///   and [`blob`](crate::outline::blob) for ready-made outlines).
    /// * `roundness` - `0.0` (sharp edges) to `1.0` (rim rounded as far as the
    ///   depth and outline size allow).
    /// * `detail` - controls how many pieces the rounded rim has.
    pub fn extruded_polygon(
        vertices: Vec<Vec2>,
        depth: f32,
        roundness: f32,
        detail: ShapeDetail,
    ) -> Self {
        Self::extruded_polygon_with_holes(vertices, Vec::new(), depth, roundness, detail)
    }

    /// Like [`Shape::extruded_polygon`] with per-vertex RGBA colours (one per
    /// vertex, linear). They multiply the gradient colour, so cobbles can carry
    /// tinted tops. Mismatched lengths are ignored.
    pub fn extruded_polygon_colored(
        vertices: Vec<Vec2>,
        colors: Vec<[f32; 4]>,
        depth: f32,
        roundness: f32,
        detail: ShapeDetail,
    ) -> Self {
        let colors = (colors.len() == vertices.len()).then_some(colors);
        Self {
            kind: ShapeKind::ExtrudedPolygon {
                outer: vertices,
                holes: Vec::new(),
                colors,
                depth,
                roundness,
            },
            detail,
        }
    }

    /// Creates a flat polygon with one or more holes (for example a wall with a
    /// door opening). Either winding is accepted for every ring.
    pub fn arbitrary_polygon_with_holes(
        outer: Vec<Vec2>,
        holes: Vec<Vec<Vec2>>,
        detail: ShapeDetail,
    ) -> Self {
        Self::extruded_polygon_with_holes(outer, holes, 0.0, 0.0, detail)
    }

    /// Creates a polygon with holes extruded like [`Shape::extruded_polygon`].
    pub fn extruded_polygon_with_holes(
        outer: Vec<Vec2>,
        holes: Vec<Vec<Vec2>>,
        depth: f32,
        roundness: f32,
        detail: ShapeDetail,
    ) -> Self {
        Self {
            kind: ShapeKind::ExtrudedPolygon {
                outer,
                holes,
                colors: None,
                depth,
                roundness,
            },
            detail,
        }
    }

    /// Unified constructor for extruded polygons supporting holes **and** colours
    /// in a single call.
    ///
    /// This replaces the fragmented API of [`Shape::extruded_polygon`],
    /// [`Shape::extruded_polygon_colored`], and [`Shape::extruded_polygon_with_holes`]
    /// by accepting all options at once. When `colors` is provided, its length
    /// must match `outer`; otherwise it is ignored.
    pub fn extruded_polygon_with_options(
        outer: Vec<Vec2>,
        holes: Vec<Vec<Vec2>>,
        colors: Option<Vec<[f32; 4]>>,
        depth: f32,
        roundness: f32,
        detail: ShapeDetail,
    ) -> Self {
        let colors = colors.and_then(|c| (c.len() == outer.len()).then_some(c));
        Self {
            kind: ShapeKind::ExtrudedPolygon {
                outer,
                holes,
                colors,
                depth,
                roundness,
            },
            detail,
        }
    }

    /// Merges `parts` into one mesh (one draw call).
    ///
    /// Each part is built, moved by its transform and tinted with its optional
    /// colour (applied as vertex colours that multiply the gradient). The compound
    /// is `Hash + Eq`, so it can be cached like any other shape.
    pub fn compound(parts: Vec<ShapePart>) -> Self {
        Self {
            kind: ShapeKind::Compound { parts },
            detail: ShapeDetail::default(),
        }
    }

    /// Revolves a profile of `(radius, height)` points around the Y axis.
    ///
    /// Points run along the silhouette from one end to the other; the profile is
    /// reordered bottom-to-top if necessary so normals face outward. Corners
    /// sharper than about 35 degrees stay crisp, gentle ones are smoothed, and
    /// open ends away from the axis get flat caps. `segments` is the number of
    /// slices around the axis (at least 3; `0` uses the segment count of
    /// `detail`).
    pub fn lathe(profile: Vec<Vec2>, segments: u32, detail: ShapeDetail) -> Self {
        Self {
            kind: ShapeKind::Lathe { profile, segments },
            detail,
        }
    }

    /// Creates a flat polyline `thickness` wide in the XY plane (facing +Z) with
    /// mitred joints, for 2D curves, guide lines and connection noodles. Use
    /// [`Shape::build_2d`] with `Mesh2d`.
    ///
    /// UVs run along the line in `u` (0 to 1) and across it in `v`, so a
    /// `dual_directional` gradient along `X` fades along the curve. `closed` joins
    /// the last point back to the first. `segments` controls cross-section detail
    /// (at least 3). The flat counterpart of [`Shape::polyline3d`].
    pub fn polyline_2d(points: Vec<Vec2>, thickness: f32, closed: bool, segments: u32) -> Self {
        Self {
            kind: ShapeKind::Polyline2D {
                points,
                thickness,
                closed,
                segments,
            },
            detail: ShapeDetail::default(),
        }
    }

    /// Creates a flat Bézier curve `thickness` wide through any number of control
    /// points (degree = points - 1) in the XY plane. `segments` sets how finely the
    /// curve is sampled (at least 4). The flat counterpart of the 3D Bézier curve.
    pub fn bezier_2d(control_points: Vec<Vec2>, thickness: f32, segments: u32) -> Self {
        Self {
            kind: ShapeKind::Bezier2D {
                control_points,
                thickness,
                segments,
            },
            detail: ShapeDetail::default(),
        }
    }

    /// Creates a Catmull-Rom spline rendered as a tube along the curve path.
    ///
    /// This is the one-shot convenience counterpart of [`Shape::polyline3d`] and
    /// [`Shape::bezier_curve`]: pass control points and get a smooth tube without
    /// manually building a [`CatmullRomSpline`](crate::CatmullRomSpline).
    ///
    /// # Arguments
    /// * `points` - Control points the curve passes through (minimum 2 for open, 3 for closed).
    /// * `thickness` - Tube radius.
    /// * `closed` - Whether the spline forms a closed loop.
    /// * `segments` - Cross-section segments around the tube.
    /// * `detail` - Ignored; uses `segments` for detail.
    ///
    /// If fewer than 2 points are given (or fewer than 3 for closed), an empty
    /// mesh is produced with a warning.
    pub fn catmull_rom_3d(
        points: Vec<Vec3>,
        thickness: f32,
        closed: bool,
        segments: u32,
        detail: ShapeDetail,
    ) -> Self {
        Self {
            kind: ShapeKind::CatmullRom3D {
                points,
                thickness,
                closed,
                segments,
            },
            detail,
        }
    }

    /// A shape made of a caller-supplied `mesh`, identified by `key`.
    ///
    /// The shape is `Eq` and `Hash` by `key` (and detail) only; the mesh itself is
    /// never compared. The caller owns key uniqueness: two different meshes with
    /// the same key share one cache entry. Bounds and [`Shape::is_round`] are
    /// measured from the mesh positions once, here, so [`Shape::bounds`] does not
    /// build anything. [`Shape::build`] and [`Shape::build_2d`] return a copy of
    /// the mesh, [`Shape::lerp`] takes the nearer shape, and the shape works as a
    /// [`ShapePart`] of [`Shape::compound`].
    pub fn from_mesh(mesh: Arc<Mesh>, key: u64) -> Self {
        let bounds = mesh_bounds(&mesh);
        let round = mesh_is_round(&mesh, &bounds);
        Self {
            kind: ShapeKind::Mesh {
                mesh,
                key,
                bounds,
                round,
            },
            detail: ShapeDetail::default(),
        }
    }

    /// Sweeps `profile` along `path` as one mesh: roads, rails, pipes and walls
    /// along a curve.
    ///
    /// `profile` is a closed outline in the local XY plane, `x` across the path
    /// and `y` up (the frame's `right` and `up`, as in
    /// [`SplineFrame`](crate::SplineFrame)); it is wound counter-clockwise for you.
    /// `path` is a polyline; for a baked spline pass the frame positions. Repeated
    /// path points are dropped.
    ///
    /// Frames come from parallel transport starting at [`SweepOptions::up`].
    /// Joints are mitred, so the sides keep a constant width through a bend (the
    /// mitre is clamped to 4 times the width at very sharp turns; the inside of a
    /// tight bend with a wide profile can still overlap itself). Normals are smooth
    /// along the path; across the profile a corner sharper than about 40 degrees
    /// stays crisp and gentler ones are smoothed. UVs run across the profile in
    /// `u` and along the path in `v`.
    ///
    /// With fewer than two distinct path points (three when closed) or fewer than
    /// three distinct profile points the mesh is empty and a warning is logged.
    /// Two sweeps blend with [`Shape::lerp`] only as the nearer one.
    pub fn sweep(profile: Vec<Vec2>, path: Vec<Vec3>, options: SweepOptions) -> Self {
        let detail = options.detail;
        Self {
            kind: ShapeKind::Sweep {
                profile,
                path,
                frames: None,
                options,
            },
            detail,
        }
    }

    /// Sweeps `profile` through cross-sections you supply, instead of letting
    /// the sweep derive frames from a polyline.
    ///
    /// Use it when the frames already exist, for example the baked frames of a
    /// spline with its own banking: the sweep keeps their `right` and `up`
    /// (and `scale`) as given and does not run parallel transport,
    /// [`SweepOptions::up`] is unused, and [`SweepOptions::roll_along`],
    /// [`SweepOptions::twist`] and [`SweepOptions::scale_along`] still apply on
    /// top of them. Frames in between (at the extra rings that smoothness,
    /// openings and tint need) are blended from their neighbours. Frames at the
    /// same position as the one before are dropped.
    ///
    /// `right x up` should point along the path (see [`SweepFrame`]); a
    /// closed sweep joins the last frame back to the first and does not
    /// correct the frames at the seam, so make them meet. Everything else works
    /// as in [`Shape::sweep`].
    pub fn sweep_frames(profile: Vec<Vec2>, frames: &[SweepFrame], options: SweepOptions) -> Self {
        let detail = options.detail;
        Self {
            kind: ShapeKind::Sweep {
                profile,
                path: frames.iter().map(|f| f.position).collect(),
                frames: Some(frames.to_vec()),
                options,
            },
            detail,
        }
    }

    /// Alias for [`Shape::sweep`]: sweeps `profile` along a polyline `path`.
    ///
    /// The name makes the relationship with [`Shape::sweep_with_frames`] clearer:
    /// `sweep_path` derives frames from a polyline, `sweep_with_frames` accepts
    /// pre-baked frames.
    pub fn sweep_path(profile: Vec<Vec2>, path: Vec<Vec3>, options: SweepOptions) -> Self {
        Self::sweep(profile, path, options)
    }

    /// Alias for [`Shape::sweep_frames`]: sweeps `profile` through pre-baked frames.
    ///
    /// The name makes the relationship with [`Shape::sweep_path`] clearer:
    /// `sweep_path` derives frames from a polyline, `sweep_with_frames` accepts
    /// pre-baked frames (e.g. from a baked spline).
    pub fn sweep_with_frames(profile: Vec<Vec2>, frames: &[SweepFrame], options: SweepOptions) -> Self {
        Self::sweep_frames(profile, frames, options)
    }

    /// A terrain chunk: a grid of `size.x * size.y` height samples spread over
    /// `extent` world units and centred on the origin in XZ. Shorthand for
    /// [`Shape::heightfield_with_options`] with no holes and no border.
    pub fn heightfield(heights: Arc<[f32]>, size: UVec2, extent: Vec2) -> Self {
        Self::heightfield_with_options(heights, size, extent, HeightfieldOptions::default())
    }

    /// A terrain chunk: a grid of `size.x * size.y` height samples spread over
    /// `extent` world units and centred on the origin in XZ.
    ///
    /// `heights[x + y * size.x]` is the sample at column `x` and row `y`, with rows
    /// running along +Z. The mesh has smooth normals from central differences
    /// (with a [`HeightfieldOptions::border`], the edge vertices use the
    /// neighbours' heights too, so chunks that share an edge share its normals),
    /// UVs from 0 to 1 and skirts that hang a tenth of the larger extent below
    /// the edges of the chunk to hide cracks between neighbouring chunks.
    /// Cells flagged in [`HeightfieldOptions::holes`] have no triangles and no
    /// skirt; [`HeightfieldOptions::hole_edge`] picks whether the outline of a
    /// hole is a staircase of whole cells or is cut along the outline. The skirt
    /// depth is not an option: a chunk without skirts shows cracks.
    ///
    /// [`Shape::bounds`] includes the skirts and [`Shape::surface_bounds`] does
    /// not; both ignore vertices that only touch holes.
    ///
    /// The shape is `Eq` and `Hash` by `size`, `extent`, the heights, the hole
    /// mask and the border, all compared by value: there is no key to keep in
    /// step, so changing anything that changes the mesh gives a different shape
    /// and a stale mesh can never be shared. The heights, mask and border are
    /// walked once, here, into a fingerprint, so hashing the shape afterwards
    /// costs the same at any size (a hash is taken on every
    /// [`MeshCache`](crate::MeshCache) lookup). Building the shape costs
    /// about 1 ns per sample (4 us for a 65 x 65 chunk); comparing two shapes
    /// that share the same `Arc` of heights is free, and comparing two equal
    /// but separate ones walks the heights once. A `size` below 2 in either axis, a
    /// `heights` slice, hole mask or border of the wrong length gives an empty
    /// mesh and a warning when built.
    pub fn heightfield_with_options(
        heights: Arc<[f32]>,
        size: UVec2,
        extent: Vec2,
        options: HeightfieldOptions,
    ) -> Self {
        let holes: Option<Arc<[bool]>> = options.holes.map(Arc::from);
        let hole_edge = options.hole_edge;
        let border = options.border.map(Arc::new);
        let fingerprint = {
            use std::hash::Hasher;
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            heights.len().hash(&mut hasher);
            for h in heights.iter() {
                h.to_bits().hash(&mut hasher);
            }
            holes.hash(&mut hasher);
            hole_edge.hash(&mut hasher);
            border.as_deref().map(HeightfieldBorder::identity_bits).hash(&mut hasher);
            hasher.finish()
        };
        let valid =
            geometry::heightfield_check(size, heights.len(), holes.as_deref(), border.as_deref()).is_ok();
        let bounds = if valid {
            geometry::heightfield_bounds(size, extent, &heights, holes.as_deref())
        } else {
            (ShapeBounds::default(), ShapeBounds::default())
        };
        Self {
            kind: ShapeKind::Heightfield {
                size,
                extent,
                heights,
                holes,
                hole_edge,
                border,
                fingerprint,
                bounds,
            },
            detail: ShapeDetail::default(),
        }
    }

    /// Outline helper: see [`outline::arch`](crate::outline::arch).
    pub fn arch(width: f32, height: f32, segments: u32) -> Vec<Vec2> {
        crate::outline::arch(width, height, segments)
    }

    /// Outline helper: see [`outline::rounded_rectangle`](crate::outline::rounded_rectangle).
    pub fn rounded_rectangle(size: Vec2, radius: f32, segments: u32) -> Vec<Vec2> {
        crate::outline::rounded_rectangle(size, radius, segments)
    }

    /// Outline helper: see [`outline::blob`](crate::outline::blob).
    pub fn blob(radius: f32, wobble: f32, seed: u64) -> Vec<Vec2> {
        crate::outline::blob(radius, wobble, seed)
    }

    /// Axis-aligned bounds of the geometry [`Shape::build`] produces, in local
    /// space (before any `Transform`).
    ///
    /// The bounds are exact for every shape kind (rounded edges, partial tori,
    /// compounds, lathes and curves included). Shapes whose box has a closed
    /// form (sphere, ellipsoid, cuboid, cylinder, cone, capsule, rectangle, disc
    /// and ellipse), [`Shape::from_mesh`] and [`Shape::heightfield`] never build
    /// anything. For every other kind the mesh is built and measured on each call.
    pub fn bounds(&self) -> ShapeBounds {
        match &self.kind {
            ShapeKind::Mesh { bounds, .. } => *bounds,
            ShapeKind::Heightfield { bounds, .. } => bounds.1,
            kind => match closed_form_bounds(kind) {
                Some(bounds) => bounds,
                None => mesh_bounds(&self.build()),
            },
        }
    }

    /// Like [`Shape::bounds`], without the skirts of a heightfield: the bounds
    /// of the land itself. Every other shape returns what [`Shape::bounds`]
    /// does.
    pub fn surface_bounds(&self) -> ShapeBounds {
        match &self.kind {
            ShapeKind::Heightfield { bounds, .. } => bounds.0,
            _ => self.bounds(),
        }
    }

    /// Whether the shape's outline is round: spheres, ellipsoids, discs,
    /// ellipses and rings (and their rims). Useful for ray tests that
    /// approximate such shapes with an ellipsoid inside [`Shape::bounds`].
    pub fn is_round(&self) -> bool {
        if let ShapeKind::Mesh { round, .. } = &self.kind {
            return *round;
        }
        matches!(
            self.kind,
            ShapeKind::Sphere { .. }
                | ShapeKind::Ellipsoid { .. }
                | ShapeKind::Disc { .. }
                | ShapeKind::DiscRim { .. }
                | ShapeKind::Ellipse { .. }
                | ShapeKind::EllipseRim { .. }
                | ShapeKind::Ring { .. }
        )
    }

    /// Like [`Shape::bounds`], for the flat geometry of [`Shape::build_2d`].
    pub fn bounds_2d(&self) -> ShapeBounds {
        match &self.kind {
            ShapeKind::Mesh { bounds, .. } => *bounds,
            ShapeKind::Heightfield { bounds, .. } => bounds.1,
            // `build_2d` returns the ordinary mesh for these kinds.
            kind => match closed_form_bounds(kind) {
                Some(bounds) => bounds,
                None => mesh_bounds(&self.build_2d()),
            },
        }
    }

    /// Proves that this shape is flat, returning a [`FlatShape`] on which the
    /// 2D conversions ([`FlatShape::build_2d`], [`FlatShape::bounds_2d`],
    /// [`FlatShape::sdf_2d`]) exist.
    ///
    /// Flat shapes are the 2D primitives (rectangle, disc, ellipse, polygon,
    /// rims, star, ring, wedge, arbitrary polygon), extruded polygons (drawn
    /// without depth), 2D polylines and Bézier curves, and compounds made only
    /// of flat parts.
    pub fn as_flat(&self) -> Result<FlatShape<'_>, NotFlat> {
        if self.kind.is_flat() {
            Ok(FlatShape { shape: self })
        } else {
            Err(NotFlat)
        }
    }

    /// Builds flat geometry for `Mesh2d`: in the XY plane, facing +Z, with
    /// gradient UVs and no billboarding. Pair it with
    /// [`GradMeshMaterial2d`](crate::GradMeshMaterial2d) (see [`Style::build_2d`](crate::Style::build_2d))
    /// and `Camera2d`.
    ///
    /// The 2D primitives (rectangle, disc, polygon, ring, star, wedge, rims,
    /// arbitrary polygons) return the same geometry as [`Shape::build`], which is
    /// already flat. Extruded polygons return their flat outline without depth,
    /// and compounds and planar-UV wrappers apply this to their parts. Solid 3D
    /// primitives have no flat form and return their ordinary mesh.
    ///
    /// Prefer [`Shape::as_flat`] and [`FlatShape::build_2d`], which refuse the solids
    /// instead of drawing them.
    ///
    /// `Transform` scale on the XY plane resizes the shape and
    /// `Transform.translation.z` orders it against other 2D meshes.
    pub fn build_2d(&self) -> Mesh {
        match &self.kind {
            ShapeKind::ExtrudedPolygon {
                outer,
                holes,
                colors,
                ..
            } => geometry::polygon_mesh(outer, colors.as_deref(), holes, 0.0, 0.0, 1),
            ShapeKind::Compound { parts } => {
                let built: Vec<_> = parts
                    .iter()
                    .map(|p| (p.shape.build_2d(), p.transform, p.tint))
                    .collect();
                geometry::merge_meshes(&built)
            }
            _ => self.build(),
        }
    }

    /// Builds the shape into a Bevy `Mesh`.
    pub fn build(&self) -> Mesh {
        match &self.kind {
            ShapeKind::Sphere { radius } => build_sphere(*radius, self.detail),
            ShapeKind::Ellipsoid { size } => build_ellipsoid(*size, self.detail),
            ShapeKind::Cuboid { size, roundness } => build_cuboid(*size, *roundness, self.detail),
            ShapeKind::Cylinder {
                radius,
                height,
                roundness,
            } => build_cylinder(*radius, *height, *roundness, self.detail),
            ShapeKind::Cone {
                radius,
                height,
                roundness,
            } => {
                if *roundness > 0.0 {
                    build_rounded_cone(*radius, *height, *roundness, self.detail)
                } else {
                    build_cone(*radius, *height, self.detail)
                }
            }
            ShapeKind::Capsule { radius, height } => build_capsule(*radius, *height, self.detail),
            ShapeKind::Hemisphere { radius, capped } => {
                build_hemisphere(*radius, *capped, self.detail)
            }
            ShapeKind::Torus {
                radius,
                thickness,
                start,
                span,
            } => build_torus(*radius, *thickness, *start, *span, self.detail),
            ShapeKind::RegularPrism {
                sides,
                radius,
                height,
                roundness,
            } => build_regular_prism(*sides, *radius, *height, *roundness, self.detail),
            ShapeKind::Pyramid {
                sides,
                radius,
                height,
                roundness,
            } => {
                if *roundness > 0.0 {
                    build_rounded_pyramid(*sides, *radius, *height, *roundness, self.detail)
                } else {
                    build_pyramid(*sides, *radius, *height, self.detail)
                }
            }
            // 2D shapes
            ShapeKind::Rectangle { size } => build_rectangle(*size, self.detail),
            ShapeKind::RectangleRim { size, rim_width } => {
                build_rectangle_rim(*size, *rim_width, self.detail)
            }
            ShapeKind::Disc { radius } => build_disc(*radius, self.detail),
            ShapeKind::DiscRim { radius, rim_width } => {
                build_disc_rim(*radius, *rim_width, self.detail)
            }
            ShapeKind::Polygon { sides, radius } => build_polygon(*sides, *radius, self.detail),
            ShapeKind::PolygonRim {
                sides,
                radius,
                rim_width,
            } => build_polygon_rim(*sides, *radius, *rim_width, self.detail),
            ShapeKind::Ellipse { radius } => build_ellipse(*radius, self.detail),
            ShapeKind::EllipseRim { radius, rim_width } => {
                build_ellipse_rim(*radius, *rim_width, self.detail)
            }
            ShapeKind::ArbitraryPolygon { vertices, colors } => {
                geometry::polygon_mesh(vertices, colors.as_deref(), &[], 0.0, 0.0, 1)
            }
            ShapeKind::Polyline3D {
                points,
                thickness,
                closed,
                segments,
            } => build_polyline3d(points.clone(), *thickness, *closed, *segments),
            ShapeKind::BezierCurve {
                control_points,
                thickness,
                segments,
            } => build_bezier_curve(control_points.clone(), *thickness, *segments),
            ShapeKind::CatmullRom3D {
                points,
                thickness,
                closed,
                segments,
            } => build_catmull_rom_3d(points.clone(), *thickness, *closed, *segments),
            ShapeKind::Star {
                points,
                outer_radius,
                inner_radius,
            } => build_star(*points, *outer_radius, *inner_radius, self.detail),
            ShapeKind::Ring {
                outer_radius,
                inner_radius,
            } => build_ring(*outer_radius, *inner_radius, self.detail),
            ShapeKind::Wedge { radius, angle } => build_wedge(*radius, *angle, self.detail),
            ShapeKind::TaperedCylinder {
                top_radius,
                bottom_radius,
                height,
            } => build_tapered_cylinder(*top_radius, *bottom_radius, *height, self.detail),
            ShapeKind::ExtrudedPolygon {
                outer,
                holes,
                colors,
                depth,
                roundness,
            } => geometry::polygon_mesh(
                outer,
                colors.as_deref(),
                holes,
                *depth,
                *roundness,
                bevel_steps(self.detail),
            ),
            ShapeKind::Compound { parts } => {
                let built: Vec<_> = parts
                    .iter()
                    .map(|p| (p.shape.build(), p.transform, p.tint))
                    .collect();
                geometry::merge_meshes(&built)
            }
            ShapeKind::Lathe { profile, segments } => geometry::lathe_mesh(
                profile,
                if *segments == 0 {
                    self.detail.segments()
                } else {
                    *segments
                },
            ),
            ShapeKind::Polyline2D {
                points,
                thickness,
                closed,
                segments: _,
            } => geometry::polyline_2d_mesh(points, *thickness, *closed),
            ShapeKind::Bezier2D {
                control_points,
                thickness,
                segments,
            } => geometry::polyline_2d_mesh(
                &geometry::bezier_2d_points(control_points, *segments),
                *thickness,
                false,
            ),
            ShapeKind::Mesh { mesh, .. } => Mesh::clone(mesh),
            ShapeKind::Sweep {
                profile,
                path,
                frames,
                options,
            } => geometry::sweep_mesh(profile, path, frames.as_deref(), options),
            ShapeKind::Heightfield {
                size,
                extent,
                heights,
                holes,
                hole_edge,
                border,
                ..
            } => geometry::heightfield_mesh(
                *size,
                *extent,
                heights,
                holes.as_deref(),
                *hole_edge,
                border.as_deref(),
            ),
        }
    }
}

// --- Mesh builders ---

fn build_sphere(radius: f32, detail: ShapeDetail) -> Mesh {
    let segments = detail.segments() as usize;
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for lat in 0..=segments {
        let theta = lat as f32 * PI / segments as f32;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        for lon in 0..=segments {
            let phi = lon as f32 * 2.0 * PI / segments as f32;
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();

            let nx = cos_phi * sin_theta;
            let ny = cos_theta;
            let nz = sin_phi * sin_theta;

            vertices.push(Vec3::new(nx, ny, nz) * radius);
            normals.push(Vec3::new(nx, ny, nz));
            uvs.push(Vec2::new(
                lon as f32 / segments as f32,
                lat as f32 / segments as f32,
            ));
        }
    }

    for lat in 0..segments {
        for lon in 0..segments {
            let first = lat * (segments + 1) + lon;
            let second = first + segments + 1;
            indices.extend_from_slice(&[first as u32, second as u32, (first + 1) as u32]);
            indices.extend_from_slice(&[second as u32, (second + 1) as u32, (first + 1) as u32]);
        }
    }

    build_mesh(vertices, normals, uvs, indices)
}

fn build_ellipsoid(size: Vec3, detail: ShapeDetail) -> Mesh {
    let segments = detail.segments() as usize;
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let half = size / 2.0;

    for lat in 0..=segments {
        let theta = lat as f32 * PI / segments as f32;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        for lon in 0..=segments {
            let phi = lon as f32 * 2.0 * PI / segments as f32;
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();

            let nx = cos_phi * sin_theta;
            let ny = cos_theta;
            let nz = sin_phi * sin_theta;

            vertices.push(Vec3::new(nx * half.x, ny * half.y, nz * half.z));
            // Normalize ellipsoid normal
            let len = (nx * nx / (half.x * half.x)
                + ny * ny / (half.y * half.y)
                + nz * nz / (half.z * half.z))
                .sqrt();
            normals.push(
                Vec3::new(
                    nx / (half.x * half.x * len),
                    ny / (half.y * half.y * len),
                    nz / (half.z * half.z * len),
                )
                .normalize(),
            );
            uvs.push(Vec2::new(
                lon as f32 / segments as f32,
                lat as f32 / segments as f32,
            ));
        }
    }

    for lat in 0..segments {
        for lon in 0..segments {
            let first = lat * (segments + 1) + lon;
            let second = first + segments + 1;
            indices.extend_from_slice(&[first as u32, second as u32, (first + 1) as u32]);
            indices.extend_from_slice(&[second as u32, (second + 1) as u32, (first + 1) as u32]);
        }
    }

    build_mesh(vertices, normals, uvs, indices)
}

fn build_cuboid(size: Vec3, roundness: f32, detail: ShapeDetail) -> Mesh {
    // For simplicity, use Bevy's rounded box approach: generate a box with chamfered edges
    // when roundness > 0. For roundness = 0, generate a simple box.
    if roundness <= 0.0 {
        build_simple_box(size)
    } else {
        // Rounded cuboid: subdivide each face based on detail and displace vertices
        let segments = (detail.segments() * roundness as u32).max(2) as usize;
        build_rounded_box(size, roundness, segments)
    }
}

fn build_simple_box(size: Vec3) -> Mesh {
    let (hx, hy, hz) = (size.x / 2.0, size.y / 2.0, size.z / 2.0);

    // 24 vertices (one per face corner, no sharing for correct normals)
    let vertices: [Vec3; 24] = [
        // Front face
        Vec3::new(-hx, -hy, hz),
        Vec3::new(hx, -hy, hz),
        Vec3::new(hx, hy, hz),
        Vec3::new(-hx, hy, hz),
        // Back face
        Vec3::new(hx, -hy, -hz),
        Vec3::new(-hx, -hy, -hz),
        Vec3::new(-hx, hy, -hz),
        Vec3::new(hx, hy, -hz),
        // Top face
        Vec3::new(-hx, hy, -hz),
        Vec3::new(-hx, hy, hz),
        Vec3::new(hx, hy, hz),
        Vec3::new(hx, hy, -hz),
        // Bottom face
        Vec3::new(-hx, -hy, hz),
        Vec3::new(hx, -hy, hz),
        Vec3::new(hx, -hy, -hz),
        Vec3::new(-hx, -hy, -hz),
        // Right face
        Vec3::new(hx, -hy, -hz),
        Vec3::new(hx, -hy, hz),
        Vec3::new(hx, hy, hz),
        Vec3::new(hx, hy, -hz),
        // Left face
        Vec3::new(-hx, -hy, hz),
        Vec3::new(-hx, -hy, -hz),
        Vec3::new(-hx, hy, -hz),
        Vec3::new(-hx, hy, hz),
    ];

    let normals: [Vec3; 6] = [
        Vec3::new(0.0, 0.0, 1.0),  // Front
        Vec3::new(0.0, 0.0, -1.0), // Back
        Vec3::new(0.0, 1.0, 0.0),  // Top
        Vec3::new(0.0, -1.0, 0.0), // Bottom
        Vec3::new(1.0, 0.0, 0.0),  // Right
        Vec3::new(-1.0, 0.0, 0.0), // Left
    ];

    let mut all_normals = Vec::new();
    for face in 0..6 {
        for _ in 0..4 {
            all_normals.push(normals[face]);
        }
    }

    let uvs: [Vec2; 24] = std::array::from_fn(|i| {
        let corner = i % 4;
        Vec2::new(
            if corner & 1 != 0 { 1.0 } else { 0.0 },
            if corner & 2 != 0 { 1.0 } else { 0.0 },
        )
    });

    // All triangles use CCW winding when viewed from outside the box.
    // Bottom, Right, and Left faces need reversed winding for outward normals.
    let indices: [u32; 36] = [
        0, 1, 2, 2, 3, 0, // Front (+Z)
        4, 5, 6, 6, 7, 4, // Back (-Z)
        8, 9, 10, 10, 11, 8, // Top (+Y)
        12, 14, 13, 14, 12, 15, // Bottom (-Y) - reversed winding
        16, 18, 17, 18, 16, 19, // Right (+X) - reversed winding
        20, 22, 21, 22, 20, 23, // Left (-X) - reversed winding
    ];

    build_mesh(
        vertices.to_vec(),
        all_normals,
        uvs.to_vec(),
        indices.to_vec(),
    )
}

fn build_rounded_box(size: Vec3, roundness: f32, segments: usize) -> Mesh {
    // Rounded cuboid using a fillet-based rounding algorithm.
    //
    // Each face is generated as a grid where the edge coordinates concentrate
    // vertices in the fillet regions. Each vertex starts at a cube corner position,
    // then is rounded by clamping toward the face center and placing the result on
    // an elliptical arc from the clamped center to the original corner.
    //
    // The radii are capped at 0.25 per axis (MAXIMUM_NORMALIZED_FILLET_RADIUS) so
    // fillets never overlap. At roundness=0.0 the box is sharp; at roundness=1.0
    // each corner is rounded with radius 0.25 of the unit box.
    //
    // Algorithm (per vertex):
    //   1. Compute cube_point = axis_u * u + axis_v * v + normal * 0.5
    //   2. Clamp cube_point to [-0.5+rx, 0.5-rx] etc. -> center
    //   3. direction = normalize((cube_point - center) / radii)
    //   4. position = center + radii * direction
    //   5. normal = normalize(direction / radii)
    //
    // This produces true spherical/elliptical fillets at edges and corners while
    // preserving flat face centers.

    const MAXIMUM_NORMALIZED_FILLET_RADIUS: f32 = 0.25;
    let r = roundness.clamp(0.0, 1.0);

    // Per-axis radii in unit-box space (box spans [-0.5, 0.5] on each axis).
    // At r=0.0, radii=0 -> sharp box. At r=1.0, radii=0.25 -> maximum fillet.
    let radii = Vec3::splat((r * 0.5).min(MAXIMUM_NORMALIZED_FILLET_RADIUS));

    // Divisions per fillet edge (how many subdivisions in each rounded corner region).
    // Derived from the segment count.
    let divisions = (segments / 2).max(1);

    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Define the 6 faces:
    // (axis_u, axis_v, normal) where normal points outward from the face.
    let faces: [(Vec3, Vec3, Vec3); 6] = [
        // Back face (-Z)
        (Vec3::X, Vec3::Y, Vec3::new(0.0, 0.0, -1.0)),
        // Front face (+Z)
        (Vec3::new(-1.0, 0.0, 0.0), Vec3::Y, Vec3::Z),
        // Left face (-X)
        (
            Vec3::Z,
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
        ),
        // Right face (+X)
        (
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::X,
        ),
        // Bottom face (-Y)
        (
            Vec3::X,
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, -1.0, 0.0),
        ),
        // Top face (+Y)
        (
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::Y,
        ),
    ];

    for &(axis_u, axis_v, normal) in faces.iter() {
        let radius_u = axis_value(&axis_u, &radii);
        let radius_v = axis_value(&axis_v, &radii);

        let edge_vertices = divisions * 2 + 2;
        let base_idx = vertices.len() as u32;

        // Generate grid vertices for this face
        for row in 0..edge_vertices {
            let v = rounded_box_edge_coordinate(row, radius_v, divisions);
            for column in 0..edge_vertices {
                let u = rounded_box_edge_coordinate(column, radius_u, divisions);

                // Cube point on the unit box face (before rounding)
                let cube_point = axis_u * u + axis_v * v + normal * 0.5;

                // Apply rounding: clamp toward center, then place on elliptical arc
                let (pos, norm) = rounded_box_round_point(&cube_point, &radii, &normal);

                // Scale from unit box to actual size
                let scaled_pos = pos * size;

                vertices.push(scaled_pos);
                normals.push(norm);
                uvs.push(Vec2::new(u + 0.5, v + 0.5));
            }
        }

        // Generate indices for this face.
        // Triangle order: idx, idx+edge_vertices, idx+1
        // and idx+1, idx+edge_vertices, idx+edge_vertices+1.
        // In Bevy's right-handed coordinate system, this produces CCW winding
        // for all faces (axis_v x axis_u points in the face normal direction).
        for row in 0..edge_vertices - 1 {
            for column in 0..edge_vertices - 1 {
                let idx = base_idx + row as u32 * edge_vertices as u32 + column as u32;
                indices.extend_from_slice(&[
                    idx,
                    idx + edge_vertices as u32,
                    idx + 1,
                    idx + 1,
                    idx + edge_vertices as u32,
                    idx + edge_vertices as u32 + 1,
                ]);
            }
        }
    }

    let mut mesh = build_mesh(vertices, normals, uvs, indices);
    // Recompute normals from the rounded geometry to ensure they match the
    // actual vertex positions. The analytical normals from _rounded_box_round_point
    // are close but not exact for the deformed mesh.
    mesh.compute_normals();
    mesh
}

/// Extract the radii component corresponding to the given axis.
/// Picks the x, y, or z component based on which axis component is nonzero.
fn axis_value(axis: &Vec3, values: &Vec3) -> f32 {
    if axis.x.abs() > 0.5 {
        values.x
    } else if axis.y.abs() > 0.5 {
        values.y
    } else {
        values.z
    }
}

/// Compute the coordinate along one edge of a rounded box face.
///
/// The edge has `divisions*2+2` vertices distributed across three regions:
/// - First `divisions+1` steps (index 0..=divisions): fillet from -0.5 to -0.5 + radius
/// - Remaining steps (index divisions+1..=2*divisions+1): flat region + second fillet
///   from 0.5 - radius to 0.5
///
/// Maps a grid index to a coordinate along a rounded edge, concentrating
fn rounded_box_edge_coordinate(index: usize, radius: f32, divisions: usize) -> f32 {
    if index <= divisions {
        // Fillet region: from -0.5 to -0.5 + radius
        -0.5 + radius * (index as f32) / (divisions as f32)
    } else {
        // Flat region + second fillet: from 0.5 - radius to 0.5
        0.5 + radius * ((index as f32) - (divisions as f32) * 2.0 - 1.0) / (divisions as f32)
    }
}

/// Round a point on a unit cube face by clamping toward center and placing on an
/// elliptical arc. Returns (position, normal).
///
/// Computes the rounded position and normal for a vertex on a rounded box face.
/// When the point is in the flat region (cube_point == center after clamping),
/// the face_normal parameter provides the fallback normal.
fn rounded_box_round_point(cube_point: &Vec3, radii: &Vec3, face_normal: &Vec3) -> (Vec3, Vec3) {
    // If radii is zero, no rounding: return the cube point with the face normal.
    let epsilon = 1e-10;
    if radii.x < epsilon && radii.y < epsilon && radii.z < epsilon {
        return (*cube_point, *face_normal);
    }

    // Clamp the cube point toward the center, leaving room for the fillet radius.
    let center = Vec3::new(
        cube_point.x.clamp(-0.5 + radii.x, 0.5 - radii.x),
        cube_point.y.clamp(-0.5 + radii.y, 0.5 - radii.y),
        cube_point.z.clamp(-0.5 + radii.z, 0.5 - radii.z),
    );

    // Direction from clamped center to the original cube point, normalized by radii.
    // This gives a point on the unit sphere in radii-scaled space.
    let offset = Vec3::new(
        (cube_point.x - center.x) / radii.x,
        (cube_point.y - center.y) / radii.y,
        (cube_point.z - center.z) / radii.z,
    );
    let direction = offset.normalize_or(*face_normal);

    // Position on the elliptical arc: center + radii * direction
    let position = Vec3::new(
        center.x + radii.x * direction.x,
        center.y + radii.y * direction.y,
        center.z + radii.z * direction.z,
    );

    // Normal: normalize(direction / radii). For uniform radii this equals direction.
    // When in the flat region (direction == face_normal fallback), use the face normal.
    let normal = Vec3::new(
        direction.x / radii.x,
        direction.y / radii.y,
        direction.z / radii.z,
    )
    .normalize_or(*face_normal);

    (position, normal)
}

fn build_cylinder(radius: f32, height: f32, _roundness: f32, detail: ShapeDetail) -> Mesh {
    let segments = detail.segments() as usize;
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let half_h = height / 2.0;

    // Side vertices
    for i in 0..=segments {
        let angle = i as f32 * 2.0 * PI / segments as f32;
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        // Top ring
        vertices.push(Vec3::new(cos_a * radius, half_h, sin_a * radius));
        normals.push(Vec3::new(cos_a, 0.0, sin_a));
        uvs.push(Vec2::new(i as f32 / segments as f32, 1.0));

        // Bottom ring
        vertices.push(Vec3::new(cos_a * radius, -half_h, sin_a * radius));
        normals.push(Vec3::new(cos_a, 0.0, sin_a));
        uvs.push(Vec2::new(i as f32 / segments as f32, 0.0));
    }

    let _side_base = 0;
    // Side indices
    for i in 0..segments {
        let a = i * 2;
        let b = a + 1;
        let c = a + 2;
        let d = a + 3;
        indices.extend_from_slice(&[a as u32, c as u32, b as u32, b as u32, c as u32, d as u32]);
    }

    // Top cap
    let top_center = vertices.len() as u32;
    vertices.push(Vec3::new(0.0, half_h, 0.0));
    normals.push(Vec3::new(0.0, 1.0, 0.0));
    uvs.push(Vec2::new(0.5, 0.5));

    for i in 0..=segments {
        let angle = i as f32 * 2.0 * PI / segments as f32;
        vertices.push(Vec3::new(
            angle.cos() * radius,
            half_h,
            angle.sin() * radius,
        ));
        normals.push(Vec3::new(0.0, 1.0, 0.0));
        uvs.push(Vec2::new(0.5 + angle.cos() * 0.5, 0.5 + angle.sin() * 0.5));
    }

    for i in 0..segments {
        indices.extend_from_slice(&[
            top_center,
            top_center + 1 + i as u32,
            top_center + 2 + i as u32,
        ]);
    }

    // Bottom cap
    let bottom_center = vertices.len() as u32;
    vertices.push(Vec3::new(0.0, -half_h, 0.0));
    normals.push(Vec3::new(0.0, -1.0, 0.0));
    uvs.push(Vec2::new(0.5, 0.5));

    for i in 0..=segments {
        let angle = i as f32 * 2.0 * PI / segments as f32;
        vertices.push(Vec3::new(
            angle.cos() * radius,
            -half_h,
            angle.sin() * radius,
        ));
        normals.push(Vec3::new(0.0, -1.0, 0.0));
        uvs.push(Vec2::new(0.5 + angle.cos() * 0.5, 0.5 + angle.sin() * 0.5));
    }

    for i in 0..segments {
        indices.extend_from_slice(&[
            bottom_center,
            bottom_center + 2 + i as u32,
            bottom_center + 1 + i as u32,
        ]);
    }

    build_mesh(vertices, normals, uvs, indices)
}

fn build_cone(radius: f32, height: f32, detail: ShapeDetail) -> Mesh {
    let segments = detail.segments() as usize;
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let half_h = height / 2.0;

    // Compute the side normal: the cone side slopes inward from base to apex.
    // The normal points outward, perpendicular to the side surface.
    // Slope angle: from apex (0, half_h) to base edge (radius, -half_h)
    // The side vector is (radius, -height), so the outward normal is (height, radius) normalized
    let side_nx = height;
    let side_ny = -radius;
    let side_len = (side_nx * side_nx + side_ny * side_ny).sqrt();
    let side_nx = side_nx / side_len;
    let side_ny = side_ny / side_len;

    // Apex
    vertices.push(Vec3::new(0.0, half_h, 0.0));
    normals.push(Vec3::new(0.0, 1.0, 0.0));
    uvs.push(Vec2::new(0.5, 1.0));

    // Base ring
    for i in 0..=segments {
        let angle = i as f32 * 2.0 * PI / segments as f32;
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        vertices.push(Vec3::new(cos_a * radius, -half_h, sin_a * radius));

        // Side normal: rotate the side normal around Y axis by the vertex angle
        normals.push(Vec3::new(cos_a * side_nx, side_ny, sin_a * side_nx).normalize());

        uvs.push(Vec2::new(i as f32 / segments as f32, 0.0));
    }

    // Side triangles
    for i in 0..segments {
        indices.extend_from_slice(&[0, 1 + i as u32, 2 + i as u32]);
    }

    // Base cap
    let base_center = vertices.len() as u32;
    vertices.push(Vec3::new(0.0, -half_h, 0.0));
    normals.push(Vec3::new(0.0, -1.0, 0.0));
    uvs.push(Vec2::new(0.5, 0.5));

    for i in 0..=segments {
        let angle = i as f32 * 2.0 * PI / segments as f32;
        vertices.push(Vec3::new(
            angle.cos() * radius,
            -half_h,
            angle.sin() * radius,
        ));
        normals.push(Vec3::new(0.0, -1.0, 0.0));
        uvs.push(Vec2::new(0.5 + angle.cos() * 0.5, 0.5 + angle.sin() * 0.5));
    }

    for i in 0..segments {
        indices.extend_from_slice(&[
            base_center,
            base_center + 2 + i as u32,
            base_center + 1 + i as u32,
        ]);
    }

    build_mesh(vertices, normals, uvs, indices)
}

fn build_capsule(radius: f32, height: f32, detail: ShapeDetail) -> Mesh {
    let segments = detail.segments() as usize;
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let cylinder_height = (height - radius * 2.0).max(0.0);
    let half_cyl = cylinder_height / 2.0;

    // Generate capsule by stacking: top hemisphere + cylinder + bottom hemisphere
    // Use equal angular steps for smooth transitions
    let lat_segments = (segments / 2).max(1); // half for each hemisphere

    let ring_verts = segments + 1;

    // Top hemisphere (from pole down to equator)
    // lat=0 is the pole, lat=lat_segments is the equator at y=half_cyl
    for lat in 0..=lat_segments {
        let theta = lat as f32 * PI / 2.0 / lat_segments as f32;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        for lon in 0..=segments {
            let phi = lon as f32 * 2.0 * PI / segments as f32;
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();

            let nx = cos_phi * sin_theta;
            let ny = cos_theta;
            let nz = sin_phi * sin_theta;

            let y = half_cyl + radius * ny;
            vertices.push(Vec3::new(nx * radius, y, nz * radius));
            normals.push(Vec3::new(nx, ny, nz));
            uvs.push(Vec2::new(
                lon as f32 / segments as f32,
                1.0 - lat as f32 / (lat_segments as f32 * 2.0 + 1.0),
            ));
        }
    }

    // Cylinder middle section - two rings: top ring first, then bottom ring
    let cyl_top_start = vertices.len() / ring_verts;
    for lon in 0..=segments {
        let phi = lon as f32 * 2.0 * PI / segments as f32;
        let cos_phi = phi.cos();
        let sin_phi = phi.sin();

        vertices.push(Vec3::new(cos_phi * radius, half_cyl, sin_phi * radius));
        normals.push(Vec3::new(cos_phi, 0.0, sin_phi));
        uvs.push(Vec2::new(lon as f32 / segments as f32, 0.5));
    }
    for lon in 0..=segments {
        let phi = lon as f32 * 2.0 * PI / segments as f32;
        let cos_phi = phi.cos();
        let sin_phi = phi.sin();

        vertices.push(Vec3::new(cos_phi * radius, -half_cyl, sin_phi * radius));
        normals.push(Vec3::new(cos_phi, 0.0, sin_phi));
        uvs.push(Vec2::new(lon as f32 / segments as f32, 0.4));
    }

    // Bottom hemisphere (from equator down to pole)
    // lat=0 is the equator at y=-half_cyl, lat=lat_segments is the pole
    let bottom_start = vertices.len() / ring_verts;
    for lat in 0..=lat_segments {
        let theta = PI / 2.0 + lat as f32 * PI / 2.0 / lat_segments as f32;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        for lon in 0..=segments {
            let phi = lon as f32 * 2.0 * PI / segments as f32;
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();

            let nx = cos_phi * sin_theta;
            let ny = cos_theta;
            let nz = sin_phi * sin_theta;

            let y = -half_cyl + radius * ny;
            vertices.push(Vec3::new(nx * radius, y, nz * radius));
            normals.push(Vec3::new(nx, ny, nz));
            uvs.push(Vec2::new(
                lon as f32 / segments as f32,
                lat as f32 / (lat_segments as f32 * 2.0 + 1.0),
            ));
        }
    }

    // Build indices for all sections

    // Top hemisphere indices
    for lat in 0..lat_segments {
        for lon in 0..segments {
            let a = lat * ring_verts + lon;
            let b = a + ring_verts;
            indices.extend_from_slice(&[a as u32, b as u32, (a + 1) as u32]);
            indices.extend_from_slice(&[b as u32, (b + 1) as u32, (a + 1) as u32]);
        }
    }

    // Cylinder indices - connects top ring to bottom ring
    let cyl_bot_start = (cyl_top_start + 1) * ring_verts;
    for lon in 0..segments {
        let a = cyl_top_start * ring_verts + lon;
        let b = cyl_bot_start + lon;
        let c = cyl_bot_start + lon + 1;
        let d = cyl_top_start * ring_verts + lon + 1;
        indices.extend_from_slice(&[a as u32, b as u32, d as u32]);
        indices.extend_from_slice(&[b as u32, c as u32, d as u32]);
    }

    // Bottom hemisphere indices
    for lat in 0..lat_segments {
        for lon in 0..segments {
            let a = bottom_start * ring_verts + lat * ring_verts + lon;
            let b = a + ring_verts;
            indices.extend_from_slice(&[a as u32, b as u32, (a + 1) as u32]);
            indices.extend_from_slice(&[b as u32, (b + 1) as u32, (a + 1) as u32]);
        }
    }

    build_mesh(vertices, normals, uvs, indices)
}

fn build_hemisphere(radius: f32, capped: bool, detail: ShapeDetail) -> Mesh {
    let segments = detail.segments() as usize;
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Generate upper half of sphere (theta from 0 to PI/2)
    let lat_segments = segments / 2;

    for lat in 0..=lat_segments {
        let theta = lat as f32 * PI / 2.0 / lat_segments as f32;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        for lon in 0..=segments {
            let phi = lon as f32 * 2.0 * PI / segments as f32;
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();

            let nx = cos_phi * sin_theta;
            let ny = cos_theta;
            let nz = sin_phi * sin_theta;

            vertices.push(Vec3::new(nx * radius, ny * radius, nz * radius));
            normals.push(Vec3::new(nx, ny, nz));
            uvs.push(Vec2::new(
                lon as f32 / segments as f32,
                lat as f32 / lat_segments as f32,
            ));
        }
    }

    let ring_verts = segments + 1;
    for lat in 0..lat_segments {
        for lon in 0..segments {
            let a = lat * ring_verts + lon;
            let b = a + ring_verts;
            indices.extend_from_slice(&[a as u32, b as u32, (a + 1) as u32]);
            indices.extend_from_slice(&[b as u32, (b + 1) as u32, (a + 1) as u32]);
        }
    }

    // Flat cap
    if capped {
        let cap_center = vertices.len() as u32;
        vertices.push(Vec3::ZERO);
        normals.push(Vec3::new(0.0, -1.0, 0.0));
        uvs.push(Vec2::new(0.5, 0.5));

        for i in 0..=segments {
            let angle = i as f32 * 2.0 * PI / segments as f32;
            vertices.push(Vec3::new(angle.cos() * radius, 0.0, angle.sin() * radius));
            normals.push(Vec3::new(0.0, -1.0, 0.0));
            uvs.push(Vec2::new(0.5 + angle.cos() * 0.5, 0.5 + angle.sin() * 0.5));
        }

        for i in 0..segments {
            indices.extend_from_slice(&[
                cap_center,
                cap_center + 2 + i as u32,
                cap_center + 1 + i as u32,
            ]);
        }
    }

    build_mesh(vertices, normals, uvs, indices)
}

fn build_torus(radius: f32, thickness: f32, start: f32, span: f32, detail: ShapeDetail) -> Mesh {
    let major_segments = detail.segments() as usize;
    let minor_segments = (detail.segments() / 2).max(4) as usize;
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let _end = start + span;

    for i in 0..=major_segments {
        let u = i as f32 / major_segments as f32;
        let theta = start + u * span;
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        for j in 0..=minor_segments {
            let v = j as f32 / minor_segments as f32;
            let phi = v * 2.0 * PI;
            let cos_phi = phi.cos();
            let sin_phi = phi.sin();

            let x = (radius + thickness * cos_phi) * cos_theta;
            let y = thickness * sin_phi;
            let z = (radius + thickness * cos_phi) * sin_theta;

            vertices.push(Vec3::new(x, y, z));
            normals.push(Vec3::new(cos_phi * cos_theta, sin_phi, cos_phi * sin_theta));
            uvs.push(Vec2::new(u, v));
        }
    }

    let ring_verts = minor_segments + 1;
    for i in 0..major_segments {
        for j in 0..minor_segments {
            let a = i * ring_verts + j;
            let b = a + ring_verts;
            indices.extend_from_slice(&[a as u32, b as u32, (a + 1) as u32]);
            indices.extend_from_slice(&[b as u32, (b + 1) as u32, (a + 1) as u32]);
        }
    }

    // Cap the tube ends when the arc is partial (span < TAU)
    if span < std::f32::consts::TAU - 1e-6 {
        // Start cap - fan from center of first ring
        let start_center = vertices.len();
        // Compute center of the tube cross-section at start angle
        let cos_start = start.cos();
        let sin_start = start.sin();
        let center_pos = Vec3::new(radius * cos_start, 0.0, radius * sin_start);
        let center_normal = Vec3::new(cos_start, 0.0, sin_start);
        vertices.push(center_pos);
        normals.push(center_normal);
        uvs.push(Vec2::new(0.5, 0.5));

        for j in 0..=minor_segments {
            let _v = j as f32 / minor_segments as f32;
            let _phi = _v * 2.0 * PI;
            // Order: outer ring vertices in CCW order
            indices.extend_from_slice(&[
                start_center as u32,
                j as u32,
                (j + 1) as u32 % (minor_segments + 1) as u32,
            ]);
        }

        // End cap - fan from center of last ring
        let end_center = vertices.len();
        let cos_end = (start + span).cos();
        let sin_end = (start + span).sin();
        let end_center_pos = Vec3::new(radius * cos_end, 0.0, radius * sin_end);
        let end_center_normal = Vec3::new(cos_end, 0.0, sin_end);
        vertices.push(end_center_pos);
        normals.push(end_center_normal);
        uvs.push(Vec2::new(0.5, 0.5));

        let last_ring_start = major_segments * ring_verts;
        for j in 0..minor_segments {
            // Reverse winding for end cap (facing outward)
            indices.extend_from_slice(&[
                end_center as u32,
                (last_ring_start + j + 1) as u32,
                (last_ring_start + j) as u32,
            ]);
        }
    }

    build_mesh(vertices, normals, uvs, indices)
}

/// Sweeps a rounded profile around a regular polygon (or a circle when
/// `smooth`), producing the inscribed solid "core xor ball of radius `rho`".
///
/// `core` lists the profile's corner points as `(apothem radius, y)` from the
/// bottom axis point to the top axis point, counter-clockwise in the (r, y)
/// half plane. Every corner of the core becomes an arc of the final profile
/// and, for polygons, every vertical polygon corner becomes an arc too, so
/// the surface is smooth where edges used to be sharp.
fn build_rounded_loft(
    sides: usize,
    smooth: bool,
    core: &[(f32, f32)],
    rho: f32,
    detail: ShapeDetail,
) -> Mesh {
    use std::f32::consts::TAU;

    let arc_steps = bevel_steps(detail) as usize;
    let cos_half = if smooth {
        1.0
    } else {
        (PI / sides as f32).cos()
    };

    // Drop coincident neighbours so degenerate cores (full roundness) work.
    let mut pts: Vec<(f32, f32)> = Vec::with_capacity(core.len());
    for &p in core {
        if pts
            .last()
            .is_none_or(|q| (q.0 - p.0).hypot(q.1 - p.1) > 1e-5)
        {
            pts.push(p);
        }
    }
    let edge_normals: Vec<(f32, f32)> = pts
        .windows(2)
        .map(|w| {
            let (dx, dy) = (w[1].0 - w[0].0, w[1].1 - w[0].1);
            let len = dx.hypot(dy);
            (dy / len, -dx / len)
        })
        .collect();

    // Profile: (circumradius, y, normal radial, normal y) bottom to top.
    let mut profile: Vec<(f32, f32, f32, f32)> = Vec::new();
    for (k, &(x, y)) in pts.iter().enumerate() {
        let n_in = if k == 0 {
            (0.0, -1.0)
        } else {
            edge_normals[k - 1]
        };
        let n_out = if k + 1 == pts.len() {
            (0.0, 1.0)
        } else {
            edge_normals[k]
        };
        let a_in = n_in.1.atan2(n_in.0);
        let a_out = n_out.1.atan2(n_out.0);
        let mut delta = (a_out - a_in).rem_euclid(TAU);
        if delta > TAU - 1e-4 {
            delta = 0.0;
        }
        let steps = if delta < 1e-4 {
            0
        } else {
            ((delta / (PI * 0.5)) * arc_steps as f32).ceil().max(1.0) as usize
        };
        for s in 0..=steps {
            let a = a_in + delta * s as f32 / steps.max(1) as f32;
            profile.push((x / cos_half, y, a.cos(), a.sin()));
        }
    }

    // Ring layout: (corner angle, normal angle, u).
    let mut ring: Vec<(f32, f32)> = Vec::new();
    if smooth {
        for i in 0..sides {
            let a = i as f32 * TAU / sides as f32;
            ring.push((a, a));
        }
    } else {
        let half = PI / sides as f32;
        for i in 0..sides {
            let a = i as f32 * TAU / sides as f32;
            for j in 0..=arc_steps {
                ring.push((a, a - half + 2.0 * half * j as f32 / arc_steps as f32));
            }
        }
    }
    let ring_len = ring.len();
    let seam = ring[0];
    ring.push(seam);

    let mut vertices = Vec::with_capacity(profile.len() * ring.len());
    let mut normals = Vec::with_capacity(vertices.capacity());
    let mut uvs = Vec::with_capacity(vertices.capacity());
    let (mut y_min, mut y_max) = (f32::MAX, f32::MIN);
    for &(_, y, _, ny) in &profile {
        y_min = y_min.min(y + rho * ny);
        y_max = y_max.max(y + rho * ny);
    }
    let y_span = (y_max - y_min).max(1e-6);
    for &(c, y, nr, ny) in &profile {
        for (e, &(a, theta)) in ring.iter().enumerate() {
            let (ct, st) = (theta.cos(), theta.sin());
            let pos = Vec3::new(
                c * a.cos() + rho * nr * ct,
                y + rho * ny,
                c * a.sin() + rho * nr * st,
            );
            vertices.push(pos);
            normals.push(Vec3::new(nr * ct, ny, nr * st).normalize_or(Vec3::Y));
            uvs.push(Vec2::new(
                e as f32 / ring_len as f32,
                (pos.y - y_min) / y_span,
            ));
        }
    }

    let stride = ring.len();
    let mut indices: Vec<u32> = Vec::new();
    // Slivers whose area is float noise (collapsed rings at the poles) are
    // dropped; the threshold scales with the size of the solid.
    let reach = profile.iter().fold(y_span, |m, p| m.max(p.0 + rho));
    let min_area_sq = 1e-10 * reach.powi(4);
    let mut push_tri = |a: usize, b: usize, c: usize| {
        let area = (vertices[b] - vertices[a])
            .cross(vertices[c] - vertices[a])
            .length_squared();
        if area > min_area_sq {
            indices.extend_from_slice(&[a as u32, b as u32, c as u32]);
        }
    };
    for k in 0..profile.len().saturating_sub(1) {
        for e in 0..stride - 1 {
            let lo = k * stride + e;
            let hi = (k + 1) * stride + e;
            push_tri(hi, hi + 1, lo);
            push_tri(lo, hi + 1, lo + 1);
        }
    }

    build_mesh(vertices, normals, uvs, indices)
}

/// Core of a rounded tapering solid (cone or pyramid) in apothem space: the
/// profile triangle with its base and sloped side moved inwards by `rho`.
/// Returns `(core, rho)`.
fn tapered_core(apothem: f32, height: f32, roundness: f32) -> (Vec<(f32, f32)>, f32) {
    let s = apothem.hypot(height);
    let rho_max = height * apothem / (apothem + s);
    let rho = roundness.clamp(0.0, 1.0) * rho_max;
    let half = height / 2.0;
    let base_x = (apothem - rho * (s + apothem) / height).max(0.0);
    let apex_y = (half - rho * s / apothem).max(-half + rho);
    (
        vec![(0.0, -half + rho), (base_x, -half + rho), (0.0, apex_y)],
        rho,
    )
}

fn build_rounded_cone(radius: f32, height: f32, roundness: f32, detail: ShapeDetail) -> Mesh {
    let (core, rho) = tapered_core(radius, height, roundness);
    build_rounded_loft(detail.segments() as usize, true, &core, rho, detail)
}

fn build_rounded_pyramid(
    sides: u32,
    radius: f32,
    height: f32,
    roundness: f32,
    detail: ShapeDetail,
) -> Mesh {
    let sides = sides.max(3) as usize;
    let apothem = radius * (PI / sides as f32).cos();
    let (core, rho) = tapered_core(apothem, height, roundness);
    build_rounded_loft(sides, false, &core, rho, detail)
}

fn build_rounded_prism(
    sides: u32,
    radius: f32,
    height: f32,
    roundness: f32,
    detail: ShapeDetail,
) -> Mesh {
    let sides = sides.max(3) as usize;
    let apothem = radius * (PI / sides as f32).cos();
    let half = height / 2.0;
    let rho = roundness.clamp(0.0, 1.0) * apothem.min(half);
    let (cx, cy) = ((apothem - rho).max(0.0), (half - rho).max(0.0));
    let core = [(0.0, -cy), (cx, -cy), (cx, cy), (0.0, cy)];
    build_rounded_loft(sides, false, &core, rho, detail)
}

fn build_regular_prism(
    sides: u32,
    radius: f32,
    height: f32,
    roundness: f32,
    detail: ShapeDetail,
) -> Mesh {
    if roundness > 0.0 {
        return build_rounded_prism(sides, radius, height, roundness, detail);
    }
    let sides = sides.max(3) as usize;
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let half_h = height / 2.0;

    // Side vertices
    for i in 0..=sides {
        let angle = i as f32 * 2.0 * PI / sides as f32;
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        // Top
        vertices.push(Vec3::new(cos_a * radius, half_h, sin_a * radius));
        let normal = Vec3::new(cos_a, 0.0, sin_a);
        normals.push(normal);
        uvs.push(Vec2::new(i as f32 / sides as f32, 1.0));

        // Bottom
        vertices.push(Vec3::new(cos_a * radius, -half_h, sin_a * radius));
        normals.push(normal);
        uvs.push(Vec2::new(i as f32 / sides as f32, 0.0));
    }

    // Side indices
    for i in 0..sides {
        let a = (i * 2) as u32;
        let b = a + 1;
        let c = a + 2;
        let d = a + 3;
        indices.extend_from_slice(&[a as u32, c as u32, b as u32, b as u32, c as u32, d as u32]);
    }

    // Top cap
    let top_center = vertices.len() as u32;
    vertices.push(Vec3::new(0.0, half_h, 0.0));
    normals.push(Vec3::new(0.0, 1.0, 0.0));
    uvs.push(Vec2::new(0.5, 0.5));

    for i in 0..=sides {
        let angle = i as f32 * 2.0 * PI / sides as f32;
        vertices.push(Vec3::new(
            angle.cos() * radius,
            half_h,
            angle.sin() * radius,
        ));
        normals.push(Vec3::new(0.0, 1.0, 0.0));
        uvs.push(Vec2::new(0.5 + angle.cos() * 0.5, 0.5 + angle.sin() * 0.5));
    }

    for i in 0..sides {
        indices.extend_from_slice(&[
            top_center,
            top_center + 1 + i as u32,
            top_center + 2 + i as u32,
        ]);
    }

    // Bottom cap
    let bottom_center = vertices.len() as u32;
    vertices.push(Vec3::new(0.0, -half_h, 0.0));
    normals.push(Vec3::new(0.0, -1.0, 0.0));
    uvs.push(Vec2::new(0.5, 0.5));

    for i in 0..=sides {
        let angle = i as f32 * 2.0 * PI / sides as f32;
        vertices.push(Vec3::new(
            angle.cos() * radius,
            -half_h,
            angle.sin() * radius,
        ));
        normals.push(Vec3::new(0.0, -1.0, 0.0));
        uvs.push(Vec2::new(0.5 + angle.cos() * 0.5, 0.5 + angle.sin() * 0.5));
    }

    for i in 0..sides {
        indices.extend_from_slice(&[
            bottom_center,
            bottom_center + 2 + i as u32,
            bottom_center + 1 + i as u32,
        ]);
    }

    build_mesh(vertices, normals, uvs, indices)
}

fn build_pyramid(sides: u32, radius: f32, height: f32, _detail: ShapeDetail) -> Mesh {
    let sides = sides.max(3) as usize;
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let half_h = height / 2.0;
    let apex = Vec3::new(0.0, half_h, 0.0);

    // Each side face gets its own 3 vertices (apex + 2 base corners) so normals
    // don't bleed between adjacent faces. This produces hard edges between faces.
    for i in 0..sides {
        let base_i = i as f32 * 2.0 * PI / sides as f32;
        let base_next = (i + 1) as f32 * 2.0 * PI / sides as f32;

        let base_i_pos = Vec3::new(base_i.cos() * radius, -half_h, base_i.sin() * radius);
        let base_next_pos = Vec3::new(base_next.cos() * radius, -half_h, base_next.sin() * radius);

        // Compute face normal from cross product of two edges
        let edge1 = base_i_pos - apex;
        let edge2 = base_next_pos - apex;
        let face_normal = edge1.cross(edge2).normalize();

        let base_idx = vertices.len() as u32;

        // Apex vertex for this face
        vertices.push(apex);
        normals.push(face_normal);
        uvs.push(Vec2::new(0.5, 1.0));

        // Base corner 1
        vertices.push(base_i_pos);
        normals.push(face_normal);
        uvs.push(Vec2::new(i as f32 / sides as f32, 0.0));

        // Base corner 2
        vertices.push(base_next_pos);
        normals.push(face_normal);
        uvs.push(Vec2::new((i + 1) as f32 / sides as f32, 0.0));

        // Triangle: apex, base_i, base_next
        indices.extend_from_slice(&[base_idx, base_idx + 1, base_idx + 2]);
    }

    // Base cap - separate vertices with downward normal
    let base_center = vertices.len() as u32;
    vertices.push(Vec3::new(0.0, -half_h, 0.0));
    normals.push(Vec3::new(0.0, -1.0, 0.0));
    uvs.push(Vec2::new(0.5, 0.5));

    for i in 0..=sides {
        let angle = i as f32 * 2.0 * PI / sides as f32;
        vertices.push(Vec3::new(
            angle.cos() * radius,
            -half_h,
            angle.sin() * radius,
        ));
        normals.push(Vec3::new(0.0, -1.0, 0.0));
        uvs.push(Vec2::new(0.5 + angle.cos() * 0.5, 0.5 + angle.sin() * 0.5));
    }

    for i in 0..sides {
        indices.extend_from_slice(&[
            base_center,
            base_center + 2 + i as u32,
            base_center + 1 + i as u32,
        ]);
    }

    build_mesh(vertices, normals, uvs, indices)
}

// --- 2D Mesh builders ---

fn build_rectangle(size: Vec2, _detail: ShapeDetail) -> Mesh {
    let (hx, hy) = (size.x / 2.0, size.y / 2.0);
    let vertices = vec![
        Vec3::new(-hx, -hy, 0.0),
        Vec3::new(hx, -hy, 0.0),
        Vec3::new(hx, hy, 0.0),
        Vec3::new(-hx, hy, 0.0),
    ];
    let normals = vec![Vec3::Z; 4];
    let uvs = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(0.0, 1.0),
    ];
    let indices = vec![0, 1, 2, 0, 2, 3];
    build_mesh(vertices, normals, uvs, indices)
}

fn build_rectangle_rim(size: Vec2, rim_width: f32, _detail: ShapeDetail) -> Mesh {
    let (hx, hy) = (size.x / 2.0, size.y / 2.0);
    let (ix, iy) = (hx - rim_width, hy - rim_width);
    if ix <= 0.0 || iy <= 0.0 {
        return build_rectangle(size, _detail);
    }
    // Outer ring (CCW) + inner ring (CW) for a hollow rectangle
    let vertices = vec![
        // Outer ring
        Vec3::new(-hx, -hy, 0.0),
        Vec3::new(hx, -hy, 0.0),
        Vec3::new(hx, hy, 0.0),
        Vec3::new(-hx, hy, 0.0),
        // Inner ring
        Vec3::new(-ix, -iy, 0.0),
        Vec3::new(-ix, iy, 0.0),
        Vec3::new(ix, iy, 0.0),
        Vec3::new(ix, -iy, 0.0),
    ];
    let normals = vec![Vec3::Z; 8];
    let uvs = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(0.0, 1.0),
        Vec2::new(0.2, 0.2),
        Vec2::new(0.2, 0.8),
        Vec2::new(0.8, 0.8),
        Vec2::new(0.8, 0.2),
    ];
    // Connect outer to inner rings
    let indices = vec![
        // Top edge
        0, 3, 5, 0, 5, 4, // Right edge
        1, 2, 6, 1, 6, 7, // Bottom edge
        2, 3, 5, 2, 5, 6, // Left edge
        0, 4, 7, 0, 7, 1,
    ];
    build_mesh(vertices, normals, uvs, indices)
}

fn build_disc(radius: f32, detail: ShapeDetail) -> Mesh {
    let segments = detail.segments() as usize;
    let mut vertices = vec![Vec3::ZERO]; // center vertex
    let mut normals = vec![Vec3::Z];
    let mut uvs = vec![Vec2::new(0.5, 0.5)];
    let mut indices: Vec<u32> = Vec::new();

    for i in 0..=segments {
        let angle = i as f32 * 2.0 * PI / segments as f32;
        vertices.push(Vec3::new(angle.cos() * radius, angle.sin() * radius, 0.0));
        normals.push(Vec3::Z);
        uvs.push(Vec2::new(0.5 + angle.cos() * 0.5, 0.5 + angle.sin() * 0.5));
    }

    for i in 0..segments {
        indices.extend_from_slice(&[0, 1 + i as u32, 2 + i as u32]);
    }

    build_mesh(vertices, normals, uvs, indices)
}

fn build_disc_rim(radius: f32, rim_width: f32, detail: ShapeDetail) -> Mesh {
    let segments = detail.segments() as usize;
    let inner_radius = (radius - rim_width).max(0.0);
    if inner_radius <= 0.0 {
        return build_disc(radius, detail);
    }

    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Outer ring
    for i in 0..=segments {
        let angle = i as f32 * 2.0 * PI / segments as f32;
        vertices.push(Vec3::new(angle.cos() * radius, angle.sin() * radius, 0.0));
        normals.push(Vec3::Z);
        uvs.push(Vec2::new(0.5 + angle.cos() * 0.5, 0.5 + angle.sin() * 0.5));
    }
    // Inner ring
    for i in 0..=segments {
        let angle = i as f32 * 2.0 * PI / segments as f32;
        vertices.push(Vec3::new(
            angle.cos() * inner_radius,
            angle.sin() * inner_radius,
            0.0,
        ));
        normals.push(Vec3::Z);
        uvs.push(Vec2::new(0.5 + angle.cos() * 0.2, 0.5 + angle.sin() * 0.2));
    }

    let outer = segments as u32 + 1;
    for i in 0..segments {
        let a = i as u32;
        let b = i as u32 + 1;
        let c = outer + i as u32;
        let d = outer + i as u32 + 1;
        indices.extend_from_slice(&[a, b, c, b, d, c]);
    }

    build_mesh(vertices, normals, uvs, indices)
}

fn build_polygon(sides: u32, radius: f32, _detail: ShapeDetail) -> Mesh {
    let sides = sides.max(3);
    let mut vertices = vec![Vec3::ZERO]; // center
    let mut normals = vec![Vec3::Z];
    let mut uvs = vec![Vec2::new(0.5, 0.5)];
    let mut indices: Vec<u32> = Vec::new();

    for i in 0..=sides {
        let angle = i as f32 * 2.0 * PI / sides as f32;
        vertices.push(Vec3::new(angle.cos() * radius, angle.sin() * radius, 0.0));
        normals.push(Vec3::Z);
        uvs.push(Vec2::new(0.5 + angle.cos() * 0.5, 0.5 + angle.sin() * 0.5));
    }

    for i in 0..sides {
        indices.extend_from_slice(&[0, 1 + i, 2 + i]);
    }

    build_mesh(vertices, normals, uvs, indices)
}

fn build_polygon_rim(sides: u32, radius: f32, rim_width: f32, _detail: ShapeDetail) -> Mesh {
    let sides = sides.max(3);
    let inner_radius = (radius - rim_width).max(0.0);
    if inner_radius <= 0.0 {
        return build_polygon(sides, radius, _detail);
    }

    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Outer ring
    for i in 0..=sides {
        let angle = i as f32 * 2.0 * PI / sides as f32;
        vertices.push(Vec3::new(angle.cos() * radius, angle.sin() * radius, 0.0));
        normals.push(Vec3::Z);
        uvs.push(Vec2::new(0.5 + angle.cos() * 0.5, 0.5 + angle.sin() * 0.5));
    }
    // Inner ring
    for i in 0..=sides {
        let angle = i as f32 * 2.0 * PI / sides as f32;
        vertices.push(Vec3::new(
            angle.cos() * inner_radius,
            angle.sin() * inner_radius,
            0.0,
        ));
        normals.push(Vec3::Z);
        uvs.push(Vec2::new(0.5 + angle.cos() * 0.2, 0.5 + angle.sin() * 0.2));
    }

    let outer = (sides + 1) as u32;
    for i in 0..sides {
        let a = i as u32;
        let b = i as u32 + 1;
        let c = outer + i as u32;
        let d = outer + i as u32 + 1;
        indices.extend_from_slice(&[a, b, c, b, d, c]);
    }

    build_mesh(vertices, normals, uvs, indices)
}

fn build_ellipse(radius: Vec2, detail: ShapeDetail) -> Mesh {
    let segments = detail.segments() as usize;
    let mut vertices = vec![Vec3::ZERO]; // center
    let mut normals = vec![Vec3::Z];
    let mut uvs = vec![Vec2::new(0.5, 0.5)];
    let mut indices: Vec<u32> = Vec::new();

    for i in 0..=segments {
        let angle = i as f32 * 2.0 * PI / segments as f32;
        vertices.push(Vec3::new(
            angle.cos() * radius.x,
            angle.sin() * radius.y,
            0.0,
        ));
        normals.push(Vec3::Z);
        uvs.push(Vec2::new(0.5 + angle.cos() * 0.5, 0.5 + angle.sin() * 0.5));
    }

    for i in 0..segments {
        indices.extend_from_slice(&[0, 1 + i as u32, 2 + i as u32]);
    }

    build_mesh(vertices, normals, uvs, indices)
}

fn build_ellipse_rim(radius: Vec2, rim_width: f32, detail: ShapeDetail) -> Mesh {
    let segments = detail.segments() as usize;
    let inner_x = (radius.x - rim_width).max(0.0);
    let inner_y = (radius.y - rim_width).max(0.0);
    if inner_x <= 0.0 || inner_y <= 0.0 {
        return build_ellipse(radius, detail);
    }

    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Outer ring
    for i in 0..=segments {
        let angle = i as f32 * 2.0 * PI / segments as f32;
        vertices.push(Vec3::new(
            angle.cos() * radius.x,
            angle.sin() * radius.y,
            0.0,
        ));
        normals.push(Vec3::Z);
        uvs.push(Vec2::new(0.5 + angle.cos() * 0.5, 0.5 + angle.sin() * 0.5));
    }
    // Inner ring
    for i in 0..=segments {
        let angle = i as f32 * 2.0 * PI / segments as f32;
        vertices.push(Vec3::new(angle.cos() * inner_x, angle.sin() * inner_y, 0.0));
        normals.push(Vec3::Z);
        uvs.push(Vec2::new(0.5 + angle.cos() * 0.2, 0.5 + angle.sin() * 0.2));
    }

    let outer = segments as u32 + 1;
    for i in 0..segments {
        let a = i as u32;
        let b = i as u32 + 1;
        let c = outer + i as u32;
        let d = outer + i as u32 + 1;
        indices.extend_from_slice(&[a, b, c, b, d, c]);
    }

    build_mesh(vertices, normals, uvs, indices)
}

fn build_mesh(vertices: Vec<Vec3>, normals: Vec<Vec3>, uvs: Vec<Vec2>, indices: Vec<u32>) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(bevy::render::mesh::Indices::U32(indices));
    // Note: we do NOT call compute_normals() here because each primitive builder
    // already computes correct analytical normals. Recomputing would smooth across
    // hard edges (e.g., cube corners, pyramid faces) and cause shading artifacts.
    mesh
}

/// Build a 3D polyline mesh as tube geometry.
/// Delegates to `PolylineConfig::build_gpu_mesh()` from `polyline_gpu.rs`.
fn build_polyline3d(points: Vec<Vec3>, thickness: f32, closed: bool, segments: u32) -> Mesh {
    use crate::polyline_gpu::PolylineConfig;
    let config = PolylineConfig {
        points,
        thickness,
        closed,
        segments,
    };
    config.build_gpu_mesh()
}

/// Build a Bézier curve mesh as a tube along the curve path.
/// Tessellates the curve into points, then builds cylinder segments between them.
fn build_bezier_curve(control_points: Vec<Vec3>, thickness: f32, segments: u32) -> Mesh {
    if control_points.len() < 2 {
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<Vec3>::new());
        return mesh;
    }

    // Tessellate the Bézier curve into a series of points
    let tessellated = tessellate_bezier(&control_points, 48);

    if tessellated.len() < 2 {
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<Vec3>::new());
        return mesh;
    }

    // Share the polyline tube builder: continuous rings, smooth frames, round caps.
    build_polyline3d(tessellated, thickness, false, segments)
}

/// Build a Catmull-Rom spline mesh as a tube along the curve path.
/// Uses the spline module to bake the curve, then sweeps a circular profile.
fn build_catmull_rom_3d(points: Vec<Vec3>, thickness: f32, closed: bool, segments: u32) -> Mesh {
    use crate::spline::CatmullRomSpline;
    
    // Need at least 3 points for closed, 2 for open
    let spline = if closed {
        CatmullRomSpline::closed(points).ok()
    } else {
        CatmullRomSpline::open(points).ok()
    };
    
    let Some(spline) = spline else {
        bevy::log::warn!("catmull_rom_3d requires at least 2 points (3 for closed), got fewer");
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<Vec3>::new());
        return mesh;
    };
    
    // Bake the spline with a step size that gives smooth results
    let baked = spline.bake(0.1);
    let frames = baked.sweep_frames(None, None);
    
    if frames.len() < 2 {
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<Vec3>::new());
        return mesh;
    }
    
    // Build a circular profile for the tube
    let profile_segs = segments.max(4);
    let mut profile = Vec::with_capacity(profile_segs as usize);
    for i in 0..profile_segs {
        let angle = i as f32 * std::f32::consts::TAU / profile_segs as f32;
        profile.push(Vec2::new(angle.cos() * thickness, angle.sin() * thickness));
    }
    
    let options = SweepOptions::default()
        .with_closed(closed)
        .with_caps(!closed);
    
    geometry::sweep_mesh(&profile, &frames.iter().map(|f| f.position).collect::<Vec<_>>(), Some(&frames), &options)
}

/// Tessellate a Bézier curve into a series of points.
/// Supports any number of control points (degree = points - 1) using De Casteljau's algorithm,
/// so every control point influences the curve.
fn tessellate_bezier(control_points: &[Vec3], num_segments: u32) -> Vec<Vec3> {
    if control_points.len() < 2 {
        return Vec::new();
    }
    let n = num_segments.max(1);
    let mut scratch = vec![Vec3::ZERO; control_points.len()];
    (0..=n)
        .map(|i| {
            let t = i as f32 / n as f32;
            scratch.copy_from_slice(control_points);
            for level in (1..scratch.len()).rev() {
                for k in 0..level {
                    scratch[k] = scratch[k].lerp(scratch[k + 1], t);
                }
            }
            scratch[0]
        })
        .collect()
}

/// Build an N-pointed star as a 2D flat quad.
fn build_star(points: u32, outer_radius: f32, inner_radius: f32, _detail: ShapeDetail) -> Mesh {
    let n = points.max(2) as usize;
    let mut vertices = vec![Vec3::ZERO]; // center vertex
    let mut normals = vec![Vec3::new(0.0, 0.0, 1.0)];
    let mut uvs = vec![Vec2::new(0.5, 0.5)]; // center UV

    // Build outer ring of vertices: for each point, an outer vertex and an inner vertex
    let mut outer_verts = Vec::new();
    for i in 0..n {
        let angle_outer = i as f32 * 2.0 * PI / n as f32 - PI / 2.0; // start from top
        let angle_inner = (i as f32 + 0.5) * 2.0 * PI / n as f32 - PI / 2.0;

        let ox = angle_outer.cos() * outer_radius;
        let oy = angle_outer.sin() * outer_radius;
        outer_verts.push(Vec2::new(ox, oy));

        let ix = angle_inner.cos() * inner_radius;
        let iy = angle_inner.sin() * inner_radius;
        outer_verts.push(Vec2::new(ix, iy));
    }

    let center_idx = 0u32;
    let mut indices = Vec::new();

    // Add outer vertices and triangulate from center
    for (i, v) in outer_verts.iter().enumerate() {
        let idx = (i + 1) as u32;
        vertices.push(Vec3::new(v.x, v.y, 0.0));
        normals.push(Vec3::new(0.0, 0.0, 1.0));
        uvs.push(Vec2::new(
            v.x / outer_radius * 0.5 + 0.5,
            v.y / outer_radius * 0.5 + 0.5,
        ));
        // Each triangle: center, outer, inner (or center, inner, next outer)
        if i + 1 < outer_verts.len() {
            indices.extend_from_slice(&[center_idx, idx, idx + 1]);
        }
    }
    // Close the last triangle
    if !outer_verts.is_empty() {
        indices.extend_from_slice(&[center_idx, outer_verts.len() as u32, 1]);
    }

    build_mesh(vertices, normals, uvs, indices)
}

/// Build a ring/annulus as a 2D flat quad with a hole.
fn build_ring(outer_radius: f32, inner_radius: f32, detail: ShapeDetail) -> Mesh {
    if inner_radius >= outer_radius || outer_radius <= 0.0 {
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<Vec3>::new());
        return mesh;
    }
    let segments = detail.segments() as usize;
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();

    // Generate outer and inner ring vertices
    for i in 0..=segments {
        let angle = i as f32 * 2.0 * PI / segments as f32;
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        // Outer vertex
        vertices.push(Vec3::new(cos_a * outer_radius, sin_a * outer_radius, 0.0));
        normals.push(Vec3::new(0.0, 0.0, 1.0));
        uvs.push(Vec2::new(cos_a * 0.5 + 0.5, sin_a * 0.5 + 0.5));

        // Inner vertex
        vertices.push(Vec3::new(cos_a * inner_radius, sin_a * inner_radius, 0.0));
        normals.push(Vec3::new(0.0, 0.0, 1.0));
        uvs.push(Vec2::new(
            cos_a * (inner_radius / outer_radius) * 0.5 + 0.5,
            sin_a * (inner_radius / outer_radius) * 0.5 + 0.5,
        ));
    }

    // Generate quad indices between outer and inner rings
    for i in 0..segments {
        let o1 = (i * 2) as u32;
        let o2 = ((i + 1) * 2) as u32;
        let i1 = (i * 2 + 1) as u32;
        let i2 = ((i + 1) * 2 + 1) as u32;
        // Two triangles per quad
        indices.extend_from_slice(&[o1, o2, i1]);
        indices.extend_from_slice(&[i1, o2, i2]);
    }

    build_mesh(vertices, normals, uvs, indices)
}

/// Build a wedge/pie-slice as a 2D flat quad.
fn build_wedge(radius: f32, angle: f32, detail: ShapeDetail) -> Mesh {
    if angle <= 0.0 || radius <= 0.0 {
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<Vec3>::new());
        return mesh;
    }
    let segments = if angle >= 2.0 * PI - 0.001 {
        // Full circle case: use disc-like triangulation
        detail.segments() as usize
    } else {
        (detail.segments() as f32 * angle / (2.0 * PI)).ceil() as usize
    };
    let segments = segments.max(2);
    let mut vertices = vec![Vec3::ZERO]; // center
    let mut normals = vec![Vec3::new(0.0, 0.0, 1.0)];
    let mut uvs = vec![Vec2::new(0.5, 0.5)];
    let mut indices = Vec::new();

    for i in 0..=segments {
        let t = i as f32 / segments as f32;
        let a = t * angle;
        let x = a.cos() * radius;
        let y = a.sin() * radius;
        let idx = (i + 1) as u32;
        vertices.push(Vec3::new(x, y, 0.0));
        normals.push(Vec3::new(0.0, 0.0, 1.0));
        uvs.push(Vec2::new(
            t * 0.5 + 0.5 * (a.cos() * 0.5 + 0.5),
            t * 0.5 + 0.5 * (a.sin() * 0.5 + 0.5),
        ));
        if i < segments {
            indices.extend_from_slice(&[0, idx, idx + 1]);
        }
    }

    build_mesh(vertices, normals, uvs, indices)
}

/// Build a tapered cylinder (cone-frustum with flat top and bottom).
fn build_tapered_cylinder(
    top_radius: f32,
    bottom_radius: f32,
    height: f32,
    detail: ShapeDetail,
) -> Mesh {
    let segments = detail.segments() as usize;
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();

    let half_h = height / 2.0;

    // Generate side vertices (bottom and top rings)
    for i in 0..=segments {
        let angle = i as f32 * 2.0 * PI / segments as f32;
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let u = i as f32 / segments as f32;

        // Bottom ring
        vertices.push(Vec3::new(
            cos_a * bottom_radius,
            -half_h,
            sin_a * bottom_radius,
        ));
        let bn = Vec3::new(cos_a, 0.0, sin_a).normalize();
        normals.push(bn);
        uvs.push(Vec2::new(u, 0.0));

        // Top ring
        vertices.push(Vec3::new(cos_a * top_radius, half_h, sin_a * top_radius));
        let tn = Vec3::new(cos_a, 0.0, sin_a).normalize();
        normals.push(tn);
        uvs.push(Vec2::new(u, 1.0));
    }

    // Side faces: quads between bottom and top rings
    for i in 0..segments {
        let b1 = (i * 2) as u32;
        let b2 = ((i + 1) * 2) as u32;
        let t1 = (i * 2 + 1) as u32;
        let t2 = ((i + 1) * 2 + 1) as u32;
        // Compute proper normal for this face
        let angle1 = i as f32 * 2.0 * PI / segments as f32;
        let angle2 = (i + 1) as f32 * 2.0 * PI / segments as f32;
        let mid_angle = (angle1 + angle2) / 2.0;
        // Normal points outward, slightly tilted
        let dx = mid_angle.cos() * top_radius - mid_angle.cos() * bottom_radius;
        let dy = height;
        let dz = mid_angle.sin() * top_radius - mid_angle.sin() * bottom_radius;
        let side_normal = Vec3::new(dx, dy, dz).normalize();
        // Override normals for this face's vertices
        normals[b1 as usize] = side_normal;
        normals[b2 as usize] = side_normal;
        normals[t1 as usize] = side_normal;
        normals[t2 as usize] = side_normal;

        indices.extend_from_slice(&[b1, t1, b2]);
        indices.extend_from_slice(&[b2, t1, t2]);
    }

    // Bottom cap (pointing down)
    let bottom_center_idx = vertices.len() as u32;
    vertices.push(Vec3::new(0.0, -half_h, 0.0));
    normals.push(Vec3::new(0.0, -1.0, 0.0));
    uvs.push(Vec2::new(0.5, 0.5));
    for i in 0..segments {
        let i1 = (i * 2) as u32;
        let i2 = ((i + 1) * 2) as u32;
        indices.extend_from_slice(&[bottom_center_idx, i1, i2]);
    }

    // Top cap (pointing up)
    let top_center_idx = vertices.len() as u32;
    vertices.push(Vec3::new(0.0, half_h, 0.0));
    normals.push(Vec3::new(0.0, 1.0, 0.0));
    uvs.push(Vec2::new(0.5, 0.5));
    for i in 0..segments {
        let i1 = (i * 2 + 1) as u32;
        let i2 = ((i + 1) * 2 + 1) as u32;
        indices.extend_from_slice(&[top_center_idx, i2, i1]);
    }

    build_mesh(vertices, normals, uvs, indices)
}

fn mesh_bounds(mesh: &Mesh) -> ShapeBounds {
    use bevy::render::mesh::VertexAttributeValues;
    match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        Some(VertexAttributeValues::Float32x3(p)) => {
            ShapeBounds::from_points(p.iter().map(|p| Vec3::from_array(*p)))
        }
        _ => ShapeBounds::default(),
    }
}

/// The box of the kinds whose mesh is centred on the origin and reaches its
/// full size at every detail level (every [`ShapeDetail`] has a segment count
/// divisible by four, so the rings hit all four axis extremes). A test compares
/// each of them with the measured mesh.
fn closed_form_bounds(kind: &ShapeKind) -> Option<ShapeBounds> {
    let centered = |half: Vec3| ShapeBounds {
        min: -half,
        max: half,
    };
    Some(match kind {
        ShapeKind::Sphere { radius } => centered(Vec3::splat(*radius)),
        ShapeKind::Ellipsoid { size } => centered(*size / 2.0),
        ShapeKind::Cuboid { size, .. } => centered(*size / 2.0),
        // Rounding blunts the apex and rim of a cone and the rim of a cylinder,
        // so only the sharp versions have this box.
        ShapeKind::Cylinder {
            radius,
            height,
            roundness,
        }
        | ShapeKind::Cone {
            radius,
            height,
            roundness,
        } if *roundness == 0.0 => centered(Vec3::new(*radius, *height / 2.0, *radius)),
        ShapeKind::Capsule { radius, height } => centered(Vec3::new(*radius, *height / 2.0, *radius)),
        ShapeKind::Rectangle { size } => centered(size.extend(0.0) / 2.0),
        ShapeKind::Disc { radius } => centered(Vec3::new(*radius, *radius, 0.0)),
        ShapeKind::Ellipse { radius } => centered(radius.extend(0.0)),
        _ => return None,
    })
}

/// Whether every vertex lies within about 5% of the ellipsoid inscribed in
/// `bounds`, which is how [`Shape::from_mesh`] guesses [`Shape::is_round`].
fn mesh_is_round(mesh: &Mesh, bounds: &ShapeBounds) -> bool {
    use bevy::render::mesh::VertexAttributeValues;
    let Some(VertexAttributeValues::Float32x3(p)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION)
    else {
        return false;
    };
    let half = bounds.half_extents();
    if p.is_empty() || half.max_element() <= 0.0 {
        return false;
    }
    let center = bounds.center();
    let norm = |v: f32, h: f32| if h > 1e-6 { v / h } else { 0.0 };
    p.iter().all(|p| {
        let d = Vec3::from_array(*p) - center;
        let q = Vec3::new(norm(d.x, half.x), norm(d.y, half.y), norm(d.z, half.z));
        (q.length() - 1.0).abs() <= 0.05
    })
}

fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

fn lerp_vec2(a: Vec2, b: Vec2, t: f32) -> Vec2 {
    Vec2::new(lerp_f32(a.x, b.x, t), lerp_f32(a.y, b.y, t))
}

fn lerp_vec3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    Vec3::new(
        lerp_f32(a.x, b.x, t),
        lerp_f32(a.y, b.y, t),
        lerp_f32(a.z, b.z, t),
    )
}

/// Picks one of two values based on a step function at `t == 0.5`.
///
/// Returns `a` when `t < 0.5` and `b` otherwise. Used for integer counts
/// (sides, segments, points) and boolean flags (capped, closed) that cannot
/// be meaningfully interpolated — a prism cannot have 6.3 sides.
///
/// # Animation implications
///
/// Because this is a hard step, animating `t` from 0.0 to 1.0 will show the
/// first value unchanged until `t` crosses 0.5, then instantly jump to the
/// second value. For example, lerping a 4-sided prism to an 8-sided prism
/// will display the 4-sided version for the first half of the animation and
/// the 8-sided version for the second half, with no gradual transition.
///
/// To avoid the jump in visible animations, consider these strategies:
///
/// * **Quantise `t`** before calling [`Shape::lerp`]: round to a small set of
///   steps so the switch happens at a predictable frame rather than mid-animation.
/// * **Pre-compute a keyframe table**: build the intermediate shapes at fixed `t`
///   values (e.g. 0.0, 0.25, 0.5, 1.0) and switch between them, rather than
///   calling `lerp` every frame. See [`Shape::lerp`] for more on keyframing.
/// * **Animate a continuous proxy** instead: for example, animate `Transform`
///   scale on a fixed shape rather than morphing the shape itself.
fn pick<T>(t: f32, a: T, b: T) -> T {
    if t < 0.5 { a } else { b }
}

/// Interpolates two equally long point lists; `None` when the lengths differ.
fn lerp_points<P: Copy>(a: &[P], b: &[P], t: f32, f: fn(P, P, f32) -> P) -> Option<Vec<P>> {
    (a.len() == b.len()).then(|| a.iter().zip(b).map(|(a, b)| f(*a, *b, t)).collect())
}

type Rgba = [f32; 4];

fn lerp_rgba(a: Rgba, b: Rgba, t: f32) -> Rgba {
    std::array::from_fn(|i| lerp_f32(a[i], b[i], t))
}

fn signed_area(p: &[Vec2]) -> f32 {
    let n = p.len();
    (0..n).map(|i| p[i].perp_dot(p[(i + 1) % n])).sum::<f32>() * 0.5
}

/// Describes `n` points on a closed ring of `m <= n` vertices as
/// `(vertex, fraction towards the next vertex)`. Every original vertex is kept;
/// the extra points go onto the edges with the longest pieces.
fn ring_samples(p: &[Vec2], n: usize) -> Vec<(usize, f32)> {
    let m = p.len();
    let lens: Vec<f32> = (0..m).map(|i| p[i].distance(p[(i + 1) % m])).collect();
    let mut pieces = vec![1usize; m];
    for _ in m..n {
        let mut best = 0;
        let mut best_len = -1.0;
        for i in 0..m {
            let len = lens[i] / pieces[i] as f32;
            if len > best_len {
                best_len = len;
                best = i;
            }
        }
        pieces[best] += 1;
    }
    let mut out = Vec::with_capacity(n);
    for (i, &k) in pieces.iter().enumerate() {
        out.extend((0..k).map(|j| (i, j as f32 / k as f32)));
    }
    out
}

fn sample_ring<T: Copy>(v: &[T], samples: &[(usize, f32)], f: fn(T, T, f32) -> T) -> Vec<T> {
    let m = v.len();
    samples
        .iter()
        .map(|&(i, frac)| {
            if frac == 0.0 {
                v[i]
            } else {
                f(v[i], v[(i + 1) % m], frac)
            }
        })
        .collect()
}

/// Pairing of two closed rings with different vertex counts: both are grown to
/// the larger count, `b` is flipped to match the winding of `a` and rotated to
/// the start that moves the points the least.
struct RingMorph {
    a: Vec<(usize, f32)>,
    b: Vec<(usize, f32)>,
    reverse_b: bool,
}

impl RingMorph {
    fn new(a: &[Vec2], b: &[Vec2]) -> Option<Self> {
        if a.len() < 3 || b.len() < 3 {
            return None;
        }
        let n = a.len().max(b.len());
        let reverse_b = signed_area(a) * signed_area(b) < 0.0;
        let b_pts = Self::orient(b, reverse_b);
        let sa = ring_samples(a, n);
        let mut sb = ring_samples(&b_pts, n);
        let pa = sample_ring(a, &sa, lerp_vec2);
        let pb = sample_ring(&b_pts, &sb, lerp_vec2);
        let cost = |shift: usize| -> f32 {
            (0..n)
                .map(|k| pa[k].distance_squared(pb[(k + shift) % n]))
                .sum()
        };
        let shift = (0..n).min_by(|&x, &y| cost(x).total_cmp(&cost(y)))?;
        sb.rotate_left(shift);
        Some(Self {
            a: sa,
            b: sb,
            reverse_b,
        })
    }

    fn orient<T: Copy>(v: &[T], reverse: bool) -> Vec<T> {
        if reverse {
            v.iter().rev().copied().collect()
        } else {
            v.to_vec()
        }
    }

    fn blend<T: Copy>(&self, a: &[T], b: &[T], t: f32, f: fn(T, T, f32) -> T) -> Vec<T> {
        let pa = sample_ring(a, &self.a, f);
        let pb = sample_ring(&Self::orient(b, self.reverse_b), &self.b, f);
        pa.iter().zip(&pb).map(|(a, b)| f(*a, *b, t)).collect()
    }
}

/// Morphs two closed polygons whose vertex counts differ, with optional
/// per-vertex colours (a missing colour list counts as white). `None` when a
/// ring has fewer than three vertices or a colour list does not match its
/// vertices.
fn morph_rings(
    a: &[Vec2],
    ca: Option<&[Rgba]>,
    b: &[Vec2],
    cb: Option<&[Rgba]>,
    t: f32,
) -> Option<(Vec<Vec2>, Option<Vec<Rgba>>)> {
    let morph = RingMorph::new(a, b)?;
    let vertices = morph.blend(a, b, t, lerp_vec2);
    let colors = if ca.is_none() && cb.is_none() {
        None
    } else {
        let ca = ca.map_or_else(|| vec![[1.0; 4]; a.len()], <[Rgba]>::to_vec);
        let cb = cb.map_or_else(|| vec![[1.0; 4]; b.len()], <[Rgba]>::to_vec);
        if ca.len() != a.len() || cb.len() != b.len() {
            return None;
        }
        Some(morph.blend(&ca, &cb, t, lerp_rgba))
    };
    Some((vertices, colors))
}

/// Blends two closed rings: index by index when the counts match, otherwise by
/// growing both to the larger count (see [`RingMorph`]).
fn lerp_rings(a: &[Vec2], b: &[Vec2], t: f32) -> Option<Vec<Vec2>> {
    if a.len() == b.len() {
        lerp_points(a, b, t, lerp_vec2)
    } else {
        morph_rings(a, None, b, None, t).map(|(v, _)| v)
    }
}

impl ShapeKind {
    /// Interpolates two shapes of the same kind. Returns `None` when the kinds
    /// differ or when point lists of different lengths cannot be paired up.
    fn lerp(&self, other: &Self, t: f32) -> Option<Self> {
        use ShapeKind as K;
        let l = |a: &f32, b: &f32| lerp_f32(*a, *b, t);
        Some(match (self, other) {
            (K::Sphere { radius: a }, K::Sphere { radius: b }) => K::Sphere { radius: l(a, b) },
            (K::Ellipsoid { size: a }, K::Ellipsoid { size: b }) => K::Ellipsoid {
                size: lerp_vec3(*a, *b, t),
            },
            (
                K::Cuboid {
                    size: s1,
                    roundness: r1,
                },
                K::Cuboid {
                    size: s2,
                    roundness: r2,
                },
            ) => K::Cuboid {
                size: lerp_vec3(*s1, *s2, t),
                roundness: l(r1, r2),
            },
            (
                K::Cylinder {
                    radius: r1,
                    height: h1,
                    roundness: n1,
                },
                K::Cylinder {
                    radius: r2,
                    height: h2,
                    roundness: n2,
                },
            ) => K::Cylinder {
                radius: l(r1, r2),
                height: l(h1, h2),
                roundness: l(n1, n2),
            },
            (
                K::Cone {
                    radius: r1,
                    height: h1,
                    roundness: o1,
                },
                K::Cone {
                    radius: r2,
                    height: h2,
                    roundness: o2,
                },
            ) => K::Cone {
                radius: l(r1, r2),
                height: l(h1, h2),
                roundness: l(o1, o2),
            },
            (
                K::Capsule {
                    radius: r1,
                    height: h1,
                },
                K::Capsule {
                    radius: r2,
                    height: h2,
                },
            ) => K::Capsule {
                radius: l(r1, r2),
                height: l(h1, h2),
            },
            (
                K::Hemisphere {
                    radius: r1,
                    capped: c1,
                },
                K::Hemisphere {
                    radius: r2,
                    capped: c2,
                },
            ) => K::Hemisphere {
                radius: l(r1, r2),
                capped: pick(t, *c1, *c2),
            },
            (
                K::Torus {
                    radius: r1,
                    thickness: k1,
                    start: s1,
                    span: p1,
                },
                K::Torus {
                    radius: r2,
                    thickness: k2,
                    start: s2,
                    span: p2,
                },
            ) => K::Torus {
                radius: l(r1, r2),
                thickness: l(k1, k2),
                start: l(s1, s2),
                span: l(p1, p2),
            },
            (
                K::RegularPrism {
                    sides: n1,
                    radius: r1,
                    height: h1,
                    roundness: o1,
                },
                K::RegularPrism {
                    sides: n2,
                    radius: r2,
                    height: h2,
                    roundness: o2,
                },
            ) => K::RegularPrism {
                sides: pick(t, *n1, *n2),
                radius: l(r1, r2),
                height: l(h1, h2),
                roundness: l(o1, o2),
            },
            (
                K::Pyramid {
                    sides: n1,
                    radius: r1,
                    height: h1,
                    roundness: o1,
                },
                K::Pyramid {
                    sides: n2,
                    radius: r2,
                    height: h2,
                    roundness: o2,
                },
            ) => K::Pyramid {
                sides: pick(t, *n1, *n2),
                radius: l(r1, r2),
                height: l(h1, h2),
                roundness: l(o1, o2),
            },
            (K::Rectangle { size: a }, K::Rectangle { size: b }) => K::Rectangle {
                size: lerp_vec2(*a, *b, t),
            },
            (
                K::RectangleRim {
                    size: s1,
                    rim_width: w1,
                },
                K::RectangleRim {
                    size: s2,
                    rim_width: w2,
                },
            ) => K::RectangleRim {
                size: lerp_vec2(*s1, *s2, t),
                rim_width: l(w1, w2),
            },
            (K::Disc { radius: a }, K::Disc { radius: b }) => K::Disc { radius: l(a, b) },
            (
                K::DiscRim {
                    radius: r1,
                    rim_width: w1,
                },
                K::DiscRim {
                    radius: r2,
                    rim_width: w2,
                },
            ) => K::DiscRim {
                radius: l(r1, r2),
                rim_width: l(w1, w2),
            },
            (
                K::Polygon {
                    sides: n1,
                    radius: r1,
                },
                K::Polygon {
                    sides: n2,
                    radius: r2,
                },
            ) => K::Polygon {
                sides: pick(t, *n1, *n2),
                radius: l(r1, r2),
            },
            (
                K::PolygonRim {
                    sides: n1,
                    radius: r1,
                    rim_width: w1,
                },
                K::PolygonRim {
                    sides: n2,
                    radius: r2,
                    rim_width: w2,
                },
            ) => K::PolygonRim {
                sides: pick(t, *n1, *n2),
                radius: l(r1, r2),
                rim_width: l(w1, w2),
            },
            (K::Ellipse { radius: a }, K::Ellipse { radius: b }) => K::Ellipse {
                radius: lerp_vec2(*a, *b, t),
            },
            (
                K::EllipseRim {
                    radius: r1,
                    rim_width: w1,
                },
                K::EllipseRim {
                    radius: r2,
                    rim_width: w2,
                },
            ) => K::EllipseRim {
                radius: lerp_vec2(*r1, *r2, t),
                rim_width: l(w1, w2),
            },
            (
                K::ArbitraryPolygon {
                    vertices: v1,
                    colors: c1,
                },
                K::ArbitraryPolygon {
                    vertices: v2,
                    colors: c2,
                },
            ) if v1.len() != v2.len() => {
                let (vertices, colors) =
                    morph_rings(v1, c1.as_deref(), v2, c2.as_deref(), t)?;
                K::ArbitraryPolygon { vertices, colors }
            }
            (
                K::ArbitraryPolygon {
                    vertices: v1,
                    colors: c1,
                },
                K::ArbitraryPolygon {
                    vertices: v2,
                    colors: c2,
                },
            ) => K::ArbitraryPolygon {
                vertices: lerp_points(v1, v2, t, lerp_vec2)?,
                colors: match (c1, c2) {
                    (Some(a), Some(b)) if a.len() == b.len() => Some(
                        a.iter()
                            .zip(b)
                            .map(|(a, b)| std::array::from_fn(|i| lerp_f32(a[i], b[i], t)))
                            .collect(),
                    ),
                    _ => pick(t, c1.clone(), c2.clone()),
                },
            },
            (
                K::Polyline3D {
                    points: p1,
                    thickness: k1,
                    closed: c1,
                    segments: s1,
                },
                K::Polyline3D {
                    points: p2,
                    thickness: k2,
                    closed: c2,
                    segments: s2,
                },
            ) => K::Polyline3D {
                points: lerp_points(p1, p2, t, lerp_vec3)?,
                thickness: l(k1, k2),
                closed: pick(t, *c1, *c2),
                segments: pick(t, *s1, *s2),
            },
            (
                K::BezierCurve {
                    control_points: p1,
                    thickness: k1,
                    segments: s1,
                },
                K::BezierCurve {
                    control_points: p2,
                    thickness: k2,
                    segments: s2,
                },
            ) => K::BezierCurve {
                control_points: lerp_points(p1, p2, t, lerp_vec3)?,
                thickness: l(k1, k2),
                segments: pick(t, *s1, *s2),
            },
            (
                K::Star {
                    points: n1,
                    outer_radius: o1,
                    inner_radius: i1,
                },
                K::Star {
                    points: n2,
                    outer_radius: o2,
                    inner_radius: i2,
                },
            ) => K::Star {
                points: pick(t, *n1, *n2),
                outer_radius: l(o1, o2),
                inner_radius: l(i1, i2),
            },
            (
                K::Ring {
                    outer_radius: o1,
                    inner_radius: i1,
                },
                K::Ring {
                    outer_radius: o2,
                    inner_radius: i2,
                },
            ) => K::Ring {
                outer_radius: l(o1, o2),
                inner_radius: l(i1, i2),
            },
            (
                K::Wedge {
                    radius: r1,
                    angle: a1,
                },
                K::Wedge {
                    radius: r2,
                    angle: a2,
                },
            ) => K::Wedge {
                radius: l(r1, r2),
                angle: l(a1, a2),
            },
            (
                K::TaperedCylinder {
                    top_radius: t1,
                    bottom_radius: b1,
                    height: h1,
                },
                K::TaperedCylinder {
                    top_radius: t2,
                    bottom_radius: b2,
                    height: h2,
                },
            ) => K::TaperedCylinder {
                top_radius: l(t1, t2),
                bottom_radius: l(b1, b2),
                height: l(h1, h2),
            },
            (
                K::ExtrudedPolygon {
                    outer: o1,
                    holes: h1,
                    colors: c1,
                    depth: d1,
                    roundness: r1,
                },
                K::ExtrudedPolygon {
                    outer: o2,
                    holes: h2,
                    colors: c2,
                    depth: d2,
                    roundness: r2,
                },
            ) => {
                if h1.len() != h2.len() {
                    return None;
                }
                let holes = h1
                    .iter()
                    .zip(h2)
                    .map(|(a, b)| lerp_rings(a, b, t))
                    .collect::<Option<Vec<_>>>()?;
                let (outer, colors) = if o1.len() != o2.len() {
                    morph_rings(o1, c1.as_deref(), o2, c2.as_deref(), t)?
                } else {
                    (lerp_points(o1, o2, t, lerp_vec2)?, {
                        // When vertex counts match, blend colors directly.
                        // If only one side has colors, treat the missing side as white,
                        // consistent with `morph_rings` behavior for different counts.
                        let white = [1.0, 1.0, 1.0, 1.0];
                        match (c1, c2) {
                            (None, None) => None,
                            (Some(a), Some(b)) => {
                                Some(a.iter().zip(b).map(|(x, y)| lerp_rgba(*x, *y, t)).collect())
                            }
                            (Some(a), None) => {
                                // Blend `a`'s colors towards white
                                Some(a.iter().map(|x| lerp_rgba(*x, white, t)).collect())
                            }
                            (None, Some(b)) => {
                                // Blend white towards `b`'s colors
                                Some(b.iter().map(|y| lerp_rgba(white, *y, t)).collect())
                            }
                        }
                    })
                };
                K::ExtrudedPolygon {
                    outer,
                    holes,
                    colors,
                    depth: l(d1, d2),
                    roundness: l(r1, r2),
                }
            }
            (
                K::Lathe {
                    profile: p1,
                    segments: s1,
                },
                K::Lathe {
                    profile: p2,
                    segments: s2,
                },
            ) => K::Lathe {
                profile: lerp_points(p1, p2, t, lerp_vec2)?,
                segments: pick(t, *s1, *s2),
            },
            (
                K::Polyline2D {
                    points: p1,
                    thickness: t1,
                    closed: c1,
                    segments: s1,
                },
                K::Polyline2D {
                    points: p2,
                    thickness: t2,
                    closed: c2,
                    segments: s2,
                },
            ) => K::Polyline2D {
                points: lerp_points(p1, p2, t, lerp_vec2)?,
                thickness: l(t1, t2),
                closed: pick(t, *c1, *c2),
                segments: pick(t, *s1, *s2),
            },
            (
                K::Bezier2D {
                    control_points: p1,
                    thickness: t1,
                    segments: s1,
                },
                K::Bezier2D {
                    control_points: p2,
                    thickness: t2,
                    segments: s2,
                },
            ) => K::Bezier2D {
                control_points: lerp_points(p1, p2, t, lerp_vec2)?,
                thickness: l(t1, t2),
                segments: pick(t, *s1, *s2),
            },
            _ => return None,
        })
    }
}

impl Shape {
    /// Blends this shape towards `other` by `t` (clamped to `0.0..=1.0`).
    ///
    /// * Two shapes of the **same kind** interpolate every continuous parameter
    ///   (radius, height, size, roundness, rim width, angles, point positions).
    /// * Integer counts (`sides`, `points`, `segments`), flags (`capped`,
    ///   `closed`) and the [`ShapeDetail`] snap to whichever endpoint is nearer.
    /// * Closed polygons ([`Shape::arbitrary_polygon`] and
    ///   [`Shape::extruded_polygon`], including holes) with **different vertex
    ///   counts** morph too: both outlines are grown to the larger count by
    ///   splitting their longest edges, the second outline is flipped to the
    ///   winding of the first and rotated to the start that moves points the
    ///   least. Per-vertex colours blend alongside (a missing list counts as
    ///   white). Polygons with fewer than three vertices cannot be morphed.
    /// * Filled flat shapes of **different kinds** morph as outlines, so a
    ///   polygon can turn into a star, a disc rim can turn into a rectangle,
    ///   and a ring can turn into a disc.  The blend in between is an arbitrary
    ///   polygon, and the [`ShapeDetail`] gives the segment count of a disc,
    ///   ellipse, ring or wedge.  Rims use their outer edge; a ring's inner
    ///   hole is ignored.  Supported kinds: rectangle, disc, ellipse, polygon,
    ///   star, arbitrary polygon, rectangle rim, disc rim, ellipse rim, polygon
    ///   rim, ring and wedge.
    /// * Other shapes of different kinds (solids), open point lists (polylines,
    ///   curves, lathe profiles) of different lengths, or polygons with a
    ///   different number of holes cannot be blended and yield the nearer
    ///   endpoint.
    ///
    /// `t == 0.0` returns a copy of `self` and `t == 1.0` a copy of `other`.
    ///
    /// Signed-distance-field shapes ([`SdfMaterial2d`](crate::SdfMaterial2d)) are
    /// not `Shape`s: they are a material drawn on one shared quad, so they have
    /// no mesh to blend. Blend them with [`SdfMaterial2d::lerp`](crate::SdfMaterial2d::lerp),
    /// which costs no new mesh.
    ///
    /// # Performance: every blended value is a distinct mesh
    ///
    /// Calling `lerp` with a different `t` each frame produces a new `Shape`
    /// value each frame, which produces a new mesh on `build()`. At 60 Hz this
    /// is 60 new meshes per second. The asset pipeline will eventually
    /// garbage-collect old handles, but the CPU still tessellates every frame.
    ///
    /// To animate smoothly without this cost, use one of these strategies:
    ///
    /// 1. **Animate a continuous proxy**: Instead of morphing the shape, keep
    ///    a fixed mesh and animate its `Transform` (scale, position, rotation).
    ///    This is free — the GPU handles it with no new meshes.
    ///
    /// 2. **Pre-compute a keyframe table**: Build the intermediate shapes once
    ///    at fixed `t` values, store them, and switch between each frame:
    ///
    ///    ```ignore
    ///    // Build once (e.g. in Startup system)
    ///    let keyframes: Vec<Shape> = (0..=10)
    ///        .map(|i| a.lerp(&b, i as f32 / 10.0))
    ///        .collect();
    ///    // Each frame, pick the nearest keyframe
    ///    let idx = (t * 10.0).round() as usize;
    ///    let shape = &keyframes[idx.min(keyframes.len() - 1)];
    ///    ```
    ///
    ///    This limits mesh generation to the number of keyframes (11 in the
    ///    example) and makes each frame a cheap array lookup. Adjust the
    ///    keyframe count to trade smoothness for performance.
    ///
    /// 3. **Quantise `t`**: Round `t` to a small set of steps before calling
    ///    `lerp`, so repeated frames reuse the same `Shape` value. Combined
    ///    with a small cache (`HashMap<f32, Shape>`), this avoids rebuilding
    ///    meshes for repeated `t` values.
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let t = if t.is_nan() { 0.0 } else { t.clamp(0.0, 1.0) };
        if t <= 0.0 {
            return self.clone();
        }
        if t >= 1.0 {
            return other.clone();
        }
        let blended = if std::mem::discriminant(&self.kind) == std::mem::discriminant(&other.kind) {
            self.kind.lerp(&other.kind, t)
        } else {
            self.lerp_flat_outlines(other, t)
        };
        match blended {
            Some(kind) => Self {
                kind,
                detail: if t < 0.5 { self.detail } else { other.detail },
            },
            None if t < 0.5 => self.clone(),
            None => other.clone(),
        }
    }

    /// The outline of a filled flat shape, counter-clockwise, with its
    /// per-vertex colours if it has any.  For rims and rings the outer edge
    /// is returned (the inner hole is ignored), so a rim can cross-morph into
    /// a filled shape of a different kind.
    fn flat_outline(&self) -> Option<(Vec<Vec2>, Option<Vec<Rgba>>)> {
        let round = |radius: Vec2| {
            let n = self.detail.segments() as usize;
            (0..n)
                .map(|i| {
                    let a = i as f32 * 2.0 * PI / n as f32;
                    Vec2::new(a.cos() * radius.x, a.sin() * radius.y)
                })
                .collect::<Vec<_>>()
        };
        Some(match &self.kind {
            ShapeKind::Rectangle { size } => {
                let h = *size * 0.5;
                (
                    vec![Vec2::new(-h.x, -h.y), Vec2::new(h.x, -h.y), Vec2::new(h.x, h.y), Vec2::new(-h.x, h.y)],
                    None,
                )
            }
            ShapeKind::RectangleRim { size, rim_width } => {
                // Outer edge of the rim
                let h = *size * 0.5 + *rim_width;
                (
                    vec![Vec2::new(-h.x, -h.y), Vec2::new(h.x, -h.y), Vec2::new(h.x, h.y), Vec2::new(-h.x, h.y)],
                    None,
                )
            }
            ShapeKind::Disc { radius } => (round(Vec2::splat(*radius)), None),
            ShapeKind::DiscRim { radius, rim_width } => {
                // Outer edge of the rim
                (round(Vec2::splat(*radius + *rim_width)), None)
            }
            ShapeKind::Ellipse { radius } => (round(*radius), None),
            ShapeKind::EllipseRim { radius, rim_width } => {
                // Outer edge of the rim (approximate: add rim_width to both axes)
                (round(*radius + Vec2::splat(*rim_width)), None)
            }
            ShapeKind::Polygon { sides, radius } => {
                let n = (*sides).max(3) as usize;
                (
                    (0..n)
                        .map(|i| {
                            let a = i as f32 * 2.0 * PI / n as f32;
                            Vec2::new(a.cos() * radius, a.sin() * radius)
                        })
                        .collect(),
                    None,
                )
            }
            ShapeKind::PolygonRim { sides, radius, rim_width } => {
                let n = (*sides).max(3) as usize;
                // Outer edge of the rim
                (
                    (0..n)
                        .map(|i| {
                            let a = i as f32 * 2.0 * PI / n as f32;
                            Vec2::new(a.cos() * (radius + rim_width), a.sin() * (radius + rim_width))
                        })
                        .collect(),
                    None,
                )
            }
            ShapeKind::Star {
                points,
                outer_radius,
                inner_radius,
            } => {
                let n = (*points).max(2) as usize;
                (
                    (0..n * 2)
                        .map(|i| {
                            let a = i as f32 * PI / n as f32 - PI / 2.0;
                            let r = if i % 2 == 0 { *outer_radius } else { *inner_radius };
                            Vec2::new(a.cos() * r, a.sin() * r)
                        })
                        .collect(),
                    None,
                )
            }
            ShapeKind::Ring { outer_radius, inner_radius: _ } => {
                // Outer edge only; inner hole ignored for outline morphing
                (round(Vec2::splat(*outer_radius)), None)
            }
            ShapeKind::Wedge { radius, angle } => {
                // Wedge outline: centre -> tip along start angle -> arc -> centre
                let n = self.detail.segments() as usize;
                let mut verts = Vec::with_capacity(n + 2);
                verts.push(Vec2::ZERO); // centre
                for i in 0..=n {
                    let a = i as f32 * *angle / n as f32;
                    verts.push(Vec2::new(a.cos() * radius, a.sin() * radius));
                }
                (verts, None)
            }
            ShapeKind::ArbitraryPolygon { vertices, colors } => (vertices.clone(), colors.clone()),
            _ => return None,
        })
    }

    /// Blends two different kinds of filled flat shape as outlines, the way
    /// two arbitrary polygons of different vertex counts blend.  Supported
    /// kinds: rectangle, disc, ellipse, polygon, star, arbitrary polygon,
    /// as well as rims (rectangle, disc, ellipse, polygon), rings and wedges.
    /// For rims and rings only the outer edge participates in the morph.
    fn lerp_flat_outlines(&self, other: &Self, t: f32) -> Option<ShapeKind> {
        let (a, ca) = self.flat_outline()?;
        let (b, cb) = other.flat_outline()?;
        let (vertices, colors) = morph_rings(&a, ca.as_deref(), &b, cb.as_deref(), t)?;
        Some(ShapeKind::ArbitraryPolygon { vertices, colors })
    }
}

#[cfg(test)]
mod lerp_tests {
    use super::*;

    const D: ShapeDetail = ShapeDetail::Medium;

    #[test]
    fn endpoints_are_exact() {
        let a = Shape::sphere(1.0, D);
        let b = Shape::sphere(3.0, D);
        assert_eq!(a.lerp(&b, 0.0), a);
        assert_eq!(a.lerp(&b, 1.0), b);
        assert_eq!(a.lerp(&b, f32::NAN), a);
        assert_eq!(a.lerp(&b, 5.0), b);
    }

    #[test]
    fn same_kind_interpolates_parameters() {
        let a = Shape::cuboid(Vec3::new(1.0, 2.0, 3.0), 0.0, D);
        let b = Shape::cuboid(Vec3::new(3.0, 6.0, 5.0), 0.5, D);
        assert_eq!(
            a.lerp(&b, 0.5),
            Shape::cuboid(Vec3::new(2.0, 4.0, 4.0), 0.25, D)
        );
        let c = Shape::cylinder(1.0, 2.0, D);
        let d = Shape::cylinder(3.0, 6.0, D);
        assert_eq!(c.lerp(&d, 0.5), Shape::cylinder(2.0, 4.0, D));
    }

    #[test]
    fn counts_and_detail_snap_to_nearer_endpoint() {
        let a = Shape::regular_prism(4, 1.0, 1.0, 0.0, ShapeDetail::Low);
        let b = Shape::regular_prism(8, 3.0, 1.0, 0.0, ShapeDetail::High);
        assert_eq!(
            a.lerp(&b, 0.25),
            Shape::regular_prism(4, 1.5, 1.0, 0.0, ShapeDetail::Low)
        );
        assert_eq!(
            a.lerp(&b, 0.75),
            Shape::regular_prism(8, 2.5, 1.0, 0.0, ShapeDetail::High)
        );
    }

    #[test]
    fn different_flat_kinds_morph_as_outlines() {
        let polygon = Shape::polygon(5, 2.0, D);
        let star = Shape::star(5, 2.0, 1.0, D);
        assert_eq!(polygon.lerp(&star, 0.0), polygon);
        assert_eq!(polygon.lerp(&star, 1.0), star);
        let mid = polygon.lerp(&star, 0.5);
        assert!(matches!(mid.kind, ShapeKind::ArbitraryPolygon { .. }));
        assert!(!mid.build().attribute(Mesh::ATTRIBUTE_POSITION).unwrap().is_empty());
        // The middle sits between the ends: smaller than the star's tips, wider
        // than the star's dents.
        let b = mid.bounds();
        assert!(b.size().x > 2.0 && b.size().x <= 4.0 + 1e-4);
        // A rectangle turns into a disc, which takes its segment count from the detail.
        let rect = Shape::rectangle(Vec2::new(4.0, 2.0), D);
        let disc = Shape::disc(1.0, ShapeDetail::Low);
        let mid = rect.lerp(&disc, 0.5);
        match &mid.kind {
            ShapeKind::ArbitraryPolygon { vertices, .. } => assert!(vertices.len() >= 12),
            other => panic!("unexpected {other:?}"),
        }
        // Rings now morph as outlines (outer edge only).
        let ring = Shape::ring(2.0, 1.0, D);
        let mid = star.lerp(&ring, 0.5);
        assert!(matches!(mid.kind, ShapeKind::ArbitraryPolygon { .. }));
        // Solids still snap.
        let sphere = Shape::sphere(1.0, D);
        assert_eq!(star.lerp(&sphere, 0.4), star);
        assert_eq!(star.lerp(&sphere, 0.6), sphere);
    }

    #[test]
    fn different_kinds_yield_nearer_shape() {
        let a = Shape::sphere(1.0, D);
        let b = Shape::disc(1.0, D);
        assert_eq!(a.lerp(&b, 0.4), a);
        assert_eq!(a.lerp(&b, 0.6), b);
    }

    #[test]
    fn point_lists_blend_only_when_same_length() {
        let a = Shape::polyline3d(vec![Vec3::ZERO, Vec3::X], 0.1, false, 8, D);
        let b = Shape::polyline3d(vec![Vec3::Y, Vec3::X + Vec3::Y], 0.3, false, 8, D);
        let mid = a.lerp(&b, 0.5);
        assert_eq!(
            mid,
            Shape::polyline3d(
                vec![Vec3::Y * 0.5, Vec3::X + Vec3::Y * 0.5],
                0.2,
                false,
                8,
                D
            )
        );
        let c = Shape::polyline3d(vec![Vec3::ZERO, Vec3::X, Vec3::Z], 0.1, false, 8, D);
        assert_eq!(a.lerp(&c, 0.4), a);
        assert_eq!(a.lerp(&c, 0.6), c);
    }

    #[test]
    fn star_blend_keeps_point_count_of_nearer_end() {
        let a = Shape::star(5, 1.0, 0.4, D);
        let b = Shape::star(7, 2.0, 0.8, D);
        assert_eq!(a.lerp(&b, 0.25), Shape::star(5, 1.25, 0.5, D));
    }

    #[test]
    fn rim_can_morph_into_filled_shape() {
        // A disc rim can cross-morph into a rectangle
        let rim = Shape::disc_rim(1.0, 0.3, D);
        let rect = Shape::rectangle(Vec2::splat(2.6), D);
        let mid = rim.lerp(&rect, 0.5);
        assert!(matches!(mid.kind, ShapeKind::ArbitraryPolygon { .. }));
        let mesh = mid.build();
        assert!(mesh.get_vertex_buffer_size() > 0);

        // A ring can cross-morph into a disc
        let ring = Shape::ring(2.0, 1.0, D);
        let disc = Shape::disc(2.0, D);
        let mid = ring.lerp(&disc, 0.5);
        assert!(matches!(mid.kind, ShapeKind::ArbitraryPolygon { .. }));
    }

    #[test]
    fn wedge_can_morph_into_disc() {
        let wedge = Shape::wedge(2.0, std::f32::consts::PI, D);
        let disc = Shape::disc(2.0, D);
        let mid = wedge.lerp(&disc, 0.5);
        assert!(matches!(mid.kind, ShapeKind::ArbitraryPolygon { .. }));
        let mesh = mid.build();
        assert!(mesh.get_vertex_buffer_size() > 0);
    }

    /// Verifies that ExtrudedPolygon lerp handles missing colors consistently
    /// regardless of vertex count: missing colors are treated as white, matching
    /// the `morph_rings` behavior.
    #[test]
    fn extruded_polygon_color_blending() {
        let outer = vec![
            Vec2::new(-1.0, -1.0),
            Vec2::new( 1.0, -1.0),
            Vec2::new( 1.0,  1.0),
            Vec2::new(-1.0,  1.0),
        ];
        let colors: Vec<[f32; 4]> = outer.iter().map(|_| [1.0, 0.0, 0.0, 1.0]).collect();

        // Same vertex count: one has colors, the other doesn't.
        // Missing colors are treated as white, so the blend succeeds.
        let with_color = Shape::extruded_polygon_with_options(
            outer.clone(), vec![], Some(colors.clone()),
            1.0, 0.0, D,
        );
        let no_color = Shape::extruded_polygon_with_options(
            outer.clone(), vec![], None,
            1.0, 0.0, D,
        );
        let mid = with_color.lerp(&no_color, 0.5);
        // The blend now succeeds: colors are blended towards white.
        match &mid.kind {
            ShapeKind::ExtrudedPolygon { outer: b_outer, colors: b_colors, .. } => {
                assert_eq!(b_outer.len(), 4, "same-count lerp keeps vertex count");
                let blended_colors = b_colors.as_ref().expect("colors should be present when blending with white");
                assert_eq!(blended_colors.len(), 4);
                // Red [1,0,0,1] blended with white [1,1,1,1] at t=0.5 → [1, 0.5, 0.5, 1]
                for c in blended_colors {
                    assert!((c[0] - 1.0).abs() < 0.01, "red channel should stay at 1.0");
                    assert!((c[1] - 0.5).abs() < 0.01, "green channel should blend to 0.5");
                    assert!((c[2] - 0.5).abs() < 0.01, "blue channel should blend to 0.5");
                }
            }
            other => panic!("expected ExtrudedPolygon, got {other:?}"),
        }

        // Reverse direction: no color → with color
        let mid2 = no_color.lerp(&with_color, 0.5);
        match &mid2.kind {
            ShapeKind::ExtrudedPolygon { colors: b_colors, .. } => {
                let blended_colors = b_colors.as_ref().expect("colors should be present");
                // White blended with red at t=0.5 → [1, 0.5, 0.5, 1]
                for c in blended_colors {
                    assert!((c[0] - 1.0).abs() < 0.01);
                    assert!((c[1] - 0.5).abs() < 0.01);
                    assert!((c[2] - 0.5).abs() < 0.01);
                }
            }
            other => panic!("expected ExtrudedPolygon, got {other:?}"),
        }

        // Different vertex count: morph_rings treats missing colors as white,
        // so the blend succeeds.
        let outer_5 = vec![
            Vec2::new(0.0, -1.0),
            Vec2::new( 0.95, -0.31),
            Vec2::new( 0.59,  0.81),
            Vec2::new(-0.59,  0.81),
            Vec2::new(-0.95, -0.31),
        ];
        let no_color_5 = Shape::extruded_polygon_with_options(
            outer_5.clone(), vec![], None,
            1.0, 0.0, D,
        );
        let blended = with_color.lerp(&no_color_5, 0.5);
        match &blended.kind {
            ShapeKind::ExtrudedPolygon { outer: b_outer, colors: b_colors, .. } => {
                assert_eq!(b_outer.len(), 5, "morph_rings grows to the larger count");
                assert!(b_colors.is_some(), "missing colors are treated as white, so colors should be present");
            }
            other => panic!("expected ExtrudedPolygon, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(shape: &Shape) -> FlatShape<'_> {
        shape.as_flat().unwrap()
    }

    #[test]
    fn solids_are_not_flat() {
        let d = ShapeDetail::Medium;
        assert_eq!(Shape::sphere(1.0, d).as_flat().unwrap_err(), NotFlat);
        let mixed = Shape::compound(vec![
            ShapePart::new(Shape::disc(1.0, d)),
            ShapePart::new(Shape::sphere(1.0, d)),
        ]);
        assert!(mixed.as_flat().is_err());
        let flat_parts = Shape::compound(vec![ShapePart::new(Shape::disc(1.0, d))]);
        assert!(flat_parts.as_flat().is_ok());
    }

    #[test]
    fn sdf_2d_maps_flat_shapes() {
        let d = ShapeDetail::Medium;
        let rect = Shape::rectangle(Vec2::new(40.0, 20.0), d);
        assert_eq!(flat(&rect).sdf_2d().unwrap().size(), Vec2::new(40.0, 20.0));
        let disc = Shape::disc(10.0, d);
        assert_eq!(flat(&disc).sdf_2d().unwrap().size(), Vec2::splat(20.0));
        let rim = Shape::disc_rim(10.0, 3.0, d);
        assert_eq!(flat(&rim).sdf_2d().unwrap().uniform.shape.y, 3.0);
        let ring = Shape::ring(10.0, 6.0, d);
        let ring = flat(&ring).sdf_2d().unwrap();
        assert_eq!(ring.uniform.shape.y, 4.0);
        assert_eq!(ring.size(), Vec2::splat(20.0));
        let wedge = Shape::wedge(10.0, 1.5, d);
        let wedge = flat(&wedge).sdf_2d().unwrap();
        assert_eq!(wedge.uniform.shape.z, 0.0);
        assert_eq!(wedge.uniform.shape.w, 1.5);
        for shape in [
            Shape::ellipse(Vec2::new(8.0, 4.0), d),
            Shape::polygon(6, 10.0, d),
            Shape::star(5, 10.0, 4.0, d),
        ] {
            assert!(flat(&shape).sdf_2d().is_ok());
        }
    }

    #[test]
    fn sdf_2d_polygon_rim_matches_mesh_thickness() {
        let shape = Shape::polygon_rim(6, 10.0, 2.0, ShapeDetail::Medium);
        let rim = flat(&shape).sdf_2d().unwrap();
        let expected = 2.0 * (PI / 6.0).cos();
        assert!((rim.uniform.shape.y - expected).abs() < 1e-5);
    }

    #[test]
    fn sdf_2d_arbitrary_polygon_errors_say_why() {
        let d = ShapeDetail::Medium;
        let triangle = vec![Vec2::ZERO, Vec2::new(4.0, 0.0), Vec2::new(0.0, 4.0)];
        let shape = Shape::arbitrary_polygon(triangle, d);
        assert!(flat(&shape).sdf_2d().is_ok());
        let many: Vec<Vec2> = (0..17).map(|i| Vec2::new(i as f32, (i * i) as f32)).collect();
        let shape = Shape::arbitrary_polygon(many, d);
        assert_eq!(
            flat(&shape).sdf_2d().unwrap_err(),
            SdfConvertError::Polygon(crate::SdfPolygonError::TooManyVertices(17))
        );
        let shape = Shape::arbitrary_polygon(vec![Vec2::ZERO, Vec2::X], d);
        assert_eq!(
            flat(&shape).sdf_2d().unwrap_err(),
            SdfConvertError::Polygon(crate::SdfPolygonError::TooFewVertices(2))
        );
    }

    #[test]
    fn sdf_2d_rejects_what_it_cannot_draw_with_a_reason() {
        let d = ShapeDetail::Medium;
        for shape in [
            Shape::disc(0.0, d),
            Shape::disc(f32::NAN, d),
            Shape::ring(5.0, 5.0, d),
            Shape::wedge(5.0, 0.0, d),
            Shape::disc_rim(5.0, f32::INFINITY, d),
        ] {
            assert_eq!(flat(&shape).sdf_2d().unwrap_err(), SdfConvertError::Degenerate);
        }
        let compound = Shape::compound(vec![ShapePart::new(Shape::disc(1.0, d))]);
        assert_eq!(flat(&compound).sdf_2d().unwrap_err(), SdfConvertError::Unsupported);
    }

    #[test]
    fn sphere_has_valid_topology() {
        let mesh = Shape::sphere(1.0, ShapeDetail::Medium).build();
        assert!(
            mesh.get_vertex_buffer_size() > 0,
            "Sphere must have vertices"
        );
        assert!(mesh.indices().is_some(), "Sphere must have indices");
    }

    #[test]
    fn capsule_generates_mesh() {
        let mesh = Shape::capsule(0.5, 2.0, ShapeDetail::Medium).build();
        assert!(
            mesh.get_vertex_buffer_size() > 0,
            "Capsule must have vertices"
        );
    }

    #[test]
    fn torus_generates_mesh() {
        let full = Shape::torus(1.0, 0.3, 0.0, std::f32::consts::TAU, ShapeDetail::Medium).build();
        assert!(
            full.get_vertex_buffer_size() > 0,
            "Torus must have vertices"
        );
    }

    #[test]
    fn hemisphere_capped_has_cap() {
        let capped = Shape::hemisphere(1.0, true, ShapeDetail::Medium).build();
        let uncapped = Shape::hemisphere(1.0, false, ShapeDetail::Medium).build();
        assert!(
            capped.get_vertex_buffer_size() > uncapped.get_vertex_buffer_size(),
            "Capped hemisphere should have more vertices"
        );
    }

    #[test]
    fn detail_levels_increase_vertices() {
        let low = Shape::sphere(1.0, ShapeDetail::Low).build();
        let high = Shape::sphere(1.0, ShapeDetail::High).build();
        assert!(
            high.get_vertex_buffer_size() > low.get_vertex_buffer_size(),
            "Higher detail should have more vertices"
        );
    }

    #[test]
    fn all_primitives_generate_meshes() {
        let primitives = [
            Shape::sphere(1.0, ShapeDetail::Medium),
            Shape::ellipsoid(Vec3::ONE, ShapeDetail::Medium),
            Shape::cuboid(Vec3::ONE, 0.0, ShapeDetail::Medium),
            Shape::cuboid(Vec3::ONE, 0.5, ShapeDetail::Medium),
            Shape::cylinder(0.5, 1.0, ShapeDetail::Medium),
            Shape::cone(0.5, 1.0, ShapeDetail::Medium),
            Shape::capsule(0.3, 1.0, ShapeDetail::Medium),
            Shape::hemisphere(0.5, true, ShapeDetail::Medium),
            Shape::torus(0.5, 0.2, 0.0, std::f32::consts::TAU, ShapeDetail::Medium),
            Shape::regular_prism(6, 0.5, 1.0, 0.0, ShapeDetail::Medium),
            Shape::pyramid(4, 0.5, 1.0, ShapeDetail::Medium),
        ];
        for prim in primitives {
            let mesh = prim.build();
            assert!(
                mesh.get_vertex_buffer_size() > 0,
                "Primitive must generate vertices"
            );
        }
    }

    #[test]
    fn rounded_cuboid_has_more_vertices_than_simple() {
        let simple = Shape::cuboid(Vec3::ONE, 0.0, ShapeDetail::Medium).build();
        let rounded = Shape::cuboid(Vec3::ONE, 0.5, ShapeDetail::Medium).build();
        assert!(
            rounded.get_vertex_buffer_size() > simple.get_vertex_buffer_size(),
            "Rounded cuboid should have more vertices than simple box"
        );
    }

    #[test]
    fn rounded_cuboid_has_smooth_normals() {
        let mesh = Shape::cuboid(Vec3::ONE, 0.8, ShapeDetail::High).build();
        let normals_data = mesh
            .attribute(Mesh::ATTRIBUTE_NORMAL)
            .expect("Missing normals");
        let normals: Vec<[f32; 3]> = match normals_data {
            bevy::render::mesh::VertexAttributeValues::Float32x3(v) => v.clone(),
            _ => panic!("Expected Float32x3 normals"),
        };
        // Check that normals are not all axis-aligned (which would indicate no rounding).
        // With the rounding algorithm, smooth normals appear at corners and edges where the
        // fillet curves. Flat face centers retain axis-aligned normals. We check that
        // a significant fraction of normals have at least two nonzero components,
        // indicating the fillet regions are generating smooth normals.
        let non_axis_aligned = normals
            .iter()
            .filter(|n| {
                let abs = [n[0].abs(), n[1].abs(), n[2].abs()];
                // Count how many components are significant (> 0.1)
                let significant = abs.iter().filter(|&&a| a > 0.1).count();
                // Non-axis-aligned normals have at least 2 significant components
                // and are not dominated by a single component
                significant >= 2 && abs[0].max(abs[1]).max(abs[2]) < 0.99
            })
            .count();
        assert!(
            non_axis_aligned > normals.len() / 8,
            "Rounded cuboid should have non-axis-aligned normals in fillet regions, got {} of {}",
            non_axis_aligned,
            normals.len()
        );
    }

    #[test]
    fn disc_rim_has_hole() {
        let solid = Shape::disc(1.0, ShapeDetail::Medium).build();
        let rim = Shape::disc_rim(1.0, 0.3, ShapeDetail::Medium).build();
        // Rim should have more vertices (outer + inner ring) than solid (center + ring)
        assert!(
            rim.get_vertex_buffer_size() > solid.get_vertex_buffer_size(),
            "Disc rim should have more vertices than solid disc"
        );
    }

    #[test]
    fn polygon_sides_match() {
        for sides in 3..=8 {
            let mesh = Shape::polygon(sides, 1.0, ShapeDetail::Medium).build();
            let verts = mesh.count_vertices();
            // Center + sides + 1 (closing vertex) = sides + 2
            assert_eq!(
                verts as usize,
                (sides + 2) as usize,
                "{}-sided polygon should have {} vertices",
                sides,
                sides + 2
            );
        }
    }

    #[test]
    fn all_2d_shapes_generate_meshes() {
        let shapes = [
            Shape::rectangle(Vec2::new(2.0, 1.0), ShapeDetail::Medium),
            Shape::rectangle_rim(Vec2::new(2.0, 1.0), 0.2, ShapeDetail::Medium),
            Shape::disc(1.0, ShapeDetail::Medium),
            Shape::disc_rim(1.0, 0.3, ShapeDetail::Medium),
            Shape::polygon(6, 1.0, ShapeDetail::Medium),
            Shape::polygon_rim(6, 1.0, 0.3, ShapeDetail::Medium),
            Shape::ellipse(Vec2::new(1.5, 1.0), ShapeDetail::Medium),
            Shape::ellipse_rim(Vec2::new(1.5, 1.0), 0.2, ShapeDetail::Medium),
        ];
        for shape in shapes {
            let mesh = shape.build();
            assert!(mesh.count_vertices() > 0, "2D shape must generate vertices");
        }
    }

    #[test]
    fn rounded_cuboid_roundness_affects_geometry() {
        let low_round = Shape::cuboid(Vec3::ONE, 0.1, ShapeDetail::Medium).build();
        let high_round = Shape::cuboid(Vec3::ONE, 0.9, ShapeDetail::Medium).build();

        let low_verts: Vec<[f32; 3]> = match low_round
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .expect("Missing positions")
        {
            bevy::render::mesh::VertexAttributeValues::Float32x3(v) => v.clone(),
            _ => panic!("Expected Float32x3 positions"),
        };
        let high_verts: Vec<[f32; 3]> = match high_round
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .expect("Missing positions")
        {
            bevy::render::mesh::VertexAttributeValues::Float32x3(v) => v.clone(),
            _ => panic!("Expected Float32x3 positions"),
        };

        // Higher roundness should pull vertices more toward the center
        let low_avg_dist: f32 = low_verts
            .iter()
            .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
            .sum::<f32>()
            / low_verts.len() as f32;
        let high_avg_dist: f32 = high_verts
            .iter()
            .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
            .sum::<f32>()
            / high_verts.len() as f32;
        assert!(
            high_avg_dist < low_avg_dist,
            "Higher roundness should pull vertices closer to center: low={:.3}, high={:.3}",
            low_avg_dist,
            high_avg_dist
        );
    }

    #[test]
    fn torus_arc_has_end_caps() {
        // Partial torus should have more vertices than full torus due to end cap centers
        let full = Shape::torus(1.0, 0.3, 0.0, std::f32::consts::TAU, ShapeDetail::Medium).build();
        let partial = Shape::torus(1.0, 0.3, 0.0, std::f32::consts::PI, ShapeDetail::Medium).build();
        assert!(
            partial.count_vertices() > full.count_vertices(),
            "Partial torus should have end cap vertices: full={}, partial={}",
            full.count_vertices(),
            partial.count_vertices()
        );
    }

    #[test]
    fn torus_full_has_no_end_caps() {
        // Full torus (span = TAU) should not have end caps
        let full = Shape::torus(1.0, 0.3, 0.0, std::f32::consts::TAU, ShapeDetail::Medium).build();
        let segments = ShapeDetail::Medium.segments() as usize;
        let minor = (segments / 2).max(4);
        // Full torus: (major+1) * (minor+1) vertices, no caps
        let expected = (segments + 1) * (minor + 1);
        assert_eq!(
            full.count_vertices() as usize,
            expected,
            "Full torus should have exactly {} vertices, got {}",
            expected,
            full.count_vertices()
        );
    }

    #[test]
    fn pyramid_faces_have_uniform_normals() {
        // Each face should have 3 vertices with identical normals (no bleeding)
        let mesh = Shape::pyramid(3, 1.0, 1.0, ShapeDetail::Medium).build();
        let normals_data = mesh
            .attribute(Mesh::ATTRIBUTE_NORMAL)
            .expect("Missing normals");
        let normals: Vec<[f32; 3]> = match normals_data {
            bevy::render::mesh::VertexAttributeValues::Float32x3(v) => v.clone(),
            _ => panic!("Expected Float32x3 normals"),
        };
        let indices = mesh.indices().expect("Missing indices");
        let idx = match indices {
            bevy::render::mesh::Indices::U32(i) => i.clone(),
            _ => panic!("Expected U32 indices"),
        };

        // Group vertices by triangle and check each triangle has uniform normals
        let mut uniform_triangles = 0;
        let mut total_triangles = 0;
        for chunk in idx.chunks(3) {
            total_triangles += 1;
            let i0 = chunk[0] as usize;
            let i1 = chunk[1] as usize;
            let i2 = chunk[2] as usize;
            let n0 = &normals[i0];
            let n1 = &normals[i1];
            let n2 = &normals[i2];
            let same = |a: &[f32; 3], b: &[f32; 3]| {
                (a[0] - b[0]).abs() < 1e-6
                    && (a[1] - b[1]).abs() < 1e-6
                    && (a[2] - b[2]).abs() < 1e-6
            };
            if same(n0, n1) && same(n1, n2) {
                uniform_triangles += 1;
            }
        }
        assert_eq!(
            uniform_triangles, total_triangles,
            "All pyramid triangles should have uniform normals (no shared-vertex bleeding), got {} of {}",
            uniform_triangles, total_triangles
        );
    }

    #[test]
    fn pyramid_vertex_count_matches_faces() {
        // N-sided pyramid: N faces * 3 vertices + base center + (N+1) base ring vertices
        for sides in 3..=6 {
            let mesh = Shape::pyramid(sides, 1.0, 1.0, ShapeDetail::Medium).build();
            let expected = (sides as usize) * 3 + 1 + (sides as usize + 1);
            assert_eq!(
                mesh.count_vertices() as usize,
                expected,
                "{}-sided pyramid should have {} vertices",
                sides,
                expected
            );
        }
    }

    #[test]
    fn rounded_cuboid_all_triangles_face_outward() {
        // Check that every triangle's geometric normal (computed from vertex positions)
        // points outward (away from the mesh center). Inverted triangles would cause
        // a "folded" visual artifact due to backface culling or incorrect lighting.
        for roundness in [0.0, 0.25, 0.5, 0.8, 1.0] {
            let mesh = Shape::cuboid(Vec3::ONE, roundness, ShapeDetail::High).build();
            let positions_data = mesh
                .attribute(Mesh::ATTRIBUTE_POSITION)
                .expect("Missing positions");
            let positions: Vec<[f32; 3]> = match positions_data {
                bevy::render::mesh::VertexAttributeValues::Float32x3(v) => v.clone(),
                _ => panic!("Expected Float32x3 positions"),
            };
            let indices = mesh.indices().expect("Missing indices");
            let idx = match indices {
                bevy::render::mesh::Indices::U32(i) => i.clone(),
                _ => panic!("Expected U32 indices"),
            };

            let mut inward_count = 0;
            let mut total = 0;
            for chunk in idx.chunks(3) {
                let a = positions[chunk[0] as usize];
                let b = positions[chunk[1] as usize];
                let c = positions[chunk[2] as usize];

                // Centroid of the triangle
                let cx = (a[0] + b[0] + c[0]) / 3.0;
                let cy = (a[1] + b[1] + c[1]) / 3.0;
                let cz = (a[2] + b[2] + c[2]) / 3.0;

                // Edge vectors
                let e1x = b[0] - a[0];
                let e1y = b[1] - a[1];
                let e1z = b[2] - a[2];
                let e2x = c[0] - a[0];
                let e2y = c[1] - a[1];
                let e2z = c[2] - a[2];

                // Cross product (geometric face normal)
                let nx = e1y * e2z - e1z * e2y;
                let ny = e1z * e2x - e1x * e2z;
                let nz = e1x * e2y - e1y * e2x;
                let len = (nx * nx + ny * ny + nz * nz).sqrt();
                if len < 1e-12 {
                    continue; // degenerate triangle
                }
                let nx = nx / len;
                let ny = ny / len;
                let nz = nz / len;

                // Vector from centroid to origin (inward direction)
                let to_origin_len = (cx * cx + cy * cy + cz * cz).sqrt();
                if to_origin_len < 1e-12 {
                    continue; // centroid at origin, skip
                }
                let tox = -cx / to_origin_len;
                let toy = -cy / to_origin_len;
                let toz = -cz / to_origin_len;

                // If face normal points toward the origin, it's inward-facing
                let dot = nx * tox + ny * toy + nz * toz;
                total += 1;
                if dot > 0.0 {
                    inward_count += 1;
                }
            }

            assert_eq!(
                inward_count, 0,
                "Rounded cuboid (roundness={}) has {} inward-facing triangles out of {}",
                roundness, inward_count, total
            );
        }
    }

    #[test]
    fn rounded_cuboid_no_degenerate_triangles() {
        // Check that no triangles are degenerate (near-zero area)
        for roundness in [0.0, 0.5, 1.0] {
            let mesh = Shape::cuboid(Vec3::ONE, roundness, ShapeDetail::High).build();
            let positions_data = mesh
                .attribute(Mesh::ATTRIBUTE_POSITION)
                .expect("Missing positions");
            let positions: Vec<[f32; 3]> = match positions_data {
                bevy::render::mesh::VertexAttributeValues::Float32x3(v) => v.clone(),
                _ => panic!("Expected Float32x3 positions"),
            };
            let indices = mesh.indices().expect("Missing indices");
            let idx = match indices {
                bevy::render::mesh::Indices::U32(i) => i.clone(),
                _ => panic!("Expected U32 indices"),
            };

            let mut degenerate = 0;
            let mut total = 0;
            for chunk in idx.chunks(3) {
                let a = positions[chunk[0] as usize];
                let b = positions[chunk[1] as usize];
                let c = positions[chunk[2] as usize];

                let e1x = b[0] - a[0];
                let e1y = b[1] - a[1];
                let e1z = b[2] - a[2];
                let e2x = c[0] - a[0];
                let e2y = c[1] - a[1];
                let e2z = c[2] - a[2];

                let nx = e1y * e2z - e1z * e2y;
                let ny = e1z * e2x - e1x * e2z;
                let nz = e1x * e2y - e1y * e2x;
                let area = (nx * nx + ny * ny + nz * nz).sqrt() / 2.0;

                total += 1;
                if area < 1e-8 {
                    degenerate += 1;
                }
            }

            assert_eq!(
                degenerate, 0,
                "Rounded cuboid (roundness={}) has {} degenerate triangles out of {}",
                roundness, degenerate, total
            );
        }
    }

    #[test]
    fn arbitrary_polygon_convex() {
        // A simple convex pentagon
        let vertices = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(0.0, 2.0),
            Vec2::new(1.0, 3.0),
        ];
        let mesh = Shape::arbitrary_polygon(vertices, ShapeDetail::Medium).build();
        let vertex_count = mesh.count_vertices();
        assert_eq!(vertex_count, 5, "Should have 5 vertices");
        let indices = mesh.indices().expect("Should have indices");
        let idx = match indices {
            bevy::render::mesh::Indices::U32(i) => i.clone(),
            _ => panic!("Expected U32 indices"),
        };
        // 5 vertices -> 3 triangles -> 9 indices
        assert_eq!(idx.len(), 9, "Should have 9 indices (3 triangles)");
    }

    #[test]
    fn arbitrary_polygon_triangle() {
        let vertices = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.5, 1.0),
        ];
        let mesh = Shape::arbitrary_polygon(vertices, ShapeDetail::Medium).build();
        assert_eq!(mesh.count_vertices(), 3);
        let indices = mesh.indices().expect("Should have indices");
        let idx = match indices {
            bevy::render::mesh::Indices::U32(i) => i.clone(),
            _ => panic!("Expected U32 indices"),
        };
        assert_eq!(idx.len(), 3, "Triangle should have 3 indices");
    }

    #[test]
    fn arbitrary_polygon_empty() {
        // Fewer than 3 vertices should produce an empty mesh
        let vertices = vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0)];
        let mesh = Shape::arbitrary_polygon(vertices, ShapeDetail::Medium).build();
        assert_eq!(mesh.count_vertices(), 0);
    }

    #[test]
    fn arbitrary_polygon_colored() {
        let vertices = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.5, 1.0),
        ];
        let colors = vec![
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ];
        let mesh = Shape::arbitrary_polygon_colored(vertices, colors, ShapeDetail::Medium).build();
        assert_eq!(mesh.count_vertices(), 3);
        assert!(
            mesh.attribute(Mesh::ATTRIBUTE_COLOR).is_some(),
            "Should have color attribute"
        );
    }

    #[test]
    fn arbitrary_polygon_concave() {
        // L-shaped concave polygon
        let vertices = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 2.0),
            Vec2::new(0.0, 2.0),
        ];
        let mesh = Shape::arbitrary_polygon(vertices, ShapeDetail::Medium).build();
        assert_eq!(mesh.count_vertices(), 6);
        let indices = mesh.indices().expect("Should have indices");
        let idx = match indices {
            bevy::render::mesh::Indices::U32(i) => i.clone(),
            _ => panic!("Expected U32 indices"),
        };
        // 6 vertices -> 4 triangles -> 12 indices
        assert_eq!(
            idx.len(),
            12,
            "Concave polygon should have 12 indices (4 triangles)"
        );
    }

    fn polygon_mesh_of(outline: &[Vec2]) -> Mesh {
        Shape::arbitrary_polygon(outline.to_vec(), ShapeDetail::Medium).build()
    }

    fn near_collinear_outlines() -> Vec<Vec<Vec2>> {
        use crate::geometry::test_support::*;
        vec![
            jigsaw_piece(20, 1e-5),
            jigsaw_piece(30, 1e-4),
            staircase(15, 1e-5),
            comb(12, 1e-5),
        ]
    }

    #[test]
    fn arbitrary_polygon_jigsaw_piece_is_one_faithful_polygon() {
        use crate::geometry::test_support::*;
        let outline = jigsaw_piece(20, 1e-5);
        assert!(outline.len() > 100);
        assert_faithful_fill(&outline, &polygon_mesh_of(&outline));
    }

    #[test]
    fn arbitrary_polygon_near_collinear_outlines_fill_exactly() {
        use crate::geometry::test_support::assert_faithful_fill;
        for outline in near_collinear_outlines() {
            assert_faithful_fill(&outline, &polygon_mesh_of(&outline));
        }
    }

    #[test]
    fn arbitrary_polygon_clockwise_matches_counter_clockwise() {
        use crate::geometry::test_support::assert_faithful_fill;
        for mut outline in near_collinear_outlines() {
            outline.reverse();
            assert_faithful_fill(&outline, &polygon_mesh_of(&outline));
        }
        let mut l_shape = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 2.0),
            Vec2::new(0.0, 2.0),
        ];
        l_shape.reverse();
        assert_faithful_fill(&l_shape, &polygon_mesh_of(&l_shape));
    }

    #[test]
    fn arbitrary_polygon_start_vertex_does_not_matter() {
        use crate::geometry::test_support::*;
        let outline = jigsaw_piece(20, 1e-5);
        for shift in [1, 7, 33, outline.len() / 2, outline.len() - 1] {
            let mut rotated = outline.clone();
            rotated.rotate_left(shift);
            assert_faithful_fill(&rotated, &polygon_mesh_of(&rotated));
        }
    }

    #[test]
    fn arbitrary_polygon_ignores_closing_and_repeated_points() {
        use crate::geometry::test_support::*;
        let outline = jigsaw_piece(20, 1e-5);
        let mut messy = Vec::new();
        for (i, p) in outline.iter().enumerate() {
            messy.push(*p);
            if i % 9 == 0 {
                messy.push(*p);
            }
        }
        messy.push(outline[0]);
        assert_faithful_fill(&outline, &polygon_mesh_of(&messy));
    }

    #[test]
    fn arbitrary_polygon_touching_itself_at_a_vertex() {
        use crate::geometry::test_support::assert_faithful_fill;
        // Two triangles joined at (1, 1).
        let pinched = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(0.0, 2.0),
            Vec2::new(1.0, 1.0),
        ];
        assert_faithful_fill(&pinched, &polygon_mesh_of(&pinched));
    }

    #[test]
    fn arbitrary_polygon_all_collinear_is_empty() {
        let line: Vec<Vec2> = (0..12).map(|i| Vec2::new(i as f32, i as f32 * 0.5)).collect();
        let mesh = polygon_mesh_of(&line);
        assert_eq!(mesh.count_vertices(), 0);
    }

    #[test]
    fn arbitrary_polygon_bow_tie_terminates() {
        let bow_tie = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(0.0, 2.0),
        ];
        let mesh = polygon_mesh_of(&bow_tie);
        let n = mesh.count_vertices() as u32;
        if let Some(bevy::render::mesh::Indices::U32(idx)) = mesh.indices() {
            assert!(idx.iter().all(|&i| i < n));
        }
    }

    #[test]
    fn arbitrary_polygon_colors_stay_with_their_vertices() {
        use crate::geometry::test_support::*;
        let color_of = |p: Vec2| [p.x / 100.0, p.y / 100.0, 0.5, 1.0];
        for reverse in [false, true] {
            let mut outline = jigsaw_piece(20, 1e-5);
            if reverse {
                outline.reverse();
            }
            let colors: Vec<[f32; 4]> = outline.iter().map(|&p| color_of(p)).collect();
            let mesh = Shape::arbitrary_polygon_colored(
                outline.clone(),
                colors,
                ShapeDetail::Medium,
            )
            .build();
            assert_faithful_fill(&outline, &mesh);
            let pos = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
                Some(bevy::mesh::VertexAttributeValues::Float32x3(v)) => v.clone(),
                _ => panic!("positions"),
            };
            let cols = match mesh.attribute(Mesh::ATTRIBUTE_COLOR) {
                Some(bevy::mesh::VertexAttributeValues::Float32x4(v)) => v.clone(),
                _ => panic!("Should have color attribute"),
            };
            assert_eq!(cols.len(), pos.len());
            for (p, c) in pos.iter().zip(&cols) {
                let want = color_of(Vec2::new(p[0], p[1]));
                assert!((0..4).all(|k| (c[k] - want[k]).abs() < 1e-6), "colour moved off its vertex");
            }
        }
    }

    #[test]
    fn arbitrary_polygon_uvs_are_bounding_box_normalised() {
        use crate::geometry::test_support::jigsaw_piece;
        let outline = jigsaw_piece(20, 1e-5);
        let mesh = polygon_mesh_of(&outline);
        let pos = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            Some(bevy::mesh::VertexAttributeValues::Float32x3(v)) => v.clone(),
            _ => panic!("positions"),
        };
        let uvs = match mesh.attribute(Mesh::ATTRIBUTE_UV_0) {
            Some(bevy::mesh::VertexAttributeValues::Float32x2(v)) => v.clone(),
            _ => panic!("uvs"),
        };
        let normals = match mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
            Some(bevy::mesh::VertexAttributeValues::Float32x3(v)) => v.clone(),
            _ => panic!("normals"),
        };
        let lo = outline.iter().fold(Vec2::splat(f32::MAX), |a, p| a.min(*p));
        let hi = outline.iter().fold(Vec2::splat(f32::MIN), |a, p| a.max(*p));
        for ((p, uv), n) in pos.iter().zip(&uvs).zip(&normals) {
            let want = (Vec2::new(p[0], p[1]) - lo) / (hi - lo);
            assert!((want - Vec2::new(uv[0], uv[1])).length() < 1e-5);
            assert_eq!(*n, [0.0, 0.0, 1.0]);
        }
    }

    #[test]
    fn arbitrary_polygon_square_is_two_triangles() {
        let square = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];
        let mesh = polygon_mesh_of(&square);
        match mesh.indices() {
            Some(bevy::render::mesh::Indices::U32(i)) => assert_eq!(i.len(), 6),
            _ => panic!("Expected U32 indices"),
        }
    }

    #[test]
    fn polyline3d_basic() {
        let points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ];
        let mesh = Shape::polyline3d(points, 0.1, false, 8, ShapeDetail::Medium).build();
        assert!(mesh.count_vertices() > 0, "Polyline must have vertices");
        assert!(mesh.indices().is_some(), "Polyline must have indices");
    }

    #[test]
    fn polyline3d_closed() {
        let points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let mesh = Shape::polyline3d(points, 0.05, true, 8, ShapeDetail::Medium).build();
        assert!(
            mesh.count_vertices() > 0,
            "Closed polyline must have vertices"
        );
    }

    #[test]
    fn polyline3d_degenerate() {
        // Single point should produce an empty or minimal mesh
        let points = vec![Vec3::new(0.0, 0.0, 0.0)];
        let mesh = Shape::polyline3d(points, 0.1, false, 8, ShapeDetail::Medium).build();
        // Single point still generates a sphere at that point
        assert!(mesh.count_vertices() > 0);
    }

    #[test]
    fn polyline3d_hash_stability() {
        let points1 = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
        let points2 = vec![Vec3::new(0.001, 0.0, 0.0), Vec3::new(1.001, 0.0, 0.0)];
        let shape1 = Shape::polyline3d(points1, 0.1, false, 8, ShapeDetail::Medium);
        let shape2 = Shape::polyline3d(points2, 0.1, false, 8, ShapeDetail::Medium);
        // Quantized to 2 decimal places, these should be equal
        assert_eq!(shape1, shape2, "Quantized points should match");
    }

    #[test]
    fn bezier_curve_basic() {
        // Linear curve (2 control points)
        let control_points = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)];
        let mesh = Shape::bezier_curve(control_points, 0.1, 8, ShapeDetail::Medium).build();
        assert!(mesh.count_vertices() > 0, "Bézier curve must have vertices");
        assert!(mesh.indices().is_some(), "Bézier curve must have indices");
    }

    #[test]
    fn bezier_curve_cubic() {
        // Cubic Bézier (4 control points)
        let control_points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
            Vec3::new(2.0, -2.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        ];
        let mesh = Shape::bezier_curve(control_points, 0.1, 8, ShapeDetail::Medium).build();
        assert!(mesh.count_vertices() > 0, "Cubic Bézier must have vertices");
    }

    #[test]
    fn bezier_curve_degenerate() {
        // Single control point should produce an empty mesh
        let control_points = vec![Vec3::new(0.0, 0.0, 0.0)];
        let mesh = Shape::bezier_curve(control_points, 0.1, 8, ShapeDetail::Medium).build();
        assert_eq!(
            mesh.count_vertices(),
            0,
            "Single point Bézier should be empty"
        );
    }

    #[test]
    fn bezier_curve_hash_stability() {
        let cp1 = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        let cp2 = vec![
            Vec3::new(0.004, 0.0, 0.0),
            Vec3::new(1.004, 2.0, 0.0),
            Vec3::new(2.004, 0.0, 0.0),
        ];
        let shape1 = Shape::bezier_curve(cp1, 0.1, 8, ShapeDetail::Medium);
        let shape2 = Shape::bezier_curve(cp2, 0.1, 8, ShapeDetail::Medium);
        // Quantized to 2 decimal places, these should be equal
        assert_eq!(shape1, shape2, "Quantized control points should match");
    }

    #[test]
    fn tessellate_bezier_linear() {
        let cp = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)];
        let points = tessellate_bezier(&cp, 4);
        assert_eq!(
            points.len(),
            5,
            "Linear Bézier with 4 segments produces 5 points"
        );
        assert!((points[0].x - 0.0).abs() < 1e-6);
        assert!((points[4].x - 2.0).abs() < 1e-6);
    }

    #[test]
    fn tessellate_bezier_cubic_midpoint() {
        // Cubic Bézier from (0,0,0) to (4,0,0) with symmetric control points
        // should pass through (2,0,0) at t=0.5
        let cp = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(3.0, -1.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
        ];
        let points = tessellate_bezier(&cp, 4);
        // At t=0.5 (index 2), the point should be (2, 0, 0)
        assert!((points[2].x - 2.0).abs() < 1e-6, "Midpoint X should be 2.0");
        assert!((points[2].y).abs() < 1e-6, "Midpoint Y should be 0.0");
    }

    // --- New shape tests ---

    #[test]
    fn star_generates_mesh() {
        let mesh = Shape::star(5, 1.0, 0.4, ShapeDetail::Medium).build();
        assert!(mesh.count_vertices() > 0, "Star must have vertices");
        assert!(mesh.indices().is_some(), "Star must have indices");
    }

    #[test]
    fn star_degenerate_2_points() {
        // 2-point star should still produce a valid mesh
        let mesh = Shape::star(2, 1.0, 0.3, ShapeDetail::Medium).build();
        assert!(mesh.count_vertices() > 0, "2-point star must have vertices");
    }

    #[test]
    fn star_hash_stability() {
        let s1 = Shape::star(5, 1.0, 0.4, ShapeDetail::Medium);
        let s2 = Shape::star(5, 1.0, 0.4, ShapeDetail::Medium);
        assert_eq!(s1, s2, "Identical stars should be equal");
    }

    #[test]
    fn ring_generates_mesh() {
        let mesh = Shape::ring(1.0, 0.5, ShapeDetail::Medium).build();
        assert!(mesh.count_vertices() > 0, "Ring must have vertices");
        assert!(mesh.indices().is_some(), "Ring must have indices");
    }

    #[test]
    fn ring_equal_radii_empty() {
        // Ring with equal radii should produce empty mesh
        let mesh = Shape::ring(1.0, 1.0, ShapeDetail::Medium).build();
        assert_eq!(
            mesh.count_vertices(),
            0,
            "Equal radius ring should be empty"
        );
    }

    #[test]
    fn ring_hash_stability() {
        let r1 = Shape::ring(1.0, 0.5, ShapeDetail::Medium);
        let r2 = Shape::ring(1.0, 0.5, ShapeDetail::Medium);
        assert_eq!(r1, r2, "Identical rings should be equal");
    }

    #[test]
    fn wedge_generates_mesh() {
        let mesh = Shape::wedge(1.0, PI / 2.0, ShapeDetail::Medium).build();
        assert!(mesh.count_vertices() > 0, "Wedge must have vertices");
        assert!(mesh.indices().is_some(), "Wedge must have indices");
    }

    #[test]
    fn wedge_full_circle() {
        // 360 degwedge should be like a disc
        let mesh = Shape::wedge(1.0, 2.0 * PI, ShapeDetail::Medium).build();
        assert!(mesh.count_vertices() > 0, "Full wedge must have vertices");
    }

    #[test]
    fn wedge_zero_angle_empty() {
        let mesh = Shape::wedge(1.0, 0.0, ShapeDetail::Medium).build();
        assert_eq!(mesh.count_vertices(), 0, "Zero angle wedge should be empty");
    }

    #[test]
    fn wedge_hash_stability() {
        let w1 = Shape::wedge(1.0, PI / 3.0, ShapeDetail::Medium);
        let w2 = Shape::wedge(1.0, PI / 3.0, ShapeDetail::Medium);
        assert_eq!(w1, w2, "Identical wedges should be equal");
    }

    #[test]
    fn tapered_cylinder_generates_mesh() {
        let mesh = Shape::tapered_cylinder(0.5, 1.0, 2.0, ShapeDetail::Medium).build();
        assert!(
            mesh.count_vertices() > 0,
            "Tapered cylinder must have vertices"
        );
        assert!(
            mesh.indices().is_some(),
            "Tapered cylinder must have indices"
        );
    }

    #[test]
    fn tapered_cylinder_equal_radii_like_cylinder() {
        // Equal radii should produce a regular cylinder
        let mesh = Shape::tapered_cylinder(1.0, 1.0, 2.0, ShapeDetail::Medium).build();
        assert!(
            mesh.count_vertices() > 0,
            "Equal radius tapered cylinder must have vertices"
        );
    }

    #[test]
    fn tapered_cylinder_hash_stability() {
        let t1 = Shape::tapered_cylinder(0.5, 1.0, 2.0, ShapeDetail::Medium);
        let t2 = Shape::tapered_cylinder(0.5, 1.0, 2.0, ShapeDetail::Medium);
        assert_eq!(t1, t2, "Identical tapered cylinders should be equal");
    }
}

#[cfg(test)]
mod extended_shape_tests {
    use super::*;
    use crate::plugin::MeshCache;
    use bevy::asset::Assets;
    use bevy::mesh::{Indices, VertexAttributeValues};
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    const D: ShapeDetail = ShapeDetail::Medium;

    fn hash_of(shape: &Shape) -> u64 {
        let mut h = DefaultHasher::new();
        shape.hash(&mut h);
        h.finish()
    }

    fn door_wall() -> Shape {
        Shape::extruded_polygon_with_holes(
            Shape::rounded_rectangle(Vec2::new(4.0, 3.0), 0.2, 3),
            vec![Shape::arch(1.0, 1.6, 8)],
            0.5,
            0.3,
            D,
        )
    }

    fn tower() -> Shape {
        Shape::compound(vec![
            ShapePart::new(Shape::lathe(
                vec![Vec2::new(1.0, 0.0), Vec2::new(0.8, 2.0), Vec2::new(1.1, 2.5)],
                16,
                D,
            )),
            ShapePart::new(Shape::cone(1.3, 1.0, D))
                .with_transform(Transform::from_xyz(0.0, 3.0, 0.0))
                .with_tint(Color::srgb(0.9, 0.3, 0.3)),
        ])
    }

    fn assert_pickable(mesh: &Mesh) {
        let Some(VertexAttributeValues::Float32x3(positions)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("missing positions");
        };
        assert!(!positions.is_empty());
        assert!(positions.iter().flatten().all(|v| v.is_finite()));
        // Panics when attribute lengths disagree.
        assert_eq!(mesh.count_vertices(), positions.len());
        assert!(mesh.contains_attribute(Mesh::ATTRIBUTE_NORMAL));
        assert!(mesh.contains_attribute(Mesh::ATTRIBUTE_UV_0));
        let indices: Vec<usize> = match mesh.indices().expect("indices") {
            Indices::U16(v) => v.iter().map(|&i| i as usize).collect(),
            Indices::U32(v) => v.iter().map(|&i| i as usize).collect(),
        };
        assert!(!indices.is_empty() && indices.len() % 3 == 0);
        assert!(indices.iter().all(|&i| i < positions.len()));
    }

    #[test]
    fn new_shapes_are_hash_and_eq() {
        assert_eq!(door_wall(), door_wall());
        assert_eq!(hash_of(&door_wall()), hash_of(&door_wall()));
        assert_eq!(tower(), tower());
        assert_eq!(hash_of(&tower()), hash_of(&tower()));
        let moved = Shape::compound(vec![ShapePart::new(Shape::sphere(1.0, D))
            .with_transform(Transform::from_xyz(1.0, 0.0, 0.0))]);
        let other = Shape::compound(vec![ShapePart::new(Shape::sphere(1.0, D))
            .with_transform(Transform::from_xyz(2.0, 0.0, 0.0))]);
        assert_ne!(moved, other);
        let lathe = Shape::lathe(vec![Vec2::X, Vec2::Y], 8, D);
        assert_eq!(lathe, Shape::lathe(vec![Vec2::X, Vec2::Y], 8, D));
        assert_ne!(lathe, Shape::lathe(vec![Vec2::X, Vec2::Y], 9, D));
    }

    #[test]
    fn compounds_and_extrusions_share_one_cached_handle() {
        let mut cache = MeshCache::default();
        let mut meshes = Assets::<Mesh>::default();
        let a = cache.get_or_build_handle(&tower(), &mut meshes);
        let b = cache.get_or_build_handle(&tower(), &mut meshes);
        let c = cache.get_or_build_handle(&door_wall(), &mut meshes);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(meshes.len(), 2);
    }

    #[test]
    fn every_shape_builds_a_pickable_mesh() {
        let shapes = [
            Shape::sphere(1.0, D),
            Shape::cuboid(Vec3::ONE, 0.2, D),
            Shape::cylinder(1.0, 2.0, D),
            Shape::cone(1.0, 2.0, D),
            Shape::capsule(0.5, 2.0, D),
            Shape::torus(1.0, 0.2, 0.0, std::f32::consts::TAU, D),
            Shape::pyramid(4, 1.0, 1.0, D),
            Shape::disc(1.0, D),
            Shape::rectangle(Vec2::ONE, D),
            Shape::star(5, 1.0, 0.5, D),
            Shape::ring(1.0, 0.5, D),
            Shape::arbitrary_polygon(Shape::blob(1.0, 0.3, 3), D),
            Shape::extruded_polygon(Shape::arch(1.0, 2.0, 8), 0.3, 0.5, D),
            Shape::extruded_polygon_colored(
                Shape::blob(1.0, 0.3, 9),
                vec![[1.0, 0.8, 0.8, 1.0]; crate::outline::BLOB_VERTICES],
                0.2,
                1.0,
                D,
            ),
            Shape::arbitrary_polygon_with_holes(
                Shape::rounded_rectangle(Vec2::new(4.0, 3.0), 0.3, 3),
                vec![Shape::arch(1.0, 1.5, 6)],
                D,
            ),
            door_wall(),
            tower(),
            Shape::lathe(vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.5), Vec2::new(0.0, 1.0)], 12, D),
            Shape::polyline_2d(vec![Vec2::ZERO, Vec2::new(1.0, 0.5), Vec2::new(2.0, 0.0)], 0.1, false, 8),
            Shape::polyline_2d(Shape::rounded_rectangle(Vec2::splat(2.0), 0.3, 3), 0.1, true, 8),
            Shape::bezier_2d(vec![Vec2::ZERO, Vec2::new(1.0, 2.0), Vec2::new(2.0, 0.0)], 0.1, 16),
        ];
        for shape in &shapes {
            assert_pickable(&shape.build());
            assert_pickable(&shape.build_2d());
        }
    }

    #[test]
    fn build_2d_is_flat_and_faces_plus_z() {
        for shape in [
            Shape::rectangle(Vec2::new(2.0, 1.0), D),
            Shape::disc(1.0, D),
            Shape::arbitrary_polygon_with_holes(
                Shape::rounded_rectangle(Vec2::new(4.0, 3.0), 0.3, 3),
                vec![Shape::arch(1.0, 1.5, 6)],
                D,
            ),
            door_wall(),
        ] {
            let mesh = shape.build_2d();
            let Some(VertexAttributeValues::Float32x3(positions)) =
                mesh.attribute(Mesh::ATTRIBUTE_POSITION)
            else {
                panic!("positions");
            };
            assert!(positions.iter().all(|p| p[2] == 0.0), "{shape:?}");
            let Some(VertexAttributeValues::Float32x3(normals)) =
                mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
            else {
                panic!("normals");
            };
            assert!(normals.iter().all(|n| n[2] > 0.99));
            let Some(VertexAttributeValues::Float32x2(uvs)) = mesh.attribute(Mesh::ATTRIBUTE_UV_0)
            else {
                panic!("uvs");
            };
            assert!(uvs.iter().flatten().all(|v| (-1e-4..=1.0001).contains(v)));
        }
    }

    #[test]
    fn extruded_polygon_adds_thickness_and_door_wall_has_a_hole() {
        let flat = Shape::arbitrary_polygon_with_holes(
            Shape::rounded_rectangle(Vec2::new(4.0, 3.0), 0.0, 1),
            vec![Shape::arch(1.0, 1.0, 8)],
            D,
        )
        .build();
        let solid = Shape::arbitrary_polygon(Shape::rounded_rectangle(Vec2::new(4.0, 3.0), 0.0, 1), D)
            .build();
        // The hole costs extra vertices and the opening is not covered by triangles.
        assert!(flat.count_vertices() > solid.count_vertices());

        let wall = door_wall().build();
        let Some(VertexAttributeValues::Float32x3(p)) = wall.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!();
        };
        let (min_z, max_z) = p.iter().fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(v[2]), b.max(v[2])));
        assert!((min_z + 0.25).abs() < 1e-5 && (max_z - 0.25).abs() < 1e-5);
    }

    #[test]
    fn extruded_shapes_lerp_between_matching_outlines() {
        let a = Shape::extruded_polygon(Shape::rounded_rectangle(Vec2::splat(2.0), 0.3, 2), 0.2, 0.0, D);
        let b = Shape::extruded_polygon(Shape::rounded_rectangle(Vec2::splat(4.0), 0.3, 2), 0.6, 1.0, D);
        let mid = a.lerp(&b, 0.5);
        assert_eq!(
            mid,
            Shape::extruded_polygon(Shape::rounded_rectangle(Vec2::splat(3.0), 0.3, 2), 0.4, 0.5, D)
        );
        // Outlines with different vertex counts now morph between the two.
        let c = Shape::extruded_polygon(Shape::arch(1.0, 1.0, 4), 0.2, 0.0, D);
        let morphed = a.lerp(&c, 0.25);
        assert_ne!(morphed, a);
        assert_ne!(morphed, c);
        // Outlines that cannot be paired up (fewer than three vertices) snap.
        let degenerate = Shape::extruded_polygon(vec![Vec2::ZERO, Vec2::X], 0.2, 0.0, D);
        assert_eq!(a.lerp(&degenerate, 0.25), a);
        assert_eq!(a.lerp(&degenerate, 0.75), degenerate);
    }

    #[test]
    fn compound_merges_parts_into_one_mesh_with_tints() {
        let mesh = tower().build();
        assert_pickable(&mesh);
        let a = Shape::lathe(
            vec![Vec2::new(1.0, 0.0), Vec2::new(0.8, 2.0), Vec2::new(1.1, 2.5)],
            16,
            D,
        )
        .build();
        let b = Shape::cone(1.3, 1.0, D).build();
        assert_eq!(mesh.count_vertices(), a.count_vertices() + b.count_vertices());
        let Some(VertexAttributeValues::Float32x4(colors)) = mesh.attribute(Mesh::ATTRIBUTE_COLOR)
        else {
            panic!("tinted compound needs vertex colours");
        };
        assert_eq!(colors[0], [1.0; 4]);
        assert!(colors[a.count_vertices()][1] < 0.5);
    }

    #[test]
    fn lathe_with_zero_segments_uses_detail() {
        let a = Shape::lathe(vec![Vec2::X, Vec2::new(1.0, 1.0)], 0, ShapeDetail::Low).build();
        let b = Shape::lathe(vec![Vec2::X, Vec2::new(1.0, 1.0)], ShapeDetail::Low.segments(), D).build();
        assert_eq!(a.count_vertices(), b.count_vertices());
    }
}

#[cfg(test)]
mod morph_and_bounds_tests {
    use super::*;

    const D: ShapeDetail = ShapeDetail::Medium;

    fn square(half: f32) -> Vec<Vec2> {
        vec![
            Vec2::new(-half, -half),
            Vec2::new(half, -half),
            Vec2::new(half, half),
            Vec2::new(-half, half),
        ]
    }

    fn triangle() -> Vec<Vec2> {
        vec![Vec2::new(0.0, -1.0), Vec2::new(1.0, 1.0), Vec2::new(-1.0, 1.0)]
    }

    fn outline(shape: &Shape) -> (&Vec<Vec2>, &Option<Vec<[f32; 4]>>) {
        match &shape.kind {
            ShapeKind::ArbitraryPolygon { vertices, colors } => (vertices, colors),
            ShapeKind::ExtrudedPolygon { outer, colors, .. } => (outer, colors),
            other => panic!("not a polygon: {other:?}"),
        }
    }

    #[test]
    fn bounds_of_a_cuboid_are_exact() {
        let b = Shape::cuboid(Vec3::new(2.0, 4.0, 6.0), 0.0, D).bounds();
        assert!((b.min - Vec3::new(-1.0, -2.0, -3.0)).abs().max_element() < 1e-4);
        assert!((b.max - Vec3::new(1.0, 2.0, 3.0)).abs().max_element() < 1e-4);
        assert!((b.size() - Vec3::new(2.0, 4.0, 6.0)).abs().max_element() < 1e-4);
        assert!(b.center().abs().max_element() < 1e-4);
        assert!((b.half_extents() - Vec3::new(1.0, 2.0, 3.0)).abs().max_element() < 1e-4);
    }

    #[test]
    fn bounds_follow_the_generated_geometry() {
        let sphere = Shape::sphere(2.0, ShapeDetail::High).bounds();
        assert!(sphere.size().min_element() > 3.8 && sphere.size().max_element() <= 4.0001);
        let rounded = Shape::cuboid(Vec3::splat(2.0), 0.5, D).bounds();
        assert!(rounded.size().max_element() <= 2.0001);
        let moved = Shape::compound(vec![
            ShapePart::new(Shape::cuboid(Vec3::ONE, 0.0, D)),
            ShapePart::new(Shape::cuboid(Vec3::ONE, 0.0, D))
                .with_transform(Transform::from_xyz(3.0, 0.0, 0.0)),
        ])
        .bounds();
        assert!((moved.min.x + 0.5).abs() < 1e-4 && (moved.max.x - 3.5).abs() < 1e-4);
    }

    #[test]
    fn bounds_2d_gives_the_flat_footprint() {
        let b = Shape::rectangle(Vec2::new(4.0, 2.0), D).bounds_2d();
        assert!((b.rect_xy().size() - Vec2::new(4.0, 2.0)).abs().max_element() < 1e-4);
        assert!(b.size().z.abs() < 1e-4);
        let empty = Shape::arbitrary_polygon(vec![Vec2::ZERO, Vec2::X], D).bounds();
        assert_eq!(empty, ShapeBounds::default());
    }

    #[test]
    fn polygons_with_different_counts_morph() {
        let a = Shape::arbitrary_polygon(triangle(), D);
        let b = Shape::arbitrary_polygon(square(1.0), D);
        assert_eq!(a.lerp(&b, 0.0), a);
        assert_eq!(a.lerp(&b, 1.0), b);
        let mid = a.lerp(&b, 0.5);
        let (verts, colors) = outline(&mid);
        assert_eq!(verts.len(), 4);
        assert!(colors.is_none());
        assert_ne!(mid, a);
        assert_ne!(mid, b);
        let low = a.lerp(&b, 0.25);
        assert_eq!(outline(&low).0.len(), 4);
        assert!(!mid.build().attribute(Mesh::ATTRIBUTE_POSITION).unwrap().is_empty());
    }

    #[test]
    fn subdivided_outline_morphs_to_the_same_outline() {
        let sq = square(1.0);
        let dense: Vec<Vec2> = (0..4)
            .flat_map(|i| [sq[i], (sq[i] + sq[(i + 1) % 4]) * 0.5])
            .collect();
        let a = Shape::arbitrary_polygon(sq, D);
        let b = Shape::arbitrary_polygon(dense, D);
        let mid = a.lerp(&b, 0.5);
        let verts = outline(&mid).0.clone();
        assert_eq!(verts.len(), 8);
        assert!((signed_area(&verts) - 4.0).abs() < 1e-4);
        assert!(verts.iter().all(|p| p.abs().max_element() > 0.999));
    }

    #[test]
    fn opposite_winding_and_start_do_not_twist_the_morph() {
        let a = triangle();
        let mut b = square(1.0);
        b.reverse();
        b.rotate_left(2);
        let (verts, _) = morph_rings(&a, None, &b, None, 0.5).unwrap();
        assert_eq!(verts.len(), 4);
        let area_a = signed_area(&a);
        let area_b = signed_area(&square(1.0));
        let area = signed_area(&verts);
        assert!(area.signum() == area_a.signum());
        assert!(area > area_a.abs().min(area_b.abs()) * 0.8);
        assert!(area < area_a.abs().max(area_b.abs()) * 1.2);
    }

    #[test]
    fn colours_blend_and_missing_colours_count_as_white() {
        let red = [1.0, 0.0, 0.0, 1.0];
        let a = Shape::arbitrary_polygon_colored(triangle(), vec![red; 3], D);
        let b = Shape::arbitrary_polygon(square(1.0), D);
        let mid = a.lerp(&b, 0.5);
        let colors = outline(&mid).1.clone().expect("colours survive the morph");
        assert_eq!(colors.len(), 4);
        for c in colors {
            assert!((c[0] - 1.0).abs() < 1e-5 && (c[1] - 0.5).abs() < 1e-5);
        }
    }

    #[test]
    fn extruded_polygons_and_holes_morph_across_counts() {
        let hole_a = vec![Shape::blob(0.2, 0.0, 1)];
        let hole_b = vec![square(0.2)];
        let a = Shape::extruded_polygon_with_holes(triangle(), hole_a.clone(), 0.2, 0.0, D);
        let b = Shape::extruded_polygon_with_holes(square(1.0), hole_b, 0.4, 0.0, D);
        let mid = a.lerp(&b, 0.5);
        match &mid.kind {
            ShapeKind::ExtrudedPolygon {
                outer,
                holes,
                depth,
                ..
            } => {
                assert_eq!(outer.len(), 4);
                assert_eq!(holes.len(), 1);
                assert!(holes[0].len() >= 4);
                assert!((depth - 0.3).abs() < 1e-5);
            }
            other => panic!("unexpected {other:?}"),
        }
        let no_hole = Shape::extruded_polygon(square(1.0), 0.4, 0.0, D);
        assert_eq!(a.lerp(&no_hole, 0.4), a);
        assert_eq!(a.lerp(&no_hole, 0.6), no_hole);
    }

    #[test]
    fn degenerate_polygons_and_open_lists_still_snap() {
        let a = Shape::arbitrary_polygon(vec![Vec2::ZERO, Vec2::X], D);
        let b = Shape::arbitrary_polygon(square(1.0), D);
        assert_eq!(a.lerp(&b, 0.4), a);
        assert_eq!(a.lerp(&b, 0.6), b);
        let p = Shape::polyline3d(vec![Vec3::ZERO, Vec3::X], 0.1, false, 8, D);
        let q = Shape::polyline3d(vec![Vec3::ZERO, Vec3::X, Vec3::Y], 0.1, false, 8, D);
        assert_eq!(p.lerp(&q, 0.4), p);
    }
}

#[cfg(test)]
mod rounded_solid_tests {
    use super::*;

    const D: ShapeDetail = ShapeDetail::Medium;

    fn positions(mesh: &Mesh) -> Vec<Vec3> {
        match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            Some(bevy::render::mesh::VertexAttributeValues::Float32x3(p)) => {
                p.iter().map(|p| Vec3::from_array(*p)).collect()
            }
            _ => panic!("no positions"),
        }
    }

    fn normals(mesh: &Mesh) -> Vec<Vec3> {
        match mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
            Some(bevy::render::mesh::VertexAttributeValues::Float32x3(p)) => {
                p.iter().map(|p| Vec3::from_array(*p)).collect()
            }
            _ => panic!("no normals"),
        }
    }

    fn indices(mesh: &Mesh) -> Vec<u32> {
        mesh.indices().unwrap().iter().map(|i| i as u32).collect()
    }

    /// Triangles whose winding disagrees with their vertex normals.
    fn inward_triangles(mesh: &Mesh) -> (usize, usize) {
        let (p, n, idx) = (positions(mesh), normals(mesh), indices(mesh));
        let mut bad = 0;
        for t in idx.chunks(3) {
            let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
            let face = (p[b] - p[a]).cross(p[c] - p[a]);
            if face.length_squared() < 1e-14 {
                continue;
            }
            if face.dot(n[a] + n[b] + n[c]) < 0.0 {
                bad += 1;
            }
        }
        (bad, idx.len() / 3)
    }

    fn assert_sound(mesh: &Mesh, what: &str) {
        let p = positions(mesh);
        assert!(!p.is_empty(), "{what}: empty");
        assert!(p.iter().all(|v| v.is_finite()), "{what}: non-finite");
        assert!(normals(mesh).iter().all(|v| v.is_finite()), "{what}: bad normals");
        let (bad, total) = inward_triangles(mesh);
        assert!(total > 0, "{what}: no triangles");
        assert_eq!(bad, 0, "{what}: {bad}/{total} triangles face inwards");
    }

    #[test]
    fn zero_roundness_is_the_sharp_shape() {
        assert_eq!(Shape::cone(1.0, 2.0, D), Shape::rounded_cone(1.0, 2.0, 0.0, D));
        assert_eq!(
            Shape::pyramid(4, 1.0, 2.0, D),
            Shape::rounded_pyramid(4, 1.0, 2.0, 0.0, D)
        );
        let sharp = Shape::cone(1.0, 2.0, D).build();
        assert_eq!(
            positions(&sharp),
            positions(&Shape::rounded_cone(1.0, 2.0, 0.0, D).build())
        );
    }

    #[test]
    fn roundness_changes_identity_and_blends() {
        let a = Shape::rounded_cone(1.0, 2.0, 0.0, D);
        let b = Shape::rounded_cone(1.0, 2.0, 1.0, D);
        assert_ne!(a, b);
        assert_eq!(a.lerp(&b, 0.5), Shape::rounded_cone(1.0, 2.0, 0.5, D));
        let p = Shape::rounded_pyramid(4, 1.0, 2.0, 0.2, D);
        let q = Shape::rounded_pyramid(4, 1.0, 2.0, 0.6, D);
        assert_eq!(p.lerp(&q, 0.5), Shape::rounded_pyramid(4, 1.0, 2.0, 0.4, D));
    }

    #[test]
    fn rounded_solids_face_outwards_for_every_roundness() {
        for r in [0.1, 0.25, 0.5, 0.75, 1.0] {
            assert_sound(&Shape::rounded_cone(1.0, 2.0, r, D).build(), &format!("cone {r}"));
            for sides in [3, 4, 6] {
                assert_sound(
                    &Shape::rounded_pyramid(sides, 1.0, 1.5, r, D).build(),
                    &format!("pyramid {sides} {r}"),
                );
                assert_sound(
                    &Shape::regular_prism(sides, 1.0, 1.5, r, D).build(),
                    &format!("prism {sides} {r}"),
                );
            }
        }
    }

    #[test]
    fn rounded_solids_stay_inside_the_sharp_shape() {
        let sharp = Shape::cone(1.0, 2.0, D).bounds();
        let soft = Shape::rounded_cone(1.0, 2.0, 0.6, D).bounds();
        assert!(soft.max.y < sharp.max.y - 0.05, "apex should be blunted");
        assert!(soft.min.y >= sharp.min.y - 1e-4);
        assert!(soft.max.x <= sharp.max.x + 1e-4);
        let sharp = Shape::pyramid(4, 1.0, 2.0, D).bounds();
        let soft = Shape::rounded_pyramid(4, 1.0, 2.0, 0.6, D).bounds();
        assert!(soft.max.y < sharp.max.y - 0.05);
        assert!(soft.min.y >= sharp.min.y - 1e-4);
        assert!(soft.max.x <= sharp.max.x + 1e-4 && soft.max.z <= sharp.max.z + 1e-4);
        let sharp = Shape::regular_prism(6, 1.0, 2.0, 0.0, D).bounds();
        let soft = Shape::regular_prism(6, 1.0, 2.0, 0.5, D).bounds();
        assert!((soft.max.y - sharp.max.y).abs() < 1e-4, "height is kept");
        assert!(soft.max.x <= sharp.max.x + 1e-4);
    }

    #[test]
    fn prism_roundness_is_no_longer_ignored() {
        let sharp = Shape::regular_prism(4, 1.0, 2.0, 0.0, D).build();
        let soft = Shape::regular_prism(4, 1.0, 2.0, 0.5, D).build();
        assert_ne!(positions(&sharp), positions(&soft));
        // The sharp prism has corners on its rim; the rounded one pulls them in.
        let reach = |m: &Mesh| positions(m).iter().map(|p| p.length()).fold(0.0, f32::max);
        assert!(reach(&soft) < reach(&sharp) - 0.05);
    }

    #[test]
    fn full_roundness_pyramid_is_a_blob_without_degenerate_output() {
        let mesh = Shape::rounded_pyramid(5, 1.0, 1.0, 1.0, D).build();
        assert_sound(&mesh, "full pyramid");
        for n in normals(&mesh) {
            assert!((n.length() - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn rounded_solids_are_pickable() {
        for shape in [
            Shape::rounded_cone(1.0, 2.0, 0.5, D),
            Shape::rounded_pyramid(4, 1.0, 2.0, 0.5, D),
            Shape::regular_prism(6, 1.0, 2.0, 0.5, D),
        ] {
            let mesh = shape.build();
            let n = positions(&mesh).len() as u32;
            assert!(indices(&mesh).iter().all(|i| *i < n));
            assert!(mesh.attribute(Mesh::ATTRIBUTE_UV_0).is_some());
        }
    }

    #[test]
    fn is_round_matches_the_outline_kind() {
        assert!(Shape::sphere(1.0, D).is_round());
        assert!(Shape::ellipsoid(Vec3::new(1.0, 2.0, 3.0), D).is_round());
        assert!(Shape::disc(1.0, D).is_round());
        assert!(Shape::ring(1.0, 0.5, D).is_round());
        assert!(!Shape::cuboid(Vec3::ONE, 0.0, D).is_round());
        assert!(!Shape::rectangle(Vec2::ONE, D).is_round());
        assert!(!Shape::rounded_cone(1.0, 2.0, 0.5, D).is_round());
    }

    #[test]
    fn bounds_convert_to_a_corner_pair() {
        let (lo, hi): (Vec3, Vec3) = Shape::cuboid(Vec3::new(2.0, 4.0, 6.0), 0.0, D).bounds().into();
        assert!((hi - lo - Vec3::new(2.0, 4.0, 6.0)).abs().max_element() < 1e-4);
    }
}

#[cfg(test)]
mod from_mesh_tests {
    use super::*;
    use bevy::mesh::VertexAttributeValues;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    const D: ShapeDetail = ShapeDetail::Medium;

    fn hash_of(s: &Shape) -> u64 {
        let mut h = DefaultHasher::new();
        s.hash(&mut h);
        h.finish()
    }

    fn positions(mesh: &Mesh) -> Vec<[f32; 3]> {
        match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            Some(VertexAttributeValues::Float32x3(p)) => p.clone(),
            _ => Vec::new(),
        }
    }

    fn normals(mesh: &Mesh) -> Vec<[f32; 3]> {
        match mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
            Some(VertexAttributeValues::Float32x3(n)) => n.clone(),
            _ => Vec::new(),
        }
    }

    fn mesh_shape(source: &Shape, key: u64) -> Shape {
        Shape::from_mesh(Arc::new(source.build()), key)
    }

    #[test]
    fn equality_and_hash_follow_the_key() {
        let a = mesh_shape(&Shape::sphere(1.0, D), 7);
        let same_key = mesh_shape(&Shape::cuboid(Vec3::ONE, 0.0, D), 7);
        let other_key = mesh_shape(&Shape::sphere(1.0, D), 8);
        assert_eq!(a, same_key);
        assert_eq!(hash_of(&a), hash_of(&same_key));
        assert_ne!(a, other_key);
        assert_ne!(a, Shape::sphere(1.0, D));
    }

    #[test]
    fn bounds_match_the_positions_without_building() {
        let mesh = Shape::cuboid(Vec3::new(2.0, 4.0, 6.0), 0.0, D).build();
        let expected = ShapeBounds::from_points(positions(&mesh).into_iter().map(Vec3::from_array));
        let shape = Shape::from_mesh(Arc::new(mesh), 1);
        assert_eq!(shape.bounds(), expected);
        assert_eq!(shape.bounds_2d(), expected);
        assert!((shape.bounds().size() - Vec3::new(2.0, 4.0, 6.0)).length() < 1e-5);
    }

    #[test]
    fn is_round_tells_a_sphere_from_a_box() {
        assert!(mesh_shape(&Shape::sphere(1.5, ShapeDetail::High), 1).is_round());
        assert!(!mesh_shape(&Shape::cuboid(Vec3::ONE, 0.0, D), 2).is_round());
    }

    #[test]
    fn build_returns_the_mesh() {
        let source = Shape::sphere(1.0, D).build();
        let shape = Shape::from_mesh(Arc::new(source.clone()), 3);
        assert_eq!(positions(&shape.build()), positions(&source));
        assert_eq!(positions(&shape.build_2d()), positions(&source));
    }

    #[test]
    fn lerp_takes_the_nearer_shape() {
        let a = mesh_shape(&Shape::sphere(1.0, D), 1);
        let b = mesh_shape(&Shape::sphere(2.0, D), 2);
        assert_eq!(a.lerp(&b, 0.3), a);
        assert_eq!(a.lerp(&b, 0.7), b);
    }

    #[test]
    fn works_inside_a_compound() {
        let part = mesh_shape(&Shape::cuboid(Vec3::ONE, 0.0, D), 1);
        let single = part.build();
        let compound = Shape::compound(vec![
            ShapePart::new(part.clone()),
            ShapePart::new(part)
                .with_transform(Transform::from_xyz(5.0, 0.0, 0.0)),
        ]);
        assert_eq!(
            positions(&compound.build()).len(),
            positions(&single).len() * 2
        );
        assert!((compound.bounds().max.x - 5.5).abs() < 1e-5);
    }

    #[test]
    fn mesh_without_normals_or_uvs_still_compounds() {
        let mut bare = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        bare.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        );
        let compound = Shape::compound(vec![ShapePart::new(Shape::from_mesh(Arc::new(bare), 9))]);
        assert_eq!(positions(&compound.build()).len(), 3);
    }

    fn field(sx: u32, sy: u32, f: impl Fn(u32, u32) -> f32) -> Arc<[f32]> {
        (0..sy)
            .flat_map(|y| (0..sx).map(move |x| (x, y)))
            .map(|(x, y)| f(x, y))
            .collect::<Vec<_>>()
            .into()
    }

    fn heightfield(sx: u32, sy: u32, f: impl Fn(u32, u32) -> f32) -> Shape {
        Shape::heightfield(field(sx, sy, f), UVec2::new(sx, sy), Vec2::new(8.0, 6.0))
    }

    fn with_options(sx: u32, sy: u32, f: impl Fn(u32, u32) -> f32, o: HeightfieldOptions) -> Shape {
        Shape::heightfield_with_options(field(sx, sy, f), UVec2::new(sx, sy), Vec2::new(8.0, 6.0), o)
    }

    fn tri_count(mesh: &Mesh) -> usize {
        mesh.indices().map_or(0, |i| i.len() / 3)
    }

    #[test]
    fn a_hole_has_no_triangles_and_no_skirt_under_it() {
        // 3x3 samples, 2x2 cells; the cell at (0, 0) is a hole (a chunk corner).
        let holes = vec![true, false, false, false];
        let full = heightfield(3, 3, |_, _| 0.0).build();
        let holed = with_options(3, 3, |_, _| 0.0, HeightfieldOptions::default().with_holes(holes)).build();
        // One grid cell (2 triangles) and its two edge skirt segments (4 triangles) are gone.
        assert_eq!(tri_count(&full) - tri_count(&holed), 2 + 4);
        // The grid vertex (0, 0) touches only the hole, so it is not meshed.
        let p: Vec<Vec3> = positions(&holed).into_iter().map(Vec3::from_array).collect();
        assert!(!p.iter().any(|p| (p.x + 4.0).abs() < 1e-5 && (p.z + 3.0).abs() < 1e-5 && p.y.abs() < 1e-5));
    }

    #[test]
    fn a_fully_holed_chunk_is_empty_without_a_warning_shape() {
        let all = vec![true; 4];
        let shape = with_options(3, 3, |_, _| 1.0, HeightfieldOptions::default().with_holes(all));
        assert_eq!(tri_count(&shape.build()), 0);
        assert_eq!(shape.bounds(), ShapeBounds::default());
    }

    #[test]
    fn bounds_ignore_what_holes_remove() {
        // A tall spike at (0, 0) that only the hole at cell (0, 0) touches.
        let f = |x: u32, y: u32| if (x, y) == (0, 0) { 50.0 } else { 1.0 };
        let holes = vec![true, false, false, false];
        let shape = with_options(3, 3, f, HeightfieldOptions::default().with_holes(holes));
        assert_eq!(shape.surface_bounds().max.y, 1.0);
        let built = ShapeBounds::from_points(positions(&shape.build()).into_iter().map(Vec3::from_array));
        assert!((shape.bounds().min - built.min).length() < 1e-5);
        assert!((shape.bounds().max - built.max).length() < 1e-5);
    }

    #[test]
    fn closed_form_bounds_match_the_built_mesh_at_every_detail() {
        let details = [
            ShapeDetail::Minimal,
            ShapeDetail::Low,
            ShapeDetail::Medium,
            ShapeDetail::High,
            ShapeDetail::Extreme,
        ];
        for d in details {
            let shapes = [
                ("sphere", Shape::sphere(1.5, d)),
                ("ellipsoid", Shape::ellipsoid(Vec3::new(2.0, 3.0, 5.0), d)),
                ("cuboid", Shape::cuboid(Vec3::new(2.0, 4.0, 6.0), 0.0, d)),
                ("rounded cuboid", Shape::cuboid(Vec3::new(2.0, 4.0, 6.0), 0.5, d)),
                ("cylinder", Shape::cylinder(1.5, 3.0, d)),
                ("cone", Shape::cone(1.5, 3.0, d)),
                ("capsule", Shape::capsule(0.75, 3.0, d)),
                ("rectangle", Shape::rectangle(Vec2::new(3.0, 2.0), d)),
                ("disc", Shape::disc(1.5, d)),
                ("ellipse", Shape::ellipse(Vec2::new(2.0, 1.0), d)),
            ];
            for (name, shape) in shapes {
                let closed = closed_form_bounds(&shape.kind)
                    .unwrap_or_else(|| panic!("{name} has no closed form"));
                let measured = mesh_bounds(&shape.build());
                assert!(
                    (closed.min - measured.min).length() < 1e-4
                        && (closed.max - measured.max).length() < 1e-4,
                    "{name} at {d:?}: closed form {closed:?}, mesh {measured:?}"
                );
                assert_eq!(shape.bounds(), closed);
                assert_eq!(shape.bounds_2d(), closed);
            }
        }
    }

    #[test]
    fn closed_form_bounds_do_not_build_mesh() {
        // Closed-form shapes return bounds without building a mesh.
        let sphere = Shape::sphere(2.0, D);
        let closed = sphere.bounds();
        assert_eq!(closed.min, Vec3::splat(-2.0));
        assert_eq!(closed.max, Vec3::splat(2.0));
        // A kind without a closed form builds the mesh to measure bounds.
        let star = Shape::star(5, 2.0, 1.0, D);
        let measured = star.bounds();
        assert!(measured.size().x > 0.0 && measured.size().y > 0.0);
    }

    #[test]
    fn surface_bounds_leave_out_the_skirts() {
        let shape = heightfield(3, 3, |x, y| (x + y) as f32);
        assert_eq!(shape.surface_bounds().min.y, 0.0);
        assert!(shape.bounds().min.y < 0.0);
        assert_eq!(shape.surface_bounds().max, shape.bounds().max);
        assert_eq!(Shape::sphere(1.0, D).surface_bounds(), Shape::sphere(1.0, D).bounds());
    }

    #[test]
    fn holes_and_border_are_part_of_the_identity() {
        let a = heightfield(3, 3, |_, _| 0.0);
        let holed = with_options(3, 3, |_, _| 0.0, HeightfieldOptions::default().with_holes(vec![true, false, false, false]));
        let other = with_options(3, 3, |_, _| 0.0, HeightfieldOptions::default().with_holes(vec![false, true, false, false]));
        assert_ne!(a, holed);
        assert_ne!(holed, other);
        assert_ne!(hash_of(&holed), hash_of(&other));
        let border = HeightfieldBorder::new(vec![0.0; 3], vec![0.0; 3], vec![0.0; 3], vec![0.0; 3]);
        let mut raised = border.clone();
        raised.east[1] = 1.0;
        let b1 = with_options(3, 3, |_, _| 0.0, HeightfieldOptions::default().with_border(border));
        let b2 = with_options(3, 3, |_, _| 0.0, HeightfieldOptions::default().with_border(raised));
        assert_ne!(b1, b2);
        assert_ne!(hash_of(&b1), hash_of(&b2));
    }

    #[test]
    fn invalid_hole_masks_and_borders_build_an_empty_mesh() {
        let bad_holes = with_options(3, 3, |_, _| 0.0, HeightfieldOptions::default().with_holes(vec![false; 3]));
        assert_eq!(tri_count(&bad_holes.build()), 0);
        let bad_border = HeightfieldBorder::new(vec![0.0; 3], vec![0.0; 2], vec![0.0; 3], vec![0.0; 3]);
        let shape = with_options(3, 3, |_, _| 0.0, HeightfieldOptions::default().with_border(bad_border));
        assert_eq!(tri_count(&shape.build()), 0);
    }

    /// A 9 x 9 grid (8 x 8 cells) over 8 x 8 world units, one unit per cell,
    /// with `holes` given as a predicate over the cell.
    fn edged(edge: HoleEdge, hole: impl Fn(u32, u32) -> bool) -> Shape {
        let holes: Vec<bool> = (0..8).flat_map(|y| (0..8).map(move |x| (x, y))).map(|(x, y)| hole(x, y)).collect();
        let heights: Arc<[f32]> = (0..9).flat_map(|y| (0..9).map(move |x| 0.1 * (x + y) as f32)).collect::<Vec<_>>().into();
        Shape::heightfield_with_options(
            heights,
            UVec2::new(9, 9),
            Vec2::new(8.0, 8.0),
            HeightfieldOptions::default().with_holes(holes).with_hole_edge(edge),
        )
    }

    fn triangles(mesh: &Mesh) -> Vec<[Vec3; 3]> {
        let p: Vec<Vec3> = positions(mesh).into_iter().map(Vec3::from_array).collect();
        let idx: Vec<u32> = mesh.indices().unwrap().iter().map(|i| i as u32).collect();
        idx.chunks(3).map(|t| [p[t[0] as usize], p[t[1] as usize], p[t[2] as usize]]).collect()
    }

    fn face_normal(t: &[Vec3; 3]) -> Vec3 {
        (t[1] - t[0]).cross(t[2] - t[0])
    }

    /// Area of the land seen from above, and the triangles that cover `at`.
    fn land_area(mesh: &Mesh) -> f32 {
        triangles(mesh).iter().map(|t| face_normal(t)).filter(|n| n.y > 1e-6).map(|n| n.y * 0.5).sum()
    }

    fn land_covers(mesh: &Mesh, at: Vec2) -> bool {
        triangles(mesh).iter().filter(|t| face_normal(t).y > 1e-6).any(|t| {
            let [a, b, c] = t.map(|v| Vec2::new(v.x, v.z));
            let side = |p: Vec2, q: Vec2| (q - p).perp_dot(at - p);
            let (s0, s1, s2) = (side(a, b), side(b, c), side(c, a));
            (s0 <= 0.0 && s1 <= 0.0 && s2 <= 0.0) || (s0 >= 0.0 && s1 >= 0.0 && s2 >= 0.0)
        })
    }

    fn a_blob(x: u32, y: u32) -> bool {
        // A round hole in the middle and another one against the west edge.
        let d = Vec2::new(x as f32 + 0.5 - 4.5, y as f32 + 0.5 - 3.5).length();
        d < 2.2 || (x == 0 && (2..5).contains(&y))
    }

    #[test]
    fn hole_edge_is_part_of_the_identity() {
        let cells = edged(HoleEdge::Cells, a_blob);
        let rounded = edged(HoleEdge::Rounded, a_blob);
        let smooth = edged(HoleEdge::Smooth, a_blob);
        assert_eq!(cells, edged(HoleEdge::default(), a_blob));
        assert_ne!(cells, rounded);
        assert_ne!(rounded, smooth);
        assert_ne!(hash_of(&cells), hash_of(&rounded));
        assert_ne!(hash_of(&rounded), hash_of(&smooth));
    }

    #[test]
    fn hole_edge_changes_nothing_without_holes() {
        let plain = heightfield(5, 5, |x, y| (x + y) as f32).build();
        let heights: Arc<[f32]> = (0..25).map(|i| ((i % 5) + (i / 5)) as f32).collect::<Vec<_>>().into();
        let smooth = Shape::heightfield_with_options(
            heights,
            UVec2::new(5, 5),
            Vec2::new(8.0, 6.0),
            HeightfieldOptions::default().with_hole_edge(HoleEdge::Smooth),
        )
        .build();
        assert_eq!(positions(&plain), positions(&smooth));
        assert_eq!(tri_count(&plain), tri_count(&smooth));
    }

    #[test]
    fn cut_outlines_face_up_and_skirts_only_hang_from_the_chunk_border() {
        for edge in [HoleEdge::Rounded, HoleEdge::Smooth] {
            let mesh = edged(edge, a_blob).build();
            for t in triangles(&mesh) {
                let n = face_normal(&t);
                assert!(n.y > -1e-5, "{edge:?}: a triangle faces down: {t:?}");
                if n.y.abs() <= 1e-5 && n.length() > 1e-6 {
                    // A skirt: all three corners on the same chunk border line, facing out.
                    let on = |f: &dyn Fn(Vec3) -> f32| t.iter().all(|v| (f(*v).abs() - 4.0).abs() < 1e-4);
                    assert!(on(&|v| v.x) || on(&|v| v.z), "{edge:?}: a wall under a hole: {t:?}");
                    let mid = (t[0] + t[1] + t[2]) / 3.0;
                    assert!(n.dot(Vec3::new(mid.x, 0.0, mid.z)) > 0.0, "{edge:?}: skirt faces inwards: {t:?}");
                }
            }
        }
    }

    #[test]
    fn the_outline_is_cut_through_the_cells() {
        let cells = land_area(&edged(HoleEdge::Cells, a_blob).build());
        let rounded = land_area(&edged(HoleEdge::Rounded, a_blob).build());
        let smooth = land_area(&edged(HoleEdge::Smooth, a_blob).build());
        // Chamfering the staircase gives land back; a whole-cell mask never does.
        assert!(rounded > cells + 0.5, "cells {cells}, rounded {rounded}");
        assert!(smooth > cells + 0.5, "cells {cells}, smooth {smooth}");
        // No level makes more land than the chunk has.
        assert!(smooth <= 64.0 * 1.2 && rounded <= 64.0 * 1.2);
    }

    #[test]
    fn no_level_removes_a_hole_or_an_island() {
        for edge in [HoleEdge::Cells, HoleEdge::Rounded, HoleEdge::Smooth] {
            let centre = Vec2::new(0.5, 0.5);
            let hole = edged(edge, |x, y| (x, y) == (4, 4)).build();
            assert!(!land_covers(&hole, centre), "{edge:?}: a one-cell hole closed up");
            assert!(land_covers(&hole, Vec2::new(-2.5, -2.5)));
            let island = edged(edge, |x, y| (x, y) != (4, 4)).build();
            assert!(land_covers(&island, centre), "{edge:?}: a one-cell island vanished");
            assert!(!land_covers(&island, Vec2::new(-2.5, -2.5)));
        }
    }

    #[test]
    fn cut_bounds_contain_the_built_mesh() {
        for edge in [HoleEdge::Rounded, HoleEdge::Smooth] {
            let shape = edged(edge, a_blob);
            let built = ShapeBounds::from_points(positions(&shape.build()).into_iter().map(Vec3::from_array));
            let b = shape.bounds();
            assert!(
                built.min.cmpge(b.min - Vec3::splat(1e-4)).all() && built.max.cmple(b.max + Vec3::splat(1e-4)).all(),
                "{edge:?}: bounds {b:?}, mesh {built:?}"
            );
        }
    }

    #[test]
    fn neighbouring_chunks_share_the_normals_of_their_edge() {
        // One continuous ridge z = |x - 2| sampled over 5 columns, cut into two
        // chunks of 3 columns that share column 2 (the crest).
        let h = |x: i32| ((x - 2).abs()) as f32;
        let chunk = |x0: i32, border: Option<HeightfieldBorder>| {
            let heights: Arc<[f32]> = (0..3).flat_map(|_| (0..3).map(|x| h(x0 + x))).collect::<Vec<_>>().into();
            let mut o = HeightfieldOptions::default();
            o.border = border;
            Shape::heightfield_with_options(heights, UVec2::new(3, 3), Vec2::new(2.0, 2.0), o)
        };
        let left_border = HeightfieldBorder::new(vec![h(-1); 3], vec![h(3); 3], vec![h(0), h(1), h(2)], vec![h(0), h(1), h(2)]);
        let right_border = HeightfieldBorder::new(vec![h(1); 3], vec![h(5); 3], vec![h(2), h(3), h(4)], vec![h(2), h(3), h(4)]);
        let (left, right) = (chunk(0, Some(left_border)), chunk(2, Some(right_border)));
        // The crest column: x index 2 of the left chunk, x index 0 of the right one.
        let ln = Vec3::from_array(normals(&left.build())[2]);
        let rn = Vec3::from_array(normals(&right.build())[0]);
        assert!((ln - rn).length() < 1e-6, "{ln:?} vs {rn:?}");
        // Without the border the crest is lopsided: the two normals differ.
        let (left, right) = (chunk(0, None), chunk(2, None));
        let (ln, rn) = (
            Vec3::from_array(normals(&left.build())[2]),
            Vec3::from_array(normals(&right.build())[0]),
        );
        assert!((ln - rn).length() > 0.1);
    }

    #[test]
    fn heightfield_vertex_count_includes_skirts() {
        let mesh = heightfield(5, 4, |_, _| 0.0).build();
        // The skirts of two edges share their vertex at each of the 4 corners.
        assert_eq!(positions(&mesh).len(), 5 * 4 + 2 * (5 + 4) - 4);
        let grid_tris = 4 * 3 * 2;
        let skirt_tris = 2 * (4 + 3) * 2;
        assert_eq!(mesh.indices().unwrap().len(), (grid_tris + skirt_tris) * 3);
    }

    #[test]
    fn flat_heightfield_normals_point_up() {
        let mesh = heightfield(4, 4, |_, _| 2.0).build();
        for n in normals(&mesh).iter().take(16) {
            assert!((Vec3::from_array(*n) - Vec3::Y).length() < 1e-5);
        }
    }

    #[test]
    fn sloped_heightfield_normals_lean_downhill() {
        // Height rises with x, so the normal tilts towards -x.
        let mesh = heightfield(4, 4, |x, _| x as f32).build();
        let n = Vec3::from_array(normals(&mesh)[5]);
        assert!(n.x < -0.1 && n.y > 0.0 && n.z.abs() < 1e-5);
    }

    #[test]
    fn heightfield_triangles_face_up() {
        let mesh = heightfield(3, 3, |_, _| 0.0).build();
        let p: Vec<Vec3> = positions(&mesh).into_iter().map(Vec3::from_array).collect();
        let idx: Vec<u32> = mesh.indices().unwrap().iter().map(|i| i as u32).collect();
        // The first 2x2 cells (8 triangles) are the grid.
        for tri in idx.chunks(3).take(8) {
            let n = (p[tri[1] as usize] - p[tri[0] as usize])
                .cross(p[tri[2] as usize] - p[tri[0] as usize]);
            assert!(n.y > 0.0, "grid triangle faces down: {n:?}");
        }
    }

    #[test]
    fn skirts_hang_below_the_edge_and_face_outwards() {
        let mesh = heightfield(3, 3, |_, _| 1.0).build();
        let p: Vec<Vec3> = positions(&mesh).into_iter().map(Vec3::from_array).collect();
        let idx: Vec<u32> = mesh.indices().unwrap().iter().map(|i| i as u32).collect();
        let centre = Vec3::new(0.0, 1.0, 0.0);
        for tri in idx.chunks(3).skip(8) {
            let (a, b, c) = (p[tri[0] as usize], p[tri[1] as usize], p[tri[2] as usize]);
            let n = (b - a).cross(c - a);
            let mid = (a + b + c) / 3.0;
            assert!(n.dot(mid - centre) > 0.0, "skirt faces inwards: {n:?}");
            assert!(n.y.abs() < 1e-5);
        }
        let lowest = p.iter().map(|p| p.y).fold(f32::MAX, f32::min);
        assert!((lowest - (1.0 - 0.8)).abs() < 1e-5);
    }

    #[test]
    fn heightfield_bounds_include_heights_and_skirts() {
        let shape = heightfield(3, 3, |x, y| (x + y) as f32);
        let b = shape.bounds();
        assert_eq!(b.min.x, -4.0);
        assert_eq!(b.max.z, 3.0);
        assert_eq!(b.max.y, 4.0);
        let built = ShapeBounds::from_points(
            positions(&shape.build()).into_iter().map(Vec3::from_array),
        );
        assert!((b.min - built.min).length() < 1e-5 && (b.max - built.max).length() < 1e-5);
    }

    #[test]
    fn equal_heightfield_parameters_give_equal_shapes() {
        let a = heightfield(3, 3, |_, _| 0.0);
        let b = heightfield(3, 3, |_, _| 0.0);
        assert_eq!(a, b);
        assert_eq!(hash_of(&a), hash_of(&b));
        let moved = Shape::heightfield(field(3, 3, |_, _| 0.0), UVec2::new(3, 3), Vec2::new(8.0, 7.0));
        assert_ne!(a, moved);
        // Different heights are a different shape, with no key to bump.
        let raised = heightfield(3, 3, |x, y| if x == 1 && y == 1 { 0.5 } else { 0.0 });
        assert_ne!(a, raised);
        assert_ne!(hash_of(&a), hash_of(&raised));
        // The same heights in a separate allocation are the same shape.
        let again = Shape::heightfield(field(3, 3, |_, _| 0.0), UVec2::new(3, 3), Vec2::new(8.0, 6.0));
        assert_eq!(a, again);
        assert_eq!(hash_of(&a), hash_of(&again));
    }

    #[test]
    fn invalid_heightfield_builds_an_empty_mesh() {
        let one_wide = heightfield(1, 4, |_, _| 0.0);
        assert!(positions(&one_wide.build()).is_empty());
        let wrong_len = Shape::heightfield(vec![0.0; 5].into(), UVec2::new(3, 3), Vec2::splat(1.0));
        assert!(positions(&wrong_len.build()).is_empty());
        assert_eq!(wrong_len.bounds(), ShapeBounds::default());
    }
}

#[cfg(test)]
mod sweep_tests {
    use super::*;
    use bevy::mesh::{Indices, VertexAttributeValues};
    use std::collections::HashMap;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    fn positions(mesh: &Mesh) -> Vec<Vec3> {
        match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            Some(VertexAttributeValues::Float32x3(p)) => {
                p.iter().map(|p| Vec3::from_array(*p)).collect()
            }
            _ => Vec::new(),
        }
    }

    fn normals(mesh: &Mesh) -> Vec<Vec3> {
        match mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
            Some(VertexAttributeValues::Float32x3(p)) => {
                p.iter().map(|p| Vec3::from_array(*p)).collect()
            }
            _ => Vec::new(),
        }
    }

    fn indices(mesh: &Mesh) -> Vec<u32> {
        match mesh.indices() {
            Some(Indices::U32(v)) => v.clone(),
            Some(Indices::U16(v)) => v.iter().map(|&i| u32::from(i)).collect(),
            None => Vec::new(),
        }
    }

    /// A 2 wide, 0.2 tall box outline.
    fn slab() -> Vec<Vec2> {
        vec![
            Vec2::new(-1.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 0.2),
            Vec2::new(-1.0, 0.2),
        ]
    }

    fn straight() -> Vec<Vec3> {
        vec![Vec3::ZERO, Vec3::new(6.0, 0.0, 0.0)]
    }

    fn near(a: Vec3, b: Vec3) -> bool {
        a.distance(b) < 1e-4
    }

    fn has_vertex(mesh: &Mesh, p: Vec3) -> bool {
        positions(mesh).iter().any(|v| near(*v, p))
    }

    /// Open boundary edges after welding vertices by position; 0 means closed.
    fn boundary_edges(mesh: &Mesh) -> usize {
        let key = |p: Vec3| {
            [
                (p.x * 1000.0).round() as i64,
                (p.y * 1000.0).round() as i64,
                (p.z * 1000.0).round() as i64,
            ]
        };
        let p = positions(mesh);
        let mut edges: HashMap<([i64; 3], [i64; 3]), i32> = HashMap::new();
        for t in indices(mesh).chunks_exact(3) {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                let (ka, kb) = (key(p[a as usize]), key(p[b as usize]));
                if ka == kb {
                    continue;
                }
                let (lo, hi, dir) = if ka < kb { (ka, kb, 1) } else { (kb, ka, -1) };
                *edges.entry((lo, hi)).or_default() += dir;
            }
        }
        // A closed, consistently wound surface uses every edge once per
        // direction, so the signed count cancels out.
        edges.values().filter(|&&v| v != 0).count()
    }

    fn hash_of(s: &Shape) -> u64 {
        let mut h = DefaultHasher::new();
        s.hash(&mut h);
        h.finish()
    }

    #[test]
    fn straight_sweep_counts_and_length() {
        let shape = Shape::sweep(slab(), straight(), SweepOptions::default());
        let mesh = shape.build();
        // 4 corners all hard: 2 vertices each per ring, 2 rings, plus 2 caps of 4.
        assert_eq!(positions(&mesh).len(), 2 * 8 + 2 * 4);
        let b = shape.bounds();
        assert!((b.size() - Vec3::new(6.0, 0.2, 2.0)).length() < 1e-4, "{:?}", b.size());
        let no_caps = Shape::sweep(
            slab(),
            straight(),
            SweepOptions {
                caps: false,
                ..default()
            },
        )
        .build();
        assert_eq!(positions(&no_caps).len(), 16);
        // Caps add two triangles per cap for a quad profile.
        assert_eq!(indices(&mesh).len() - indices(&no_caps).len(), 2 * 2 * 3);
    }

    #[test]
    fn triangles_face_the_way_their_normals_point() {
        for options in [
            SweepOptions::default(),
            SweepOptions {
                closed: true,
                ..default()
            },
        ] {
            let path = if options.closed {
                vec![
                    Vec3::new(-3.0, 0.0, -3.0),
                    Vec3::new(3.0, 0.0, -3.0),
                    Vec3::new(3.0, 0.0, 3.0),
                    Vec3::new(-3.0, 0.0, 3.0),
                ]
            } else {
                vec![Vec3::ZERO, Vec3::new(4.0, 0.0, 0.0), Vec3::new(4.0, 2.0, 4.0)]
            };
            let mesh = Shape::sweep(slab(), path, options).build();
            let (p, n) = (positions(&mesh), normals(&mesh));
            for t in indices(&mesh).chunks_exact(3) {
                let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
                let face = (p[b] - p[a]).cross(p[c] - p[a]);
                if face.length() < 1e-6 {
                    continue;
                }
                let avg = n[a] + n[b] + n[c];
                assert!(face.dot(avg) > 0.0, "triangle {t:?} faces against its normals");
            }
        }
    }

    #[test]
    fn sweeps_are_watertight() {
        let open = Shape::sweep(
            slab(),
            vec![Vec3::ZERO, Vec3::new(4.0, 0.0, 0.0), Vec3::new(4.0, 0.0, 4.0)],
            SweepOptions::default(),
        )
        .build();
        assert_eq!(boundary_edges(&open), 0);
        let closed = Shape::sweep(
            slab(),
            vec![
                Vec3::new(-3.0, 0.0, -3.0),
                Vec3::new(3.0, 0.0, -3.0),
                Vec3::new(3.0, 0.0, 3.0),
                Vec3::new(-3.0, 0.0, 3.0),
            ],
            SweepOptions {
                closed: true,
                ..default()
            },
        )
        .build();
        assert_eq!(boundary_edges(&closed), 0);
        let uncapped = Shape::sweep(
            slab(),
            straight(),
            SweepOptions {
                caps: false,
                ..default()
            },
        )
        .build();
        assert_eq!(boundary_edges(&uncapped), 8, "an open tube shows both rims");
    }

    #[test]
    fn mitred_bend_keeps_the_width() {
        let mesh = Shape::sweep(
            slab(),
            vec![Vec3::ZERO, Vec3::new(4.0, 0.0, 0.0), Vec3::new(4.0, 0.0, 4.0)],
            SweepOptions::default(),
        )
        .build();
        // Outer and inner corners of the joint sit exactly one unit from both legs.
        assert!(has_vertex(&mesh, Vec3::new(5.0, 0.2, -1.0)));
        assert!(has_vertex(&mesh, Vec3::new(3.0, 0.2, 1.0)));
        assert!(has_vertex(&mesh, Vec3::new(5.0, 0.0, -1.0)));
        // Every vertex is exactly one unit from the line of one of the two legs
        // in plan view: the sides keep a constant width through the bend.
        for p in positions(&mesh) {
            let d = p.z.abs().min((p.x - 4.0).abs());
            assert!((d - 1.0).abs() < 1e-3, "vertex {p:?} is {d} from the leg lines");
        }
    }

    #[test]
    fn sharp_turn_does_not_spike() {
        let angle = 150.0_f32.to_radians();
        let path = vec![
            Vec3::ZERO,
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0) + Vec3::new(angle.cos(), 0.0, angle.sin()) * 4.0,
        ];
        let mesh = Shape::sweep(slab(), path.clone(), SweepOptions::default()).build();
        let to_path = |p: Vec3| {
            path.windows(2)
                .map(|w| {
                    let ab = w[1] - w[0];
                    let t = ((p - w[0]).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
                    Vec3::new(p.x, 0.0, p.z).distance(w[0] + ab * t)
                })
                .fold(f32::MAX, f32::min)
        };
        // The mitre tip is the farthest point from the path: 1 / cos(75 deg) ~ 3.9
        // half-widths, never beyond the clamp of 4.
        let farthest = positions(&mesh).iter().map(|p| to_path(*p)).fold(0.0, f32::max);
        assert!(farthest > 3.0, "the sharp joint should widen, got {farthest}");
        assert!(farthest <= 4.0 + 1e-3, "mitre grew to {farthest}");
        assert!(positions(&mesh).iter().all(|p| p.is_finite()));
    }

    #[test]
    fn closed_loop_has_no_seam_gap() {
        let path: Vec<Vec3> = (0..8)
            .map(|i| {
                let a = i as f32 / 8.0 * std::f32::consts::TAU;
                Vec3::new(a.cos() * 5.0, 0.0, a.sin() * 5.0)
            })
            .collect();
        let mesh = Shape::sweep(
            slab(),
            path,
            SweepOptions {
                closed: true,
                caps: true,
                ..default()
            },
        )
        .build();
        // No caps on a closed sweep: 8 rings of 8 vertices.
        assert_eq!(positions(&mesh).len(), 8 * 8);
        assert_eq!(boundary_edges(&mesh), 0);
        // A circle sweep is symmetric: the profile at ring 0 and ring 4 mirror.
        let b = Shape::sweep(
            slab(),
            (0..8)
                .map(|i| {
                    let a = i as f32 / 8.0 * std::f32::consts::TAU;
                    Vec3::new(a.cos() * 5.0, 0.0, a.sin() * 5.0)
                })
                .collect(),
            SweepOptions {
                closed: true,
                ..default()
            },
        )
        .bounds();
        assert!((b.size().y - 0.2).abs() < 1e-4, "frames drifted: {:?}", b.size());
    }

    #[test]
    fn closed_loop_with_a_tilting_path_still_welds() {
        // A loop that leaves the horizontal plane accumulates transport twist.
        let path: Vec<Vec3> = (0..12)
            .map(|i| {
                let a = i as f32 / 12.0 * std::f32::consts::TAU;
                Vec3::new(a.cos() * 5.0, (a * 2.0).sin() * 2.0, a.sin() * 5.0)
            })
            .collect();
        let mesh = Shape::sweep(
            slab(),
            path,
            SweepOptions {
                closed: true,
                ..default()
            },
        )
        .build();
        assert_eq!(boundary_edges(&mesh), 0);
        assert!(positions(&mesh).iter().all(|p| p.is_finite()));
    }

    #[test]
    fn twist_rotates_the_end_profile() {
        let mesh = Shape::sweep(
            slab(),
            straight(),
            SweepOptions {
                twist: 0.25,
                caps: false,
                ..default()
            },
        )
        .build();
        let end: Vec<Vec3> = positions(&mesh)
            .into_iter()
            .filter(|p| (p.x - 6.0).abs() < 1e-4)
            .collect();
        assert!(!end.is_empty());
        let z = end.iter().map(|p| p.z.abs()).fold(0.0, f32::max);
        let y = end.iter().map(|p| p.y.abs()).fold(0.0, f32::max);
        // A quarter turn swaps the wide axis (z) for the tall one (y).
        assert!(z < 0.25, "z extent {z}");
        assert!(y > 0.9, "y extent {y}");
    }

    #[test]
    fn scale_along_scales_the_profile() {
        let options = SweepOptions {
            caps: false,
            ..default()
        }
        .scale_with(2, |t| Vec2::splat(1.0 + t));
        let mesh = Shape::sweep(slab(), straight(), options).build();
        let width_at = |x: f32| {
            positions(&mesh)
                .iter()
                .filter(|p| (p.x - x).abs() < 1e-4)
                .map(|p| p.z.abs())
                .fold(0.0, f32::max)
        };
        assert!((width_at(0.0) - 1.0).abs() < 1e-4);
        assert!((width_at(6.0) - 2.0).abs() < 1e-4);
    }

    #[test]
    fn tint_by_length_writes_vertex_colours() {
        let options = SweepOptions {
            tint_by_length: Some(vec![(0.0, Color::BLACK), (1.0, Color::WHITE)]),
            ..default()
        };
        let mesh = Shape::sweep(slab(), straight(), options).build();
        let Some(VertexAttributeValues::Float32x4(colors)) =
            mesh.attribute(Mesh::ATTRIBUTE_COLOR)
        else {
            panic!("tinted sweep has no vertex colours");
        };
        let p = positions(&mesh);
        let at = |x: f32| {
            let i = (0..p.len()).find(|&i| (p[i].x - x).abs() < 1e-4).unwrap();
            colors[i]
        };
        assert!(at(0.0)[0] < 0.01);
        assert!(at(6.0)[0] > 0.99);
        let mid = at(3.0)[0];
        assert!((0.1..0.9).contains(&mid), "mid tint {mid}");
        // Untinted sweeps carry no colour attribute.
        let plain = Shape::sweep(slab(), straight(), SweepOptions::default()).build();
        assert!(plain.attribute(Mesh::ATTRIBUTE_COLOR).is_none());
    }

    #[test]
    fn rounded_profiles_are_smooth_and_boxes_are_crisp() {
        let circle: Vec<Vec2> = (0..12)
            .map(|i| {
                let a = i as f32 / 12.0 * std::f32::consts::TAU;
                Vec2::new(a.cos(), a.sin()) * 0.5
            })
            .collect();
        let pipe = Shape::sweep(
            circle,
            straight(),
            SweepOptions {
                caps: false,
                ..default()
            },
        )
        .build();
        // One vertex per profile point per ring, plus the u seam at the first.
        assert_eq!(positions(&pipe).len(), 2 * 13);
        let boxy = Shape::sweep(slab(), straight(), SweepOptions { caps: false, ..default() }).build();
        assert_eq!(positions(&boxy).len(), 2 * 8);
    }

    #[test]
    fn degenerate_input_gives_an_empty_mesh() {
        let one_point = Shape::sweep(slab(), vec![Vec3::ZERO], SweepOptions::default());
        assert!(positions(&one_point.build()).is_empty());
        let two_profile_points = Shape::sweep(
            vec![Vec2::ZERO, Vec2::X],
            straight(),
            SweepOptions::default(),
        );
        assert!(positions(&two_profile_points.build()).is_empty());
        let closed_two = Shape::sweep(
            slab(),
            straight(),
            SweepOptions {
                closed: true,
                ..default()
            },
        );
        assert!(positions(&closed_two.build()).is_empty());
    }

    #[test]
    fn repeated_path_points_are_dropped() {
        let clean = Shape::sweep(slab(), straight(), SweepOptions::default()).build();
        let repeated = Shape::sweep(
            slab(),
            vec![Vec3::ZERO, Vec3::ZERO, Vec3::new(6.0, 0.0, 0.0), Vec3::new(6.0, 0.0, 0.0)],
            SweepOptions::default(),
        )
        .build();
        assert_eq!(positions(&repeated), positions(&clean));
    }

    #[test]
    fn clockwise_profiles_are_rewound() {
        let mut cw = slab();
        cw.reverse();
        let a = Shape::sweep(slab(), straight(), SweepOptions::default()).build();
        let b = Shape::sweep(cw, straight(), SweepOptions::default()).build();
        assert_eq!(positions(&a).len(), positions(&b).len());
        let (p, n) = (positions(&b), normals(&b));
        for t in indices(&b).chunks_exact(3) {
            let (i, j, k) = (t[0] as usize, t[1] as usize, t[2] as usize);
            let face = (p[j] - p[i]).cross(p[k] - p[i]);
            assert!(face.dot(n[i] + n[j] + n[k]) > 0.0);
        }
    }

    #[test]
    fn up_hint_orients_the_profile() {
        let flat = Shape::sweep(slab(), straight(), SweepOptions::default()).bounds();
        let sideways = Shape::sweep(
            slab(),
            straight(),
            SweepOptions {
                up: Vec3::Z,
                ..default()
            },
        )
        .bounds();
        assert!((flat.size().y - 0.2).abs() < 1e-4);
        assert!((sideways.size().z - 0.2).abs() < 1e-4);
        // A path along the up hint still gets a valid frame.
        let vertical = Shape::sweep(
            slab(),
            vec![Vec3::ZERO, Vec3::new(0.0, 5.0, 0.0)],
            SweepOptions::default(),
        );
        assert!(positions(&vertical.build()).iter().all(|p| p.is_finite()));
    }

    #[test]
    fn equal_sweeps_are_equal_and_hash_alike() {
        let a = Shape::sweep(slab(), straight(), SweepOptions::default());
        let b = Shape::sweep(slab(), straight(), SweepOptions::default());
        assert_eq!(a, b);
        assert_eq!(hash_of(&a), hash_of(&b));
        let twisted = Shape::sweep(
            slab(),
            straight(),
            SweepOptions {
                twist: 0.5,
                ..default()
            },
        );
        assert_ne!(a, twisted);
        let longer = Shape::sweep(
            slab(),
            vec![Vec3::ZERO, Vec3::new(7.0, 0.0, 0.0)],
            SweepOptions::default(),
        );
        assert_ne!(a, longer);
        assert_eq!(a.lerp(&longer, 0.2), a);
        assert_eq!(a.lerp(&longer, 0.8), longer);
    }

    fn default<T: Default>() -> T {
        T::default()
    }

    fn extent_y(mesh: &Mesh, keep: impl Fn(Vec3) -> bool) -> f32 {
        let ys: Vec<f32> = positions(mesh).into_iter().filter(|p| keep(*p)).map(|p| p.y).collect();
        ys.iter().cloned().fold(f32::MIN, f32::max) - ys.iter().cloned().fold(f32::MAX, f32::min)
    }

    #[test]
    fn a_constant_roll_turns_the_profile_onto_its_side() {
        let mesh = Shape::sweep(
            slab(),
            straight(),
            SweepOptions::default().with_roll_along(vec![(0.0, std::f32::consts::FRAC_PI_2), (1.0, std::f32::consts::FRAC_PI_2)]),
        )
        .build();
        // The 2 wide, 0.2 tall slab now stands 2 tall and 0.2 wide.
        let size = ShapeBounds::from_points(positions(&mesh)).size();
        assert!((size.y - 2.0).abs() < 1e-4 && (size.z - 0.2).abs() < 1e-4, "{size:?}");
    }

    #[test]
    fn roll_follows_its_keyframes_along_the_path() {
        let mesh = Shape::sweep(
            slab(),
            straight(),
            SweepOptions::default().with_roll_along(vec![(0.0, 0.0), (1.0, std::f32::consts::FRAC_PI_2)]),
        )
        .build();
        assert!((extent_y(&mesh, |p| p.x < 1e-4) - 0.2).abs() < 1e-4);
        assert!((extent_y(&mesh, |p| p.x > 6.0 - 1e-4) - 2.0).abs() < 1e-4);
    }

    #[test]
    fn roll_with_samples_a_closure_and_roll_is_part_of_the_identity() {
        let o = SweepOptions::default().roll_with(5, |r| r * 2.0);
        assert_eq!(o.roll_along.as_ref().unwrap().len(), 5);
        assert_eq!(o.roll_along.as_ref().unwrap()[4], (1.0, 2.0));
        let a = Shape::sweep(slab(), straight(), o.clone());
        let b = Shape::sweep(slab(), straight(), o.clone().with_roll_along(vec![(0.0, 0.0), (1.0, 1.0)]));
        assert_eq!(a, Shape::sweep(slab(), straight(), o));
        assert_ne!(a, b);
        assert_ne!(hash_of(&a), hash_of(&b));
    }

    fn straight_frames(bank_end: f32) -> Vec<SweepFrame> {
        // `right x up` points along +X, like the frames the sweep derives.
        let end_up = Vec3::new(0.0, bank_end.cos(), bank_end.sin());
        let end_right = end_up.cross(Vec3::X).normalize();
        vec![
            SweepFrame::new(Vec3::ZERO, -Vec3::Z, Vec3::Y),
            SweepFrame::new(Vec3::new(6.0, 0.0, 0.0), end_right, end_up),
        ]
    }

    #[test]
    fn frames_equal_to_the_derived_ones_give_the_same_mesh() {
        let derived = Shape::sweep(slab(), straight(), SweepOptions::default()).build();
        let given = Shape::sweep_frames(slab(), &straight_frames(0.0), SweepOptions::default()).build();
        let (a, b) = (positions(&derived), positions(&given));
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(&b) {
            assert!(near(*a, *b), "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn caller_frames_bank_the_road_and_scale_it() {
        let mut frames = straight_frames(0.5);
        frames[1] = frames[1].with_scale(Vec2::new(2.0, 1.0));
        let mesh = Shape::sweep_frames(slab(), &frames, SweepOptions::default()).build();
        // The start ring is flat; the end ring is banked by 0.5 rad and twice as wide.
        assert!((extent_y(&mesh, |p| p.x < 1e-4) - 0.2).abs() < 1e-4);
        let banked = extent_y(&mesh, |p| p.x > 6.0 - 1e-4);
        assert!((banked - (4.0 * 0.5_f32.sin() + 0.2 * 0.5_f32.cos())).abs() < 1e-3, "{banked}");
        assert_eq!(boundary_edges(&mesh), 0);
    }

    #[test]
    fn frames_are_part_of_the_identity() {
        let a = Shape::sweep_frames(slab(), &straight_frames(0.0), SweepOptions::default());
        let b = Shape::sweep_frames(slab(), &straight_frames(0.1), SweepOptions::default());
        let plain = Shape::sweep(slab(), straight(), SweepOptions::default());
        assert_eq!(a, Shape::sweep_frames(slab(), &straight_frames(0.0), SweepOptions::default()));
        assert_ne!(a, b);
        assert_ne!(hash_of(&a), hash_of(&b));
        assert_ne!(a, plain);
    }

    #[test]
    fn a_left_handed_frame_does_not_turn_the_mesh_inside_out() {
        // `right x up` points along -X here: against the path.
        let frames = [
            SweepFrame::new(Vec3::ZERO, Vec3::Z, Vec3::Y),
            SweepFrame::new(Vec3::new(6.0, 0.0, 0.0), Vec3::Z, Vec3::Y),
        ];
        let mesh = Shape::sweep_frames(slab(), &frames, SweepOptions::default()).build();
        assert!(!positions(&mesh).is_empty());
        assert_eq!(boundary_edges(&mesh), 0);
        let (p, n) = (positions(&mesh), normals(&mesh));
        for t in indices(&mesh).chunks_exact(3) {
            let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
            let face = (p[b] - p[a]).cross(p[c] - p[a]);
            if face.length() > 1e-6 {
                assert!(face.dot(n[a] + n[b] + n[c]) > 0.0);
            }
        }
        // `up` is kept, so the slab still sits on top of the path.
        assert!(extent_y(&mesh, |_| true) > 0.19 && positions(&mesh).iter().all(|p| p.y > -1e-4));
    }

    fn square_loop() -> Vec<Vec3> {
        vec![
            Vec3::new(-3.0, 0.0, -3.0),
            Vec3::new(3.0, 0.0, -3.0),
            Vec3::new(3.0, 0.0, 3.0),
            Vec3::new(-3.0, 0.0, 3.0),
        ]
    }

    #[test]
    fn a_closed_sweep_never_shows_a_seam() {
        let plain = positions(&Shape::sweep(slab(), square_loop(), SweepOptions::default().with_closed(true)).build());
        // A twist that is not whole turns is rounded to the nearest turn.
        let twisted = Shape::sweep(slab(), square_loop(), SweepOptions::default().with_closed(true).with_twist(0.4));
        let twisted = positions(&twisted.build());
        assert_eq!(plain.len(), twisted.len());
        assert!(plain.iter().zip(&twisted).all(|(a, b)| near(*a, *b)));
        // Roll that does not come back is spread over the loop, so a roll from
        // 0 to 0.3 rad across the loop is no roll at all.
        let rolled = Shape::sweep(
            slab(),
            square_loop(),
            SweepOptions::default().with_closed(true).with_roll_along(vec![(0.0, 0.0), (1.0, 0.3)]),
        );
        // (Rolled sweeps get extra rings, so look at the profile, not the vertex list.)
        let rolled = rolled.build();
        assert!((extent_y(&rolled, |_| true) - 0.2).abs() < 1e-4);
        // Whole turns still twist.
        let one_turn = Shape::sweep(slab(), square_loop(), SweepOptions::default().with_closed(true).with_twist(1.0));
        let one_turn = one_turn.build();
        assert_eq!(boundary_edges(&one_turn), 0);
        assert!(!positions(&one_turn).iter().zip(&plain).all(|(a, b)| near(*a, *b)));
    }
}

#[cfg(test)]
mod opening_tests {
    use super::*;
    use bevy::mesh::{Indices, VertexAttributeValues};
    use std::collections::HashMap;

    fn positions(mesh: &Mesh) -> Vec<Vec3> {
        match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            Some(VertexAttributeValues::Float32x3(p)) => {
                p.iter().map(|p| Vec3::from_array(*p)).collect()
            }
            _ => Vec::new(),
        }
    }

    fn normals(mesh: &Mesh) -> Vec<Vec3> {
        match mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
            Some(VertexAttributeValues::Float32x3(p)) => {
                p.iter().map(|p| Vec3::from_array(*p)).collect()
            }
            _ => Vec::new(),
        }
    }

    fn indices(mesh: &Mesh) -> Vec<u32> {
        match mesh.indices() {
            Some(Indices::U32(v)) => v.clone(),
            Some(Indices::U16(v)) => v.iter().map(|&i| u32::from(i)).collect(),
            None => Vec::new(),
        }
    }

    /// A 0.4 thick, 2 high wall section.
    fn wall() -> Vec<Vec2> {
        vec![
            Vec2::new(-0.2, 0.0),
            Vec2::new(0.2, 0.0),
            Vec2::new(0.2, 2.0),
            Vec2::new(-0.2, 2.0),
        ]
    }

    fn straight() -> Vec<Vec3> {
        vec![Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0)]
    }

    fn with(openings: Vec<SweepOpening>) -> SweepOptions {
        SweepOptions {
            openings,
            ..SweepOptions::default()
        }
    }

    fn build(path: Vec<Vec3>, options: SweepOptions) -> Mesh {
        Shape::sweep(wall(), path, options).build()
    }

    /// Open boundary edges after welding vertices by position; 0 means a closed,
    /// consistently wound surface.
    fn boundary_edges(mesh: &Mesh) -> usize {
        let key = |p: Vec3| {
            [
                (p.x * 1000.0).round() as i64,
                (p.y * 1000.0).round() as i64,
                (p.z * 1000.0).round() as i64,
            ]
        };
        let p = positions(mesh);
        let mut edges: HashMap<([i64; 3], [i64; 3]), i32> = HashMap::new();
        for t in indices(mesh).chunks_exact(3) {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                let (ka, kb) = (key(p[a as usize]), key(p[b as usize]));
                if ka == kb {
                    continue;
                }
                let (lo, hi, dir) = if ka < kb { (ka, kb, 1) } else { (kb, ka, -1) };
                *edges.entry((lo, hi)).or_default() += dir;
            }
        }
        edges.values().filter(|&&v| v != 0).count()
    }

    /// Number of triangles a ray crosses.
    fn ray_hits(mesh: &Mesh, origin: Vec3, dir: Vec3) -> usize {
        let p = positions(mesh);
        indices(mesh)
            .chunks_exact(3)
            .filter(|t| {
                let (a, b, c) = (p[t[0] as usize], p[t[1] as usize], p[t[2] as usize]);
                let (e1, e2) = (b - a, c - a);
                let h = dir.cross(e2);
                let det = e1.dot(h);
                if det.abs() < 1e-9 {
                    return false;
                }
                let f = 1.0 / det;
                let s = origin - a;
                let u = f * s.dot(h);
                if !(0.0..=1.0).contains(&u) {
                    return false;
                }
                let q = s.cross(e1);
                let v = f * dir.dot(q);
                if v < 0.0 || u + v > 1.0 {
                    return false;
                }
                f * e2.dot(q) > 1e-6
            })
            .count()
    }

    /// Rays are nudged off round numbers so they never run along the shared
    /// diagonal of two triangles.
    fn through(mesh: &Mesh, x: f32, y: f32) -> usize {
        ray_hits(mesh, Vec3::new(x + 0.0137, y + 0.0071, -5.0), Vec3::Z)
    }

    #[test]
    fn a_window_is_a_real_hole() {
        let plain = build(straight(), SweepOptions::default());
        let holed = build(
            straight(),
            with(vec![SweepOpening::rect(0.4..0.6, 0.8, 1.4)]),
        );
        // Solid wall: the ray crosses both faces. Through the window: nothing.
        assert_eq!(through(&plain, 5.0, 1.1), 2);
        assert_eq!(through(&holed, 5.0, 1.1), 0);
        // Beside and above the window the wall is still there.
        assert_eq!(through(&holed, 3.0, 1.1), 2);
        assert_eq!(through(&holed, 5.0, 1.7), 2);
        assert_eq!(through(&holed, 5.0, 0.4), 2);
        assert!(indices(&holed).len() > indices(&plain).len());
    }

    #[test]
    fn a_window_leaves_a_watertight_surface() {
        let holed = build(
            straight(),
            with(vec![SweepOpening::rect(0.4..0.6, 0.8, 1.4)]),
        );
        assert_eq!(boundary_edges(&holed), 0);
    }

    #[test]
    fn a_window_has_reveal_faces_on_the_inside() {
        let holed = build(
            straight(),
            with(vec![SweepOpening::rect(0.4..0.6, 0.8, 1.4)]),
        );
        let (p, n) = (positions(&holed), normals(&holed));
        let in_hole = |v: Vec3| (4.0..=6.0).contains(&v.x) && (0.79..=1.41).contains(&v.y);
        // Sill faces up at the bottom, lintel faces down at the top, jambs face
        // along the wall into the hole.
        let facing = |dir: Vec3, y: Option<f32>, x: Option<f32>| {
            indices(&holed).chunks_exact(3).any(|t| {
                let c = (p[t[0] as usize] + p[t[1] as usize] + p[t[2] as usize]) / 3.0;
                let avg = (n[t[0] as usize] + n[t[1] as usize] + n[t[2] as usize]).normalize();
                in_hole(c)
                    && avg.dot(dir) > 0.99
                    && y.is_none_or(|y| (c.y - y).abs() < 1e-3)
                    && x.is_none_or(|x| (c.x - x).abs() < 1e-3)
            })
        };
        assert!(facing(Vec3::Y, Some(0.8), None), "no sill");
        assert!(facing(Vec3::NEG_Y, Some(1.4), None), "no lintel");
        assert!(facing(Vec3::X, None, Some(4.0)), "no left jamb");
        assert!(facing(Vec3::NEG_X, None, Some(6.0)), "no right jamb");
    }

    #[test]
    fn a_gate_cuts_the_bottom_open() {
        let gate = build(
            straight(),
            with(vec![SweepOpening::rect(0.4..0.6, 0.0, 1.4)]),
        );
        assert_eq!(boundary_edges(&gate), 0);
        // Through the gate, up to the lintel, nothing blocks the ray; the wall
        // above it is solid.
        assert_eq!(through(&gate, 5.0, 0.1), 0);
        assert_eq!(through(&gate, 5.0, 1.3), 0);
        assert_eq!(through(&gate, 5.0, 1.7), 2);
        assert_eq!(through(&gate, 3.0, 0.1), 2);
        // No floor remains between the jambs: a ray dropped from inside the gate
        // leaves the wall, while beside the gate it hits the wall's bottom face.
        let down = ray_hits(&gate, Vec3::new(5.0137, 1.0, 0.0071), Vec3::NEG_Y);
        assert_eq!(down, 0, "the gate has no bottom face");
        let down_beside = ray_hits(&gate, Vec3::new(3.0137, 1.0, 0.0071), Vec3::NEG_Y);
        assert_eq!(down_beside, 1, "the solid wall keeps its bottom face");
    }

    #[test]
    fn an_arched_gate_is_watertight_and_open() {
        let gate = build(
            straight(),
            with(vec![SweepOpening::arch(0.35..0.65, 0.0, 1.0, 1.7, 8)]),
        );
        assert_eq!(boundary_edges(&gate), 0);
        assert_eq!(through(&gate, 5.0, 0.5), 0);
        assert_eq!(through(&gate, 5.0, 1.5), 0);
        assert_eq!(through(&gate, 5.0, 1.9), 2);
        // Near the arch's shoulder the wall closes in.
        assert_eq!(through(&gate, 3.6, 1.6), 2);
    }

    #[test]
    fn openings_follow_a_bend() {
        let path = vec![
            Vec3::ZERO,
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(5.0, 0.0, 5.0),
            Vec3::new(10.0, 0.0, 5.0),
        ];
        let options = with(vec![
            SweepOpening::arch(0.1..0.3, 0.0, 1.0, 1.6, 6),
            SweepOpening::rect(0.45..0.55, 0.6, 1.2),
            SweepOpening::rect(0.7..0.9, 0.6, 1.2),
        ]);
        let mesh = build(path, options);
        assert_eq!(boundary_edges(&mesh), 0);
        let (p, n) = (positions(&mesh), normals(&mesh));
        for t in indices(&mesh).chunks_exact(3) {
            let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
            let face = (p[b] - p[a]).cross(p[c] - p[a]);
            if face.length() < 1e-6 {
                continue;
            }
            assert!(
                face.dot(n[a] + n[b] + n[c]) > 0.0,
                "triangle {t:?} faces against its normals"
            );
        }
    }

    #[test]
    fn openings_on_a_closed_loop_stay_watertight() {
        let path: Vec<Vec3> = (0..8)
            .map(|i| {
                let a = i as f32 / 8.0 * std::f32::consts::TAU;
                Vec3::new(a.cos() * 5.0, 0.0, a.sin() * 5.0)
            })
            .collect();
        let options = SweepOptions {
            closed: true,
            openings: vec![
                SweepOpening::arch(0.2..0.3, 0.0, 1.0, 1.6, 6),
                SweepOpening::rect(0.6..0.7, 0.6, 1.2),
            ],
            ..SweepOptions::default()
        };
        let mesh = build(path, options);
        assert_eq!(boundary_edges(&mesh), 0);
    }

    #[test]
    fn overlapping_and_invalid_openings_are_ignored() {
        let first = SweepOpening::rect(0.3..0.5, 0.5, 1.5);
        let alone = build(straight(), with(vec![first.clone()]));
        let overlapping = build(
            straight(),
            with(vec![first.clone(), SweepOpening::rect(0.4..0.6, 0.2, 1.0)]),
        );
        assert_eq!(positions(&overlapping), positions(&alone));
        let invalid = build(
            straight(),
            with(vec![
                first,
                SweepOpening {
                    t: 0.7..0.8,
                    outline: vec![Vec2::ZERO, Vec2::X],
                },
                SweepOpening::rect(0.9..0.9, 0.5, 1.0),
            ]),
        );
        assert_eq!(positions(&invalid), positions(&alone));
    }

    #[test]
    fn openings_change_identity() {
        let a = Shape::sweep(wall(), straight(), SweepOptions::default());
        let b = Shape::sweep(
            wall(),
            straight(),
            with(vec![SweepOpening::rect(0.4..0.6, 0.8, 1.4)]),
        );
        let c = Shape::sweep(
            wall(),
            straight(),
            with(vec![SweepOpening::rect(0.4..0.6, 0.8, 1.4)]),
        );
        assert_ne!(a, b);
        assert_eq!(b, c);
    }

    #[test]
    fn an_opening_at_the_path_end_is_kept_inside() {
        let mesh = build(
            straight(),
            with(vec![SweepOpening::rect(0.0..0.2, 0.0, 1.4)]),
        );
        assert_eq!(boundary_edges(&mesh), 0);
        assert!(positions(&mesh).iter().all(|p| p.is_finite()));
        assert_eq!(through(&mesh, 1.0, 0.5), 0);
    }

    // --- Catmull-Rom 3D convenience constructor tests ---

    #[test]
    fn catmull_rom_3d_closed_produces_mesh() {
        let pts = vec![
            Vec3::ZERO,
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 2.0),
            Vec3::new(0.0, 0.0, 2.0),
        ];
        let shape = Shape::catmull_rom_3d(pts, 0.1, true, 8, ShapeDetail::Medium);
        let mesh = shape.build();
        assert!(
            mesh.get_vertex_buffer_size() > 0,
            "Closed Catmull-Rom spline must produce vertices"
        );
    }

    #[test]
    fn catmull_rom_3d_open_produces_mesh() {
        let pts = vec![
            Vec3::ZERO,
            Vec3::new(1.0, 0.5, 0.0),
            Vec3::new(2.0, 0.0, 1.0),
            Vec3::new(3.0, 0.5, 0.0),
        ];
        let shape = Shape::catmull_rom_3d(pts, 0.1, false, 8, ShapeDetail::Medium);
        let mesh = shape.build();
        assert!(
            mesh.get_vertex_buffer_size() > 0,
            "Open Catmull-Rom spline must produce vertices"
        );
    }

    #[test]
    fn catmull_rom_3d_too_few_points_empty() {
        // Closed needs 3, open needs 2
        let shape_closed = Shape::catmull_rom_3d(vec![Vec3::ZERO, Vec3::X], 0.1, true, 8, ShapeDetail::Medium);
        let mesh_closed = shape_closed.build();
        assert!(
            mesh_closed.get_vertex_buffer_size() == 0,
            "Closed Catmull-Rom with 2 points should be empty"
        );
        let shape_open = Shape::catmull_rom_3d(vec![Vec3::ZERO], 0.1, false, 8, ShapeDetail::Medium);
        let mesh_open = shape_open.build();
        assert!(
            mesh_open.get_vertex_buffer_size() == 0,
            "Open Catmull-Rom with 1 point should be empty"
        );
    }

    #[test]
    fn catmull_rom_3d_equality() {
        let pts = vec![Vec3::ZERO, Vec3::X, Vec3::new(1.0, 0.0, 1.0), Vec3::Z];
        let a = Shape::catmull_rom_3d(pts.clone(), 0.1, true, 8, ShapeDetail::Medium);
        let b = Shape::catmull_rom_3d(pts, 0.1, true, 8, ShapeDetail::Medium);
        assert_eq!(a, b, "Identical Catmull-Rom shapes should be equal");
    }

    // --- Torus detail tests ---

    #[test]
    fn torus_detail_affects_vertices() {
        let low = Shape::torus(1.0, 0.3, 0.0, std::f32::consts::TAU, ShapeDetail::Low).build();
        let high = Shape::torus(1.0, 0.3, 0.0, std::f32::consts::TAU, ShapeDetail::High).build();
        assert!(
            high.get_vertex_buffer_size() > low.get_vertex_buffer_size(),
            "Higher detail torus should have more vertices"
        );
    }

    // --- Polyline2D segments test ---

    #[test]
    fn polyline_2d_segments_param() {
        let pts = vec![Vec2::ZERO, Vec2::X, Vec2::new(1.0, 1.0)];
        let shape = Shape::polyline_2d(pts, 0.1, false, 8);
        let mesh = shape.build();
        assert!(mesh.get_vertex_buffer_size() > 0, "Polyline2D must produce vertices");
    }

    // --- Bezier2D segments test ---

    #[test]
    fn bezier_2d_segments_param() {
        let pts = vec![Vec2::ZERO, Vec2::new(1.0, 2.0), Vec2::new(2.0, 0.0)];
        let shape = Shape::bezier_2d(pts, 0.1, 16);
        let mesh = shape.build();
        assert!(mesh.get_vertex_buffer_size() > 0, "Bezier2D must produce vertices");
    }

    // --- Extruded polygon with options test ---

    #[test]
    fn extruded_polygon_with_options_holes_and_colors() {
        let outer = vec![
            Vec2::new(-1.0, -1.0),
            Vec2::new(1.0, -1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(-1.0, 1.0),
        ];
        let holes = vec![vec![
            Vec2::new(-0.3, -0.3),
            Vec2::new(0.3, -0.3),
            Vec2::new(0.3, 0.3),
            Vec2::new(-0.3, 0.3),
        ]];
        let colors = Some(vec![[1.0, 0.0, 0.0, 1.0]; 4]);
        let shape = Shape::extruded_polygon_with_options(outer, holes, colors, 0.5, 0.0, ShapeDetail::Medium);
        let mesh = shape.build();
        assert!(mesh.get_vertex_buffer_size() > 0, "Extruded polygon with options must produce vertices");
    }

    // --- Sweep alias tests ---

    #[test]
    fn sweep_path_alias_matches_sweep() {
        let profile = vec![Vec2::new(-0.5, 0.0), Vec2::new(0.5, 0.0), Vec2::new(0.0, 1.0)];
        let path = vec![Vec3::ZERO, Vec3::X, Vec3::new(1.0, 0.0, 1.0)];
        let opts = SweepOptions::default();
        let a = Shape::sweep(profile.clone(), path.clone(), opts.clone());
        let b = Shape::sweep_path(profile, path, opts);
        assert_eq!(a, b, "sweep_path should produce identical shape to sweep");
    }

    #[test]
    fn sweep_with_frames_alias_matches_sweep_frames() {
        let profile = vec![Vec2::new(-0.5, 0.0), Vec2::new(0.5, 0.0), Vec2::new(0.0, 1.0)];
        let frames = [
            SweepFrame::new(Vec3::ZERO, Vec3::Z, Vec3::Y),
            SweepFrame::new(Vec3::X, Vec3::Z, Vec3::Y),
        ];
        let opts = SweepOptions::default();
        let a = Shape::sweep_frames(profile.clone(), &frames, opts.clone());
        let b = Shape::sweep_with_frames(profile, &frames, opts);
        assert_eq!(a, b, "sweep_with_frames should produce identical shape to sweep_frames");
    }
}
