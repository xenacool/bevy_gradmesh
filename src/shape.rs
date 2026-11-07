//! Procedural mesh primitives with configurable detail levels.
//!
//! Each primitive generates watertight geometry with proper normals and UVs.

use bevy::color::{Color, ColorToComponents};
use bevy::math::{Rect, Vec2, Vec3};
use bevy::render::mesh::{Mesh, PrimitiveTopology};
use bevy::transform::components::Transform;
use bevy_asset::RenderAssetUsages;

use crate::geometry;
use std::f32::consts::PI;
use std::hash::Hash;

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
    },
    /// Flat thick Bézier curve in the XY plane.
    Bezier2D {
        control_points: Vec<Vec2>,
        thickness: f32,
    },
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
                },
                Self::Polyline2D {
                    points: p2,
                    thickness: t2,
                    closed: c2,
                },
            ) => eq_vec2s(p1, p2) && t1.to_bits() == t2.to_bits() && c1 == c2,
            (
                Self::Bezier2D {
                    control_points: p1,
                    thickness: t1,
                },
                Self::Bezier2D {
                    control_points: p2,
                    thickness: t2,
                },
            ) => eq_vec2s(p1, p2) && t1.to_bits() == t2.to_bits(),
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
            } => {
                hash_vec2s(points, state);
                thickness.to_bits().hash(state);
                closed.hash(state);
            }
            Self::Bezier2D {
                control_points,
                thickness,
            } => {
                hash_vec2s(control_points, state);
                thickness.to_bits().hash(state);
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

    /// Creates a cuboid with the given size and optional roundness (0.0–1.0).
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
    pub fn torus(radius: f32, thickness: f32, start: f32, span: f32) -> Self {
        Self {
            kind: ShapeKind::Torus {
                radius,
                thickness,
                start,
                span,
            },
            detail: ShapeDetail::Medium,
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
    /// The polygon is triangulated using ear-clipping. Vertices should be
    /// ordered counter-clockwise for correct winding.
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
    /// `inner_radius` controls how deep the indentations are (0.0–`outer_radius`).
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
    /// the last point back to the first. The flat counterpart of
    /// [`Shape::polyline3d`].
    pub fn polyline_2d(points: Vec<Vec2>, thickness: f32, closed: bool) -> Self {
        Self {
            kind: ShapeKind::Polyline2D {
                points,
                thickness,
                closed,
            },
            detail: ShapeDetail::default(),
        }
    }

    /// Creates a flat Bézier curve `thickness` wide through any number of control
    /// points (degree = points - 1) in the XY plane. `detail` sets how finely the
    /// curve is sampled. The flat counterpart of the 3D Bézier curve.
    pub fn bezier_2d(control_points: Vec<Vec2>, thickness: f32, detail: ShapeDetail) -> Self {
        Self {
            kind: ShapeKind::Bezier2D {
                control_points,
                thickness,
            },
            detail,
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
    /// The bounds are measured from the generated vertices, so they are exact
    /// for every shape kind (rounded edges, partial tori, compounds, lathes and
    /// curves included) but cost one mesh build; cache the result when reading
    /// it every frame.
    pub fn bounds(&self) -> ShapeBounds {
        mesh_bounds(&self.build())
    }

    /// Whether the shape's outline is round: spheres, ellipsoids, discs,
    /// ellipses and rings (and their rims). Useful for ray tests that
    /// approximate such shapes with an ellipsoid inside [`Shape::bounds`].
    pub fn is_round(&self) -> bool {
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
        mesh_bounds(&self.build_2d())
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
                build_arbitrary_polygon(vertices.clone(), colors.clone())
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
            } => geometry::polyline_2d_mesh(points, *thickness, *closed),
            ShapeKind::Bezier2D {
                control_points,
                thickness,
            } => geometry::polyline_2d_mesh(
                &geometry::bezier_2d_points(control_points, self.detail.segments() * 2),
                *thickness,
                false,
            ),
        }
    }
}

// ─── Mesh builders ───

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
        12, 14, 13, 14, 12, 15, // Bottom (-Y) — reversed winding
        16, 18, 17, 18, 16, 19, // Right (+X) — reversed winding
        20, 22, 21, 22, 20, 23, // Left (-X) — reversed winding
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
    //   2. Clamp cube_point to [-0.5+rx, 0.5-rx] etc. → center
    //   3. direction = normalize((cube_point - center) / radii)
    //   4. position = center + radii * direction
    //   5. normal = normalize(direction / radii)
    //
    // This produces true spherical/elliptical fillets at edges and corners while
    // preserving flat face centers.

    const MAXIMUM_NORMALIZED_FILLET_RADIUS: f32 = 0.25;
    let r = roundness.clamp(0.0, 1.0);

    // Per-axis radii in unit-box space (box spans [-0.5, 0.5] on each axis).
    // At r=0.0, radii=0 → sharp box. At r=1.0, radii=0.25 → maximum fillet.
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
        // for all faces (axis_v × axis_u points in the face normal direction).
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

    // Cylinder middle section — two rings: top ring first, then bottom ring
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

    // Cylinder indices — connects top ring to bottom ring
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
        // Start cap — fan from center of first ring
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

        // End cap — fan from center of last ring
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
/// `smooth`), producing the inscribed solid "core ⊕ ball of radius `rho`".
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

    // Base cap — separate vertices with downward normal
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

// ─── 2D Mesh builders ───

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

/// Build an arbitrary polygon mesh using ear-clipping triangulation.
fn build_arbitrary_polygon(vertices: Vec<Vec2>, colors: Option<Vec<[f32; 4]>>) -> Mesh {
    // Need at least 3 vertices for a polygon
    if vertices.len() < 3 {
        // Return an empty mesh
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<Vec3>::new());
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, Vec::<Vec3>::new());
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, Vec::<Vec2>::new());
        return mesh;
    }

    let n = vertices.len();

    // Triangulate using ear-clipping
    let indices = ear_clip_triangulate(&vertices);

    // Build 3D vertices (flat on Z=0 plane)
    let mut positions = Vec::with_capacity(n);
    let mut normals = Vec::with_capacity(n);
    let mut uvs = Vec::with_capacity(n);

    // Compute bounding box for UV mapping
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for v in &vertices {
        if v.x < min_x {
            min_x = v.x;
        }
        if v.y < min_y {
            min_y = v.y;
        }
        if v.x > max_x {
            max_x = v.x;
        }
        if v.y > max_y {
            max_y = v.y;
        }
    }
    let width = (max_x - min_x).max(0.001);
    let height = (max_y - min_y).max(0.001);

    for v in &vertices {
        positions.push(Vec3::new(v.x, v.y, 0.0));
        normals.push(Vec3::Z);
        uvs.push(Vec2::new((v.x - min_x) / width, (v.y - min_y) / height));
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(bevy::render::mesh::Indices::U32(indices));

    // Store per-vertex colors if provided
    if let Some(colors) = colors {
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    }

    mesh
}

/// Ear-clipping triangulation for simple 2D polygons.
/// Returns triangle indices into the vertex array.
fn ear_clip_triangulate(vertices: &[Vec2]) -> Vec<u32> {
    let n = vertices.len();
    if n < 3 {
        return Vec::new();
    }

    // Working copy of vertex indices
    let mut indices: Vec<usize> = (0..n).collect();

    let mut triangles = Vec::new();
    let mut ear_count = 0; // Safety counter to avoid infinite loops

    while indices.len() > 3 {
        ear_count += 1;
        if ear_count > n * n {
            // Degenerate polygon; fallback to fan triangulation
            break;
        }

        let len = indices.len();
        let mut found_ear = false;

        for i in 0..len {
            let prev = indices[(i + len - 1) % len];
            let curr = indices[i];
            let next = indices[(i + 1) % len];

            if is_ear(vertices, prev, curr, next, &indices) {
                // This is an ear — add triangle and remove the ear tip
                triangles.push(prev as u32);
                triangles.push(curr as u32);
                triangles.push(next as u32);
                indices.remove(i);
                found_ear = true;
                break;
            }
        }

        if !found_ear {
            // No ear found — degenerate polygon; fallback to fan
            break;
        }
    }

    // Add the final triangle
    if indices.len() == 3 {
        triangles.push(indices[0] as u32);
        triangles.push(indices[1] as u32);
        triangles.push(indices[2] as u32);
    } else if indices.len() > 3 {
        // Fallback: fan triangulation from first vertex
        let first = indices[0];
        for i in 1..indices.len() - 1 {
            triangles.push(first as u32);
            triangles.push(indices[i] as u32);
            triangles.push(indices[i + 1] as u32);
        }
    }

    triangles
}

/// Check if vertex `curr` forms an "ear" in the polygon.
fn is_ear(vertices: &[Vec2], prev: usize, curr: usize, next: usize, all_indices: &[usize]) -> bool {
    let p = vertices[prev];
    let q = vertices[curr];
    let r = vertices[next];

    // Check that the triangle is convex (left turn for CCW polygon)
    let cross = cross_product_2d(p, q, r);
    if cross <= 0.0 {
        return false;
    }

    // Check that no other vertex is inside the ear triangle
    for &idx in all_indices {
        if idx == prev || idx == curr || idx == next {
            continue;
        }
        let s = vertices[idx];
        if point_in_triangle(s, p, q, r) {
            return false;
        }
    }

    true
}

/// 2D cross product of vectors (q-p) and (r-q).
fn cross_product_2d(p: Vec2, q: Vec2, r: Vec2) -> f32 {
    (q.x - p.x) * (r.y - q.y) - (q.y - p.y) * (r.x - q.x)
}

/// Check if point p is inside triangle (a, b, c).
fn point_in_triangle(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool {
    let d1 = sign(p, a, b);
    let d2 = sign(p, b, c);
    let d3 = sign(p, c, a);

    let has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
    let has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);

    !(has_neg && has_pos)
}

fn sign(p1: Vec2, p2: Vec2, p3: Vec2) -> f32 {
    (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)
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

/// Counts and flags cannot be blended; they follow whichever end is nearer.
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
                        match (c1, c2) {
                            (None, None) => None,
                            (Some(a), Some(b)) if a.len() == b.len() => {
                                Some(a.iter().zip(b).map(|(x, y)| lerp_rgba(*x, *y, t)).collect())
                            }
                            _ => return None,
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
                },
                K::Polyline2D {
                    points: p2,
                    thickness: t2,
                    closed: c2,
                },
            ) => K::Polyline2D {
                points: lerp_points(p1, p2, t, lerp_vec2)?,
                thickness: l(t1, t2),
                closed: pick(t, *c1, *c2),
            },
            (
                K::Bezier2D {
                    control_points: p1,
                    thickness: t1,
                },
                K::Bezier2D {
                    control_points: p2,
                    thickness: t2,
                },
            ) => K::Bezier2D {
                control_points: lerp_points(p1, p2, t, lerp_vec2)?,
                thickness: l(t1, t2),
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
    /// * Shapes of **different kinds**, open point lists (polylines, curves,
    ///   lathe profiles) of different lengths, or polygons with a different
    ///   number of holes cannot be blended and yield the nearer endpoint.
    ///
    /// `t == 0.0` returns a copy of `self` and `t == 1.0` a copy of `other`.
    ///
    /// Every distinct blended value is a distinct mesh. To animate smoothly
    /// without generating a new mesh per frame, prefer animating the
    /// `Transform` scale of a fixed shape, or quantise `t` before calling this.
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let t = if t.is_nan() { 0.0 } else { t.clamp(0.0, 1.0) };
        if t <= 0.0 {
            return self.clone();
        }
        if t >= 1.0 {
            return other.clone();
        }
        match self.kind.lerp(&other.kind, t) {
            Some(kind) => Self {
                kind,
                detail: if t < 0.5 { self.detail } else { other.detail },
            },
            None if t < 0.5 => self.clone(),
            None => other.clone(),
        }
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let full = Shape::torus(1.0, 0.3, 0.0, std::f32::consts::TAU).build();
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
            Shape::torus(0.5, 0.2, 0.0, std::f32::consts::TAU),
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
        let full = Shape::torus(1.0, 0.3, 0.0, std::f32::consts::TAU).build();
        let partial = Shape::torus(1.0, 0.3, 0.0, std::f32::consts::PI).build();
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
        let full = Shape::torus(1.0, 0.3, 0.0, std::f32::consts::TAU).build();
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
        // 5 vertices → 3 triangles → 9 indices
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
        // 6 vertices → 4 triangles → 12 indices
        assert_eq!(
            idx.len(),
            12,
            "Concave polygon should have 12 indices (4 triangles)"
        );
    }

    #[test]
    fn ear_clip_triangulate_convex() {
        // Square: 4 vertices → 2 triangles
        let vertices = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];
        let indices = ear_clip_triangulate(&vertices);
        assert_eq!(
            indices.len(),
            6,
            "Square should produce 2 triangles (6 indices)"
        );
    }

    #[test]
    fn ear_clip_triangulate_triangle() {
        let vertices = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.5, 1.0),
        ];
        let indices = ear_clip_triangulate(&vertices);
        assert_eq!(
            indices.len(),
            3,
            "Triangle should produce 1 triangle (3 indices)"
        );
    }

    #[test]
    fn ear_clip_triangulate_empty() {
        let vertices: Vec<Vec2> = vec![];
        let indices = ear_clip_triangulate(&vertices);
        assert!(indices.is_empty());
    }

    #[test]
    fn point_in_triangle_basic() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(1.0, 0.0);
        let c = Vec2::new(0.0, 1.0);
        assert!(point_in_triangle(Vec2::new(0.1, 0.1), a, b, c));
        assert!(!point_in_triangle(Vec2::new(0.6, 0.6), a, b, c));
        assert!(!point_in_triangle(Vec2::new(-0.1, 0.0), a, b, c));
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

    // ─── New shape tests ───

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
        // 360° wedge should be like a disc
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
            Shape::torus(1.0, 0.2, 0.0, std::f32::consts::TAU),
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
            Shape::polyline_2d(vec![Vec2::ZERO, Vec2::new(1.0, 0.5), Vec2::new(2.0, 0.0)], 0.1, false),
            Shape::polyline_2d(Shape::rounded_rectangle(Vec2::splat(2.0), 0.3, 3), 0.1, true),
            Shape::bezier_2d(vec![Vec2::ZERO, Vec2::new(1.0, 2.0), Vec2::new(2.0, 0.0)], 0.1, D),
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
