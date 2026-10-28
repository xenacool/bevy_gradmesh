//! Procedural mesh primitives with configurable detail levels.
//!
//! Each primitive generates watertight geometry with proper normals and UVs.

use bevy::math::{Vec2, Vec3};
use bevy::render::mesh::{Mesh, PrimitiveTopology};
use bevy_asset::RenderAssetUsages;
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

#[derive(Debug, Clone)]
enum ShapeKind {
    Sphere { radius: f32 },
    Ellipsoid { size: Vec3 },
    Cuboid { size: Vec3, roundness: f32 },
    Cylinder { radius: f32, height: f32, roundness: f32 },
    Cone { radius: f32, height: f32 },
    Capsule { radius: f32, height: f32 },
    Hemisphere { radius: f32, capped: bool },
    Torus { radius: f32, thickness: f32, start: f32, span: f32 },
    RegularPrism { sides: u32, radius: f32, height: f32, roundness: f32 },
    Pyramid { sides: u32, radius: f32, height: f32 },
    // 2D shapes (billboarded flat quads)
    Rectangle { size: Vec2 },
    RectangleRim { size: Vec2, rim_width: f32 },
    Disc { radius: f32 },
    DiscRim { radius: f32, rim_width: f32 },
    Polygon { sides: u32, radius: f32 },
    PolygonRim { sides: u32, radius: f32, rim_width: f32 },
    Ellipse { radius: Vec2 },
    EllipseRim { radius: Vec2, rim_width: f32 },
    /// Arbitrary polygon with optional per-vertex colors.
    ArbitraryPolygon { vertices: Vec<Vec2>, colors: Option<Vec<[f32; 4]>> },
    /// 3D polyline rendered as tube geometry.
    Polyline3D { points: Vec<Vec3>, thickness: f32, closed: bool, segments: u32 },
    /// Bézier curve rendered as a tube along the curve path.
    BezierCurve { control_points: Vec<Vec3>, thickness: f32, segments: u32 },
    /// N-pointed star (2D flat quad, like polygon).
    Star { points: u32, outer_radius: f32, inner_radius: f32 },
    /// Ring/annulus (2D flat quad with a hole).
    Ring { outer_radius: f32, inner_radius: f32 },
    /// Wedge/slice of a cylinder (2D flat pie slice).
    Wedge { radius: f32, angle: f32 },
    /// Tapered cylinder (cone-frustum-like, but with flat top and bottom).
    TaperedCylinder { top_radius: f32, bottom_radius: f32, height: f32 },
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
            (Self::Sphere { radius: r1 }, Self::Sphere { radius: r2 }) => r1.to_bits() == r2.to_bits(),
            (Self::Ellipsoid { size: s1 }, Self::Ellipsoid { size: s2 }) => {
                s1.x.to_bits() == s2.x.to_bits() && s1.y.to_bits() == s2.y.to_bits() && s1.z.to_bits() == s2.z.to_bits()
            }
            (Self::Cuboid { size: s1, roundness: r1 }, Self::Cuboid { size: s2, roundness: r2 }) => {
                s1.x.to_bits() == s2.x.to_bits() && s1.y.to_bits() == s2.y.to_bits() && s1.z.to_bits() == s2.z.to_bits()
                    && r1.to_bits() == r2.to_bits()
            }
            (Self::Cylinder { radius: r1, height: h1, roundness: rn1 }, Self::Cylinder { radius: r2, height: h2, roundness: rn2 }) => {
                r1.to_bits() == r2.to_bits() && h1.to_bits() == h2.to_bits() && rn1.to_bits() == rn2.to_bits()
            }
            (Self::Cone { radius: r1, height: h1 }, Self::Cone { radius: r2, height: h2 }) => {
                r1.to_bits() == r2.to_bits() && h1.to_bits() == h2.to_bits()
            }
            (Self::Capsule { radius: r1, height: h1 }, Self::Capsule { radius: r2, height: h2 }) => {
                r1.to_bits() == r2.to_bits() && h1.to_bits() == h2.to_bits()
            }
            (Self::Hemisphere { radius: r1, capped: c1 }, Self::Hemisphere { radius: r2, capped: c2 }) => {
                r1.to_bits() == r2.to_bits() && c1 == c2
            }
            (Self::Torus { radius: r1, thickness: t1, start: s1, span: sp1 }, Self::Torus { radius: r2, thickness: t2, start: s2, span: sp2 }) => {
                r1.to_bits() == r2.to_bits() && t1.to_bits() == t2.to_bits()
                    && s1.to_bits() == s2.to_bits() && sp1.to_bits() == sp2.to_bits()
            }
            (Self::RegularPrism { sides: s1, radius: r1, height: h1, roundness: rn1 }, Self::RegularPrism { sides: s2, radius: r2, height: h2, roundness: rn2 }) => {
                s1 == s2 && r1.to_bits() == r2.to_bits() && h1.to_bits() == h2.to_bits() && rn1.to_bits() == rn2.to_bits()
            }
            (Self::Pyramid { sides: s1, radius: r1, height: h1 }, Self::Pyramid { sides: s2, radius: r2, height: h2 }) => {
                s1 == s2 && r1.to_bits() == r2.to_bits() && h1.to_bits() == h2.to_bits()
            }
            (Self::Rectangle { size: s1 }, Self::Rectangle { size: s2 }) => {
                s1.x.to_bits() == s2.x.to_bits() && s1.y.to_bits() == s2.y.to_bits()
            }
            (Self::RectangleRim { size: s1, rim_width: r1 }, Self::RectangleRim { size: s2, rim_width: r2 }) => {
                s1.x.to_bits() == s2.x.to_bits() && s1.y.to_bits() == s2.y.to_bits() && r1.to_bits() == r2.to_bits()
            }
            (Self::Disc { radius: r1 }, Self::Disc { radius: r2 }) => r1.to_bits() == r2.to_bits(),
            (Self::DiscRim { radius: r1, rim_width: rw1 }, Self::DiscRim { radius: r2, rim_width: rw2 }) => {
                r1.to_bits() == r2.to_bits() && rw1.to_bits() == rw2.to_bits()
            }
            (Self::Polygon { sides: s1, radius: r1 }, Self::Polygon { sides: s2, radius: r2 }) => {
                s1 == s2 && r1.to_bits() == r2.to_bits()
            }
            (Self::PolygonRim { sides: s1, radius: r1, rim_width: rw1 }, Self::PolygonRim { sides: s2, radius: r2, rim_width: rw2 }) => {
                s1 == s2 && r1.to_bits() == r2.to_bits() && rw1.to_bits() == rw2.to_bits()
            }
            (Self::Ellipse { radius: r1 }, Self::Ellipse { radius: r2 }) => {
                r1.x.to_bits() == r2.x.to_bits() && r1.y.to_bits() == r2.y.to_bits()
            }
            (Self::EllipseRim { radius: r1, rim_width: rw1 }, Self::EllipseRim { radius: r2, rim_width: rw2 }) => {
                r1.x.to_bits() == r2.x.to_bits() && r1.y.to_bits() == r2.y.to_bits() && rw1.to_bits() == rw2.to_bits()
            }
            (Self::ArbitraryPolygon { vertices: v1, colors: c1 }, Self::ArbitraryPolygon { vertices: v2, colors: c2 }) => {
                v1.len() == v2.len() && v1.iter().zip(v2.iter()).all(|(a, b)| {
                    a.x.to_bits() == b.x.to_bits() && a.y.to_bits() == b.y.to_bits()
                }) && c1 == c2
            }
            (Self::Polyline3D { points: p1, thickness: t1, closed: c1, segments: s1 },
             Self::Polyline3D { points: p2, thickness: t2, closed: c2, segments: s2 }) => {
                p1.len() == p2.len()
                    && p1.iter().zip(p2.iter()).all(|(a, b)| {
                        // Quantize to 2 decimal places for stable comparison
                        let qa = Vec3::new((a.x * 100.0).round() / 100.0, (a.y * 100.0).round() / 100.0, (a.z * 100.0).round() / 100.0);
                        let qb = Vec3::new((b.x * 100.0).round() / 100.0, (b.y * 100.0).round() / 100.0, (b.z * 100.0).round() / 100.0);
                        qa.x.to_bits() == qb.x.to_bits() && qa.y.to_bits() == qb.y.to_bits() && qa.z.to_bits() == qb.z.to_bits()
                    })
                    && t1.to_bits() == t2.to_bits()
                    && c1 == c2
                    && s1 == s2
            }
            (Self::BezierCurve { control_points: c1, thickness: t1, segments: s1 },
             Self::BezierCurve { control_points: c2, thickness: t2, segments: s2 }) => {
                c1.len() == c2.len()
                    && c1.iter().zip(c2.iter()).all(|(a, b)| {
                        // Quantize to 2 decimal places for stable comparison
                        let qa = Vec3::new((a.x * 100.0).round() / 100.0, (a.y * 100.0).round() / 100.0, (a.z * 100.0).round() / 100.0);
                        let qb = Vec3::new((b.x * 100.0).round() / 100.0, (b.y * 100.0).round() / 100.0, (b.z * 100.0).round() / 100.0);
                        qa.x.to_bits() == qb.x.to_bits() && qa.y.to_bits() == qb.y.to_bits() && qa.z.to_bits() == qb.z.to_bits()
                    })
                    && t1.to_bits() == t2.to_bits()
                    && s1 == s2
            }
            (Self::Star { points: p1, outer_radius: o1, inner_radius: i1 },
             Self::Star { points: p2, outer_radius: o2, inner_radius: i2 }) => {
                p1 == p2 && o1.to_bits() == o2.to_bits() && i1.to_bits() == i2.to_bits()
            }
            (Self::Ring { outer_radius: o1, inner_radius: i1 },
             Self::Ring { outer_radius: o2, inner_radius: i2 }) => {
                o1.to_bits() == o2.to_bits() && i1.to_bits() == i2.to_bits()
            }
            (Self::Wedge { radius: r1, angle: a1 },
             Self::Wedge { radius: r2, angle: a2 }) => {
                r1.to_bits() == r2.to_bits() && a1.to_bits() == a2.to_bits()
            }
            (Self::TaperedCylinder { top_radius: t1, bottom_radius: b1, height: h1 },
             Self::TaperedCylinder { top_radius: t2, bottom_radius: b2, height: h2 }) => {
                t1.to_bits() == t2.to_bits() && b1.to_bits() == b2.to_bits() && h1.to_bits() == h2.to_bits()
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
            Self::Cylinder { radius, height, roundness } => {
                radius.to_bits().hash(state);
                height.to_bits().hash(state);
                roundness.to_bits().hash(state);
            }
            Self::Cone { radius, height } => {
                radius.to_bits().hash(state);
                height.to_bits().hash(state);
            }
            Self::Capsule { radius, height } => {
                radius.to_bits().hash(state);
                height.to_bits().hash(state);
            }
            Self::Hemisphere { radius, capped } => {
                radius.to_bits().hash(state);
                capped.hash(state);
            }
            Self::Torus { radius, thickness, start, span } => {
                radius.to_bits().hash(state);
                thickness.to_bits().hash(state);
                start.to_bits().hash(state);
                span.to_bits().hash(state);
            }
            Self::RegularPrism { sides, radius, height, roundness } => {
                sides.hash(state);
                radius.to_bits().hash(state);
                height.to_bits().hash(state);
                roundness.to_bits().hash(state);
            }
            Self::Pyramid { sides, radius, height } => {
                sides.hash(state);
                radius.to_bits().hash(state);
                height.to_bits().hash(state);
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
            Self::PolygonRim { sides, radius, rim_width } => {
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
            Self::Polyline3D { points, thickness, closed, segments } => {
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
            Self::BezierCurve { control_points, thickness, segments } => {
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
            Self::Star { points, outer_radius, inner_radius } => {
                points.hash(state);
                outer_radius.to_bits().hash(state);
                inner_radius.to_bits().hash(state);
            }
            Self::Ring { outer_radius, inner_radius } => {
                outer_radius.to_bits().hash(state);
                inner_radius.to_bits().hash(state);
            }
            Self::Wedge { radius, angle } => {
                radius.to_bits().hash(state);
                angle.to_bits().hash(state);
            }
            Self::TaperedCylinder { top_radius, bottom_radius, height } => {
                top_radius.to_bits().hash(state);
                bottom_radius.to_bits().hash(state);
                height.to_bits().hash(state);
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
        Self {
            kind: ShapeKind::Cone { radius, height },
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
    pub fn regular_prism(sides: u32, radius: f32, height: f32, roundness: f32, detail: ShapeDetail) -> Self {
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
        Self {
            kind: ShapeKind::Pyramid {
                sides,
                radius,
                height,
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
            kind: ShapeKind::PolygonRim { sides, radius, rim_width },
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
            kind: ShapeKind::ArbitraryPolygon { vertices, colors: None },
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

    /// Builds the shape into a Bevy `Mesh`.
    pub fn build(&self) -> Mesh {
        match &self.kind {
            ShapeKind::Sphere { radius } => build_sphere(*radius, self.detail),
            ShapeKind::Ellipsoid { size } => build_ellipsoid(*size, self.detail),
            ShapeKind::Cuboid { size, roundness } => {
                build_cuboid(*size, *roundness, self.detail)
            }
            ShapeKind::Cylinder {
                radius,
                height,
                roundness,
            } => build_cylinder(*radius, *height, *roundness, self.detail),
            ShapeKind::Cone { radius, height } => build_cone(*radius, *height, self.detail),
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
            } => build_pyramid(*sides, *radius, *height, self.detail),
            // 2D shapes
            ShapeKind::Rectangle { size } => build_rectangle(*size, self.detail),
            ShapeKind::RectangleRim { size, rim_width } => {
                build_rectangle_rim(*size, *rim_width, self.detail)
            }
            ShapeKind::Disc { radius } => build_disc(*radius, self.detail),
            ShapeKind::DiscRim { radius, rim_width } => {
                build_disc_rim(*radius, *rim_width, self.detail)
            }
            ShapeKind::Polygon { sides, radius } => {
                build_polygon(*sides, *radius, self.detail)
            }
            ShapeKind::PolygonRim { sides, radius, rim_width } => {
                build_polygon_rim(*sides, *radius, *rim_width, self.detail)
            }
            ShapeKind::Ellipse { radius } => build_ellipse(*radius, self.detail),
            ShapeKind::EllipseRim { radius, rim_width } => {
                build_ellipse_rim(*radius, *rim_width, self.detail)
            }
            ShapeKind::ArbitraryPolygon { vertices, colors } => {
                build_arbitrary_polygon(vertices.clone(), colors.clone())
            }
            ShapeKind::Polyline3D { points, thickness, closed, segments } => {
                build_polyline3d(points.clone(), *thickness, *closed, *segments)
            }
            ShapeKind::BezierCurve { control_points, thickness, segments } => {
                build_bezier_curve(control_points.clone(), *thickness, *segments)
            }
            ShapeKind::Star { points, outer_radius, inner_radius } => {
                build_star(*points, *outer_radius, *inner_radius, self.detail)
            }
            ShapeKind::Ring { outer_radius, inner_radius } => {
                build_ring(*outer_radius, *inner_radius, self.detail)
            }
            ShapeKind::Wedge { radius, angle } => {
                build_wedge(*radius, *angle, self.detail)
            }
            ShapeKind::TaperedCylinder { top_radius, bottom_radius, height } => {
                build_tapered_cylinder(*top_radius, *bottom_radius, *height, self.detail)
            }
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
            uvs.push(Vec2::new(lon as f32 / segments as f32, lat as f32 / segments as f32));
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
            normals.push(Vec3::new(
                nx / (half.x * half.x * len),
                ny / (half.y * half.y * len),
                nz / (half.z * half.z * len),
            ).normalize());
            uvs.push(Vec2::new(lon as f32 / segments as f32, lat as f32 / segments as f32));
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
        Vec3::new(-hx, -hy, hz), Vec3::new(hx, -hy, hz), Vec3::new(hx, hy, hz), Vec3::new(-hx, hy, hz),
        // Back face
        Vec3::new(hx, -hy, -hz), Vec3::new(-hx, -hy, -hz), Vec3::new(-hx, hy, -hz), Vec3::new(hx, hy, -hz),
        // Top face
        Vec3::new(-hx, hy, -hz), Vec3::new(-hx, hy, hz), Vec3::new(hx, hy, hz), Vec3::new(hx, hy, -hz),
        // Bottom face
        Vec3::new(-hx, -hy, hz), Vec3::new(hx, -hy, hz), Vec3::new(hx, -hy, -hz), Vec3::new(-hx, -hy, -hz),
        // Right face
        Vec3::new(hx, -hy, -hz), Vec3::new(hx, -hy, hz), Vec3::new(hx, hy, hz), Vec3::new(hx, hy, -hz),
        // Left face
        Vec3::new(-hx, -hy, hz), Vec3::new(-hx, -hy, -hz), Vec3::new(-hx, hy, -hz), Vec3::new(-hx, hy, hz),
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
        0, 1, 2, 2, 3, 0,       // Front (+Z)
        4, 5, 6, 6, 7, 4,       // Back (-Z)
        8, 9, 10, 10, 11, 8,    // Top (+Y)
        12, 14, 13, 14, 12, 15, // Bottom (-Y) — reversed winding
        16, 18, 17, 18, 16, 19, // Right (+X) — reversed winding
        20, 22, 21, 22, 20, 23, // Left (-X) — reversed winding
    ];

    build_mesh(vertices.to_vec(), all_normals, uvs.to_vec(), indices.to_vec())
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
        (Vec3::Z, Vec3::new(0.0, -1.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)),
        // Right face (+X)
        (Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, -1.0, 0.0), Vec3::X),
        // Bottom face (-Y)
        (Vec3::X, Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, -1.0, 0.0)),
        // Top face (+Y)
        (Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0), Vec3::Y),
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
                    idx, idx + edge_vertices as u32, idx + 1,
                    idx + 1, idx + edge_vertices as u32, idx + edge_vertices as u32 + 1,
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
    ).normalize_or(*face_normal);

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
        vertices.push(Vec3::new(angle.cos() * radius, half_h, angle.sin() * radius));
        normals.push(Vec3::new(0.0, 1.0, 0.0));
        uvs.push(Vec2::new(
            0.5 + angle.cos() * 0.5,
            0.5 + angle.sin() * 0.5,
        ));
    }

    for i in 0..segments {
        indices.extend_from_slice(&[top_center, top_center + 1 + i as u32, top_center + 2 + i as u32]);
    }

    // Bottom cap
    let bottom_center = vertices.len() as u32;
    vertices.push(Vec3::new(0.0, -half_h, 0.0));
    normals.push(Vec3::new(0.0, -1.0, 0.0));
    uvs.push(Vec2::new(0.5, 0.5));

    for i in 0..=segments {
        let angle = i as f32 * 2.0 * PI / segments as f32;
        vertices.push(Vec3::new(angle.cos() * radius, -half_h, angle.sin() * radius));
        normals.push(Vec3::new(0.0, -1.0, 0.0));
        uvs.push(Vec2::new(
            0.5 + angle.cos() * 0.5,
            0.5 + angle.sin() * 0.5,
        ));
    }

    for i in 0..segments {
        indices.extend_from_slice(&[bottom_center, bottom_center + 2 + i as u32, bottom_center + 1 + i as u32]);
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
        vertices.push(Vec3::new(angle.cos() * radius, -half_h, angle.sin() * radius));
        normals.push(Vec3::new(0.0, -1.0, 0.0));
        uvs.push(Vec2::new(
            0.5 + angle.cos() * 0.5,
            0.5 + angle.sin() * 0.5,
        ));
    }

    for i in 0..segments {
        indices.extend_from_slice(&[base_center, base_center + 2 + i as u32, base_center + 1 + i as u32]);
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
            vertices.push(Vec3::new(
                angle.cos() * radius,
                0.0,
                angle.sin() * radius,
            ));
            normals.push(Vec3::new(0.0, -1.0, 0.0));
            uvs.push(Vec2::new(
                0.5 + angle.cos() * 0.5,
                0.5 + angle.sin() * 0.5,
            ));
        }

        for i in 0..segments {
            indices.extend_from_slice(&[cap_center, cap_center + 2 + i as u32, cap_center + 1 + i as u32]);
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
            indices.extend_from_slice(&[start_center as u32, j as u32, (j + 1) as u32 % (minor_segments + 1) as u32]);
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

fn build_regular_prism(sides: u32, radius: f32, height: f32, _roundness: f32, _detail: ShapeDetail) -> Mesh {
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
        vertices.push(Vec3::new(angle.cos() * radius, half_h, angle.sin() * radius));
        normals.push(Vec3::new(0.0, 1.0, 0.0));
        uvs.push(Vec2::new(
            0.5 + angle.cos() * 0.5,
            0.5 + angle.sin() * 0.5,
        ));
    }

    for i in 0..sides {
        indices.extend_from_slice(&[top_center, top_center + 1 + i as u32, top_center + 2 + i as u32]);
    }

    // Bottom cap
    let bottom_center = vertices.len() as u32;
    vertices.push(Vec3::new(0.0, -half_h, 0.0));
    normals.push(Vec3::new(0.0, -1.0, 0.0));
    uvs.push(Vec2::new(0.5, 0.5));

    for i in 0..=sides {
        let angle = i as f32 * 2.0 * PI / sides as f32;
        vertices.push(Vec3::new(angle.cos() * radius, -half_h, angle.sin() * radius));
        normals.push(Vec3::new(0.0, -1.0, 0.0));
        uvs.push(Vec2::new(
            0.5 + angle.cos() * 0.5,
            0.5 + angle.sin() * 0.5,
        ));
    }

    for i in 0..sides {
        indices.extend_from_slice(&[bottom_center, bottom_center + 2 + i as u32, bottom_center + 1 + i as u32]);
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
        vertices.push(Vec3::new(angle.cos() * radius, -half_h, angle.sin() * radius));
        normals.push(Vec3::new(0.0, -1.0, 0.0));
        uvs.push(Vec2::new(
            0.5 + angle.cos() * 0.5,
            0.5 + angle.sin() * 0.5,
        ));
    }

    for i in 0..sides {
        indices.extend_from_slice(&[base_center, base_center + 2 + i as u32, base_center + 1 + i as u32]);
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
        Vec3::new(-hx, -hy, 0.0), Vec3::new(hx, -hy, 0.0),
        Vec3::new(hx, hy, 0.0), Vec3::new(-hx, hy, 0.0),
        // Inner ring
        Vec3::new(-ix, -iy, 0.0), Vec3::new(-ix, iy, 0.0),
        Vec3::new(ix, iy, 0.0), Vec3::new(ix, -iy, 0.0),
    ];
    let normals = vec![Vec3::Z; 8];
    let uvs = vec![
        Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(1.0, 1.0), Vec2::new(0.0, 1.0),
        Vec2::new(0.2, 0.2), Vec2::new(0.2, 0.8), Vec2::new(0.8, 0.8), Vec2::new(0.8, 0.2),
    ];
    // Connect outer to inner rings
    let indices = vec![
        // Top edge
        0, 3, 5, 0, 5, 4,
        // Right edge
        1, 2, 6, 1, 6, 7,
        // Bottom edge
        2, 3, 5, 2, 5, 6,
        // Left edge
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
        vertices.push(Vec3::new(angle.cos() * inner_radius, angle.sin() * inner_radius, 0.0));
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
        vertices.push(Vec3::new(angle.cos() * inner_radius, angle.sin() * inner_radius, 0.0));
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
        vertices.push(Vec3::new(angle.cos() * radius.x, angle.sin() * radius.y, 0.0));
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
        vertices.push(Vec3::new(angle.cos() * radius.x, angle.sin() * radius.y, 0.0));
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
fn build_arbitrary_polygon(
    vertices: Vec<Vec2>,
    colors: Option<Vec<[f32; 4]>>,
) -> Mesh {
    // Need at least 3 vertices for a polygon
    if vertices.len() < 3 {
        // Return an empty mesh
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            Vec::<Vec3>::new(),
        );
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            Vec::<Vec3>::new(),
        );
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_UV_0,
            Vec::<Vec2>::new(),
        );
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
        if v.x < min_x { min_x = v.x; }
        if v.y < min_y { min_y = v.y; }
        if v.x > max_x { max_x = v.x; }
        if v.y > max_y { max_y = v.y; }
    }
    let width = (max_x - min_x).max(0.001);
    let height = (max_y - min_y).max(0.001);

    for v in &vertices {
        positions.push(Vec3::new(v.x, v.y, 0.0));
        normals.push(Vec3::Z);
        uvs.push(Vec2::new(
            (v.x - min_x) / width,
            (v.y - min_y) / height,
        ));
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
fn is_ear(
    vertices: &[Vec2],
    prev: usize,
    curr: usize,
    next: usize,
    all_indices: &[usize],
) -> bool {
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

fn build_mesh(
    vertices: Vec<Vec3>,
    normals: Vec<Vec3>,
    uvs: Vec<Vec2>,
    indices: Vec<u32>,
) -> Mesh {
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
        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<Vec3>::new());
        return mesh;
    }

    // Tessellate the Bézier curve into a series of points
    let tessellated = tessellate_bezier(&control_points, 32);

    // Build cylinder segments between consecutive points
    if tessellated.len() < 2 {
        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<Vec3>::new());
        return mesh;
    }

    let seg = segments.max(3);
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for i in 0..tessellated.len() - 1 {
        let start = tessellated[i];
        let end = tessellated[i + 1];
        let dir = (end - start).normalize();
        let seg_len = start.distance(end);

        if seg_len < 1e-6 {
            continue;
        }

        // Build a cylinder segment between start and end

        // Find two perpendicular axes for the tube cross-section
        let (up, right) = perpendicular_axes(&dir);

        let base_idx = positions.len() as u32;

        // Generate vertices for this segment
        for j in 0..seg {
            let angle = j as f32 * std::f32::consts::TAU / seg as f32;
            let cos_a = angle.cos();
            let sin_a = angle.sin();

            let radial = up * cos_a + right * sin_a;
            let uv = bevy::math::Vec2::new(j as f32 / seg as f32, i as f32 / (tessellated.len() - 2) as f32);

            // Start cap vertex
            positions.push(start + radial * thickness);
            normals.push(radial);
            uvs.push(uv);

            // End cap vertex
            positions.push(end + radial * thickness);
            normals.push(radial);
            uvs.push(bevy::math::Vec2::new(j as f32 / seg as f32, (i + 1) as f32 / (tessellated.len() - 2) as f32));
        }

        // Generate indices for this segment
        for j in 0..seg {
            let next = (j + 1) % seg;
            let a = base_idx + j * 2;
            let b = base_idx + j * 2 + 1;
            let c = base_idx + next * 2;
            let d = base_idx + next * 2 + 1;
            indices.extend_from_slice(&[a, c, b]);
            indices.extend_from_slice(&[b, c, d]);
        }
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(bevy::render::mesh::Indices::U32(indices));
    mesh
}

/// Find two perpendicular axes for a given direction vector.
fn perpendicular_axes(dir: &Vec3) -> (Vec3, Vec3) {
    let up = if dir.abs().y < 0.9 {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let right = dir.cross(up).normalize();
    let up = right.cross(*dir).normalize();
    (up, right)
}

/// Tessellate a Bézier curve into a series of points.
/// Supports linear (2 points), quadratic (3 points), and cubic (4+ points) curves.
fn tessellate_bezier(control_points: &[Vec3], num_segments: u32) -> Vec<Vec3> {
    let n = num_segments.max(1) as f32;
    let mut points = Vec::new();

    match control_points.len() {
        2 => {
            // Linear interpolation
            for i in 0..=num_segments {
                let t = i as f32 / n;
                points.push(control_points[0] * (1.0 - t) + control_points[1] * t);
            }
        }
        3 => {
            // Quadratic Bézier
            for i in 0..=num_segments {
                let t = i as f32 / n;
                let u = 1.0 - t;
                points.push(
                    control_points[0] * u * u
                        + control_points[1] * 2.0 * u * t
                        + control_points[2] * t * t,
                );
            }
        }
        _ if control_points.len() >= 4 => {
            // Cubic Bézier (use first 4 control points)
            for i in 0..=num_segments {
                let t = i as f32 / n;
                let u = 1.0 - t;
                points.push(
                    control_points[0] * u * u * u
                        + control_points[1] * 3.0 * u * u * t
                        + control_points[2] * 3.0 * u * t * t
                        + control_points[3] * t * t * t,
                );
            }
        }
        _ => {}
    }

    points
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
        uvs.push(Vec2::new(v.x / outer_radius * 0.5 + 0.5, v.y / outer_radius * 0.5 + 0.5));
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
        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
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
        uvs.push(Vec2::new(cos_a * (inner_radius / outer_radius) * 0.5 + 0.5, sin_a * (inner_radius / outer_radius) * 0.5 + 0.5));
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
        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
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
        uvs.push(Vec2::new(t * 0.5 + 0.5 * (a.cos() * 0.5 + 0.5), t * 0.5 + 0.5 * (a.sin() * 0.5 + 0.5)));
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
        vertices.push(Vec3::new(cos_a * bottom_radius, -half_h, sin_a * bottom_radius));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_has_valid_topology() {
        let mesh = Shape::sphere(1.0, ShapeDetail::Medium).build();
        assert!(mesh.get_vertex_buffer_size() > 0, "Sphere must have vertices");
        assert!(mesh.indices().is_some(), "Sphere must have indices");
    }

    #[test]
    fn capsule_generates_mesh() {
        let mesh = Shape::capsule(0.5, 2.0, ShapeDetail::Medium).build();
        assert!(mesh.get_vertex_buffer_size() > 0, "Capsule must have vertices");
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
        let normals_data = mesh.attribute(Mesh::ATTRIBUTE_NORMAL).expect("Missing normals");
        let normals: Vec<[f32; 3]> = match normals_data {
            bevy::render::mesh::VertexAttributeValues::Float32x3(v) => v.clone(),
            _ => panic!("Expected Float32x3 normals"),
        };
        // Check that normals are not all axis-aligned (which would indicate no rounding).
        // With the rounding algorithm, smooth normals appear at corners and edges where the
        // fillet curves. Flat face centers retain axis-aligned normals. We check that
        // a significant fraction of normals have at least two nonzero components,
        // indicating the fillet regions are generating smooth normals.
        let non_axis_aligned = normals.iter().filter(|n| {
            let abs = [n[0].abs(), n[1].abs(), n[2].abs()];
            // Count how many components are significant (> 0.1)
            let significant = abs.iter().filter(|&&a| a > 0.1).count();
            // Non-axis-aligned normals have at least 2 significant components
            // and are not dominated by a single component
            significant >= 2 && abs[0].max(abs[1]).max(abs[2]) < 0.99
        }).count();
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
            assert!(
                mesh.count_vertices() > 0,
                "2D shape must generate vertices"
            );
        }
    }

    #[test]
    fn rounded_cuboid_roundness_affects_geometry() {
        let low_round = Shape::cuboid(Vec3::ONE, 0.1, ShapeDetail::Medium).build();
        let high_round = Shape::cuboid(Vec3::ONE, 0.9, ShapeDetail::Medium).build();

        let low_verts: Vec<[f32; 3]> = match low_round.attribute(Mesh::ATTRIBUTE_POSITION).expect("Missing positions") {
            bevy::render::mesh::VertexAttributeValues::Float32x3(v) => v.clone(),
            _ => panic!("Expected Float32x3 positions"),
        };
        let high_verts: Vec<[f32; 3]> = match high_round.attribute(Mesh::ATTRIBUTE_POSITION).expect("Missing positions") {
            bevy::render::mesh::VertexAttributeValues::Float32x3(v) => v.clone(),
            _ => panic!("Expected Float32x3 positions"),
        };

        // Higher roundness should pull vertices more toward the center
        let low_avg_dist: f32 = low_verts.iter().map(|v| (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt()).sum::<f32>() / low_verts.len() as f32;
        let high_avg_dist: f32 = high_verts.iter().map(|v| (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt()).sum::<f32>() / high_verts.len() as f32;
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
        let normals_data = mesh.attribute(Mesh::ATTRIBUTE_NORMAL).expect("Missing normals");
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
                (a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6 && (a[2] - b[2]).abs() < 1e-6
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
            let positions_data =
                mesh.attribute(Mesh::ATTRIBUTE_POSITION).expect("Missing positions");
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
            let positions_data =
                mesh.attribute(Mesh::ATTRIBUTE_POSITION).expect("Missing positions");
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
        assert!(mesh.attribute(Mesh::ATTRIBUTE_COLOR).is_some(), "Should have color attribute");
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
        assert_eq!(idx.len(), 12, "Concave polygon should have 12 indices (4 triangles)");
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
        assert_eq!(indices.len(), 6, "Square should produce 2 triangles (6 indices)");
    }

    #[test]
    fn ear_clip_triangulate_triangle() {
        let vertices = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.5, 1.0),
        ];
        let indices = ear_clip_triangulate(&vertices);
        assert_eq!(indices.len(), 3, "Triangle should produce 1 triangle (3 indices)");
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
        assert!(mesh.count_vertices() > 0, "Closed polyline must have vertices");
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
        let points1 = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        ];
        let points2 = vec![
            Vec3::new(0.001, 0.0, 0.0),
            Vec3::new(1.001, 0.0, 0.0),
        ];
        let shape1 = Shape::polyline3d(points1, 0.1, false, 8, ShapeDetail::Medium);
        let shape2 = Shape::polyline3d(points2, 0.1, false, 8, ShapeDetail::Medium);
        // Quantized to 2 decimal places, these should be equal
        assert_eq!(shape1, shape2, "Quantized points should match");
    }

    #[test]
    fn bezier_curve_basic() {
        // Linear curve (2 control points)
        let control_points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
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
        assert_eq!(mesh.count_vertices(), 0, "Single point Bézier should be empty");
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
        assert_eq!(points.len(), 5, "Linear Bézier with 4 segments produces 5 points");
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
        assert_eq!(mesh.count_vertices(), 0, "Equal radius ring should be empty");
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
        assert!(mesh.count_vertices() > 0, "Tapered cylinder must have vertices");
        assert!(mesh.indices().is_some(), "Tapered cylinder must have indices");
    }

    #[test]
    fn tapered_cylinder_equal_radii_like_cylinder() {
        // Equal radii should produce a regular cylinder
        let mesh = Shape::tapered_cylinder(1.0, 1.0, 2.0, ShapeDetail::Medium).build();
        assert!(mesh.count_vertices() > 0, "Equal radius tapered cylinder must have vertices");
    }

    #[test]
    fn tapered_cylinder_hash_stability() {
        let t1 = Shape::tapered_cylinder(0.5, 1.0, 2.0, ShapeDetail::Medium);
        let t2 = Shape::tapered_cylinder(0.5, 1.0, 2.0, ShapeDetail::Medium);
        assert_eq!(t1, t2, "Identical tapered cylinders should be equal");
    }
}
