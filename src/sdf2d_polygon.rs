//! Polygon, star and arbitrary polygon constructors for [`SdfMaterial2d`].
//!
//! A regular polygon points a vertex up (+Y) and a star points an outer tip up.
//! An arbitrary polygon keeps the positions it was given. Rim, arc, gradients
//! and colours work as they do on rectangles and ellipses. Dashes are not
//! supported on these kinds, so a rim with a dash pattern draws solid.

use std::fmt;

use bevy::math::{Vec2, Vec4};

use crate::sdf2d::{SdfKind, SdfMaterial2d};

/// Fewest sides or points a polygon or star can have.
const MIN_SIDES: u32 = 3;
/// Most sides or points a polygon or star can have.
const MAX_SIDES: u32 = 64;
/// Fewest vertices an arbitrary polygon can have.
const MIN_VERTICES: usize = 3;
/// Most vertices an arbitrary polygon can have (the uniform holds eight pairs).
const MAX_VERTICES: usize = 16;
/// Smallest bounds extent per axis, so the quad stays drawable for flat input.
const MIN_EXTENT: f32 = 1.0;

/// Why [`SdfMaterial2d::polygon_from_points`] rejected its input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdfPolygonError {
    /// Fewer than 3 vertices were given.
    TooFewVertices(usize),
    /// More than 16 vertices were given.
    TooManyVertices(usize),
}

impl fmt::Display for SdfPolygonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewVertices(count) => {
                write!(f, "polygon needs at least {MIN_VERTICES} vertices, got {count}")
            }
            Self::TooManyVertices(count) => {
                write!(f, "polygon allows at most {MAX_VERTICES} vertices, got {count}")
            }
        }
    }
}

impl std::error::Error for SdfPolygonError {}

fn clamp_sides(count: u32) -> f32 {
    count.clamp(MIN_SIDES, MAX_SIDES) as f32
}

impl SdfMaterial2d {
    /// A filled regular polygon with `sides` sides (3 to 64) and the given
    /// circumradius. One vertex points up.
    #[must_use]
    pub fn polygon(sides: u32, radius: f32) -> Self {
        let radius = radius.max(0.0);
        let mut material = Self::new(SdfKind::Polygon, Vec2::splat(radius * 2.0));
        material.uniform.params.x = clamp_sides(sides);
        material
    }

    /// A filled star with `points` tips (3 to 64). `outer_radius` is the
    /// distance to the tips and `inner_radius` the distance to the notches. An
    /// inner radius at or above the outer radius gives a regular polygon with
    /// twice as many sides.
    #[must_use]
    pub fn star(points: u32, outer_radius: f32, inner_radius: f32) -> Self {
        let outer_radius = outer_radius.max(0.0);
        let mut material = Self::new(SdfKind::Star, Vec2::splat(outer_radius * 2.0));
        material.uniform.params.x = clamp_sides(points);
        material.uniform.params.y = inner_ratio(outer_radius, inner_radius);
        material
    }

    /// A filled polygon through `points`, in order. The outline may be concave
    /// and may be wound either way. Holes are not supported.
    ///
    /// The shape is centred on the middle of its bounding box, so the points'
    /// absolute position is not kept. Place the entity where the middle of the
    /// bounds should sit.
    ///
    /// # Errors
    ///
    /// Returns [`SdfPolygonError`] for fewer than 3 or more than 16 points.
    pub fn polygon_from_points(points: &[Vec2]) -> Result<Self, SdfPolygonError> {
        if points.len() < MIN_VERTICES {
            return Err(SdfPolygonError::TooFewVertices(points.len()));
        }
        if points.len() > MAX_VERTICES {
            return Err(SdfPolygonError::TooManyVertices(points.len()));
        }
        let (min, max) = bounds_of(points);
        let center = (min + max) * 0.5;
        let size = (max - min).max(Vec2::splat(MIN_EXTENT));
        let mut material = Self::new(SdfKind::ArbitraryPolygon, size);
        material.uniform.params.w = points.len() as f32;
        material.uniform.vertices = pack_vertices(points, center);
        Ok(material)
    }
}

fn bounds_of(points: &[Vec2]) -> (Vec2, Vec2) {
    let mut min = points[0];
    let mut max = points[0];
    for point in points {
        min = min.min(*point);
        max = max.max(*point);
    }
    (min, max)
}

/// Stores two vertices per `Vec4`, relative to `center`. Unused slots stay zero.
fn pack_vertices(points: &[Vec2], center: Vec2) -> [Vec4; 8] {
    let mut packed = [Vec4::ZERO; 8];
    for (index, point) in points.iter().enumerate() {
        let relative = *point - center;
        let slot = &mut packed[index / 2];
        if index % 2 == 0 {
            slot.x = relative.x;
            slot.y = relative.y;
        } else {
            slot.z = relative.x;
            slot.w = relative.y;
        }
    }
    packed
}

fn inner_ratio(outer_radius: f32, inner_radius: f32) -> f32 {
    if outer_radius <= 0.0 {
        return 1.0;
    }
    (inner_radius / outer_radius).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::color::Color;

    #[test]
    fn polygon_sets_kind_count_and_size() {
        let m = SdfMaterial2d::polygon(6, 100.0);
        assert_eq!(m.uniform.shape.x, SdfKind::Polygon.gpu_code());
        assert_eq!(m.uniform.params.x, 6.0);
        assert_eq!(m.size(), Vec2::splat(200.0));
    }

    #[test]
    fn star_sets_kind_count_ratio_and_size() {
        let m = SdfMaterial2d::star(5, 100.0, 40.0);
        assert_eq!(m.uniform.shape.x, SdfKind::Star.gpu_code());
        assert_eq!(m.uniform.params.x, 5.0);
        assert!((m.uniform.params.y - 0.4).abs() < 1e-6);
        assert_eq!(m.size(), Vec2::splat(200.0));
    }

    #[test]
    fn counts_clamp_to_the_supported_range() {
        assert_eq!(SdfMaterial2d::polygon(2, 10.0).uniform.params.x, 3.0);
        assert_eq!(SdfMaterial2d::polygon(500, 10.0).uniform.params.x, 64.0);
        assert_eq!(SdfMaterial2d::star(1, 10.0, 4.0).uniform.params.x, 3.0);
        assert_eq!(SdfMaterial2d::star(65, 10.0, 4.0).uniform.params.x, 64.0);
    }

    #[test]
    fn star_with_large_inner_radius_gets_ratio_one() {
        assert_eq!(SdfMaterial2d::star(5, 10.0, 10.0).uniform.params.y, 1.0);
        assert_eq!(SdfMaterial2d::star(5, 10.0, 25.0).uniform.params.y, 1.0);
        assert_eq!(SdfMaterial2d::star(5, 0.0, 5.0).uniform.params.y, 1.0);
    }

    #[test]
    fn zero_radius_keeps_the_quad_finite() {
        let m = SdfMaterial2d::polygon(5, 0.0);
        assert!(m.quad_scale().is_finite());
        assert!(m.quad_scale().x > 0.0);
    }

    #[test]
    fn rim_arc_and_colors_keep_kind_and_params() {
        let base = SdfMaterial2d::star(7, 50.0, 20.0);
        let styled = base
            .clone()
            .with_rim(4.0)
            .with_arc(0.5, 2.0)
            .with_colors(Color::BLACK, Color::WHITE);
        assert_eq!(styled.uniform.shape.x, base.uniform.shape.x);
        assert_eq!(styled.uniform.params, base.uniform.params);
    }

    #[test]
    fn lerp_between_polygons_switches_params_at_the_midpoint() {
        let a = SdfMaterial2d::polygon(3, 10.0);
        let b = SdfMaterial2d::polygon(8, 10.0);
        assert_eq!(a.lerp(&b, 0.25).uniform.params.x, 3.0);
        assert_eq!(a.lerp(&b, 0.75).uniform.params.x, 8.0);
    }

    fn l_shape() -> Vec<Vec2> {
        vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 2.0),
            Vec2::new(0.0, 2.0),
        ]
    }

    #[test]
    fn concave_polygon_packs_vertices_relative_to_the_bounds_centre() {
        let m = SdfMaterial2d::polygon_from_points(&l_shape()).unwrap();
        let u = m.uniform;
        assert_eq!(u.shape.x, SdfKind::ArbitraryPolygon.gpu_code());
        assert_eq!(u.params.w, 6.0);
        assert_eq!(m.size(), Vec2::splat(2.0));
        assert_eq!(u.vertices[0], Vec4::new(-1.0, -1.0, 1.0, -1.0));
        assert_eq!(u.vertices[1], Vec4::new(1.0, 0.0, 0.0, 0.0));
        assert_eq!(u.vertices[2], Vec4::new(0.0, 1.0, -1.0, 1.0));
        assert_eq!(u.vertices[3], Vec4::ZERO);
    }

    #[test]
    fn fewer_than_three_vertices_are_rejected() {
        for count in 0..3 {
            let points = vec![Vec2::ONE; count];
            let error = SdfMaterial2d::polygon_from_points(&points).unwrap_err();
            assert_eq!(error, SdfPolygonError::TooFewVertices(count));
        }
    }

    #[test]
    fn more_than_sixteen_vertices_are_rejected_with_the_count() {
        let points = vec![Vec2::ONE; 17];
        let error = SdfMaterial2d::polygon_from_points(&points).unwrap_err();
        assert_eq!(error, SdfPolygonError::TooManyVertices(17));
        assert!(error.to_string().contains("17"));
    }

    #[test]
    fn sixteen_vertices_fill_every_slot() {
        let points: Vec<Vec2> = (0..16).map(|i| Vec2::new(i as f32, (i % 3) as f32)).collect();
        let m = SdfMaterial2d::polygon_from_points(&points).unwrap();
        assert_eq!(m.uniform.params.w, 16.0);
        assert_ne!(m.uniform.vertices[7], Vec4::ZERO);
    }

    #[test]
    fn identical_points_still_give_a_finite_quad() {
        let points = vec![Vec2::new(5.0, 5.0); 4];
        let m = SdfMaterial2d::polygon_from_points(&points).unwrap();
        assert_eq!(m.size(), Vec2::ONE);
        assert!(m.quad_scale().is_finite());
    }

    #[test]
    fn lerp_keeps_polygon_kind_against_a_disc() {
        let polygon = SdfMaterial2d::polygon(5, 10.0);
        let disc = SdfMaterial2d::disc(10.0);
        let near_polygon = polygon.lerp(&disc, 0.25).uniform;
        assert_eq!(near_polygon.shape.x, SdfKind::Polygon.gpu_code());
        let near_disc = polygon.lerp(&disc, 0.75).uniform;
        assert_eq!(near_disc.shape.x, SdfKind::Ellipse.gpu_code());
    }
}
