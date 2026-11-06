//! Outline helpers for polygons.
//!
//! Each function returns a closed outline as `Vec<Vec2>`, wound
//! counter-clockwise and centred on the origin, ready for
//! [`Shape::arbitrary_polygon`](crate::Shape::arbitrary_polygon),
//! [`Shape::extruded_polygon`](crate::Shape::extruded_polygon) and
//! [`Shape::arbitrary_polygon_with_holes`](crate::Shape::arbitrary_polygon_with_holes).
//! The last point is not repeated.

use bevy::math::Vec2;
use std::f32::consts::{PI, TAU};

/// Removes consecutive (and wrap-around) points that coincide.
fn dedup_close(points: &mut Vec<Vec2>) {
    points.dedup_by(|a, b| a.distance_squared(*b) < 1e-10);
    while points.len() > 1 && points[0].distance_squared(points[points.len() - 1]) < 1e-10 {
        points.pop();
    }
}

/// A rectangle with a semicircular top, `width` wide and `height` tall in total.
///
/// The arch is a semicircle of radius `width / 2`; when the shape is shorter
/// than that it becomes a semi-ellipse as tall as the shape. `segments` is the
/// number of straight pieces in the arch (at least 2). The bounding box is
/// centred on the origin.
pub fn arch(width: f32, height: f32, segments: u32) -> Vec<Vec2> {
    let hw = width.abs() * 0.5;
    let hh = height.abs() * 0.5;
    let radius = hw.min(hh * 2.0);
    let centre_y = hh - radius;
    let segments = segments.max(2);

    let mut points = vec![Vec2::new(-hw, -hh), Vec2::new(hw, -hh)];
    for i in 0..=segments {
        let angle = PI * i as f32 / segments as f32;
        points.push(Vec2::new(hw * angle.cos(), centre_y + radius * angle.sin()));
    }
    dedup_close(&mut points);
    points
}

/// A rectangle of `size` with rounded corners.
///
/// `radius` is clamped to half the shorter side; each corner uses `segments`
/// straight pieces (at least 1). A radius of 0 gives a plain rectangle.
pub fn rounded_rectangle(size: Vec2, radius: f32, segments: u32) -> Vec<Vec2> {
    let half = size.abs() * 0.5;
    let radius = radius.clamp(0.0, half.x.min(half.y));
    if radius <= 0.0 {
        return vec![
            Vec2::new(-half.x, -half.y),
            Vec2::new(half.x, -half.y),
            Vec2::new(half.x, half.y),
            Vec2::new(-half.x, half.y),
        ];
    }
    let segments = segments.max(1);
    let inner = half - Vec2::splat(radius);
    // Corner centres in counter-clockwise order, with the arc start angle.
    let corners = [
        (Vec2::new(inner.x, -inner.y), -PI * 0.5),
        (Vec2::new(inner.x, inner.y), 0.0),
        (Vec2::new(-inner.x, inner.y), PI * 0.5),
        (Vec2::new(-inner.x, -inner.y), PI),
    ];
    let mut points = Vec::with_capacity(4 * (segments as usize + 1));
    for (centre, start) in corners {
        for i in 0..=segments {
            let angle = start + PI * 0.5 * i as f32 / segments as f32;
            points.push(centre + Vec2::new(angle.cos(), angle.sin()) * radius);
        }
    }
    dedup_close(&mut points);
    points
}

/// SplitMix64 step: a tiny deterministic generator so outlines depend only on `seed`.
fn next_random(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 40) as f32 / (1u64 << 24) as f32
}

/// Number of vertices in a [`blob`] outline.
pub const BLOB_VERTICES: usize = 24;

/// A jittered, roughly round outline, e.g. a cobblestone.
///
/// `radius` is the mean radius and `wobble` (clamped to `0.0..=0.6`) the relative
/// size of the irregularities. The same `seed` always gives the same outline.
/// The outline is star-shaped around the origin, so it never self-intersects.
pub fn blob(radius: f32, wobble: f32, seed: u64) -> Vec<Vec2> {
    let wobble = if wobble.is_nan() { 0.0 } else { wobble.clamp(0.0, 0.6) };
    let mut state = seed ^ 0xA076_1D64_78BD_642F;

    // Smooth low-frequency lobes plus a little per-vertex jitter.
    let lobes: [(f32, f32); 3] = [2.0, 3.0, 5.0].map(|k| {
        let phase = next_random(&mut state) * TAU;
        (k, phase)
    });
    let weights = [0.5, 0.3, 0.2];
    let mut points = Vec::with_capacity(BLOB_VERTICES);
    for i in 0..BLOB_VERTICES {
        let angle = TAU * i as f32 / BLOB_VERTICES as f32;
        let lobe: f32 = lobes
            .iter()
            .zip(weights)
            .map(|((k, phase), w)| w * (k * angle + phase).sin())
            .sum();
        let jitter = (next_random(&mut state) * 2.0 - 1.0) * 0.25;
        let r = radius * (1.0 + wobble * (0.8 * lobe + jitter).clamp(-1.0, 1.0));
        points.push(Vec2::new(angle.cos(), angle.sin()) * r.max(0.0));
    }
    points
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signed_area(p: &[Vec2]) -> f32 {
        (0..p.len())
            .map(|i| p[i].perp_dot(p[(i + 1) % p.len()]))
            .sum::<f32>()
            * 0.5
    }

    fn bounds(p: &[Vec2]) -> (Vec2, Vec2) {
        p.iter().fold((Vec2::splat(f32::MAX), Vec2::splat(f32::MIN)), |(lo, hi), v| {
            (lo.min(*v), hi.max(*v))
        })
    }

    #[test]
    fn arch_is_ccw_and_centred() {
        let p = arch(2.0, 3.0, 8);
        assert!(signed_area(&p) > 0.0);
        let (lo, hi) = bounds(&p);
        assert!((lo - Vec2::new(-1.0, -1.5)).length() < 1e-5, "{lo:?}");
        assert!((hi - Vec2::new(1.0, 1.5)).length() < 1e-5, "{hi:?}");
        // Area = rectangle (2 x 2) + half disc.
        let expected = 2.0 * 2.0 + PI * 0.5;
        assert!((signed_area(&p) - expected).abs() < 0.1);
    }

    #[test]
    fn arch_handles_squat_shapes() {
        let p = arch(2.0, 0.5, 6);
        assert!(p.len() >= 3);
        assert!(signed_area(&p) > 0.0);
    }

    #[test]
    fn rounded_rectangle_is_ccw_and_centred() {
        let p = rounded_rectangle(Vec2::new(4.0, 2.0), 0.5, 4);
        assert!(signed_area(&p) > 0.0);
        let (lo, hi) = bounds(&p);
        assert!((lo - Vec2::new(-2.0, -1.0)).length() < 1e-5);
        assert!((hi - Vec2::new(2.0, 1.0)).length() < 1e-5);
        assert_eq!(rounded_rectangle(Vec2::new(4.0, 2.0), 0.0, 4).len(), 4);
        // Radius larger than the half side clamps instead of inverting.
        assert!(signed_area(&rounded_rectangle(Vec2::new(4.0, 2.0), 9.0, 4)) > 0.0);
    }

    #[test]
    fn blob_is_deterministic_and_simple() {
        let a = blob(1.0, 0.3, 7);
        assert_eq!(a, blob(1.0, 0.3, 7));
        assert_ne!(a, blob(1.0, 0.3, 8));
        assert!(signed_area(&a) > 0.0);
        assert_eq!(a.len(), BLOB_VERTICES);
        assert!(a.iter().all(|p| p.length() > 0.3 && p.length() < 1.7));
        // Zero wobble is a regular polygon.
        assert!(blob(2.0, 0.0, 1).iter().all(|p| (p.length() - 2.0).abs() < 1e-5));
    }
}
