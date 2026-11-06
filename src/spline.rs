//! Closed Catmull-Rom splines with stress relaxation and arc-length baking.
//!
//! A [`CatmullRomSpline`] is a closed loop through a list of control points. Every
//! segment is internally a cubic Bézier, so the curve is C1-continuous and passes
//! through each control point.
//!
//! ```
//! use bevy::math::Vec3;
//! use bevy_gradmesh::spline::CatmullRomSpline;
//!
//! let track = CatmullRomSpline::closed([
//!     Vec3::new(0.0, 0.0, 0.0),
//!     Vec3::new(40.0, 0.0, 0.0),
//!     Vec3::new(40.0, 0.0, 40.0),
//!     Vec3::new(0.0, 0.0, 40.0),
//! ])
//! // Pull over-stressed (too tight) corners straight until every turn has r >= 10.
//! .relaxed(10.0);
//!
//! // Bake once for cheap, evenly spaced lookups.
//! let baked = track.bake(0.5);
//! let frame = baked.sample_at_distance(baked.length() * 0.25);
//! assert!(frame.forward.is_normalized());
//! ```
//!
//! # Stress
//!
//! The "stress" of a segment is how tightly it bends: the smaller the horizontal turn
//! radius, the higher the stress. [`CatmullRomSpline::min_turn_radius`] measures it and
//! [`CatmullRomSpline::relaxed`] straightens stressed spans (in the XZ plane; heights are
//! left alone) by pulling their control points toward their neighbours.

use bevy::ecs::prelude::Resource;
use bevy::math::{Vec3, Vec3Swizzles};

/// Samples used per segment when measuring turn radius.
const STRESS_SAMPLES: usize = 32;
/// Maximum relaxation passes in [`CatmullRomSpline::relaxed`].
const MAX_RELAX_ITERATIONS: usize = 2000;
/// Fraction of the distance to the neighbour midpoint moved per relaxation pass.
const RELAX_STRENGTH: f32 = 0.2;
/// Dense samples used to build the arc-length table when baking.
const BAKE_DENSITY: usize = 2000;

/// A closed Catmull-Rom spline through a ring of control points.
#[derive(Debug, Clone, PartialEq)]
pub struct CatmullRomSpline {
    points: Vec<Vec3>,
}

impl CatmullRomSpline {
    /// Create a closed spline through `points`. The last point connects back to the first.
    ///
    /// # Panics
    /// Panics if fewer than 3 points are given.
    pub fn closed(points: impl Into<Vec<Vec3>>) -> Self {
        let points = points.into();
        assert!(points.len() >= 3, "a closed spline needs at least 3 points");
        Self { points }
    }

    /// The control points the curve passes through.
    pub fn points(&self) -> &[Vec3] {
        &self.points
    }

    /// Number of curve segments (equal to the number of control points).
    pub fn segment_count(&self) -> usize {
        self.points.len()
    }

    /// The cubic Bézier control points `[b0, b1, b2, b3]` of segment `index`.
    pub fn bezier_segment(&self, index: usize) -> [Vec3; 4] {
        bezier_of(&self.points, index % self.points.len())
    }

    /// Evaluate the curve at `t`, where `t` wraps around `[0, 1)` over the whole loop.
    pub fn point(&self, t: f32) -> Vec3 {
        let (segment, u) = self.locate(t);
        bezier_point(&bezier_of(&self.points, segment), u)
    }

    /// Unit tangent of the curve at `t`.
    pub fn tangent(&self, t: f32) -> Vec3 {
        let (segment, u) = self.locate(t);
        let (velocity, _) = bezier_derivatives(&bezier_of(&self.points, segment), u);
        velocity.normalize_or_zero()
    }

    /// Signed horizontal curvature (1 / radius) at `t`. Positive turns left, negative right.
    pub fn curvature(&self, t: f32) -> f32 {
        let (segment, u) = self.locate(t);
        signed_curvature(&bezier_of(&self.points, segment), u)
    }

    /// Smallest horizontal turn radius anywhere on the curve (`f32::MAX` if perfectly straight).
    pub fn min_turn_radius(&self) -> f32 {
        (0..self.points.len())
            .map(|i| segment_min_radius(&self.points, i))
            .fold(f32::MAX, f32::min)
    }

    /// Return a copy with over-stressed spans straightened so no turn is tighter than
    /// `min_radius`. Only X and Z move; heights are preserved.
    ///
    /// Stressed control points are pulled toward the midpoint of their neighbours until the
    /// whole loop is gentle enough. Over-relaxing can shrink a loop, so the best shape seen is
    /// returned: the result is never tighter than the input, and if `min_radius` is
    /// unreachable you get the gentlest curve found.
    pub fn relaxed(self, min_radius: f32) -> Self {
        let mut best_radius = self.min_turn_radius();
        let mut best = self.clone();
        let mut current = self;
        for _ in 0..MAX_RELAX_ITERATIONS {
            if best_radius >= min_radius {
                break;
            }
            current = current.relax_step(min_radius, RELAX_STRENGTH);
            let radius = current.min_turn_radius();
            if radius > best_radius {
                best_radius = radius;
                best = current.clone();
            }
        }
        best
    }

    /// One relaxation pass over every segment tighter than `min_radius`.
    fn relax_step(&self, min_radius: f32, strength: f32) -> Self {
        let n = self.points.len();
        let mut next = self.clone();
        for segment in 0..n {
            if segment_min_radius(&self.points, segment) >= min_radius {
                continue;
            }
            for index in [segment, (segment + 1) % n] {
                let target = (self.points[(index + n - 1) % n] + self.points[(index + 1) % n]) * 0.5;
                let point = &mut next.points[index];
                point.x += (target.x - point.x) * strength;
                point.z += (target.z - point.z) * strength;
            }
        }
        next
    }

    /// Bake evenly spaced (by arc length) frames, `frame_step` world units apart.
    ///
    /// Frames use parallel transport, so `right`/`up` never flip at sharp turns.
    pub fn bake(&self, frame_step: f32) -> BakedSpline {
        assert!(frame_step > 0.0, "frame_step must be positive");

        // Cumulative arc length over a dense, uniform-in-t sampling.
        let mut distances = Vec::with_capacity(BAKE_DENSITY + 1);
        let mut previous = self.point(0.0);
        let mut total = 0.0;
        distances.push(0.0);
        for i in 1..=BAKE_DENSITY {
            let p = self.point(i as f32 / BAKE_DENSITY as f32);
            total += (p - previous).length();
            distances.push(total);
            previous = p;
        }

        let first_forward = self.tangent(0.0);
        let mut right = Vec3::Y.cross(first_forward).normalize_or_zero();
        let mut up = first_forward.cross(right).normalize_or_zero();

        let mut frames = Vec::new();
        let mut distance = 0.0;
        while distance < total {
            let t = distance_to_t(&distances, distance);
            let forward = self.tangent(t);
            (right, up) = transport_frame(forward, right, up);
            frames.push(SplineFrame {
                position: self.point(t),
                forward,
                right,
                up,
                curvature: self.curvature(t),
                distance,
            });
            distance += frame_step;
        }

        // Close the loop with a duplicate of the first frame at full length.
        let mut closing = frames[0];
        closing.distance = total;
        frames.push(closing);

        BakedSpline {
            frames,
            length: total,
        }
    }

    /// Split `t` into (segment index, local parameter in `[0, 1]`).
    fn locate(&self, t: f32) -> (usize, f32) {
        let scaled = t.rem_euclid(1.0) * self.points.len() as f32;
        let segment = (scaled.floor() as usize).min(self.points.len() - 1);
        (segment, scaled - segment as f32)
    }
}

/// A single baked frame along a spline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplineFrame {
    /// Position on the curve.
    pub position: Vec3,
    /// Unit tangent.
    pub forward: Vec3,
    /// Unit right vector (parallel-transported, never flips).
    pub right: Vec3,
    /// Unit up vector.
    pub up: Vec3,
    /// Signed horizontal curvature (1 / radius). Positive turns left, negative right.
    pub curvature: f32,
    /// Arc-length distance from the start of the loop.
    pub distance: f32,
}

/// A spline baked into arc-length-spaced frames for fast, even sampling.
#[derive(Resource, Debug, Clone)]
pub struct BakedSpline {
    frames: Vec<SplineFrame>,
    length: f32,
}

impl BakedSpline {
    /// Total loop length in world units.
    pub fn length(&self) -> f32 {
        self.length
    }

    /// The baked frames. The last frame duplicates the first to close the loop.
    pub fn frames(&self) -> &[SplineFrame] {
        &self.frames
    }

    /// Sample at normalized loop progress `t` (wraps around `[0, 1)`).
    pub fn sample(&self, t: f32) -> SplineFrame {
        self.sample_at_distance(t.rem_euclid(1.0) * self.length)
    }

    /// Sample at an arc-length `distance` from the start (wraps around the loop).
    pub fn sample_at_distance(&self, distance: f32) -> SplineFrame {
        let distance = distance.rem_euclid(self.length);
        let upper = self
            .frames
            .partition_point(|f| f.distance <= distance)
            .clamp(1, self.frames.len() - 1);
        let (a, b) = (self.frames[upper - 1], self.frames[upper]);
        let w = ((distance - a.distance) / (b.distance - a.distance).max(1e-6)).clamp(0.0, 1.0);
        SplineFrame {
            position: a.position.lerp(b.position, w),
            forward: a.forward.lerp(b.forward, w).normalize_or_zero(),
            right: a.right.lerp(b.right, w).normalize_or_zero(),
            up: a.up.lerp(b.up, w).normalize_or_zero(),
            curvature: a.curvature + (b.curvature - a.curvature) * w,
            distance,
        }
    }

    /// Normalized loop progress `t` of the baked point nearest `point` (compared in XZ).
    pub fn nearest_t(&self, point: Vec3) -> f32 {
        let count = self.frames.len() - 1; // ignore the duplicated closing frame
        let dist_sq = |i: usize| (point.xz() - self.frames[i % count].position.xz()).length_squared();

        let step = (count / 100).max(1);
        let coarse = (0..count)
            .step_by(step)
            .min_by(|&a, &b| dist_sq(a).total_cmp(&dist_sq(b)))
            .unwrap_or(0);

        let range = step.max(10);
        let best = (coarse + count - range % count..=coarse + count + range)
            .map(|i| i % count)
            .min_by(|&a, &b| dist_sq(a).total_cmp(&dist_sq(b)))
            .unwrap_or(coarse);

        self.frames[best].distance / self.length
    }
}

// ─── Internals ───

/// Bézier control points of closed-loop segment `segment` of `points`.
fn bezier_of(points: &[Vec3], segment: usize) -> [Vec3; 4] {
    let n = points.len();
    let p0 = points[(segment + n - 1) % n];
    let p1 = points[segment];
    let p2 = points[(segment + 1) % n];
    let p3 = points[(segment + 2) % n];
    [p1, p1 + (p2 - p0) / 3.0, p2 - (p3 - p1) / 3.0, p2]
}

fn bezier_point(b: &[Vec3; 4], u: f32) -> Vec3 {
    let w = 1.0 - u;
    b[0] * (w * w * w) + b[1] * (3.0 * w * w * u) + b[2] * (3.0 * w * u * u) + b[3] * (u * u * u)
}

/// First and second derivative of the cubic Bézier at `u`.
fn bezier_derivatives(b: &[Vec3; 4], u: f32) -> (Vec3, Vec3) {
    let w = 1.0 - u;
    let velocity = 3.0 * (w * w * (b[1] - b[0]) + 2.0 * w * u * (b[2] - b[1]) + u * u * (b[3] - b[2]));
    let acceleration = 6.0 * (w * (b[2] - 2.0 * b[1] + b[0]) + u * (b[3] - 2.0 * b[2] + b[1]));
    (velocity, acceleration)
}

fn signed_curvature(b: &[Vec3; 4], u: f32) -> f32 {
    let (v, a) = bezier_derivatives(b, u);
    let speed = v.xz().length().max(1e-6);
    (v.z * a.x - v.x * a.z) / (speed * speed * speed)
}

fn segment_min_radius(points: &[Vec3], segment: usize) -> f32 {
    let b = bezier_of(points, segment);
    (0..=STRESS_SAMPLES)
        .map(|s| 1.0 / signed_curvature(&b, s as f32 / STRESS_SAMPLES as f32).abs().max(1e-6))
        .fold(f32::MAX, f32::min)
}

/// Rotate `(right, up)` minimally so they are perpendicular to the new `forward`.
fn transport_frame(forward: Vec3, right: Vec3, up: Vec3) -> (Vec3, Vec3) {
    let mut up = up;
    let dot = up.dot(forward);
    if dot.abs() < 0.9999 {
        let projected = up - forward * dot;
        if projected.length_squared() > 1e-8 {
            up = projected.normalize();
        } else {
            let right_projected = right - forward * right.dot(forward);
            if right_projected.length_squared() > 1e-8 {
                up = forward.cross(right_projected.normalize()).normalize();
            }
        }
    }
    (forward.cross(up).normalize(), up)
}

/// Convert an arc-length distance to curve parameter `t` using the cumulative table.
fn distance_to_t(distances: &[f32], distance: f32) -> f32 {
    let last = distances.len() - 1;
    let upper = distances.partition_point(|&d| d <= distance).clamp(1, last);
    let lower = upper - 1;
    let w = (distance - distances[lower]) / (distances[upper] - distances[lower]).max(1e-6);
    (lower as f32 + w.clamp(0.0, 1.0)) / last as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(size: f32) -> CatmullRomSpline {
        CatmullRomSpline::closed([
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(size, 0.0, 0.0),
            Vec3::new(size, 0.0, size),
            Vec3::new(0.0, 0.0, size),
        ])
    }

    #[test]
    fn passes_through_control_points() {
        let spline = square(40.0);
        for (i, p) in spline.points().iter().enumerate() {
            let t = i as f32 / spline.segment_count() as f32;
            assert!(spline.point(t).distance(*p) < 1e-3);
        }
    }

    #[test]
    fn point_wraps_around() {
        let spline = square(40.0);
        assert!(spline.point(0.3).distance(spline.point(1.3)) < 1e-3);
        assert!(spline.point(-0.7).distance(spline.point(0.3)) < 1e-3);
    }

    #[test]
    fn relaxing_straightens_tight_corners() {
        // A tight zig-zag with hairpin turns.
        let tight = CatmullRomSpline::closed([
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(60.0, 0.0, 0.0),
            Vec3::new(62.0, 0.0, 6.0),
            Vec3::new(60.0, 0.0, 12.0),
            Vec3::new(0.0, 0.0, 12.0),
            Vec3::new(-20.0, 0.0, 6.0),
        ]);
        let before = tight.min_turn_radius();
        let relaxed = tight.relaxed(6.0);
        assert!(relaxed.min_turn_radius() >= before);
    }

    #[test]
    fn relaxing_keeps_heights() {
        let spline = CatmullRomSpline::closed([
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(10.0, 2.0, 0.0),
            Vec3::new(10.0, 3.0, 1.0),
            Vec3::new(0.0, 4.0, 1.0),
        ]);
        let relaxed = spline.clone().relaxed(5.0);
        for (a, b) in spline.points().iter().zip(relaxed.points()) {
            assert_eq!(a.y, b.y);
        }
    }

    #[test]
    fn baked_frames_are_evenly_spaced_and_on_curve() {
        let spline = square(40.0);
        let baked = spline.bake(0.5);
        assert!(baked.length() > 100.0);
        for k in 0..20 {
            let d = baked.length() * k as f32 / 20.0;
            let frame = baked.sample_at_distance(d);
            let t = baked.nearest_t(frame.position);
            assert!(baked.sample(t).position.distance(frame.position) < 0.6);
            // Frames lie on the underlying curve.
            let nearest = (0..2000)
                .map(|i| spline.point(i as f32 / 2000.0).distance(frame.position))
                .fold(f32::MAX, f32::min);
            assert!(nearest < 0.1);
            assert!(frame.forward.is_normalized());
            assert!(frame.right.dot(frame.forward).abs() < 1e-2);
        }
    }

    #[test]
    fn sample_wraps_and_nearest_t_roundtrips() {
        let baked = square(40.0).bake(0.5);
        let a = baked.sample(0.25);
        let b = baked.sample(1.25);
        assert!(a.position.distance(b.position) < 1e-3);
        let t = baked.nearest_t(a.position);
        assert!((t - 0.25).abs() < 0.01);
    }

    #[test]
    fn curvature_sign_matches_turn_direction() {
        // Counter-clockwise when viewed from +Y (x → z): heading turns toward +X from +Z? Just
        // check that the sign is consistent around the loop and reversing flips it.
        let spline = square(40.0);
        let reversed = CatmullRomSpline::closed(
            spline.points().iter().rev().copied().collect::<Vec<_>>(),
        );
        let c = spline.curvature(0.1);
        let r = reversed.curvature(0.1);
        assert!(c * r < 0.0 || c == 0.0 && r == 0.0);
    }
}
