//! Catmull-Rom splines, closed or open, with scalar channels, stress relaxation
//! and arc-length baking.
//!
//! A [`CatmullRomSpline`] is a curve through a list of control points, either a
//! closed loop ([`CatmullRomSpline::closed`]) or an open path
//! ([`CatmullRomSpline::open`]). Every segment is internally a cubic Bézier, so
//! the curve is C1-continuous and passes through each control point. Open paths
//! have one-sided tangents at their two ends.
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
//! // One width per control point, interpolated like the position.
//! .with_channel(&[6.0, 8.0, 6.0, 8.0])
//! // Pull over-stressed (too tight) corners straight until every turn has r >= 10.
//! .relaxed(10.0);
//!
//! // Bake once for cheap, evenly spaced lookups.
//! let baked = track.bake(0.5);
//! let frame = baked.sample_at_distance(baked.length() * 0.25);
//! assert!(frame.forward.is_normalized());
//! let width = baked.channel_at_distance(0, baked.length() * 0.25).unwrap();
//! assert!((6.0..=8.0).contains(&width));
//! ```
//!
//! # Channels
//!
//! [`CatmullRomSpline::with_channel`] attaches one number per control point
//! (width, roll, tilt, a colour component...). A channel is interpolated with
//! the same Catmull-Rom weights as the position and follows the same wrap rule
//! (it wraps around a closed loop and has one-sided ends on an open path).
//! [`CatmullRomSpline::bake`] samples every channel at every frame.
//!
//! # Stress
//!
//! The "stress" of a segment is how tightly it bends: the smaller the horizontal turn
//! radius, the higher the stress. [`CatmullRomSpline::min_turn_radius`] measures it and
//! [`CatmullRomSpline::relaxed`] straightens stressed spans (in the XZ plane; heights are
//! left alone) by pulling their control points toward their neighbours. Relaxing never
//! collapses the curve: see [`CatmullRomSpline::relaxed_within`].

use std::ops::{Add, Mul, Range, Sub};

use bevy::ecs::prelude::Resource;
use bevy::math::{Vec2, Vec3, Vec3Swizzles};

use crate::shape::SweepFrame;

/// Samples used per segment when measuring turn radius.
const STRESS_SAMPLES: usize = 32;
/// Maximum relaxation passes in [`CatmullRomSpline::relaxed`].
const MAX_RELAX_ITERATIONS: usize = 2000;
/// Fraction of the distance to the neighbour midpoint moved per relaxation pass.
const RELAX_STRENGTH: f32 = 0.2;
/// Dense samples per curve segment used to build the arc-length table when
/// baking, so long and intricate splines are measured as finely as short ones.
const BAKE_SAMPLES_PER_SEGMENT: usize = 128;

/// The share of its length [`CatmullRomSpline::relaxed`] keeps at least:
/// relaxing a curve that is far too sharp stops there instead of shrinking it
/// to a point. Use [`CatmullRomSpline::relaxed_within`] to ask for another
/// share.
pub const RELAXED_KEEP: f32 = 0.5;

/// A Catmull-Rom spline through a list of control points, closed or open, with
/// optional scalar channels.
#[derive(Debug, Clone, PartialEq)]
pub struct CatmullRomSpline {
    points: Vec<Vec3>,
    closed: bool,
    channels: Vec<Vec<f32>>,
}

/// What [`CatmullRomSpline::relaxed_within`] returns when no curve that keeps
/// enough of its length is gentle enough: the gentlest one it found.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Relaxed {
    /// The gentlest curve found that keeps the requested share of the length.
    pub best: CatmullRomSpline,
    /// Its smallest horizontal turn radius ([`CatmullRomSpline::min_turn_radius`]).
    pub radius: f32,
}

impl std::fmt::Display for Relaxed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "no curve keeps enough of its length and is gentle enough; the best has a turn radius of {}",
            self.radius
        )
    }
}

impl std::error::Error for Relaxed {}

impl CatmullRomSpline {
    /// Create a closed spline through `points`. The last point connects back to the first.
    ///
    /// # Panics
    /// Panics if fewer than 3 points are given.
    pub fn closed(points: impl Into<Vec<Vec3>>) -> Self {
        let points = points.into();
        assert!(points.len() >= 3, "a closed spline needs at least 3 points");
        Self {
            points,
            closed: true,
            channels: Vec::new(),
        }
    }

    /// Create an open spline from the first to the last of `points`. The end
    /// tangents are one-sided (they point along the first and last chord), and
    /// [`CatmullRomSpline::bake`] puts a frame exactly on the last point.
    ///
    /// # Panics
    /// Panics if fewer than 2 points are given.
    pub fn open(points: impl Into<Vec<Vec3>>) -> Self {
        let points = points.into();
        assert!(points.len() >= 2, "an open spline needs at least 2 points");
        Self {
            points,
            closed: false,
            channels: Vec::new(),
        }
    }

    /// Adds a scalar channel with one value per control point. See the module
    /// documentation. Channels are numbered in the order they were added.
    ///
    /// # Panics
    /// Panics if `values` does not have one entry per control point.
    #[must_use]
    pub fn with_channel(mut self, values: &[f32]) -> Self {
        assert_eq!(
            values.len(),
            self.points.len(),
            "a channel needs one value per control point"
        );
        self.channels.push(values.to_vec());
        self
    }

    /// Whether the curve is a closed loop.
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// The control points the curve passes through.
    pub fn points(&self) -> &[Vec3] {
        &self.points
    }

    /// Number of channels.
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// The control-point values of channel `index`.
    pub fn channel_values(&self, index: usize) -> Option<&[f32]> {
        self.channels.get(index).map(Vec::as_slice)
    }

    /// Number of curve segments: one per control point when closed, one fewer
    /// when open.
    pub fn segment_count(&self) -> usize {
        if self.closed {
            self.points.len()
        } else {
            self.points.len() - 1
        }
    }

    /// The cubic Bézier control points `[b0, b1, b2, b3]` of segment `index`
    /// (taken modulo [`CatmullRomSpline::segment_count`]).
    pub fn bezier_segment(&self, index: usize) -> [Vec3; 4] {
        bezier_of(&self.points, index % self.segment_count(), self.closed)
    }

    /// Evaluate the curve at `t`. On a closed loop `t` wraps around `[0, 1)`
    /// over the whole loop; on an open path it is clamped to `[0, 1]`.
    pub fn point(&self, t: f32) -> Vec3 {
        let (segment, u) = self.locate(t);
        bezier_point(&bezier_of(&self.points, segment, self.closed), u)
    }

    /// Unit tangent of the curve at `t`.
    pub fn tangent(&self, t: f32) -> Vec3 {
        let (segment, u) = self.locate(t);
        let (velocity, _) = bezier_derivatives(&bezier_of(&self.points, segment, self.closed), u);
        velocity.normalize_or_zero()
    }

    /// Signed horizontal curvature (1 / radius) at `t`. Positive turns left, negative right.
    pub fn curvature(&self, t: f32) -> f32 {
        let (segment, u) = self.locate(t);
        signed_curvature(&bezier_of(&self.points, segment, self.closed), u)
    }

    /// Value of channel `index` at `t` (wrapping or clamping like
    /// [`CatmullRomSpline::point`]), or `None` if there is no such channel.
    pub fn channel(&self, index: usize, t: f32) -> Option<f32> {
        let values = self.channels.get(index)?;
        let (segment, u) = self.locate(t);
        Some(bezier_point(&bezier_of(values, segment, self.closed), u))
    }

    /// Smallest horizontal turn radius anywhere on the curve (`f32::MAX` if perfectly straight).
    pub fn min_turn_radius(&self) -> f32 {
        (0..self.segment_count())
            .map(|i| segment_min_radius(&self.points, i, self.closed))
            .fold(f32::MAX, f32::min)
    }

    /// Return a copy with over-stressed spans straightened so no turn is tighter than
    /// `min_radius`. Only X and Z move; heights and channels are preserved, and the
    /// end points of an open path stay where they are.
    ///
    /// Stressed control points are pulled toward the midpoint of their neighbours until the
    /// whole curve is gentle enough. The result is never tighter than the input, and it
    /// never collapses: it keeps at least [`RELAXED_KEEP`] of the length of the control
    /// polygon, so a curve that is far too sharp is relaxed as far as that allows and
    /// no further. If `min_radius` cannot be reached you get the gentlest curve found;
    /// use [`CatmullRomSpline::relaxed_within`] to learn that, or to keep more length.
    pub fn relaxed(self, min_radius: f32) -> Self {
        match self.relaxed_within(min_radius, RELAXED_KEEP) {
            Ok(curve) => curve,
            Err(relaxed) => relaxed.best,
        }
    }

    /// The gentlest curve that keeps at least the fraction `keep` of the length
    /// of the control polygon (`0.0..=1.0`; values outside are clamped, and NaN
    /// keeps everything), relaxed as [`CatmullRomSpline::relaxed`] describes.
    ///
    /// Returns `Ok` when no turn is tighter than `min_radius`. Otherwise no curve
    /// that keeps enough length reaches it, and the error carries the gentlest
    /// curve found and its radius, so a caller can reject the design or accept it.
    pub fn relaxed_within(self, min_radius: f32, keep: f32) -> Result<Self, Relaxed> {
        let keep = if keep.is_nan() { 1.0 } else { keep.clamp(0.0, 1.0) };
        let floor = self.polygon_length() * keep;
        let mut best_radius = self.min_turn_radius();
        let mut best = self.clone();
        let mut current = self;
        for _ in 0..MAX_RELAX_ITERATIONS {
            if best_radius >= min_radius {
                break;
            }
            current = current.relax_step(min_radius, RELAX_STRENGTH);
            if current.polygon_length() < floor {
                break;
            }
            let radius = current.min_turn_radius();
            if radius > best_radius {
                best_radius = radius;
                best = current.clone();
            }
        }
        if best_radius >= min_radius {
            Ok(best)
        } else {
            Err(Relaxed {
                best,
                radius: best_radius,
            })
        }
    }

    /// Length of the polygon through the control points.
    fn polygon_length(&self) -> f32 {
        let mut total: f32 = self.points.windows(2).map(|w| w[0].distance(w[1])).sum();
        if self.closed {
            total += self.points[self.points.len() - 1].distance(self.points[0]);
        }
        total
    }

    /// One relaxation pass over every segment tighter than `min_radius`.
    fn relax_step(&self, min_radius: f32, strength: f32) -> Self {
        let n = self.points.len();
        let mut next = self.clone();
        for segment in 0..self.segment_count() {
            if segment_min_radius(&self.points, segment, self.closed) >= min_radius {
                continue;
            }
            for index in [segment, (segment + 1) % n] {
                if !self.closed && (index == 0 || index == n - 1) {
                    continue;
                }
                let target = (self.points[(index + n - 1) % n] + self.points[(index + 1) % n]) * 0.5;
                let point = &mut next.points[index];
                point.x += (target.x - point.x) * strength;
                point.z += (target.z - point.z) * strength;
            }
        }
        next
    }

    /// Bake evenly spaced (by arc length) frames, `frame_step` world units apart,
    /// with the first frame's up direction taken from `+Y`. See
    /// [`CatmullRomSpline::bake_with_up`].
    ///
    /// # Panics
    /// Panics if `frame_step` is not positive, or if the spline has no length.
    pub fn bake(&self, frame_step: f32) -> BakedSpline {
        self.bake_with_up(frame_step, Vec3::Y)
    }

    /// Bake evenly spaced (by arc length) frames, `frame_step` world units apart.
    ///
    /// Frames use parallel transport starting from `up` (a direction that is
    /// parallel to the first tangent falls back to any perpendicular one), so
    /// `right`/`up` never flip at sharp turns. A closed loop ends with a
    /// duplicate of its first frame at full length; an open path ends with a
    /// frame exactly on its last point. Every channel is sampled at every frame.
    ///
    /// # Panics
    /// Panics if `frame_step` is not positive, or if the spline has no length.
    pub fn bake_with_up(&self, frame_step: f32, up: Vec3) -> BakedSpline {
        assert!(frame_step > 0.0, "frame_step must be positive");

        // Cumulative arc length over a dense, uniform-in-t sampling.
        let samples = self.segment_count() * BAKE_SAMPLES_PER_SEGMENT;
        let mut distances = Vec::with_capacity(samples + 1);
        let mut previous = self.point(0.0);
        let mut total = 0.0;
        distances.push(0.0);
        for i in 1..=samples {
            let p = self.point(i as f32 / samples as f32);
            total += (p - previous).length();
            distances.push(total);
            previous = p;
        }
        assert!(total > 0.0, "a spline needs some length to be baked");

        let first_forward = self.tangent(0.0);
        let mut right = up
            .cross(first_forward)
            .try_normalize()
            .unwrap_or_else(|| first_forward.any_orthonormal_vector());
        let mut up = first_forward.cross(right).normalize_or_zero();

        let mut frames = Vec::new();
        let mut channels: Vec<Vec<f32>> = vec![Vec::new(); self.channels.len()];
        let mut push_frame = |distance: f32, t: f32, right: &mut Vec3, up: &mut Vec3| {
            let forward = self.tangent(t);
            (*right, *up) = transport_frame(forward, *right, *up);
            frames.push(SplineFrame {
                position: self.point(t),
                forward,
                right: *right,
                up: *up,
                curvature: self.curvature(t),
                distance,
            });
            for (c, values) in channels.iter_mut().enumerate() {
                values.push(self.channel(c, t).unwrap_or(0.0));
            }
        };

        let mut distance = 0.0;
        if self.closed {
            while distance < total {
                let t = distance_to_t(&distances, distance);
                push_frame(distance, t, &mut right, &mut up);
                distance += frame_step;
            }
            // Close the loop with a duplicate of the first frame at full length.
            let mut closing = frames[0];
            closing.distance = total;
            frames.push(closing);
            for values in &mut channels {
                values.push(values[0]);
            }
        } else {
            // A frame too close to the end would nearly duplicate the last one.
            while distance < total - frame_step * 0.01 {
                let t = distance_to_t(&distances, distance);
                push_frame(distance, t, &mut right, &mut up);
                distance += frame_step;
            }
            push_frame(total, 1.0, &mut right, &mut up);
        }

        BakedSpline {
            frames,
            length: total,
            closed: self.closed,
            channels,
        }
    }

    /// Split `t` into (segment index, local parameter in `[0, 1]`).
    fn locate(&self, t: f32) -> (usize, f32) {
        let segments = self.segment_count();
        let t = if self.closed {
            t.rem_euclid(1.0)
        } else {
            t.clamp(0.0, 1.0)
        };
        let scaled = t * segments as f32;
        let segment = (scaled.floor() as usize).min(segments - 1);
        (segment, scaled - segment as f32)
    }
}

/// A single baked frame along a spline.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct SplineFrame {
    /// Position on the curve.
    pub position: Vec3,
    /// Unit tangent.
    pub forward: Vec3,
    /// Unit right vector (parallel-transported, never flips).
    pub right: Vec3,
    /// Unit up vector.
    pub up: Vec3,
    /// Signed horizontal curvature (1 / radius), measured in the XZ plane only:
    /// the height of the curve does not enter it. Positive turns left, negative
    /// right.
    pub curvature: f32,
    /// Arc-length distance from the start of the curve.
    pub distance: f32,
}

/// A spline baked into arc-length-spaced frames for fast, even sampling.
#[derive(Resource, Debug, Clone)]
pub struct BakedSpline {
    frames: Vec<SplineFrame>,
    length: f32,
    closed: bool,
    channels: Vec<Vec<f32>>,
}

impl BakedSpline {
    /// Total length in world units.
    pub fn length(&self) -> f32 {
        self.length
    }

    /// Whether the spline is a closed loop: distances and `t` wrap around when
    /// sampling, and the last frame duplicates the first. On an open path they
    /// are clamped, and the last frame lies on the last point.
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// The baked frames.
    pub fn frames(&self) -> &[SplineFrame] {
        &self.frames
    }

    /// Number of channels baked.
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Channel `index` at every baked frame, in the order of
    /// [`BakedSpline::frames`].
    pub fn channel(&self, index: usize) -> Option<&[f32]> {
        self.channels.get(index).map(Vec::as_slice)
    }

    /// Sample at normalized progress `t` (wraps around `[0, 1)` on a closed loop,
    /// clamps to `[0, 1]` on an open path).
    pub fn sample(&self, t: f32) -> SplineFrame {
        self.sample_at_distance(t * self.length)
    }

    /// Sample at an arc-length `distance` from the start (wraps around a closed
    /// loop, clamps on an open path).
    pub fn sample_at_distance(&self, distance: f32) -> SplineFrame {
        let (a, b, w, distance) = self.locate(distance);
        let (a, b) = (self.frames[a], self.frames[b]);
        SplineFrame {
            position: a.position.lerp(b.position, w),
            forward: a.forward.lerp(b.forward, w).normalize_or_zero(),
            right: a.right.lerp(b.right, w).normalize_or_zero(),
            up: a.up.lerp(b.up, w).normalize_or_zero(),
            curvature: a.curvature + (b.curvature - a.curvature) * w,
            distance,
        }
    }

    /// Value of channel `index` at an arc-length `distance`, wrapping or
    /// clamping like [`BakedSpline::sample_at_distance`]; `None` if there is
    /// no such channel.
    pub fn channel_at_distance(&self, index: usize, distance: f32) -> Option<f32> {
        let values = self.channels.get(index)?;
        let (a, b, w, _) = self.locate(distance);
        Some(values[a] + (values[b] - values[a]) * w)
    }

    /// The baked frames as [`SweepFrame`]s for
    /// [`Shape::sweep_frames`](crate::Shape::sweep_frames), so a road, kerb or
    /// rail follows the spline with its own banking and width.
    ///
    /// `width` names a channel that scales the profile across (`scale.x`) and
    /// `roll` names one that rolls it about the tangent, in radians, positive
    /// turning `right` towards `up` as [`SweepOptions::roll_along`](crate::SweepOptions::roll_along)
    /// does. `None` leaves that part alone. A closed spline gives its duplicate
    /// closing frame too, which a closed sweep drops.
    ///
    /// # Panics
    /// Panics if a named channel does not exist.
    pub fn sweep_frames(&self, width: Option<usize>, roll: Option<usize>) -> Vec<SweepFrame> {
        for (what, index) in [("width", width), ("roll", roll)] {
            if let Some(i) = index {
                assert!(i < self.channels.len(), "the {what} channel {i} does not exist");
            }
        }
        self.frames
            .iter()
            .enumerate()
            .map(|(k, f)| {
                let (mut right, mut up) = (f.right, f.up);
                if let Some(i) = roll {
                    let (s, c) = self.channels[i][k].sin_cos();
                    (right, up) = (right * c + up * s, up * c - right * s);
                }
                let sx = width.map_or(1.0, |i| self.channels[i][k]);
                SweepFrame::new(f.position, right, up).with_scale(Vec2::new(sx, 1.0))
            })
            .collect()
    }

    /// Normalized progress `t` of the baked point nearest `point`, in 3D.
    ///
    /// Heights count, so a point on a road below a bridge matches the road and
    /// not the bridge, unless the bridge is closer. This searches the whole
    /// curve; to follow a moving object across a road that crosses itself
    /// within a few metres, use [`BakedSpline::nearest_distance_near`].
    pub fn nearest_t(&self, point: Vec3) -> f32 {
        let count = self.frames.len() - usize::from(self.closed); // ignore the duplicated closing frame
        let dist_sq = |i: usize| (point - self.frames[i % count].position).length_squared();

        let step = (count / 100).max(1);
        let coarse = (0..count)
            .step_by(step)
            .min_by(|&a, &b| dist_sq(a).total_cmp(&dist_sq(b)))
            .unwrap_or(0);

        let spacing = self.length / (self.frames.len() - 1).max(1) as f32;
        let window = step.max(10) as f32 * spacing;
        self.nearest_distance_near(point, self.frames[coarse].distance, window) / self.length
    }

    /// Normalized progress `t` of the point of the curve nearest `point` (in 3D)
    /// among those with `t` in `range`. On a closed loop `range` may run
    /// past `1.0` or below `0.0` and wraps; on an open path it is clamped to
    /// `0.0..=1.0`. A reversed range is read the other way round.
    pub fn nearest_t_in(&self, point: Vec3, range: Range<f32>) -> f32 {
        self.nearest_between(point, range.start * self.length, range.end * self.length) / self.length
    }

    /// Arc-length distance of the point of the curve nearest `point` (in 3D)
    /// within `window` of the distance `around`, on either side: a search
    /// that stays on the right part of a road that crosses over itself. Feed it
    /// the previous answer to follow a kart. On a closed loop the window wraps.
    pub fn nearest_distance_near(&self, point: Vec3, around: f32, window: f32) -> f32 {
        let window = window.abs();
        self.nearest_between(point, around - window, around + window)
    }

    /// Nearest point in 3D among those between arc-length distances `from` and
    /// `to` (unwrapped on a closed loop), refined between frames.
    fn nearest_between(&self, point: Vec3, from: f32, to: f32) -> f32 {
        let p = point;
        let len = self.length;
        let (from, to) = if to < from { (to, from) } else { (from, to) };
        let (from, to) = if self.closed {
            if to - from >= len { (0.0, len) } else { (from, to) }
        } else {
            (from.clamp(0.0, len), to.clamp(0.0, len))
        };
        let laps = if self.closed {
            (from / len).floor() as i32..=(to / len).floor() as i32
        } else {
            0..=0
        };
        let mut best = (f32::MAX, from);
        for lap in laps {
            let shift = lap as f32 * len;
            let (lo, hi) = ((from - shift).max(0.0), (to - shift).min(len));
            if lo > hi {
                continue;
            }
            let first = self
                .frames
                .partition_point(|f| f.distance <= lo)
                .saturating_sub(1)
                .min(self.frames.len() - 2);
            for k in first..self.frames.len() - 1 {
                let (a, b) = (&self.frames[k], &self.frames[k + 1]);
                if a.distance > hi {
                    break;
                }
                let (d0, d1) = (a.distance.max(lo), b.distance.min(hi));
                if d0 > d1 {
                    continue;
                }
                let span = (b.distance - a.distance).max(1e-6);
                let at = |d: f32| a.position.lerp(b.position, (d - a.distance) / span);
                let (pa, pb) = (at(d0), at(d1));
                let ab = pb - pa;
                let w = if ab.length_squared() > 1e-12 {
                    ((p - pa).dot(ab) / ab.length_squared()).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let d2 = (p - (pa + ab * w)).length_squared();
                if d2 < best.0 {
                    best = (d2, d0 + (d1 - d0) * w + shift);
                }
            }
        }
        if self.closed {
            best.1.rem_euclid(len)
        } else {
            best.1
        }
    }

    /// The frames on either side of `distance`, the blend weight between them
    /// and the distance after wrapping or clamping.
    fn locate(&self, distance: f32) -> (usize, usize, f32, f32) {
        let distance = if self.closed {
            distance.rem_euclid(self.length)
        } else {
            distance.clamp(0.0, self.length)
        };
        let upper = self
            .frames
            .partition_point(|f| f.distance <= distance)
            .clamp(1, self.frames.len() - 1);
        let (a, b) = (&self.frames[upper - 1], &self.frames[upper]);
        let w = ((distance - a.distance) / (b.distance - a.distance).max(1e-6)).clamp(0.0, 1.0);
        (upper - 1, upper, w, distance)
    }
}

// ─── Internals ───

/// Bézier control points of segment `segment` of `points` (a closed loop, or an
/// open path whose end tangents are one-sided: the missing neighbour is the
/// reflection of the next point).
fn bezier_of<T>(points: &[T], segment: usize, closed: bool) -> [T; 4]
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<f32, Output = T>,
{
    let n = points.len();
    let at = |i: isize| -> T {
        if closed {
            points[i.rem_euclid(n as isize) as usize]
        } else if i < 0 {
            points[0] * 2.0 - points[1]
        } else if i >= n as isize {
            points[n - 1] * 2.0 - points[n - 2]
        } else {
            points[i as usize]
        }
    };
    let s = segment as isize;
    let (p0, p1, p2, p3) = (at(s - 1), at(s), at(s + 1), at(s + 2));
    // Catmull-Rom tangents are (p2 - p0) / 2 and (p3 - p1) / 2 per unit of
    // parameter; a cubic Bézier's end tangent is three times its control arm.
    [p1, p1 + (p2 - p0) * (1.0 / 6.0), p2 - (p3 - p1) * (1.0 / 6.0), p2]
}

fn bezier_point<T>(b: &[T; 4], u: f32) -> T
where
    T: Copy + Add<Output = T> + Mul<f32, Output = T>,
{
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

fn segment_min_radius(points: &[Vec3], segment: usize, closed: bool) -> f32 {
    let b = bezier_of(points, segment, closed);
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
    fn evenly_spaced_collinear_points_are_traversed_at_a_constant_speed() {
        // A Catmull-Rom spline reproduces a straight line exactly: a tangent that
        // is too long makes the curve overshoot and bend between the points.
        let points: Vec<Vec3> = (0..6).map(|i| Vec3::new(i as f32 * 2.0, 0.0, 0.0)).collect();
        let spline = CatmullRomSpline::open(points);
        for i in 0..=50 {
            let t = i as f32 / 50.0;
            let p = spline.point(t);
            assert!((p.x - t * 10.0).abs() < 1e-4, "x({t}) = {}", p.x);
            assert!(p.z.abs() < 1e-6);
        }
    }

    #[test]
    fn a_sampled_circle_keeps_close_to_its_radius() {
        let (n, radius) = (12, 5.0_f32);
        let spline = CatmullRomSpline::closed(
            (0..n)
                .map(|i| {
                    let a = i as f32 / n as f32 * std::f32::consts::TAU;
                    Vec3::new(a.cos() * radius, 0.0, a.sin() * radius)
                })
                .collect::<Vec<_>>(),
        );
        let min = spline.min_turn_radius();
        // The curve ripples between its control points, so allow some slack.
        assert!((radius * 0.75..radius * 1.25).contains(&min), "min radius {min}");
        for i in 0..200 {
            let r = spline.point(i as f32 / 200.0).xz().length();
            assert!((r - radius).abs() < 0.05 * radius, "radius {r}");
        }
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

#[cfg(test)]
mod open_and_channel_tests {
    use super::*;

    fn arc() -> CatmullRomSpline {
        CatmullRomSpline::open([
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 4.0),
            Vec3::new(20.0, 0.0, 0.0),
            Vec3::new(30.0, 0.0, 6.0),
        ])
    }

    /// A long thin loop: the strip along z = 0 and the strip along z = 2.
    fn strips() -> CatmullRomSpline {
        CatmullRomSpline::closed([
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(33.0, 0.0, 0.0),
            Vec3::new(66.0, 0.0, 0.0),
            Vec3::new(100.0, 0.0, 0.0),
            Vec3::new(100.0, 0.0, 2.0),
            Vec3::new(66.0, 0.0, 2.0),
            Vec3::new(33.0, 0.0, 2.0),
            Vec3::new(0.0, 0.0, 2.0),
        ])
    }

    #[test]
    fn an_open_spline_runs_from_its_first_to_its_last_point() {
        let s = arc();
        assert!(!s.is_closed());
        assert_eq!(s.segment_count(), 3);
        assert!(s.point(0.0).distance(s.points()[0]) < 1e-4);
        assert!(s.point(1.0).distance(s.points()[3]) < 1e-4);
        // t is clamped, not wrapped.
        assert!(s.point(1.7).distance(s.points()[3]) < 1e-4);
        assert!(s.point(-3.0).distance(s.points()[0]) < 1e-4);
        // One-sided end tangents point along the first and last chord.
        assert!(s.tangent(0.0).dot((s.points()[1] - s.points()[0]).normalize()) > 0.999);
        assert!(s.tangent(1.0).dot((s.points()[3] - s.points()[2]).normalize()) > 0.999);
    }

    #[test]
    fn baking_an_open_spline_puts_a_frame_on_the_last_point() {
        let s = arc();
        let baked = s.bake(1.0);
        assert!(!baked.is_closed());
        let last = baked.frames().last().unwrap();
        assert!(last.position.distance(s.points()[3]) < 1e-3);
        assert!((last.distance - baked.length()).abs() < 1e-4);
        assert!(baked.frames()[0].position.distance(s.points()[0]) < 1e-4);
        // Distances clamp instead of wrapping.
        assert!(baked.sample_at_distance(baked.length() + 50.0).position.distance(s.points()[3]) < 1e-3);
        assert!(baked.sample_at_distance(-5.0).position.distance(s.points()[0]) < 1e-3);
        assert!(baked.sample(2.0).position.distance(s.points()[3]) < 1e-3);
        // Frames are spaced `frame_step` apart, except possibly the last one.
        for w in baked.frames().windows(2).take(baked.frames().len() - 2) {
            assert!((w[1].distance - w[0].distance - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn channels_follow_the_curve_and_the_wrap_rule() {
        let closed = strips().with_channel(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        for i in 0..8 {
            let t = i as f32 / 8.0;
            assert!((closed.channel(0, t).unwrap() - (i + 1) as f32).abs() < 1e-4);
        }
        // Wraps like the position.
        assert!((closed.channel(0, 0.3).unwrap() - closed.channel(0, 1.3).unwrap()).abs() < 1e-4);
        assert!(closed.channel(1, 0.0).is_none());
        assert_eq!(closed.channel_count(), 1);
        let open = arc().with_channel(&[0.0, 1.0, 0.0, 1.0]);
        assert!((open.channel(0, 0.0).unwrap()).abs() < 1e-4);
        assert!((open.channel(0, 1.0).unwrap() - 1.0).abs() < 1e-4);
        assert!((open.channel(0, 5.0).unwrap() - 1.0).abs() < 1e-4);
    }

    #[test]
    #[should_panic(expected = "one value per control point")]
    fn a_channel_needs_one_value_per_point() {
        let _ = arc().with_channel(&[1.0, 2.0]);
    }

    #[test]
    fn baking_samples_every_channel_at_every_frame() {
        let baked = arc().with_channel(&[0.0, 1.0, 0.0, 1.0]).with_channel(&[5.0; 4]).bake(2.0);
        assert_eq!(baked.channel_count(), 2);
        assert_eq!(baked.channel(0).unwrap().len(), baked.frames().len());
        assert!(baked.channel(2).is_none());
        assert!(baked.channel(1).unwrap().iter().all(|v| (v - 5.0).abs() < 1e-4));
        let f = baked.frames()[3];
        assert!((baked.channel_at_distance(0, f.distance).unwrap() - baked.channel(0).unwrap()[3]).abs() < 1e-5);
        assert!(baked.channel_at_distance(9, 0.0).is_none());
        // A closed bake closes its channels too.
        let closed = strips().with_channel(&[1.0; 8]).bake(3.0);
        assert_eq!(closed.channel(0).unwrap().len(), closed.frames().len());
    }

    #[test]
    fn the_windowed_search_stays_on_its_part_of_the_road() {
        let baked = strips().bake(0.5);
        // Closer to the z = 2 strip than to the z = 0 one.
        let p = Vec3::new(50.0, 0.0, 1.6);
        let global = baked.sample(baked.nearest_t(p)).position;
        assert!((global.z - 2.0).abs() < 0.8, "{global:?}");
        let near = baked.nearest_distance_near(p, 50.0, 10.0);
        let on_a = baked.sample_at_distance(near).position;
        assert!(on_a.z < 0.8 && (on_a.x - 50.0).abs() < 2.0, "{on_a:?}");
        // The same through a range of t.
        let t = baked.nearest_t_in(p, 0.0..0.4);
        assert!(baked.sample(t).position.z < 0.8);
        // A reversed range is the same range.
        assert_eq!(baked.nearest_t_in(p, 0.4..0.0), t);
    }

    #[test]
    fn nearest_queries_count_height_so_a_road_is_not_matched_to_the_bridge_above() {
        // The strip along z = 0 is a road, the one along z = 2 a bridge 6 m above it.
        let bridge = CatmullRomSpline::closed([
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(33.0, 0.0, 0.0),
            Vec3::new(66.0, 0.0, 0.0),
            Vec3::new(100.0, 0.0, 0.0),
            Vec3::new(100.0, 6.0, 2.0),
            Vec3::new(66.0, 6.0, 2.0),
            Vec3::new(33.0, 6.0, 2.0),
            Vec3::new(0.0, 6.0, 2.0),
        ])
        .bake(0.5);
        // In XZ this point is nearer the bridge (z = 2); in 3D it is on the road.
        let on_the_road = Vec3::new(50.0, 0.1, 1.2);
        let found = bridge.sample(bridge.nearest_t(on_the_road)).position;
        assert!(found.y < 1.0, "matched the bridge: {found:?}");
        let on_the_bridge = Vec3::new(50.0, 5.9, 1.2);
        let found = bridge.sample(bridge.nearest_t(on_the_bridge)).position;
        assert!(found.y > 5.0, "matched the road: {found:?}");
    }

    #[test]
    fn the_window_wraps_around_a_closed_loop_and_clamps_on_an_open_path() {
        let baked = strips().bake(0.5);
        let start = Vec3::new(1.0, 0.0, 0.0);
        let d = baked.nearest_distance_near(start, -3.0, 10.0);
        assert!(baked.sample_at_distance(d).position.distance(start) < 0.8);
        let p = Vec3::new(29.0, 0.0, 5.0);
        let open = arc().bake(0.5);
        let d = open.nearest_distance_near(p, open.length() - 1.0, 100.0);
        assert!(d <= open.length() + 1e-4);
        assert!(open.sample_at_distance(d).position.distance(Vec3::new(30.0, 0.0, 6.0)) < 3.0);
    }

    #[test]
    fn relaxing_does_not_collapse_a_loop() {
        // Far too sharp for a 100 m radius.
        let tiny = CatmullRomSpline::closed([
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 4.0),
            Vec3::new(0.0, 0.0, 4.0),
        ]);
        let before = tiny.polygon_length();
        let relaxed = tiny.clone().relaxed(100.0);
        assert!(relaxed.polygon_length() >= before * RELAXED_KEEP - 1e-4);
        assert!(relaxed.min_turn_radius() >= tiny.min_turn_radius());
        let err = tiny.clone().relaxed_within(100.0, 0.8).unwrap_err();
        assert!(err.best.polygon_length() >= before * 0.8 - 1e-4);
        assert!((err.best.min_turn_radius() - err.radius).abs() < 1e-4);
        assert!(err.radius < 100.0);
        assert!(!err.to_string().is_empty());
        // Keeping everything changes nothing.
        let same = tiny.clone().relaxed_within(100.0, 1.0).unwrap_err();
        assert_eq!(same.best, tiny);
        // A curve that is gentle enough comes back as `Ok`.
        let gentle = CatmullRomSpline::closed([
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(400.0, 0.0, 0.0),
            Vec3::new(400.0, 0.0, 400.0),
            Vec3::new(0.0, 0.0, 400.0),
        ]);
        assert!(gentle.clone().relaxed_within(10.0, 0.9).is_ok());
    }

    #[test]
    fn relaxing_an_open_path_keeps_its_ends_and_channels() {
        let zigzag = CatmullRomSpline::open([
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 2.0),
            Vec3::new(10.0, 0.0, 2.0),
        ])
        .with_channel(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let relaxed = zigzag.clone().relaxed(3.0);
        assert_eq!(relaxed.points()[0], zigzag.points()[0]);
        assert_eq!(relaxed.points()[5], zigzag.points()[5]);
        assert_eq!(relaxed.channel_values(0), zigzag.channel_values(0));
        assert!(relaxed.min_turn_radius() >= zigzag.min_turn_radius());
    }

    #[test]
    fn baked_width_and_roll_become_sweep_frames() {
        let baked = arc()
            .with_channel(&[1.0, 2.0, 3.0, 4.0])
            .with_channel(&[std::f32::consts::FRAC_PI_2; 4])
            .bake(1.0);
        let plain = baked.sweep_frames(None, None);
        let both = baked.sweep_frames(Some(0), Some(1));
        assert_eq!(both.len(), baked.frames().len());
        for ((p, b), f) in plain.iter().zip(&both).zip(baked.frames()) {
            assert!(p.right.distance(f.right) < 1e-5 && p.up.distance(f.up) < 1e-5);
            assert_eq!(p.scale, Vec2::ONE);
            // A quarter turn of roll: the new `right` is the old `up`.
            assert!(b.right.distance(f.up) < 1e-5);
            assert!(b.up.distance(-f.right) < 1e-5);
            assert_eq!(b.position, f.position);
        }
        assert!((both[0].scale.x - 1.0).abs() < 1e-4);
        assert!((both.last().unwrap().scale.x - 4.0).abs() < 1e-3);
        // And they sweep into a real road.
        let road = crate::Shape::sweep_frames(
            vec![Vec2::new(-1.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(1.0, 0.1), Vec2::new(-1.0, 0.1)],
            &baked.sweep_frames(Some(0), None),
            crate::SweepOptions::default(),
        );
        assert!(road.bounds().size().x > 25.0);
    }

    #[test]
    #[should_panic(expected = "does not exist")]
    fn naming_a_missing_channel_is_a_mistake() {
        let _ = arc().bake(1.0).sweep_frames(Some(0), None);
    }

    #[test]
    fn baking_from_another_up_direction_starts_there() {
        let line = CatmullRomSpline::open([Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0)]);
        let baked = line.bake_with_up(1.0, Vec3::Z);
        assert!(baked.frames()[0].up.dot(Vec3::Z) > 0.99);
        // Parallel to the path: still a valid frame.
        let odd = line.bake_with_up(1.0, Vec3::X);
        assert!(odd.frames()[0].right.is_normalized() && odd.frames()[0].up.is_normalized());
    }
}
