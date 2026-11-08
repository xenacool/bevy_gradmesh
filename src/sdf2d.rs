//! Signed-distance-field 2D shapes drawn on one shared quad.
//!
//! [`SdfMaterial2d`] is a `Material2d` whose fragment shader evaluates the shape
//! (rounded rectangle or ellipse, optionally a rim and/or a partial arc) from
//! parameters in a uniform buffer. Every shape uses the *same* unit quad mesh
//! ([`unit_quad`]); the quad is scaled to the shape's size by `Transform`.
//!
//! Compared with building a mesh per shape ([`Shape::build_2d`](crate::Shape::build_2d)):
//!
//! * edges are anti-aliased from screen-space derivatives, so no MSAA is needed;
//! * radius, rim width, corner radius and arc span can be animated every frame by
//!   changing a few floats - no new mesh, no mesh cache entry;
//! * every shape shares one mesh asset.
//!
//! ```ignore
//! let quad = meshes.add(unit_quad());
//! let material = SdfMaterial2d::ellipse(Vec2::splat(120.0))
//!     .with_rim(10.0)
//!     .with_arc(0.0, std::f32::consts::PI)
//!     .with_colors(Color::srgb(1.0, 0.3, 0.2), Color::srgb(1.0, 0.9, 0.3));
//! commands.spawn((
//!     Mesh2d(quad.clone()),
//!     Transform::from_scale(material.quad_scale()),
//!     MeshMaterial2d(materials.add(material)),
//! ));
//! ```
//!
//! The material is always alpha-blended. `Transform` scale **must** be
//! [`SdfMaterial2d::quad_scale`]; scale the shape through the material instead.

use bevy::app::{App, Plugin};
use bevy::asset::{Asset, AssetPath};
use bevy::color::Color;
use bevy::math::{Vec2, Vec3, Vec4};
use bevy::mesh::Mesh;
use bevy::prelude::Rectangle;
use bevy::reflect::TypePath;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;
use bevy::sprite_render::{AlphaMode2d, Material2d, Material2dPlugin};

use crate::mix_space::MixSpace;

/// Which signed distance function the shader evaluates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SdfKind {
    /// Rectangle with optional rounded corners.
    #[default]
    Rect,
    /// Ellipse (a circle when both radii are equal).
    Ellipse,
}

impl SdfKind {
    const fn gpu_code(self) -> f32 {
        match self {
            Self::Rect => 0.0,
            Self::Ellipse => 1.0,
        }
    }
}

/// Uniform buffer layout read by `sdf2d.wesl`.
#[derive(Debug, Clone, Copy, PartialEq, ShaderType)]
pub struct SdfUniform {
    /// Primary colour, sRGB components + alpha.
    pub color: Vec4,
    /// Secondary colour, sRGB components + alpha.
    pub secondary_color: Vec4,
    /// `x` kind code, `y` rim width (0 = filled), `z` arc start (rad), `w` arc span (rad).
    pub shape: Vec4,
    /// `xy` shape size, `z` corner radius, `w` quad padding per side.
    pub size: Vec4,
    /// `xy` gradient direction (`x` is the start angle for angular modes), `z` colour
    /// mode (see [`SdfGradient`]), `w` mix-space code.
    pub gradient: Vec4,
    /// `x` dash length (> 0) or minus the dash count (< 0), `0` for no dashes;
    /// `y` gap, `z` offset, `w` snap code (see [`Dash2d`]).
    pub dash: Vec4,
}

/// How a [`SdfMaterial2d`] colours its pixels.
///
/// The numeric codes are the values of `SdfUniform::gradient.z`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdfGradient {
    /// Linear between the two colours along the gradient direction.
    Linear,
    /// Radial, fitted to the shape's bounds: the second colour is reached at the
    /// edge midpoints of the bounding rectangle (the outline of an ellipse).
    Radial,
    /// Angular and mirrored: the first colour at the start angle, the second
    /// half a turn later, and back again, so there is no seam.
    Angular,
    /// Angular sweep: one full turn from the first to the second colour, with a
    /// hard seam at the start angle.
    AngularSweep,
}

impl SdfGradient {
    const fn gpu_code(self) -> f32 {
        match self {
            Self::Linear => 1.0,
            Self::Radial => 2.0,
            Self::Angular => 3.0,
            Self::AngularSweep => 4.0,
        }
    }
}

/// How a dash pattern is fitted to the outline it runs along.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DashSnap {
    /// Use the pattern as given. A closed outline is cut where it starts and
    /// ends, so the last dash may be partial.
    Off,
    /// Round the number of periods to a whole number and rescale the period, so
    /// the pattern repeats cleanly with no seam.
    #[default]
    Tiling,
    /// Like [`DashSnap::Tiling`] on closed outlines. On a partial arc the first
    /// dash starts at the arc start and the last dash ends at the arc end.
    EndToEnd,
}

impl DashSnap {
    const fn gpu_code(self) -> f32 {
        match self {
            Self::Off => 0.0,
            Self::Tiling => 1.0,
            Self::EndToEnd => 2.0,
        }
    }
}

/// A dash pattern for the rim of a [`SdfMaterial2d`] (see [`SdfMaterial2d::with_dash`]).
///
/// Dashes are measured along the outline: counter-clockwise from +X, in the
/// shape's world units. For circles and rounded rectangles the length is exact;
/// on an eccentric ellipse the dashes are spaced evenly by angle, so they
/// stretch where the outline is flatter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dash2d {
    /// Dash length in world units. `0` means the pattern is given by `count`.
    pub length: f32,
    /// Number of dashes, used when `length` is `0`.
    pub count: u32,
    /// Gap between dashes: world units in length mode, or a `0..1` share of one
    /// period in count mode.
    pub gap: f32,
    /// Shifts the pattern along the outline, as a share (`0..1`) of one period.
    pub offset: f32,
    /// How the pattern is fitted to the outline.
    pub snap: DashSnap,
}

impl Dash2d {
    /// `count` dashes around the outline, separated by gaps of `gap` (a `0..1`
    /// share of each period).
    #[must_use]
    pub fn by_count(count: u32, gap: f32) -> Self {
        Self {
            length: 0.0,
            count,
            gap,
            offset: 0.0,
            snap: DashSnap::Tiling,
        }
    }

    /// Dashes `length` long separated by gaps of `gap`, both in world units.
    #[must_use]
    pub fn by_length(length: f32, gap: f32) -> Self {
        Self {
            length,
            count: 0,
            gap,
            offset: 0.0,
            snap: DashSnap::Tiling,
        }
    }

    /// Sets the snap mode.
    #[must_use]
    pub fn with_snap(mut self, snap: DashSnap) -> Self {
        self.snap = snap;
        self
    }

    /// Shifts the pattern by `offset` periods.
    #[must_use]
    pub fn with_offset(mut self, offset: f32) -> Self {
        self.offset = offset;
        self
    }

    /// Whether the pattern draws any dashes.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.length > 0.0 || self.count > 0
    }

    /// `(period, dash length)` once the pattern is fitted to `span` world units of
    /// outline (`is_arc` for a partial arc, where `span` is the arc length).
    ///
    /// The shader runs the same maths (`dash_pattern` in `sdf2d.wesl`); this copy
    /// lets the snapping rules be tested without a GPU.
    #[must_use]
    pub fn pattern(&self, span: f32, is_arc: bool) -> (f32, f32) {
        if self.length <= 0.0 {
            let n = self.count.max(1) as f32;
            let r = self.gap.clamp(0.0, 0.95);
            let period = if is_arc && self.snap == DashSnap::EndToEnd {
                span / (n - r)
            } else {
                span / n
            };
            return (period, period * (1.0 - r));
        }
        let (len, gap) = (self.length, self.gap.max(0.0));
        let t0 = len + gap;
        match self.snap {
            DashSnap::Off => (t0, len),
            DashSnap::EndToEnd if is_arc => {
                let n = ((span + gap) / t0).round().max(1.0);
                let k = span / (n * len + (n - 1.0) * gap);
                (t0 * k, len * k)
            }
            _ => {
                let n = (span / t0).round().max(1.0);
                let k = span / (n * t0);
                (t0 * k, len * k)
            }
        }
    }

    /// The period after snapping; see [`Dash2d::pattern`].
    #[must_use]
    pub fn snapped_period(&self, span: f32, is_arc: bool) -> f32 {
        self.pattern(span, is_arc).0
    }

    fn gpu_value(&self) -> Vec4 {
        let x = if self.length > 0.0 {
            self.length
        } else {
            -(self.count as f32)
        };
        Vec4::new(x, self.gap.max(0.0), self.offset, self.snap.gpu_code())
    }
}

/// Default quad padding per side, in world units (pixels with the default `Camera2d`).
const DEFAULT_PADDING: f32 = 2.0;

/// A 2D shape evaluated per pixel from a signed distance field.
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct SdfMaterial2d {
    /// Shape, colour and gradient parameters.
    #[uniform(0)]
    pub uniform: SdfUniform,
}

impl SdfMaterial2d {
    fn new(kind: SdfKind, size: Vec2) -> Self {
        Self {
            uniform: SdfUniform {
                color: Vec4::ONE,
                secondary_color: Vec4::ONE,
                shape: Vec4::new(kind.gpu_code(), 0.0, 0.0, std::f32::consts::TAU),
                size: Vec4::new(size.x, size.y, 0.0, DEFAULT_PADDING),
                gradient: Vec4::new(0.0, 1.0, 0.0, MixSpace::default().gpu_code()),
                dash: Vec4::ZERO,
            },
        }
    }

    /// A filled rectangle of the given size.
    #[must_use]
    pub fn rect(size: Vec2) -> Self {
        Self::new(SdfKind::Rect, size)
    }

    /// A filled ellipse with the given radii.
    #[must_use]
    pub fn ellipse(radii: Vec2) -> Self {
        Self::new(SdfKind::Ellipse, radii * 2.0)
    }

    /// A filled circle with the given radius.
    #[must_use]
    pub fn disc(radius: f32) -> Self {
        Self::ellipse(Vec2::splat(radius))
    }

    /// Sets the corner radius of a rectangle (clamped to half the shortest side).
    #[must_use]
    pub fn with_corner_radius(mut self, radius: f32) -> Self {
        self.uniform.size.z = radius.max(0.0);
        self
    }

    /// Turns the shape into a rim (hollow outline) of the given width, drawn
    /// inside the shape's outline. `0.0` restores the filled shape.
    #[must_use]
    pub fn with_rim(mut self, width: f32) -> Self {
        self.uniform.shape.y = width.max(0.0);
        self
    }

    /// Clips the shape to the angular sector starting at `start` radians
    /// (counter-clockwise from +X) and covering `span` radians. A span of
    /// `TAU` or more draws the full shape.
    #[must_use]
    pub fn with_arc(mut self, start: f32, span: f32) -> Self {
        self.uniform.shape.z = start;
        self.uniform.shape.w = span.clamp(0.0, std::f32::consts::TAU);
        self
    }

    /// Sets a single flat colour.
    #[must_use]
    pub fn with_color(mut self, color: Color) -> Self {
        let c = srgb_vec4(color);
        self.uniform.color = c;
        self.uniform.secondary_color = c;
        self.uniform.gradient.z = 0.0;
        self
    }

    /// Sets a two-colour gradient along the current gradient direction
    /// (default: bottom to top, `+Y`). An already chosen radial or angular mode
    /// is kept, so the colours may be set before or after it.
    #[must_use]
    pub fn with_colors(mut self, from: Color, to: Color) -> Self {
        self.uniform.color = srgb_vec4(from);
        self.uniform.secondary_color = srgb_vec4(to);
        if self.uniform.gradient.z < 0.5 {
            self.uniform.gradient.z = SdfGradient::Linear.gpu_code();
        }
        self
    }

    /// Fills the shape with a radial gradient fitted to its bounds: the first
    /// colour at the centre, the second at the edge midpoints (clamped beyond, so
    /// the corners of a rectangle stay at the second colour). Blends in the
    /// current [`MixSpace`].
    #[must_use]
    pub fn with_radial_gradient(mut self) -> Self {
        self.uniform.gradient.z = SdfGradient::Radial.gpu_code();
        self
    }

    /// Fills the shape with an angular gradient around its centre, starting at
    /// `start` radians (counter-clockwise from +X). The gradient is mirrored: the
    /// second colour is reached half a turn later and it fades back, so there is
    /// no seam. Use [`SdfMaterial2d::with_angular_sweep`] for a one-way sweep.
    #[must_use]
    pub fn with_angular_gradient(mut self, start: f32) -> Self {
        self.uniform.gradient.x = start;
        self.uniform.gradient.z = SdfGradient::Angular.gpu_code();
        self
    }

    /// Like [`SdfMaterial2d::with_angular_gradient`], but one turn runs from the
    /// first to the second colour with a hard seam at `start`.
    #[must_use]
    pub fn with_angular_sweep(mut self, start: f32) -> Self {
        self.uniform.gradient.x = start;
        self.uniform.gradient.z = SdfGradient::AngularSweep.gpu_code();
        self
    }

    /// Dashes the rim. Has no effect on a filled shape: call
    /// [`SdfMaterial2d::with_rim`] too. Dashes follow the outline of discs,
    /// ellipses and rounded rectangles and respect partial arcs. A pattern with
    /// neither `length` nor `count` draws a solid rim.
    #[must_use]
    pub fn with_dash(mut self, dash: Dash2d) -> Self {
        self.uniform.dash = if dash.is_enabled() {
            dash.gpu_value()
        } else {
            Vec4::ZERO
        };
        self
    }

    /// Length of the full outline in world units: the exact perimeter of a
    /// rounded rectangle, or Ramanujan's approximation for an ellipse.
    #[must_use]
    pub fn perimeter(&self) -> f32 {
        let size = self.size();
        if self.uniform.shape.x == SdfKind::Ellipse.gpu_code() {
            let (a, b) = (size.x * 0.5, size.y * 0.5);
            let h = (a - b) * (a - b) / ((a + b) * (a + b)).max(1e-5);
            return std::f32::consts::PI * (a + b) * (1.0 + 3.0 * h / (10.0 + (4.0 - 3.0 * h).sqrt()));
        }
        let r = self.uniform.size.z.min(size.x.min(size.y) * 0.5);
        2.0 * (size.x + size.y) - 8.0 * r + std::f32::consts::TAU * r
    }

    /// Sets the gradient direction in the shape's frame (+Y up).
    #[must_use]
    pub fn with_gradient_direction(mut self, direction: Vec2) -> Self {
        self.uniform.gradient.x = direction.x;
        self.uniform.gradient.y = direction.y;
        self
    }

    /// Sets the colour space the gradient is mixed in.
    #[must_use]
    pub fn with_mix_space(mut self, space: MixSpace) -> Self {
        self.uniform.gradient.w = space.gpu_code();
        self
    }

    /// Sets how far the quad extends past the shape on each side, so the anti-aliased
    /// fringe is not clipped. Defaults to 2 world units.
    #[must_use]
    pub fn with_padding(mut self, padding: f32) -> Self {
        self.uniform.size.w = padding.max(0.0);
        self
    }

    /// The shape size in world units (without padding).
    #[must_use]
    pub fn size(&self) -> Vec2 {
        Vec2::new(self.uniform.size.x, self.uniform.size.y)
    }

    /// The `Transform` scale to give the shared [`unit_quad`] so it covers the
    /// shape plus its padding.
    #[must_use]
    pub fn quad_scale(&self) -> Vec3 {
        let pad = self.uniform.size.w * 2.0;
        Vec3::new(self.uniform.size.x + pad, self.uniform.size.y + pad, 1.0)
    }

    /// Resizes the shape. Keep `Transform` scale in sync via [`Self::quad_scale`].
    pub fn set_size(&mut self, size: Vec2) {
        self.uniform.size.x = size.x;
        self.uniform.size.y = size.y;
    }

    /// Blends this material towards `other` by `t` (clamped to `0.0..=1.0`;
    /// `t == 0.0` is a copy of `self` and `t == 1.0` a copy of `other`).
    ///
    /// * Size, corner radius, rim width, arc, padding, the gradient direction
    ///   and the dash length, gap and offset blend linearly. The two colours
    ///   mix in `self`'s [`MixSpace`], like [`Style::lerp`](crate::Style::lerp).
    /// * A circle blends into a rectangle through its corner radius (a circle
    ///   is a rectangle whose corners take up the whole side), so the outline
    ///   never jumps. Any other pairing of a rectangle and an ellipse cannot
    ///   be blended and switches at `t == 0.5`.
    /// * The gradient mode, the mix space, the dash snapping, and whether
    ///   there are dashes at all or how they are given (by length or by count)
    ///   switch at `t == 0.5`. A count of dashes stays a whole number.
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let t = if t.is_nan() { 0.0 } else { t.clamp(0.0, 1.0) };
        if t <= 0.0 {
            return self.clone();
        }
        if t >= 1.0 {
            return other.clone();
        }
        let (mut a, mut b) = (self.uniform, other.uniform);
        let near = if t < 0.5 { a } else { b };
        let mix = |x: f32, y: f32| x + (y - x) * t;
        let is_rect = |u: &SdfUniform| u.shape.x < 0.5;
        let is_circle = |u: &SdfUniform| !is_rect(u) && (u.size.x - u.size.y).abs() < 1e-6;
        let as_rect = |mut u: SdfUniform| {
            u.shape.x = SdfKind::Rect.gpu_code();
            u.size.z = u.size.x * 0.5;
            u
        };
        if is_circle(&a) && is_rect(&b) {
            a = as_rect(a);
        } else if is_circle(&b) && is_rect(&a) {
            b = as_rect(b);
        }
        let kind = if is_rect(&a) == is_rect(&b) { a.shape.x } else { near.shape.x };

        let space = MixSpace::from_gpu_code(self.uniform.gradient.w);
        let color_of = |v: Vec4| Color::srgba(v.x, v.y, v.z, v.w);
        let mixed = |x: Vec4, y: Vec4| srgb_vec4(space.mix(color_of(x), color_of(y), t));

        let mut gradient = Vec4::new(
            mix(a.gradient.x, b.gradient.x),
            mix(a.gradient.y, b.gradient.y),
            near.gradient.z,
            near.gradient.w,
        );
        if a.gradient.z == b.gradient.z && a.gradient.z == SdfGradient::Linear.gpu_code() {
            let d = Vec2::new(gradient.x, gradient.y).normalize_or(Vec2::new(near.gradient.x, near.gradient.y));
            gradient.x = d.x;
            gradient.y = d.y;
        }

        let dashes = |u: &SdfUniform| u.dash.x != 0.0;
        let dash = if dashes(&a) && dashes(&b) && (a.dash.x > 0.0) == (b.dash.x > 0.0) {
            let x = mix(a.dash.x, b.dash.x);
            let x = if x < 0.0 { x.round().min(-1.0) } else { x };
            Vec4::new(x, mix(a.dash.y, b.dash.y), mix(a.dash.z, b.dash.z), near.dash.w)
        } else {
            near.dash
        };

        Self {
            uniform: SdfUniform {
                color: mixed(a.color, b.color),
                secondary_color: mixed(a.secondary_color, b.secondary_color),
                shape: Vec4::new(kind, mix(a.shape.y, b.shape.y), mix(a.shape.z, b.shape.z), mix(a.shape.w, b.shape.w)),
                size: Vec4::new(
                    mix(a.size.x, b.size.x),
                    mix(a.size.y, b.size.y),
                    mix(a.size.z, b.size.z),
                    mix(a.size.w, b.size.w),
                ),
                gradient,
                dash,
            },
        }
    }
}

fn srgb_vec4(color: Color) -> Vec4 {
    let c = color.to_srgba();
    Vec4::new(c.red, c.green, c.blue, c.alpha)
}

/// The one quad mesh shared by every [`SdfMaterial2d`] entity (1×1, centred).
#[must_use]
pub fn unit_quad() -> Mesh {
    Mesh::from(Rectangle::new(1.0, 1.0))
}

impl Material2d for SdfMaterial2d {
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Path(
            AssetPath::from_path_buf(bevy::asset::embedded_path!("shader/sdf2d.wesl"))
                .with_source("embedded"),
        )
    }

    fn alpha_mode(&self) -> AlphaMode2d {
        AlphaMode2d::Blend
    }
}

/// Registers [`SdfMaterial2d`]. [`GradMeshPlugin`](crate::GradMeshPlugin) adds it
/// automatically when Bevy's 2D mesh renderer is present; it is safe to add twice.
pub struct SdfMaterial2dPlugin;

impl Plugin for SdfMaterial2dPlugin {
    fn build(&self, app: &mut App) {
        if app.is_plugin_added::<Material2dPlugin<SdfMaterial2d>>() {
            return;
        }
        bevy::asset::embedded_asset!(app, "shader/sdf2d.wesl");
        app.add_plugins(Material2dPlugin::<SdfMaterial2d>::default());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disc_is_full_ellipse_with_diameter_size() {
        let m = SdfMaterial2d::disc(50.0);
        assert_eq!(m.size(), Vec2::splat(100.0));
        assert_eq!(m.uniform.shape.x, SdfKind::Ellipse.gpu_code());
        assert!((m.uniform.shape.w - std::f32::consts::TAU).abs() < 1e-6);
    }

    #[test]
    fn quad_scale_includes_padding_on_both_sides() {
        let m = SdfMaterial2d::rect(Vec2::new(100.0, 40.0)).with_padding(3.0);
        assert_eq!(m.quad_scale(), Vec3::new(106.0, 46.0, 1.0));
    }

    #[test]
    fn builders_clamp_their_inputs() {
        let m = SdfMaterial2d::rect(Vec2::splat(10.0))
            .with_rim(-4.0)
            .with_corner_radius(-1.0)
            .with_arc(1.0, 100.0);
        assert_eq!(m.uniform.shape.y, 0.0);
        assert_eq!(m.uniform.size.z, 0.0);
        assert!((m.uniform.shape.w - std::f32::consts::TAU).abs() < 1e-6);
        assert_eq!(m.uniform.shape.z, 1.0);
    }

    #[test]
    fn colors_set_gradient_mode() {
        let flat = SdfMaterial2d::disc(1.0).with_color(Color::WHITE);
        assert_eq!(flat.uniform.gradient.z, 0.0);
        let grad = flat.with_colors(Color::BLACK, Color::WHITE);
        assert_eq!(grad.uniform.gradient.z, 1.0);
        assert_eq!(grad.uniform.color, Vec4::new(0.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn gradient_modes_have_their_codes() {
        let base = SdfMaterial2d::disc(1.0).with_colors(Color::BLACK, Color::WHITE);
        assert_eq!(base.clone().with_radial_gradient().uniform.gradient.z, 2.0);
        let angular = base.clone().with_angular_gradient(0.5);
        assert_eq!(angular.uniform.gradient.z, 3.0);
        assert_eq!(angular.uniform.gradient.x, 0.5);
        let sweep = base.with_angular_sweep(1.5);
        assert_eq!(sweep.uniform.gradient.z, 4.0);
        assert_eq!(sweep.uniform.gradient.x, 1.5);
    }

    #[test]
    fn colors_keep_a_chosen_gradient_mode() {
        let m = SdfMaterial2d::disc(1.0)
            .with_radial_gradient()
            .with_colors(Color::BLACK, Color::WHITE);
        assert_eq!(m.uniform.gradient.z, 2.0);
        let flat = m.with_color(Color::WHITE);
        assert_eq!(flat.uniform.gradient.z, 0.0);
    }

    #[test]
    fn mix_space_applies_to_the_new_gradients() {
        let m = SdfMaterial2d::disc(1.0)
            .with_radial_gradient()
            .with_mix_space(MixSpace::Oklch);
        assert_eq!(m.uniform.gradient.w, MixSpace::Oklch.gpu_code());
    }

    #[test]
    fn dash_uniform_packing() {
        let by_length = Dash2d::by_length(12.0, 5.0)
            .with_offset(0.25)
            .with_snap(DashSnap::EndToEnd);
        let m = SdfMaterial2d::disc(50.0).with_rim(4.0).with_dash(by_length);
        assert_eq!(m.uniform.dash, Vec4::new(12.0, 5.0, 0.25, 2.0));

        let by_count = SdfMaterial2d::disc(50.0).with_dash(Dash2d::by_count(8, 0.3));
        assert_eq!(by_count.uniform.dash, Vec4::new(-8.0, 0.3, 0.0, 1.0));

        let none = SdfMaterial2d::disc(50.0)
            .with_dash(by_length)
            .with_dash(Dash2d::by_count(0, 0.5));
        assert_eq!(none.uniform.dash, Vec4::ZERO);
    }

    #[test]
    fn tiling_rounds_the_period_count() {
        let d = Dash2d::by_length(10.0, 5.0).with_snap(DashSnap::Tiling);
        // 100 / 15 = 6.67 -> 7 periods of 14.2857.
        let (period, dash) = d.pattern(100.0, false);
        assert!((period * 7.0 - 100.0).abs() < 1e-3);
        assert!((dash / period - 10.0 / 15.0).abs() < 1e-5);
        // Tiny shapes keep at least one period.
        let (period, _) = d.pattern(3.0, false);
        assert!((period - 3.0).abs() < 1e-5);
    }

    #[test]
    fn off_keeps_the_given_pattern() {
        let d = Dash2d::by_length(10.0, 5.0).with_snap(DashSnap::Off);
        assert_eq!(d.pattern(100.0, false), (15.0, 10.0));
        assert_eq!(d.snapped_period(100.0, true), 15.0);
    }

    #[test]
    fn end_to_end_fits_dashes_to_an_arc() {
        let d = Dash2d::by_length(10.0, 5.0).with_snap(DashSnap::EndToEnd);
        for span in [37.0_f32, 100.0, 251.3, 8.0] {
            let (period, dash) = d.pattern(span, true);
            let gap = period - dash;
            let n = ((span + gap) / period).round();
            // n dashes and n - 1 gaps fill the arc exactly.
            assert!((n * dash + (n - 1.0) * gap - span).abs() < 1e-3, "span {span}");
        }
        // On a closed outline it tiles like `Tiling`.
        let tiled = Dash2d::by_length(10.0, 5.0);
        assert_eq!(d.pattern(100.0, false), tiled.pattern(100.0, false));
    }

    #[test]
    fn count_mode_divides_the_span() {
        let d = Dash2d::by_count(10, 0.25);
        let (period, dash) = d.pattern(200.0, false);
        assert!((period - 20.0).abs() < 1e-4);
        assert!((dash - 15.0).abs() < 1e-4);
        // End to end on an arc: 10 dashes and 9 gaps.
        let e = d.with_snap(DashSnap::EndToEnd);
        let (period, dash) = e.pattern(200.0, true);
        assert!((10.0 * dash + 9.0 * (period - dash) - 200.0).abs() < 1e-3);
    }

    #[test]
    fn empty_dash_is_disabled() {
        assert!(!Dash2d::by_count(0, 0.5).is_enabled());
        assert!(!Dash2d::by_length(0.0, 3.0).is_enabled());
        assert!(Dash2d::by_count(1, 0.0).is_enabled());
    }

    #[test]
    fn perimeter_matches_known_shapes() {
        let circle = SdfMaterial2d::disc(10.0).perimeter();
        assert!((circle - std::f32::consts::TAU * 10.0).abs() < 1e-3);
        let rect = SdfMaterial2d::rect(Vec2::new(40.0, 20.0)).perimeter();
        assert!((rect - 120.0).abs() < 1e-4);
        // A rectangle with the corner radius at half the short side is a stadium.
        let stadium = SdfMaterial2d::rect(Vec2::new(40.0, 20.0))
            .with_corner_radius(10.0)
            .perimeter();
        assert!((stadium - (40.0 + std::f32::consts::TAU * 10.0)).abs() < 1e-3);
    }

    #[test]
    fn lerp_blends_continuous_values_and_keeps_endpoints() {
        let a = SdfMaterial2d::rect(Vec2::new(40.0, 20.0))
            .with_corner_radius(2.0)
            .with_rim(1.0)
            .with_padding(2.0)
            .with_color(Color::srgb(1.0, 0.0, 0.0));
        let b = SdfMaterial2d::rect(Vec2::new(80.0, 40.0))
            .with_corner_radius(6.0)
            .with_rim(5.0)
            .with_padding(4.0)
            .with_color(Color::srgb(0.0, 0.0, 1.0));
        assert_eq!(a.lerp(&b, 0.0).uniform, a.uniform);
        assert_eq!(a.lerp(&b, 1.0).uniform, b.uniform);
        assert_eq!(a.lerp(&b, f32::NAN).uniform, a.uniform);
        let m = a.lerp(&b, 0.5).uniform;
        assert_eq!(m.size, Vec4::new(60.0, 30.0, 4.0, 3.0));
        assert_eq!(m.shape.y, 3.0);
        assert!(m.color.x > 0.0 && m.color.x < 1.0 && m.color.z > 0.0 && m.color.z < 1.0);
    }

    #[test]
    fn a_circle_becomes_a_rectangle_through_its_corner_radius() {
        let circle = SdfMaterial2d::disc(10.0);
        let rect = SdfMaterial2d::rect(Vec2::new(20.0, 20.0));
        let m = circle.lerp(&rect, 0.25).uniform;
        assert_eq!(m.shape.x, SdfKind::Rect.gpu_code());
        // Corners start fully round (radius 10) and shrink to square.
        assert!((m.size.z - 7.5).abs() < 1e-4, "{}", m.size.z);
        let back = rect.lerp(&circle, 0.75).uniform;
        assert!((back.size.z - 7.5).abs() < 1e-4);
    }

    #[test]
    fn kinds_that_cannot_blend_switch_halfway() {
        let ellipse = SdfMaterial2d::ellipse(Vec2::new(20.0, 10.0));
        let rect = SdfMaterial2d::rect(Vec2::new(40.0, 20.0));
        assert_eq!(ellipse.lerp(&rect, 0.4).uniform.shape.x, SdfKind::Ellipse.gpu_code());
        assert_eq!(ellipse.lerp(&rect, 0.6).uniform.shape.x, SdfKind::Rect.gpu_code());
    }

    #[test]
    fn dash_counts_stay_whole_and_dash_modes_switch() {
        let ring = |dash: Dash2d| SdfMaterial2d::disc(10.0).with_rim(2.0).with_dash(dash);
        let a = ring(Dash2d::by_count(10, 0.2));
        let b = ring(Dash2d::by_count(15, 0.4));
        let m = a.lerp(&b, 0.2).uniform.dash;
        assert_eq!(m.x, -11.0);
        assert!((m.y - 0.24).abs() < 1e-5);
        // Count against length cannot blend: the nearer pattern wins.
        let c = ring(Dash2d::by_length(3.0, 1.0));
        assert_eq!(a.lerp(&c, 0.4).uniform.dash, a.uniform.dash);
        assert_eq!(a.lerp(&c, 0.6).uniform.dash, c.uniform.dash);
        // No dashes against dashes: the nearer one wins too.
        let none = SdfMaterial2d::disc(10.0).with_rim(2.0);
        assert_eq!(none.lerp(&a, 0.4).uniform.dash, Vec4::ZERO);
    }

    #[test]
    fn opposite_gradient_directions_do_not_collapse() {
        let up = SdfMaterial2d::rect(Vec2::splat(10.0)).with_colors(Color::BLACK, Color::WHITE);
        let down = up.clone().with_gradient_direction(Vec2::NEG_Y);
        let g = up.lerp(&down, 0.5).uniform.gradient;
        assert!((Vec2::new(g.x, g.y).length() - 1.0).abs() < 1e-5);
    }
}
