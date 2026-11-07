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
//!   changing a few floats — no new mesh, no mesh cache entry;
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
use bevy::prelude::Rectangle;
use bevy::mesh::Mesh;
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
    /// `xy` gradient direction, `z` colour mode (0 single, 1 linear), `w` mix-space code.
    pub gradient: Vec4,
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
    /// (default: bottom to top, `+Y`).
    #[must_use]
    pub fn with_colors(mut self, from: Color, to: Color) -> Self {
        self.uniform.color = srgb_vec4(from);
        self.uniform.secondary_color = srgb_vec4(to);
        self.uniform.gradient.z = 1.0;
        self
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
}
