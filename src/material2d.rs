//! 2D gradient material for `Mesh2d` / `Camera2d` scenes.
//!
//! [`GradMeshMaterial2d`] is a `Material2d` that evaluates the same
//! dual-directional and dual-radial gradients, colour mix spaces and dash
//! patterns as [`GradMeshMaterial`](crate::GradMeshMaterial), unlit and with alpha
//! blending. Pair it with [`Shape::build_2d`](crate::Shape::build_2d):
//!
//! ```ignore
//! commands.spawn((
//!     Mesh2d(meshes.add(Shape::disc(1.0, ShapeDetail::Medium).build_2d())),
//!     MeshMaterial2d(materials.add(style.build_2d())),
//!     Transform::from_xyz(0.0, 0.0, 5.0).with_scale(Vec3::new(80.0, 80.0, 1.0)),
//! ));
//! ```
//!
//! * `Transform` scale and rotation apply on the XY plane; gradients and dashes
//!   are computed from the mesh UVs, so they stretch with the shape.
//! * Draw order follows `Transform.translation.z`.
//! * [`GradMeshPlugin`](crate::GradMeshPlugin) registers the material
//!   automatically when Bevy's 2D mesh rendering is present; otherwise add
//!   [`GradMesh2dPlugin`] yourself.

use bevy::app::{App, Plugin};
use bevy::asset::{Asset, AssetPath};
use bevy::reflect::TypePath;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;
use bevy::sprite_render::{AlphaMode2d, Material2d, Material2dPlugin};

use crate::appearance::AppearanceSnapshot;
use crate::material::{GradMeshUniformBuffer, Vec4};
use crate::style::Style;

/// Unlit gradient `Material2d`.
///
/// Gradients are evaluated in the mesh's UV frame: for
/// [`Style::dual_directional`] the direction's X and Y components point from
/// the dark side to the lit side with +Y towards the top of the shape, so the
/// direction space is always local to the shape.
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct GradMeshMaterial2d {
    /// Colours, direction, flags and dash parameters (same layout as the 3D material).
    #[uniform(0)]
    pub uniforms: GradMeshUniformBuffer,
    /// Appearance animation: `x` speed, `y` phase, `z` type code.
    #[uniform(1)]
    pub appearance: Vec4,
    /// Whether the material is drawn with alpha blending.
    pub translucent: bool,
}

impl GradMeshMaterial2d {
    /// Creates an opaque white material animated by `appearance`.
    ///
    /// Use [`Style::build_2d`] for gradients and colours, then
    /// [`GradMeshMaterial2d::with_appearance`] to animate them.
    pub fn new(appearance: AppearanceSnapshot) -> Self {
        Style::opaque(bevy::color::Color::WHITE)
            .build_2d()
            .with_appearance(appearance)
    }

    /// Replaces the appearance animation (colour pulse, colour cycle or
    /// gradient shift). The animation clock is Bevy's render time.
    #[must_use]
    pub fn with_appearance(mut self, appearance: AppearanceSnapshot) -> Self {
        self.appearance = appearance.to_vec4();
        self
    }

    /// The appearance animation currently set.
    pub fn appearance_snapshot(&self) -> AppearanceSnapshot {
        AppearanceSnapshot::from_vec4(self.appearance)
    }
}

impl Style {
    /// Builds a [`GradMeshMaterial2d`] from this style.
    ///
    /// Colours, gradient mode, mix space, alpha and dashes carry over. PBR fields
    /// (roughness, metallic, emissive) have no effect on the unlit 2D material.
    pub fn build_2d(&self) -> GradMeshMaterial2d {
        GradMeshMaterial2d {
            uniforms: self.build_uniforms(),
            appearance: AppearanceSnapshot::static_appearance().to_vec4(),
            translucent: self.is_translucent(),
        }
    }
}

fn gradmesh2d_shader_ref() -> ShaderRef {
    ShaderRef::Path(
        AssetPath::from_path_buf(bevy::asset::embedded_path!("shader/gradmesh2d.wesl"))
            .with_source("embedded"),
    )
}

impl Material2d for GradMeshMaterial2d {
    fn fragment_shader() -> ShaderRef {
        gradmesh2d_shader_ref()
    }

    fn alpha_mode(&self) -> AlphaMode2d {
        if self.translucent {
            AlphaMode2d::Blend
        } else {
            AlphaMode2d::Opaque
        }
    }
}

/// Registers [`GradMeshMaterial2d`].
///
/// [`GradMeshPlugin`](crate::GradMeshPlugin) adds this automatically when
/// Bevy's 2D mesh renderer is present. Add it manually for apps that add
/// `GradMeshPlugin` before the render plugins or that only use 2D. It is safe to
/// add alongside `GradMeshPlugin`.
pub struct GradMesh2dPlugin;

impl Plugin for GradMesh2dPlugin {
    fn build(&self, app: &mut App) {
        if app.is_plugin_added::<Material2dPlugin<GradMeshMaterial2d>>() {
            return;
        }
        bevy::asset::embedded_asset!(app, "shader/gradmesh2d.wesl");
        app.add_plugins(Material2dPlugin::<GradMeshMaterial2d>::default());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::color::Color;
    use bevy::math::Vec3;

    #[test]
    fn style_translates_to_2d_material() {
        let style = Style::dual_directional(Color::BLACK, Color::WHITE, Vec3::X);
        let m = style.build_2d();
        assert!(!m.translucent);
        assert_eq!(m.uniforms.flags.x, 1.0);
        assert_eq!(m.uniforms.direction.x, 1.0);
        assert!(matches!(m.alpha_mode(), AlphaMode2d::Opaque));
    }

    #[test]
    fn translucent_style_blends() {
        let m = Style::shaded(Color::WHITE, 0.3).with_alpha(0.5).build_2d();
        assert!(m.translucent);
        assert!(matches!(m.alpha_mode(), AlphaMode2d::Blend));
        assert!((m.uniforms.color.w - 0.5).abs() < 1e-6);
        assert!((m.uniforms.secondary_color.w - 0.5).abs() < 1e-6);
    }

    #[test]
    fn appearance_round_trips() {
        let a = AppearanceSnapshot::color_pulse(2.0, 0.25);
        let m = GradMeshMaterial2d::new(a);
        assert_eq!(m.appearance_snapshot(), a);
    }
}
