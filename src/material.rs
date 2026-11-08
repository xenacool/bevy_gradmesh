//! BevyGradmesh custom shader material for dual-directional gradient shading.
//!
//! Extends Bevy's `StandardMaterial` with gradient shading modes:
//! single color, dual directional, and dual radial.
//!
//! Uses Bevy 0.20's `ExtendedMaterial` pipeline with a `.wesl` shader template
//! that imports Bevy's PBR functions and overrides the base color with a
//! gradient-computed color before lighting.

use bevy::asset::Asset;
use bevy::math::Vec3;
use bevy::reflect::Reflect;
use bevy::render::render_resource::{AsBindGroup, ShaderType};

#[cfg(feature = "pbr")]
use crate::mix_space::MixSpace;

/// 16-byte aligned vec4 for GPU uniform buffers.
///
/// `encase` requires uniform struct fields to have stride ≥ 16 bytes,
/// so `[f32; 4]` cannot be used directly with `#[uniform()]`. This named-field
/// struct derives `ShaderType`, which ensures each instance is exactly 16 bytes
/// with proper alignment.
#[derive(Debug, Clone, Copy, Default, PartialEq, Reflect, ShaderType)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }
    pub const fn as_array(&self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }
    pub const fn from_array(arr: [f32; 4]) -> Self {
        Self {
            x: arr[0],
            y: arr[1],
            z: arr[2],
            w: arr[3],
        }
    }
}

/// Shader uniform data for the GradMesh material extension.
///
/// This is a CPU-side descriptor; the actual GPU uniform is `GradMeshExtension`.
#[derive(Debug, Clone, Copy, Default, Reflect)]
pub struct GradMeshUniform {
    /// Primary color (dark/shadow side for dual_directional).
    pub color: [f32; 4],
    /// Secondary color (light/highlight side for dual_directional).
    pub secondary_color: [f32; 4],
    /// Gradient direction (world space, X, Y, Z, padding).
    pub direction: [f32; 4],
    /// [color_mode, blend_mode, emission_strength, padding]
    pub flags: [f32; 4],
}

impl GradMeshUniform {
    /// Create a new uniform with all fields set.
    pub fn new(
        color: [f32; 4],
        secondary_color: [f32; 4],
        direction: [f32; 4],
        flags: [f32; 4],
    ) -> Self {
        Self {
            color,
            secondary_color,
            direction,
            flags,
        }
    }
    /// Get the color mode.
    #[inline]
    pub fn color_mode(&self) -> f32 {
        self.flags[0]
    }
    /// Get the blend mode.
    #[inline]
    pub fn blend_mode(&self) -> f32 {
        self.flags[1]
    }
    /// Get the emission strength.
    #[inline]
    pub fn emission_strength(&self) -> f32 {
        self.flags[2]
    }
}

/// GPU uniform buffer for the GradMesh material extension.
///
/// All fields are `Vec4` (16 bytes each via `ShaderType`; 15 in total), so the total struct
/// is 240 bytes. This single struct is bound at slot 100, avoiding conflicts
/// with the base StandardMaterial's bindings (0–99).
#[derive(Debug, Clone, Copy, Default, Reflect, ShaderType)]
pub struct GradMeshUniformBuffer {
    pub color: Vec4,
    pub secondary_color: Vec4,
    pub direction: Vec4,
    pub flags: Vec4,
    pub dash: Vec4,
    /// World-space center of the shape, used for object-space gradient computation.
    /// Stored as (x, y, z, 0) - the shape's translation in world coordinates.
    pub world_center: Vec4,
    /// Procedural noise: `(scale, amount, 0, 0)`; `amount == 0` disables it.
    pub noise: Vec4,
    /// Procedural stripes: `(direction.xyz, frequency)`.
    pub stripes: Vec4,
    /// Stripe strength: `(amount, 0, 0, 0)`; `amount == 0` disables stripes.
    pub stripes_amount: Vec4,
    /// Rim light: `(linear rgb, power)`; `power == 0` disables it.
    pub rim: Vec4,
    /// Height-band colours (linear rgba), lowest band first. Used when the
    /// colour mode is `3`; unused slots repeat the last band.
    pub bands: [Vec4; 4],
    /// Height each band starts at; unused slots hold `f32::MAX`.
    pub band_heights: Vec4,
}

impl GradMeshUniformBuffer {
    pub fn new(
        color: Vec4,
        secondary_color: Vec4,
        direction: Vec4,
        flags: Vec4,
        dash: Vec4,
    ) -> Self {
        Self {
            color,
            secondary_color,
            direction,
            flags,
            dash,
            world_center: Vec4::default(),
            noise: Vec4::default(),
            stripes: Vec4::default(),
            stripes_amount: Vec4::default(),
            rim: Vec4::default(),
            bands: [Vec4::default(); 4],
            band_heights: Vec4::default(),
        }
    }

    pub fn new_with_center(
        color: Vec4,
        secondary_color: Vec4,
        direction: Vec4,
        flags: Vec4,
        dash: Vec4,
        world_center: Vec4,
    ) -> Self {
        Self {
            color,
            secondary_color,
            direction,
            flags,
            dash,
            world_center,
            noise: Vec4::default(),
            stripes: Vec4::default(),
            stripes_amount: Vec4::default(),
            rim: Vec4::default(),
            bands: [Vec4::default(); 4],
            band_heights: Vec4::default(),
        }
    }
}

/// Material extension that adds dual-directional gradient shading to StandardMaterial.
///
/// Uses a single `GradMeshUniformBuffer` bound at slot 100, containing all 4 Vec4
/// fields (64 bytes total). This avoids the binding slot conflict that occurs
/// when multiple `#[uniform(100)]` fields are declared separately.
#[derive(Debug, Clone, Reflect, Asset, AsBindGroup)]
#[reflect(Clone)]
pub struct GradMeshExtension {
    #[uniform(100)]
    pub uniforms: GradMeshUniformBuffer,
}

impl Default for GradMeshExtension {
    fn default() -> Self {
        Self {
            uniforms: GradMeshUniformBuffer::default(),
        }
    }
}

impl GradMeshExtension {
    /// Create from a `GradMeshUniform` descriptor.
    pub fn from_uniform(u: &GradMeshUniform) -> Self {
        Self {
            uniforms: GradMeshUniformBuffer::new(
                Vec4::from_array(u.color),
                Vec4::from_array(u.secondary_color),
                Vec4::from_array(u.direction),
                Vec4::from_array(u.flags),
                Vec4::new(0.0, 0.0, 0.0, 0.0),
            ),
        }
    }

    /// Create from a `GradMeshUniform` descriptor with a world-space center.
    pub fn from_uniform_with_center(u: &GradMeshUniform, world_center: Vec3) -> Self {
        Self {
            uniforms: GradMeshUniformBuffer::new_with_center(
                Vec4::from_array(u.color),
                Vec4::from_array(u.secondary_color),
                Vec4::from_array(u.direction),
                Vec4::from_array(u.flags),
                Vec4::new(0.0, 0.0, 0.0, 0.0),
                Vec4::new(world_center.x, world_center.y, world_center.z, 0.0),
            ),
        }
    }
}

/// Uniform values for every material preset, independent of `bevy_pbr`.
///
/// The PBR constructors in `pbr_material` and the 2D material both build on
/// these, so the same style produces the same gradient on every path.
pub(crate) mod presets {
    use super::*;

    pub fn opaque(r: f32, g: f32, b: f32) -> GradMeshUniformBuffer {
        GradMeshUniformBuffer::new(
            Vec4::new(r, g, b, 1.0),
            Vec4::new(r, g, b, 1.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 0.0),
        )
    }

    pub fn transparent(r: f32, g: f32, b: f32, alpha: f32) -> GradMeshUniformBuffer {
        GradMeshUniformBuffer::new(
            Vec4::new(r, g, b, alpha),
            Vec4::new(r, g, b, alpha),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 0.0),
        )
    }

    pub fn additive(r: f32, g: f32, b: f32) -> GradMeshUniformBuffer {
        GradMeshUniformBuffer::new(
            Vec4::new(r, g, b, 1.0),
            Vec4::new(r, g, b, 1.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 2.0, 1.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 0.0),
        )
    }

    pub fn dual_directional(
        dark: [f32; 3],
        light: [f32; 3],
        direction: [f32; 3],
    ) -> GradMeshUniformBuffer {
        let dir = Vec3::new(direction[0], direction[1], direction[2]);
        let dir_norm = dir.normalize();
        GradMeshUniformBuffer::new(
            Vec4::new(dark[0], dark[1], dark[2], 1.0),
            Vec4::new(light[0], light[1], light[2], 1.0),
            Vec4::new(dir_norm.x, dir_norm.y, dir_norm.z, 0.0),
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 0.0),
        )
    }

    pub fn dual_radial(inner: [f32; 3], outer: [f32; 3]) -> GradMeshUniformBuffer {
        GradMeshUniformBuffer::new(
            Vec4::new(inner[0], inner[1], inner[2], 1.0),
            Vec4::new(outer[0], outer[1], outer[2], 1.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(2.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 0.0),
        )
    }

    pub fn dual_radial_alpha(inner: [f32; 4], outer: [f32; 4]) -> GradMeshUniformBuffer {
        GradMeshUniformBuffer::new(
            Vec4::new(inner[0], inner[1], inner[2], inner[3]),
            Vec4::new(outer[0], outer[1], outer[2], outer[3]),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(2.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 0.0),
        )
    }

    pub fn multiplicative(r: f32, g: f32, b: f32) -> GradMeshUniformBuffer {
        GradMeshUniformBuffer::new(
            Vec4::new(r, g, b, 1.0),
            Vec4::new(r, g, b, 1.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 3.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 0.0),
        )
    }

    pub fn screen(r: f32, g: f32, b: f32) -> GradMeshUniformBuffer {
        GradMeshUniformBuffer::new(
            Vec4::new(r, g, b, 1.0),
            Vec4::new(r, g, b, 1.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 4.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 0.0),
        )
    }

    pub fn dither(r: f32, g: f32, b: f32, alpha: f32) -> GradMeshUniformBuffer {
        GradMeshUniformBuffer::new(
            Vec4::new(r, g, b, alpha),
            Vec4::new(r, g, b, alpha),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 5.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 0.0),
        )
    }

    pub fn subtractive(r: f32, g: f32, b: f32) -> GradMeshUniformBuffer {
        GradMeshUniformBuffer::new(
            Vec4::new(r, g, b, 1.0),
            Vec4::new(r, g, b, 1.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 6.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 0.0),
        )
    }

    pub fn lighten(r: f32, g: f32, b: f32) -> GradMeshUniformBuffer {
        GradMeshUniformBuffer::new(
            Vec4::new(r, g, b, 1.0),
            Vec4::new(r, g, b, 1.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 7.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 0.0),
        )
    }

    pub fn color_dodge(r: f32, g: f32, b: f32) -> GradMeshUniformBuffer {
        GradMeshUniformBuffer::new(
            Vec4::new(r, g, b, 1.0),
            Vec4::new(r, g, b, 1.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 8.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 0.0),
        )
    }

    pub fn linear_burn(r: f32, g: f32, b: f32) -> GradMeshUniformBuffer {
        GradMeshUniformBuffer::new(
            Vec4::new(r, g, b, 1.0),
            Vec4::new(r, g, b, 1.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 9.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 0.0),
        )
    }

    pub fn color_burn(r: f32, g: f32, b: f32) -> GradMeshUniformBuffer {
        GradMeshUniformBuffer::new(
            Vec4::new(r, g, b, 1.0),
            Vec4::new(r, g, b, 1.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 10.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 0.0),
        )
    }

    pub fn darken(r: f32, g: f32, b: f32) -> GradMeshUniformBuffer {
        GradMeshUniformBuffer::new(
            Vec4::new(r, g, b, 1.0),
            Vec4::new(r, g, b, 1.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 11.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 0.0),
        )
    }

    pub fn dashed(
        r: f32,
        g: f32,
        b: f32,
        dash_count: u32,
        spacing: f32,
        offset: f32,
        dash_type: u32,
        modifier: f32,
        shape_type: u32,
    ) -> GradMeshUniformBuffer {
        let repeat_span = (dash_count.max(1)) as f32;
        let gap_ratio = spacing.clamp(0.0, 1.0);
        let type_code = 0.1 + 0.3 * (dash_type.min(2)) as f32 + 0.075 * modifier.clamp(-1.0, 1.0);
        // Encode shape_type in flags.z so the shader knows which perimeter function to use.
        let shape_type_code = shape_type.min(2) as f32;
        GradMeshUniformBuffer::new(
            Vec4::new(r, g, b, 1.0),
            Vec4::new(r, g, b, 1.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, shape_type_code, 0.0),
            Vec4::new(repeat_span, type_code, gap_ratio, offset),
        )
    }
}

#[cfg(feature = "pbr")]
pub use pbr_material::*;

/// Items that need `bevy_pbr`: the extended material and its constructors.
#[cfg(feature = "pbr")]
mod pbr_material {
    use super::*;
    use bevy::asset::AssetPath;
    use bevy::color::Color;
    use bevy::mesh::MeshVertexBufferLayoutRef;
    use bevy::pbr::{
        ExtendedMaterial, MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline,
        StandardMaterial,
    };
    use bevy::render::render_resource::{RenderPipelineDescriptor, SpecializedMeshPipelineError};
    use bevy::shader::ShaderRef;
    use bevy_material::AlphaMode;

    fn gradmesh_shader_ref() -> ShaderRef {
        ShaderRef::Path(
            AssetPath::from_path_buf(bevy::asset::embedded_path!("shader/gradmesh.wesl"))
                .with_source("embedded"),
        )
    }

    impl MaterialExtension for GradMeshExtension {
        fn fragment_shader() -> ShaderRef {
            gradmesh_shader_ref()
        }

        fn deferred_fragment_shader() -> ShaderRef {
            gradmesh_shader_ref()
        }

        fn specialize(
            _pipeline: &MaterialExtensionPipeline,
            descriptor: &mut RenderPipelineDescriptor,
            _layout: &MeshVertexBufferLayoutRef,
            _key: MaterialExtensionKey<Self>,
        ) -> Result<(), SpecializedMeshPipelineError> {
            // The shader file has multiple @fragment entry points ("fragment" and "instanced_fragment").
            // Explicitly tell the pipeline which one to use for the material extension.
            if let Some(ref mut fragment) = descriptor.fragment {
                fragment.entry_point = Some("fragment".into());
            }
            Ok(())
        }
    }

    /// Type alias for the full extended material.
    pub type GradMeshMaterial = ExtendedMaterial<StandardMaterial, GradMeshExtension>;

    /// Sets the region a gradient material blends across.
    ///
    /// By default gradients assume a roughly unit-sized shape centred at the world
    /// origin, which makes them look flat on larger shapes. `center` is the middle of
    /// the gradient and `half_extent` is the distance from the centre to where the
    /// second colour is fully reached (along the gradient direction).
    pub fn with_gradient_bounds(
        mut material: GradMeshMaterial,
        center: Vec3,
        half_extent: f32,
    ) -> GradMeshMaterial {
        material.extension.uniforms.world_center =
            Vec4::new(center.x, center.y, center.z, half_extent.max(1e-4));
        material
    }

    /// Selects the colour space gradients are blended in.
    ///
    /// Materials blend in [`MixSpace::Oklab`] unless told otherwise. Single-colour
    /// materials are unaffected.
    #[must_use]
    pub fn with_mix_space(mut material: GradMeshMaterial, space: MixSpace) -> GradMeshMaterial {
        material.extension.uniforms.flags.w = space.gpu_code();
        material
    }

    /// Creates a flat opaque GradMeshMaterial with a single color.
    pub fn opaque(r: f32, g: f32, b: f32) -> GradMeshMaterial {
        GradMeshMaterial {
            base: StandardMaterial {
                base_color: Color::srgb(r, g, b).into(),
                perceptual_roughness: 0.7,
                cull_mode: None,
                ..StandardMaterial::default()
            },
            extension: GradMeshExtension {
                uniforms: presets::opaque(r, g, b),
            },
        }
    }

    /// Creates a transparent GradMeshMaterial with alpha.
    pub fn transparent(r: f32, g: f32, b: f32, alpha: f32) -> GradMeshMaterial {
        GradMeshMaterial {
            base: StandardMaterial {
                base_color: Color::srgba(r, g, b, alpha).into(),
                perceptual_roughness: 0.7,
                cull_mode: None,
                alpha_mode: AlphaMode::Blend,
                ..StandardMaterial::default()
            },
            extension: GradMeshExtension {
                uniforms: presets::transparent(r, g, b, alpha),
            },
        }
    }

    /// Creates an additive (emissive) GradMeshMaterial.
    pub fn additive(r: f32, g: f32, b: f32) -> GradMeshMaterial {
        GradMeshMaterial {
            base: StandardMaterial {
                base_color: Color::srgb(r, g, b).into(),
                perceptual_roughness: 0.3,
                // Values around 2-3 stay coloured under the default tone mapper; much
                // higher values saturate to pure white.
                emissive: bevy::color::LinearRgba::rgb(r, g, b) * 2.5,
                cull_mode: None,
                ..StandardMaterial::default()
            },
            extension: GradMeshExtension {
                uniforms: presets::additive(r, g, b),
            },
        }
    }

    /// Creates a dual-directional gradient GradMeshMaterial.
    ///
    /// `dark` is the shadow-side color, `light` is the highlight-side color,
    /// and `direction` defines the gradient axis in world space. Colours blend in
    /// OKLAB; use [`with_mix_space`] to pick another space.
    pub fn dual_directional(
        dark: [f32; 3],
        light: [f32; 3],
        direction: [f32; 3],
    ) -> GradMeshMaterial {
        GradMeshMaterial {
            base: StandardMaterial {
                base_color: Color::srgb(dark[0], dark[1], dark[2]).into(),
                perceptual_roughness: 0.7,
                cull_mode: None,
                ..StandardMaterial::default()
            },
            extension: GradMeshExtension {
                uniforms: presets::dual_directional(dark, light, direction),
            },
        }
    }

    /// Creates a dual-radial gradient GradMeshMaterial (inner/outer color blend).
    pub fn dual_radial(inner: [f32; 3], outer: [f32; 3]) -> GradMeshMaterial {
        GradMeshMaterial {
            base: StandardMaterial {
                base_color: Color::srgb(inner[0], inner[1], inner[2]).into(),
                perceptual_roughness: 0.3,
                cull_mode: None,
                ..StandardMaterial::default()
            },
            extension: GradMeshExtension {
                uniforms: presets::dual_radial(inner, outer),
            },
        }
    }

    /// Creates a dual-radial gradient GradMeshMaterial with alpha on both colours.
    ///
    /// `inner` and `outer` are `[r, g, b, a]`; the material alpha-blends.
    pub fn dual_radial_alpha(inner: [f32; 4], outer: [f32; 4]) -> GradMeshMaterial {
        GradMeshMaterial {
            base: StandardMaterial {
                base_color: Color::srgba(inner[0], inner[1], inner[2], inner[3]).into(),
                perceptual_roughness: 0.3,
                cull_mode: None,
                alpha_mode: AlphaMode::Blend,
                ..StandardMaterial::default()
            },
            extension: GradMeshExtension {
                uniforms: presets::dual_radial_alpha(inner, outer),
            },
        }
    }

    /// Creates a multiplicative blend GradMeshMaterial (darkens the result).
    pub fn multiplicative(r: f32, g: f32, b: f32) -> GradMeshMaterial {
        GradMeshMaterial {
            base: StandardMaterial {
                base_color: Color::srgb(r, g, b).into(),
                perceptual_roughness: 0.7,
                cull_mode: None,
                ..StandardMaterial::default()
            },
            extension: GradMeshExtension {
                uniforms: presets::multiplicative(r, g, b),
            },
        }
    }

    /// Creates a screen blend GradMeshMaterial (lightens the result).
    pub fn screen(r: f32, g: f32, b: f32) -> GradMeshMaterial {
        GradMeshMaterial {
            base: StandardMaterial {
                base_color: Color::srgb(r, g, b).into(),
                perceptual_roughness: 0.7,
                cull_mode: None,
                ..StandardMaterial::default()
            },
            extension: GradMeshExtension {
                uniforms: presets::screen(r, g, b),
            },
        }
    }

    /// Creates a dither blend GradMeshMaterial (alpha-to-coverage via Bayer matrix).
    pub fn dither(r: f32, g: f32, b: f32, alpha: f32) -> GradMeshMaterial {
        GradMeshMaterial {
            base: StandardMaterial {
                base_color: Color::srgba(r, g, b, alpha).into(),
                perceptual_roughness: 0.7,
                cull_mode: None,
                alpha_mode: AlphaMode::Blend,
                ..StandardMaterial::default()
            },
            extension: GradMeshExtension {
                uniforms: presets::dither(r, g, b, alpha),
            },
        }
    }

    /// Creates a subtractive blend GradMeshMaterial (subtracts color from background).
    pub fn subtractive(r: f32, g: f32, b: f32) -> GradMeshMaterial {
        GradMeshMaterial {
            base: StandardMaterial {
                base_color: Color::srgb(r, g, b).into(),
                perceptual_roughness: 0.7,
                cull_mode: None,
                ..StandardMaterial::default()
            },
            extension: GradMeshExtension {
                uniforms: presets::subtractive(r, g, b),
            },
        }
    }

    /// Creates a lighten blend GradMeshMaterial (takes maximum of source and destination).
    pub fn lighten(r: f32, g: f32, b: f32) -> GradMeshMaterial {
        GradMeshMaterial {
            base: StandardMaterial {
                base_color: Color::srgb(r, g, b).into(),
                perceptual_roughness: 0.7,
                cull_mode: None,
                ..StandardMaterial::default()
            },
            extension: GradMeshExtension {
                uniforms: presets::lighten(r, g, b),
            },
        }
    }

    /// Creates a color dodge blend GradMeshMaterial (brightens destination by source).
    pub fn color_dodge(r: f32, g: f32, b: f32) -> GradMeshMaterial {
        GradMeshMaterial {
            base: StandardMaterial {
                base_color: Color::srgb(r, g, b).into(),
                perceptual_roughness: 0.7,
                cull_mode: None,
                ..StandardMaterial::default()
            },
            extension: GradMeshExtension {
                uniforms: presets::color_dodge(r, g, b),
            },
        }
    }

    /// Creates a linear burn blend GradMeshMaterial (darkens destination by source).
    pub fn linear_burn(r: f32, g: f32, b: f32) -> GradMeshMaterial {
        GradMeshMaterial {
            base: StandardMaterial {
                base_color: Color::srgb(r, g, b).into(),
                perceptual_roughness: 0.7,
                cull_mode: None,
                ..StandardMaterial::default()
            },
            extension: GradMeshExtension {
                uniforms: presets::linear_burn(r, g, b),
            },
        }
    }

    /// Creates a color burn blend GradMeshMaterial (deepens destination by source).
    pub fn color_burn(r: f32, g: f32, b: f32) -> GradMeshMaterial {
        GradMeshMaterial {
            base: StandardMaterial {
                base_color: Color::srgb(r, g, b).into(),
                perceptual_roughness: 0.7,
                cull_mode: None,
                ..StandardMaterial::default()
            },
            extension: GradMeshExtension {
                uniforms: presets::color_burn(r, g, b),
            },
        }
    }

    /// Creates a darken blend GradMeshMaterial (takes minimum of source and destination).
    pub fn darken(r: f32, g: f32, b: f32) -> GradMeshMaterial {
        GradMeshMaterial {
            base: StandardMaterial {
                base_color: Color::srgb(r, g, b).into(),
                perceptual_roughness: 0.7,
                cull_mode: None,
                ..StandardMaterial::default()
            },
            extension: GradMeshExtension {
                uniforms: presets::darken(r, g, b),
            },
        }
    }

    /// Creates a GradMeshMaterial with dash pattern.
    ///
    /// `dash_count` is the number of dashes (1-4096), `spacing` is the gap ratio (0-1),
    /// `offset` shifts the dash pattern, `dash_type` selects normal/chevron/rounded,
    /// `modifier` controls chevron V-shape or rounded end-cap intensity (-1..1),
    /// and `shape_type` selects the perimeter progress function (0=generic, 1=disc, 2=rectangle).
    pub fn dashed(
        r: f32,
        g: f32,
        b: f32,
        dash_count: u32,
        spacing: f32,
        offset: f32,
        dash_type: u32,
        modifier: f32,
        shape_type: u32,
    ) -> GradMeshMaterial {
        GradMeshMaterial {
            base: StandardMaterial {
                base_color: Color::srgb(r, g, b).into(),
                perceptual_roughness: 0.7,
                cull_mode: None,
                ..StandardMaterial::default()
            },
            extension: GradMeshExtension {
                uniforms: presets::dashed(
                    r, g, b, dash_count, spacing, offset, dash_type, modifier, shape_type,
                ),
            },
        }
    }

    /// Creates a `StandardMaterial` from a `GradMeshUniform` descriptor (fallback for non-shader use).
    pub fn gradmesh_to_standard(uniform: &GradMeshUniform) -> StandardMaterial {
        let r = uniform.color[0];
        let g = uniform.color[1];
        let b = uniform.color[2];
        let a = uniform.color[3];
        StandardMaterial {
            base_color: Color::srgba(r, g, b, a).into(),
            perceptual_roughness: 0.7,
            ..StandardMaterial::default()
        }
    }
}
