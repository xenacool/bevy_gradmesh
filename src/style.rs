//! Material style descriptors for procedural shading.
//!
//! Provides style methods: opaque, transparent, dual_directional, dual_radial.

use bevy::asset::{Assets, Handle};
use bevy::color::Color;
use bevy::math::Vec3;
use bevy::prelude::Shader;

use crate::material::{
    additive, dashed, dither, dual_directional, dual_directional_oklab, dual_radial, dual_radial_oklab,
    multiplicative, opaque, screen, transparent, GradMeshMaterial,
};

/// Blend mode for material rendering.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum BlendMode {
    #[default]
    Opaque,
    Transparent,
    Additive,
    Multiplicative,
    Screen,
    Dither,
    /// Custom shader passthrough — uses the provided shader for fragment rendering.
    Custom(Handle<Shader>),
}

impl std::hash::Hash for BlendMode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        if let BlendMode::Custom(handle) = self {
            handle.id().hash(state);
        }
    }
}

/// Color mode for gradient shading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorMode {
    #[default]
    Single,
    DualDirectional,
    DualRadial,
}

/// Dash type for outlined shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DashType {
    #[default]
    Normal,
    Chevron,
    Rounded,
}

/// Dash pattern configuration.
#[derive(Debug, Clone, Copy, Default)]
pub struct Dash {
    /// Number of dashes (1-4096) or 0 for no dash.
    pub count: u32,
    /// Gap ratio between dashes (0.0-1.0).
    pub spacing: f32,
    /// Offset into the dash pattern.
    pub offset: f32,
    /// Dash shape style.
    pub style: DashType,
}

impl Dash {
    /// Creates a dash pattern with a fixed number of dashes.
    pub fn fixed(count: u32, spacing: f32) -> Self {
        Self { count, spacing, offset: 0.0, style: DashType::Normal }
    }

    /// Creates a dash pattern with chevron-style dashes.
    pub fn chevron(count: u32, spacing: f32) -> Self {
        Self { count, spacing, offset: 0.0, style: DashType::Chevron }
    }

    /// Creates a dash pattern with rounded dashes.
    pub fn rounded(count: u32, spacing: f32) -> Self {
        Self { count, spacing, offset: 0.0, style: DashType::Rounded }
    }

    /// Returns true if the dash pattern is enabled.
    pub fn is_enabled(&self) -> bool { self.count > 0 }
}

/// Material style descriptor.
///
/// Describes how a primitive should be shaded. Use the factory methods
/// (`opaque`, `transparent`, `dual_directional`, `dual_radial`) to create styles.
#[derive(Debug, Clone)]
pub struct Style {
    pub blend_mode: BlendMode,
    pub color_mode: ColorMode,
    /// Primary color (or dark color for dual modes).
    pub color: Color,
    /// Secondary color (light/inner color for dual modes).
    pub secondary_color: Option<Color>,
    /// Direction vector for dual_directional shading (normalized).
    pub direction: Vec3,
    /// Whether to use OKLAB color interpolation for gradients.
    pub oklab: bool,
    /// Dash pattern for outlined shapes.
    pub dash: Option<Dash>,
}

impl Style {
    /// Creates a flat opaque style with a single color.
    pub fn opaque(color: Color) -> Self {
        Self {
            blend_mode: BlendMode::Opaque,
            color_mode: ColorMode::Single,
            color,
            secondary_color: None,
            direction: Vec3::Y,
            oklab: false,
            dash: None,
        }
    }

    /// Creates a transparent style with alpha blending.
    pub fn transparent(color: Color) -> Self {
        Self {
            blend_mode: BlendMode::Transparent,
            color_mode: ColorMode::Single,
            color,
            secondary_color: None,
            direction: Vec3::Y,
            oklab: false,
            dash: None,
        }
    }

    /// Creates a dual-directional gradient style.
    ///
    /// `negative` is the dark color (shadow side), `positive` is the light color (lit side).
    /// `direction` is the light direction in world space (default: Vec3::Y).
    pub fn dual_directional(negative: Color, positive: Color, direction: Vec3) -> Self {
        Self {
            blend_mode: BlendMode::Opaque,
            color_mode: ColorMode::DualDirectional,
            color: negative,
            secondary_color: Some(positive),
            direction: direction.normalize_or(Vec3::Y),
            oklab: false,
            dash: None,
        }
    }

    /// Creates a dual-directional gradient style with default light direction (Vec3::Y).
    pub fn dual_directional_up(negative: Color, positive: Color) -> Self {
        Self::dual_directional(negative, positive, Vec3::Y)
    }

    /// Creates a dual-directional gradient style with OKLAB interpolation.
    pub fn dual_directional_oklab(negative: Color, positive: Color, direction: Vec3) -> Self {
        Self {
            blend_mode: BlendMode::Opaque,
            color_mode: ColorMode::DualDirectional,
            color: negative,
            secondary_color: Some(positive),
            direction: direction.normalize_or(Vec3::Y),
            oklab: true,
            dash: None,
        }
    }

    /// Creates a dual-radial gradient style (inner/outer colors).
    pub fn dual_radial(inner: Color, outer: Color) -> Self {
        Self {
            blend_mode: BlendMode::Opaque,
            color_mode: ColorMode::DualRadial,
            color: inner,
            secondary_color: Some(outer),
            direction: Vec3::Y,
            oklab: false,
            dash: None,
        }
    }

    /// Creates a multiplicative blend style (darkens).
    pub fn multiplicative(color: Color) -> Self {
        Self {
            blend_mode: BlendMode::Multiplicative,
            color_mode: ColorMode::Single,
            color,
            secondary_color: None,
            direction: Vec3::Y,
            oklab: false,
            dash: None,
        }
    }

    /// Creates a screen blend style (lightens).
    pub fn screen(color: Color) -> Self {
        Self {
            blend_mode: BlendMode::Screen,
            color_mode: ColorMode::Single,
            color,
            secondary_color: None,
            direction: Vec3::Y,
            oklab: false,
            dash: None,
        }
    }

    /// Creates a dither blend style (alpha-to-coverage).
    pub fn dither(color: Color) -> Self {
        Self {
            blend_mode: BlendMode::Dither,
            color_mode: ColorMode::Single,
            color,
            secondary_color: None,
            direction: Vec3::Y,
            oklab: false,
            dash: None,
        }
    }

    /// Creates a custom shader style.
    ///
    /// The provided shader handle replaces the built-in GradMesh fragment shader.
    /// The custom shader must accept the same vertex output as the built-in shader
    /// (`VertexOutput` from Bevy PBR).
    pub fn custom(shader: Handle<Shader>, color: Color) -> Self {
        Self {
            blend_mode: BlendMode::Custom(shader),
            color_mode: ColorMode::Single,
            color,
            secondary_color: None,
            direction: Vec3::Y,
            oklab: false,
            dash: None,
        }
    }

    /// Sets a dash pattern on this style.
    pub fn with_dash(mut self, dash: Dash) -> Self {
        self.dash = Some(dash);
        self
    }

    /// Enables OKLAB color interpolation.
    pub fn with_oklab(mut self) -> Self {
        self.oklab = true;
        self
    }

    /// Builds a `GradMeshMaterial` from this style and registers it in the asset server.
    ///
    /// Returns a handle to the newly created material.
    pub fn build_material(&self, materials: &mut Assets<GradMeshMaterial>) -> Handle<GradMeshMaterial> {
        let material = self.build();
        materials.add(material)
    }

    /// Builds a `GradMeshMaterial` from this style without registering it.
    pub fn build(&self) -> GradMeshMaterial {
        // Extract color components using to_srgba() which handles all Color variants
        let srgba = self.color.to_srgba();
        let (r, g, b, a) = (srgba.red, srgba.green, srgba.blue, srgba.alpha);
        let (sr, sg, sb, sa) = self.secondary_color
            .map_or((r, g, b, a), |c| {
                let s = c.to_srgba();
                (s.red, s.green, s.blue, s.alpha)
            });

        // Build based on color mode and blend mode
        let base_material = match self.color_mode {
            ColorMode::DualDirectional => {
                if self.oklab {
                    dual_directional_oklab(
                        [r, g, b], [sr, sg, sb],
                        [self.direction.x, self.direction.y, self.direction.z],
                    )
                } else {
                    dual_directional(
                        [r, g, b], [sr, sg, sb],
                        [self.direction.x, self.direction.y, self.direction.z],
                    )
                }
            }
            ColorMode::DualRadial => {
                if self.oklab {
                    dual_radial_oklab([r, g, b, a], [sr, sg, sb, sa], 1.0)
                } else {
                    dual_radial([r, g, b], [sr, sg, sb])
                }
            }
            ColorMode::Single => {
                match &self.blend_mode {
                    BlendMode::Opaque => opaque(r, g, b),
                    BlendMode::Transparent => transparent(r, g, b, a),
                    BlendMode::Additive => additive(r, g, b),
                    BlendMode::Multiplicative => multiplicative(r, g, b),
                    BlendMode::Screen => screen(r, g, b),
                    BlendMode::Dither => dither(r, g, b, a),
                    BlendMode::Custom(_) => opaque(r, g, b),
                }
            }
        };

        // Apply dash pattern if set
        if let Some(ref dash) = self.dash {
            if dash.is_enabled() {
                let dash_type_code = match dash.style {
                    DashType::Normal => 0,
                    DashType::Chevron => 1,
                    DashType::Rounded => 2,
                };
                return dashed(
                    r, g, b,
                    dash.count,
                    dash.spacing,
                    dash.offset,
                    dash_type_code,
                );
            }
        }

        base_material
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opaque_style_has_single_color_mode() {
        let style = Style::opaque(Color::srgb(1.0, 0.0, 0.0));
        assert_eq!(style.color_mode, ColorMode::Single);
        assert_eq!(style.blend_mode, BlendMode::Opaque);
        assert!(style.secondary_color.is_none());
    }

    #[test]
    fn dual_directional_has_two_colors() {
        let dark = Color::srgb(0.3, 0.2, 0.1);
        let light = Color::srgb(0.96, 0.8, 0.69);
        let style = Style::dual_directional(dark, light, Vec3::Y);
        assert_eq!(style.color_mode, ColorMode::DualDirectional);
        assert_eq!(style.color, dark);
        assert_eq!(style.secondary_color, Some(light));
    }

    #[test]
    fn dual_directional_normalizes_direction() {
        let style = Style::dual_directional(
            Color::srgb(0.0, 0.0, 0.0),
            Color::srgb(1.0, 1.0, 1.0),
            Vec3::new(3.0, 4.0, 0.0),
        );
        let len = style.direction.length();
        assert!((len - 1.0).abs() < 0.001, "Direction should be normalized");
    }

    #[test]
    fn transparent_style_has_correct_blend_mode() {
        let style = Style::transparent(Color::srgba(1.0, 0.0, 0.0, 0.5));
        assert_eq!(style.blend_mode, BlendMode::Transparent);
    }

    #[test]
    fn dual_radial_style() {
        let style = Style::dual_radial(Color::srgb(1.0, 1.0, 1.0), Color::srgb(0.0, 0.0, 0.0));
        assert_eq!(style.color_mode, ColorMode::DualRadial);
    }
}
