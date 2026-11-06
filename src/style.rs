//! Material style descriptors for procedural shading.
//!
//! Provides style methods: opaque, transparent, dual_directional, dual_radial.

use bevy::asset::{Assets, Handle};
use bevy::color::Color;
use bevy::math::Vec3;
use bevy::prelude::Shader;

use crate::material::{
    GradMeshMaterial, additive, color_burn, color_dodge, darken, dashed, dither, dual_directional,
    dual_radial, dual_radial_alpha, lighten, linear_burn, multiplicative, opaque, screen,
    subtractive, transparent, with_mix_space,
};
use crate::mix_space::MixSpace;

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
    Subtractive,
    Lighten,
    ColorDodge,
    LinearBurn,
    ColorBurn,
    Darken,
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

/// Shape type for perimeter-aware dash patterns.
///
/// Tells the shader which perimeter progress function to use so dashes
/// follow the actual shape outline instead of a diagonal UV sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShapeType {
    #[default]
    /// No shape hint — uses generic diagonal UV progress.
    Generic,
    /// Disc/circle — uses angular progress around center.
    Disc,
    /// Rectangle — uses perimeter tracing around edges.
    Rectangle,
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
    /// Modifier that controls the intensity of chevron V-shape or rounded
    /// end-cap curvature.  Range: -1.0..1.0; 0.0 disables the effect.
    pub modifier: f32,
}

impl Dash {
    /// Creates a dash pattern with a fixed number of dashes.
    pub fn fixed(count: u32, spacing: f32) -> Self {
        Self {
            count,
            spacing,
            offset: 0.0,
            style: DashType::Normal,
            modifier: 0.0,
        }
    }

    /// Creates a dash pattern with chevron-style dashes.
    ///
    /// The `modifier` controls the V-shape intensity (default 0.8).
    /// Range: -1.0..1.0.
    pub fn chevron(count: u32, spacing: f32) -> Self {
        Self {
            count,
            spacing,
            offset: 0.0,
            style: DashType::Chevron,
            modifier: 0.8,
        }
    }

    /// Creates a dash pattern with rounded end-caps.
    ///
    /// The `modifier` controls the rounding intensity (default 0.6).
    /// Range: -1.0..1.0.
    pub fn rounded(count: u32, spacing: f32) -> Self {
        Self {
            count,
            spacing,
            offset: 0.0,
            style: DashType::Rounded,
            modifier: 0.6,
        }
    }

    /// Returns true if the dash pattern is enabled.
    pub fn is_enabled(&self) -> bool {
        self.count > 0
    }
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
    /// Colour space gradients are blended in (OKLAB by default).
    pub space: MixSpace,
    /// Dash pattern for outlined shapes.
    pub dash: Option<Dash>,
    /// Shape type hint for perimeter-aware dash patterns.
    pub shape_type: ShapeType,
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
            space: MixSpace::default(),
            dash: None,
            shape_type: ShapeType::Generic,
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
            space: MixSpace::default(),
            dash: None,
            shape_type: ShapeType::Generic,
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
            space: MixSpace::default(),
            dash: None,
            shape_type: ShapeType::Generic,
        }
    }

    /// Creates a dual-directional gradient style with default light direction (Vec3::Y).
    pub fn dual_directional_up(negative: Color, positive: Color) -> Self {
        Self::dual_directional(negative, positive, Vec3::Y)
    }

    /// Creates a dual-radial gradient style (inner/outer colors).
    pub fn dual_radial(inner: Color, outer: Color) -> Self {
        Self {
            blend_mode: BlendMode::Opaque,
            color_mode: ColorMode::DualRadial,
            color: inner,
            secondary_color: Some(outer),
            direction: Vec3::Y,
            space: MixSpace::default(),
            dash: None,
            shape_type: ShapeType::Generic,
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
            space: MixSpace::default(),
            dash: None,
            shape_type: ShapeType::Generic,
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
            space: MixSpace::default(),
            dash: None,
            shape_type: ShapeType::Generic,
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
            space: MixSpace::default(),
            dash: None,
            shape_type: ShapeType::Generic,
        }
    }

    /// Creates a subtractive blend style (subtracts color from background).
    pub fn subtractive(color: Color) -> Self {
        Self {
            blend_mode: BlendMode::Subtractive,
            color_mode: ColorMode::Single,
            color,
            secondary_color: None,
            direction: Vec3::Y,
            space: MixSpace::default(),
            dash: None,
            shape_type: ShapeType::Generic,
        }
    }

    /// Creates a lighten blend style (takes maximum of source and destination).
    pub fn lighten(color: Color) -> Self {
        Self {
            blend_mode: BlendMode::Lighten,
            color_mode: ColorMode::Single,
            color,
            secondary_color: None,
            direction: Vec3::Y,
            space: MixSpace::default(),
            dash: None,
            shape_type: ShapeType::Generic,
        }
    }

    /// Creates a color dodge blend style (brightens destination by source).
    pub fn color_dodge(color: Color) -> Self {
        Self {
            blend_mode: BlendMode::ColorDodge,
            color_mode: ColorMode::Single,
            color,
            secondary_color: None,
            direction: Vec3::Y,
            space: MixSpace::default(),
            dash: None,
            shape_type: ShapeType::Generic,
        }
    }

    /// Creates a linear burn blend style (darkens destination by source).
    pub fn linear_burn(color: Color) -> Self {
        Self {
            blend_mode: BlendMode::LinearBurn,
            color_mode: ColorMode::Single,
            color,
            secondary_color: None,
            direction: Vec3::Y,
            space: MixSpace::default(),
            dash: None,
            shape_type: ShapeType::Generic,
        }
    }

    /// Creates a color burn blend style (deepens destination by source).
    pub fn color_burn(color: Color) -> Self {
        Self {
            blend_mode: BlendMode::ColorBurn,
            color_mode: ColorMode::Single,
            color,
            secondary_color: None,
            direction: Vec3::Y,
            space: MixSpace::default(),
            dash: None,
            shape_type: ShapeType::Generic,
        }
    }

    /// Creates a darken blend style (takes minimum of source and destination).
    pub fn darken(color: Color) -> Self {
        Self {
            blend_mode: BlendMode::Darken,
            color_mode: ColorMode::Single,
            color,
            secondary_color: None,
            direction: Vec3::Y,
            space: MixSpace::default(),
            dash: None,
            shape_type: ShapeType::Generic,
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
            space: MixSpace::default(),
            dash: None,
            shape_type: ShapeType::Generic,
        }
    }

    /// Sets a dash pattern on this style.
    pub fn with_dash(mut self, dash: Dash) -> Self {
        self.dash = Some(dash);
        self
    }

    /// Sets the colour space gradients are blended in.
    #[must_use]
    pub fn with_mix_space(mut self, space: MixSpace) -> Self {
        self.space = space;
        self
    }

    /// Blends this style towards `other` by `t` (clamped to `0.0..=1.0`).
    ///
    /// * Colours mix in `self.space`; the result keeps that space.
    /// * Continuous values (direction, dash spacing, offset and modifier) interpolate.
    /// * Discrete values (blend mode, colour mode, dash count and style, shape
    ///   type) follow whichever endpoint is nearer.
    /// * A missing secondary colour mixes as if it equalled the primary colour.
    ///
    /// `t == 0.0` reproduces `self` and `t == 1.0` reproduces `other`, up to
    /// colour-space round-trip error (the blended colours are returned in
    /// `self.space`'s colour type) and except that `space` always stays
    /// `self.space`.
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let t = if t.is_nan() { 0.0 } else { t.clamp(0.0, 1.0) };
        let near = if t < 0.5 { self } else { other };
        let mix = |a: f32, b: f32| a * (1.0 - t) + b * t;

        let color = self.space.mix(self.color, other.color, t);
        let secondary_color = match (self.secondary_color, other.secondary_color) {
            (None, None) => None,
            (a, b) => Some(self.space.mix(
                a.unwrap_or(self.color),
                b.unwrap_or(other.color),
                t,
            )),
        };
        let direction = self
            .direction
            .lerp(other.direction, t)
            .normalize_or(near.direction);
        let dash = match (self.dash, other.dash) {
            (Some(a), Some(b)) => Some(Dash {
                count: near.dash.map_or(a.count, |d| d.count),
                spacing: mix(a.spacing, b.spacing),
                offset: mix(a.offset, b.offset),
                style: near.dash.map_or(a.style, |d| d.style),
                modifier: mix(a.modifier, b.modifier),
            }),
            _ => near.dash,
        };

        Self {
            blend_mode: near.blend_mode.clone(),
            color_mode: near.color_mode,
            color,
            secondary_color,
            direction,
            space: self.space,
            dash,
            shape_type: near.shape_type,
        }
    }

    /// Sets the shape type for perimeter-aware dash patterns.
    ///
    /// When a dash pattern is applied, the shape type tells the shader which
    /// perimeter progress function to use so dashes follow the actual shape
    /// outline (e.g., angular progress for discs, edge tracing for rectangles).
    pub fn with_shape_type(mut self, shape_type: ShapeType) -> Self {
        self.shape_type = shape_type;
        self
    }

    /// Builds a `GradMeshMaterial` from this style and registers it in the asset server.
    ///
    /// Returns a handle to the newly created material.
    pub fn build_material(
        &self,
        materials: &mut Assets<GradMeshMaterial>,
    ) -> Handle<GradMeshMaterial> {
        let material = self.build();
        materials.add(material)
    }

    /// Builds a `GradMeshMaterial` from this style without registering it.
    pub fn build(&self) -> GradMeshMaterial {
        // Extract color components using to_srgba() which handles all Color variants
        let srgba = self.color.to_srgba();
        let (r, g, b, a) = (srgba.red, srgba.green, srgba.blue, srgba.alpha);
        let (sr, sg, sb, sa) = self.secondary_color.map_or((r, g, b, a), |c| {
            let s = c.to_srgba();
            (s.red, s.green, s.blue, s.alpha)
        });

        // Build based on color mode and blend mode
        let base_material = match self.color_mode {
            ColorMode::DualDirectional => with_mix_space(
                dual_directional(
                    [r, g, b],
                    [sr, sg, sb],
                    [self.direction.x, self.direction.y, self.direction.z],
                ),
                self.space,
            ),
            ColorMode::DualRadial => with_mix_space(
                if a < 1.0 || sa < 1.0 {
                    dual_radial_alpha([r, g, b, a], [sr, sg, sb, sa])
                } else {
                    dual_radial([r, g, b], [sr, sg, sb])
                },
                self.space,
            ),
            ColorMode::Single => match &self.blend_mode {
                BlendMode::Opaque => opaque(r, g, b),
                BlendMode::Transparent => transparent(r, g, b, a),
                BlendMode::Additive => additive(r, g, b),
                BlendMode::Multiplicative => multiplicative(r, g, b),
                BlendMode::Screen => screen(r, g, b),
                BlendMode::Dither => dither(r, g, b, a),
                BlendMode::Subtractive => subtractive(r, g, b),
                BlendMode::Lighten => lighten(r, g, b),
                BlendMode::ColorDodge => color_dodge(r, g, b),
                BlendMode::LinearBurn => linear_burn(r, g, b),
                BlendMode::ColorBurn => color_burn(r, g, b),
                BlendMode::Darken => darken(r, g, b),
                BlendMode::Custom(_) => opaque(r, g, b),
            },
        };

        // Apply dash pattern if set
        if let Some(ref dash) = self.dash {
            if dash.is_enabled() {
                let dash_type_code = match dash.style {
                    DashType::Normal => 0,
                    DashType::Chevron => 1,
                    DashType::Rounded => 2,
                };
                let shape_type_code = match self.shape_type {
                    ShapeType::Generic => 0,
                    ShapeType::Disc => 1,
                    ShapeType::Rectangle => 2,
                };
                return dashed(
                    r,
                    g,
                    b,
                    dash.count,
                    dash.spacing,
                    dash.offset,
                    dash_type_code,
                    dash.modifier,
                    shape_type_code,
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
    fn styles_default_to_oklab() {
        assert_eq!(Style::opaque(Color::WHITE).space, MixSpace::Oklab);
        let g = Style::dual_directional_up(Color::BLACK, Color::WHITE);
        assert_eq!(g.space, MixSpace::Oklab);
        assert_eq!(g.build().extension.uniforms.flags.w, 0.0);
    }

    #[test]
    fn build_encodes_mix_space() {
        for space in MixSpace::ALL {
            let m = Style::dual_radial(Color::BLACK, Color::WHITE)
                .with_mix_space(space)
                .build();
            assert_eq!(m.extension.uniforms.flags.w, space.gpu_code());
        }
    }

    #[test]
    fn lerp_mixes_colors_in_self_space() {
        let a = Style::dual_directional_up(Color::srgb(1.0, 0.0, 0.0), Color::srgb(0.0, 0.0, 1.0))
            .with_mix_space(MixSpace::Oklch);
        let b = Style::dual_directional_up(Color::srgb(0.0, 1.0, 0.0), Color::srgb(1.0, 1.0, 0.0));
        let mid = a.lerp(&b, 0.5);
        assert_eq!(mid.space, MixSpace::Oklch);
        assert_eq!(
            mid.color,
            MixSpace::Oklch.mix(a.color, b.color, 0.5),
        );
        let close = |x: Color, y: Color| {
            let (x, y) = (x.to_srgba(), y.to_srgba());
            (x.red - y.red).abs() + (x.green - y.green).abs() + (x.blue - y.blue).abs() < 1e-3
        };
        let (l0, l1) = (a.lerp(&b, 0.0), a.lerp(&b, 1.0));
        assert!(close(l0.color, a.color));
        assert!(close(l1.secondary_color.unwrap(), b.secondary_color.unwrap()));
    }

    #[test]
    fn lerp_snaps_discrete_fields() {
        let a = Style::opaque(Color::WHITE);
        let b = Style::transparent(Color::BLACK).with_shape_type(ShapeType::Disc);
        assert_eq!(a.lerp(&b, 0.25).blend_mode, BlendMode::Opaque);
        assert_eq!(a.lerp(&b, 0.75).blend_mode, BlendMode::Transparent);
        assert_eq!(a.lerp(&b, 0.75).shape_type, ShapeType::Disc);
    }

    #[test]
    fn lerp_blends_dash_parameters() {
        let a = Style::opaque(Color::WHITE).with_dash(Dash::fixed(8, 0.2));
        let b = Style::opaque(Color::WHITE).with_dash(Dash::fixed(16, 0.6));
        let mid = a.lerp(&b, 0.25).dash.unwrap();
        assert_eq!(mid.count, 8);
        assert!((mid.spacing - 0.3).abs() < 1e-6);
        let none = Style::opaque(Color::WHITE);
        assert!(a.lerp(&none, 0.25).dash.is_some());
        assert!(a.lerp(&none, 0.75).dash.is_none());
    }

    #[test]
    fn lerp_direction_stays_normalized() {
        let a = Style::dual_directional(Color::BLACK, Color::WHITE, Vec3::X);
        let b = Style::dual_directional(Color::BLACK, Color::WHITE, Vec3::NEG_X);
        let mid = a.lerp(&b, 0.5);
        assert!((mid.direction.length() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn dual_radial_style() {
        let style = Style::dual_radial(Color::srgb(1.0, 1.0, 1.0), Color::srgb(0.0, 0.0, 0.0));
        assert_eq!(style.color_mode, ColorMode::DualRadial);
    }
}
