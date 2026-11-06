//! Material style descriptors for procedural shading.
//!
//! Provides style methods: opaque, transparent, dual_directional, dual_radial.

use bevy::asset::{Assets, Handle};
use bevy::color::{Alpha, Color, LinearRgba};
use bevy::math::Vec3;
use bevy::prelude::Shader;
use bevy_material::AlphaMode;

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

/// Space a gradient direction is expressed in.
///
/// The gradient is evaluated per fragment as the projection of the fragment's
/// world position (relative to the gradient centre, see
/// [`with_gradient_bounds`](crate::material::with_gradient_bounds)) onto the
/// direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DirectionSpace {
    /// The direction is a fixed world-space vector (the default). Rotating the
    /// entity does not rotate the gradient, so it behaves like a light that stays
    /// put: a mesh turned upside down is lit from the same world direction.
    #[default]
    World,
    /// The direction is expressed in the mesh's local frame and is rotated by the
    /// entity's `GlobalTransform` every frame. Use this for pieces built Y-up
    /// and placed in a Z-up scene by a rotation: the gradient follows the piece
    /// so it keeps its intended lit and shadow sides.
    Local,
}

/// Dash pattern configuration.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
pub struct Style {
    pub blend_mode: BlendMode,
    pub color_mode: ColorMode,
    /// Primary color (or dark color for dual modes).
    pub color: Color,
    /// Secondary color (light/inner color for dual modes).
    pub secondary_color: Option<Color>,
    /// Direction vector for dual_directional shading (normalized).
    ///
    /// Interpreted in [`Style::direction_space`] (world space by default).
    pub direction: Vec3,
    /// Space `direction` is expressed in.
    pub direction_space: DirectionSpace,
    /// Colour space gradients are blended in (OKLAB by default).
    pub space: MixSpace,
    /// Dash pattern for outlined shapes.
    pub dash: Option<Dash>,
    /// Shape type hint for perimeter-aware dash patterns.
    pub shape_type: ShapeType,
    /// Perceptual roughness override (`None` keeps the constructor's default).
    pub perceptual_roughness: Option<f32>,
    /// Metallic factor, `0.0` (dielectric) to `1.0` (metal).
    pub metallic: f32,
    /// Emissive colour override (`None` keeps the constructor's default).
    pub emissive: Option<LinearRgba>,
    /// Exposure applied to Bevy lightmaps (`None` keeps Bevy's default of `1.0`).
    pub lightmap_exposure: Option<f32>,
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
            direction_space: DirectionSpace::World,
            perceptual_roughness: None,
            metallic: 0.0,
            emissive: None,
            lightmap_exposure: None,
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
            direction_space: DirectionSpace::World,
            perceptual_roughness: None,
            metallic: 0.0,
            emissive: None,
            lightmap_exposure: None,
        }
    }

    /// Creates a dual-directional gradient style.
    ///
    /// `negative` is the dark color (shadow side), `positive` is the light color (lit side).
    /// `direction` points from the dark side towards the lit side and is expressed
    /// in **world space** unless [`Style::with_direction_space`] selects
    /// [`DirectionSpace::Local`] (default direction: `Vec3::Y`). In world space the
    /// gradient does not follow the entity's rotation; in local space it does.
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
            direction_space: DirectionSpace::World,
            perceptual_roughness: None,
            metallic: 0.0,
            emissive: None,
            lightmap_exposure: None,
        }
    }

    /// Creates a dual-directional gradient style with default light direction (Vec3::Y).
    pub fn dual_directional_up(negative: Color, positive: Color) -> Self {
        Self::dual_directional(negative, positive, Vec3::Y)
    }

    /// Creates a dual-directional gradient from a single colour.
    ///
    /// `color` is used on the lit side and `color * dark_factor` (linear RGB,
    /// alpha unchanged) on the shadow side, blended in the style's mix space.
    /// `dark_factor` is clamped to `0.0..=1.0`. The direction defaults to `Vec3::Y`;
    /// change it with [`Style::with_direction`].
    pub fn shaded(color: Color, dark_factor: f32) -> Self {
        let factor = if dark_factor.is_nan() {
            1.0
        } else {
            dark_factor.clamp(0.0, 1.0)
        };
        let lit = color.to_linear();
        let dark = Color::LinearRgba(LinearRgba {
            red: lit.red * factor,
            green: lit.green * factor,
            blue: lit.blue * factor,
            alpha: lit.alpha,
        });
        Self::dual_directional(dark, color, Vec3::Y)
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
            direction_space: DirectionSpace::World,
            perceptual_roughness: None,
            metallic: 0.0,
            emissive: None,
            lightmap_exposure: None,
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
            direction_space: DirectionSpace::World,
            perceptual_roughness: None,
            metallic: 0.0,
            emissive: None,
            lightmap_exposure: None,
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
            direction_space: DirectionSpace::World,
            perceptual_roughness: None,
            metallic: 0.0,
            emissive: None,
            lightmap_exposure: None,
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
            direction_space: DirectionSpace::World,
            perceptual_roughness: None,
            metallic: 0.0,
            emissive: None,
            lightmap_exposure: None,
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
            direction_space: DirectionSpace::World,
            perceptual_roughness: None,
            metallic: 0.0,
            emissive: None,
            lightmap_exposure: None,
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
            direction_space: DirectionSpace::World,
            perceptual_roughness: None,
            metallic: 0.0,
            emissive: None,
            lightmap_exposure: None,
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
            direction_space: DirectionSpace::World,
            perceptual_roughness: None,
            metallic: 0.0,
            emissive: None,
            lightmap_exposure: None,
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
            direction_space: DirectionSpace::World,
            perceptual_roughness: None,
            metallic: 0.0,
            emissive: None,
            lightmap_exposure: None,
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
            direction_space: DirectionSpace::World,
            perceptual_roughness: None,
            metallic: 0.0,
            emissive: None,
            lightmap_exposure: None,
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
            direction_space: DirectionSpace::World,
            perceptual_roughness: None,
            metallic: 0.0,
            emissive: None,
            lightmap_exposure: None,
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
            direction_space: DirectionSpace::World,
            perceptual_roughness: None,
            metallic: 0.0,
            emissive: None,
            lightmap_exposure: None,
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

    /// Sets the gradient direction (normalized; falls back to `Vec3::Y` for a
    /// zero vector). See [`Style::with_direction_space`] for the space it uses.
    #[must_use]
    pub fn with_direction(mut self, direction: Vec3) -> Self {
        self.direction = direction.normalize_or(Vec3::Y);
        self
    }

    /// Chooses whether the gradient direction is a fixed world-space vector or
    /// rotates with the entity. See [`DirectionSpace`].
    #[must_use]
    pub fn with_direction_space(mut self, space: DirectionSpace) -> Self {
        self.direction_space = space;
        self
    }

    /// Overrides the perceptual roughness (`0.0` mirror-like, `1.0` fully matte).
    #[must_use]
    pub fn with_perceptual_roughness(mut self, roughness: f32) -> Self {
        self.perceptual_roughness = Some(roughness.clamp(0.0, 1.0));
        self
    }

    /// Sets the metallic factor (`0.0` dielectric, `1.0` metal).
    #[must_use]
    pub fn with_metallic(mut self, metallic: f32) -> Self {
        self.metallic = metallic.clamp(0.0, 1.0);
        self
    }

    /// Sets the emissive colour (linear RGB; values above 1 glow).
    #[must_use]
    pub fn with_emissive(mut self, emissive: LinearRgba) -> Self {
        self.emissive = Some(emissive);
        self
    }

    /// Sets the exposure applied when the entity carries a Bevy `Lightmap`.
    ///
    /// Gradient materials honour lightmaps: the baked texture supplies the
    /// indirect diffuse light and the gradient colours the surface on top. The
    /// mesh needs `UV_1`; see [`Shape::with_planar_uv1`](crate::Shape::with_planar_uv1).
    #[must_use]
    pub fn with_lightmap_exposure(mut self, exposure: f32) -> Self {
        self.lightmap_exposure = Some(exposure.max(0.0));
        self
    }

    /// Multiplies the alpha of both gradient colours by `alpha` (clamped to
    /// `0.0..=1.0`). [`Style::lerp`] blends the resulting colours, so the alpha
    /// fades smoothly.
    ///
    /// An opaque single-colour style becomes a [`BlendMode::Transparent`] style
    /// when `alpha < 1`. Gradient styles render translucent whenever either
    /// colour has alpha below 1.
    ///
    /// # Shadows
    ///
    /// Translucent materials still render in Bevy's shadow pass (only alpha-tested
    /// there), so a ghost preview would cast a full shadow. Check
    /// [`Style::casts_shadows`] and add Bevy's `NotShadowCaster` component to the
    /// entity when it returns `false`.
    #[must_use]
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        let alpha = if alpha.is_nan() { 1.0 } else { alpha.clamp(0.0, 1.0) };
        self.color = self.color.with_alpha(self.color.alpha() * alpha);
        if let Some(secondary) = self.secondary_color {
            self.secondary_color = Some(secondary.with_alpha(secondary.alpha() * alpha));
        }
        if alpha < 1.0
            && self.color_mode == ColorMode::Single
            && self.blend_mode == BlendMode::Opaque
        {
            self.blend_mode = BlendMode::Transparent;
        }
        self
    }

    /// Returns true if the built material blends with what is behind it.
    pub fn is_translucent(&self) -> bool {
        match self.blend_mode {
            BlendMode::Transparent | BlendMode::Dither => true,
            _ => {
                self.color_mode != ColorMode::Single
                    && (self.color.alpha() < 1.0
                        || self.secondary_color.is_some_and(|c| c.alpha() < 1.0))
            }
        }
    }

    /// Whether entities using this style should cast shadows.
    ///
    /// `false` for translucent styles (see [`Style::is_translucent`]): add
    /// `bevy::light::NotShadowCaster` to the entity in that case, because Bevy
    /// decides shadow casting per entity rather than per material.
    pub fn casts_shadows(&self) -> bool {
        !self.is_translucent()
    }

    /// Hash consistent with `PartialEq` for bit-identical styles, used by the
    /// material pool.
    pub(crate) fn cache_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.blend_mode.hash(&mut h);
        std::mem::discriminant(&self.color_mode).hash(&mut h);
        let mut color = |c: Color| {
            let c = c.to_srgba();
            for v in [c.red, c.green, c.blue, c.alpha] {
                v.to_bits().hash(&mut h);
            }
        };
        color(self.color);
        color(self.secondary_color.unwrap_or(Color::NONE));
        for v in [
            self.direction.x,
            self.direction.y,
            self.direction.z,
            self.metallic,
            self.perceptual_roughness.unwrap_or(-1.0),
            self.lightmap_exposure.unwrap_or(-1.0),
        ] {
            v.to_bits().hash(&mut h);
        }
        self.direction_space.hash(&mut h);
        self.space.hash(&mut h);
        std::mem::discriminant(&self.shape_type).hash(&mut h);
        if let Some(d) = self.dash {
            d.count.hash(&mut h);
            d.spacing.to_bits().hash(&mut h);
            d.offset.to_bits().hash(&mut h);
            d.modifier.to_bits().hash(&mut h);
            std::mem::discriminant(&d.style).hash(&mut h);
        }
        h.finish()
    }

    /// Roughness the constructors use when none is set explicitly.
    fn default_roughness(&self) -> f32 {
        match (self.color_mode, &self.blend_mode) {
            (ColorMode::DualRadial, _) | (ColorMode::Single, BlendMode::Additive) => 0.3,
            _ => 0.7,
        }
    }

    /// Emissive colour the constructors use when none is set explicitly.
    fn default_emissive(&self) -> LinearRgba {
        if self.color_mode == ColorMode::Single && self.blend_mode == BlendMode::Additive {
            let c = self.color.to_linear();
            LinearRgba::rgb(c.red, c.green, c.blue) * 2.5
        } else {
            LinearRgba::BLACK
        }
    }

    /// Blends this style towards `other` by `t` (clamped to `0.0..=1.0`).
    ///
    /// * Colours mix in `self.space`; the result keeps that space.
    /// * Continuous values (direction, dash spacing, offset and modifier,
    ///   roughness, metallic and emissive) interpolate.
    /// * Discrete values (blend mode, colour mode, direction space, dash count
    ///   and style, shape type) follow whichever endpoint is nearer.
    /// * Colour alpha blends with the colours, so `with_alpha` styles fade smoothly.
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
        let perceptual_roughness = match (self.perceptual_roughness, other.perceptual_roughness) {
            (None, None) => None,
            (a, b) => Some(mix(
                a.unwrap_or_else(|| self.default_roughness()),
                b.unwrap_or_else(|| other.default_roughness()),
            )),
        };
        let emissive = match (self.emissive, other.emissive) {
            (None, None) => None,
            (a, b) => {
                let a = a.unwrap_or_else(|| self.default_emissive());
                let b = b.unwrap_or_else(|| other.default_emissive());
                Some(LinearRgba {
                    red: mix(a.red, b.red),
                    green: mix(a.green, b.green),
                    blue: mix(a.blue, b.blue),
                    alpha: mix(a.alpha, b.alpha),
                })
            }
        };
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
            direction_space: near.direction_space,
            space: self.space,
            dash,
            shape_type: near.shape_type,
            perceptual_roughness,
            metallic: mix(self.metallic, other.metallic),
            emissive,
            lightmap_exposure: match (self.lightmap_exposure, other.lightmap_exposure) {
                (None, None) => None,
                (a, b) => Some(mix(a.unwrap_or(1.0), b.unwrap_or(1.0))),
            },
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
    ///
    /// Roughness, metallic and emissive are written to the `StandardMaterial`
    /// base, translucent gradients use alpha blending, and the direction space is
    /// encoded for the shader.
    pub fn build(&self) -> GradMeshMaterial {
        let mut material = self.build_base();

        let uniforms = &mut material.extension.uniforms;
        if self.direction_space == DirectionSpace::Local {
            uniforms.direction.w = 1.0;
        }
        if self.color_mode != ColorMode::Single && self.is_translucent() {
            let a = self.color.alpha();
            let sa = self.secondary_color.map_or(a, |c| c.alpha());
            uniforms.color.w = a;
            uniforms.secondary_color.w = sa;
            if uniforms.flags.y == 0.0 {
                uniforms.flags.y = 1.0;
            }
            material.base.alpha_mode = AlphaMode::Blend;
            material.base.base_color = material.base.base_color.with_alpha(a);
        }

        if let Some(roughness) = self.perceptual_roughness {
            material.base.perceptual_roughness = roughness;
        }
        material.base.metallic = self.metallic;
        if let Some(emissive) = self.emissive {
            material.base.emissive = emissive;
        }
        if let Some(exposure) = self.lightmap_exposure {
            material.base.lightmap_exposure = exposure;
        }
        material
    }

    fn build_base(&self) -> GradMeshMaterial {
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
    fn styles_compare_by_value() {
        let a = Style::dual_directional_up(Color::BLACK, Color::WHITE).with_metallic(0.5);
        assert_eq!(a, a.clone());
        assert_ne!(a, a.clone().with_metallic(0.6));
        assert_ne!(a, a.clone().with_dash(Dash::fixed(4, 0.2)));
        assert_eq!(Dash::fixed(4, 0.2), Dash::fixed(4, 0.2));
        assert_eq!(a.cache_hash(), a.clone().cache_hash());
        assert_eq!(
            crate::appearance::AppearanceSnapshot::color_pulse(1.0, 0.5),
            crate::appearance::AppearanceSnapshot::color_pulse(1.0, 0.5)
        );
    }

    #[test]
    fn shaded_darkens_the_shadow_side() {
        let lit = Color::srgb(0.8, 0.6, 0.4);
        let s = Style::shaded(lit, 0.35);
        assert_eq!(s.color_mode, ColorMode::DualDirectional);
        assert_eq!(s.secondary_color, Some(lit));
        let (dark, light) = (s.color.to_linear(), lit.to_linear());
        assert!((dark.red - light.red * 0.35).abs() < 1e-5);
        assert!((dark.blue - light.blue * 0.35).abs() < 1e-5);
        assert_eq!(dark.alpha, light.alpha);
        // Out-of-range and NaN factors are tamed.
        assert_eq!(Style::shaded(lit, 3.0).color.to_linear(), light);
        assert_eq!(Style::shaded(lit, f32::NAN).color.to_linear(), light);
    }

    #[test]
    fn pbr_fields_reach_the_material_and_blend() {
        let rough = Style::opaque(Color::WHITE);
        assert_eq!(rough.build().base.perceptual_roughness, 0.7);
        let a = Style::opaque(Color::WHITE)
            .with_perceptual_roughness(0.2)
            .with_metallic(1.0)
            .with_emissive(LinearRgba::rgb(2.0, 0.0, 0.0));
        let m = a.build();
        assert_eq!(m.base.perceptual_roughness, 0.2);
        assert_eq!(m.base.metallic, 1.0);
        assert_eq!(m.base.emissive, LinearRgba::rgb(2.0, 0.0, 0.0));

        let b = Style::opaque(Color::BLACK)
            .with_perceptual_roughness(1.0)
            .with_metallic(0.0)
            .with_emissive(LinearRgba::BLACK);
        let mid = a.lerp(&b, 0.5).build();
        assert!((mid.base.perceptual_roughness - 0.6).abs() < 1e-6);
        assert!((mid.base.metallic - 0.5).abs() < 1e-6);
        assert!((mid.base.emissive.red - 1.0).abs() < 1e-6);
        // A style without overrides blends from the constructor default.
        let from_default = Style::opaque(Color::WHITE).lerp(&b, 0.5).build();
        assert!((from_default.base.perceptual_roughness - 0.85).abs() < 1e-6);
        assert_eq!(a.lerp(&b, 0.0).metallic, 1.0);
    }

    #[test]
    fn lightmap_exposure_is_written_to_the_material() {
        assert_eq!(Style::opaque(Color::WHITE).build().base.lightmap_exposure, 1.0);
        let m = Style::opaque(Color::WHITE).with_lightmap_exposure(4000.0).build();
        assert_eq!(m.base.lightmap_exposure, 4000.0);
    }

    #[test]
    fn direction_space_is_encoded_and_documented_default_is_world() {
        let world = Style::dual_directional(Color::BLACK, Color::WHITE, Vec3::Z);
        assert_eq!(world.direction_space, DirectionSpace::World);
        assert_eq!(world.build().extension.uniforms.direction.w, 0.0);
        let local = world.with_direction_space(DirectionSpace::Local);
        let u = local.build().extension.uniforms;
        assert_eq!(u.direction.w, 1.0);
        assert_eq!((u.direction.x, u.direction.y, u.direction.z), (0.0, 0.0, 1.0));
    }

    #[test]
    fn with_alpha_scales_both_colors_and_blends() {
        let s = Style::shaded(Color::srgb(1.0, 1.0, 1.0), 0.5).with_alpha(0.5);
        assert_eq!(s.color.alpha(), 0.5);
        assert_eq!(s.secondary_color.unwrap().alpha(), 0.5);
        assert!(s.is_translucent());
        assert!(!s.casts_shadows());
        let m = s.build();
        assert_eq!(m.base.alpha_mode, AlphaMode::Blend);
        assert_eq!(m.extension.uniforms.color.w, 0.5);
        assert_eq!(m.extension.uniforms.secondary_color.w, 0.5);
        assert_eq!(m.extension.uniforms.flags.y, 1.0);

        // Alpha multiplies and lerp fades it.
        let faded = s.clone().with_alpha(0.5);
        assert!((faded.color.alpha() - 0.25).abs() < 1e-6);
        let opaque = Style::shaded(Color::WHITE, 0.5);
        let mid = opaque.lerp(&s, 0.5);
        assert!((mid.color.alpha() - 0.75).abs() < 1e-3, "{}", mid.color.alpha());

        // Single-colour opaque styles switch to the transparent blend mode.
        let single = Style::opaque(Color::WHITE).with_alpha(0.4);
        assert_eq!(single.blend_mode, BlendMode::Transparent);
        assert!((single.build().extension.uniforms.color.w - 0.4).abs() < 1e-6);
        assert_eq!(Style::opaque(Color::WHITE).with_alpha(1.0).blend_mode, BlendMode::Opaque);
        assert!(Style::opaque(Color::WHITE).casts_shadows());
    }

    #[test]
    fn pool_shares_materials_between_equal_styles() {
        use crate::material_interning::GradMeshMaterialPool;
        let mut pool = GradMeshMaterialPool::default();
        let mut assets = Assets::<GradMeshMaterial>::default();
        let a = Style::shaded(Color::srgb(0.5, 0.5, 0.5), 0.4);
        let h1 = pool.get_or_add(&a, &mut assets);
        let h2 = pool.get_or_add(&a.clone(), &mut assets);
        let h3 = pool.get_or_add(&a.clone().with_metallic(1.0), &mut assets);
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
        assert_eq!(assets.len(), 2);
        assert_eq!(pool.hits(), 1);
        assert_eq!(pool.misses(), 2);
    }

    #[test]
    fn dual_radial_style() {
        let style = Style::dual_radial(Color::srgb(1.0, 1.0, 1.0), Color::srgb(0.0, 0.0, 0.0));
        assert_eq!(style.color_mode, ColorMode::DualRadial);
    }
}
