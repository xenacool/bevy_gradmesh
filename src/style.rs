//! Material style descriptors for procedural shading.
//!
//! Provides style methods: opaque, transparent, dual_directional, dual_radial.

use bevy::asset::Handle;
#[cfg(feature = "pbr")]
use bevy::asset::Assets;
use bevy::color::{Alpha, Color, LinearRgba};
use bevy::math::Vec3;
use bevy::prelude::Shader;
#[cfg(feature = "pbr")]
use bevy_material::AlphaMode;

#[cfg(feature = "pbr")]
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
    /// Custom shader passthrough - uses the provided shader for fragment rendering.
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
    /// Colour by the world height of each fragment (see [`Style::with_height_bands`]).
    HeightBands,
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
    /// No shape hint - uses generic diagonal UV progress.
    Generic,
    /// Disc/circle - uses angular progress around center.
    Disc,
    /// Rectangle - uses perimeter tracing around edges.
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

/// Procedural tint variation: smooth value noise in the entity's local space.
///
/// Gives stone, plaster and thatch some variety without textures or extra
/// meshes. Set with [`Style::with_noise`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Noise {
    /// Noise cells per unit of local space (higher is finer grained).
    pub scale: f32,
    /// How strongly the noise darkens and brightens the colour (`0.0..=1.0`).
    pub amount: f32,
}

/// Procedural stripes: a sine pattern across the entity's local space.
///
/// Gives timber grain, thatch and bands without textures. Set with
/// [`Style::with_stripes`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stripes {
    /// Direction the stripes advance in, in local space (normalized).
    pub direction: Vec3,
    /// Stripe periods per unit of local space.
    pub frequency: f32,
    /// How strongly the stripes darken and brighten the colour (`0.0..=1.0`).
    pub amount: f32,
}

/// View-dependent rim light, added as emission towards the silhouette.
///
/// Set with [`Style::with_rim`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rim {
    /// Rim colour (linear; values above 1 glow).
    pub color: LinearRgba,
    /// Falloff exponent: higher values hug the silhouette more tightly.
    pub power: f32,
}

/// Most bands a [`Style::with_height_bands`] style can hold.
pub const MAX_HEIGHT_BANDS: usize = 4;

/// Colours chosen by world height, set with [`Style::with_height_bands`].
///
/// Unused slots repeat the last band's colour at an unreachable height, so the
/// shader needs no band count.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeightBands {
    /// Band colours (linear), lowest band first.
    pub colors: [LinearRgba; MAX_HEIGHT_BANDS],
    /// Height each band starts at, ascending. Unused slots hold `f32::MAX`.
    pub heights: [f32; MAX_HEIGHT_BANDS],
    /// Number of real bands (`1..=4`).
    pub count: usize,
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
    /// Procedural noise tint (`GradMeshMaterial` only).
    pub noise: Option<Noise>,
    /// Procedural stripe tint (`GradMeshMaterial` only).
    pub stripes: Option<Stripes>,
    /// View-dependent rim light (`GradMeshMaterial` only).
    pub rim: Option<Rim>,
    /// Colours by world height (`GradMeshMaterial` only), see [`Style::with_height_bands`].
    pub height_bands: Option<HeightBands>,
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
            noise: None,
            stripes: None,
            rim: None,
            height_bands: None,
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
            noise: None,
            stripes: None,
            rim: None,
            height_bands: None,
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
            noise: None,
            stripes: None,
            rim: None,
            height_bands: None,
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
            noise: None,
            stripes: None,
            rim: None,
            height_bands: None,
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
            noise: None,
            stripes: None,
            rim: None,
            height_bands: None,
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
            noise: None,
            stripes: None,
            rim: None,
            height_bands: None,
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
            noise: None,
            stripes: None,
            rim: None,
            height_bands: None,
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
            noise: None,
            stripes: None,
            rim: None,
            height_bands: None,
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
            noise: None,
            stripes: None,
            rim: None,
            height_bands: None,
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
            noise: None,
            stripes: None,
            rim: None,
            height_bands: None,
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
            noise: None,
            stripes: None,
            rim: None,
            height_bands: None,
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
            noise: None,
            stripes: None,
            rim: None,
            height_bands: None,
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
            noise: None,
            stripes: None,
            rim: None,
            height_bands: None,
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
            noise: None,
            stripes: None,
            rim: None,
            height_bands: None,
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

    /// Adds procedural noise: smooth tint variation in the entity's local space.
    ///
    /// `scale` is the number of noise cells per local unit and `amount`
    /// (`0.0..=1.0`) how far the colour is pushed darker and lighter. Because
    /// the pattern lives in local space it sticks to the entity as it moves,
    /// rotates and scales. Applies to [`GradMeshMaterial`]; `Style::lerp` blends it.
    #[must_use]
    pub fn with_noise(mut self, scale: f32, amount: f32) -> Self {
        self.noise = Some(Noise {
            scale: scale.max(0.0),
            amount: amount.clamp(0.0, 1.0),
        });
        self
    }

    /// Adds procedural stripes along `direction` (local space, normalized;
    /// falls back to `Vec3::Y` for a zero vector).
    ///
    /// `frequency` is the number of stripe periods per local unit and `amount`
    /// (`0.0..=1.0`) how far the colour is pushed darker and lighter.
    /// Applies to [`GradMeshMaterial`]; `Style::lerp` blends it.
    #[must_use]
    pub fn with_stripes(mut self, direction: Vec3, frequency: f32, amount: f32) -> Self {
        self.stripes = Some(Stripes {
            direction: direction.normalize_or(Vec3::Y),
            frequency: frequency.max(0.0),
            amount: amount.clamp(0.0, 1.0),
        });
        self
    }

    /// Adds a view-dependent rim light: `color` is added as emission where the
    /// surface turns away from the camera, shaped by `power` (clamped to at
    /// least `0.1`; higher values give a thinner rim).
    ///
    /// Use it to highlight a shape (for example the nearest petal of a radial
    /// menu) without a second mesh. Applies to [`GradMeshMaterial`];
    /// `Style::lerp` fades it in and out.
    #[must_use]
    pub fn with_rim(mut self, color: Color, power: f32) -> Self {
        self.rim = Some(Rim {
            color: color.to_linear(),
            power: power.max(0.1),
        });
        self
    }

    /// Colours the shape by the world height of each fragment: a band's colour
    /// holds from its height up to the next band, with a hard step between them
    /// (no blending). Fragments below the first band take its colour.
    ///
    /// `bands` is `(height, colour)` pairs in any order (they are sorted by
    /// height). At most [`MAX_HEIGHT_BANDS`] (4) fit in the shader: given more,
    /// the four lowest are kept and a warning is logged, the same in debug and
    /// release builds. An empty slice leaves the style unchanged. The
    /// style's own colour no longer shows, but noise, stripes, rim light and
    /// vertex tints still apply. Heights are in world space, so it suits terrain
    /// that stays put. Applies to [`GradMeshMaterial`] only; the instanced,
    /// standalone and `Mesh2d` paths ignore it. [`Style::lerp`] blends band
    /// colours and heights when both styles have bands.
    ///
    /// To state the heights as fractions of a shape's own height, so that
    /// re-scaling the shape keeps the look, see [`Style::with_height_bands_on`].
    #[must_use]
    pub fn with_height_bands(mut self, bands: &[(f32, Color)]) -> Self {
        if bands.is_empty() {
            return self;
        }
        let mut sorted: Vec<(f32, LinearRgba)> = bands
            .iter()
            .map(|(h, c)| (if h.is_nan() { 0.0 } else { *h }, c.to_linear()))
            .collect();
        sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
        if sorted.len() > MAX_HEIGHT_BANDS {
            bevy::log::warn!(
                "height bands: {} given but at most {MAX_HEIGHT_BANDS} fit; keeping the lowest {MAX_HEIGHT_BANDS}",
                sorted.len()
            );
            sorted.truncate(MAX_HEIGHT_BANDS);
        }
        let last = sorted[sorted.len() - 1].1;
        let mut out = HeightBands {
            colors: [last; MAX_HEIGHT_BANDS],
            heights: [f32::MAX; MAX_HEIGHT_BANDS],
            count: sorted.len(),
        };
        for (i, (h, c)) in sorted.into_iter().enumerate() {
            out.colors[i] = c;
            out.heights[i] = h;
        }
        self.height_bands = Some(out);
        self.color_mode = ColorMode::HeightBands;
        self
    }

    /// Like [`Style::with_height_bands`], with each band's height given as a
    /// fraction of the vertical extent of the land of `shape` as placed by
    /// `transform`: `0.0` is its lowest point and `1.0` its highest, and
    /// fractions outside `0.0..=1.0` reach below or above it.
    ///
    /// The fractions are resolved here, against [`Shape::surface_bounds`](crate::Shape::surface_bounds), so
    /// they cannot be left unresolved in the style and the skirts of a
    /// heightfield (which hang below the land) never shift the bands. The
    /// resolved heights are world heights, like the absolute form: the
    /// `transform` should be the one the shape is shown with, and a shape that is
    /// tilted is measured along the world `Y` axis through its centre.
    /// A shape that is flat (or empty) has no extent to spread the fractions
    /// over, so all bands start at its height and only the highest shows.
    /// Build the style again when the shape's heights or transform change; the
    /// result is an ordinary style that [`Style::lerp`] blends and the pool shares.
    #[must_use]
    pub fn with_height_bands_on(
        self,
        bands: &[(f32, Color)],
        shape: &crate::shape::Shape,
        transform: &bevy::transform::components::Transform,
    ) -> Self {
        let b = shape.surface_bounds();
        let c = b.center();
        let world_height = |fraction: f32| {
            let fraction = if fraction.is_nan() { 0.0 } else { fraction };
            let local = b.min.y + (b.max.y - b.min.y) * fraction;
            transform.transform_point(Vec3::new(c.x, local, c.z)).y
        };
        let absolute: Vec<(f32, Color)> = bands.iter().map(|&(f, col)| (world_height(f), col)).collect();
        self.with_height_bands(&absolute)
    }

    /// Like [`Style::with_height_bands`], with each band's height given as a
    /// fraction of an explicit world height range `lowest..highest`: `0.0` is
    /// `lowest` and `1.0` is `highest`, and fractions outside `0.0..=1.0` reach
    /// below or above the range.
    ///
    /// Use it for terrain made of several chunks: the range belongs to the
    /// whole terrain, so every chunk resolves the same fractions to the same
    /// world heights and colours agree along shared edges, which
    /// [`Style::with_height_bands_on`] cannot promise because it measures one
    /// shape. Re-scaling the terrain's heights then only means changing the
    /// range. A range that is empty, reversed or not finite has nothing to
    /// spread the fractions over, so all bands start at `lowest` (or at `0.0`
    /// if that is not finite) and only the highest shows.
    #[must_use]
    pub fn with_height_bands_in(self, bands: &[(f32, Color)], lowest: f32, highest: f32) -> Self {
        let lowest = if lowest.is_finite() { lowest } else { 0.0 };
        let span = if highest.is_finite() { (highest - lowest).max(0.0) } else { 0.0 };
        let absolute: Vec<(f32, Color)> = bands
            .iter()
            .map(|&(f, col)| (lowest + span * if f.is_nan() { 0.0 } else { f }, col))
            .collect();
        self.with_height_bands(&absolute)
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
        let alpha = if alpha.is_nan() {
            1.0
        } else {
            alpha.clamp(0.0, 1.0)
        };
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
    #[cfg_attr(not(feature = "pbr"), allow(dead_code))]
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
        ] {
            v.to_bits().hash(&mut h);
        }
        if let Some(b) = &self.height_bands {
            b.count.hash(&mut h);
            for i in 0..MAX_HEIGHT_BANDS {
                b.heights[i].to_bits().hash(&mut h);
                let c = b.colors[i];
                for v in [c.red, c.green, c.blue, c.alpha] {
                    v.to_bits().hash(&mut h);
                }
            }
        }
        self.direction_space.hash(&mut h);
        if let Some(n) = self.noise {
            [1.0, n.scale, n.amount].map(f32::to_bits).hash(&mut h);
        }
        if let Some(st) = self.stripes {
            [
                2.0,
                st.direction.x,
                st.direction.y,
                st.direction.z,
                st.frequency,
                st.amount,
            ]
            .map(f32::to_bits)
            .hash(&mut h);
        }
        if let Some(r) = self.rim {
            [
                3.0,
                r.color.red,
                r.color.green,
                r.color.blue,
                r.color.alpha,
                r.power,
            ]
            .map(f32::to_bits)
            .hash(&mut h);
        }
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
            (a, b) => Some(
                self.space
                    .mix(a.unwrap_or(self.color), b.unwrap_or(other.color), t),
            ),
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

        // Surface detail fades in and out: a missing effect counts as the other
        // endpoint's effect with zero strength.
        let noise = if t <= 0.0 {
            self.noise
        } else if t >= 1.0 {
            other.noise
        } else {
            match (self.noise, other.noise) {
                (None, None) => None,
                (a, b) => {
                    let fill = |n: Option<Noise>, o: Option<Noise>| {
                        n.unwrap_or(Noise {
                            scale: o.map_or(1.0, |o| o.scale),
                            amount: 0.0,
                        })
                    };
                    let (a, b) = (fill(a, b), fill(b, a));
                    Some(Noise {
                        scale: mix(a.scale, b.scale),
                        amount: mix(a.amount, b.amount),
                    })
                }
            }
        };
        let stripes = if t <= 0.0 {
            self.stripes
        } else if t >= 1.0 {
            other.stripes
        } else {
            match (self.stripes, other.stripes) {
                (None, None) => None,
                (a, b) => {
                    let fill = |n: Option<Stripes>, o: Option<Stripes>| {
                        n.unwrap_or(Stripes {
                            direction: o.map_or(Vec3::Y, |o| o.direction),
                            frequency: o.map_or(1.0, |o| o.frequency),
                            amount: 0.0,
                        })
                    };
                    let (a, b) = (fill(a, b), fill(b, a));
                    Some(Stripes {
                        direction: a
                            .direction
                            .lerp(b.direction, t)
                            .normalize_or(near.stripes.map_or(a.direction, |s| s.direction)),
                        frequency: mix(a.frequency, b.frequency),
                        amount: mix(a.amount, b.amount),
                    })
                }
            }
        };
        let rim = if t <= 0.0 {
            self.rim
        } else if t >= 1.0 {
            other.rim
        } else {
            match (self.rim, other.rim) {
                (None, None) => None,
                (a, b) => {
                    let fill = |n: Option<Rim>, o: Option<Rim>| {
                        n.unwrap_or(Rim {
                            color: LinearRgba::BLACK,
                            power: o.map_or(1.0, |o| o.power),
                        })
                    };
                    let (a, b) = (fill(a, b), fill(b, a));
                    Some(Rim {
                        color: LinearRgba {
                            red: mix(a.color.red, b.color.red),
                            green: mix(a.color.green, b.color.green),
                            blue: mix(a.color.blue, b.color.blue),
                            alpha: mix(a.color.alpha, b.color.alpha),
                        },
                        power: mix(a.power, b.power),
                    })
                }
            }
        };
        let height_bands = if t <= 0.0 {
            self.height_bands
        } else if t >= 1.0 {
            other.height_bands
        } else {
            match (self.height_bands, other.height_bands) {
                (Some(a), Some(b)) => {
                    let mut out = a;
                    for i in 0..MAX_HEIGHT_BANDS {
                        let (ca, cb) = (a.colors[i], b.colors[i]);
                        out.colors[i] = LinearRgba {
                            red: mix(ca.red, cb.red),
                            green: mix(ca.green, cb.green),
                            blue: mix(ca.blue, cb.blue),
                            alpha: mix(ca.alpha, cb.alpha),
                        };
                        out.heights[i] = if a.heights[i] == f32::MAX || b.heights[i] == f32::MAX {
                            near.height_bands.map_or(f32::MAX, |n| n.heights[i])
                        } else {
                            mix(a.heights[i], b.heights[i])
                        };
                    }
                    out.count = near.height_bands.map_or(a.count, |n| n.count);
                    Some(out)
                }
                _ => near.height_bands,
            }
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
            noise,
            stripes,
            rim,
            height_bands,
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
    #[cfg(feature = "pbr")]
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
    #[cfg(feature = "pbr")]
    pub fn build(&self) -> GradMeshMaterial {
        let mut material = self.build_base();

        let uniforms = &mut material.extension.uniforms;
        self.apply_height_bands(uniforms);
        if self.direction_space == DirectionSpace::Local {
            uniforms.direction.w = 1.0;
        }
        if let Some(n) = self.noise {
            uniforms.noise = crate::material::Vec4::new(n.scale, n.amount, 0.0, 0.0);
        }
        if let Some(st) = self.stripes {
            uniforms.stripes = crate::material::Vec4::new(
                st.direction.x,
                st.direction.y,
                st.direction.z,
                st.frequency,
            );
            uniforms.stripes_amount = crate::material::Vec4::new(st.amount, 0.0, 0.0, 0.0);
        }
        if let Some(r) = self.rim {
            uniforms.rim =
                crate::material::Vec4::new(r.color.red, r.color.green, r.color.blue, r.power);
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
        material
    }

    /// Writes the height-band mode code, colours and heights into `uniforms`.
    fn apply_height_bands(&self, uniforms: &mut crate::material::GradMeshUniformBuffer) {
        use crate::material::Vec4;
        let Some(bands) = self.height_bands.filter(|_| self.color_mode == ColorMode::HeightBands)
        else {
            return;
        };
        uniforms.flags.x = 3.0;
        for (slot, c) in uniforms.bands.iter_mut().zip(bands.colors) {
            *slot = Vec4::new(c.red, c.green, c.blue, c.alpha);
        }
        uniforms.band_heights = Vec4::from_array(bands.heights);
    }

    /// Builds the shader uniforms for this style without any `bevy_pbr` types.
    ///
    /// This is what the 2D materials use; it matches the uniforms that
    /// [`Style::build`] writes into the extended material.
    pub fn build_uniforms(&self) -> crate::material::GradMeshUniformBuffer {
        use crate::material::{Vec4, presets};

        let srgba = self.color.to_srgba();
        let (r, g, b, a) = (srgba.red, srgba.green, srgba.blue, srgba.alpha);
        let (sr, sg, sb, sa) = self.secondary_color.map_or((r, g, b, a), |c| {
            let s = c.to_srgba();
            (s.red, s.green, s.blue, s.alpha)
        });

        let mut uniforms = 'base: {
            if let Some(dash) = self.dash.as_ref().filter(|d| d.is_enabled()) {
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
                break 'base presets::dashed(
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
            let mut u = match self.color_mode {
                ColorMode::DualDirectional => presets::dual_directional(
                    [r, g, b],
                    [sr, sg, sb],
                    [self.direction.x, self.direction.y, self.direction.z],
                ),
                ColorMode::DualRadial => {
                    if a < 1.0 || sa < 1.0 {
                        presets::dual_radial_alpha([r, g, b, a], [sr, sg, sb, sa])
                    } else {
                        presets::dual_radial([r, g, b], [sr, sg, sb])
                    }
                }
                ColorMode::Single | ColorMode::HeightBands => match &self.blend_mode {
                    BlendMode::Opaque | BlendMode::Custom(_) => presets::opaque(r, g, b),
                    BlendMode::Transparent => presets::transparent(r, g, b, a),
                    BlendMode::Additive => presets::additive(r, g, b),
                    BlendMode::Multiplicative => presets::multiplicative(r, g, b),
                    BlendMode::Screen => presets::screen(r, g, b),
                    BlendMode::Dither => presets::dither(r, g, b, a),
                    BlendMode::Subtractive => presets::subtractive(r, g, b),
                    BlendMode::Lighten => presets::lighten(r, g, b),
                    BlendMode::ColorDodge => presets::color_dodge(r, g, b),
                    BlendMode::LinearBurn => presets::linear_burn(r, g, b),
                    BlendMode::ColorBurn => presets::color_burn(r, g, b),
                    BlendMode::Darken => presets::darken(r, g, b),
                },
            };
            if self.color_mode != ColorMode::Single {
                u.flags.w = self.space.gpu_code();
            }
            u
        };

        if self.direction_space == DirectionSpace::Local {
            uniforms.direction.w = 1.0;
        }
        if let Some(n) = self.noise {
            uniforms.noise = Vec4::new(n.scale, n.amount, 0.0, 0.0);
        }
        if let Some(st) = self.stripes {
            uniforms.stripes =
                Vec4::new(st.direction.x, st.direction.y, st.direction.z, st.frequency);
            uniforms.stripes_amount = Vec4::new(st.amount, 0.0, 0.0, 0.0);
        }
        if let Some(r) = self.rim {
            uniforms.rim = Vec4::new(r.color.red, r.color.green, r.color.blue, r.power);
        }
        if self.color_mode != ColorMode::Single && self.is_translucent() {
            let a = self.color.alpha();
            let sa = self.secondary_color.map_or(a, |c| c.alpha());
            uniforms.color.w = a;
            uniforms.secondary_color.w = sa;
            if uniforms.flags.y == 0.0 {
                uniforms.flags.y = 1.0;
            }
        }
        self.apply_height_bands(&mut uniforms);
        uniforms
    }

    #[cfg(feature = "pbr")]
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
            ColorMode::Single | ColorMode::HeightBands => match &self.blend_mode {
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
        assert_eq!(mid.color, MixSpace::Oklch.mix(a.color, b.color, 0.5),);
        let close = |x: Color, y: Color| {
            let (x, y) = (x.to_srgba(), y.to_srgba());
            (x.red - y.red).abs() + (x.green - y.green).abs() + (x.blue - y.blue).abs() < 1e-3
        };
        let (l0, l1) = (a.lerp(&b, 0.0), a.lerp(&b, 1.0));
        assert!(close(l0.color, a.color));
        assert!(close(
            l1.secondary_color.unwrap(),
            b.secondary_color.unwrap()
        ));
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
    fn direction_space_is_encoded_and_documented_default_is_world() {
        let world = Style::dual_directional(Color::BLACK, Color::WHITE, Vec3::Z);
        assert_eq!(world.direction_space, DirectionSpace::World);
        assert_eq!(world.build().extension.uniforms.direction.w, 0.0);
        let local = world.with_direction_space(DirectionSpace::Local);
        let u = local.build().extension.uniforms;
        assert_eq!(u.direction.w, 1.0);
        assert_eq!(
            (u.direction.x, u.direction.y, u.direction.z),
            (0.0, 0.0, 1.0)
        );
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
        assert!(
            (mid.color.alpha() - 0.75).abs() < 1e-3,
            "{}",
            mid.color.alpha()
        );

        // Single-colour opaque styles switch to the transparent blend mode.
        let single = Style::opaque(Color::WHITE).with_alpha(0.4);
        assert_eq!(single.blend_mode, BlendMode::Transparent);
        assert!((single.build().extension.uniforms.color.w - 0.4).abs() < 1e-6);
        assert_eq!(
            Style::opaque(Color::WHITE).with_alpha(1.0).blend_mode,
            BlendMode::Opaque
        );
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

#[cfg(test)]
mod height_band_tests {
    use super::*;

    const GRASS: Color = Color::srgb(0.2, 0.6, 0.2);
    const ROCK: Color = Color::srgb(0.5, 0.5, 0.5);
    const SNOW: Color = Color::srgb(0.95, 0.95, 1.0);

    fn terrain() -> Style {
        Style::opaque(Color::WHITE).with_height_bands(&[(0.0, GRASS), (1.0, ROCK), (2.0, SNOW)])
    }

    #[test]
    fn bands_are_sorted_and_padded() {
        let s = Style::opaque(Color::WHITE).with_height_bands(&[(2.0, SNOW), (0.0, GRASS), (1.0, ROCK)]);
        assert_eq!(s.color_mode, ColorMode::HeightBands);
        let b = s.height_bands.unwrap();
        assert_eq!(b.count, 3);
        assert_eq!(b.heights, [0.0, 1.0, 2.0, f32::MAX]);
        assert_eq!(b.colors[0], GRASS.to_linear());
        assert_eq!(b.colors[2], SNOW.to_linear());
        // The unused slot repeats the last band.
        assert_eq!(b.colors[3], SNOW.to_linear());
    }

    #[test]
    fn empty_bands_change_nothing() {
        let plain = Style::opaque(Color::WHITE);
        assert_eq!(plain.clone().with_height_bands(&[]), plain);
    }

    #[test]
    fn uniforms_carry_the_bands() {
        let u = terrain().build_uniforms();
        assert_eq!(u.flags.x, 3.0);
        assert_eq!(u.band_heights.as_array(), [0.0, 1.0, 2.0, f32::MAX]);
        let rock = ROCK.to_linear();
        assert_eq!(u.bands[1].as_array(), [rock.red, rock.green, rock.blue, rock.alpha]);
        // Other styles leave the band slots empty and the mode code alone.
        let single = Style::opaque(ROCK).build_uniforms();
        assert_eq!(single.flags.x, 0.0);
        assert_eq!(single.band_heights.as_array(), [0.0; 4]);
    }

    #[cfg(feature = "pbr")]
    #[test]
    fn built_material_matches_the_uniforms() {
        let style = terrain();
        let material = style.build();
        assert_eq!(material.extension.uniforms.flags.x, 3.0);
        assert_eq!(
            material.extension.uniforms.band_heights,
            style.build_uniforms().band_heights
        );
    }

    #[test]
    fn different_bands_are_different_styles() {
        let a = terrain();
        let b = Style::opaque(Color::WHITE).with_height_bands(&[(0.0, GRASS), (1.5, ROCK), (2.0, SNOW)]);
        assert_ne!(a, b);
        assert_ne!(a.cache_hash(), b.cache_hash());
        assert_eq!(a.cache_hash(), terrain().cache_hash());
    }

    #[test]
    fn lerp_blends_band_colours_and_heights() {
        let a = terrain();
        let b = Style::opaque(Color::WHITE).with_height_bands(&[(0.0, SNOW), (3.0, ROCK), (4.0, GRASS)]);
        // The endpoints keep the bands exactly (the plain colour may change colour
        // space, which `Style::lerp` always does).
        assert_eq!(a.lerp(&b, 0.0).height_bands, a.height_bands);
        assert_eq!(a.lerp(&b, 1.0).height_bands, b.height_bands);
        let mid = a.lerp(&b, 0.5).height_bands.unwrap();
        assert_eq!(mid.heights[..3], [0.0, 2.0, 3.0]);
        assert_eq!(mid.heights[3], f32::MAX);
    }

    #[test]
    fn more_than_four_bands_keep_the_lowest_four_in_every_build() {
        // Given out of order: the lowest four by height stay, whatever the order.
        let five: Vec<(f32, Color)> = [4.0, 0.0, 3.0, 1.0, 2.0].iter().map(|&h| (h, GRASS)).collect();
        let b = Style::opaque(Color::WHITE).with_height_bands(&five).height_bands.unwrap();
        assert_eq!(b.count, 4);
        assert_eq!(b.heights, [0.0, 1.0, 2.0, 3.0]);
    }

    fn land(top: f32) -> crate::shape::Shape {
        use bevy::math::{UVec2, Vec2};
        // Heights 0 and `top`, with skirts hanging 0.1 * 10 = 1 below.
        let heights: std::sync::Arc<[f32]> = vec![0.0, top, 0.0, top].into();
        crate::shape::Shape::heightfield(heights, UVec2::new(2, 2), Vec2::splat(10.0))
    }

    fn palette(shape: &crate::shape::Shape, transform: &bevy::transform::components::Transform) -> Style {
        Style::opaque(Color::WHITE).with_height_bands_on(&[(0.0, GRASS), (0.5, ROCK), (1.0, SNOW)], shape, transform)
    }

    #[test]
    fn relative_bands_follow_the_land_not_its_skirts() {
        use bevy::transform::components::Transform;
        let b = palette(&land(8.0), &Transform::IDENTITY).height_bands.unwrap();
        // The land spans 0..8; the skirts (down to -1) do not count.
        assert_eq!(b.count, 3);
        assert_eq!(b.heights[..3], [0.0, 4.0, 8.0]);
    }

    #[test]
    fn rescaled_heights_keep_the_look() {
        use bevy::transform::components::Transform;
        let low = palette(&land(8.0), &Transform::IDENTITY).height_bands.unwrap();
        let tall = palette(&land(80.0), &Transform::IDENTITY).height_bands.unwrap();
        assert_eq!(tall.heights[..3], [0.0, 40.0, 80.0]);
        assert_eq!(low.colors, tall.colors);
        // The same land scaled by its transform lands on the same world heights.
        let scaled = palette(&land(8.0), &Transform::from_scale(Vec3::new(1.0, 10.0, 1.0)))
            .height_bands
            .unwrap();
        assert_eq!(scaled.heights[..3], [0.0, 40.0, 80.0]);
        // And a lifted placement moves the bands with it.
        let lifted = palette(&land(8.0), &Transform::from_xyz(0.0, 100.0, 0.0)).height_bands.unwrap();
        assert_eq!(lifted.heights[..3], [100.0, 104.0, 108.0]);
    }

    #[test]
    fn fractions_outside_the_range_reach_past_the_land() {
        use bevy::transform::components::Transform;
        let s = Style::opaque(Color::WHITE).with_height_bands_on(
            &[(-0.5, GRASS), (1.5, SNOW)],
            &land(8.0),
            &Transform::IDENTITY,
        );
        assert_eq!(s.height_bands.unwrap().heights[..2], [-4.0, 12.0]);
    }

    #[test]
    fn bands_in_an_explicit_range_agree_between_chunks() {
        let bands = [(0.0, GRASS), (0.5, ROCK), (1.0, SNOW)];
        // Two chunks of one terrain with different local heights get the same
        // bands from the terrain's range, which `with_height_bands_on` would not give.
        let a = Style::opaque(Color::WHITE).with_height_bands_in(&bands, -10.0, 30.0);
        let b = Style::opaque(Color::WHITE).with_height_bands_in(&bands, -10.0, 30.0);
        assert_eq!(a, b);
        assert_eq!(a.height_bands.unwrap().heights[..3], [-10.0, 10.0, 30.0]);
        // Fractions outside the range reach past it.
        let past = Style::opaque(Color::WHITE).with_height_bands_in(&[(-0.5, GRASS), (1.5, SNOW)], 0.0, 8.0);
        assert_eq!(past.height_bands.unwrap().heights[..2], [-4.0, 12.0]);
        // A bad range collapses to its lowest point instead of producing NaN.
        for (lo, hi) in [(5.0, 5.0), (5.0, 1.0), (5.0, f32::NAN), (f32::NAN, 1.0)] {
            let s = Style::opaque(Color::WHITE).with_height_bands_in(&bands, lo, hi).height_bands.unwrap();
            assert!(s.heights[..3].iter().all(|h| h.is_finite()), "{lo} {hi}");
        }
    }

    #[test]
    fn a_flat_or_empty_shape_collapses_the_bands_to_its_height() {
        use bevy::transform::components::Transform;
        let flat = palette(&land(0.0), &Transform::IDENTITY).height_bands.unwrap();
        assert_eq!(flat.heights[..3], [0.0, 0.0, 0.0]);
        let empty = palette(&crate::shape::Shape::from_mesh(
            std::sync::Arc::new(bevy::mesh::Mesh::new(
                bevy::mesh::PrimitiveTopology::TriangleList,
                bevy_asset::RenderAssetUsages::default(),
            )),
            1,
        ), &Transform::IDENTITY)
        .height_bands
        .unwrap();
        assert!(empty.heights[..3].iter().all(|h| h.is_finite()));
    }
}

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    /// Two colours are close enough to be considered equal after a colour-space round-trip.
    fn color_close(a: &Color, b: &Color) -> bool {
        let (a, b) = (a.to_srgba(), b.to_srgba());
        (a.red - b.red).abs() < 1e-3
            && (a.green - b.green).abs() < 1e-3
            && (a.blue - b.blue).abs() < 1e-3
            && (a.alpha - b.alpha).abs() < 1e-3
    }

    /// Generate a random sRGB colour component in [0, 1].
    fn color_component() -> impl Strategy<Value = f32> {
        (0.0f32..=1.0f32).prop_map(|v| v)
    }

    /// Generate a random sRGB colour.
    fn random_color() -> impl Strategy<Value = Color> {
        (
            color_component(),
            color_component(),
            color_component(),
            color_component(),
        )
            .prop_map(|(r, g, b, a)| Color::srgba(r, g, b, a))
    }

    /// Generate a random unit direction vector.
    fn random_direction() -> impl Strategy<Value = Vec3> {
        (0.0f32..=std::f32::consts::TAU, 0.0f32..=std::f32::consts::PI)
            .prop_map(|(theta, phi)| {
                let sin_phi = phi.sin();
                Vec3::new(sin_phi * theta.cos(), phi.sin(), sin_phi * theta.sin())
                    .normalize_or(Vec3::Y)
            })
    }

    /// Generate a simple random style (opaque or dual_directional).
    fn random_style() -> impl Strategy<Value = Style> {
        prop_oneof![
            // Opaque style
            2 => (random_color(), random_direction())
                .prop_map(|(color, direction)| {
                    Style::opaque(color).with_direction(direction)
                }),
            // Dual directional style
            2 => (random_color(), random_color(), random_direction())
                .prop_map(|(dark, light, direction)| {
                    Style::dual_directional(dark, light, direction)
                }),
        ]
    }

    proptest! {
        #[test]
        /// Lens law 1: lerp at t=0 returns self (colours may differ slightly due to colour-space round-trip).
        fn lerp_t0_is_self(
            a in random_style(),
            b in random_style(),
        ) {
            let result = a.lerp(&b, 0.0);

            // Colours may shift slightly due to colour-space conversion
            prop_assert!(color_close(&result.color, &a.color),
                "color mismatch at t=0: {:?} vs {:?}", result.color, a.color);

            // Secondary colour: lerp always creates Some when either endpoint has it,
            // but the colour value at t=0 should match a's colour (or a.color if a has None).
            if let Some(ref r) = result.secondary_color {
                let expected = a.secondary_color.unwrap_or(a.color);
                prop_assert!(color_close(r, &expected),
                    "secondary_color value mismatch at t=0: {:?} vs {:?}", r, expected);
            }

            // Discrete fields must match exactly
            prop_assert_eq!(result.blend_mode, a.blend_mode, "blend_mode mismatch at t=0");
            prop_assert_eq!(result.color_mode, a.color_mode, "color_mode mismatch at t=0");
            prop_assert_eq!(result.direction_space, a.direction_space, "direction_space mismatch at t=0");
            prop_assert_eq!(result.space, a.space, "space mismatch at t=0");
            prop_assert_eq!(result.shape_type, a.shape_type, "shape_type mismatch at t=0");

            // Continuous fields
            prop_assert!((result.direction - a.direction).length() < 1e-5,
                "direction mismatch at t=0: {:?} vs {:?}", result.direction, a.direction);
            prop_assert!((result.metallic - a.metallic).abs() < 1e-6,
                "metallic mismatch at t=0");

            // Optional fields
            prop_assert_eq!(result.perceptual_roughness, a.perceptual_roughness,
                "perceptual_roughness mismatch at t=0");
            prop_assert_eq!(result.dash, a.dash, "dash mismatch at t=0");
            prop_assert_eq!(result.noise, a.noise, "noise mismatch at t=0");
            prop_assert_eq!(result.stripes, a.stripes, "stripes mismatch at t=0");
            prop_assert_eq!(result.rim, a.rim, "rim mismatch at t=0");
            prop_assert_eq!(result.height_bands, a.height_bands, "height_bands mismatch at t=0");
        }
    }

    proptest! {
        #[test]
        /// Lens law 2: lerp at t=1 returns other (except space stays self.space).
        fn lerp_t1_is_other(
            a in random_style(),
            b in random_style(),
        ) {
            let result = a.lerp(&b, 1.0);

            // Colours may shift slightly
            prop_assert!(color_close(&result.color, &b.color),
                "color mismatch at t=1: {:?} vs {:?}", result.color, b.color);

            // Secondary colour: lerp always creates Some when either endpoint has it,
            // but the colour value at t=1 should match b's colour (or b.color if b has None).
            if let Some(ref r) = result.secondary_color {
                let expected = b.secondary_color.unwrap_or(b.color);
                prop_assert!(color_close(r, &expected),
                    "secondary_color value mismatch at t=1: {:?} vs {:?}", r, expected);
            }

            // Discrete fields follow other
            prop_assert_eq!(result.blend_mode, b.blend_mode, "blend_mode mismatch at t=1");
            prop_assert_eq!(result.color_mode, b.color_mode, "color_mode mismatch at t=1");
            prop_assert_eq!(result.direction_space, b.direction_space, "direction_space mismatch at t=1");
            prop_assert_eq!(result.shape_type, b.shape_type, "shape_type mismatch at t=1");

            // Space always stays self.space (documented behaviour)
            prop_assert_eq!(result.space, a.space, "space should stay self.space at t=1");

            // Continuous fields
            prop_assert!((result.direction - b.direction).length() < 1e-5,
                "direction mismatch at t=1");
            prop_assert!((result.metallic - b.metallic).abs() < 1e-6,
                "metallic mismatch at t=1");

            // Optional fields
            prop_assert_eq!(result.perceptual_roughness, b.perceptual_roughness,
                "perceptual_roughness mismatch at t=1");
            prop_assert_eq!(result.dash, b.dash, "dash mismatch at t=1");
            prop_assert_eq!(result.noise, b.noise, "noise mismatch at t=1");
            prop_assert_eq!(result.stripes, b.stripes, "stripes mismatch at t=1");
            prop_assert_eq!(result.rim, b.rim, "rim mismatch at t=1");
            prop_assert_eq!(result.height_bands, b.height_bands, "height_bands mismatch at t=1");
        }
    }

    proptest! {
        #[test]
        /// Symmetry: a.lerp(b, t) and b.lerp(a, 1-t) produce equivalent continuous values.
        fn lerp_symmetry(
            a in random_style(),
            b in random_style(),
            t in 0.0f32..=1.0f32,
        ) {
            let ab = a.lerp(&b, t);
            let ba = b.lerp(&a, 1.0 - t);

            // Colours should be close
            prop_assert!(color_close(&ab.color, &ba.color),
                "color asymmetry at t={}: {:?} vs {:?}", t, ab.color, ba.color);

            // Direction should be close (both normalized)
            prop_assert!((ab.direction - ba.direction).length() < 1e-4,
                "direction asymmetry at t={}", t);

            // Metallic should match
            prop_assert!((ab.metallic - ba.metallic).abs() < 1e-6,
                "metallic asymmetry at t={}", t);
        }
    }

    proptest! {
        #[test]
        /// Direction is always normalized after lerp.
        fn lerp_direction_normalized(
            a in random_style(),
            b in random_style(),
            t in 0.0f32..=1.0f32,
        ) {
            let result = a.lerp(&b, t);
            let len = result.direction.length();
            prop_assert!((len - 1.0).abs() < 1e-5,
                "direction not normalized: length={} at t={}", len, t);
        }
    }

    proptest! {
        #[test]
        /// Metallic stays in [0, 1] after lerp.
        fn lerp_metallic_in_range(
            a in random_style(),
            b in random_style(),
            t in 0.0f32..=1.0f32,
        ) {
            let result = a.lerp(&b, t);
            prop_assert!(result.metallic >= 0.0 && result.metallic <= 1.0,
                "metallic out of range: {} at t={}", result.metallic, t);
        }
    }

    proptest! {
        #[test]
        /// Perceptual roughness, when present, stays in [0, 1].
        fn lerp_roughness_in_range(
            a_color in random_color(),
            a_dir in random_direction(),
            a_rough in 0.0f32..=1.0f32,
            b_color in random_color(),
            b_dir in random_direction(),
            b_rough in 0.0f32..=1.0f32,
            t in 0.0f32..=1.0f32,
        ) {
            let a = Style::opaque(a_color).with_direction(a_dir).with_perceptual_roughness(a_rough);
            let b = Style::opaque(b_color).with_direction(b_dir).with_perceptual_roughness(b_rough);
            let result = a.lerp(&b, t);
            if let Some(r) = result.perceptual_roughness {
                prop_assert!(r >= 0.0 && r <= 1.0,
                    "roughness out of range: {} at t={}", r, t);
            }
        }
    }
}
