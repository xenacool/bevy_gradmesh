//! Colour space used when blending two colours.
//!
//! The same [`MixSpace`] selects the space for shader gradients (dual
//! directional / dual radial) and for CPU-side blending ([`MixSpace::mix`],
//! `Style::lerp`), so a gradient and an animated blend of the same two colours
//! always agree.

use bevy::color::{Color, LinearRgba, Mix, Oklaba, Oklcha, Srgba};

/// Colour space in which two colours are interpolated.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum MixSpace {
    /// Perceptually uniform; avoids muddy midpoints. The default.
    #[default]
    Oklab,
    /// Perceptually uniform, interpolating hue around the colour wheel along
    /// the shortest arc. Keeps chroma, so blends stay vivid.
    Oklch,
    /// Physically correct light mixing.
    LinearRgb,
    /// Mixes the gamma-encoded components as picked in a colour picker. Cheap,
    /// but darkens midpoints between saturated colours.
    Srgb,
}

impl MixSpace {
    /// Every space, in GPU-code order.
    pub const ALL: [Self; 4] = [Self::Oklab, Self::Oklch, Self::LinearRgb, Self::Srgb];

    /// Code stored in the `w` component of the shader `flags` vector.
    ///
    /// `Oklab` is `0.0` on purpose: zero-initialised buffers get the default.
    #[must_use]
    pub const fn gpu_code(self) -> f32 {
        match self {
            Self::Oklab => 0.0,
            Self::Oklch => 1.0,
            Self::LinearRgb => 2.0,
            Self::Srgb => 3.0,
        }
    }

    /// Inverse of [`MixSpace::gpu_code`]; unknown codes decode to the default.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_gpu_code(code: f32) -> Self {
        match code.round() as i32 {
            1 => Self::Oklch,
            2 => Self::LinearRgb,
            3 => Self::Srgb,
            _ => Self::Oklab,
        }
    }

    /// Interpolates `a` towards `b` by `t` in this space (alpha included).
    #[must_use]
    pub fn mix(self, a: Color, b: Color, t: f32) -> Color {
        match self {
            Self::Oklab => Oklaba::from(a).mix(&Oklaba::from(b), t).into(),
            Self::Oklch => Oklcha::from(a).mix(&Oklcha::from(b), t).into(),
            Self::LinearRgb => LinearRgba::from(a).mix(&LinearRgba::from(b), t).into(),
            Self::Srgb => Srgba::from(a).mix(&Srgba::from(b), t).into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::color::Alpha;

    fn close(a: Color, b: Color) -> bool {
        let (a, b) = (a.to_srgba(), b.to_srgba());
        (a.red - b.red).abs() < 1e-3
            && (a.green - b.green).abs() < 1e-3
            && (a.blue - b.blue).abs() < 1e-3
            && (a.alpha - b.alpha).abs() < 1e-3
    }

    #[test]
    fn default_is_oklab_and_encodes_as_zero() {
        assert_eq!(MixSpace::default(), MixSpace::Oklab);
        assert_eq!(MixSpace::default().gpu_code(), 0.0);
    }

    #[test]
    fn gpu_code_round_trips() {
        for space in MixSpace::ALL {
            assert_eq!(MixSpace::from_gpu_code(space.gpu_code()), space);
        }
        assert_eq!(MixSpace::from_gpu_code(99.0), MixSpace::Oklab);
    }

    #[test]
    fn gpu_codes_are_unique() {
        for (i, a) in MixSpace::ALL.iter().enumerate() {
            for b in &MixSpace::ALL[i + 1..] {
                assert_ne!(a.gpu_code(), b.gpu_code());
            }
        }
    }

    #[test]
    fn endpoints_are_preserved_in_every_space() {
        let a = Color::srgb(1.0, 0.8, 0.9);
        let b = Color::srgb(0.2, 0.3, 0.9);
        for space in MixSpace::ALL {
            assert!(close(space.mix(a, b, 0.0), a), "{space:?} t=0");
            assert!(close(space.mix(a, b, 1.0), b), "{space:?} t=1");
        }
    }

    #[test]
    fn spaces_disagree_at_the_midpoint() {
        let a = Color::srgb(1.0, 0.0, 0.0);
        let b = Color::srgb(0.0, 1.0, 0.0);
        let mids: Vec<_> = MixSpace::ALL
            .iter()
            .map(|s| s.mix(a, b, 0.5).to_srgba())
            .collect();
        for (i, x) in mids.iter().enumerate() {
            for y in &mids[i + 1..] {
                assert!(
                    (x.red - y.red).abs() + (x.green - y.green).abs() + (x.blue - y.blue).abs()
                        > 1e-3
                );
            }
        }
    }

    #[test]
    fn oklch_takes_the_shortest_hue_arc() {
        // Red and blue-ish magenta sit either side of the hue wrap; the midpoint
        // must stay in the red-magenta range and not sweep through green.
        let a = Color::srgb(1.0, 0.0, 0.2);
        let b = Color::srgb(1.0, 0.0, 0.8);
        let mid = MixSpace::Oklch.mix(a, b, 0.5).to_srgba();
        assert!(mid.green < mid.red && mid.green < mid.blue + 0.2);
        assert!(mid.red > 0.5);
    }

    #[test]
    fn alpha_is_interpolated() {
        let a = Color::srgba(1.0, 1.0, 1.0, 0.0);
        let b = Color::srgba(1.0, 1.0, 1.0, 1.0);
        for space in MixSpace::ALL {
            assert!((space.mix(a, b, 0.5).alpha() - 0.5).abs() < 1e-3);
        }
    }
}
