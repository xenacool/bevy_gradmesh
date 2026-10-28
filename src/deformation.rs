//! Vertex shader deformation system for GPU-side shape animation.
//!
//! Enables runtime shape deformation (wobble, pulse, morph) via per-instance
//! shader attributes, avoiding CPU mesh regeneration for animated shapes.
//!
//! # Design
//!
//! Each instance carries a `DeformationParams` vec4 encoding:
//! - `x`: deformation type (0=None, 1=Wobble, 2=Pulse, 3=Morph)
//! - `y`: amplitude (displacement magnitude)
//! - `z`: frequency (oscillation speed)
//! - `w`: phase (time offset)
//!
//! The vertex shader reads this attribute and displaces vertices before
//! the world-space transform, keeping deformation entirely on the GPU.

use bevy::ecs::prelude::Resource;
use crate::material::Vec4;

/// Deformation type identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeformationType {
    #[default]
    /// No deformation.
    None,
    /// Sinusoidal displacement along vertex normal.
    Wobble,
    /// Uniform scale oscillation (breathing effect).
    Pulse,
    /// Vertex displacement toward a target direction.
    Morph,
}

impl DeformationType {
    /// Returns the numeric code for shader encoding.
    pub fn as_f32(self) -> f32 {
        match self {
            DeformationType::None => 0.0,
            DeformationType::Wobble => 1.0,
            DeformationType::Pulse => 2.0,
            DeformationType::Morph => 3.0,
        }
    }

    /// Decode from a float value read from the shader.
    pub fn from_f32(value: f32) -> Self {
        match value.round() as i32 {
            1 => DeformationType::Wobble,
            2 => DeformationType::Pulse,
            3 => DeformationType::Morph,
            _ => DeformationType::None,
        }
    }
}

/// Deformation parameters for a single instance.
///
/// Encoded as a single `vec4` for GPU upload:
/// - `x`: type code (0=None, 1=Wobble, 2=Pulse, 3=Morph)
/// - `y`: amplitude
/// - `z`: frequency
/// - `w`: phase
#[derive(Debug, Clone, Copy, Default)]
pub struct DeformationParams {
    pub r#type: DeformationType,
    pub amplitude: f32,
    pub frequency: f32,
    pub phase: f32,
}

impl DeformationParams {
    /// No deformation.
    pub fn none() -> Self {
        Self::default()
    }

    /// Sinusoidal wobble along vertex normal.
    ///
    /// # Arguments
    /// * `amplitude` - Displacement magnitude in world units.
    /// * `frequency` - Oscillations per second.
    /// * `phase` - Time offset in seconds.
    pub fn wobble(amplitude: f32, frequency: f32, phase: f32) -> Self {
        Self {
            r#type: DeformationType::Wobble,
            amplitude,
            frequency,
            phase,
        }
    }

    /// Uniform scale oscillation (breathing/pulsing effect).
    ///
    /// # Arguments
    /// * `amplitude` - Scale oscillation magnitude (e.g., 0.1 = ±10% scale).
    /// * `frequency` - Oscillations per second.
    /// * `phase` - Time offset in seconds.
    pub fn pulse(amplitude: f32, frequency: f32, phase: f32) -> Self {
        Self {
            r#type: DeformationType::Pulse,
            amplitude,
            frequency,
            phase,
        }
    }

    /// Morph displacement toward a target direction.
    ///
    /// # Arguments
    /// * `amplitude` - Maximum displacement magnitude.
    /// * `frequency` - Oscillation speed.
    /// * `phase` - Time offset.
    pub fn morph(amplitude: f32, frequency: f32, phase: f32) -> Self {
        Self {
            r#type: DeformationType::Morph,
            amplitude,
            frequency,
            phase,
        }
    }

    /// Encode to a Vec4 for GPU upload.
    pub fn to_vec4(self) -> Vec4 {
        Vec4::new(
            self.r#type.as_f32(),
            self.amplitude,
            self.frequency,
            self.phase,
        )
    }

    /// Decode from a Vec4 read from GPU data.
    pub fn from_vec4(v: Vec4) -> Self {
        Self {
            r#type: DeformationType::from_f32(v.x),
            amplitude: v.y,
            frequency: v.z,
            phase: v.w,
        }
    }

    /// Returns true if this deformation is active (not None).
    pub fn is_active(self) -> bool {
        self.r#type != DeformationType::None
    }
}

/// Global deformation configuration resource.
///
/// Provides elapsed time to the vertex shader for animation.
#[derive(Debug, Clone, Copy, Resource)]
pub struct Deformation {
    /// Global elapsed time in seconds for deformation animation.
    pub time: f32,
    /// Whether deformation is globally enabled.
    pub enabled: bool,
}

impl Default for Deformation {
    fn default() -> Self {
        Self {
            time: 0.0,
            enabled: true,
        }
    }
}

impl Deformation {
    /// Advance time by the given delta.
    pub fn tick(&mut self, delta: f32) {
        if self.enabled {
            self.time += delta;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deformation_type_encoding() {
        assert_eq!(DeformationType::None.as_f32(), 0.0);
        assert_eq!(DeformationType::Wobble.as_f32(), 1.0);
        assert_eq!(DeformationType::Pulse.as_f32(), 2.0);
        assert_eq!(DeformationType::Morph.as_f32(), 3.0);
    }

    #[test]
    fn deformation_type_decoding() {
        assert_eq!(DeformationType::from_f32(0.0), DeformationType::None);
        assert_eq!(DeformationType::from_f32(1.0), DeformationType::Wobble);
        assert_eq!(DeformationType::from_f32(2.0), DeformationType::Pulse);
        assert_eq!(DeformationType::from_f32(3.0), DeformationType::Morph);
        assert_eq!(DeformationType::from_f32(99.0), DeformationType::None);
    }

    #[test]
    fn deformation_params_round_trip() {
        let params = DeformationParams::wobble(0.5, 2.0, 1.5);
        let vec4 = params.to_vec4();
        let decoded = DeformationParams::from_vec4(vec4);
        assert_eq!(decoded.r#type, DeformationType::Wobble);
        assert!((decoded.amplitude - 0.5).abs() < f32::EPSILON);
        assert!((decoded.frequency - 2.0).abs() < f32::EPSILON);
        assert!((decoded.phase - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn deformation_params_pulse() {
        let params = DeformationParams::pulse(0.1, 1.0, 0.0);
        assert!(params.is_active());
        let vec4 = params.to_vec4();
        assert!((vec4.x - 2.0).abs() < f32::EPSILON);
        assert!((vec4.y - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn deformation_params_none() {
        let params = DeformationParams::none();
        assert!(!params.is_active());
        let vec4 = params.to_vec4();
        assert!((vec4.x - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn deformation_tick() {
        let mut deform = Deformation::default();
        assert!((deform.time - 0.0).abs() < f32::EPSILON);
        deform.tick(0.016);
        assert!((deform.time - 0.016).abs() < f32::EPSILON);
        deform.enabled = false;
        deform.tick(1.0);
        assert!((deform.time - 0.016).abs() < f32::EPSILON);
    }

    #[test]
    fn deformation_morph() {
        let params = DeformationParams::morph(1.0, 0.5, 3.14);
        assert!(params.is_active());
        assert_eq!(params.r#type, DeformationType::Morph);
    }
}
