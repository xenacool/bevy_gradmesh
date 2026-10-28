//! Appearance Snapshots — per-instance color/gradient animation.
//!
//! Enables runtime color animation without per-frame CPU updates by encoding
//! animation parameters as a `vec4` per instance. The vertex shader resolves
//! the color at render time using the global deformation time resource.
//!
//! # Design
//!
//! Each instance carries an `AppearanceSnapshot` vec4 encoding:
//! - `x`: animation speed (cycles per second; 0.0 = static color)
//! - `y`: phase offset (0.0–1.0, time shift for staggered animation)
//! - `z`: animation type (0=Static, 1=ColorPulse, 2=ColorCycle, 3=GradientShift)
//! - `w`: reserved for future use
//!
//! When speed > 0, the shader interpolates between the instance's primary and
//! secondary colors based on `sin(time * speed + phase * 2π)`.

use crate::material::Vec4;

/// Animation type for appearance snapshots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AppearanceType {
    #[default]
    /// No animation — static color.
    Static,
    /// Pulse between primary and secondary color.
    ColorPulse,
    /// Cycle through hues (shifts the color continuously).
    ColorCycle,
    /// Shift the gradient direction over time.
    GradientShift,
}

impl AppearanceType {
    /// Returns the numeric code for shader encoding.
    pub fn as_f32(self) -> f32 {
        match self {
            AppearanceType::Static => 0.0,
            AppearanceType::ColorPulse => 1.0,
            AppearanceType::ColorCycle => 2.0,
            AppearanceType::GradientShift => 3.0,
        }
    }

    /// Decode from a float value.
    pub fn from_f32(value: f32) -> Self {
        match value.round() as i32 {
            1 => AppearanceType::ColorPulse,
            2 => AppearanceType::ColorCycle,
            3 => AppearanceType::GradientShift,
            _ => AppearanceType::Static,
        }
    }
}

/// Appearance animation parameters for a single instance.
///
/// Encoded as a single `vec4` for GPU upload:
/// - `x`: speed (cycles per second)
/// - `y`: phase offset (0.0–1.0)
/// - `z`: type code (0=Static, 1=ColorPulse, 2=ColorCycle, 3=GradientShift)
/// - `w`: reserved
#[derive(Debug, Clone, Copy, Default)]
pub struct AppearanceSnapshot {
    pub speed: f32,
    pub phase: f32,
    pub r#type: AppearanceType,
}

impl AppearanceSnapshot {
    /// No animation — static color.
    pub fn static_appearance() -> Self {
        Self::default()
    }

    /// Pulse between primary and secondary color.
    ///
    /// # Arguments
    /// * `speed` - Cycles per second (e.g., 1.0 = one full cycle per second).
    /// * `phase` - Time offset (0.0–1.0) for staggered animation across instances.
    pub fn color_pulse(speed: f32, phase: f32) -> Self {
        Self {
            speed,
            phase: phase.clamp(0.0, 1.0),
            r#type: AppearanceType::ColorPulse,
        }
    }

    /// Cycle through color hues over time.
    ///
    /// # Arguments
    /// * `speed` - Cycles per second.
    /// * `phase` - Starting hue offset (0.0–1.0).
    pub fn color_cycle(speed: f32, phase: f32) -> Self {
        Self {
            speed,
            phase: phase.clamp(0.0, 1.0),
            r#type: AppearanceType::ColorCycle,
        }
    }

    /// Shift the gradient direction over time.
    ///
    /// # Arguments
    /// * `speed` - Rotation speed in cycles per second.
    /// * `phase` - Starting angle offset (0.0–1.0 maps to 0–2π).
    pub fn gradient_shift(speed: f32, phase: f32) -> Self {
        Self {
            speed,
            phase: phase.clamp(0.0, 1.0),
            r#type: AppearanceType::GradientShift,
        }
    }

    /// Encode to a Vec4 for GPU upload.
    pub fn to_vec4(self) -> Vec4 {
        Vec4::new(self.speed, self.phase, self.r#type.as_f32(), 0.0)
    }

    /// Decode from a Vec4 read from GPU data.
    pub fn from_vec4(v: Vec4) -> Self {
        Self {
            speed: v.x,
            phase: v.y,
            r#type: AppearanceType::from_f32(v.z),
        }
    }

    /// Returns true if this appearance is animated (not static).
    pub fn is_animated(self) -> bool {
        self.r#type != AppearanceType::Static && self.speed > 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn appearance_type_encoding() {
        assert_eq!(AppearanceType::Static.as_f32(), 0.0);
        assert_eq!(AppearanceType::ColorPulse.as_f32(), 1.0);
        assert_eq!(AppearanceType::ColorCycle.as_f32(), 2.0);
        assert_eq!(AppearanceType::GradientShift.as_f32(), 3.0);
    }

    #[test]
    fn appearance_type_decoding() {
        assert_eq!(AppearanceType::from_f32(0.0), AppearanceType::Static);
        assert_eq!(AppearanceType::from_f32(1.0), AppearanceType::ColorPulse);
        assert_eq!(AppearanceType::from_f32(2.0), AppearanceType::ColorCycle);
        assert_eq!(AppearanceType::from_f32(3.0), AppearanceType::GradientShift);
        assert_eq!(AppearanceType::from_f32(99.0), AppearanceType::Static);
    }

    #[test]
    fn appearance_snapshot_round_trip() {
        let snap = AppearanceSnapshot::color_pulse(2.0, 0.5);
        let vec4 = snap.to_vec4();
        let decoded = AppearanceSnapshot::from_vec4(vec4);
        assert_eq!(decoded.r#type, AppearanceType::ColorPulse);
        assert!((decoded.speed - 2.0).abs() < f32::EPSILON);
        assert!((decoded.phase - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn appearance_snapshot_static() {
        let snap = AppearanceSnapshot::static_appearance();
        assert!(!snap.is_animated());
        let vec4 = snap.to_vec4();
        assert!((vec4.x - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn appearance_snapshot_animated() {
        let snap = AppearanceSnapshot::color_cycle(1.5, 0.25);
        assert!(snap.is_animated());
    }

    #[test]
    fn appearance_snapshot_zero_speed_not_animated() {
        let snap = AppearanceSnapshot {
            speed: 0.0,
            phase: 0.0,
            r#type: AppearanceType::ColorPulse,
        };
        assert!(!snap.is_animated());
    }

    #[test]
    fn appearance_phase_clamped() {
        let snap = AppearanceSnapshot::color_pulse(1.0, 5.0);
        assert!(snap.phase <= 1.0);
        let snap2 = AppearanceSnapshot::color_pulse(1.0, -2.0);
        assert!(snap2.phase >= 0.0);
    }
}
