//! Distance-based Level of Detail (LOD) with dithering cross-fade.
//!
//! When an instance crosses an LOD threshold, both the current and next LOD
//! level are rendered simultaneously with a dither mask that fades between
//! them, eliminating visual pop-in.
//!
//! # Design
//!
//! 1. **LODConfig**: Configurable distance thresholds and transition width.
//! 2. **compute_lod_level()**: Given a distance, returns the current LOD level
//!    and a transition weight (0.0 = fully current, 1.0 = fully next).
//! 3. **dither_crossfade()**: WGSL function that generates a Bayer dither
//!    pattern to blend between two LOD levels during transitions.
//!
//! # Usage
//!
//! The render node computes per-batch LOD levels from camera distance.
//! When a batch is in transition, it spawns a second render item for the
//! next LOD level, and the fragment shader uses the dither pattern to
//! blend between them.

use bevy::ecs::prelude::Resource;
use bevy::math::Vec3;

/// LOD level indices. Higher index = higher detail (more vertices).
/// LOD 0 = Minimal (fewest vertices, used for far objects)
/// LOD 4 = Extreme (most vertices, used for close objects)
pub const LOD_MINIMAL: u8 = 0;
pub const LOD_LOW: u8 = 1;
pub const LOD_MEDIUM: u8 = 2;
pub const LOD_HIGH: u8 = 3;
pub const LOD_EXTREME: u8 = 4;

/// Invert an LOD level: close objects get high detail, far objects get low detail.
/// The raw compute_lod returns 0 for close and 4 for far.
/// This function inverts so 4 (Extreme) is close and 0 (Minimal) is far.
#[inline]
pub fn invert_lod_level(level: u8) -> u8 {
    LOD_EXTREME - level
}

/// Number of LOD levels.
pub const LOD_LEVEL_COUNT: usize = 5;

/// Configuration for distance-based LOD.
#[derive(Debug, Clone, Copy, Resource)]
pub struct LodConfig {
    /// Distance thresholds for each LOD transition.
    /// `thresholds[i]` is the distance at which LOD `i` transitions to `i+1`.
    /// For 5 levels, there are 4 thresholds.
    pub thresholds: [f32; 4],
    /// Width of the transition zone in world units.
    /// During this zone, both LOD levels are rendered with dither cross-fade.
    pub transition_width: f32,
    /// Hysteresis factor (0.0–1.0). Prevents rapid LOD oscillation by
    /// requiring the distance to move further before switching back.
    pub hysteresis: f32,
}

impl Default for LodConfig {
    fn default() -> Self {
        Self {
            // Default thresholds: switch at 10, 30, 60, 100 units
            thresholds: [10.0, 30.0, 60.0, 100.0],
            // Transition over 5 units
            transition_width: 5.0,
            // 20% hysteresis
            hysteresis: 0.2,
        }
    }
}

impl LodConfig {
    /// Compute the LOD level and transition weight for a given distance.
    ///
    /// Returns `(lod_level, transition_weight)` where:
    /// - `lod_level`: The current LOD level (0=Minimal for far, 4=Extreme for close).
    /// - `transition_weight`: 0.0 means fully at current LOD, 1.0 means
    ///   fully at the next LOD. Values between 0.0 and 1.0 indicate
    ///   a transition zone where dither cross-fade should be applied.
    ///
    /// If `previous_lod` is provided, hysteresis is applied to prevent
    /// rapid oscillation.
    ///
    /// Note: Close objects get high detail (Extreme=4), far objects get low detail (Minimal=0).
    pub fn compute_lod(
        &self,
        distance: f32,
        previous_lod: Option<u8>,
    ) -> (u8, f32) {
        // Find the first threshold that the distance exceeds.
        // This determines the raw LOD level (0=closer, higher=farther).
        let mut raw_level = 0u8;
        let mut transition_weight = 0.0;

        for (i, &threshold) in self.thresholds.iter().enumerate() {
            let level = i as u8;
            let hysteresis_offset = if let Some(prev) = previous_lod {
                // If we were at a higher raw LOD (farther), require moving farther to switch
                if prev > level {
                    threshold * self.hysteresis
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let effective_threshold = threshold + hysteresis_offset;

            if distance > effective_threshold + self.transition_width {
                // Fully past this threshold, continue to check further thresholds
                raw_level = level + 1;
            } else if distance > effective_threshold {
                // In the transition zone for this threshold — stop here
                raw_level = level;
                let progress = (distance - effective_threshold) / self.transition_width;
                transition_weight = progress.clamp(0.0, 1.0);
                break;
            } else {
                // Before this threshold — stop here, no transition
                raw_level = level;
                transition_weight = 0.0;
                break;
            }
        }

        raw_level = raw_level.min(LOD_EXTREME);

        // Invert: raw 0 (close) -> Extreme(4), raw 4 (far) -> Minimal(0)
        let lod_level = invert_lod_level(raw_level);

        (lod_level, transition_weight)
    }

    /// Compute the camera-to-instance distance for LOD determination.
    pub fn distance_from_camera(camera_pos: &Vec3, instance_pos: &Vec3) -> f32 {
        camera_pos.distance(*instance_pos)
    }
}

/// Result of LOD computation for a single instance.
#[derive(Debug, Clone, Copy)]
pub struct LodResult {
    /// Current LOD level (0=Minimal, 4=Extreme).
    pub lod_level: u8,
    /// Transition weight (0.0 = fully current, 1.0 = fully next).
    /// If > 0.0 and < 1.0, dither cross-fade is active.
    pub transition_weight: f32,
    /// Whether this instance is in a transition zone.
    pub in_transition: bool,
}

impl LodResult {
    /// Returns true if dither cross-fade should be applied.
    pub fn needs_dither(&self) -> bool {
        self.in_transition && self.transition_weight > 0.0 && self.transition_weight < 1.0
    }
}

impl LodConfig {
    /// Compute LOD with full result including transition state.
    pub fn compute_lod_result(
        &self,
        distance: f32,
        previous_lod: Option<u8>,
    ) -> LodResult {
        let (lod_level, transition_weight) = self.compute_lod(distance, previous_lod);
        let in_transition = transition_weight > 0.0 && transition_weight < 1.0;
        LodResult {
            lod_level,
            transition_weight,
            in_transition,
        }
    }
}

/// Bayer 4x4 dither matrix for cross-fade blending.
/// Used in the WGSL shader to generate the dither pattern.
pub const BAYER_DITHER_4X4: [[f32; 4]; 4] = [
    [ 0.0/16.0,  8.0/16.0,  2.0/16.0, 10.0/16.0],
    [12.0/16.0,  4.0/16.0, 14.0/16.0,  6.0/16.0],
    [ 3.0/16.0, 11.0/16.0,  1.0/16.0,  9.0/16.0],
    [15.0/16.0,  7.0/16.0, 13.0/16.0,  5.0/16.0],
];

/// Generate the WGSL code for the Bayer dither function.
/// This is included in the instanced shader for LOD cross-fade.
pub fn dither_wgsl_snippet() -> &'static str {
    r#"
// Bayer 4x4 dither matrix for LOD cross-fade
const bayer_dither_4x4 = array<array<f32, 4>, 4>(
    array<f32, 4>(0.0/16.0,  8.0/16.0,  2.0/16.0, 10.0/16.0),
    array<f32, 4>(12.0/16.0, 4.0/16.0, 14.0/16.0,  6.0/16.0),
    array<f32, 4>( 3.0/16.0, 11.0/16.0, 1.0/16.0,  9.0/16.0),
    array<f32, 4>(15.0/16.0,  7.0/16.0, 13.0/16.0, 5.0/16.0),
);

/// Apply Bayer dither cross-fade between two LOD levels.
/// Returns 1.0 if the fragment should be visible, 0.0 if it should be discarded.
/// When transition_weight is 0.0, always returns 1.0 (no dithering).
/// When transition_weight is 1.0, always returns 1.0 (fully transitioned).
fn lod_dither_mask(frag_coord: vec2<f32>, transition_weight: f32) -> f32 {
    if (transition_weight <= 0.0 || transition_weight >= 1.0) {
        return 1.0;
    }
    
    let x = i32(frag_coord.x) % 4;
    let y = i32(frag_coord.y) % 4;
    let dither_value = bayer_dither_4x4[y][x];
    
    // If the dither value is less than the transition weight, show this fragment
    // This creates a smooth visual transition between LOD levels
    return select(1.0, 0.0, dither_value > transition_weight);
}
"#
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lod_default_config() {
        let config = LodConfig::default();
        assert_eq!(config.thresholds, [10.0, 30.0, 60.0, 100.0]);
        assert!((config.transition_width - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn lod_close_distance_is_extreme() {
        let config = LodConfig::default();
        let (lod, weight) = config.compute_lod(1.0, None);
        // Close objects get Extreme detail (level 4)
        assert_eq!(lod, LOD_EXTREME);
        assert!((weight - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn lod_far_distance_is_minimal() {
        let config = LodConfig::default();
        let (lod, _weight) = config.compute_lod(200.0, None);
        // Far objects get Minimal detail (level 0)
        assert_eq!(lod, LOD_MINIMAL);
    }

    #[test]
    fn lod_transition_zone() {
        let config = LodConfig {
            thresholds: [10.0, 30.0, 60.0, 100.0],
            transition_width: 5.0,
            hysteresis: 0.0,
        };
        // At distance 12.5, halfway through the first transition zone (10-15)
        // Raw level 0 transitions to 1, inverted: Extreme(4) transitions to High(3)
        // The current LOD is the higher detail level (Extreme=4) during transition
        let (lod, weight) = config.compute_lod(12.5, None);
        assert_eq!(lod, LOD_EXTREME); // Inverted from raw level 0
        assert!((weight - 0.5).abs() < 0.01);
    }

    #[test]
    fn lod_hysteresis_prevents_oscillation() {
        let config = LodConfig {
            thresholds: [10.0, 30.0, 60.0, 100.0],
            transition_width: 5.0,
            hysteresis: 0.2,
        };
        // Previously at raw LOD 3 (inverted = Low), now at 11.0
        // With hysteresis, the effective threshold shifts
        let (lod, _weight) = config.compute_lod(11.0, Some(3));
        // Should stay at a higher detail level due to hysteresis
        assert!(lod >= LOD_LOW);
    }

    #[test]
    fn lod_result_needs_dither() {
        let config = LodConfig::default();
        
        // In transition
        let result = config.compute_lod_result(12.5, None);
        assert!(result.needs_dither());
        
        // Not in transition (very close)
        let result = config.compute_lod_result(1.0, None);
        assert!(!result.needs_dither());
        
        // Fully transitioned (past the zone)
        let result = config.compute_lod_result(20.0, None);
        assert!(!result.needs_dither());
    }

    #[test]
    fn bayer_dither_matrix_values() {
        // Verify the Bayer matrix values are in [0, 1)
        for row in &BAYER_DITHER_4X4 {
            for &val in row {
                assert!(val >= 0.0 && val < 1.0);
            }
        }
    }

    #[test]
    fn lod_distance_from_camera() {
        let camera = Vec3::new(0.0, 0.0, 0.0);
        let instance = Vec3::new(3.0, 4.0, 0.0);
        let dist = LodConfig::distance_from_camera(&camera, &instance);
        assert!((dist - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn invert_lod_level_symmetry() {
        assert_eq!(invert_lod_level(LOD_MINIMAL), LOD_EXTREME);
        assert_eq!(invert_lod_level(LOD_EXTREME), LOD_MINIMAL);
        assert_eq!(invert_lod_level(LOD_MEDIUM), LOD_MEDIUM);
    }
}
