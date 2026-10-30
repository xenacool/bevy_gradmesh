//! Material interning to reduce descriptor pool pressure on mobile GPUs.
//!
//! On mobile GPUs (e.g., Mali-G715), each unique `GradMeshMaterial` asset creates
//! a new descriptor set allocation.  When many shapes share identical material
//! parameters, this wastes descriptor pool space and can trigger
//! `VK_ERROR_FRAGMENTED_POOL` crashes.
//!
//! This module provides two mechanisms to deduplicate materials:
//!
//! 1. **`GradMeshMaterialKey`** — a hashable struct representing the uniform
//!    parameters of a `GradMeshMaterial`.  Use `get_or_insert()` to find an
//!    existing material or create a new one.
//!
//! 2. **`GradMeshMaterialPool`** — a Bevy `Resource` that manages a cache of
//!    deduplicated materials automatically.  Add it via `GradMeshPlugin` or
//!    manually with `app.init_resource::<GradMeshMaterialPool>()`.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use bevy::asset::{Assets, Handle};
use bevy::ecs::prelude::{Local, Resource};

use crate::material::{GradMeshExtension, GradMeshMaterial, GradMeshUniformBuffer, Vec4};

/// Hashable key representing the uniform parameters of a `GradMeshMaterial`.
///
/// Two materials with the same key produce identical shader output and can
/// safely share the same asset handle, reducing descriptor set allocations.
#[derive(Debug, Clone)]
pub struct GradMeshMaterialKey {
    /// Primary color (dark/shadow side).
    pub color: Vec4,
    /// Secondary color (light/highlight side).
    pub secondary_color: Vec4,
    /// Gradient direction in world space.
    pub direction: Vec4,
    /// Shader flags: `[color_mode, blend_mode, emission_strength, oklab]`.
    pub flags: Vec4,
    /// Dash pattern parameters.
    pub dash: Vec4,
}

impl GradMeshMaterialKey {
    /// Create a key from a `GradMeshUniformBuffer`.
    pub fn from_uniforms(uniforms: &GradMeshUniformBuffer) -> Self {
        Self {
            color: uniforms.color,
            secondary_color: uniforms.secondary_color,
            direction: uniforms.direction,
            flags: uniforms.flags,
            dash: uniforms.dash,
        }
    }

    /// Create a key from a `GradMeshMaterial`.
    pub fn from_material(material: &GradMeshMaterial) -> Self {
        Self::from_uniforms(&material.extension.uniforms)
    }

    /// Build a `GradMeshMaterial` from this key.
    pub fn build(&self) -> GradMeshMaterial {
        GradMeshMaterial {
            base: crate::material::opaque(1.0, 1.0, 1.0).base,
            extension: GradMeshExtension {
                uniforms: GradMeshUniformBuffer::new(
                    self.color,
                    self.secondary_color,
                    self.direction,
                    self.flags,
                    self.dash,
                ),
            },
        }
    }

    /// Find an existing material with these parameters in the asset store,
    /// or create and insert a new one.
    ///
    /// This is the primary method for material deduplication.  Call it instead
    /// of `assets.add()` directly to avoid creating duplicate materials.
    pub fn get_or_insert(&self, assets: &mut Assets<GradMeshMaterial>) -> Handle<GradMeshMaterial> {
        // Collect matching asset IDs first to avoid borrow conflicts
        let mut matched_id: Option<bevy::asset::AssetId<GradMeshMaterial>> = None;
        for (id, material) in assets.iter() {
            let existing_key = Self::from_material(material);
            if existing_key == *self {
                matched_id = Some(id);
                break;
            }
        }
        // Return existing handle or create a new material
        if let Some(id) = matched_id {
            return assets.get_strong_handle(id).expect("asset should exist after iteration");
        }
        let material = self.build();
        assets.add(material)
    }
}

impl PartialEq for GradMeshMaterialKey {
    fn eq(&self, other: &Self) -> bool {
        // Use approximate float comparison for uniform values
        Self::approx_eq(&self.color, &other.color)
            && Self::approx_eq(&self.secondary_color, &other.secondary_color)
            && Self::approx_eq(&self.direction, &other.direction)
            && Self::approx_eq(&self.flags, &other.flags)
            && Self::approx_eq(&self.dash, &other.dash)
    }
}

impl Eq for GradMeshMaterialKey {}

impl Hash for GradMeshMaterialKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the bit patterns for consistent hashing
        self.color.x.to_bits().hash(state);
        self.color.y.to_bits().hash(state);
        self.color.z.to_bits().hash(state);
        self.color.w.to_bits().hash(state);
        self.secondary_color.x.to_bits().hash(state);
        self.secondary_color.y.to_bits().hash(state);
        self.secondary_color.z.to_bits().hash(state);
        self.secondary_color.w.to_bits().hash(state);
        self.direction.x.to_bits().hash(state);
        self.direction.y.to_bits().hash(state);
        self.direction.z.to_bits().hash(state);
        self.direction.w.to_bits().hash(state);
        self.flags.x.to_bits().hash(state);
        self.flags.y.to_bits().hash(state);
        self.flags.z.to_bits().hash(state);
        self.flags.w.to_bits().hash(state);
        self.dash.x.to_bits().hash(state);
        self.dash.y.to_bits().hash(state);
        self.dash.z.to_bits().hash(state);
        self.dash.w.to_bits().hash(state);
    }
}

impl GradMeshMaterialKey {
    /// Approximate float comparison with a small epsilon for uniform values.
    fn approx_eq(a: &Vec4, b: &Vec4) -> bool {
        const EPSILON: f32 = 1e-4;
        (a.x - b.x).abs() < EPSILON
            && (a.y - b.y).abs() < EPSILON
            && (a.z - b.z).abs() < EPSILON
            && (a.w - b.w).abs() < EPSILON
    }
}

/// A Bevy `Resource` that caches deduplicated `GradMeshMaterial` instances.
///
/// Unlike `GradMeshMaterialKey::get_or_insert()`, which scans all assets on
/// each call, this pool maintains a `HashMap` for O(1) lookups.
///
/// # Example
///
/// ```ignore
/// use bevy::ecs::system::ResMut;
/// use bevy_gradmesh::material_interning::GradMeshMaterialPool;
///
/// fn spawn_shape(mut pool: ResMut<GradMeshMaterialPool>) {
///     let key = GradMeshMaterialKey::from_uniforms(&uniforms);
///     let handle = pool.get_or_insert(&key, &mut materials);
/// }
/// ```
#[derive(Resource, Default, Debug)]
pub struct GradMeshMaterialPool {
    /// Cache of material keys to their asset handles.
    cache: HashMap<GradMeshMaterialKey, Handle<GradMeshMaterial>>,
    /// Total lookups performed.
    total_lookups: u64,
    /// Cache hits (existing material found).
    hits: u64,
    /// Cache misses (new material created).
    misses: u64,
}

impl GradMeshMaterialPool {
    /// Find an existing material with these parameters, or create a new one.
    ///
    /// Returns the cached `Handle<GradMeshMaterial>`.
    pub fn get_or_insert(
        &mut self,
        key: &GradMeshMaterialKey,
        assets: &mut Assets<GradMeshMaterial>,
    ) -> Handle<GradMeshMaterial> {
        self.total_lookups += 1;

        if let Some(handle) = self.cache.get(key) {
            self.hits += 1;
            return handle.clone();
        }

        self.misses += 1;
        let handle = key.get_or_insert(assets);
        self.cache.insert(key.clone(), handle.clone());
        handle
    }

    /// Returns the number of cached materials.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Returns true if the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Clears the cache.  Note that the materials remain in the asset store;
    /// only the pool's internal index is cleared.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.hits = 0;
        self.misses = 0;
        self.total_lookups = 0;
    }

    /// Returns the total number of lookups.
    pub fn total_lookups(&self) -> u64 {
        self.total_lookups
    }

    /// Returns the number of cache hits.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Returns the number of cache misses.
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Returns the cache hit rate as a float (0.0–1.0).
    pub fn hit_rate(&self) -> f64 {
        if self.total_lookups == 0 {
            0.0
        } else {
            self.hits as f64 / self.total_lookups as f64
        }
    }
}

// ─── Material count monitoring ───

/// Warning threshold for unique material count.
///
/// On mobile GPUs like Mali-G715, descriptor pool fragmentation occurs after
/// ~20-30 unique shader specializations.  This threshold provides a safety
/// margin to warn developers before the crash point.
pub const MATERIAL_WARNING_THRESHOLD: usize = 50;

/// System that monitors the number of unique `GradMeshMaterial` instances
/// and logs a warning when the count exceeds the threshold.
///
/// Add this system to your app's `Update` schedule to enable monitoring:
///
/// ```ignore
/// app.add_systems(Update, warn_excessive_materials);
/// ```
pub fn warn_excessive_materials(
    materials: bevy::ecs::system::Res<Assets<GradMeshMaterial>>,
    mut warned: Local<bool>,
) {
    let count = materials.len();
    if count > MATERIAL_WARNING_THRESHOLD && !*warned {
        bevy::log::warn!(
            "bevy_gradmesh: {} unique GradMeshMaterial instances detected (threshold: {}). \
             On mobile GPUs, this may cause descriptor pool fragmentation crashes. \
             Consider using GradMeshMaterialPool for deduplication.",
            count,
            MATERIAL_WARNING_THRESHOLD
        );
        *warned = true;
    }
    // Reset warning flag when count drops below threshold so it can re-trigger
    if count <= MATERIAL_WARNING_THRESHOLD / 2 {
        *warned = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_key_equality() {
        let key1 = GradMeshMaterialKey {
            color: Vec4::new(1.0, 0.0, 0.0, 1.0),
            secondary_color: Vec4::new(0.0, 1.0, 0.0, 1.0),
            direction: Vec4::new(0.0, 1.0, 0.0, 0.0),
            flags: Vec4::new(1.0, 0.0, 0.0, 0.0),
            dash: Vec4::new(0.0, 0.0, 0.0, 0.0),
        };
        let key2 = GradMeshMaterialKey {
            color: Vec4::new(1.0, 0.0, 0.0, 1.0),
            secondary_color: Vec4::new(0.0, 1.0, 0.0, 1.0),
            direction: Vec4::new(0.0, 1.0, 0.0, 0.0),
            flags: Vec4::new(1.0, 0.0, 0.0, 0.0),
            dash: Vec4::new(0.0, 0.0, 0.0, 0.0),
        };
        assert_eq!(key1, key2);
    }

    #[test]
    fn material_key_inequality_different_color() {
        let key1 = GradMeshMaterialKey {
            color: Vec4::new(1.0, 0.0, 0.0, 1.0),
            secondary_color: Vec4::new(0.0, 1.0, 0.0, 1.0),
            direction: Vec4::new(0.0, 1.0, 0.0, 0.0),
            flags: Vec4::new(1.0, 0.0, 0.0, 0.0),
            dash: Vec4::new(0.0, 0.0, 0.0, 0.0),
        };
        let key2 = GradMeshMaterialKey {
            color: Vec4::new(0.0, 0.0, 1.0, 1.0),
            secondary_color: Vec4::new(0.0, 1.0, 0.0, 1.0),
            direction: Vec4::new(0.0, 1.0, 0.0, 0.0),
            flags: Vec4::new(1.0, 0.0, 0.0, 0.0),
            dash: Vec4::new(0.0, 0.0, 0.0, 0.0),
        };
        assert_ne!(key1, key2);
    }

    #[test]
    fn material_key_approximate_equality() {
        let key1 = GradMeshMaterialKey {
            color: Vec4::new(1.0, 0.0, 0.0, 1.0),
            secondary_color: Vec4::new(0.0, 1.0, 0.0, 1.0),
            direction: Vec4::new(0.0, 1.0, 0.0, 0.0),
            flags: Vec4::new(1.0, 0.0, 0.0, 0.0),
            dash: Vec4::new(0.0, 0.0, 0.0, 0.0),
        };
        let key2 = GradMeshMaterialKey {
            color: Vec4::new(1.0 + 1e-5, 0.0 + 1e-5, 0.0 + 1e-5, 1.0),
            secondary_color: Vec4::new(0.0, 1.0, 0.0, 1.0),
            direction: Vec4::new(0.0, 1.0, 0.0, 0.0),
            flags: Vec4::new(1.0, 0.0, 0.0, 0.0),
            dash: Vec4::new(0.0, 0.0, 0.0, 0.0),
        };
        // Small floating-point differences should still be equal
        assert_eq!(key1, key2);
    }

    #[test]
    fn material_pool_deduplicates() {
        let mut pool = GradMeshMaterialPool::default();
        let mut assets = Assets::<GradMeshMaterial>::default();

        let key = GradMeshMaterialKey {
            color: Vec4::new(1.0, 0.0, 0.0, 1.0),
            secondary_color: Vec4::new(0.0, 1.0, 0.0, 1.0),
            direction: Vec4::new(0.0, 1.0, 0.0, 0.0),
            flags: Vec4::new(1.0, 0.0, 0.0, 0.0),
            dash: Vec4::new(0.0, 0.0, 0.0, 0.0),
        };

        let handle1 = pool.get_or_insert(&key, &mut assets);
        let handle2 = pool.get_or_insert(&key, &mut assets);

        assert_eq!(handle1.id(), handle2.id());
        assert_eq!(pool.hits(), 1);
        assert_eq!(pool.misses(), 1);
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn material_pool_different_keys_create_different_materials() {
        let mut pool = GradMeshMaterialPool::default();
        let mut assets = Assets::<GradMeshMaterial>::default();

        let key1 = GradMeshMaterialKey {
            color: Vec4::new(1.0, 0.0, 0.0, 1.0),
            secondary_color: Vec4::new(0.0, 1.0, 0.0, 1.0),
            direction: Vec4::new(0.0, 1.0, 0.0, 0.0),
            flags: Vec4::new(1.0, 0.0, 0.0, 0.0),
            dash: Vec4::new(0.0, 0.0, 0.0, 0.0),
        };
        let key2 = GradMeshMaterialKey {
            color: Vec4::new(0.0, 0.0, 1.0, 1.0),
            secondary_color: Vec4::new(0.0, 1.0, 0.0, 1.0),
            direction: Vec4::new(0.0, 1.0, 0.0, 0.0),
            flags: Vec4::new(1.0, 0.0, 0.0, 0.0),
            dash: Vec4::new(0.0, 0.0, 0.0, 0.0),
        };

        let handle1 = pool.get_or_insert(&key1, &mut assets);
        let handle2 = pool.get_or_insert(&key2, &mut assets);

        assert_ne!(handle1.id(), handle2.id());
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn material_pool_hit_rate() {
        let mut pool = GradMeshMaterialPool::default();
        let mut assets = Assets::<GradMeshMaterial>::default();

        let key = GradMeshMaterialKey {
            color: Vec4::new(1.0, 0.0, 0.0, 1.0),
            secondary_color: Vec4::new(0.0, 1.0, 0.0, 1.0),
            direction: Vec4::new(0.0, 1.0, 0.0, 0.0),
            flags: Vec4::new(1.0, 0.0, 0.0, 0.0),
            dash: Vec4::new(0.0, 0.0, 0.0, 0.0),
        };

        assert_eq!(pool.hit_rate(), 0.0);

        pool.get_or_insert(&key, &mut assets);
        assert_eq!(pool.hit_rate(), 0.0);

        pool.get_or_insert(&key, &mut assets);
        assert!((pool.hit_rate() - 0.5).abs() < f64::EPSILON);

        pool.get_or_insert(&key, &mut assets);
        let expected = 2.0 / 3.0;
        assert!((pool.hit_rate() - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn material_key_from_uniforms() {
        let uniforms = GradMeshUniformBuffer::new(
            Vec4::new(1.0, 0.5, 0.0, 1.0),
            Vec4::new(0.0, 0.5, 1.0, 1.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 0.0),
        );
        let key = GradMeshMaterialKey::from_uniforms(&uniforms);
        assert_eq!(key.color, Vec4::new(1.0, 0.5, 0.0, 1.0));
        assert_eq!(key.secondary_color, Vec4::new(0.0, 0.5, 1.0, 1.0));
        assert_eq!(key.direction, Vec4::new(0.0, 1.0, 0.0, 0.0));
    }

    #[test]
    fn material_pool_clear() {
        let mut pool = GradMeshMaterialPool::default();
        let mut assets = Assets::<GradMeshMaterial>::default();

        let key = GradMeshMaterialKey {
            color: Vec4::new(1.0, 0.0, 0.0, 1.0),
            secondary_color: Vec4::new(0.0, 1.0, 0.0, 1.0),
            direction: Vec4::new(0.0, 1.0, 0.0, 0.0),
            flags: Vec4::new(1.0, 0.0, 0.0, 0.0),
            dash: Vec4::new(0.0, 0.0, 0.0, 0.0),
        };

        pool.get_or_insert(&key, &mut assets);
        assert_eq!(pool.len(), 1);

        pool.clear();
        assert_eq!(pool.len(), 0);
        assert_eq!(pool.hits(), 0);
        assert_eq!(pool.misses(), 0);
    }
}
