//! Material interning to reduce descriptor pool pressure on mobile GPUs.
//!
//! On mobile GPUs (e.g., Mali-G715), each unique `GradMeshMaterial` asset creates
//! a new descriptor set allocation.  When many shapes share identical material
//! parameters, this wastes descriptor pool space and can trigger
//! `VK_ERROR_FRAGMENTED_POOL` crashes.
//!
//! This module provides two mechanisms to deduplicate materials:
//!
//! 1. **`GradMeshMaterialKey`** - a hashable struct representing the uniform
//!    parameters of a `GradMeshMaterial`.  Use `get_or_insert()` to find an
//!    existing material or create a new one.
//!
//! 2. **`GradMeshMaterialPool`** - a Bevy `Resource` that manages a cache of
//!    deduplicated materials automatically.  Add it via `GradMeshPlugin` or
//!    manually with `app.init_resource::<GradMeshMaterialPool>()`.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use bevy::asset::{AssetId, Assets, Handle};
use bevy::ecs::prelude::{Local, Resource};

use crate::material::{GradMeshExtension, GradMeshMaterial, GradMeshUniformBuffer, Vec4};
use crate::style::Style;

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
    /// Procedural noise parameters.
    pub noise: Vec4,
    /// Procedural stripe direction and frequency.
    pub stripes: Vec4,
    /// Procedural stripe amount.
    pub stripes_amount: Vec4,
    /// Rim light colour and power.
    pub rim: Vec4,
    /// Height-band colours.
    pub bands: [Vec4; 4],
    /// Height-band start heights.
    pub band_heights: Vec4,
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
            noise: uniforms.noise,
            stripes: uniforms.stripes,
            stripes_amount: uniforms.stripes_amount,
            rim: uniforms.rim,
            bands: uniforms.bands,
            band_heights: uniforms.band_heights,
        }
    }

    /// Create a key from a `GradMeshMaterial`.
    pub fn from_material(material: &GradMeshMaterial) -> Self {
        Self::from_uniforms(&material.extension.uniforms)
    }

    /// Build a `GradMeshMaterial` from this key.
    pub fn build(&self) -> GradMeshMaterial {
        let mut uniforms = GradMeshUniformBuffer::new(
            self.color,
            self.secondary_color,
            self.direction,
            self.flags,
            self.dash,
        );
        uniforms.noise = self.noise;
        uniforms.stripes = self.stripes;
        uniforms.stripes_amount = self.stripes_amount;
        uniforms.rim = self.rim;
        uniforms.bands = self.bands;
        uniforms.band_heights = self.band_heights;
        GradMeshMaterial {
            base: bevy::pbr::StandardMaterial {
                base_color: bevy::color::Color::srgb(1.0, 1.0, 1.0).into(),
                perceptual_roughness: 0.7,
                cull_mode: None,
                ..bevy::pbr::StandardMaterial::default()
            },
            extension: GradMeshExtension { uniforms },
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
            return assets
                .get_strong_handle(id)
                .expect("asset should exist after iteration");
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
            && Self::approx_eq(&self.noise, &other.noise)
            && Self::approx_eq(&self.stripes, &other.stripes)
            && Self::approx_eq(&self.stripes_amount, &other.stripes_amount)
            && Self::approx_eq(&self.rim, &other.rim)
            && self
                .bands
                .iter()
                .zip(&other.bands)
                .all(|(a, b)| Self::approx_eq(a, b))
            && Self::approx_eq(&self.band_heights, &other.band_heights)
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
        for v in [&self.noise, &self.stripes, &self.stripes_amount, &self.rim]
            .into_iter()
            .chain(self.bands.iter())
            .chain(std::iter::once(&self.band_heights))
        {
            v.x.to_bits().hash(state);
            v.y.to_bits().hash(state);
            v.z.to_bits().hash(state);
            v.w.to_bits().hash(state);
        }
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
/// The pool does **not** keep its materials alive: it remembers their ids and
/// hands out strong handles, so a material that no node refers to any more is
/// freed by Bevy and forgotten here. A screen that animates colours therefore
/// creates many materials over time but holds only the live ones, and
/// [`warn_excessive_materials`] counts only those. Hold the handle for as long
/// as the material is shown; asking again for an equal style while it is alive
/// returns the same handle.
///
/// There is nothing to manage: no entry needs to be removed by hand and the pool
/// has no `clear`, because dead entries are swept as it grows.
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
    cache: HashMap<GradMeshMaterialKey, AssetId<GradMeshMaterial>>,
    /// Materials keyed by the `Style` they were built from (bucketed by hash).
    styles: HashMap<u64, Vec<(Style, AssetId<GradMeshMaterial>)>>,
    /// Entry count at which dead entries are swept out next.
    prune_at: usize,
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

        if let Some(id) = self.cache.get(key).copied() {
            if let Some(handle) = assets.get_strong_handle(id) {
                self.hits += 1;
                return handle;
            }
            self.cache.remove(key);
        }

        self.misses += 1;
        let handle = key.get_or_insert(assets);
        self.cache.insert(key.clone(), handle.id());
        self.prune_when_due(assets);
        handle
    }

    /// Returns the shared material for `style`, building it on the first request.
    ///
    /// Equal styles (`Style: PartialEq`) return the same handle, so syncing a
    /// node every frame with an unchanged style allocates no material asset.
    pub fn get_or_add(
        &mut self,
        style: &Style,
        assets: &mut Assets<GradMeshMaterial>,
    ) -> Handle<GradMeshMaterial> {
        self.total_lookups += 1;
        let bucket = self.styles.entry(style.cache_hash()).or_default();
        // Entries whose asset was freed are dropped instead of returned dangling.
        bucket.retain(|(_, id)| assets.contains(*id));
        let found = bucket
            .iter()
            .find(|(s, _)| s == style)
            .map(|(_, id)| *id)
            .and_then(|id| assets.get_strong_handle(id));
        if let Some(handle) = found {
            self.hits += 1;
            return handle;
        }
        self.misses += 1;
        let handle = assets.add(style.build());
        bucket.push((style.clone(), handle.id()));
        self.prune_when_due(assets);
        handle
    }

    /// Forgets every entry whose material has been freed. Lookups do this on
    /// their own as the pool grows, so calling it is only needed to shrink the
    /// pool at a moment of your choosing.
    pub fn prune(&mut self, assets: &Assets<GradMeshMaterial>) {
        self.cache.retain(|_, id| assets.contains(*id));
        for bucket in self.styles.values_mut() {
            bucket.retain(|(_, id)| assets.contains(*id));
        }
        self.styles.retain(|_, bucket| !bucket.is_empty());
    }

    /// Sweeps dead entries once the pool has doubled since the last sweep, so
    /// the work stays proportional to the number of materials created.
    fn prune_when_due(&mut self, assets: &Assets<GradMeshMaterial>) {
        if self.len() >= self.prune_at.max(64) {
            self.prune(assets);
            self.prune_at = self.len() * 2;
        }
    }

    /// Returns the number of cached materials (some of which may have been
    /// freed since; [`GradMeshMaterialPool::prune`] forgets those).
    pub fn len(&self) -> usize {
        self.cache.len() + self.styles.values().map(Vec::len).sum::<usize>()
    }

    /// Returns the number of pooled materials that still exist in `assets`,
    /// that is, that something holds a handle to.
    pub fn live_len(&self, assets: &Assets<GradMeshMaterial>) -> usize {
        self.cache.values().filter(|id| assets.contains(**id)).count()
            + self
                .styles
                .values()
                .flatten()
                .filter(|(_, id)| assets.contains(*id))
                .count()
    }

    /// Returns true if the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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

    /// Returns the cache hit rate as a float (0.0-1.0).
    pub fn hit_rate(&self) -> f64 {
        if self.total_lookups == 0 {
            0.0
        } else {
            self.hits as f64 / self.total_lookups as f64
        }
    }
}

// --- Material count monitoring ---

/// Default warning threshold for the number of live materials.
///
/// On mobile GPUs like Mali-G715, descriptor pool fragmentation occurs after
/// ~20-30 unique shader specializations.  This threshold provides a safety
/// margin to warn developers before the crash point. Change it with
/// [`MaterialWarning`].
pub const MATERIAL_WARNING_THRESHOLD: usize = 50;

/// The material count above which [`warn_excessive_materials`] warns. Insert it
/// as a resource to change the default of [`MATERIAL_WARNING_THRESHOLD`]; the
/// warning arms again once the count falls to half of it.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaterialWarning {
    /// Warn above this many live materials.
    pub threshold: usize,
}

impl Default for MaterialWarning {
    fn default() -> Self {
        Self {
            threshold: MATERIAL_WARNING_THRESHOLD,
        }
    }
}

/// System that monitors the number of live `GradMeshMaterial` assets
/// and logs a warning when the count exceeds the threshold.
///
/// Only materials that something still refers to are counted: Bevy frees an
/// asset when its last handle is dropped, and [`GradMeshMaterialPool`] does not
/// hold handles of its own, so colours that were animated through the pool and
/// are no longer shown do not add up. If the warning fires anyway, something
/// holds on to many different materials at once (for example by keeping every
/// handle it ever asked for).
///
/// Add this system to your app's `Update` schedule to enable monitoring:
///
/// ```ignore
/// app.add_systems(Update, warn_excessive_materials);
/// ```
pub fn warn_excessive_materials(
    materials: bevy::ecs::system::Res<Assets<GradMeshMaterial>>,
    config: Option<bevy::ecs::system::Res<MaterialWarning>>,
    mut warned: Local<bool>,
) {
    let threshold = config.map_or(MATERIAL_WARNING_THRESHOLD, |c| c.threshold);
    let count = materials.len();
    if count > threshold && !*warned {
        bevy::log::warn!(
            "bevy_gradmesh: {} live GradMeshMaterial instances detected (threshold: {}). \
             On mobile GPUs, this may cause descriptor pool fragmentation crashes. \
             Consider using GradMeshMaterialPool for deduplication, and drop the handles \
             of materials that are no longer shown.",
            count,
            threshold
        );
        *warned = true;
    }
    // Reset warning flag when count drops below threshold so it can re-trigger
    if count <= threshold / 2 {
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
            noise: Vec4::default(),
            stripes: Vec4::default(),
            stripes_amount: Vec4::default(),
            rim: Vec4::default(),
            bands: [Vec4::default(); 4],
            band_heights: Vec4::default(),
        };
        let key2 = GradMeshMaterialKey {
            color: Vec4::new(1.0, 0.0, 0.0, 1.0),
            secondary_color: Vec4::new(0.0, 1.0, 0.0, 1.0),
            direction: Vec4::new(0.0, 1.0, 0.0, 0.0),
            flags: Vec4::new(1.0, 0.0, 0.0, 0.0),
            dash: Vec4::new(0.0, 0.0, 0.0, 0.0),
            noise: Vec4::default(),
            stripes: Vec4::default(),
            stripes_amount: Vec4::default(),
            rim: Vec4::default(),
            bands: [Vec4::default(); 4],
            band_heights: Vec4::default(),
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
            noise: Vec4::default(),
            stripes: Vec4::default(),
            stripes_amount: Vec4::default(),
            rim: Vec4::default(),
            bands: [Vec4::default(); 4],
            band_heights: Vec4::default(),
        };
        let key2 = GradMeshMaterialKey {
            color: Vec4::new(0.0, 0.0, 1.0, 1.0),
            secondary_color: Vec4::new(0.0, 1.0, 0.0, 1.0),
            direction: Vec4::new(0.0, 1.0, 0.0, 0.0),
            flags: Vec4::new(1.0, 0.0, 0.0, 0.0),
            dash: Vec4::new(0.0, 0.0, 0.0, 0.0),
            noise: Vec4::default(),
            stripes: Vec4::default(),
            stripes_amount: Vec4::default(),
            rim: Vec4::default(),
            bands: [Vec4::default(); 4],
            band_heights: Vec4::default(),
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
            noise: Vec4::default(),
            stripes: Vec4::default(),
            stripes_amount: Vec4::default(),
            rim: Vec4::default(),
            bands: [Vec4::default(); 4],
            band_heights: Vec4::default(),
        };
        let key2 = GradMeshMaterialKey {
            color: Vec4::new(1.0 + 1e-5, 0.0 + 1e-5, 0.0 + 1e-5, 1.0),
            secondary_color: Vec4::new(0.0, 1.0, 0.0, 1.0),
            direction: Vec4::new(0.0, 1.0, 0.0, 0.0),
            flags: Vec4::new(1.0, 0.0, 0.0, 0.0),
            dash: Vec4::new(0.0, 0.0, 0.0, 0.0),
            noise: Vec4::default(),
            stripes: Vec4::default(),
            stripes_amount: Vec4::default(),
            rim: Vec4::default(),
            bands: [Vec4::default(); 4],
            band_heights: Vec4::default(),
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
            noise: Vec4::default(),
            stripes: Vec4::default(),
            stripes_amount: Vec4::default(),
            rim: Vec4::default(),
            bands: [Vec4::default(); 4],
            band_heights: Vec4::default(),
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
            noise: Vec4::default(),
            stripes: Vec4::default(),
            stripes_amount: Vec4::default(),
            rim: Vec4::default(),
            bands: [Vec4::default(); 4],
            band_heights: Vec4::default(),
        };
        let key2 = GradMeshMaterialKey {
            color: Vec4::new(0.0, 0.0, 1.0, 1.0),
            secondary_color: Vec4::new(0.0, 1.0, 0.0, 1.0),
            direction: Vec4::new(0.0, 1.0, 0.0, 0.0),
            flags: Vec4::new(1.0, 0.0, 0.0, 0.0),
            dash: Vec4::new(0.0, 0.0, 0.0, 0.0),
            noise: Vec4::default(),
            stripes: Vec4::default(),
            stripes_amount: Vec4::default(),
            rim: Vec4::default(),
            bands: [Vec4::default(); 4],
            band_heights: Vec4::default(),
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
            noise: Vec4::default(),
            stripes: Vec4::default(),
            stripes_amount: Vec4::default(),
            rim: Vec4::default(),
            bands: [Vec4::default(); 4],
            band_heights: Vec4::default(),
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

    #[cfg(feature = "pbr")]
    mod live {
        use super::*;
        use crate::style::Style;
        use bevy::app::App;
        use bevy::asset::{AssetApp, AssetPlugin};
        use bevy::color::Color;
        use bevy::app::TaskPoolPlugin;

        fn app() -> App {
            let mut app = App::new();
            app.add_plugins((TaskPoolPlugin::default(), AssetPlugin::default()))
                .init_asset::<GradMeshMaterial>();
            app
        }

        fn count(app: &App) -> usize {
            app.world().resource::<Assets<GradMeshMaterial>>().len()
        }

        #[test]
        fn the_pool_does_not_keep_unused_materials_alive() {
            let mut app = app();
            let mut pool = GradMeshMaterialPool::default();
            let style = |i: usize| Style::opaque(Color::srgb(i as f32 / 100.0, 0.5, 0.5));
            let mut shown = Vec::new();
            // A colour animation: a new colour every step, one shown at a time.
            for i in 0..200 {
                let handle = {
                    let mut assets = app.world_mut().resource_mut::<Assets<GradMeshMaterial>>();
                    pool.get_or_add(&style(i), &mut assets)
                };
                shown = vec![handle];
                app.update();
            }
            assert_eq!(count(&app), 1, "only the material still shown is alive");
            assert_eq!(shown.len(), 1);
            // Dead entries are swept as the pool grows.
            assert!(pool.len() < 200, "{} entries", pool.len());
            // Equal styles share one handle while it is alive...
            let again = {
                let mut assets = app.world_mut().resource_mut::<Assets<GradMeshMaterial>>();
                pool.get_or_add(&style(199), &mut assets)
            };
            assert_eq!(again, shown[0]);
            // ...and once nothing holds it, the material goes away and a new one is built.
            drop((again, shown));
            app.update();
            assert_eq!(count(&app), 0);
            let fresh = {
                let mut assets = app.world_mut().resource_mut::<Assets<GradMeshMaterial>>();
                pool.get_or_add(&style(199), &mut assets)
            };
            assert_eq!(count(&app), 1);
            assert!(app.world().resource::<Assets<GradMeshMaterial>>().contains(fresh.id()));
        }

        #[test]
        fn a_just_released_material_is_never_handed_back_dangling() {
            let mut app = app();
            let mut pool = GradMeshMaterialPool::default();
            let style = Style::opaque(Color::srgb(0.2, 0.4, 0.6));
            let first = {
                let mut assets = app.world_mut().resource_mut::<Assets<GradMeshMaterial>>();
                pool.get_or_add(&style, &mut assets)
            };
            // Dropped, but Bevy has not processed the drop yet.
            drop(first);
            let second = {
                let mut assets = app.world_mut().resource_mut::<Assets<GradMeshMaterial>>();
                pool.get_or_add(&style, &mut assets)
            };
            app.update();
            assert!(app.world().resource::<Assets<GradMeshMaterial>>().contains(second.id()));
        }

        #[test]
        fn the_key_path_is_pooled_without_strong_handles_too() {
            let mut app = app();
            let mut pool = GradMeshMaterialPool::default();
            let key = GradMeshMaterialKey::from_material(&Style::opaque(Color::WHITE).build());
            let h = {
                let mut assets = app.world_mut().resource_mut::<Assets<GradMeshMaterial>>();
                pool.get_or_insert(&key, &mut assets)
            };
            let h2 = {
                let mut assets = app.world_mut().resource_mut::<Assets<GradMeshMaterial>>();
                pool.get_or_insert(&key, &mut assets)
            };
            assert_eq!(h, h2);
            drop((h, h2));
            app.update();
            assert_eq!(count(&app), 0);
        }

        #[test]
        fn live_len_counts_only_materials_something_still_holds() {
            let mut app = app();
            let mut pool = GradMeshMaterialPool::default();
            let held = {
                let mut assets = app.world_mut().resource_mut::<Assets<GradMeshMaterial>>();
                let held = pool.get_or_add(&Style::opaque(Color::WHITE), &mut assets);
                drop(pool.get_or_add(&Style::opaque(Color::BLACK), &mut assets));
                held
            };
            app.update();
            let assets = app.world().resource::<Assets<GradMeshMaterial>>();
            // The pool still lists both entries until it is pruned; one is dead.
            assert_eq!(pool.len(), 2);
            assert_eq!(pool.live_len(assets), 1);
            drop(held);
        }

        #[test]
        fn the_warning_threshold_is_configurable() {
            assert_eq!(MaterialWarning::default().threshold, MATERIAL_WARNING_THRESHOLD);
        }
    }
}
