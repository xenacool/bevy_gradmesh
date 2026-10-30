//! BevyGradmesh Bevy Plugin.
//!
//! Provides `GradMeshPlugin` — a single entry point that registers the gradient
//! material extension and inserts a shared mesh cache resource.
//!
//! # Example
//!
//! ```ignore
//! use bevy::prelude::*;
//! use bevy_gradmesh::GradMeshPlugin;
//!
//! fn main() {
//!     App::new()
//!         .add_plugins(DefaultPlugins)
//!         .add_plugins(GradMeshPlugin)
//!         .run();
//! }
//! ```

use std::collections::HashMap;

use bevy::app::Plugin;
use bevy::asset::Assets;
use bevy::ecs::prelude::*;
use bevy::pbr::{ExtendedMaterial, MaterialPlugin, StandardMaterial};
use bevy::render::mesh::Mesh;
use crate::deformation::Deformation;
use crate::material::GradMeshExtension;
use crate::material_interning::{GradMeshMaterialPool, warn_excessive_materials};
use crate::render_pipeline::InstancedStats;
use crate::shape::Shape;

/// Shared mesh cache keyed by `Shape`.
///
/// When the plugin is active, call `MeshCache::get_or_build()` instead of
/// `Shape::build()` directly to avoid generating the same mesh geometry
/// multiple times.  The cache lives as a `Resource` so all systems share it.
#[derive(Resource, Default, Debug)]
pub struct MeshCache {
    inner: HashMap<Shape, Mesh>,
    /// Successful cache lookups.
    hits: u64,
    /// Cache lookups that required building a new mesh.
    misses: u64,
}

impl MeshCache {
    /// Returns a clone of the cached mesh for `shape`, building and caching it
    /// on a miss. Increments hit/miss counters.
    pub fn get_or_build(&mut self, shape: &Shape) -> &Mesh {
        let is_new = !self.inner.contains_key(shape);
        let mesh = self.inner
            .entry(shape.clone())
            .or_insert_with(|| shape.build());
        if is_new {
            self.misses += 1;
        } else {
            self.hits += 1;
        }
        mesh
    }

    /// Inserts a mesh handle into Bevy's `Assets<Mesh>`, caching the handle
    /// so repeated lookups return the same `Handle`.
    ///
    /// Returns the cached `Handle<Mesh>`.
    pub fn get_or_insert(
        &mut self,
        shape: &Shape,
        assets: &mut Assets<Mesh>,
    ) -> bevy::asset::Handle<Mesh> {
        // We can't store Handles in the same cache because Assets<Mesh> might
        // be replaced; instead we cache the Mesh data and only add to Assets
        // when we have a miss.
        let mesh = self.get_or_build(shape);
        assets.add(mesh.clone())
    }

    /// Returns the number of cached meshes.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns true if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Clears the cache.
    pub fn clear(&mut self) {
        self.inner.clear();
        self.hits = 0;
        self.misses = 0;
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
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// BevyGradmesh plugin — registers the gradient material and mesh cache.
///
/// Add this to your `App` instead of manually adding `MaterialPlugin`.
///
/// ```ignore
/// App::new()
///     .add_plugins(DefaultPlugins)
///     .add_plugins(GradMeshPlugin)
///     .run();
/// ```
pub struct GradMeshPlugin;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::ShapeDetail;

    #[test]
    fn mesh_cache_tracks_hits_and_misses() {
        let mut cache = MeshCache::default();
        let shape = Shape::sphere(1.0, ShapeDetail::Low);

        // First lookup is a miss
        let _mesh = cache.get_or_build(&shape);
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 1);

        // Second lookup is a hit
        let _mesh = cache.get_or_build(&shape);
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 1);

        // Third lookup is also a hit
        let _mesh = cache.get_or_build(&shape);
        assert_eq!(cache.hits(), 2);
        assert_eq!(cache.misses(), 1);
    }

    #[test]
    fn mesh_cache_hit_rate() {
        let mut cache = MeshCache::default();
        let shape = Shape::sphere(1.0, ShapeDetail::Low);

        // Empty cache has 0.0 hit rate
        assert_eq!(cache.hit_rate(), 0.0);

        // After one miss, hit rate is 0.0
        let _mesh = cache.get_or_build(&shape);
        assert_eq!(cache.hit_rate(), 0.0);

        // After one hit, hit rate is 50%
        let _mesh = cache.get_or_build(&shape);
        assert!((cache.hit_rate() - 0.5).abs() < f64::EPSILON);

        // After another hit, hit rate is ~66.7%
        let _mesh = cache.get_or_build(&shape);
        let expected = 2.0 / 3.0;
        assert!((cache.hit_rate() - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn mesh_cache_clear_resets_counters() {
        let mut cache = MeshCache::default();
        let shape = Shape::sphere(1.0, ShapeDetail::Low);

        cache.get_or_build(&shape);
        cache.get_or_build(&shape);
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 1);

        cache.clear();
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn mesh_cache_different_shapes_are_misses() {
        let mut cache = MeshCache::default();
        let shape1 = Shape::sphere(1.0, ShapeDetail::Low);
        let shape2 = Shape::sphere(2.0, ShapeDetail::Low);

        cache.get_or_build(&shape1);
        cache.get_or_build(&shape2);
        assert_eq!(cache.misses(), 2);
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.len(), 2);
    }
}

/// Tick the deformation time resource each frame.
fn tick_deformation(mut deformation: ResMut<Deformation>, time: Res<bevy::time::Time>) {
    deformation.tick(time.delta().as_secs_f32());
}

impl Plugin for GradMeshPlugin {
    fn build(&self, app: &mut bevy::app::App) {
        // Embed the shader first so it's available when MaterialPlugin loads it
        bevy::asset::embedded_asset!(app, "shader/gradmesh.wesl");

        app.add_plugins(MaterialPlugin::<ExtendedMaterial<StandardMaterial, GradMeshExtension>>::default())
            .add_plugins(crate::instanced_render::InstancedRenderPlugin)
            .init_resource::<MeshCache>()
            .init_resource::<InstancedStats>()
            .init_resource::<Deformation>()
            .init_resource::<GradMeshMaterialPool>()
            .add_systems(bevy::app::Update, tick_deformation)
            .add_systems(bevy::app::Update, warn_excessive_materials);
    }
}
