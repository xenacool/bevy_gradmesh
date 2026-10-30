//! Retained mode API for BevyGradmesh.
//!
//! Unlike the immediate mode (`ImmediateRecorder`) which creates and destroys
//! entities each frame, the retained mode creates shapes once and returns a
//! `GradMeshHandle` that can be updated per-frame without regenerating geometry.
//!
//! Under the hood, `RetainedWorld` uses batched instanced rendering via
//! `BatchRecorder`: shapes are accumulated into batches keyed by (mesh, blend
//! mode, render state), and per-frame updates modify batch instance data
//! directly. This eliminates both per-shape entities and per-shape material
//! assets, reducing draw calls and descriptor pool pressure.
//!
//! # Example
//!
//! ```ignore
//! use bevy::prelude::*;
//! use bevy_gradmesh::{GradMeshPlugin, retained::RetainedWorld, Shape, ShapeDetail};
//!
//! #[derive(Resource)]
//! struct World(RetainedWorld);
//!
//! fn setup(mut commands: Commands) {
//!     commands.insert_resource(World(RetainedWorld::new()));
//! }
//!
//! fn spawn_and_update(
//!     mut commands: Commands,
//!     mut meshes: ResMut<Assets<Mesh>>,
//!     mut world: ResMut<World>,
//!     time: Res<Time>,
//! ) {
//!     let handle = world.0.sphere(
//!         &mut commands, &mut meshes,
//!         Vec3::ZERO, 0.5, ShapeDetail::Medium, [1.0, 0.0, 0.0, 1.0],
//!     );
//!     world.0.set_position(handle, Vec3::new(time.elapsed_secs(), 0.0, 0.0));
//!     world.0.flush(&mut commands, &mut meshes);
//! }
//! ```

use bevy::asset::Assets;
use bevy::ecs::prelude::*;
use bevy::math::{Quat, Vec3};
use bevy::render::mesh::Mesh;

use crate::batch::{BatchRecorder, InstanceProperties};
use crate::shape::{Shape, ShapeDetail};

/// Opaque handle to a retained shape.
///
/// Unlike a simple entity reference, this handle wraps a slot index into the
/// world's parallel arrays with generation-based validation.  The world maps
/// each slot to a batch instance, so updates modify batch buffers directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GradMeshHandle {
    /// World ID for validation across multiple worlds.
    world_id: u64,
    /// Index into the world's parallel arrays.
    slot: usize,
    /// Generation counter for validity checking.
    generation: u32,
}

/// Retained mode world that renders shapes via batched instancing.
///
/// Combines the retained handle API with `BatchRecorder` internally: shapes are
/// accumulated into batches keyed by (mesh, blend mode, render state), and
/// per-frame updates modify batch instance data directly.  This eliminates both
/// per-shape entities and per-shape material assets, reducing draw calls and
/// descriptor pool pressure.
///
/// # Architecture
///
/// Uses a slot-based parallel array pattern:
/// - Parallel arrays (`positions`, `rotations`, `scales`, `properties`)
///   indexed by slot
/// - Handle → slot → (batch_key, batch_index) chain
/// - Updates like `set_position` modify the parallel arrays and mark dirty
/// - `flush()` replays all alive slots into batches and spawns entities
///
/// # Example
///
/// ```ignore
/// let mut world = RetainedWorld::new();
///
/// // Add shapes
/// let h1 = world.sphere(&mut commands, &mut meshes,
///     Vec3::ZERO, 0.5, ShapeDetail::Medium, [1.0, 0.0, 0.0, 1.0]);
/// let h2 = world.sphere(&mut commands, &mut meshes,
///     Vec3::X, 0.5, ShapeDetail::Medium, [1.0, 0.0, 0.0, 1.0]);
/// // Both share one batch → 1 draw call
///
/// // Update per frame
/// world.set_position(h1, Vec3::new(time.elapsed_secs(), 0.0, 0.0));
///
/// // Flush changes to GPU
/// world.flush(&mut commands, &mut meshes);
/// ```
pub struct RetainedWorld {
    /// Unique world ID for handle validation.
    world_id: u64,
    /// Parallel arrays — one entry per slot.
    positions: Vec<Vec3>,
    rotations: Vec<Quat>,
    scales: Vec<Vec3>,
    properties: Vec<InstanceProperties>,
    /// Shape reference for mesh caching.
    shapes: Vec<Shape>,
    /// Generation counter per slot (0 = free).
    generations: Vec<u32>,
    /// Whether a slot is currently alive.
    alive: Vec<bool>,
    /// Render state per slot.
    render_states: Vec<crate::RenderState>,
    /// Next available free slot.
    free_slots: Vec<usize>,
    /// Internal batch recorder.
    recorder: BatchRecorder,
    /// Whether data is dirty since last flush.
    dirty: bool,
}

impl RetainedWorld {
    /// Create a new retained world.
    pub fn new() -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let world_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self {
            world_id,
            positions: Vec::new(),
            rotations: Vec::new(),
            scales: Vec::new(),
            properties: Vec::new(),
            shapes: Vec::new(),
            generations: Vec::new(),
            alive: Vec::new(),
            render_states: Vec::new(),
            free_slots: Vec::new(),
            recorder: BatchRecorder::new(),
            dirty: false,
        }
    }

    /// Allocate a new slot and return its index.
    fn alloc_slot(&mut self) -> usize {
        if let Some(slot) = self.free_slots.pop() {
            self.alive[slot] = true;
            self.generations[slot] += 1;
            slot
        } else {
            let slot = self.positions.len();
            self.positions.push(Vec3::ZERO);
            self.rotations.push(Quat::IDENTITY);
            self.scales.push(Vec3::ONE);
            self.properties.push(InstanceProperties::default());
            self.shapes.push(Shape::sphere(0.5, ShapeDetail::Medium));
            self.generations.push(1);
            self.alive.push(true);
            self.render_states.push(crate::RenderState::default());
            slot
        }
    }

    /// Validate a handle and return its slot index, or None if invalid.
    fn validate_handle(&self, handle: GradMeshHandle) -> Option<usize> {
        if handle.world_id != self.world_id {
            return None;
        }
        if handle.slot >= self.alive.len() || !self.alive[handle.slot] {
            return None;
        }
        if self.generations[handle.slot] != handle.generation {
            return None;
        }
        Some(handle.slot)
    }

    // ─── Shape creation ───

    fn add_shape(
        &mut self,
        _commands: &mut Commands,
        _meshes: &mut Assets<Mesh>,
        shape: Shape,
        position: Vec3,
        rotation: Quat,
        scale: Vec3,
        properties: InstanceProperties,
        render_state: crate::RenderState,
    ) -> GradMeshHandle {
        let slot = self.alloc_slot();
        self.positions[slot] = position;
        self.rotations[slot] = rotation;
        self.scales[slot] = scale;
        self.properties[slot] = properties;
        self.shapes[slot] = shape;
        self.render_states[slot] = render_state;
        self.dirty = true;

        GradMeshHandle {
            world_id: self.world_id,
            slot,
            generation: self.generations[slot],
        }
    }

    /// Add a sphere and return a handle for updating it.
    pub fn sphere(
        &mut self,
        _commands: &mut Commands,
        _meshes: &mut Assets<Mesh>,
        position: Vec3,
        radius: f32,
        detail: ShapeDetail,
        color: [f32; 4],
    ) -> GradMeshHandle {
        let shape = Shape::sphere(radius, detail);
        let props = InstanceProperties::from_style(
            color, color, [0.0, 1.0, 0.0, 0.0],
            crate::style::ColorMode::Single, crate::style::BlendMode::Opaque, false, None,
        );
        self.add_shape(_commands, _meshes, shape, position, Quat::IDENTITY, Vec3::ONE, props, crate::RenderState::default())
    }

    /// Add a cuboid and return a handle for updating it.
    pub fn cuboid(
        &mut self,
        _commands: &mut Commands,
        _meshes: &mut Assets<Mesh>,
        position: Vec3,
        size: Vec3,
        roundness: f32,
        detail: ShapeDetail,
        color: [f32; 4],
    ) -> GradMeshHandle {
        let shape = Shape::cuboid(size, roundness, detail);
        let props = InstanceProperties::from_style(
            color, color, [0.0, 1.0, 0.0, 0.0],
            crate::style::ColorMode::Single, crate::style::BlendMode::Opaque, false, None,
        );
        self.add_shape(_commands, _meshes, shape, position, Quat::IDENTITY, Vec3::ONE, props, crate::RenderState::default())
    }

    /// Add a cylinder and return a handle for updating it.
    pub fn cylinder(
        &mut self,
        _commands: &mut Commands,
        _meshes: &mut Assets<Mesh>,
        position: Vec3,
        radius: f32,
        height: f32,
        detail: ShapeDetail,
        color: [f32; 4],
    ) -> GradMeshHandle {
        let shape = Shape::cylinder(radius, height, detail);
        let props = InstanceProperties::from_style(
            color, color, [0.0, 1.0, 0.0, 0.0],
            crate::style::ColorMode::Single, crate::style::BlendMode::Opaque, false, None,
        );
        self.add_shape(_commands, _meshes, shape, position, Quat::IDENTITY, Vec3::ONE, props, crate::RenderState::default())
    }

    /// Add a capsule and return a handle for updating it.
    pub fn capsule(
        &mut self,
        _commands: &mut Commands,
        _meshes: &mut Assets<Mesh>,
        position: Vec3,
        radius: f32,
        height: f32,
        detail: ShapeDetail,
        color: [f32; 4],
    ) -> GradMeshHandle {
        let shape = Shape::capsule(radius, height, detail);
        let props = InstanceProperties::from_style(
            color, color, [0.0, 1.0, 0.0, 0.0],
            crate::style::ColorMode::Single, crate::style::BlendMode::Opaque, false, None,
        );
        self.add_shape(_commands, _meshes, shape, position, Quat::IDENTITY, Vec3::ONE, props, crate::RenderState::default())
    }

    /// Add a cone and return a handle for updating it.
    pub fn cone(
        &mut self,
        _commands: &mut Commands,
        _meshes: &mut Assets<Mesh>,
        position: Vec3,
        radius: f32,
        height: f32,
        detail: ShapeDetail,
        color: [f32; 4],
    ) -> GradMeshHandle {
        let shape = Shape::cone(radius, height, detail);
        let props = InstanceProperties::from_style(
            color, color, [0.0, 1.0, 0.0, 0.0],
            crate::style::ColorMode::Single, crate::style::BlendMode::Opaque, false, None,
        );
        self.add_shape(_commands, _meshes, shape, position, Quat::IDENTITY, Vec3::ONE, props, crate::RenderState::default())
    }

    /// Add a torus and return a handle for updating it.
    pub fn torus(
        &mut self,
        _commands: &mut Commands,
        _meshes: &mut Assets<Mesh>,
        position: Vec3,
        radius: f32,
        thickness: f32,
        start_angle: f32,
        span: f32,
        color: [f32; 4],
    ) -> GradMeshHandle {
        let shape = Shape::torus(radius, thickness, start_angle, span);
        let props = InstanceProperties::from_style(
            color, color, [0.0, 1.0, 0.0, 0.0],
            crate::style::ColorMode::Single, crate::style::BlendMode::Opaque, false, None,
        );
        self.add_shape(_commands, _meshes, shape, position, Quat::IDENTITY, Vec3::ONE, props, crate::RenderState::default())
    }

    /// Add a generic shape with full control over properties.
    pub fn shape(
        &mut self,
        _commands: &mut Commands,
        _meshes: &mut Assets<Mesh>,
        shape: Shape,
        position: Vec3,
        rotation: Quat,
        scale: Vec3,
        properties: InstanceProperties,
        render_state: crate::RenderState,
    ) -> GradMeshHandle {
        self.add_shape(_commands, _meshes, shape, position, rotation, scale, properties, render_state)
    }

    // ─── Handle updates ───

    /// Update the position of a retained shape.
    pub fn set_position(&mut self, handle: GradMeshHandle, position: Vec3) {
        if let Some(slot) = self.validate_handle(handle) {
            self.positions[slot] = position;
            self.dirty = true;
        }
    }

    /// Update the rotation of a retained shape.
    pub fn set_rotation(&mut self, handle: GradMeshHandle, rotation: Quat) {
        if let Some(slot) = self.validate_handle(handle) {
            self.rotations[slot] = rotation;
            self.dirty = true;
        }
    }

    /// Update the scale of a retained shape.
    pub fn set_scale(&mut self, handle: GradMeshHandle, scale: Vec3) {
        if let Some(slot) = self.validate_handle(handle) {
            self.scales[slot] = scale;
            self.dirty = true;
        }
    }

    /// Update the transform of a retained shape.
    pub fn set_transform(
        &mut self,
        handle: GradMeshHandle,
        position: Vec3,
        rotation: Quat,
        scale: Vec3,
    ) {
        if let Some(slot) = self.validate_handle(handle) {
            self.positions[slot] = position;
            self.rotations[slot] = rotation;
            self.scales[slot] = scale;
            self.dirty = true;
        }
    }

    /// Update the color of a retained shape.
    pub fn set_color(&mut self, handle: GradMeshHandle, color: [f32; 4]) {
        if let Some(slot) = self.validate_handle(handle) {
            self.properties[slot].color = crate::material::Vec4::new(color[0], color[1], color[2], color[3]);
            self.dirty = true;
        }
    }

    /// Remove a retained shape.
    pub fn remove(&mut self, handle: GradMeshHandle) {
        if let Some(slot) = self.validate_handle(handle) {
            self.alive[slot] = false;
            self.free_slots.push(slot);
            self.dirty = true;
        }
    }

    /// Returns the number of active shapes.
    pub fn len(&self) -> usize {
        self.alive.iter().filter(|&&a| a).count()
    }

    /// Returns true if there are no active shapes.
    pub fn is_empty(&self) -> bool {
        self.alive.iter().all(|&a| !a)
    }

    // ─── Flush to batches ───

    /// Rebuild batches from the current parallel arrays and spawn entities.
    ///
    /// This is the key method that converts retained state into native Bevy
    /// entities. Each instance becomes a Mesh3d + Material3d entity, and Bevy's
    /// render pipeline handles batching automatically.
    ///
    /// Returns the spawned entity IDs.
    pub fn flush(
        &mut self,
        commands: &mut Commands,
        meshes: bevy::ecs::system::ResMut<Assets<Mesh>>,
        materials: bevy::ecs::system::ResMut<Assets<crate::GradMeshMaterial>>,
    ) -> Vec<Entity> {
        self.recorder.begin();

        for slot in 0..self.alive.len() {
            if !self.alive[slot] {
                continue;
            }

            self.recorder.set_position(self.positions[slot]);
            self.recorder.set_rotation(self.rotations[slot]);
            self.recorder.set_scale(self.scales[slot]);
            self.recorder.set_render_state(self.render_states[slot]);

            let shape = &self.shapes[slot];
            let props = self.properties[slot];
            self.recorder.add_instance(shape, props);
        }

        self.recorder.end();
        let spawned = self.recorder.render_batches(commands, meshes, materials, None);
        self.dirty = false;
        spawned
    }

    /// Returns true if there are pending changes since the last flush.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Returns the number of batches (draw calls) from the last flush.
    pub fn batch_count(&self) -> usize {
        self.recorder.batch_count()
    }

    /// Returns the total instance count across all batches.
    pub fn total_instances(&self) -> usize {
        self.recorder.total_instances()
    }
}

impl Default for RetainedWorld {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_equality() {
        let world = RetainedWorld::new();
        let h1 = GradMeshHandle { world_id: world.world_id, slot: 0, generation: 1 };
        let h2 = GradMeshHandle { world_id: world.world_id, slot: 0, generation: 1 };
        let h3 = GradMeshHandle { world_id: world.world_id, slot: 1, generation: 1 };
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn retained_world_starts_empty() {
        let world = RetainedWorld::new();
        assert!(world.is_empty());
        assert_eq!(world.len(), 0);
    }

    #[test]
    fn retained_handle_validation() {
        let mut world = RetainedWorld::new();

        // Create a fake handle with wrong world ID
        let bad_handle = GradMeshHandle {
            world_id: 999,
            slot: 0,
            generation: 1,
        };
        assert!(world.validate_handle(bad_handle).is_none());
    }

    #[test]
    fn retained_world_alloc_and_free() {
        let mut world = RetainedWorld::new();
        let slot1 = world.alloc_slot();
        assert_eq!(slot1, 0);
        assert!(world.alive[slot1]);

        // Free it
        world.alive[slot1] = false;
        world.free_slots.push(slot1);

        // Reallocate should reuse the slot
        let slot2 = world.alloc_slot();
        assert_eq!(slot2, 0);
        assert_eq!(world.generations[0], 2);
    }

    #[test]
    fn retained_world_grouping_reduces_batches() {
        // Verify that the BatchRecorder groups identical shapes
        let mut rec = BatchRecorder::new();
        rec.begin();

        // Add 5 identical spheres at different positions
        for i in 0..5 {
            rec.set_position(Vec3::new(i as f32, 0.0, 0.0));
            rec.sphere(0.5, ShapeDetail::Medium, [1.0, 0.0, 0.0, 1.0]);
        }

        let batches = rec.end();
        // All 5 spheres should be in 1 batch
        assert_eq!(batches.len(), 1);
        assert_eq!(batches.values().next().unwrap().count, 5);
    }

    #[test]
    fn retained_world_different_colors_same_batch() {
        // BatchRecorder groups by mesh+blend+render_state, not by color
        // (colors are per-instance properties)
        let mut rec = BatchRecorder::new();
        rec.begin();

        rec.set_position(Vec3::ZERO);
        rec.sphere(0.5, ShapeDetail::Medium, [1.0, 0.0, 0.0, 1.0]);
        rec.set_position(Vec3::X);
        rec.sphere(0.5, ShapeDetail::Medium, [0.0, 1.0, 0.0, 1.0]);
        rec.set_position(Vec3::Y);
        rec.sphere(0.5, ShapeDetail::Medium, [0.0, 0.0, 1.0, 1.0]);

        let batches = rec.end();
        // All 3 spheres share 1 batch despite different colors
        assert_eq!(batches.len(), 1);
        assert_eq!(batches.values().next().unwrap().count, 3);
    }

    #[test]
    fn retained_world_different_blend_modes_separate_batches() {
        let mut rec = BatchRecorder::new();
        rec.begin();

        rec.set_position(Vec3::ZERO);
        rec.sphere(0.5, ShapeDetail::Medium, [1.0, 0.0, 0.0, 1.0]);
        rec.set_position(Vec3::X);
        rec.sphere_transparent(0.5, ShapeDetail::Medium, [1.0, 0.0, 0.0, 0.5]);

        let batches = rec.end();
        // Opaque and transparent go to different batches
        assert_eq!(batches.len(), 2);
    }
}
