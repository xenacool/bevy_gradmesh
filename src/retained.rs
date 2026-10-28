//! Retained mode API for BevyGradmesh.
//!
//! Unlike the immediate mode (`ImmediateRecorder`) which creates and destroys
//! entities each frame, the retained mode creates shapes once and returns a
//! `GradMeshHandle` that can be updated per-frame without regenerating geometry.
//!
//! # Example
//!
//! ```ignore
//! use bevy::prelude::*;
//! use bevy_gradmesh::{GradMeshPlugin, retained::RetainedWorld, Shape, ShapeDetail, imm_opaque};
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
//!     mut materials: ResMut<Assets<GradMeshMaterial>>,
//!     mut world: ResMut<World>,
//!     time: Res<Time>,
//! ) {
//!     let handle = world.0.sphere(
//!         &mut commands, &mut meshes, &mut materials,
//!         0.5, ShapeDetail::Medium, imm_opaque(1.0, 0.0, 0.0),
//!     );
//!     world.0.set_position(handle, Vec3::new(time.elapsed_secs(), 0.0, 0.0));
//! }
//! ```

use bevy::asset::Assets;
use bevy::ecs::prelude::*;
use bevy::math::{Quat, Vec2, Vec3};
use bevy::prelude::{Mesh3d, MeshMaterial3d};
use bevy::render::mesh::Mesh;
use bevy::transform::components::{GlobalTransform, Transform};

use crate::immediate::BillboardMode;
use crate::material::GradMeshMaterial;
use crate::shape::{Shape, ShapeDetail};

/// Opaque handle to a retained shape.
///
/// Use this to update a shape's transform or style after it has been created.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GradMeshHandle {
    entity: Entity,
}

/// Retained mode world — owns a collection of shapes that persist across frames.
///
/// Shapes are created once and can be updated via their `GradMeshHandle`.
/// The world manages a root entity and mesh/material caching.
pub struct RetainedWorld {
    root: Option<Entity>,
    /// Maps handles to their current shape (for mesh caching on style changes).
    handles: Vec<GradMeshHandle>,
}

impl RetainedWorld {
    /// Create a new retained world.
    pub fn new() -> Self {
        Self {
            root: None,
            handles: Vec::new(),
        }
    }

    /// Ensure the root entity exists.
    fn ensure_root(&mut self, commands: &mut Commands) -> Entity {
        if self.root.is_none() {
            self.root = Some(commands.spawn(Transform::default()).id());
        }
        self.root.unwrap()
    }

    // ─── 3D shape creation ───

    /// Create a sphere and return a handle for updating it.
    pub fn sphere(
        &mut self,
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<GradMeshMaterial>,
        radius: f32,
        detail: ShapeDetail,
        material: GradMeshMaterial,
    ) -> GradMeshHandle {
        let shape = Shape::sphere(radius, detail);
        self.spawn_shape(commands, meshes, materials, &shape, material, Vec3::ZERO, Quat::IDENTITY, Vec3::ONE)
    }

    /// Create a cuboid and return a handle for updating it.
    pub fn cuboid(
        &mut self,
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<GradMeshMaterial>,
        size: Vec3,
        roundness: f32,
        detail: ShapeDetail,
        material: GradMeshMaterial,
    ) -> GradMeshHandle {
        let shape = Shape::cuboid(size, roundness, detail);
        self.spawn_shape(commands, meshes, materials, &shape, material, Vec3::ZERO, Quat::IDENTITY, Vec3::ONE)
    }

    /// Create a cylinder and return a handle for updating it.
    pub fn cylinder(
        &mut self,
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<GradMeshMaterial>,
        radius: f32,
        height: f32,
        detail: ShapeDetail,
        material: GradMeshMaterial,
    ) -> GradMeshHandle {
        let shape = Shape::cylinder(radius, height, detail);
        self.spawn_shape(commands, meshes, materials, &shape, material, Vec3::ZERO, Quat::IDENTITY, Vec3::ONE)
    }

    /// Create a capsule and return a handle for updating it.
    pub fn capsule(
        &mut self,
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<GradMeshMaterial>,
        radius: f32,
        height: f32,
        detail: ShapeDetail,
        material: GradMeshMaterial,
    ) -> GradMeshHandle {
        let shape = Shape::capsule(radius, height, detail);
        self.spawn_shape(commands, meshes, materials, &shape, material, Vec3::ZERO, Quat::IDENTITY, Vec3::ONE)
    }

    /// Create a cone and return a handle for updating it.
    pub fn cone(
        &mut self,
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<GradMeshMaterial>,
        radius: f32,
        height: f32,
        detail: ShapeDetail,
        material: GradMeshMaterial,
    ) -> GradMeshHandle {
        let shape = Shape::cone(radius, height, detail);
        self.spawn_shape(commands, meshes, materials, &shape, material, Vec3::ZERO, Quat::IDENTITY, Vec3::ONE)
    }

    /// Create a torus and return a handle for updating it.
    pub fn torus(
        &mut self,
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<GradMeshMaterial>,
        radius: f32,
        thickness: f32,
        start_angle: f32,
        span: f32,
        material: GradMeshMaterial,
    ) -> GradMeshHandle {
        let shape = Shape::torus(radius, thickness, start_angle, span);
        self.spawn_shape(commands, meshes, materials, &shape, material, Vec3::ZERO, Quat::IDENTITY, Vec3::ONE)
    }

    /// Create an ellipsoid and return a handle for updating it.
    pub fn ellipsoid(
        &mut self,
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<GradMeshMaterial>,
        size: Vec3,
        detail: ShapeDetail,
        material: GradMeshMaterial,
    ) -> GradMeshHandle {
        let shape = Shape::ellipsoid(size, detail);
        self.spawn_shape(commands, meshes, materials, &shape, material, Vec3::ZERO, Quat::IDENTITY, Vec3::ONE)
    }

    /// Create a hemisphere and return a handle for updating it.
    pub fn hemisphere(
        &mut self,
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<GradMeshMaterial>,
        radius: f32,
        capped: bool,
        detail: ShapeDetail,
        material: GradMeshMaterial,
    ) -> GradMeshHandle {
        let shape = Shape::hemisphere(radius, capped, detail);
        self.spawn_shape(commands, meshes, materials, &shape, material, Vec3::ZERO, Quat::IDENTITY, Vec3::ONE)
    }

    /// Create a regular prism and return a handle for updating it.
    pub fn regular_prism(
        &mut self,
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<GradMeshMaterial>,
        sides: u32,
        radius: f32,
        height: f32,
        roundness: f32,
        detail: ShapeDetail,
        material: GradMeshMaterial,
    ) -> GradMeshHandle {
        let shape = Shape::regular_prism(sides, radius, height, roundness, detail);
        self.spawn_shape(commands, meshes, materials, &shape, material, Vec3::ZERO, Quat::IDENTITY, Vec3::ONE)
    }

    /// Create a pyramid and return a handle for updating it.
    pub fn pyramid(
        &mut self,
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<GradMeshMaterial>,
        sides: u32,
        radius: f32,
        height: f32,
        detail: ShapeDetail,
        material: GradMeshMaterial,
    ) -> GradMeshHandle {
        let shape = Shape::pyramid(sides, radius, height, detail);
        self.spawn_shape(commands, meshes, materials, &shape, material, Vec3::ZERO, Quat::IDENTITY, Vec3::ONE)
    }

    // ─── 2D shape creation ───

    /// Create a disc (2D circle) and return a handle for updating it.
    pub fn disc(
        &mut self,
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<GradMeshMaterial>,
        radius: f32,
        material: GradMeshMaterial,
        billboard: BillboardMode,
    ) -> GradMeshHandle {
        let shape = Shape::disc(radius, ShapeDetail::Medium);
        self.spawn_2d_shape(commands, meshes, materials, &shape, material, billboard)
    }

    /// Create a disc rim (2D ring) and return a handle for updating it.
    pub fn disc_rim(
        &mut self,
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<GradMeshMaterial>,
        radius: f32,
        rim_width: f32,
        material: GradMeshMaterial,
        billboard: BillboardMode,
    ) -> GradMeshHandle {
        let shape = Shape::disc_rim(radius, rim_width, ShapeDetail::Medium);
        self.spawn_2d_shape(commands, meshes, materials, &shape, material, billboard)
    }

    /// Create a rectangle (2D) and return a handle for updating it.
    pub fn rectangle(
        &mut self,
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<GradMeshMaterial>,
        size: Vec2,
        material: GradMeshMaterial,
        billboard: BillboardMode,
    ) -> GradMeshHandle {
        let shape = Shape::rectangle(size, ShapeDetail::Medium);
        self.spawn_2d_shape(commands, meshes, materials, &shape, material, billboard)
    }

    /// Create a polygon (2D) and return a handle for updating it.
    pub fn polygon(
        &mut self,
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<GradMeshMaterial>,
        sides: u32,
        radius: f32,
        material: GradMeshMaterial,
        billboard: BillboardMode,
    ) -> GradMeshHandle {
        let shape = Shape::polygon(sides, radius, ShapeDetail::Minimal);
        self.spawn_2d_shape(commands, meshes, materials, &shape, material, billboard)
    }

    /// Create an ellipse (2D) and return a handle for updating it.
    pub fn ellipse(
        &mut self,
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<GradMeshMaterial>,
        radius: Vec2,
        material: GradMeshMaterial,
        billboard: BillboardMode,
    ) -> GradMeshHandle {
        let shape = Shape::ellipse(radius, ShapeDetail::Medium);
        self.spawn_2d_shape(commands, meshes, materials, &shape, material, billboard)
    }

    // ─── Generic shape creation ───

    /// Create any shape and return a handle for updating it.
    pub fn shape(
        &mut self,
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<GradMeshMaterial>,
        shape: &Shape,
        material: GradMeshMaterial,
        position: Vec3,
        rotation: Quat,
        scale: Vec3,
    ) -> GradMeshHandle {
        self.spawn_shape(commands, meshes, materials, shape, material, position, rotation, scale)
    }

    // ─── Handle updates ───

    /// Update the position of a retained shape.
    pub fn set_position(&self, handle: GradMeshHandle, position: Vec3, query: &mut Query<&mut Transform, With<GradMeshEntityMarker>>) {
        if let Ok(mut transform) = query.get_mut(handle.entity) {
            transform.translation = position;
        }
    }

    /// Update the rotation of a retained shape.
    pub fn set_rotation(&self, handle: GradMeshHandle, rotation: Quat, query: &mut Query<&mut Transform, With<GradMeshEntityMarker>>) {
        if let Ok(mut transform) = query.get_mut(handle.entity) {
            transform.rotation = rotation;
        }
    }

    /// Update the scale of a retained shape.
    pub fn set_scale(&self, handle: GradMeshHandle, scale: Vec3, query: &mut Query<&mut Transform, With<GradMeshEntityMarker>>) {
        if let Ok(mut transform) = query.get_mut(handle.entity) {
            transform.scale = scale;
        }
    }

    /// Update the transform of a retained shape.
    pub fn set_transform(
        &self,
        handle: GradMeshHandle,
        position: Vec3,
        rotation: Quat,
        scale: Vec3,
        query: &mut Query<&mut Transform, With<GradMeshEntityMarker>>,
    ) {
        if let Ok(mut transform) = query.get_mut(handle.entity) {
            transform.translation = position;
            transform.rotation = rotation;
            transform.scale = scale;
        }
    }

    /// Remove a retained shape.
    pub fn remove(&mut self, handle: GradMeshHandle, commands: &mut Commands, query: &mut Query<Entity, With<GradMeshEntityMarker>>) {
        if query.get(handle.entity).is_ok() {
            commands.entity(handle.entity).despawn();
        }
        self.handles.retain(|h| h != &handle);
    }

    /// Returns the number of retained shapes.
    pub fn len(&self) -> usize {
        self.handles.len()
    }

    /// Returns true if there are no retained shapes.
    pub fn is_empty(&self) -> bool {
        self.handles.is_empty()
    }

    // ─── Internal helpers ───

    fn spawn_shape(
        &mut self,
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<GradMeshMaterial>,
        shape: &Shape,
        mut material: GradMeshMaterial,
        position: Vec3,
        rotation: Quat,
        scale: Vec3,
    ) -> GradMeshHandle {
        // Set world center for object-space gradient computation
        material.extension.uniforms.world_center = crate::material::Vec4::new(
            position.x, position.y, position.z, 0.0,
        );

        let mesh_handle = meshes.add(shape.build());
        let mat_handle = materials.add(material);
        let root = self.ensure_root(commands);

        let entity = commands
            .entity(root)
            .with_children(|parent| {
                parent.spawn((
                    Mesh3d(mesh_handle),
                    MeshMaterial3d(mat_handle),
                    Transform::from_translation(position)
                        .with_rotation(rotation)
                        .with_scale(scale),
                    GlobalTransform::default(),
                    GradMeshEntityMarker,
                ));
            })
            .id();

        let handle = GradMeshHandle { entity };
        self.handles.push(handle);
        handle
    }

    fn spawn_2d_shape(
        &mut self,
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<GradMeshMaterial>,
        shape: &Shape,
        material: GradMeshMaterial,
        billboard: BillboardMode,
    ) -> GradMeshHandle {
        let mesh_handle = meshes.add(shape.build());
        let mat_handle = materials.add(material);
        let root = self.ensure_root(commands);

        let mut transform = Transform::from_translation(Vec3::ZERO);
        match billboard {
            BillboardMode::None => {}
            BillboardMode::FaceCameraY | BillboardMode::FaceCamera => {
                transform.rotation *= Quat::from_rotation_x(std::f32::consts::PI / 2.0);
            }
        }

        let entity = commands
            .entity(root)
            .with_children(|parent| {
                parent.spawn((
                    Mesh3d(mesh_handle),
                    MeshMaterial3d(mat_handle),
                    transform,
                    GlobalTransform::default(),
                    GradMeshEntityMarker,
                ));
            })
            .id();

        let handle = GradMeshHandle { entity };
        self.handles.push(handle);
        handle
    }
}

impl Default for RetainedWorld {
    fn default() -> Self {
        Self::new()
    }
}

/// Marker component for entities spawned by RetainedWorld.
#[derive(Component)]
pub struct GradMeshEntityMarker;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_equality() {
        let h1 = GradMeshHandle { entity: Entity::from_bits(1) };
        let h2 = GradMeshHandle { entity: Entity::from_bits(1) };
        let h3 = GradMeshHandle { entity: Entity::from_bits(2) };
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn retained_world_starts_empty() {
        let world = RetainedWorld::new();
        assert!(world.is_empty());
        assert_eq!(world.len(), 0);
    }
}
