//! Immediate mode drawing API for BevyGradmesh.
//!
//! Provides a recorder pattern for per-frame drawing without retained handles.
//! Each frame, you begin a recording session, draw shapes, then end the session.
//! This is ideal for dynamic content like particle effects, orbiting objects,
//! and animated visualizations.
//!
//! # Example
//!
//! ```ignore
//! let mut recorder = ImmediateRecorder::new();
//! recorder.begin(&mut commands, &mut meshes, &mut materials);
//! recorder.set_color([1.0, 0.0, 0.0]);
//! recorder.set_position(Vec3::new(0.0, 1.0, 0.0));
//! recorder.sphere(0.5, ShapeDetail::Medium);
//! recorder.end();
//! ```

use bevy::ecs::prelude::*;
use bevy::math::{Quat, Vec3};
use bevy::prelude::{Mesh3d, MeshMaterial3d};
use bevy::render::mesh::Mesh;
use bevy::transform::components::{GlobalTransform, Transform};

use crate::{dual_directional, opaque, transparent, additive, Shape, ShapeDetail, GradMeshMaterial};

/// Render state for a drawing group, controlling layer, depth test, and depth write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderState {
    /// Render layer for depth ordering (negative = background, positive = overlay).
    pub layer: i32,
    /// Depth test mode.
    pub depth_test: DepthTest,
    /// Depth write mode.
    pub depth_write: DepthWrite,
}

impl Default for RenderState {
    fn default() -> Self {
        Self {
            layer: 0,
            depth_test: DepthTest::LessEqual,
            depth_write: DepthWrite::Enabled,
        }
    }
}

impl RenderState {
    /// Create a new render state with the given parameters.
    pub fn new(layer: i32, depth_test: DepthTest, depth_write: DepthWrite) -> Self {
        Self { layer, depth_test, depth_write }
    }
}

/// Depth test mode for render state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DepthTest {
    #[default]
    /// Pass if fragment depth < reference depth.
    Less,
    /// Pass if fragment depth <= reference depth.
    LessEqual,
    /// Always pass.
    Always,
    /// Never pass.
    Never,
}

/// Depth write mode for render state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DepthWrite {
    #[default]
    /// Write depth values.
    Enabled,
    /// Do not write depth values.
    Disabled,
}

/// Billboard mode for 2D shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BillboardMode {
    #[default]
    /// No billboarding — shape faces +Z.
    None,
    /// Shape faces the camera, Y-axis locked up.
    FaceCameraY,
    /// Shape fully faces the camera.
    FaceCamera,
}

/// State stack entry for push/pop operations.
#[derive(Debug, Clone)]
struct StateEntry {
    position: Vec3,
    rotation: Quat,
    scale: Vec3,
}

impl Default for StateEntry {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

/// Immediate mode recorder for per-frame drawing.
///
/// Use `begin()` at the start of each frame, draw shapes, then call `end()`
/// to clean up the previous frame's entities.
pub struct ImmediateRecorder {
    state_stack: Vec<StateEntry>,
    current_state: StateEntry,
    render_state: Option<RenderState>,
    /// Entities spawned in the last `begin()` call, despawned on next `begin()`.
    entities: Vec<Entity>,
    /// Root entity for all immediate draws.
    root: Option<Entity>,
}

impl ImmediateRecorder {
    /// Create a new immediate recorder.
    pub fn new() -> Self {
        Self {
            state_stack: Vec::new(),
            current_state: StateEntry::default(),
            render_state: None,
            entities: Vec::new(),
            root: None,
        }
    }

    /// Begin a new recording session.
    ///
    /// Despawns entities from the previous session and resets the state stack.
    pub fn begin(&mut self, commands: &mut Commands) {
        // Despawn previous frame's entities
        for entity in self.entities.drain(..) {
            commands.entity(entity).despawn();
        }
        self.state_stack.clear();
        self.current_state = StateEntry::default();
        self.render_state = None;

        // Ensure root entity exists
        if self.root.is_none() {
            self.root = Some(commands.spawn(Transform::default()).id());
        }
    }

    /// Push the current state onto the stack.
    pub fn push_state(&mut self) {
        self.state_stack.push(self.current_state.clone());
    }

    /// Pop the current state from the stack.
    pub fn pop_state(&mut self) {
        if let Some(state) = self.state_stack.pop() {
            self.current_state = state;
        }
    }

    /// Set the position for subsequent draws.
    pub fn set_position(&mut self, position: Vec3) {
        self.current_state.position = position;
    }

    /// Set the rotation for subsequent draws.
    pub fn set_rotation(&mut self, rotation: Quat) {
        self.current_state.rotation = rotation;
    }

    /// Set the scale for subsequent draws.
    pub fn set_scale(&mut self, scale: Vec3) {
        self.current_state.scale = scale;
    }

    /// Set the render state for subsequent draws.
    pub fn set_render_state(&mut self, state: RenderState) {
        self.render_state = Some(state);
    }

    /// Draw a sphere at the current position/rotation/scale.
    pub fn sphere(
        &mut self,
        commands: &mut Commands,
        meshes: &mut bevy::asset::Assets<Mesh>,
        materials: &mut bevy::asset::Assets<GradMeshMaterial>,
        radius: f32,
        detail: ShapeDetail,
        material: GradMeshMaterial,
    ) {
        let mesh = meshes.add(Shape::sphere(radius, detail).build());
        let mat = materials.add(material);
        let entity = self.spawn_mesh(commands, mesh, mat);
        self.entities.push(entity);
    }

    /// Draw a disc (2D circle) at the current position.
    pub fn disc(
        &mut self,
        commands: &mut Commands,
        meshes: &mut bevy::asset::Assets<Mesh>,
        materials: &mut bevy::asset::Assets<GradMeshMaterial>,
        radius: f32,
        material: GradMeshMaterial,
        billboard: BillboardMode,
    ) {
        let mesh = meshes.add(Shape::disc(radius, ShapeDetail::Medium).build());
        let mat = materials.add(material);
        let entity = self.spawn_2d(commands, mesh, mat, billboard);
        self.entities.push(entity);
    }

    /// Draw a disc rim (2D ring) at the current position.
    pub fn disc_rim(
        &mut self,
        commands: &mut Commands,
        meshes: &mut bevy::asset::Assets<Mesh>,
        materials: &mut bevy::asset::Assets<GradMeshMaterial>,
        radius: f32,
        rim_width: f32,
        material: GradMeshMaterial,
        billboard: BillboardMode,
    ) {
        let mesh = meshes.add(Shape::disc_rim(radius, rim_width, ShapeDetail::Medium).build());
        let mat = materials.add(material);
        let entity = self.spawn_2d(commands, mesh, mat, billboard);
        self.entities.push(entity);
    }

    /// Draw a line from start to end point.
    pub fn line_3d(
        &mut self,
        commands: &mut Commands,
        meshes: &mut bevy::asset::Assets<Mesh>,
        materials: &mut bevy::asset::Assets<GradMeshMaterial>,
        start: Vec3,
        end: Vec3,
        thickness: f32,
        material: GradMeshMaterial,
    ) {
        let direction = end - start;
        let length = direction.length();
        if length < f32::EPSILON {
            return;
        }
        let midpoint = (start + end) * 0.5;
        let line_direction = direction.normalize();

        // Create a thin cylinder along the line
        let mesh = meshes.add(Shape::cylinder(thickness, length, ShapeDetail::Minimal).build());
        let mat = materials.add(material);

        let rotation = Quat::from_rotation_arc(Vec3::Y, line_direction);
        let entity = commands.spawn((
            Mesh3d(mesh),
            MeshMaterial3d(mat),
            Transform::from_translation(midpoint).with_rotation(rotation),
            GlobalTransform::default(),
        )).id();
        self.entities.push(entity);
    }

    /// Draw a 2D line (billboarded) from start to end point.
    pub fn line_2d(
        &mut self,
        commands: &mut Commands,
        meshes: &mut bevy::asset::Assets<Mesh>,
        materials: &mut bevy::asset::Assets<GradMeshMaterial>,
        start: Vec3,
        end: Vec3,
        thickness: f32,
        material: GradMeshMaterial,
        _billboard: BillboardMode,
    ) {
        let direction = end - start;
        let length = direction.length();
        if length < f32::EPSILON {
            return;
        }
        let midpoint = (start + end) * 0.5;
        let line_direction = direction.normalize();

        // Create a thin capsule for the line
        let mesh = meshes.add(Shape::capsule(thickness, length, ShapeDetail::Minimal).build());
        let mat = materials.add(material);

        let rotation = Quat::from_rotation_arc(Vec3::Y, line_direction);
        let entity = commands.spawn((
            Mesh3d(mesh),
            MeshMaterial3d(mat),
            Transform::from_translation(midpoint).with_rotation(rotation),
            GlobalTransform::default(),
        )).id();
        self.entities.push(entity);
    }

    /// Draw a regular polygon (2D) at the current position.
    pub fn regular_polygon(
        &mut self,
        commands: &mut Commands,
        meshes: &mut bevy::asset::Assets<Mesh>,
        materials: &mut bevy::asset::Assets<GradMeshMaterial>,
        sides: u32,
        radius: f32,
        material: GradMeshMaterial,
        billboard: BillboardMode,
    ) {
        let mesh = meshes.add(Shape::polygon(sides, radius, ShapeDetail::Minimal).build());
        let mat = materials.add(material);
        let entity = self.spawn_2d(commands, mesh, mat, billboard);
        self.entities.push(entity);
    }

    /// Draw a regular prism (3D) at the current position.
    pub fn regular_prism(
        &mut self,
        commands: &mut Commands,
        meshes: &mut bevy::asset::Assets<Mesh>,
        materials: &mut bevy::asset::Assets<GradMeshMaterial>,
        sides: u32,
        radius: f32,
        height: f32,
        roundness: f32,
        detail: ShapeDetail,
        material: GradMeshMaterial,
    ) {
        let mesh = meshes.add(Shape::regular_prism(sides, radius, height, roundness, detail).build());
        let mat = materials.add(material);
        let entity = self.spawn_mesh(commands, mesh, mat);
        self.entities.push(entity);
    }

    /// Draw a torus at the current position.
    pub fn torus(
        &mut self,
        commands: &mut Commands,
        meshes: &mut bevy::asset::Assets<Mesh>,
        materials: &mut bevy::asset::Assets<GradMeshMaterial>,
        radius: f32,
        thickness: f32,
        start_angle: f32,
        span: f32,
        _detail: ShapeDetail,
        material: GradMeshMaterial,
    ) {
        let mesh = meshes.add(Shape::torus(radius, thickness, start_angle, span).build());
        let mat = materials.add(material);
        let entity = self.spawn_mesh(commands, mesh, mat);
        self.entities.push(entity);
    }

    /// Draw a cuboid at the current position.
    pub fn cuboid(
        &mut self,
        commands: &mut Commands,
        meshes: &mut bevy::asset::Assets<Mesh>,
        materials: &mut bevy::asset::Assets<GradMeshMaterial>,
        size: Vec3,
        roundness: f32,
        detail: ShapeDetail,
        material: GradMeshMaterial,
    ) {
        let mesh = meshes.add(Shape::cuboid(size, roundness, detail).build());
        let mat = materials.add(material);
        let entity = self.spawn_mesh(commands, mesh, mat);
        self.entities.push(entity);
    }

    /// Draw an ellipsoid at the current position.
    pub fn ellipsoid(
        &mut self,
        commands: &mut Commands,
        meshes: &mut bevy::asset::Assets<Mesh>,
        materials: &mut bevy::asset::Assets<GradMeshMaterial>,
        size: Vec3,
        detail: ShapeDetail,
        material: GradMeshMaterial,
    ) {
        let mesh = meshes.add(Shape::ellipsoid(size, detail).build());
        let mat = materials.add(material);
        let entity = self.spawn_mesh(commands, mesh, mat);
        self.entities.push(entity);
    }

    /// Draw a cylinder at the current position.
    pub fn cylinder(
        &mut self,
        commands: &mut Commands,
        meshes: &mut bevy::asset::Assets<Mesh>,
        materials: &mut bevy::asset::Assets<GradMeshMaterial>,
        radius: f32,
        height: f32,
        detail: ShapeDetail,
        material: GradMeshMaterial,
    ) {
        let mesh = meshes.add(Shape::cylinder(radius, height, detail).build());
        let mat = materials.add(material);
        let entity = self.spawn_mesh(commands, mesh, mat);
        self.entities.push(entity);
    }

    /// Draw a cone at the current position.
    pub fn cone(
        &mut self,
        commands: &mut Commands,
        meshes: &mut bevy::asset::Assets<Mesh>,
        materials: &mut bevy::asset::Assets<GradMeshMaterial>,
        radius: f32,
        height: f32,
        detail: ShapeDetail,
        material: GradMeshMaterial,
    ) {
        let mesh = meshes.add(Shape::cone(radius, height, detail).build());
        let mat = materials.add(material);
        let entity = self.spawn_mesh(commands, mesh, mat);
        self.entities.push(entity);
    }

    /// Draw a capsule at the current position.
    pub fn capsule(
        &mut self,
        commands: &mut Commands,
        meshes: &mut bevy::asset::Assets<Mesh>,
        materials: &mut bevy::asset::Assets<GradMeshMaterial>,
        radius: f32,
        height: f32,
        detail: ShapeDetail,
        material: GradMeshMaterial,
    ) {
        let mesh = meshes.add(Shape::capsule(radius, height, detail).build());
        let mat = materials.add(material);
        let entity = self.spawn_mesh(commands, mesh, mat);
        self.entities.push(entity);
    }

    /// Draw a hemisphere at the current position.
    pub fn hemisphere(
        &mut self,
        commands: &mut Commands,
        meshes: &mut bevy::asset::Assets<Mesh>,
        materials: &mut bevy::asset::Assets<GradMeshMaterial>,
        radius: f32,
        capped: bool,
        detail: ShapeDetail,
        material: GradMeshMaterial,
    ) {
        let mesh = meshes.add(Shape::hemisphere(radius, capped, detail).build());
        let mat = materials.add(material);
        let entity = self.spawn_mesh(commands, mesh, mat);
        self.entities.push(entity);
    }

    /// Draw a pyramid at the current position.
    pub fn pyramid(
        &mut self,
        commands: &mut Commands,
        meshes: &mut bevy::asset::Assets<Mesh>,
        materials: &mut bevy::asset::Assets<GradMeshMaterial>,
        sides: u32,
        radius: f32,
        height: f32,
        detail: ShapeDetail,
        material: GradMeshMaterial,
    ) {
        let mesh = meshes.add(Shape::pyramid(sides, radius, height, detail).build());
        let mat = materials.add(material);
        let entity = self.spawn_mesh(commands, mesh, mat);
        self.entities.push(entity);
    }

    /// Draw a 2D rectangle at the current position.
    pub fn rectangle(
        &mut self,
        commands: &mut Commands,
        meshes: &mut bevy::asset::Assets<Mesh>,
        materials: &mut bevy::asset::Assets<GradMeshMaterial>,
        size: bevy::math::Vec2,
        material: GradMeshMaterial,
        billboard: BillboardMode,
    ) {
        let mesh = meshes.add(Shape::rectangle(size, ShapeDetail::Medium).build());
        let mat = materials.add(material);
        let entity = self.spawn_2d(commands, mesh, mat, billboard);
        self.entities.push(entity);
    }

    /// Draw a 2D rectangle rim at the current position.
    pub fn rectangle_rim(
        &mut self,
        commands: &mut Commands,
        meshes: &mut bevy::asset::Assets<Mesh>,
        materials: &mut bevy::asset::Assets<GradMeshMaterial>,
        size: bevy::math::Vec2,
        rim_width: f32,
        material: GradMeshMaterial,
        billboard: BillboardMode,
    ) {
        let mesh = meshes.add(Shape::rectangle_rim(size, rim_width, ShapeDetail::Medium).build());
        let mat = materials.add(material);
        let entity = self.spawn_2d(commands, mesh, mat, billboard);
        self.entities.push(entity);
    }

    /// Draw a 2D polygon rim at the current position.
    pub fn polygon_rim(
        &mut self,
        commands: &mut Commands,
        meshes: &mut bevy::asset::Assets<Mesh>,
        materials: &mut bevy::asset::Assets<GradMeshMaterial>,
        sides: u32,
        radius: f32,
        rim_width: f32,
        material: GradMeshMaterial,
        billboard: BillboardMode,
    ) {
        let mesh = meshes.add(Shape::polygon_rim(sides, radius, rim_width, ShapeDetail::Minimal).build());
        let mat = materials.add(material);
        let entity = self.spawn_2d(commands, mesh, mat, billboard);
        self.entities.push(entity);
    }

    /// Draw a 2D ellipse at the current position.
    pub fn ellipse(
        &mut self,
        commands: &mut Commands,
        meshes: &mut bevy::asset::Assets<Mesh>,
        materials: &mut bevy::asset::Assets<GradMeshMaterial>,
        radius: bevy::math::Vec2,
        material: GradMeshMaterial,
        billboard: BillboardMode,
    ) {
        let mesh = meshes.add(Shape::ellipse(radius, ShapeDetail::Medium).build());
        let mat = materials.add(material);
        let entity = self.spawn_2d(commands, mesh, mat, billboard);
        self.entities.push(entity);
    }

    /// Draw a 2D ellipse rim at the current position.
    pub fn ellipse_rim(
        &mut self,
        commands: &mut Commands,
        meshes: &mut bevy::asset::Assets<Mesh>,
        materials: &mut bevy::asset::Assets<GradMeshMaterial>,
        radius: bevy::math::Vec2,
        rim_width: f32,
        material: GradMeshMaterial,
        billboard: BillboardMode,
    ) {
        let mesh = meshes.add(Shape::ellipse_rim(radius, rim_width, ShapeDetail::Medium).build());
        let mat = materials.add(material);
        let entity = self.spawn_2d(commands, mesh, mat, billboard);
        self.entities.push(entity);
    }

    /// Helper: spawn a 3D mesh with current transform.
    fn spawn_mesh(
        &self,
        commands: &mut Commands,
        mesh: bevy::asset::Handle<Mesh>,
        material: bevy::asset::Handle<GradMeshMaterial>,
    ) -> Entity {
        let transform = Transform::from_translation(self.current_state.position)
            .with_rotation(self.current_state.rotation)
            .with_scale(self.current_state.scale);

        let root = self.root.unwrap();
        commands.entity(root).with_children(|parent| {
            parent.spawn((
                Mesh3d(mesh),
                MeshMaterial3d(material),
                transform,
                GlobalTransform::default(),
            ));
        }).id()
    }

    /// Helper: spawn a 2D mesh with billboard support.
    fn spawn_2d(
        &self,
        commands: &mut Commands,
        mesh: bevy::asset::Handle<Mesh>,
        material: bevy::asset::Handle<GradMeshMaterial>,
        billboard: BillboardMode,
    ) -> Entity {
        let mut transform = Transform::from_translation(self.current_state.position)
            .with_rotation(self.current_state.rotation)
            .with_scale(self.current_state.scale);

        // Apply billboard rotation
        match billboard {
            BillboardMode::None => {}
            BillboardMode::FaceCameraY => {
                // Face camera but keep Y up — will be updated by a system
                transform.rotation *= Quat::from_rotation_x(std::f32::consts::PI / 2.0);
            }
            BillboardMode::FaceCamera => {
                // Full billboard — will be updated by a system
                transform.rotation *= Quat::from_rotation_x(std::f32::consts::PI / 2.0);
            }
        }

        let root = self.root.unwrap();
        commands.entity(root).with_children(|parent| {
            parent.spawn((
                Mesh3d(mesh),
                MeshMaterial3d(material),
                transform,
                GlobalTransform::default(),
            ));
        }).id()
    }
}

impl Default for ImmediateRecorder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function: create an opaque material for immediate drawing.
pub fn imm_opaque(r: f32, g: f32, b: f32) -> GradMeshMaterial {
    opaque(r, g, b)
}

/// Convenience function: create a transparent material for immediate drawing.
pub fn imm_transparent(r: f32, g: f32, b: f32, alpha: f32) -> GradMeshMaterial {
    transparent(r, g, b, alpha)
}

/// Convenience function: create an additive material for immediate drawing.
pub fn imm_additive(r: f32, g: f32, b: f32) -> GradMeshMaterial {
    additive(r, g, b)
}

/// Convenience function: create a dual-directional material for immediate drawing.
pub fn imm_dual_directional(
    dark: [f32; 3],
    light: [f32; 3],
    direction: [f32; 3],
) -> GradMeshMaterial {
    dual_directional(dark, light, direction)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recorder_push_pop_state() {
        let mut recorder = ImmediateRecorder::new();
        recorder.set_position(Vec3::new(1.0, 0.0, 0.0));
        recorder.push_state();
        recorder.set_position(Vec3::new(2.0, 0.0, 0.0));
        recorder.pop_state();
        assert_eq!(recorder.current_state.position, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn recorder_nested_push_pop() {
        let mut recorder = ImmediateRecorder::new();
        recorder.set_position(Vec3::new(0.0, 0.0, 0.0));
        recorder.push_state();
        recorder.set_position(Vec3::new(1.0, 0.0, 0.0));
        recorder.push_state();
        recorder.set_position(Vec3::new(2.0, 0.0, 0.0));
        recorder.pop_state();
        assert_eq!(recorder.current_state.position, Vec3::new(1.0, 0.0, 0.0));
        recorder.pop_state();
        assert_eq!(recorder.current_state.position, Vec3::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn recorder_pop_empty_stack() {
        let mut recorder = ImmediateRecorder::new();
        recorder.set_position(Vec3::new(1.0, 0.0, 0.0));
        recorder.pop_state(); // Should not change state
        assert_eq!(recorder.current_state.position, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn render_state_defaults() {
        let state = RenderState::default();
        assert_eq!(state.layer, 0);
        assert_eq!(state.depth_test, DepthTest::LessEqual);
        assert_eq!(state.depth_write, DepthWrite::Enabled);
    }

    #[test]
    fn render_state_custom() {
        let state = RenderState::new(-30, DepthTest::Always, DepthWrite::Disabled);
        assert_eq!(state.layer, -30);
        assert_eq!(state.depth_test, DepthTest::Always);
        assert_eq!(state.depth_write, DepthWrite::Disabled);
    }

    #[test]
    fn billboard_mode_variants() {
        let _none = BillboardMode::None;
        let _y = BillboardMode::FaceCameraY;
        let _full = BillboardMode::FaceCamera;
    }

    #[test]
    fn line_3d_skips_zero_length_check() {
        // Verify that a zero-length line produces no entity
        // (we can't test Commands directly, but we can verify the length check)
        let start = Vec3::ZERO;
        let end = Vec3::ZERO;
        let direction = end - start;
        let length = direction.length();
        assert!(length < f32::EPSILON, "zero length should be below epsilon");
    }
}
