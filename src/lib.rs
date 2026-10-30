//! Procedural mesh generation and gradient materials for Bevy.
//!
//! Generates watertight meshes for common 3D primitives with configurable detail levels,
//! and provides a custom shader material for dual-directional gradient shading.
//!
//! # Quick Start
//!
//! ```ignore
//! use bevy::prelude::*;
//! use bevy_gradmesh::{GradMeshPlugin, Shape, ShapeDetail, dual_directional};
//!
//! fn main() {
//!     App::new()
//!         .add_plugins(DefaultPlugins)
//!         .add_plugins(GradMeshPlugin)
//!         .add_systems(Startup, |mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>,
//!                                  mut materials: ResMut<Assets<GradMeshMaterial>>| {
//!             // Camera
//!             commands.spawn(Camera3d::from_transform(Transform::from_xyz(0.0, 5.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y)));
//!             // Shape
//!             let mesh = meshes.add(Shape::sphere(1.0, ShapeDetail::Medium).build());
//!             let mat = materials.add(dual_directional([0.2, 0.2, 0.4], [0.8, 0.6, 1.0], [0.0, 1.0, 0.0]));
//!             commands.spawn((Mesh3d(mesh), MeshMaterial3d(mat)));
//!         })
//!         .run();
//! }
//! ```
//!
//! # Modes
//!
//! - **Direct**: Call `Shape::build()` and spawn `Mesh3d` entities manually.
//! - **Immediate**: Use `ImmediateRecorder` for per-frame draw/dispose patterns.
//! - **Retained**: Use `RetainedWorld` + `GradMeshHandle` for persistent shapes
//!   that you update per-frame.  Under the hood this uses batched instanced
//!   rendering so shapes with the same mesh share draw calls.
//! - **Batched**: Use `BatchRecorder` for manual per-frame batch accumulation.
//!   Call `render_batches()` to spawn entities from batches.

pub mod shape;
pub mod style;
pub mod material;
pub mod material_interning;
pub mod immediate;
pub mod batch;
pub mod plugin;
pub mod retained;
pub mod render_pipeline;
pub mod instanced_render;
pub mod standalone_pipeline;
pub mod culling;
pub mod polyline_gpu;
pub mod lod;
pub mod deformation;
pub mod appearance;

pub use shape::{Shape, ShapeDetail};
pub use style::{Style, BlendMode, ColorMode, DashType, Dash, ShapeType};
pub use material::{
    GradMeshMaterial, GradMeshExtension, GradMeshUniform, GradMeshUniformBuffer, Vec4, gradmesh_to_standard,
    opaque, transparent, additive, dual_directional, dual_directional_oklab, dual_radial, dual_radial_oklab,
    multiplicative, screen, dither, dashed,
};
pub use immediate::{
    ImmediateRecorder, RenderState, DepthTest, DepthWrite, BillboardMode,
    imm_opaque, imm_transparent, imm_additive, imm_dual_directional,
};
pub use batch::{
    BatchRecorder, Batch, BatchKey, InstanceTransform, InstanceProperties,
};
pub use plugin::{GradMeshPlugin, MeshCache};
pub use retained::{GradMeshHandle, RetainedWorld};
pub use render_pipeline::{
    InstancedBufferData, GradMeshBatchItem, InstancedBatch, InstancedStats,
};
pub use instanced_render::{InstancedBatchData, InstancedRenderPlugin};
pub use lod::{LodConfig, LodResult, LOD_MINIMAL, LOD_LOW, LOD_MEDIUM, LOD_HIGH, LOD_EXTREME};
pub use deformation::{DeformationType, DeformationParams, Deformation};
pub use appearance::{AppearanceSnapshot, AppearanceType};
pub use material_interning::{
    GradMeshMaterialKey, GradMeshMaterialPool, MATERIAL_WARNING_THRESHOLD, warn_excessive_materials,
};
pub use standalone_pipeline::{
    StandaloneRenderPlugin, StandalonePipeline, StandalonePipelineKey,
    StandaloneBatchData, render_standalone_batches,
};
