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

pub mod appearance;
pub mod batch;
pub mod culling;
pub mod deformation;
mod geometry;
pub mod immediate;
#[cfg(feature = "pbr")]
pub mod instanced_render;
pub mod lod;
pub mod material;
pub mod material2d;
#[cfg(feature = "pbr")]
pub mod material_interning;
pub mod mix_space;
pub mod outline;
pub mod plugin;
pub mod polyline_gpu;
mod render_pipeline;
pub mod retained;
pub mod sdf2d;
pub mod shape;
pub mod spline;
mod standalone_pipeline;
pub mod style;

pub use appearance::{AppearanceSnapshot, AppearanceType};
pub use batch::{Batch, BatchKey, BatchRecorder, InstanceProperties, InstanceTransform};
pub use deformation::{Deformation, DeformationParams, DeformationType};
pub use immediate::{BillboardMode, DepthTest, DepthWrite, RenderState};
#[cfg(feature = "pbr")]
pub use immediate::{
    ImmediateRecorder, imm_additive, imm_dual_directional, imm_opaque, imm_transparent,
};
#[cfg(feature = "pbr")]
pub use instanced_render::{InstancedBatchData, InstancedRenderPlugin};
pub use lod::{LOD_EXTREME, LOD_HIGH, LOD_LOW, LOD_MEDIUM, LOD_MINIMAL, LodConfig, LodResult};
#[cfg(feature = "pbr")]
pub use material::{
    GradMeshExtension, GradMeshMaterial, additive, dashed, dither, dual_directional, dual_radial,
    dual_radial_alpha, gradmesh_to_standard, multiplicative, opaque, screen, transparent,
    with_gradient_bounds, with_mix_space,
};
pub use material::Vec4;
#[cfg(feature = "pbr")]
pub use material_interning::{
    GradMeshMaterialKey, GradMeshMaterialPool, MATERIAL_WARNING_THRESHOLD, MaterialWarning,
    warn_excessive_materials,
};
pub use material2d::{GradMesh2dPlugin, GradMeshMaterial2d};
pub use mix_space::MixSpace;
pub use outline::{arch, blob, rounded_rectangle};
pub use plugin::{GradMeshMobilePlugin, GradMeshPlugin, MeshCache};
pub use render_pipeline::InstancedStats;
pub use retained::{GradMeshHandle, RetainedWorld};
pub use sdf2d::{
    Dash2d, DashSnap, SdfGradient, SdfKind, SdfMaterial2d, SdfMaterial2dPlugin, unit_quad,
};
pub use shape::{
    HeightfieldBorder, HeightfieldOptions, HoleEdge, Shape, ShapeBounds, ShapeDetail, ShapePart,
    SweepFrame, SweepOpening, SweepOptions,
};
pub use spline::{BakedSpline, CatmullRomSpline, SplineError, SplineFrame};
pub use standalone_pipeline::{StandaloneRenderPlugin, render_standalone_batches};
pub use style::{
    BlendMode, ColorMode, Dash, DashType, DirectionSpace, Noise, Rim, ShapeType, Stripes, Style,
};
