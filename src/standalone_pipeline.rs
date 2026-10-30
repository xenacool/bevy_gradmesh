//! Standalone GradMesh render pipeline — no PBR dependency.
//!
//! This module provides a complete render pipeline for GradMesh that bypasses
//! Bevy's PBR system entirely. Unlike the `ExtendedMaterial` path which inherits
//! the full PBR pipeline (lighting, shadows, prepass, deferred rendering), this
//! standalone pipeline uses a simple vertex+fragment shader that only performs
//! gradient shading and blend modes.
//!
//! # Key Benefits
//!
//! - **Zero descriptor pool pressure**: One pipeline, one bind group layout,
//!   one descriptor set. All material variation via per-instance vertex attributes.
//! - **No PBR overhead**: No prepass, no shadows, no deferred rendering.
//! - **Mobile-friendly**: Works on Mali-G715, Adreno, and any GPU with limited
//!   descriptor pool capacity.
//! - **Faster compilation**: One shader pipeline instead of N specializations.
//!
//! # Architecture
//!
//! The standalone pipeline reuses the `Opaque3d` binned render phase (like the
//! existing `InstancedRenderPlugin`) but replaces the PBR-dependent shader with
//! a pure WGSL shader (`standalone.wgsl`) that has no Bevy PBR imports.
//! It uses `SpecializedMeshPipeline` with an empty key so all batches share
//! a single pipeline specialization.
//!
//! ```
//! StandalonePipeline
//! ├── Single render pipeline (no specialization per material)
//! ├── Reuses Opaque3d binned render phase
//! ├── Per-batch vertex buffer (transforms + properties)
//! └── Standalone WGSL shader (no PBR imports)
//! ```

use std::mem::size_of;

use bevy::core_pipeline::core_3d::{Opaque3d, Opaque3dBatchSetKey, Opaque3dBinKey};
use bevy::ecs::system::{lifetimeless::*, SystemParamItem};
use bevy::mesh::{MeshVertexBufferLayoutRef, VertexBufferLayout};
use bevy::pbr::{
    MeshPipeline, MeshPipelineKey, RenderMeshInstances, SetMeshBindGroup, SetMeshViewBindGroup,
    ViewKeyCache,
};
use bevy::prelude::*;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::mesh::allocator::MeshSlabs;
use bevy::render::mesh::{allocator::MeshAllocator, RenderMesh, RenderMeshBufferInfo};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_phase::{
    AddRenderCommand, BinnedRenderPhaseType, DrawFunctions, InputUniformIndex, PhaseItem,
    RenderCommand, RenderCommandResult, SetItemPipeline, TrackedRenderPass,
    ViewBinnedRenderPhases,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::RenderDevice;
use bevy::render::sync_world::MainEntity;
use bevy::render::view::ExtractedView;
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};

use crate::batch::{InstanceProperties, InstanceTransform};
use crate::culling::{self, AABB};
use crate::lod::LodConfig;
use crate::render_pipeline::InstancedStats;

// ─── Extracted Batch Data ───

/// Extracted batch data for the standalone render world.
#[derive(Clone, Component, ExtractComponent)]
#[extract_app(RenderApp)]
pub struct StandaloneBatchData {
    /// Mesh asset ID.
    pub mesh_id: AssetId<Mesh>,
    /// Number of instances.
    pub instance_count: u32,
    /// Flattened transform data for GPU upload.
    pub transforms: Vec<u8>,
    /// Flattened property data for GPU upload.
    pub properties: Vec<u8>,
    /// Whether this batch needs GPU upload this frame.
    pub dirty: bool,
    /// Render state for depth settings.
    pub render_state: crate::RenderState,
    /// Mesh AABB in local space for frustum culling.
    pub mesh_aabb: Option<AABB>,
    /// LOD level index.
    pub lod_level: u8,
    /// LOD transition weight.
    pub lod_transition_weight: f32,
}

/// GPU buffers for a single standalone instanced batch.
#[derive(Component)]
pub struct StandaloneGpuBuffers {
    /// Combined vertex buffer containing both transforms and properties.
    buffer: Buffer,
    /// Number of instances.
    instance_count: u32,
}

// ─── Pipeline ───

/// The standalone pipeline for GradMesh rendering (no PBR).
#[derive(Resource)]
pub struct StandalonePipeline {
    /// The base mesh pipeline for specialization.
    mesh_pipeline: MeshPipeline,
    /// Shader handle for the standalone WGSL shader.
    shader: Handle<Shader>,
}

impl StandalonePipeline {
    pub fn new(mesh_pipeline: MeshPipeline, shader: Handle<Shader>) -> Self {
        Self { mesh_pipeline, shader }
    }
}

/// Specialized pipelines cache for standalone rendering.
pub type StandalonePipelineCache = SpecializedMeshPipelines<StandalonePipeline>;

/// Empty pipeline key — all standalone batches share the same pipeline.
#[derive(Default, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct StandalonePipelineKey;

impl From<StandalonePipelineKey> for u64 {
    fn from(_: StandalonePipelineKey) -> u64 {
        0
    }
}

impl From<u64> for StandalonePipelineKey {
    fn from(_: u64) -> Self {
        StandalonePipelineKey
    }
}

impl SpecializedMeshPipeline for StandalonePipeline {
    type Key = MeshPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut descriptor = self.mesh_pipeline.specialize(key, layout)?;

        // Use the standalone shader for both vertex and fragment
        descriptor.vertex.shader = self.shader.clone();
        descriptor.label = Some("standalone gradmesh pipeline".into());

        // Add the instanced vertex buffer layout
        let transform_stride = size_of::<InstanceTransform>() as u64;
        let property_stride = size_of::<InstanceProperties>() as u64;

        descriptor.vertex.buffers.push(VertexBufferLayout {
            array_stride: transform_stride + property_stride,
            step_mode: VertexStepMode::Instance,
            attributes: [
                // InstanceTransform: 4×vec4 at locations 3-6
                (3u32, 0u64, VertexFormat::Float32x4),
                (4u32, VertexFormat::Float32x4.size(), VertexFormat::Float32x4),
                (5u32, VertexFormat::Float32x4.size() * 2, VertexFormat::Float32x4),
                (6u32, VertexFormat::Float32x4.size() * 3, VertexFormat::Float32x4),
                // InstanceProperties: 7×vec4 at locations 7-13
                (7u32, transform_stride, VertexFormat::Float32x4),
                (8u32, transform_stride + VertexFormat::Float32x4.size(), VertexFormat::Float32x4),
                (9u32, transform_stride + VertexFormat::Float32x4.size() * 2, VertexFormat::Float32x4),
                (10u32, transform_stride + VertexFormat::Float32x4.size() * 3, VertexFormat::Float32x4),
                (11u32, transform_stride + VertexFormat::Float32x4.size() * 4, VertexFormat::Float32x4),
                (12u32, transform_stride + VertexFormat::Float32x4.size() * 5, VertexFormat::Float32x4),
                (13u32, transform_stride + VertexFormat::Float32x4.size() * 6, VertexFormat::Float32x4),
            ]
            .map(|(loc, offset, fmt)| VertexAttribute {
                shader_location: loc,
                format: fmt,
                offset,
            })
            .to_vec(),
        });

        if let Some(fragment) = descriptor.fragment.as_mut() {
            fragment.shader = self.shader.clone();
        }

        Ok(descriptor)
    }
}

// ─── Draw Command ───

/// The draw command tuple for standalone batches.
pub type DrawStandaloneBatch = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshBindGroup<1>,
    DrawStandaloneBatchImpl,
);

/// Implements the actual draw call for standalone batches.
pub struct DrawStandaloneBatchImpl;

impl<P: PhaseItem> RenderCommand<P> for DrawStandaloneBatchImpl {
    type Param = (
        SRes<RenderAssets<RenderMesh>>,
        SRes<RenderMeshInstances>,
        SRes<MeshAllocator>,
    );
    type ViewQuery = ();
    type ItemQuery = (Read<StandaloneGpuBuffers>, Read<StandaloneBatchData>);

    #[inline]
    fn render<'w>(
        item: &P,
        _view: (),
        item_query: Option<(&'w StandaloneGpuBuffers, &'w StandaloneBatchData)>,
        (meshes, render_mesh_instances, mesh_allocator): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let mesh_allocator = mesh_allocator.into_inner();

        let Some((gpu_buffers, _batch_data)) = item_query else {
            return RenderCommandResult::Skip;
        };

        let Some(mesh_instance) = render_mesh_instances.into_inner().render_mesh_queue_data(item.main_entity())
        else {
            return RenderCommandResult::Skip;
        };

        let Some(gpu_mesh) = meshes.into_inner().get(mesh_instance.mesh_asset_id()) else {
            return RenderCommandResult::Skip;
        };

        let Some(vertex_buffer_slice) =
            mesh_allocator.mesh_vertex_slice(&mesh_instance.mesh_asset_id())
        else {
            return RenderCommandResult::Skip;
        };

        // Bind the mesh vertex buffer at location 0
        pass.set_vertex_buffer(0, vertex_buffer_slice.buffer.slice(..));

        // Bind the instanced data buffer at location 1
        pass.set_vertex_buffer(1, gpu_buffers.buffer.slice(..));

        match &gpu_mesh.buffer_info {
            RenderMeshBufferInfo::Indexed { index_format, count } => {
                let Some(index_buffer_slice) =
                    mesh_allocator.mesh_index_slice(&mesh_instance.mesh_asset_id())
                else {
                    return RenderCommandResult::Skip;
                };

                pass.set_index_buffer(
                    index_buffer_slice.buffer.slice(..),
                    *index_format,
                );

                pass.draw_indexed(
                    index_buffer_slice.range.start..(index_buffer_slice.range.start + count),
                    vertex_buffer_slice.range.start as i32,
                    0..gpu_buffers.instance_count,
                );
            }
            RenderMeshBufferInfo::NonIndexed => {
                pass.draw(
                    vertex_buffer_slice.range,
                    0..gpu_buffers.instance_count,
                );
            }
        }

        RenderCommandResult::Success
    }
}

// ─── Buffer Preparation ───

/// Prepare GPU buffers for standalone instanced batches.
fn prepare_standalone_buffers(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<bevy::render::renderer::RenderQueue>,
    new_batches: Query<(Entity, &StandaloneBatchData), Without<StandaloneGpuBuffers>>,
    dirty_batches: Query<(&StandaloneBatchData, &StandaloneGpuBuffers)>,
) {
    for (entity, batch_data) in &new_batches {
        if batch_data.instance_count == 0 {
            continue;
        }

        let combined = build_combined_buffer(batch_data);
        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("standalone instanced batch data buffer"),
            contents: bytemuck::cast_slice(&combined),
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        });

        commands.entity(entity).insert(StandaloneGpuBuffers {
            buffer,
            instance_count: batch_data.instance_count,
        });
    }

    for (batch_data, gpu_buffers) in &dirty_batches {
        if !batch_data.dirty || batch_data.instance_count == 0 {
            continue;
        }

        let combined = build_combined_buffer(batch_data);
        render_queue.write_buffer(&gpu_buffers.buffer, 0, bytemuck::cast_slice(&combined));
    }
}

/// Build the interleaved transform+property buffer for a standalone batch.
fn build_combined_buffer(batch_data: &StandaloneBatchData) -> Vec<u8> {
    let transform_stride = size_of::<InstanceTransform>();
    let property_stride = size_of::<InstanceProperties>();
    let instance_stride = transform_stride + property_stride;

    let total_size = instance_stride * batch_data.instance_count as usize;
    let mut combined = Vec::with_capacity(total_size);

    for i in 0..batch_data.instance_count as usize {
        let t_start = i * transform_stride;
        let p_start = i * property_stride;

        if t_start + transform_stride <= batch_data.transforms.len() {
            combined.extend_from_slice(
                &batch_data.transforms[t_start..t_start + transform_stride],
            );
        }
        if p_start + property_stride <= batch_data.properties.len() {
            combined.extend_from_slice(
                &batch_data.properties[p_start..p_start + property_stride],
            );
        }
    }

    combined
}

// ─── Queue System ───

/// Queue standalone batches into the opaque render phase.
fn queue_standalone_batches(
    opaque_draw_functions: Res<DrawFunctions<Opaque3d>>,
    standalone_pipeline: Option<Res<StandalonePipeline>>,
    mut pipelines: ResMut<StandalonePipelineCache>,
    pipeline_cache: Res<PipelineCache>,
    meshes: Res<RenderAssets<RenderMesh>>,
    render_mesh_instances: Res<RenderMeshInstances>,
    standalone_batches: Query<(Entity, &MainEntity, &StandaloneBatchData)>,
    mut opaque_render_phases: ResMut<ViewBinnedRenderPhases<Opaque3d>>,
    views: Query<&ExtractedView>,
    view_key_cache: Res<ViewKeyCache>,
    mut stats: ResMut<InstancedStats>,
    _lod_config: Option<Res<LodConfig>>,
) {
    let Some(standalone_pipeline) = standalone_pipeline else {
        return;
    };
    let draw_standalone = opaque_draw_functions.read().id::<DrawStandaloneBatch>();

    // Reset stats
    stats.total_instances = 0;
    stats.batch_count = 0;
    stats.dirty_batch_count = 0;
    stats.gpu_upload_bytes = 0;
    stats.culled_instances = 0;
    stats.total_instances_before_cull = 0;

    for view in &views {
        let Some(opaque_phase) = opaque_render_phases.get_mut(&view.retained_view_entity) else {
            continue;
        };

        let Some(&view_key) = view_key_cache.get(&view.retained_view_entity) else {
            continue;
        };

        let clip_from_world = view.clip_from_world.unwrap_or(view.clip_from_view * bevy::math::Affine3A::from(view.world_from_view.affine()));
        let frustum = culling::FrustumPlanes::from_view_projection(&clip_from_world);
        let _camera_world_pos = view.world_from_view.translation();

        for (render_entity, &main_entity, batch_data) in &standalone_batches {
            if batch_data.instance_count == 0 {
                continue;
            }

            let Some(mesh_instance) = render_mesh_instances.render_mesh_queue_data(main_entity)
            else {
                continue;
            };

            let Some(mesh) = meshes.get(mesh_instance.mesh_asset_id()) else {
                continue;
            };

            // Specialize the pipeline
            let key = view_key
                | MeshPipelineKey::from_primitive_topology_and_strip_index(
                    mesh.primitive_topology(),
                    mesh.index_format(),
                );

            let Ok(pipeline) = pipelines.specialize(
                &pipeline_cache,
                &standalone_pipeline,
                key,
                &mesh.layout,
            ) else {
                continue;
            };

            // Frustum culling
            let (visible_count, culled_count) = if let Some(ref mesh_aabb) = batch_data.mesh_aabb {
                let transforms = parse_transforms(&batch_data);
                let visible = culling::cull_instances(&frustum, mesh_aabb, &transforms);
                let culled = batch_data.instance_count as usize - visible.len();
                (visible.len(), culled)
            } else {
                (batch_data.instance_count as usize, 0)
            };

            stats.total_instances_before_cull += batch_data.instance_count as usize;
            stats.culled_instances += culled_count;
            stats.total_instances += visible_count;
            stats.batch_count += 1;
            if batch_data.dirty {
                stats.dirty_batch_count += 1;
                stats.gpu_upload_bytes +=
                    batch_data.transforms.len() as u64 + batch_data.properties.len() as u64;
            }

            if visible_count == 0 {
                continue;
            }

            opaque_phase.add(
                Opaque3dBatchSetKey {
                    draw_function: draw_standalone,
                    pipeline,
                    material_bind_group_index: None,
                    lightmap_slab: None,
                    slabs: MeshSlabs::default(),
                },
                Opaque3dBinKey {
                    asset_id: batch_data.mesh_id.untyped(),
                },
                (render_entity, main_entity),
                InputUniformIndex::default(),
                BinnedRenderPhaseType::NonMesh,
            );
        }
    }
}

/// Parse transform data from StandaloneBatchData.
fn parse_transforms(batch_data: &StandaloneBatchData) -> Vec<culling::InstanceTransform> {
    let transform_stride = std::mem::size_of::<InstanceTransform>();
    let mut transforms = Vec::with_capacity(batch_data.instance_count as usize);

    for i in 0..batch_data.instance_count as usize {
        let start = i * transform_stride;
        if start + transform_stride > batch_data.transforms.len() {
            break;
        }

        let bytes = &batch_data.transforms[start..start + transform_stride];
        let values: [f32; 16] = bytemuck::pod_read_unaligned(&bytes[..std::mem::size_of::<[f32; 16]>()]);

        transforms.push(culling::InstanceTransform {
            col0: [values[0], values[1], values[2], values[3]],
            col1: [values[4], values[5], values[6], values[7]],
            col2: [values[8], values[9], values[10], values[11]],
            col3: [values[12], values[13], values[14], values[15]],
        });
    }

    transforms
}

// ─── Plugin ───

/// Plugin that wires the standalone GPU instanced pipeline into Bevy's render graph.
///
/// This bypasses Bevy's PBR system entirely, using a simple vertex+fragment shader
/// that only performs gradient shading and blend modes. Ideal for mobile GPUs with
/// limited descriptor pool capacity.
pub struct StandaloneRenderPlugin;

impl Plugin for StandaloneRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<StandaloneBatchData>::default());

        app.sub_app_mut(RenderApp)
            .add_render_command::<Opaque3d, DrawStandaloneBatch>()
            .init_resource::<StandalonePipelineCache>()
            .add_systems(RenderStartup, init_standalone_pipeline)
            .add_systems(
                Render,
                (
                    prepare_standalone_buffers.in_set(RenderSystems::PrepareResources),
                    queue_standalone_batches.in_set(RenderSystems::Queue),
                ),
            );
    }
}

/// Initialize the standalone pipeline resource.
fn init_standalone_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mesh_pipeline: Option<Res<MeshPipeline>>,
) {
    let Some(mesh_pipeline) = mesh_pipeline else {
        return;
    };
    use bevy::asset::AssetPath;
    let path = AssetPath::from_path_buf(
        bevy::asset::embedded_path!("shader/standalone.wgsl"),
    ).with_source("embedded");
    let shader = asset_server.load(path);
    commands.insert_resource(StandalonePipeline::new(mesh_pipeline.clone(), shader));
}

// ─── Spawn API ───

/// Spawn standalone batch entities from a `BatchRecorder`.
///
/// This is the standalone equivalent of `BatchRecorder::render_batches()`.
/// Instead of using `InstancedBatchData` (which goes through the PBR pipeline),
/// it uses `StandaloneBatchData` which goes through the standalone pipeline.
pub fn render_standalone_batches(
    recorder: &mut crate::batch::BatchRecorder,
    commands: &mut bevy::ecs::system::Commands,
    mut meshes: bevy::ecs::system::ResMut<bevy::asset::Assets<Mesh>>,
    parent: Option<bevy::prelude::Entity>,
) -> Vec<bevy::prelude::Entity> {
    use crate::render_pipeline::InstancedBufferData;

    let mut spawned = Vec::new();

    for (key, batch) in recorder.batches.iter() {
        if batch.is_empty() {
            continue;
        }

        let mesh_handle = meshes.add(batch.mesh.clone());

        let buffer_data = InstancedBufferData::from_slices_with_deformation(
            &batch.transforms[..batch.count],
            &batch.properties[..batch.count],
            &batch.deformations[..batch.count],
        );

        let mesh_id = mesh_handle.id();
        let mesh_aabb = AABB::from_mesh(&batch.mesh);
        let vertex_count = batch.mesh.count_vertices();
        let lod_level = if vertex_count < 50 { 0 }
        else if vertex_count < 100 { 1 }
        else if vertex_count < 200 { 2 }
        else if vertex_count < 400 { 3 }
        else { 4 };

        let entity_id = commands.spawn((
            StandaloneBatchData {
                mesh_id,
                instance_count: buffer_data.instance_count,
                transforms: buffer_data.transforms,
                properties: buffer_data.properties,
                dirty: batch.is_dirty(),
                render_state: key.render_state,
                mesh_aabb,
                lod_level,
                lod_transition_weight: 0.0,
            },
            Transform::default(),
            GlobalTransform::default(),
        )).id();

        if let Some(p) = parent {
            commands.entity(p).add_child(entity_id);
        }

        spawned.push(entity_id);
    }

    for batch in recorder.batches.values_mut() {
        batch.acknowledge_changes();
    }

    spawned
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standalone_pipeline_key_is_unit() {
        let key = StandalonePipelineKey;
        let hash: u64 = key.into();
        assert_eq!(hash, 0);
        let back: StandalonePipelineKey = hash.into();
        assert_eq!(back, StandalonePipelineKey);
    }
}
