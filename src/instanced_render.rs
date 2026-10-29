//! GPU instanced rendering pipeline for BevyGradmesh.
//!
//! Replaces the entity-per-instance fallback with a custom render phase that
//! issues `draw_indexed` instanced calls via the `instanced.wesl` shader.
//!
//! # Architecture
//!
//! 1. **Batch Data Extraction** — Each frame, `InstancedBatch` components are
//!    extracted to the render world via `ExtractComponent`.
//! 2. **GPU Buffers** — Per-batch transform and property data are uploaded to
//!    GPU vertex buffers with `VertexStepMode::Instance`.
//! 3. **Render Phase** — Batches are queued as `Opaque3d` phase items and
//!    rendered with a single `draw_indexed` call per batch.
//! 4. **Dirty Tracking** — Only dirty batches re-upload GPU data each frame.

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

use bevy::prelude::Shader;

use crate::batch::{InstanceProperties, InstanceTransform};
use crate::culling::{self, AABB};
use crate::lod::LodConfig;
use crate::render_pipeline::InstancedStats;


// ─── Extracted Batch Data ───

/// Extracted batch data for the render world. Contains the mesh handle,
/// instance count, and per-instance transform and property data.
#[derive(Clone, Component, ExtractComponent)]
#[extract_app(RenderApp)]
pub struct InstancedBatchData {
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
    /// LOD level index (0=Minimal, 1=Low, 2=Medium, 3=High, 4=Extreme).
    pub lod_level: u8,
    /// LOD transition weight for dither cross-fade (0.0 = fully current, 1.0 = fully next).
    pub lod_transition_weight: f32,
    /// Optional custom shader handle for fragment rendering.
    pub custom_shader: Option<Handle<Shader>>,
}

/// GPU buffers for a single instanced batch.
#[derive(Component)]
pub struct InstancedGpuBuffers {
    /// Combined vertex buffer containing both transforms and properties.
    buffer: Buffer,
    /// Number of instances.
    instance_count: u32,
}

// ─── Pipeline ───

/// The specialized pipeline for instanced GradMesh rendering.
#[derive(Resource)]
pub struct InstancedPipeline {
    /// The base mesh pipeline for specialization.
    mesh_pipeline: MeshPipeline,
    /// Shader handle for the instanced WGSL shader.
    shader: Handle<Shader>,
}

impl InstancedPipeline {
    pub fn new(mesh_pipeline: MeshPipeline, shader: Handle<Shader>) -> Self {
        Self { mesh_pipeline, shader }
    }
}

/// Specialized pipelines cache for instanced rendering.
pub type InstancedPipelineCache = SpecializedMeshPipelines<InstancedPipeline>;

// ─── Draw Command ───

/// The draw command tuple for instanced batches.
pub type DrawInstancedBatch = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshBindGroup<1>,
    DrawInstancedBatchImpl,
);

/// Implements the actual draw call for instanced batches.
pub struct DrawInstancedBatchImpl;

impl<P: PhaseItem> RenderCommand<P> for DrawInstancedBatchImpl {
    type Param = (
        SRes<RenderAssets<RenderMesh>>,
        SRes<RenderMeshInstances>,
        SRes<MeshAllocator>,
    );
    type ViewQuery = ();
    type ItemQuery = (Read<InstancedGpuBuffers>, Read<InstancedBatchData>);

    #[inline]
    fn render<'w>(
        item: &P,
        _view: (),
        item_query: Option<(&'w InstancedGpuBuffers, &'w InstancedBatchData)>,
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

                // Issue the instanced draw call
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

/// Prepare GPU buffers for instanced batches. Uploads dirty batch data.
fn prepare_instanced_buffers(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<bevy::render::renderer::RenderQueue>,
    new_batches: Query<(Entity, &InstancedBatchData), Without<InstancedGpuBuffers>>,
    dirty_batches: Query<(&InstancedBatchData, &InstancedGpuBuffers)>,
) {
    // Create buffers for new batches
    for (entity, batch_data) in &new_batches {
        if batch_data.instance_count == 0 {
            continue;
        }

        let combined = build_combined_buffer(batch_data);
        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("instanced batch data buffer"),
            contents: bytemuck::cast_slice(&combined),
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        });

        commands.entity(entity).insert(InstancedGpuBuffers {
            buffer,
            instance_count: batch_data.instance_count,
        });
    }

    // Re-upload dirty batches
    for (batch_data, gpu_buffers) in &dirty_batches {
        if !batch_data.dirty || batch_data.instance_count == 0 {
            continue;
        }

        let combined = build_combined_buffer(batch_data);
        render_queue.write_buffer(&gpu_buffers.buffer, 0, bytemuck::cast_slice(&combined));
    }
}

/// Build the interleaved transform+property buffer for a batch.
fn build_combined_buffer(batch_data: &InstancedBatchData) -> Vec<u8> {
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

/// Queue instanced batches into the opaque render phase with frustum culling and LOD.
fn queue_instanced_batches(
    opaque_draw_functions: Res<DrawFunctions<Opaque3d>>,
    instanced_pipeline: Res<InstancedPipeline>,
    mut pipelines: ResMut<InstancedPipelineCache>,
    pipeline_cache: Res<PipelineCache>,
    meshes: Res<RenderAssets<RenderMesh>>,
    render_mesh_instances: Res<RenderMeshInstances>,
    mut instanced_batches: Query<(Entity, &MainEntity, &mut InstancedBatchData)>,
    mut opaque_render_phases: ResMut<ViewBinnedRenderPhases<Opaque3d>>,
    views: Query<&ExtractedView>,
    view_key_cache: Res<ViewKeyCache>,
    mut stats: ResMut<InstancedStats>,
    lod_config: Option<Res<LodConfig>>,
) {
    let draw_instanced = opaque_draw_functions.read().id::<DrawInstancedBatch>();

    // Reset stats for this frame
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

        // Extract frustum planes from the view-projection matrix for culling
        let clip_from_world = view.clip_from_world.unwrap_or(view.clip_from_view * bevy::math::Affine3A::from(view.world_from_view.affine()));
        let frustum = culling::FrustumPlanes::from_view_projection(&clip_from_world);

        // Camera world position for LOD distance computation
        let camera_world_pos = view.world_from_view.translation();

        for (render_entity, &main_entity, mut batch_data) in &mut instanced_batches {
            if batch_data.instance_count == 0 {
                continue;
            }

            // Look up the render mesh for this batch
            let Some(mesh_instance) = render_mesh_instances.render_mesh_queue_data(main_entity)
            else {
                continue;
            };

            let Some(mesh) = meshes.get(mesh_instance.mesh_asset_id()) else {
                continue;
            };

            // Compute LOD from camera distance if LodConfig is available
            if let Some(lod_cfg) = &lod_config {
                let lod_cfg = lod_cfg.as_ref();
                // Compute average instance position for distance estimation
                let avg_pos = if let Some(ref mesh_aabb) = batch_data.mesh_aabb {
                    let transforms = parse_transforms(&batch_data);
                    if !transforms.is_empty() {
                        let mat = transforms[0].to_mat4();
                        let center = mesh_aabb.center;
                        let p = mat.mul_vec4(center.extend(1.0));
                        bevy::math::Vec3::new(p.x / p.w, p.y / p.w, p.z / p.w)
                    } else {
                        camera_world_pos
                    }
                } else {
                    camera_world_pos
                };

                let distance = LodConfig::distance_from_camera(&camera_world_pos, &avg_pos);
                let (computed_lod, transition_weight) = lod_cfg.compute_lod(distance, Some(batch_data.lod_level));
                
                // Update LOD level and transition weight
                batch_data.lod_level = computed_lod;
                batch_data.lod_transition_weight = transition_weight;
            }

            // Specialize the pipeline for this mesh
            let key = view_key
                | MeshPipelineKey::from_primitive_topology_and_strip_index(
                    mesh.primitive_topology(),
                    mesh.index_format(),
                );

            let Ok(pipeline) = pipelines.specialize(
                &pipeline_cache,
                &instanced_pipeline,
                key,
                &mesh.layout,
            ) else {
                continue;
            };

            // Perform frustum culling if mesh bounds are available
            let (visible_count, culled_count) = if let Some(ref mesh_aabb) = batch_data.mesh_aabb {
                let transforms = parse_transforms(&batch_data);
                let visible = culling::cull_instances(&frustum, mesh_aabb, &transforms);
                let culled = batch_data.instance_count as usize - visible.len();
                (visible.len(), culled)
            } else {
                (batch_data.instance_count as usize, 0)
            };

            // Update stats
            stats.total_instances_before_cull += batch_data.instance_count as usize;
            stats.culled_instances += culled_count;
            stats.total_instances += visible_count;
            stats.batch_count += 1;
            // Track LOD distribution
            let lod_idx = batch_data.lod_level as usize;
            if lod_idx < 5 {
                stats.lod_distribution[lod_idx] += visible_count;
            }
            if batch_data.dirty {
                stats.dirty_batch_count += 1;
                stats.gpu_upload_bytes +=
                    batch_data.transforms.len() as u64 + batch_data.properties.len() as u64;
            }

            // Skip batches where all instances are culled
            if visible_count == 0 {
                continue;
            }

            // Add to the opaque render phase
            opaque_phase.add(
                Opaque3dBatchSetKey {
                    draw_function: draw_instanced,
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

/// Parse transform data from InstancedBatchData into culling InstanceTransforms.
fn parse_transforms(batch_data: &InstancedBatchData) -> Vec<culling::InstanceTransform> {
    let transform_stride = std::mem::size_of::<InstanceTransform>();
    let mut transforms = Vec::with_capacity(batch_data.instance_count as usize);

    for i in 0..batch_data.instance_count as usize {
        let start = i * transform_stride;
        if start + transform_stride > batch_data.transforms.len() {
            break;
        }

        // Read 4 Vec4 columns (16 f32 values) from the transform data
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

/// Plugin that wires the GPU instanced pipeline into Bevy's render graph.
pub struct InstancedRenderPlugin;

impl Plugin for InstancedRenderPlugin {
    fn build(&self, app: &mut App) {
        // Extract batch data from main world to render world
        app.add_plugins(ExtractComponentPlugin::<InstancedBatchData>::default());

        app.sub_app_mut(RenderApp)
            .add_render_command::<Opaque3d, DrawInstancedBatch>()
            .init_resource::<InstancedPipelineCache>()
            .init_resource::<InstancedStats>()
            .add_systems(RenderStartup, init_instanced_pipeline)
            .add_systems(
                Render,
                (
                    prepare_instanced_buffers.in_set(RenderSystems::PrepareResources),
                    queue_instanced_batches.in_set(RenderSystems::Queue),
                ),
            );
    }
}

/// Initialize the instanced pipeline resource.
fn init_instanced_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mesh_pipeline: Res<MeshPipeline>,
) {
    use bevy::asset::AssetPath;
    let path = AssetPath::from_path_buf(
        bevy::asset::embedded_path!("shader/gradmesh.wesl"),
    ).with_source("embedded");
    let shader = asset_server.load(path);
    commands.insert_resource(InstancedPipeline::new(mesh_pipeline.clone(), shader));
}

// ─── Pipeline Specialization ───

impl SpecializedMeshPipeline for InstancedPipeline {
    type Key = MeshPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut descriptor = self.mesh_pipeline.specialize(key, layout)?;

        // Use the instanced shader for both vertex and fragment
        descriptor.vertex.shader = self.shader.clone();
        descriptor.label = Some("instanced gradmesh pipeline".into());

        // Add the instanced vertex buffer layout
        // This buffer contains interleaved InstanceTransform + InstanceProperties per instance
        let transform_stride = size_of::<InstanceTransform>() as u64; // 64 bytes = 4×vec4
        let property_stride = size_of::<InstanceProperties>() as u64; // 112 bytes = 7×vec4 (includes deformation + appearance)

        descriptor.vertex.buffers.push(VertexBufferLayout {
            array_stride: transform_stride + property_stride,
            step_mode: VertexStepMode::Instance,
            attributes: [
                // InstanceTransform: 4×vec4 at locations 3-6
                (3u32, 0u64, VertexFormat::Float32x4),
                (4u32, VertexFormat::Float32x4.size(), VertexFormat::Float32x4),
                (5u32, VertexFormat::Float32x4.size() * 2, VertexFormat::Float32x4),
                (6u32, VertexFormat::Float32x4.size() * 3, VertexFormat::Float32x4),
                // InstanceProperties: 7×vec4 at locations 7-13 (color, secondary, direction, flags, dash, deformation, appearance)
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
            // Use custom shader if provided, otherwise use the built-in instanced shader
            fragment.shader = self.shader.clone();
        }

        Ok(descriptor)
    }
}
