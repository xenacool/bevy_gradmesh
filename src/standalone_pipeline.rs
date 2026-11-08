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
//! a pure shader (`standalone.wesl`) that only imports `bevy_render::view::View`.
//! The pipeline is specialized only on view state (MSAA, target format) and mesh
//! topology, never on material, so all batches share a single specialization.
//!
//! This module depends on `bevy_render` and `bevy_core_pipeline` only — nothing
//! from `bevy_pbr`. The camera is bound through its own `ViewUniform` bind group
//! and meshes are bound straight from the `MeshAllocator`.
//!
//! ```text
//! StandalonePipeline
//! ├── One bind group: the camera `ViewUniform` (dynamic offset)
//! ├── Reuses Opaque3d binned render phase
//! ├── Per-batch vertex buffer (transforms + properties)
//! └── Standalone shader (no PBR imports)
//! ```

use std::mem::size_of;

use bevy::core_pipeline::core_3d::CORE_3D_DEPTH_FORMAT;
use bevy::core_pipeline::core_3d::{Opaque3d, Opaque3dBatchSetKey, Opaque3dBinKey};
use bevy::ecs::system::{SystemParamItem, lifetimeless::*};
use bevy::mesh::{MeshVertexBufferLayoutRef, VertexBufferLayout};
use bevy::prelude::*;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::mesh::allocator::MeshSlabs;
use bevy::render::mesh::{RenderMesh, RenderMeshBufferInfo, allocator::MeshAllocator};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_phase::{
    AddRenderCommand, BinnedRenderPhaseType, DrawFunctions, InputUniformIndex, PhaseItem,
    RenderCommand, RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewBinnedRenderPhases,
};
use bevy::render::render_resource::binding_types::uniform_buffer;
use bevy::render::render_resource::*;
use bevy::render::renderer::RenderDevice;
use bevy::render::sync_world::MainEntity;
use bevy::render::view::{ExtractedView, Msaa, ViewUniform, ViewUniformOffset, ViewUniforms};
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
    /// Mesh AABB in local space for frustum culling.
    pub mesh_aabb: Option<AABB>,
    /// LOD level index.
    pub lod_level: u8,
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
#[derive(Resource, Clone)]
pub struct StandalonePipeline {
    /// Layout of the camera bind group (group 0).
    view_layout: BindGroupLayoutDescriptor,
    /// Shader handle for the standalone shader.
    shader: Handle<Shader>,
}

impl StandalonePipeline {
    pub fn new(shader: Handle<Shader>) -> Self {
        let view_layout = BindGroupLayoutDescriptor::new(
            "standalone gradmesh view layout",
            &BindGroupLayoutEntries::single(
                ShaderStages::VERTEX_FRAGMENT,
                uniform_buffer::<ViewUniform>(true),
            ),
        );
        Self {
            view_layout,
            shader,
        }
    }
}

/// Camera bind group shared by every standalone batch.
#[derive(Resource)]
pub struct StandaloneViewBindGroup(BindGroup);

/// Specialized pipelines cache for standalone rendering.
pub type StandalonePipelineCache = SpecializedMeshPipelines<StandalonePipeline>;

/// Pipeline key — depends only on view state and mesh topology, never on
/// material, so all standalone batches on a view share the same pipeline.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct StandalonePipelineKey {
    /// MSAA sample count of the view.
    pub msaa_samples: u32,
    /// Colour target format of the view.
    pub target_format: TextureFormat,
    /// Primitive topology of the mesh.
    pub topology: PrimitiveTopology,
    /// Index format, required by wgpu for strip topologies.
    pub strip_index_format: Option<IndexFormat>,
}

impl StandalonePipelineKey {
    /// Build a key from the view's MSAA and target format plus a mesh's topology.
    pub fn new(
        msaa_samples: u32,
        target_format: TextureFormat,
        topology: PrimitiveTopology,
        index_format: Option<IndexFormat>,
    ) -> Self {
        Self {
            msaa_samples,
            target_format,
            topology,
            strip_index_format: if topology.is_strip() {
                index_format
            } else {
                None
            },
        }
    }
}

impl SpecializedMeshPipeline for StandalonePipeline {
    type Key = StandalonePipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        // Per-vertex mesh buffer: position, normal, uv at locations 0-2.
        let mesh_buffer_layout = layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
            Mesh::ATTRIBUTE_UV_0.at_shader_location(2),
        ])?;

        let mut descriptor = RenderPipelineDescriptor {
            label: Some("standalone gradmesh pipeline".into()),
            layout: vec![self.view_layout.clone()],
            vertex: VertexState {
                shader: self.shader.clone(),
                entry_point: Some("vs_standalone".into()),
                buffers: vec![mesh_buffer_layout],
                ..default()
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                entry_point: Some("fs_standalone".into()),
                targets: vec![Some(ColorTargetState {
                    format: key.target_format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            primitive: PrimitiveState {
                topology: key.topology,
                strip_index_format: key.strip_index_format,
                front_face: FrontFace::Ccw,
                cull_mode: None,
                ..default()
            },
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::Greater),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState {
                count: key.msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            ..default()
        };

        // Add the instanced vertex buffer layout
        let transform_stride = size_of::<InstanceTransform>() as u64;
        let property_stride = size_of::<InstanceProperties>() as u64;

        descriptor.vertex.buffers.push(VertexBufferLayout {
            array_stride: transform_stride + property_stride,
            step_mode: VertexStepMode::Instance,
            attributes: [
                // InstanceTransform: 4×vec4 at locations 3-6
                (3u32, 0u64, VertexFormat::Float32x4),
                (
                    4u32,
                    VertexFormat::Float32x4.size(),
                    VertexFormat::Float32x4,
                ),
                (
                    5u32,
                    VertexFormat::Float32x4.size() * 2,
                    VertexFormat::Float32x4,
                ),
                (
                    6u32,
                    VertexFormat::Float32x4.size() * 3,
                    VertexFormat::Float32x4,
                ),
                // InstanceProperties: 7×vec4 at locations 7-13
                (7u32, transform_stride, VertexFormat::Float32x4),
                (
                    8u32,
                    transform_stride + VertexFormat::Float32x4.size(),
                    VertexFormat::Float32x4,
                ),
                (
                    9u32,
                    transform_stride + VertexFormat::Float32x4.size() * 2,
                    VertexFormat::Float32x4,
                ),
                (
                    10u32,
                    transform_stride + VertexFormat::Float32x4.size() * 3,
                    VertexFormat::Float32x4,
                ),
                (
                    11u32,
                    transform_stride + VertexFormat::Float32x4.size() * 4,
                    VertexFormat::Float32x4,
                ),
                (
                    12u32,
                    transform_stride + VertexFormat::Float32x4.size() * 5,
                    VertexFormat::Float32x4,
                ),
                (
                    13u32,
                    transform_stride + VertexFormat::Float32x4.size() * 6,
                    VertexFormat::Float32x4,
                ),
            ]
            .map(|(loc, offset, fmt)| VertexAttribute {
                shader_location: loc,
                format: fmt,
                offset,
            })
            .to_vec(),
        });

        Ok(descriptor)
    }
}

// ─── Draw Command ───

/// The draw command tuple for standalone batches.
pub type DrawStandaloneBatch = (
    SetItemPipeline,
    SetStandaloneViewBindGroup<0>,
    DrawStandaloneBatchImpl,
);

/// Binds the camera `ViewUniform` bind group at group `I`.
pub struct SetStandaloneViewBindGroup<const I: usize>;

impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetStandaloneViewBindGroup<I> {
    type Param = SRes<StandaloneViewBindGroup>;
    type ViewQuery = Read<ViewUniformOffset>;
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        _item: &P,
        view_uniform_offset: &'w ViewUniformOffset,
        _entity: Option<()>,
        view_bind_group: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        pass.set_bind_group(
            I,
            &view_bind_group.into_inner().0,
            &[view_uniform_offset.offset],
        );
        RenderCommandResult::Success
    }
}

/// Implements the actual draw call for standalone batches.
pub struct DrawStandaloneBatchImpl;

impl<P: PhaseItem> RenderCommand<P> for DrawStandaloneBatchImpl {
    type Param = (SRes<RenderAssets<RenderMesh>>, SRes<MeshAllocator>);
    type ViewQuery = ();
    type ItemQuery = (Read<StandaloneGpuBuffers>, Read<StandaloneBatchData>);

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        item_query: Option<(&'w StandaloneGpuBuffers, &'w StandaloneBatchData)>,
        (meshes, mesh_allocator): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let mesh_allocator = mesh_allocator.into_inner();

        let Some((gpu_buffers, batch_data)) = item_query else {
            return RenderCommandResult::Skip;
        };
        let mesh_asset_id = batch_data.mesh_id;

        let Some(gpu_mesh) = meshes.into_inner().get(mesh_asset_id) else {
            return RenderCommandResult::Skip;
        };

        let Some(vertex_buffer_slice) = mesh_allocator.mesh_vertex_slice(&mesh_asset_id) else {
            return RenderCommandResult::Skip;
        };

        // Bind the mesh vertex buffer at location 0
        pass.set_vertex_buffer(0, vertex_buffer_slice.buffer.slice(..));

        // Bind the instanced data buffer at location 1
        pass.set_vertex_buffer(1, gpu_buffers.buffer.slice(..));

        match &gpu_mesh.buffer_info {
            RenderMeshBufferInfo::Indexed {
                index_format,
                count,
            } => {
                let Some(index_buffer_slice) = mesh_allocator.mesh_index_slice(&mesh_asset_id)
                else {
                    return RenderCommandResult::Skip;
                };

                pass.set_index_buffer(index_buffer_slice.buffer.slice(..), *index_format);

                pass.draw_indexed(
                    index_buffer_slice.range.start..(index_buffer_slice.range.start + count),
                    vertex_buffer_slice.range.start as i32,
                    0..gpu_buffers.instance_count,
                );
            }
            RenderMeshBufferInfo::NonIndexed => {
                pass.draw(vertex_buffer_slice.range, 0..gpu_buffers.instance_count);
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
    mut dirty_batches: Query<(&StandaloneBatchData, &mut StandaloneGpuBuffers)>,
    mut stats: ResMut<InstancedStats>,
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

        stats.buffer_allocations += 1;
        stats.dirty_batch_count += 1;
        stats.gpu_upload_bytes += combined.len() as u64;
        commands.entity(entity).insert(StandaloneGpuBuffers {
            buffer,
            instance_count: batch_data.instance_count,
        });
    }

    for (batch_data, mut gpu_buffers) in &mut dirty_batches {
        if !batch_data.dirty || batch_data.instance_count == 0 {
            continue;
        }

        let combined = build_combined_buffer(batch_data);
        // A batch that grew would write past the end of its buffer, so it gets
        // a new one; the draw call always uses the batch's current count.
        if combined.len() as u64 > gpu_buffers.buffer.size() {
            gpu_buffers.buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("standalone instanced batch data buffer"),
                contents: bytemuck::cast_slice(&combined),
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            });
            stats.buffer_reallocations += 1;
        } else {
            render_queue.write_buffer(&gpu_buffers.buffer, 0, bytemuck::cast_slice(&combined));
        }
        gpu_buffers.instance_count = batch_data.instance_count;
        stats.dirty_batch_count += 1;
        stats.gpu_upload_bytes += combined.len() as u64;
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
            combined.extend_from_slice(&batch_data.transforms[t_start..t_start + transform_stride]);
        }
        if p_start + property_stride <= batch_data.properties.len() {
            combined.extend_from_slice(&batch_data.properties[p_start..p_start + property_stride]);
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
    standalone_batches: Query<(Entity, &MainEntity, &StandaloneBatchData)>,
    mut opaque_render_phases: ResMut<ViewBinnedRenderPhases<Opaque3d>>,
    views: Query<(&ExtractedView, &Msaa)>,
    mut stats: ResMut<InstancedStats>,
    _lod_config: Option<Res<LodConfig>>,
) {
    let Some(standalone_pipeline) = standalone_pipeline else {
        return;
    };
    let draw_standalone = opaque_draw_functions.read().id::<DrawStandaloneBatch>();

    for (view, msaa) in &views {
        let Some(opaque_phase) = opaque_render_phases.get_mut(&view.retained_view_entity) else {
            continue;
        };
        stats.view_count += 1;

        let clip_from_world = view.clip_from_world.unwrap_or_else(|| {
            view.clip_from_view * bevy::math::Mat4::from(view.world_from_view.affine().inverse())
        });
        let frustum = culling::FrustumPlanes::from_view_projection(&clip_from_world);
        let _camera_world_pos = view.world_from_view.translation();

        for (render_entity, &main_entity, batch_data) in &standalone_batches {
            if batch_data.instance_count == 0 {
                continue;
            }

            let Some(mesh) = meshes.get(batch_data.mesh_id) else {
                continue;
            };

            // Specialize the pipeline on view state and mesh topology only.
            let key = StandalonePipelineKey::new(
                msaa.samples(),
                view.target_format,
                mesh.primitive_topology(),
                mesh.index_format(),
            );

            let Ok(pipeline) =
                pipelines.specialize(&pipeline_cache, &standalone_pipeline, key, &mesh.layout)
            else {
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
            // No distance-based LOD here: record the level fixed at spawn time.
            if let Some(slot) = stats.lod_distribution.get_mut(batch_data.lod_level as usize) {
                *slot += visible_count;
            }

            if visible_count == 0 {
                stats.skipped_batches += 1;
                continue;
            }
            stats.draw_calls += 1;
            stats.visible_triangles +=
                visible_count as u64 * crate::render_pipeline::triangles_per_instance(mesh);

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
        let values: [f32; 16] =
            bytemuck::pod_read_unaligned(&bytes[..std::mem::size_of::<[f32; 16]>()]);

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
        bevy::asset::embedded_asset!(app, "shader/standalone.wesl");

        app.add_plugins(ExtractComponentPlugin::<StandaloneBatchData>::default());
        if !app.is_plugin_added::<crate::render_pipeline::InstancedStatsPlugin>() {
            app.add_plugins(crate::render_pipeline::InstancedStatsPlugin);
        }

        app.sub_app_mut(RenderApp)
            .add_render_command::<Opaque3d, DrawStandaloneBatch>()
            .init_resource::<StandalonePipelineCache>()
            // Also initialised by `InstancedRenderPlugin`; `init_resource` is a no-op
            // when it already exists, so the standalone plugin works on its own.
            .init_resource::<InstancedStats>()
            .add_systems(RenderStartup, init_standalone_pipeline)
            .add_systems(
                Render,
                (
                    prepare_standalone_buffers.in_set(RenderSystems::PrepareResources),
                    prepare_standalone_view_bind_group.in_set(RenderSystems::PrepareBindGroups),
                    queue_standalone_batches.in_set(RenderSystems::Queue),
                ),
            );
    }
}

/// Initialize the standalone pipeline resource.
fn init_standalone_pipeline(mut commands: Commands, asset_server: Res<AssetServer>) {
    use bevy::asset::AssetPath;
    let path = AssetPath::from_path_buf(bevy::asset::embedded_path!("shader/standalone.wesl"))
        .with_source("embedded");
    let shader = asset_server.load(path);
    commands.insert_resource(StandalonePipeline::new(shader));
}

/// Build the camera bind group from the shared `ViewUniforms` buffer.
fn prepare_standalone_view_bind_group(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    view_uniforms: Res<ViewUniforms>,
    standalone_pipeline: Option<Res<StandalonePipeline>>,
) {
    let Some(standalone_pipeline) = standalone_pipeline else {
        return;
    };
    let Some(view_binding) = view_uniforms.uniforms.binding() else {
        return;
    };
    let bind_group = render_device.create_bind_group(
        "standalone gradmesh view bind group",
        &pipeline_cache.get_bind_group_layout(&standalone_pipeline.view_layout),
        &BindGroupEntries::single(view_binding),
    );
    commands.insert_resource(StandaloneViewBindGroup(bind_group));
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

    for batch in recorder.batches.values() {
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
        let lod_level = if vertex_count < 50 {
            0
        } else if vertex_count < 100 {
            1
        } else if vertex_count < 200 {
            2
        } else if vertex_count < 400 {
            3
        } else {
            4
        };

        let entity_id = commands
            .spawn((
                StandaloneBatchData {
                    mesh_id,
                    instance_count: buffer_data.instance_count,
                    transforms: buffer_data.transforms,
                    properties: buffer_data.properties,
                    dirty: batch.is_dirty(),
                    mesh_aabb,
                    lod_level,
                },
                Transform::default(),
                GlobalTransform::default(),
            ))
            .id();

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
    fn standalone_pipeline_key_depends_only_on_view_and_topology() {
        let a = StandalonePipelineKey::new(
            4,
            TextureFormat::Rgba8UnormSrgb,
            PrimitiveTopology::TriangleList,
            Some(IndexFormat::Uint32),
        );
        let b = StandalonePipelineKey::new(
            4,
            TextureFormat::Rgba8UnormSrgb,
            PrimitiveTopology::TriangleList,
            Some(IndexFormat::Uint16),
        );
        // Index format only matters for strip topologies.
        assert_eq!(a, b);

        let strip_a = StandalonePipelineKey::new(
            1,
            TextureFormat::Rgba8UnormSrgb,
            PrimitiveTopology::TriangleStrip,
            Some(IndexFormat::Uint32),
        );
        let strip_b = StandalonePipelineKey::new(
            1,
            TextureFormat::Rgba8UnormSrgb,
            PrimitiveTopology::TriangleStrip,
            Some(IndexFormat::Uint16),
        );
        assert_ne!(strip_a, strip_b);
        assert_ne!(a, strip_a);
    }
}
