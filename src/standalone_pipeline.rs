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

use bevy::core_pipeline::core_3d::{CORE_3D_DEPTH_FORMAT, Opaque3d, Opaque3dBatchSetKey, Opaque3dBinKey};
use bevy::ecs::system::{lifetimeless::*, SystemParamItem};
use bevy::math::Affine3A;
use bevy::mesh::{Mesh, MeshVertexBufferLayout, MeshVertexBufferLayoutRef, VertexBufferLayout};
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
use bevy::render::render_resource::{
    binding_types::uniform_buffer, BindGroupEntries, BindGroupLayoutEntries,
    *,
};
use std::borrow::Cow;
use bevy::render::renderer::RenderDevice;
use bevy::render::sync_world::MainEntity;
use bevy::render::view::{ExtractedView, Msaa};
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};
use bevy::ecs::query::ROQueryItem;
use bevy::transform::components::GlobalTransform;

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

/// The standalone pipeline for GradMesh rendering (no PBR dependency).
#[derive(Resource)]
pub struct StandalonePipeline {
    /// Shader handle for the standalone WGSL shader.
    shader: Handle<Shader>,
    /// Bind group layout descriptor for the view uniform (group 0).
    view_bind_group_layout: BindGroupLayoutDescriptor,
}

impl StandalonePipeline {
    pub fn new(shader: Handle<Shader>) -> Self {
        let view_bind_group_layout = BindGroupLayoutDescriptor::new(
            "standalone view layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX,
                (uniform_buffer::<StandaloneViewUniform>(true),),
            ),
        );
        Self {
            shader,
            view_bind_group_layout,
        }
    }
}

/// Uniform buffer sent to the standalone shader's @group(0).
#[derive(Debug, Clone, Copy, ShaderType, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct StandaloneViewUniform {
    view_proj: [f32; 16],
    time: f32,
    _pad: [f32; 3],
}

/// Specialized pipelines cache for standalone rendering.
pub type StandalonePipelineCache = SpecializedMeshPipelines<StandalonePipeline>;

/// Minimal pipeline key for standalone rendering: only topology, index format, and MSAA.
#[derive(Default, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct StandalonePipelineKey(u32);

impl StandalonePipelineKey {
    const PRIMITIVE_TOPOLOGY_TRIANGLE_LIST: u32 = 0;
    const PRIMITIVE_TOPOLOGY_POINT_LIST: u32 = 1 << 0;
    const PRIMITIVE_TOPOLOGY_LINE_LIST: u32 = 1 << 0;
    const PRIMITIVE_TOPOLOGY_LINE_STRIP: u32 = 1 << 1;

    const INDEX_FORMAT_UINT16: u32 = 1 << 2;
    const INDEX_FORMAT_UINT32: u32 = 1 << 3;

    const MSAA_2X: u32 = 1 << 4;
    const MSAA_4X: u32 = 1 << 5;
    const MSAA_8X: u32 = 1 << 6;

    fn from_mesh_and_msaa(topology: PrimitiveTopology, index_format: Option<IndexFormat>, msaa: Msaa) -> Self {
        let mut key: u32 = 0;
        match topology {
            PrimitiveTopology::TriangleList => {}
            PrimitiveTopology::PointList => key |= Self::PRIMITIVE_TOPOLOGY_POINT_LIST,
            PrimitiveTopology::LineList => key |= Self::PRIMITIVE_TOPOLOGY_LINE_LIST,
            PrimitiveTopology::LineStrip => key |= Self::PRIMITIVE_TOPOLOGY_LINE_STRIP,
            _ => {}
        }
        match index_format {
            Some(IndexFormat::Uint16) => key |= Self::INDEX_FORMAT_UINT16,
            Some(IndexFormat::Uint32) => key |= Self::INDEX_FORMAT_UINT32,
            _ => {}
        }
        match msaa {
            Msaa::Sample2 => key |= Self::MSAA_2X,
            Msaa::Sample4 => key |= Self::MSAA_4X,
            Msaa::Sample8 => key |= Self::MSAA_8X,
            Msaa::Off => {}
        }
        StandalonePipelineKey(key)
    }
}

impl From<StandalonePipelineKey> for u64 {
    fn from(key: StandalonePipelineKey) -> u64 {
        key.0 as u64
    }
}

impl From<u64> for StandalonePipelineKey {
    fn from(val: u64) -> Self {
        StandalonePipelineKey(val as u32)
    }
}

impl SpecializedMeshPipeline for StandalonePipeline {
    type Key = StandalonePipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        // Build vertex attributes from the mesh layout (position only at location 0)
        let layout_ref = &layout.0;

        // Add the instanced vertex buffer layout
        let transform_stride = size_of::<InstanceTransform>() as u64;
        let property_stride = size_of::<InstanceProperties>() as u64;

        let msaa_samples = match key.0 & (StandalonePipelineKey::MSAA_2X | StandalonePipelineKey::MSAA_4X | StandalonePipelineKey::MSAA_8X) {
            StandalonePipelineKey::MSAA_2X => 2,
            StandalonePipelineKey::MSAA_4X => 4,
            StandalonePipelineKey::MSAA_8X => 8,
            _ => 1,
        };

        let primitive_topology = if key.0 & StandalonePipelineKey::PRIMITIVE_TOPOLOGY_POINT_LIST != 0 {
            PrimitiveTopology::PointList
        } else if key.0 & StandalonePipelineKey::PRIMITIVE_TOPOLOGY_LINE_LIST != 0 {
            PrimitiveTopology::LineList
        } else if key.0 & StandalonePipelineKey::PRIMITIVE_TOPOLOGY_LINE_STRIP != 0 {
            PrimitiveTopology::LineStrip
        } else {
            PrimitiveTopology::TriangleList
        };

        let index_format = if key.0 & StandalonePipelineKey::INDEX_FORMAT_UINT32 != 0 {
            Some(IndexFormat::Uint32)
        } else if key.0 & StandalonePipelineKey::INDEX_FORMAT_UINT16 != 0 {
            Some(IndexFormat::Uint16)
        } else {
            None
        };

        Ok(RenderPipelineDescriptor {
            label: Some("standalone gradmesh pipeline".into()),
            layout: vec![self.view_bind_group_layout.clone()],
            immediate_size: 0,
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: Vec::new(),
                entry_point: Some(Cow::Borrowed("vs_standalone")),
                constants: Vec::new(),
                buffers: vec![
                    mesh_vertex_buffer_layout_from_layout(layout_ref),
                    VertexBufferLayout {
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
                    },
                ],
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs: Vec::new(),
                entry_point: Some(Cow::Borrowed("fs_standalone")),
                constants: Vec::new(),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Bgra8UnormSrgb,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                topology: primitive_topology,
                strip_index_format: index_format,
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::Less),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState {
                count: msaa_samples,
                mask: u64::MAX,
                alpha_to_coverage_enabled: false,
            },
            zero_initialize_workgroup_memory: false,
        })
    }
}

/// Build the mesh vertex buffer layout from the Arc-wrapped layout.
fn mesh_vertex_buffer_layout_from_layout(layout: &std::sync::Arc<MeshVertexBufferLayout>) -> VertexBufferLayout {
    let vb_layout = layout.layout();
    let mut attributes: Vec<VertexAttribute> = Vec::new();
    if layout.contains(Mesh::ATTRIBUTE_POSITION) {
        let desc = Mesh::ATTRIBUTE_POSITION.at_shader_location(0);
        if let Ok(found) = layout.get_layout(&[desc]) {
            attributes = found.attributes;
        }
    }
    VertexBufferLayout {
        array_stride: vb_layout.array_stride,
        step_mode: VertexStepMode::Vertex,
        attributes,
    }
}

// ─── View Uniform ───

/// Per-view bind group for the standalone pipeline.
#[derive(Component)]
pub struct StandaloneViewBindGroup {
    /// The bind group handle.
    value: BindGroup,
    /// Uniform buffer offset within the view uniform buffer.
    uniform_offset: u32,
}

/// The draw command tuple for standalone batches.
pub type DrawStandaloneBatch = (
    SetItemPipeline,
    SetStandaloneViewBindGroup<0>,
    DrawStandaloneBatchImpl,
);

/// Bind the standalone view uniform at group 0.
pub struct SetStandaloneViewBindGroup<const I: usize>;

impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetStandaloneViewBindGroup<I> {
    type Param = ();
    type ViewQuery = Read<StandaloneViewBindGroup>;
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        _item: &P,
        view_bind_group: ROQueryItem<'w, '_, Self::ViewQuery>,
        _entity: Option<()>,
        _param: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        pass.set_bind_group(I, &view_bind_group.value, &[]);
        RenderCommandResult::Success
    }
}

/// Implements the actual draw call for standalone batches.
pub struct DrawStandaloneBatchImpl;

impl<P: PhaseItem> RenderCommand<P> for DrawStandaloneBatchImpl {
    type Param = (
        SRes<RenderAssets<RenderMesh>>,
        SRes<MeshAllocator>,
    );
    type ViewQuery = ();
    type ItemQuery = (Read<StandaloneGpuBuffers>, Read<StandaloneBatchData>);

    #[inline]
    fn render<'w>(
        item: &P,
        _view: (),
        item_query: Option<(&'w StandaloneGpuBuffers, &'w StandaloneBatchData)>,
        (meshes, mesh_allocator): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let mesh_allocator = mesh_allocator.into_inner();

        let Some((gpu_buffers, batch_data)) = item_query else {
            return RenderCommandResult::Skip;
        };

        let Some(gpu_mesh) = meshes.into_inner().get(batch_data.mesh_id) else {
            return RenderCommandResult::Skip;
        };

        let Some(vertex_buffer_slice) =
            mesh_allocator.mesh_vertex_slice(&batch_data.mesh_id)
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
                    mesh_allocator.mesh_index_slice(&batch_data.mesh_id)
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

// ─── View Uniform Preparation ───

/// Prepare per-view uniform buffers and bind groups for standalone rendering.
fn prepare_standalone_view_uniforms(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<bevy::render::renderer::RenderQueue>,
    pipeline: Option<Res<StandalonePipeline>>,
    views: Query<(Entity, &ExtractedView)>,
    time: Res<Time>,
    existing: Query<Entity, With<StandaloneViewBindGroup>>,
) {
    let Some(pipeline) = pipeline else { return };

    // Remove stale bind groups
    for entity in &existing {
        commands.entity(entity).remove::<StandaloneViewBindGroup>();
    }

    let view_bind_group_layout = render_device.create_bind_group_layout(
        "standalone view layout",
        &pipeline.view_bind_group_layout.entries,
    );

    for (view_entity, view) in &views {
        let view_proj_mat: Mat4 = view.clip_from_view * view.world_from_view.affine();
        let uniform = StandaloneViewUniform {
            view_proj: view_proj_mat.to_cols_array_2d().as_flattened().try_into().unwrap_or([0.0; 16]),
            time: time.elapsed_secs() as f32,
            _pad: [0.0; 3],
        };

        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("standalone view uniform buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let bind_group = render_device.create_bind_group(
            "standalone view bind group",
            &view_bind_group_layout,
            &BindGroupEntries::single(BufferBinding {
                buffer: &buffer,
                offset: 0,
                size: None,
            }),
        );

        commands.entity(view_entity).insert(StandaloneViewBindGroup {
            value: bind_group,
            uniform_offset: 0,
        });
    }
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
    views: Query<(&ExtractedView, Option<&Msaa>)>,
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

    for (view, msaa_opt) in &views {
        let Some(opaque_phase) = opaque_render_phases.get_mut(&view.retained_view_entity) else {
            continue;
        };

        let msaa = msaa_opt.copied().unwrap_or(Msaa::Off);

        let clip_from_world = view.clip_from_world.unwrap_or(view.clip_from_view * Affine3A::from(view.world_from_view.affine()));
        let frustum = culling::FrustumPlanes::from_view_projection(&clip_from_world);
        let _camera_world_pos = view.world_from_view.translation();

        for (render_entity, &main_entity, batch_data) in &standalone_batches {
            if batch_data.instance_count == 0 {
                continue;
            }

            let Some(mesh) = meshes.get(batch_data.mesh_id) else {
                continue;
            };

            // Specialize the pipeline with topology + MSAA
            let key = StandalonePipelineKey::from_mesh_and_msaa(
                mesh.primitive_topology(),
                mesh.index_format(),
                msaa,
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
                    prepare_standalone_view_uniforms.in_set(RenderSystems::PrepareResources),
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
) {
    use bevy::asset::AssetPath;
    let path = AssetPath::from_path_buf(
        bevy::asset::embedded_path!("shader/standalone.wgsl"),
    ).with_source("embedded");
    let shader = asset_server.load(path);
    commands.insert_resource(StandalonePipeline::new(shader));
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
    fn standalone_pipeline_key_roundtrip() {
        let key = StandalonePipelineKey::default();
        let hash: u64 = key.into();
        assert_eq!(hash, 0);
        let back: StandalonePipelineKey = hash.into();
        assert_eq!(back, StandalonePipelineKey::default());
    }

    #[test]
    fn standalone_pipeline_key_from_mesh_and_msaa() {
        let key = StandalonePipelineKey::from_mesh_and_msaa(
            PrimitiveTopology::TriangleList,
            Some(IndexFormat::Uint32),
            Msaa::Sample4,
        );
        // Should have INDEX_FORMAT_UINT32 and MSAA_4X bits set
        assert_eq!(key.0, StandalonePipelineKey::INDEX_FORMAT_UINT32 | StandalonePipelineKey::MSAA_4X);
    }
}
