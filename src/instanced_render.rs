//! GPU instanced rendering pipeline for BevyGradmesh.
//!
//! Replaces the entity-per-instance fallback with a custom render phase that
//! issues `draw_indexed` instanced calls via the `instanced.wesl` shader.
//!
//! # Architecture
//!
//! 1. **Batch Data Extraction** - Each frame, `InstancedBatch` components are
//!    extracted to the render world via `ExtractComponent`.
//! 2. **GPU Buffers** - Per-batch transform and property data are uploaded to
//!    GPU vertex buffers with `VertexStepMode::Instance`.
//! 3. **Render Phase** - Batches are queued as `Opaque3d` phase items and
//!    rendered with a single `draw_indexed` call per batch.
//! 4. **Dirty Tracking** - Only dirty batches re-upload GPU data each frame.

use std::mem::size_of;

use bevy::core_pipeline::core_3d::{Opaque3d, Opaque3dBatchSetKey, Opaque3dBinKey};
use bevy::ecs::system::{SystemParamItem, lifetimeless::*};
use bevy::mesh::{MeshVertexBufferLayoutRef, VertexBufferLayout};
use bevy::pbr::{
    MeshPipeline, MeshPipelineKey, RenderMeshInstances, SetMeshBindGroup, SetMeshViewBindGroup,
    ViewKeyCache,
};
use bevy::prelude::*;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::mesh::allocator::MeshSlabs;
use bevy::render::mesh::{RenderMesh, RenderMeshBufferInfo, allocator::MeshAllocator};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_phase::{
    AddRenderCommand, BinnedRenderPhaseType, DrawFunctions, InputUniformIndex, PhaseItem,
    RenderCommand, RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewBinnedRenderPhases,
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
use crate::render_pipeline::{InstanceBuffer, InstancedStats, new_instance_buffer, upload_instances};

// --- Extracted Batch Data ---

/// Extracted batch data for the render world. Contains the mesh handle,
/// instance count, and per-instance transform and property data.
#[derive(Clone, Component, ExtractComponent)]
#[extract_app(RenderApp)]
pub struct InstancedBatchData {
    /// Mesh asset ID.
    pub mesh_id: AssetId<Mesh>,
    /// Number of instances.
    pub instance_count: u32,
    /// One transform per instance.
    pub transforms: Vec<InstanceTransform>,
    /// One set of properties per instance.
    pub properties: Vec<InstanceProperties>,
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
#[doc(hidden)]
#[derive(Component)]
pub struct InstancedGpuBuffers {
    /// Vertex buffer with one transform-plus-properties record per instance.
    buffer: InstanceBuffer,
    /// Number of instances.
    instance_count: u32,
}

// --- Pipeline ---

/// The specialized pipeline for instanced GradMesh rendering.
#[doc(hidden)]
#[derive(Resource)]
pub struct InstancedPipeline {
    /// The base mesh pipeline for specialization.
    mesh_pipeline: MeshPipeline,
    /// Shader handle for the instanced WGSL shader.
    shader: Handle<Shader>,
}

impl InstancedPipeline {
    pub fn new(mesh_pipeline: MeshPipeline, shader: Handle<Shader>) -> Self {
        Self {
            mesh_pipeline,
            shader,
        }
    }
}

/// Specialized pipelines cache for instanced rendering.
#[doc(hidden)]
pub type InstancedPipelineCache = SpecializedMeshPipelines<InstancedPipeline>;

// --- Draw Command ---

/// The draw command tuple for instanced batches.
#[doc(hidden)]
pub type DrawInstancedBatch = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshBindGroup<1>,
    DrawInstancedBatchImpl,
);

/// Implements the actual draw call for instanced batches.
#[doc(hidden)]
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

        let Some(mesh_instance) = render_mesh_instances
            .into_inner()
            .render_mesh_queue_data(item.main_entity())
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
        let Some(instance_buffer) = gpu_buffers.buffer.buffer() else {
            return RenderCommandResult::Skip;
        };
        pass.set_vertex_buffer(1, instance_buffer.slice(..));

        match &gpu_mesh.buffer_info {
            RenderMeshBufferInfo::Indexed {
                index_format,
                count,
            } => {
                let Some(index_buffer_slice) =
                    mesh_allocator.mesh_index_slice(&mesh_instance.mesh_asset_id())
                else {
                    return RenderCommandResult::Skip;
                };

                pass.set_index_buffer(index_buffer_slice.buffer.slice(..), *index_format);

                // Issue the instanced draw call
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

// --- Buffer Preparation ---

/// Prepare GPU buffers for instanced batches. Uploads dirty batch data.
fn prepare_instanced_buffers(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<bevy::render::renderer::RenderQueue>,
    new_batches: Query<(Entity, &InstancedBatchData), Without<InstancedGpuBuffers>>,
    mut dirty_batches: Query<(&InstancedBatchData, &mut InstancedGpuBuffers)>,
    mut stats: ResMut<InstancedStats>,
) {
    // Create buffers for new batches
    for (entity, batch_data) in &new_batches {
        if batch_data.instance_count == 0 {
            continue;
        }

        let mut buffer = new_instance_buffer("instanced batch data buffer");
        upload_instances(
            &mut buffer,
            &batch_data.transforms,
            &batch_data.properties,
            batch_data.instance_count,
            &render_device,
            &render_queue,
            &mut stats,
        );
        commands.entity(entity).insert(InstancedGpuBuffers {
            buffer,
            instance_count: batch_data.instance_count,
        });
    }

    // Re-upload dirty batches
    for (batch_data, mut gpu_buffers) in &mut dirty_batches {
        if !batch_data.dirty || batch_data.instance_count == 0 {
            continue;
        }

        // The buffer vector grows its GPU buffer when the batch got bigger; the draw call
        // always uses the batch's current count.
        upload_instances(
            &mut gpu_buffers.buffer,
            &batch_data.transforms,
            &batch_data.properties,
            batch_data.instance_count,
            &render_device,
            &render_queue,
            &mut stats,
        );
        gpu_buffers.instance_count = batch_data.instance_count;
    }
}

// --- Queue System ---

/// Queue instanced batches into the opaque render phase with frustum culling and LOD.
fn queue_instanced_batches(
    opaque_draw_functions: Res<DrawFunctions<Opaque3d>>,
    instanced_pipeline: Option<Res<InstancedPipeline>>,
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
    let Some(instanced_pipeline) = instanced_pipeline else {
        return;
    };
    let draw_instanced = opaque_draw_functions.read().id::<DrawInstancedBatch>();

    let mut first_view = true;
    for view in &views {
        let Some(opaque_phase) = opaque_render_phases.get_mut(&view.retained_view_entity) else {
            continue;
        };

        let Some(&view_key) = view_key_cache.get(&view.retained_view_entity) else {
            continue;
        };
        stats.view_count += 1;
        let counts_lod_changes = std::mem::take(&mut first_view);

        // Extract frustum planes from the view-projection matrix for culling
        let clip_from_world = view.clip_from_world.unwrap_or_else(|| {
            view.clip_from_view * bevy::math::Mat4::from(view.world_from_view.affine().inverse())
        });
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
                let (computed_lod, transition_weight) =
                    lod_cfg.compute_lod(distance, Some(batch_data.lod_level));

                // Update LOD level and transition weight
                if counts_lod_changes && computed_lod != batch_data.lod_level {
                    stats.lod_changed_batches += 1;
                }
                batch_data.lod_level = computed_lod;
                batch_data.lod_transition_weight = transition_weight;
            }

            // Specialize the pipeline for this mesh
            let key = view_key
                | MeshPipelineKey::from_primitive_topology_and_strip_index(
                    mesh.primitive_topology(),
                    mesh.index_format(),
                );

            let Ok(pipeline) =
                pipelines.specialize(&pipeline_cache, &instanced_pipeline, key, &mesh.layout)
            else {
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
            if batch_data.lod_transition_weight > 0.0 && batch_data.lod_transition_weight < 1.0 {
                stats.lod_transitioning_instances += visible_count;
            }

            // Skip batches where all instances are culled
            if visible_count == 0 {
                stats.skipped_batches += 1;
                continue;
            }
            stats.draw_calls += 1;
            stats.visible_triangles += visible_count as u64 * crate::render_pipeline::triangles_per_instance(mesh);

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
    batch_data
        .transforms
        .iter()
        .take(batch_data.instance_count as usize)
        .map(|t| culling::InstanceTransform {
            col0: [t.col0.x, t.col0.y, t.col0.z, t.col0.w],
            col1: [t.col1.x, t.col1.y, t.col1.z, t.col1.w],
            col2: [t.col2.x, t.col2.y, t.col2.z, t.col2.w],
            col3: [t.col3.x, t.col3.y, t.col3.z, t.col3.w],
        })
        .collect()
}

// --- Plugin ---

/// Plugin that wires the GPU instanced pipeline into Bevy's render graph.
pub struct InstancedRenderPlugin;

impl Plugin for InstancedRenderPlugin {
    fn build(&self, app: &mut App) {
        // Extract batch data from main world to render world
        app.add_plugins(ExtractComponentPlugin::<InstancedBatchData>::default());
        if !app.is_plugin_added::<crate::render_pipeline::InstancedStatsPlugin>() {
            app.add_plugins(crate::render_pipeline::InstancedStatsPlugin);
        }

        // Without a render sub-app (a headless app or a test) there is nothing to draw with,
        // so only the main-world side above is registered.
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_render_command::<Opaque3d, DrawInstancedBatch>()
            .init_resource::<InstancedPipelineCache>()
            .init_resource::<InstancedStats>()
            .add_systems(
                RenderStartup,
                init_instanced_pipeline.after(bevy::pbr::MeshPipelineSystems),
            )
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
/// Gracefully skips if MeshPipeline is not available (e.g., when bevy_pbr is not loaded).
fn init_instanced_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mesh_pipeline: Option<Res<MeshPipeline>>,
) {
    let Some(mesh_pipeline) = mesh_pipeline else {
        return;
    };
    use bevy::asset::AssetPath;
    let path = AssetPath::from_path_buf(bevy::asset::embedded_path!("shader/instanced.wesl"))
        .with_source("embedded");
    let shader = asset_server.load(path);
    commands.insert_resource(InstancedPipeline::new(mesh_pipeline.clone(), shader));
}

// --- Pipeline Specialization ---

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
        descriptor.vertex.entry_point = Some("instanced_vertex".into());
        descriptor.label = Some("instanced gradmesh pipeline".into());

        // Add the instanced vertex buffer layout
        // This buffer contains interleaved InstanceTransform + InstanceProperties per instance
        let transform_stride = size_of::<InstanceTransform>() as u64; // 64 bytes = 4xvec4
        let property_stride = size_of::<InstanceProperties>() as u64; // 112 bytes = 7xvec4 (includes deformation + appearance)

        descriptor.vertex.buffers.push(VertexBufferLayout {
            array_stride: transform_stride + property_stride,
            step_mode: VertexStepMode::Instance,
            attributes: [
                // InstanceTransform: 4xvec4 at locations 3-6
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
                // InstanceProperties: 7xvec4 at locations 7-13 (color, secondary, direction, flags, dash, deformation, appearance)
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

        if let Some(fragment) = descriptor.fragment.as_mut() {
            fragment.shader = self.shader.clone();
            fragment.entry_point = Some("instanced_fragment".into());
        }

        Ok(descriptor)
    }
}
