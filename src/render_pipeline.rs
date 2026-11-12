//! Custom instanced render pipeline for BevyGradmesh.
//!
//! Provides GPU instancing via a custom `SpecializedMeshPipeline` that reads
//! per-instance transforms and properties from storage buffers, enabling
//! single-draw-call rendering of thousands of shapes.
//!
//! # Architecture
//!
//! 1. **GPU Buffers**: Each batch has one instance vertex buffer holding an
//!    [`InstanceData`] per instance: the transform (`InstanceTransform`, 64 bytes)
//!    followed by the properties (`InstanceProperties`, 112 bytes), 176 bytes in all.
//! 2. **Custom Shader**: `instanced.wesl` reads per-instance attributes and
//!    computes world position and gradient shading per instance.
//! 3. **Render Phase Items**: Batches are queued as `GradMeshBatchItem` in the
//!    opaque render phase.
//! 4. **Draw Command**: `DrawInstancedBatch` sets the pipeline, binds buffers,
//!    and issues `draw_indexed(instance_count, 1, ...)` for instanced rendering.

use bevy::ecs::prelude::*;
use bevy::render::render_resource::{BufferUsages, BufferVec};
use bevy::render::renderer::{RenderDevice, RenderQueue};

use crate::batch::{InstanceData, InstanceProperties, InstanceTransform};
use crate::deformation::DeformationParams;

// --- GPU Buffer Data ---

/// CPU-side data for a single batch, ready for GPU upload.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct InstancedBufferData {
    /// One transform per instance.
    pub transforms: Vec<InstanceTransform>,
    /// One set of properties per instance, with the deformation filled in.
    pub properties: Vec<InstanceProperties>,
    /// Number of instances.
    pub instance_count: u32,
}

impl InstancedBufferData {
    /// Create buffer data from transforms, properties, and deformation slices.
    pub fn from_slices_with_deformation(
        transforms: &[InstanceTransform],
        properties: &[InstanceProperties],
        deformations: &[DeformationParams],
    ) -> Self {
        let instance_count = transforms
            .len()
            .min(properties.len())
            .min(deformations.len());

        let properties = properties
            .iter()
            .zip(deformations)
            .take(instance_count)
            .map(|(properties, deformation)| InstanceProperties {
                deformation: deformation.to_vec4(),
                ..*properties
            })
            .collect();

        Self {
            transforms: transforms[..instance_count].to_vec(),
            properties,
            instance_count: instance_count as u32,
        }
    }
}

// --- GPU upload ---

/// The instance vertex buffer of one batch: an [`InstanceData`] per instance.
///
/// Bevy's `BufferVec` lays each record out with encase, owns the GPU buffer and grows it
/// when a batch gets bigger.
pub(crate) type InstanceBuffer = BufferVec<InstanceData>;

/// An empty instance buffer that is bound as a vertex buffer.
pub(crate) fn new_instance_buffer(label: &str) -> InstanceBuffer {
    let mut buffer = InstanceBuffer::new(BufferUsages::VERTEX);
    buffer.set_label(Some(label));
    buffer
}

/// Writes a batch's instances into `buffer` and queues the GPU upload.
///
/// Counts the upload in `stats`: a first allocation, a reallocation when the batch grew
/// past the buffer's capacity, and the bytes written. Uses as many instances as both
/// slices hold, never more than `instance_count`.
pub(crate) fn upload_instances(
    buffer: &mut InstanceBuffer,
    transforms: &[InstanceTransform],
    properties: &[InstanceProperties],
    instance_count: u32,
    device: &RenderDevice,
    queue: &RenderQueue,
    stats: &mut InstancedStats,
) {
    let count = transforms
        .len()
        .min(properties.len())
        .min(instance_count as usize);

    let had_buffer = buffer.buffer().is_some();
    let capacity_before = buffer.capacity();

    buffer.clear();
    buffer.reserve_internal(count);
    for (transform, properties) in transforms.iter().zip(properties).take(count) {
        buffer.push(InstanceData {
            transform: *transform,
            properties: *properties,
        });
    }
    buffer.write_buffer(device, queue);

    if !had_buffer {
        stats.buffer_allocations += 1;
    } else if buffer.capacity() > capacity_before {
        stats.buffer_reallocations += 1;
    }
    stats.dirty_batch_count += 1;
    stats.gpu_upload_bytes += (count * std::mem::size_of::<InstanceData>()) as u64;
}

// --- Statistics ---

/// Performance statistics for instanced rendering.
///
/// The resource lives in the main world, where you read it, and is refreshed
/// once per frame at the end of the main schedule from the numbers the render
/// world finished the frame before (rendering is pipelined, so they are one
/// frame behind). `InstancedStats` is `#[non_exhaustive]`: fields will be
/// added, so read it, never build it with a struct literal.
///
/// Counters that depend on a camera (`batch_count`, `total_instances`,
/// `culled_instances`, `draw_calls`, `visible_triangles`, `lod_distribution`
/// and the like) are summed over every view that rendered, so two cameras
/// double them; divide by [`InstancedStats::view_count`] for a per-view
/// figure. Counters about the data itself (uploads, allocations, the mesh
/// cache, materials) are per frame and are not multiplied by the views.
#[derive(Resource, Debug, Clone, Default)]
#[non_exhaustive]
pub struct InstancedStats {
    // -- Per view, summed over views --
    /// Views that queued instanced batches last frame.
    pub view_count: usize,
    /// Instances that survived culling last frame.
    pub total_instances: usize,
    /// Batches considered last frame.
    pub batch_count: usize,
    /// Instances culled by frustum culling last frame.
    pub culled_instances: usize,
    /// Total instances before culling last frame.
    pub total_instances_before_cull: usize,
    /// Draw calls issued: one per batch that had at least one visible instance.
    pub draw_calls: usize,
    /// Batches skipped because every instance was culled.
    pub skipped_batches: usize,
    /// Triangles drawn after culling, assuming triangle lists.
    pub visible_triangles: u64,
    /// LOD distribution: count of instances per LOD level (0=Minimal, 1=Low,
    /// 2=Medium, 3=High, 4=Extreme). The standalone pipeline has no
    /// distance-based LOD, so it records the level fixed by the vertex count
    /// of each batch's mesh.
    pub lod_distribution: [usize; 5],
    /// Instances currently dither cross-fading between two LOD levels.
    pub lod_transitioning_instances: usize,
    // -- Per frame --
    /// Batches whose LOD level changed last frame (counted on the first view
    /// only), which is how often the scene visibly pops.
    pub lod_changed_batches: usize,
    /// Batches uploaded to the GPU last frame (new or changed).
    pub dirty_batch_count: usize,
    /// Bytes written to the GPU last frame.
    pub gpu_upload_bytes: u64,
    /// GPU buffers created last frame for batches that had none.
    pub buffer_allocations: usize,
    /// GPU buffers replaced last frame because their batch outgrew them.
    /// Steady growth here, with a flat `gpu_upload_bytes`, means a batch
    /// changes size every frame.
    pub buffer_reallocations: usize,
    // -- Main world: mesh cache --
    /// [`MeshCache`](crate::MeshCache) hits since it was created or cleared.
    pub cache_hits: u64,
    /// [`MeshCache`](crate::MeshCache) misses since it was created or cleared.
    pub cache_misses: u64,
    /// Meshes held by the cache.
    pub cached_meshes: usize,
    /// Bytes of vertex and index data held by the cache.
    pub cached_mesh_bytes: usize,
    /// Meshes built last frame (cache misses).
    pub mesh_builds: u32,
    /// Microseconds spent building those meshes. This is the cost of animating
    /// a shape with `Shape::lerp` or rebuilding a heightfield.
    pub mesh_build_micros: u64,
    // -- Main world: materials --
    /// Pooled materials that something still refers to. A count that keeps
    /// climbing means handles are being kept alive.
    pub live_materials: usize,
}

impl InstancedStats {
    /// Resets every counter to zero, including the cumulative cache counters.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Clears the numbers the render world rewrites every frame.
    pub(crate) fn reset_render_frame(&mut self) {
        let keep = self.main_side();
        *self = Self::default();
        self.set_main_side(keep);
    }

    /// Copies the numbers measured in the render world from `render`.
    pub(crate) fn absorb_render_side(&mut self, render: &InstancedStats) {
        let keep = self.main_side();
        *self = render.clone();
        self.set_main_side(keep);
    }

    fn main_side(&self) -> (u64, u64, usize, usize, u32, u64, usize) {
        (
            self.cache_hits,
            self.cache_misses,
            self.cached_meshes,
            self.cached_mesh_bytes,
            self.mesh_builds,
            self.mesh_build_micros,
            self.live_materials,
        )
    }

    fn set_main_side(&mut self, m: (u64, u64, usize, usize, u32, u64, usize)) {
        self.cache_hits = m.0;
        self.cache_misses = m.1;
        self.cached_meshes = m.2;
        self.cached_mesh_bytes = m.3;
        self.mesh_builds = m.4;
        self.mesh_build_micros = m.5;
        self.live_materials = m.6;
    }
}

/// Carries the render world's finished-frame numbers to the main world. The
/// same `Arc` is a resource in both worlds.
#[derive(Resource, Clone, Default)]
pub(crate) struct InstancedStatsSink(pub(crate) std::sync::Arc<std::sync::Mutex<InstancedStats>>);

/// Registers [`InstancedStats`] and the systems that keep it current.
pub(crate) struct InstancedStatsPlugin;

impl bevy::app::Plugin for InstancedStatsPlugin {
    fn build(&self, app: &mut bevy::app::App) {
        use bevy::render::{Render, RenderApp, RenderSystems};
        let sink = InstancedStatsSink::default();
        app.insert_resource(sink.clone())
            .init_resource::<InstancedStats>()
            .add_systems(bevy::app::Last, publish_render_stats);
        if let Some(render) = app.get_sub_app_mut(RenderApp) {
            render
                .insert_resource(sink)
                .init_resource::<InstancedStats>()
                .add_systems(
                    Render,
                    (
                        begin_render_stats.in_set(RenderSystems::PrepareAssets),
                        end_render_stats.in_set(RenderSystems::Cleanup),
                    ),
                );
        }
    }
}

/// Triangles one instance of `mesh` draws, assuming triangle lists.
pub(crate) fn triangles_per_instance(mesh: &bevy::render::mesh::RenderMesh) -> u64 {
    use bevy::render::mesh::RenderMeshBufferInfo;
    match &mesh.buffer_info {
        RenderMeshBufferInfo::Indexed { count, .. } => *count as u64 / 3,
        RenderMeshBufferInfo::NonIndexed => mesh.vertex_count as u64 / 3,
    }
}

fn begin_render_stats(mut stats: ResMut<InstancedStats>) {
    stats.reset_render_frame();
}

fn end_render_stats(stats: Res<InstancedStats>, sink: Res<InstancedStatsSink>) {
    if let Ok(mut shared) = sink.0.lock() {
        *shared = stats.clone();
    }
}

fn publish_render_stats(sink: Res<InstancedStatsSink>, mut stats: ResMut<InstancedStats>) {
    if let Ok(shared) = sink.0.lock() {
        stats.absorb_render_side(&shared);
    }
}

#[cfg(test)]
mod stats_tests {
    use super::*;

    #[test]
    fn render_numbers_reach_the_main_world_without_losing_main_side_numbers() {
        let mut app = bevy::app::App::new();
        app.add_plugins(InstancedStatsPlugin);
        app.world_mut().resource_mut::<InstancedStats>().cached_meshes = 7;
        {
            let sink = app.world().resource::<InstancedStatsSink>().clone();
            let mut rendered = InstancedStats::default();
            rendered.draw_calls = 3;
            rendered.visible_triangles = 120;
            *sink.0.lock().unwrap() = rendered;
        }
        app.update();
        let stats = app.world().resource::<InstancedStats>();
        assert_eq!(stats.draw_calls, 3);
        assert_eq!(stats.visible_triangles, 120);
        assert_eq!(stats.cached_meshes, 7);
    }

    #[test]
    fn reset_render_frame_keeps_the_cache_numbers() {
        let mut stats = InstancedStats::default();
        stats.cache_hits = 5;
        stats.draw_calls = 9;
        stats.reset_render_frame();
        assert_eq!(stats.cache_hits, 5);
        assert_eq!(stats.draw_calls, 0);
        stats.reset();
        assert_eq!(stats.cache_hits, 0);
    }
}
