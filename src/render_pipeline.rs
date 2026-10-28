//! Custom instanced render pipeline for BevyGradmesh.
//!
//! Provides GPU instancing via a custom `SpecializedMeshPipeline` that reads
//! per-instance transforms and properties from storage buffers, enabling
//! single-draw-call rendering of thousands of shapes.
//!
//! # Architecture
//!
//! 1. **GPU Buffers**: Each batch has two storage buffers — one for transforms
//!    (`InstanceTransform`, 64 bytes per instance) and one for properties
//!    (`InstanceProperties`, 80 bytes per instance).
//! 2. **Custom Shader**: `instanced.wesl` reads per-instance attributes and
//!    computes world position and gradient shading per instance.
//! 3. **Render Phase Items**: Batches are queued as `GradMeshBatchItem` in the
//!    opaque render phase.
//! 4. **Draw Command**: `DrawInstancedBatch` sets the pipeline, binds buffers,
//!    and issues `draw_indexed(instance_count, 1, ...)` for instanced rendering.

use bevy::ecs::prelude::*;
use bevy::render::mesh::Mesh;

use crate::batch::{InstanceProperties, InstanceTransform};
use crate::deformation::DeformationParams;

/// Path to the instanced shader.
pub const INSTANCED_SHADER_PATH: &str = "src/shader/gradmesh.wesl";

// ─── GPU Buffer Data ───

/// CPU-side buffer data for a single batch, ready for GPU upload.
#[derive(Debug, Clone)]
pub struct InstancedBufferData {
    /// Flattened transform data (64 bytes per instance).
    pub transforms: Vec<u8>,
    /// Flattened property data (96 bytes per instance, includes deformation).
    pub properties: Vec<u8>,
    /// Number of instances.
    pub instance_count: u32,
}

impl InstancedBufferData {
    /// Create buffer data from transform and property slices.
    pub fn from_slices(
        transforms: &[InstanceTransform],
        properties: &[InstanceProperties],
    ) -> Self {
        let instance_count = transforms.len().min(properties.len());

        // Flatten transforms - use the slice's contiguous memory directly
        let transforms_bytes = if instance_count > 0 {
            let ptr = transforms.as_ptr();
            unsafe {
                std::slice::from_raw_parts(
                    ptr as *const u8,
                    instance_count * std::mem::size_of::<InstanceTransform>(),
                )
            }
            .to_vec()
        } else {
            Vec::new()
        };

        // Flatten properties (now includes deformation vec4)
        let properties_bytes = if instance_count > 0 {
            let ptr = properties.as_ptr();
            unsafe {
                std::slice::from_raw_parts(
                    ptr as *const u8,
                    instance_count * std::mem::size_of::<InstanceProperties>(),
                )
            }
            .to_vec()
        } else {
            Vec::new()
        };

        Self {
            transforms: transforms_bytes,
            properties: properties_bytes,
            instance_count: instance_count as u32,
        }
    }

    /// Create buffer data from transforms, properties, and deformation slices.
    pub fn from_slices_with_deformation(
        transforms: &[InstanceTransform],
        properties: &[InstanceProperties],
        deformations: &[DeformationParams],
    ) -> Self {
        let instance_count = transforms.len().min(properties.len()).min(deformations.len());

        // Flatten transforms
        let transforms_bytes = if instance_count > 0 {
            let ptr = transforms.as_ptr();
            unsafe {
                std::slice::from_raw_parts(
                    ptr as *const u8,
                    instance_count * std::mem::size_of::<InstanceTransform>(),
                )
            }
            .to_vec()
        } else {
            Vec::new()
        };

        // Build properties with deformation encoded in the deformation field
        let properties_bytes = if instance_count > 0 {
            let mut buf = Vec::with_capacity(instance_count * std::mem::size_of::<InstanceProperties>());
            for i in 0..instance_count {
                let mut prop = properties[i];
                prop.deformation = deformations[i].to_vec4();
                buf.extend_from_slice(unsafe {
                    std::slice::from_raw_parts(
                        &prop as *const InstanceProperties as *const u8,
                        std::mem::size_of::<InstanceProperties>(),
                    )
                });
            }
            buf
        } else {
            Vec::new()
        };

        Self {
            transforms: transforms_bytes,
            properties: properties_bytes,
            instance_count: instance_count as u32,
        }
    }

    /// Returns the total size in bytes for the transform buffer.
    pub fn transform_size(&self) -> u64 {
        self.transforms.len() as u64
    }

    /// Returns the total size in bytes for the property buffer.
    pub fn property_size(&self) -> u64 {
        self.properties.len() as u64
    }
}

// ─── Batch Item for Render Phase ───

/// A render phase item representing one instanced batch.
#[derive(Clone, Debug)]
pub struct GradMeshBatchItem {
    /// Mesh asset ID.
    pub mesh_id: bevy::asset::AssetId<Mesh>,
    /// Number of instances to draw.
    pub instance_count: u32,
}

// ─── Component ───

/// Component that marks an entity as an instanced GradMesh batch.
#[derive(Component)]
pub struct InstancedBatch {
    /// Mesh ID for the batch.
    pub mesh_id: bevy::asset::AssetId<Mesh>,
    /// Number of instances.
    pub instance_count: u32,
}

// ─── Statistics ───

/// Performance statistics for instanced rendering.
#[derive(Resource, Debug, Clone, Default)]
pub struct InstancedStats {
    /// Total instances rendered last frame.
    pub total_instances: usize,
    /// Number of batches rendered last frame.
    pub batch_count: usize,
    /// Number of dirty batches that needed GPU upload.
    pub dirty_batch_count: usize,
    /// Total bytes uploaded to GPU last frame.
    pub gpu_upload_bytes: u64,
    /// Mesh cache hits since last reset.
    pub cache_hits: u64,
    /// Mesh cache misses since last reset.
    pub cache_misses: u64,
    /// Instances culled by frustum culling last frame.
    pub culled_instances: usize,
    /// Total instances before culling last frame.
    pub total_instances_before_cull: usize,
    /// LOD distribution: count of instances per LOD level (0=Minimal, 1=Low, 2=Medium, 3=High, 4=Extreme).
    pub lod_distribution: [usize; 5],
}

impl InstancedStats {
    /// Reset stats for a new frame.
    pub fn reset(&mut self) {
        self.total_instances = 0;
        self.batch_count = 0;
        self.dirty_batch_count = 0;
        self.gpu_upload_bytes = 0;
        self.cache_hits = 0;
        self.cache_misses = 0;
        self.culled_instances = 0;
        self.total_instances_before_cull = 0;
        self.lod_distribution = [0; 5];
    }

    /// Log stats to console.
    pub fn log(&self) {
        let hit_rate = if self.cache_hits + self.cache_misses > 0 {
            self.cache_hits as f64 / (self.cache_hits + self.cache_misses) as f64
        } else {
            0.0
        };
        let cull_ratio = if self.total_instances_before_cull > 0 {
            self.culled_instances as f64 / self.total_instances_before_cull as f64
        } else {
            0.0
        };
        let lod_names = ["Minimal", "Low", "Medium", "High", "Extreme"];
        let lod_parts: Vec<String> = self.lod_distribution.iter().enumerate()
            .filter(|(_, count)| **count > 0)
            .map(|(i, count)| format!("{}:{}", lod_names[i], count))
            .collect();
        let lod_display = if lod_parts.is_empty() { "none".to_string() } else { lod_parts.join(", ") };
        println!(
            "InstancedStats: instances={}, batches={}, dirty={}, upload_bytes={}, cache_hit_rate={:.2}%, culled={}/{} ({:.1}%), LOD:{}",
            self.total_instances,
            self.batch_count,
            self.dirty_batch_count,
            self.gpu_upload_bytes,
            hit_rate * 100.0,
            self.culled_instances,
            self.total_instances_before_cull,
            cull_ratio * 100.0,
            lod_display,
        );
    }
}
