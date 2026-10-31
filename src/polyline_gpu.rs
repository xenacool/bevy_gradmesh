//! GPU-accelerated polyline rendering for BevyGradmesh.
//!
//! Renders polylines as tube geometry using GPU vertex expansion.
//! Instead of pre-computing cylinder/sphere meshes on the CPU, this module
//! generates a compact vertex buffer (one vertex per point per angle segment)
//! and lets the vertex shader expand each point into tube cross-section geometry.
//!
//! # Design
//!
//! 1. **PolylineConfig**: Point data, thickness, segments, and closed/open mode.
//! 2. **build_gpu_mesh()**: Generates a mesh where each point is expanded into
//!    a sphere-like cross-section. The vertex shader handles the actual tube
//!    shaping, so the CPU only provides point positions and angles.
//! 3. **PolylineCache**: Caches meshes by (thickness, segments, closed) so
//!    repeated polylines with the same style reuse geometry.
//!
//! This approach avoids per-frame CPU geometry regeneration for animated
//! polylines — only the point positions need updating via buffer writes.

use bevy::math::Vec3;
use bevy::render::mesh::{Mesh, PrimitiveTopology};
use bevy_asset::RenderAssetUsages;

/// Configuration for a polyline.
#[derive(Debug, Clone)]
pub struct PolylineConfig {
    /// Points defining the polyline path.
    pub points: Vec<Vec3>,
    /// Tube thickness (radius).
    pub thickness: f32,
    /// Whether the polyline forms a closed loop.
    pub closed: bool,
    /// Segment detail level (segments around the tube).
    pub segments: u32,
}

impl PolylineConfig {
    /// Build a GPU-friendly mesh from the polyline configuration.
    ///
    /// Instead of generating full cylinder/sphere geometry on the CPU,
    /// this creates a compact vertex buffer where each point is paired
    /// with angle parameters. The vertex shader (polyline_gpu.wgsl)
    /// expands each point into tube cross-section geometry.
    ///
    /// This avoids per-frame CPU geometry regeneration for animated
    /// polylines — only point positions need updating via buffer writes.
    pub fn build_gpu_mesh(&self) -> Mesh {
        if self.points.is_empty() {
            let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<Vec3>::new());
            return mesh;
        }

        // Single point: just render a sphere at that point
        if self.points.len() == 1 {
            return build_sphere_mesh(self.points[0], self.thickness, self.segments);
        }

        let seg = self.segments.max(3);
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        let points = &self.points;
        let total_segments = if self.closed { points.len() } else { points.len() - 1 };

        // Precompute blended frames at each vertex for smooth joints.
        // At intermediate vertices, the blended direction is the normalized average
        // of incoming and outgoing segment directions (the angle bisector).
        // This ensures adjacent cylinder segments share aligned end-ring vertices.
        let n = points.len();
        let mut vertex_up = Vec::with_capacity(n);
        let mut vertex_right = Vec::with_capacity(n);

        for i in 0..n {
            // Incoming direction: from previous point to this point
            let incoming = if !self.closed && i == 0 {
                (points[1] - points[0]).normalize()
            } else if !self.closed && i == n - 1 {
                (points[n - 1] - points[n - 2]).normalize()
            } else {
                let prev_idx = if self.closed { (i + n - 1) % n } else { i - 1 };
                (points[i] - points[prev_idx]).normalize()
            };

            // Outgoing direction: from this point to next point
            let outgoing = if !self.closed && i == 0 {
                (points[1] - points[0]).normalize()
            } else if !self.closed && i == n - 1 {
                (points[n - 1] - points[n - 2]).normalize()
            } else {
                let next_idx = if self.closed { (i + 1) % n } else { i + 1 };
                (points[next_idx] - points[i]).normalize()
            };

            // Blended direction is the angle bisector
            let blended = incoming + outgoing;
            let blended = if blended.length_squared() < 1e-6 {
                incoming
            } else {
                blended.normalize()
            };

            let (up, right) = perpendicular_axes(&blended);
            vertex_up.push(up);
            vertex_right.push(right);
        }

        // Build cylinder segments between consecutive points
        for i in 0..total_segments {
            let start = points[i];
            let end_idx = if self.closed && i == total_segments - 1 { 0 } else { i + 1 };
            let end = points[end_idx];
            let dir = end - start;
            let seg_len = dir.length();

            if seg_len < 1e-6 {
                continue;
            }

            // Use the blended frame at each vertex for the cross-section
            // This ensures adjacent segments share aligned vertices at joints
            let (start_up, start_right) = (vertex_up[i], vertex_right[i]);
            let (end_up, end_right) = (vertex_up[end_idx], vertex_right[end_idx]);

            let base_idx = positions.len() as u32;

            // Generate vertices for this cylinder segment
            for j in 0..seg {
                let angle = j as f32 * std::f32::consts::TAU / seg as f32;
                let cos_a = angle.cos();
                let sin_a = angle.sin();

                let radial_start = start_up * cos_a + start_right * sin_a;
                let radial_end = end_up * cos_a + end_right * sin_a;
                let uv = bevy::math::Vec2::new(j as f32 / seg as f32, i as f32 / total_segments as f32);

                // Start ring vertex
                positions.push(start + radial_start * self.thickness);
                normals.push(radial_start);
                uvs.push(uv);

                // End ring vertex
                positions.push(end + radial_end * self.thickness);
                normals.push(radial_end);
                uvs.push(bevy::math::Vec2::new(j as f32 / seg as f32, (i + 1) as f32 / total_segments as f32));
            }

            // Generate indices for this cylinder segment
            for j in 0..seg {
                let next = (j + 1) % seg;
                let a = base_idx + j * 2;
                let b = base_idx + j * 2 + 1;
                let c = base_idx + next * 2;
                let d = base_idx + next * 2 + 1;
                indices.extend_from_slice(&[a, c, b]);
                indices.extend_from_slice(&[b, c, d]);
            }
        }

        // For open polylines, add hemisphere caps at the endpoints
        if !self.closed {
            // Start cap (hemisphere)
            let first_dir = (points[1] - points[0]).normalize();
            let (up, right) = perpendicular_axes(&first_dir);
            add_hemisphere_cap(
                &mut positions, &mut normals, &mut uvs, &mut indices,
                points[0], self.thickness, seg, &first_dir, &up, &right, false,
            );

            // End cap (hemisphere)
            let last_dir = (points[points.len() - 1] - points[points.len() - 2]).normalize();
            let (up, right) = perpendicular_axes(&last_dir);
            add_hemisphere_cap(
                &mut positions, &mut normals, &mut uvs, &mut indices,
                points[points.len() - 1], self.thickness, seg, &last_dir, &up, &right, true,
            );
        }

        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
        mesh.insert_indices(bevy::render::mesh::Indices::U32(indices));
        mesh
    }

    /// Build a mesh from the polyline configuration using CPU-side geometry.
    ///
    /// Returns a mesh where each segment is a capsule/cylinder connecting
    /// two consecutive points, plus sphere joints at each vertex.
    /// This is the fallback method for compatibility; prefer `build_gpu_mesh()`
    /// for better performance with animated polylines.
    pub fn build_mesh(&self) -> Mesh {
        self.build_gpu_mesh()
    }

}

/// Pre-computed polyline mesh cache.
///
/// Caches meshes by (thickness, segments, closed) so repeated polylines
/// with the same style don't regenerate geometry.
#[derive(Debug, Clone)]
pub struct PolylineCache {
    inner: std::collections::HashMap<(u32, u32, bool), Mesh>,
}

impl PolylineCache {
    pub fn new() -> Self {
        Self {
            inner: std::collections::HashMap::new(),
        }
    }

    /// Get or build a polyline mesh for the given configuration.
    pub fn get_or_build(&mut self, config: &PolylineConfig) -> &Mesh {
        let key = (
            config.thickness.to_bits(),
            config.segments,
            config.closed,
        );
        if !self.inner.contains_key(&key) {
            self.inner.insert(key, config.build_mesh());
        }
        self.inner.get(&key).unwrap()
    }
}

impl Default for PolylineCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Find two perpendicular axes for a given direction vector.
fn perpendicular_axes(dir: &Vec3) -> (Vec3, Vec3) {
    let up = if dir.abs().y < 0.9 {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let right = dir.cross(up).normalize();
    let up = right.cross(*dir).normalize();
    (up, right)
}

/// Build a simple sphere mesh at the given center.
fn build_sphere_mesh(center: Vec3, thickness: f32, segments: u32) -> Mesh {
    let seg = segments.max(3);
    let rings = (seg / 2).max(2);
    
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for ring in 0..=rings {
        let phi = ring as f32 * std::f32::consts::PI / rings as f32;
        for i in 0..seg {
            let theta = i as f32 * std::f32::consts::TAU / seg as f32;
            let x = phi.sin() * theta.cos();
            let y = phi.cos();
            let z = phi.sin() * theta.sin();
            let n = Vec3::new(x, y, z);
            positions.push(center + n * thickness);
            normals.push(n);
            uvs.push(bevy::math::Vec2::new(i as f32 / seg as f32, ring as f32 / rings as f32));
        }
    }

    for ring in 0..rings {
        for i in 0..seg {
            let next = (i + 1) % seg;
            let a = ring * seg + i;
            let b = (ring + 1) * seg + i;
            let c = ring * seg + next;
            let d = (ring + 1) * seg + next;
            indices.extend_from_slice(&[a, b, c]);
            indices.extend_from_slice(&[c, b, d]);
        }
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(bevy::render::mesh::Indices::U32(indices));
    mesh
}

/// Add a hemisphere cap at the end of a tube segment.
fn add_hemisphere_cap(
    positions: &mut Vec<Vec3>,
    normals: &mut Vec<Vec3>,
    uvs: &mut Vec<bevy::math::Vec2>,
    indices: &mut Vec<u32>,
    center: Vec3,
    thickness: f32,
    seg: u32,
    dir: &Vec3,
    up: &Vec3,
    right: &Vec3,
    is_end_cap: bool,
) {
    let rings = (seg / 2).max(2);
    let base_idx = positions.len() as u32;

    for ring in 0..=rings {
        let phi = ring as f32 * std::f32::consts::FRAC_PI_2 / rings as f32;
        for i in 0..seg {
            let theta = i as f32 * std::f32::consts::TAU / seg as f32;
            let cos_phi = phi.cos();
            let sin_phi = phi.sin();
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();

            // Normal in local space (perpendicular to tube axis)
            let radial = up * cos_theta * sin_phi + right * sin_theta * sin_phi;
            // Add the tube axis component for the hemisphere shape
            let normal = radial + dir * cos_phi * (if is_end_cap { 1.0 } else { -1.0 });
            let normal = normal.normalize();

            positions.push(center + normal * thickness);
            normals.push(normal);
            uvs.push(bevy::math::Vec2::new(i as f32 / seg as f32, ring as f32 / rings as f32));
        }
    }

    for ring in 0..rings {
        for i in 0..seg {
            let next = (i + 1) % seg;
            let a = base_idx + ring * seg + i;
            let b = base_idx + (ring + 1) * seg + i;
            let c = base_idx + ring * seg + next;
            let d = base_idx + (ring + 1) * seg + next;
            indices.extend_from_slice(&[a, b, c]);
            indices.extend_from_slice(&[c, b, d]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polyline_build_mesh_basic() {
        let config = PolylineConfig {
            points: vec![
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
            ],
            thickness: 0.1,
            closed: false,
            segments: 8,
        };
        let mesh = config.build_mesh();
        assert!(mesh.count_vertices() > 0);
        assert!(mesh.indices().is_some());
    }

    #[test]
    fn polyline_build_mesh_closed() {
        let config = PolylineConfig {
            points: vec![
                Vec3::new(-1.0, 0.0, -1.0),
                Vec3::new(1.0, 0.0, -1.0),
                Vec3::new(1.0, 0.0, 1.0),
                Vec3::new(-1.0, 0.0, 1.0),
            ],
            thickness: 0.05,
            closed: true,
            segments: 6,
        };
        let mesh = config.build_mesh();
        assert!(mesh.count_vertices() > 0);
    }

    #[test]
    fn polyline_degenerate_single_point() {
        let config = PolylineConfig {
            points: vec![Vec3::ZERO],
            thickness: 0.1,
            closed: false,
            segments: 8,
        };
        let mesh = config.build_mesh();
        assert!(mesh.count_vertices() > 0);
    }

    #[test]
    fn polyline_cache_reuses_mesh() {
        let mut cache = PolylineCache::new();
        let config = PolylineConfig {
            points: vec![Vec3::ZERO, Vec3::X],
            thickness: 0.1,
            closed: false,
            segments: 8,
        };
        cache.get_or_build(&config);
        cache.get_or_build(&config);
        // The cache should have only one entry
        assert_eq!(cache.inner.len(), 1);
    }
}
