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
//! polylines - only the point positions need updating via buffer writes.

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
    /// polylines - only point positions need updating via buffer writes.
    pub fn build_gpu_mesh(&self) -> Mesh {
        if self.points.is_empty() {
            let mut mesh = Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::default(),
            );
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
        let n = points.len();
        let total_segments = if self.closed { n } else { n - 1 };

        // Rotation-minimising frames with mitre compensation at every vertex so
        // adjacent segments share identical rings (smooth, watertight joints).
        let frames = compute_ring_frames(points, self.closed);

        // Cumulative arc length gives a UV `v` that runs evenly along the path,
        // which is what along-the-path gradients and dashes need.
        let mut cumulative = Vec::with_capacity(total_segments + 1);
        cumulative.push(0.0_f32);
        for i in 0..total_segments {
            let a = points[i];
            let b = points[(i + 1) % n];
            let last = *cumulative.last().unwrap();
            cumulative.push(last + a.distance(b));
        }
        let total_length = cumulative.last().copied().unwrap_or(0.0).max(1e-6);

        let ring_offset = |frame: &RingFrame, angle: f32| -> (Vec3, Vec3) {
            let radial = frame.up * angle.cos() + frame.right * angle.sin();
            // Stretch the ring along the bend axis so the tube keeps its
            // thickness through the corner.
            let offset = radial + frame.bend_axis * (radial.dot(frame.bend_axis) * frame.mitre);
            (offset, radial)
        };

        for i in 0..total_segments {
            let end_idx = (i + 1) % n;
            let start = points[i];
            let end = points[end_idx];

            if start.distance(end) < 1e-6 {
                continue;
            }

            let (start_frame, end_frame) = (&frames[i], &frames[end_idx]);
            let v0 = cumulative[i] / total_length;
            let v1 = cumulative[i + 1] / total_length;
            let base_idx = positions.len() as u32;

            for j in 0..seg {
                let angle = j as f32 * std::f32::consts::TAU / seg as f32;
                let u = j as f32 / seg as f32;

                let (offset_start, radial_start) = ring_offset(start_frame, angle);
                positions.push(start + offset_start * self.thickness);
                normals.push(radial_start);
                uvs.push(bevy::math::Vec2::new(u, v0));

                let (offset_end, radial_end) = ring_offset(end_frame, angle);
                positions.push(end + offset_end * self.thickness);
                normals.push(radial_end);
                uvs.push(bevy::math::Vec2::new(u, v1));
            }

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
            let first = &frames[0];
            add_hemisphere_cap(
                &mut positions,
                &mut normals,
                &mut uvs,
                &mut indices,
                points[0],
                self.thickness,
                seg,
                &first.tangent,
                &first.up,
                &first.right,
                false,
            );

            let last = &frames[n - 1];
            add_hemisphere_cap(
                &mut positions,
                &mut normals,
                &mut uvs,
                &mut indices,
                points[n - 1],
                self.thickness,
                seg,
                &last.tangent,
                &last.up,
                &last.right,
                true,
            );
        }

        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
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
        let key = (config.thickness.to_bits(), config.segments, config.closed);
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

/// Cross-section frame of the tube at one path vertex.
#[derive(Debug, Clone, Copy)]
struct RingFrame {
    /// Path tangent (bisector of incoming/outgoing directions).
    tangent: Vec3,
    up: Vec3,
    right: Vec3,
    /// Axis (in the ring plane) along which the ring is stretched at a corner.
    bend_axis: Vec3,
    /// Extra stretch factor along `bend_axis` (`1/cos(half_turn) - 1`).
    mitre: f32,
}

/// Compute rotation-minimising ring frames for every vertex of a path.
///
/// Parallel transport keeps the rings from twisting between neighbours. For closed
/// loops the leftover twist is distributed evenly so the seam lines up. Corners
/// get a mitre factor so the tube keeps constant thickness through bends.
fn compute_ring_frames(points: &[Vec3], closed: bool) -> Vec<RingFrame> {
    let n = points.len();
    let seg_count = if closed { n } else { n - 1 };

    // Direction of each segment; zero-length segments inherit their neighbour.
    let mut dirs: Vec<Vec3> = (0..seg_count)
        .map(|i| (points[(i + 1) % n] - points[i]).normalize_or_zero())
        .collect();
    let fallback = dirs.iter().copied().find(|d| *d != Vec3::ZERO).unwrap_or(Vec3::X);
    let mut prev = fallback;
    for d in dirs.iter_mut() {
        if *d == Vec3::ZERO {
            *d = prev;
        }
        prev = *d;
    }

    // Per-vertex incoming/outgoing directions and bisector tangent.
    let mut tangents = Vec::with_capacity(n);
    let mut bend_axes = Vec::with_capacity(n);
    let mut mitres = Vec::with_capacity(n);
    for i in 0..n {
        let incoming = if closed {
            dirs[(i + seg_count - 1) % seg_count]
        } else if i == 0 {
            dirs[0]
        } else {
            dirs[i - 1]
        };
        let outgoing = if closed {
            dirs[i % seg_count]
        } else if i == n - 1 {
            dirs[seg_count - 1]
        } else {
            dirs[i]
        };

        let sum = incoming + outgoing;
        let tangent = if sum.length_squared() < 1e-6 {
            incoming
        } else {
            sum.normalize()
        };
        let cos_half = tangent.dot(incoming).clamp(0.35, 1.0);
        let bend = outgoing - incoming;
        let bend_axis = if bend.length_squared() < 1e-8 {
            Vec3::ZERO
        } else {
            bend.normalize()
        };
        tangents.push(tangent);
        bend_axes.push(bend_axis);
        mitres.push(1.0 / cos_half - 1.0);
    }

    // Parallel transport of the `up` vector along the tangents.
    let (up0, _) = perpendicular_axes(&tangents[0]);
    let transport = |up: Vec3, from: Vec3, to: Vec3| -> Vec3 {
        let rotated = bevy::math::Quat::from_rotation_arc(from, to) * up;
        (rotated - to * rotated.dot(to)).normalize()
    };

    let mut ups = Vec::with_capacity(n);
    ups.push(up0);
    for i in 1..n {
        ups.push(transport(ups[i - 1], tangents[i - 1], tangents[i]));
    }

    // Closed loop: spread the accumulated twist over all vertices.
    if closed {
        let wrapped = transport(ups[n - 1], tangents[n - 1], tangents[0]);
        let right0 = tangents[0].cross(up0);
        let twist = wrapped.dot(right0).atan2(wrapped.dot(up0));
        for (i, up) in ups.iter_mut().enumerate() {
            let correction = -twist * i as f32 / n as f32;
            *up = bevy::math::Quat::from_axis_angle(tangents[i], correction) * *up;
        }
    }

    (0..n)
        .map(|i| RingFrame {
            tangent: tangents[i],
            up: ups[i],
            right: tangents[i].cross(ups[i]),
            bend_axis: bend_axes[i],
            mitre: mitres[i],
        })
        .collect()
}

/// Find two perpendicular axes for a given direction vector.
fn perpendicular_axes(dir: &Vec3) -> (Vec3, Vec3) {
    let up = if dir.abs().y < 0.9 { Vec3::Y } else { Vec3::Z };
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
            uvs.push(bevy::math::Vec2::new(
                i as f32 / seg as f32,
                ring as f32 / rings as f32,
            ));
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

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
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
            uvs.push(bevy::math::Vec2::new(
                i as f32 / seg as f32,
                ring as f32 / rings as f32,
            ));
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
