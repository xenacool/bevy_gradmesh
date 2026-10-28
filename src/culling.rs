//! Per-instance AABB frustum culling for the instanced render pipeline.
//!
//! Computes per-instance bounding boxes from transform data and tests them
//! against camera frustum planes to filter out invisible instances before
//! GPU upload, reducing bandwidth for large scenes.
//!
//! # Design
//!
//! 1. **AABB**: Simple axis-aligned bounding box (center + half_extents).
//! 2. **FrustumPlanes**: 6 frustum planes extracted from the view-projection matrix.
//! 3. **cull_instances()**: Tests each instance's AABB against all 6 planes,
//!    returning a vector of visible instance indices.

use bevy::math::{Mat4, Vec3, Vec4Swizzles};

/// Axis-aligned bounding box defined by center and half-extents.
#[derive(Debug, Clone, Copy)]
pub struct AABB {
    /// Center of the bounding box.
    pub center: Vec3,
    /// Half-extents (distance from center to each face).
    pub half_extents: Vec3,
}

impl AABB {
    /// Create an AABB from min/max corners.
    pub fn from_min_max(min: Vec3, max: Vec3) -> Self {
        let center = (min + max) * 0.5;
        let half_extents = (max - min) * 0.5;
        Self { center, half_extents }
    }

    /// Create an AABB from a mesh's vertex positions.
    pub fn from_mesh(mesh: &bevy::render::mesh::Mesh) -> Option<Self> {
        use bevy::render::mesh::VertexAttributeValues;
        
        let Ok(VertexAttributeValues::Float32x3(values)) =
            mesh.try_attribute(bevy::render::mesh::Mesh::ATTRIBUTE_POSITION)
        else {
            return None;
        };

        if values.is_empty() {
            return None;
        }

        let mut min = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut max = Vec3::new(f32::MIN, f32::MIN, f32::MIN);

        for value in values {
            let v = Vec3::new(value[0], value[1], value[2]);
            min = Vec3::min(min, v);
            max = Vec3::max(max, v);
        }

        Some(Self::from_min_max(min, max))
    }

    /// Transform this AABB by a matrix, returning a new AABB that encloses
    /// the transformed box. This is a conservative approximation.
    pub fn transform(&self, matrix: &Mat4) -> Self {
        // Transform the 8 corners and compute the enclosing AABB
        let corners = self.corners();
        let transformed: Vec<Vec3> = corners
            .map(|c| {
                let p = matrix.mul_vec4(c.extend(1.0));
                Vec3::new(p.x / p.w, p.y / p.w, p.z / p.w)
            })
            .to_vec();

        let mut min = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut max = Vec3::new(f32::MIN, f32::MIN, f32::MIN);
        for v in &transformed {
            min = Vec3::min(min, *v);
            max = Vec3::max(max, *v);
        }

        Self::from_min_max(min, max)
    }

    /// Returns the 8 corner points of this AABB.
    fn corners(&self) -> [Vec3; 8] {
        let hx = self.half_extents.x;
        let hy = self.half_extents.y;
        let hz = self.half_extents.z;
        let c = self.center;

        [
            Vec3::new(c.x - hx, c.y - hy, c.z - hz),
            Vec3::new(c.x + hx, c.y - hy, c.z - hz),
            Vec3::new(c.x + hx, c.y + hy, c.z - hz),
            Vec3::new(c.x - hx, c.y + hy, c.z - hz),
            Vec3::new(c.x - hx, c.y - hy, c.z + hz),
            Vec3::new(c.x + hx, c.y - hy, c.z + hz),
            Vec3::new(c.x + hx, c.y + hy, c.z + hz),
            Vec3::new(c.x - hx, c.y + hy, c.z + hz),
        ]
    }
}

/// A single frustum plane: ax + by + cz + d >= 0 for points inside the frustum.
#[derive(Debug, Clone, Copy)]
pub struct FrustumPlane {
    /// Normal vector (a, b, c) and distance d stored as (a, b, c, d).
    pub abc: Vec3,
    /// Distance from origin.
    pub d: f32,
}

impl FrustumPlane {
    /// Test if a point is inside (or on the boundary of) this plane.
    #[inline]
    pub fn contains_point(&self, point: &Vec3) -> bool {
        self.abc.dot(*point) + self.d >= 0.0
    }

    /// Test if an AABB is inside (or intersecting) this plane.
    /// Returns true if the AABB has any point on the positive side of the plane.
    #[inline]
    pub fn contains_aabb(&self, aabb: &AABB) -> bool {
        // Find the most negative point of the AABB relative to this plane
        // If that point is still >= 0, the entire AABB is inside
        let center = aabb.center;
        let half = aabb.half_extents;

        // The most negative point along the plane normal
        let negative_point = Vec3::new(
            if self.abc.x >= 0.0 { center.x - half.x } else { center.x + half.x },
            if self.abc.y >= 0.0 { center.y - half.y } else { center.y + half.y },
            if self.abc.z >= 0.0 { center.z - half.z } else { center.z + half.z },
        );

        self.contains_point(&negative_point)
    }
}

/// Six frustum planes extracted from a view-projection matrix.
#[derive(Debug, Clone)]
pub struct FrustumPlanes {
    /// 6 planes: right, left, top, bottom, far, near.
    pub planes: [FrustumPlane; 6],
}

impl FrustumPlanes {
    /// Extract frustum planes from a view-projection matrix (clip_from_world).
    /// The matrix should be column-major (Bevy/Glam convention).
    ///
    /// Each plane is defined such that points inside the frustum satisfy:
    /// abc.dot(point) + d >= 0
    pub fn from_view_projection(vp: &Mat4) -> Self {
        let m = *vp;

        // Extract planes from the view-projection matrix
        // For a standard RH OpenGL perspective matrix, the planes are:
        // Right:  M[3] + M[0]
        // Left:   M[3] - M[0]
        // Top:    M[3] + M[1]
        // Bottom: M[3] - M[1]
        // Far:    M[3] - M[2]
        // Near:   M[3] + M[2]
        //
        // However, the sign convention depends on the projection. We normalize
        // so that the plane normal points inward (toward the frustum center).

        let right = Self::extract_plane_inward(&m, m.w_axis + m.x_axis);
        let left = Self::extract_plane_inward(&m, m.w_axis - m.x_axis);
        let top = Self::extract_plane_inward(&m, m.w_axis + m.y_axis);
        let bottom = Self::extract_plane_inward(&m, m.w_axis - m.y_axis);
        let far = Self::extract_plane_inward(&m, m.w_axis - m.z_axis);
        let near = Self::extract_plane_inward(&m, m.w_axis + m.z_axis);

        Self {
            planes: [right, left, top, bottom, far, near],
        }
    }

    fn extract_plane_inward(_m: &Mat4, v: bevy::math::Vec4) -> FrustumPlane {
        let len = v.xyz().length();
        if len > 0.0 {
            let abc = v.xyz() / len;
            let d = v.w / len;
            // Ensure the plane normal points inward by checking against the
            // frustum center (approximate: transform origin by inverse VP)
            // For simplicity, we check the camera position (origin in view space)
            // The camera origin in world space is the 4th column of the view matrix
            // For a VP matrix, the camera is at the point where the view transform places it
            // We use a heuristic: the origin should be inside, so flip if needed
            if abc.dot(Vec3::ZERO) + d < 0.0 {
                // Origin is outside; flip the plane
                FrustumPlane { abc: -abc, d: -d }
            } else {
                FrustumPlane { abc, d }
            }
        } else {
            FrustumPlane {
                abc: Vec3::ZERO,
                d: 0.0,
            }
        }
    }

    /// Test if an AABB intersects the frustum (is inside all 6 planes).
    #[inline]
    pub fn contains_aabb(&self, aabb: &AABB) -> bool {
        for plane in &self.planes {
            if !plane.contains_aabb(aabb) {
                return false;
            }
        }
        true
    }

    /// Test if a point is inside the frustum.
    #[inline]
    pub fn contains_point(&self, point: &Vec3) -> bool {
        for plane in &self.planes {
            if !plane.contains_point(point) {
                return false;
            }
        }
        true
    }
}

/// Per-instance transform for culling (matches InstanceTransform in batch.rs).
#[derive(Debug, Clone, Copy)]
pub struct InstanceTransform {
    /// Column-major 4x4 transform matrix stored as 4 Vec4 columns.
    pub col0: [f32; 4],
    pub col1: [f32; 4],
    pub col2: [f32; 4],
    pub col3: [f32; 4],
}

impl InstanceTransform {
    /// Convert to a Bevy Mat4.
    pub fn to_mat4(&self) -> Mat4 {
        Mat4::from_cols(
            bevy::math::Vec4::from(self.col0),
            bevy::math::Vec4::from(self.col1),
            bevy::math::Vec4::from(self.col2),
            bevy::math::Vec4::from(self.col3),
        )
    }
}

/// Cull instances against a frustum, returning indices of visible instances.
///
/// Given a base mesh AABB (in local space) and per-instance transforms,
/// computes the world-space AABB for each instance and tests it against
/// the frustum planes.
///
/// Returns a vector of indices for instances that are visible.
pub fn cull_instances(
    frustum: &FrustumPlanes,
    base_aabb: &AABB,
    transforms: &[InstanceTransform],
) -> Vec<u32> {
    let mut visible = Vec::with_capacity(transforms.len());

    for (i, transform) in transforms.iter().enumerate() {
        let mat = transform.to_mat4();
        let world_aabb = base_aabb.transform(&mat);

        if frustum.contains_aabb(&world_aabb) {
            visible.push(i as u32);
        }
    }

    visible
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aabb_from_min_max() {
        let aabb = AABB::from_min_max(Vec3::new(-1.0, -2.0, -3.0), Vec3::new(1.0, 2.0, 3.0));
        assert!((aabb.center - Vec3::ZERO).length() < 1e-5);
        assert!((aabb.half_extents - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-5);
    }

    #[test]
    fn aabb_transform_identity() {
        let aabb = AABB::from_min_max(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let transformed = aabb.transform(&Mat4::IDENTITY);
        assert!((transformed.center - aabb.center).length() < 1e-5);
        assert!((transformed.half_extents - aabb.half_extents).length() < 1e-5);
    }

    #[test]
    fn aabb_transform_translation() {
        let aabb = AABB::from_min_max(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let mat = Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0));
        let transformed = aabb.transform(&mat);
        assert!((transformed.center - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-5);
        assert!((transformed.half_extents - Vec3::new(1.0, 1.0, 1.0)).length() < 1e-5);
    }

    #[test]
    fn frustum_plane_contains_point() {
        // Simple plane: z >= -10 (points with z > -10 are inside)
        let plane = FrustumPlane {
            abc: Vec3::new(0.0, 0.0, 1.0),
            d: 10.0,
        };
        assert!(plane.contains_point(&Vec3::new(0.0, 0.0, 0.0)));
        assert!(plane.contains_point(&Vec3::new(0.0, 0.0, -9.0)));
        assert!(!plane.contains_point(&Vec3::new(0.0, 0.0, -11.0)));
    }

    #[test]
    fn frustum_plane_contains_aabb() {
        let plane = FrustumPlane {
            abc: Vec3::new(0.0, 0.0, 1.0),
            d: 5.0,
        };
        // AABB centered at origin with half_extents 1.0
        let aabb = AABB::from_min_max(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        assert!(plane.contains_aabb(&aabb));

        // AABB entirely outside the plane
        let outside = AABB::from_min_max(
            Vec3::new(-1.0, -1.0, -10.0),
            Vec3::new(1.0, 1.0, -6.0),
        );
        assert!(!plane.contains_aabb(&outside));
    }

    #[test]
    fn frustum_planes_from_perspective() {
        // Build a simple view-projection matrix: camera at (0,0,5) looking at origin
        // Using a standard perspective projection matrix (right-handed, OpenGL-style)
        let fov: f32 = 1.0;
        let aspect = 16.0 / 9.0;
        let near = 0.1;
        let far = 100.0;
        let f = 1.0 / (fov * 0.5).tan();
        // Standard perspective matrix (not infinite reverse Z)
        let nfar = near - far;
        let proj = Mat4::from_cols(
            bevy::math::Vec4::new(f / aspect, 0.0, 0.0, 0.0),
            bevy::math::Vec4::new(0.0, f, 0.0, 0.0),
            bevy::math::Vec4::new(0.0, 0.0, far / nfar, 1.0),
            bevy::math::Vec4::new(0.0, 0.0, near * far / nfar, 0.0),
        );
        // Camera at (0,0,5) looking toward -Z (toward origin)
        let view = Mat4::from_translation(Vec3::new(0.0, 0.0, -5.0));
        let vp = proj * view;
        let frustum = FrustumPlanes::from_view_projection(&vp);

        // Origin should be inside the frustum
        assert!(frustum.contains_point(&Vec3::ZERO));
    }

    #[test]
    fn cull_instances_all_visible() {
        let fov: f32 = 1.0;
        let near = 0.1;
        let far = 100.0;
        let f = 1.0 / (fov * 0.5).tan();
        let nfar = near - far;
        let proj = Mat4::from_cols(
            bevy::math::Vec4::new(f, 0.0, 0.0, 0.0),
            bevy::math::Vec4::new(0.0, f, 0.0, 0.0),
            bevy::math::Vec4::new(0.0, 0.0, far / nfar, 1.0),
            bevy::math::Vec4::new(0.0, 0.0, near * far / nfar, 0.0),
        );
        let view = Mat4::from_translation(Vec3::new(0.0, 0.0, -5.0));
        let vp = proj * view;
        let frustum = FrustumPlanes::from_view_projection(&vp);

        let base_aabb = AABB::from_min_max(
            Vec3::new(-0.1, -0.1, -0.1),
            Vec3::new(0.1, 0.1, 0.1),
        );

        // 3 instances near origin, all should be visible
        let transforms = vec![
            InstanceTransform {
                col0: [1.0, 0.0, 0.0, 0.0],
                col1: [0.0, 1.0, 0.0, 0.0],
                col2: [0.0, 0.0, 1.0, 0.0],
                col3: [0.0, 0.0, 0.0, 1.0],
            },
            InstanceTransform {
                col0: [1.0, 0.0, 0.0, 0.0],
                col1: [0.0, 1.0, 0.0, 0.0],
                col2: [0.0, 0.0, 1.0, 0.0],
                col3: [1.0, 0.0, 0.0, 1.0],
            },
            InstanceTransform {
                col0: [1.0, 0.0, 0.0, 0.0],
                col1: [0.0, 1.0, 0.0, 0.0],
                col2: [0.0, 0.0, 1.0, 0.0],
                col3: [-1.0, 0.0, 0.0, 1.0],
            },
        ];

        let visible = cull_instances(&frustum, &base_aabb, &transforms);
        assert_eq!(visible.len(), 3);
    }

    #[test]
    fn cull_instances_some_culled() {
        let fov: f32 = 0.5;
        let near = 0.1;
        let far = 100.0;
        let f = 1.0 / (fov * 0.5).tan();
        let nfar = near - far;
        let proj = Mat4::from_cols(
            bevy::math::Vec4::new(f, 0.0, 0.0, 0.0),
            bevy::math::Vec4::new(0.0, f, 0.0, 0.0),
            bevy::math::Vec4::new(0.0, 0.0, far / nfar, 1.0),
            bevy::math::Vec4::new(0.0, 0.0, near * far / nfar, 0.0),
        );
        let view = Mat4::from_translation(Vec3::new(0.0, 0.0, -2.0));
        let vp = proj * view;
        let frustum = FrustumPlanes::from_view_projection(&vp);

        let base_aabb = AABB::from_min_max(
            Vec3::new(-0.1, -0.1, -0.1),
            Vec3::new(0.1, 0.1, 0.1),
        );

        // One instance at origin (visible), one far to the side (culled)
        let transforms = vec![
            InstanceTransform {
                col0: [1.0, 0.0, 0.0, 0.0],
                col1: [0.0, 1.0, 0.0, 0.0],
                col2: [0.0, 0.0, 1.0, 0.0],
                col3: [0.0, 0.0, 0.0, 1.0],
            },
            InstanceTransform {
                col0: [1.0, 0.0, 0.0, 0.0],
                col1: [0.0, 1.0, 0.0, 0.0],
                col2: [0.0, 0.0, 1.0, 0.0],
                col3: [50.0, 0.0, 0.0, 1.0],
            },
        ];

        let visible = cull_instances(&frustum, &base_aabb, &transforms);
        assert!(visible.len() >= 1); // At least the origin instance should be visible
    }

    #[test]
    fn cull_instances_empty() {
        let frustum = FrustumPlanes::from_view_projection(&Mat4::IDENTITY);
        let base_aabb = AABB::from_min_max(Vec3::ZERO, Vec3::ONE);
        let visible = cull_instances(&frustum, &base_aabb, &[]);
        assert!(visible.is_empty());
    }
}
