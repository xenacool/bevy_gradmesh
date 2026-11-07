//! Mesh builders behind the extruded-polygon, polygon-with-holes, lathe and
//! compound shapes.
//!
//! These are plain functions over vertex data; [`Shape`](crate::Shape) wraps them.

use bevy::color::{Color, ColorToComponents};
use bevy::math::{Mat3, Quat, Vec2, Vec3};
use bevy::mesh::{Indices, VertexAttributeValues};
use bevy::render::mesh::{Mesh, PrimitiveTopology};
use bevy::transform::components::Transform;
use bevy_asset::RenderAssetUsages;
use std::f32::consts::{FRAC_PI_2, TAU};

/// Corners sharper than this (cosine of the angle between adjacent edge
/// normals) get a hard edge instead of smooth shading.
const SMOOTH_COS: f32 = 0.82;

#[derive(Default)]
struct MeshData {
    positions: Vec<Vec3>,
    normals: Vec<Vec3>,
    uvs: Vec<Vec2>,
    colors: Vec<[f32; 4]>,
    indices: Vec<u32>,
}

impl MeshData {
    fn push(&mut self, p: Vec3, n: Vec3, uv: Vec2, color: [f32; 4]) -> u32 {
        let i = self.positions.len() as u32;
        self.positions.push(p);
        self.normals.push(n);
        self.uvs.push(uv);
        self.colors.push(color);
        i
    }

    fn tri(&mut self, a: u32, b: u32, c: u32) {
        self.indices.extend_from_slice(&[a, b, c]);
    }

    fn into_mesh(self, with_colors: bool) -> Mesh {
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, self.positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, self.uvs);
        if with_colors {
            mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, self.colors);
        }
        mesh.insert_indices(Indices::U32(self.indices));
        mesh
    }
}

const WHITE: [f32; 4] = [1.0; 4];

// ─── 2D ring helpers ───

fn signed_area(ring: &[Vec2]) -> f32 {
    (0..ring.len())
        .map(|i| ring[i].perp_dot(ring[(i + 1) % ring.len()]))
        .sum::<f32>()
        * 0.5
}

/// A cleaned polygon ring with optional per-vertex colours.
#[derive(Clone, Debug)]
struct Ring {
    pts: Vec<Vec2>,
    colors: Option<Vec<[f32; 4]>>,
}

impl Ring {
    /// Drops repeated points and the closing duplicate, then winds the ring
    /// counter-clockwise (`ccw`) or clockwise, keeping colours aligned.
    fn new(points: &[Vec2], colors: Option<&[[f32; 4]]>, ccw: bool) -> Option<Self> {
        let colors = colors.filter(|c| c.len() == points.len());
        let mut keep: Vec<usize> = Vec::with_capacity(points.len());
        for (i, p) in points.iter().enumerate() {
            if keep.last().is_none_or(|&k| points[k].distance_squared(*p) > 1e-10) {
                keep.push(i);
            }
        }
        while keep.len() > 1 && points[keep[0]].distance_squared(points[keep[keep.len() - 1]]) <= 1e-10
        {
            keep.pop();
        }
        if keep.len() < 3 {
            return None;
        }
        let area: f32 = {
            let pts: Vec<Vec2> = keep.iter().map(|&i| points[i]).collect();
            signed_area(&pts)
        };
        if area.abs() < 1e-12 {
            return None;
        }
        if (area > 0.0) != ccw {
            keep.reverse();
        }
        Some(Self {
            pts: keep.iter().map(|&i| points[i]).collect(),
            colors: colors.map(|c| keep.iter().map(|&i| c[i]).collect()),
        })
    }

    fn len(&self) -> usize {
        self.pts.len()
    }

    fn color(&self, i: usize) -> [f32; 4] {
        self.colors.as_ref().map_or(WHITE, |c| c[i])
    }

    /// Unit normal of edge `i -> i + 1` pointing away from the material (the
    /// material is on the left of every edge of a counter-clockwise outer ring or
    /// a clockwise hole).
    fn edge_normal(&self, i: usize) -> Vec2 {
        let d = self.pts[(i + 1) % self.len()] - self.pts[i];
        Vec2::new(d.y, -d.x).normalize_or_zero()
    }

    /// Outward normal used at vertex `k` for edge `edge`: smoothed with the
    /// neighbouring edge when the corner is gentle, the edge normal otherwise.
    fn vertex_normal(&self, k: usize, edge: usize) -> Vec2 {
        let n = self.len();
        let prev = self.edge_normal((k + n - 1) % n);
        let next = self.edge_normal(k);
        if prev.dot(next) > SMOOTH_COS {
            (prev + next).normalize_or(self.edge_normal(edge))
        } else {
            self.edge_normal(edge)
        }
    }

    /// Moves every vertex by `d` towards the material using mitred offsets.
    fn inset(&self, d: f32) -> Vec<Vec2> {
        if d == 0.0 {
            return self.pts.clone();
        }
        let n = self.len();
        (0..n)
            .map(|i| {
                let prev = self.pts[(i + n - 1) % n];
                let next = self.pts[(i + 1) % n];
                let e0 = (self.pts[i] - prev).normalize_or_zero();
                let e1 = (next - self.pts[i]).normalize_or_zero();
                let l0 = Vec2::new(-e0.y, e0.x);
                let l1 = Vec2::new(-e1.y, e1.x);
                let bis = l0 + l1;
                let miter = if bis.length_squared() < 1e-8 {
                    l0
                } else {
                    let m = bis / (1.0 + l0.dot(l1)).max(1e-3);
                    if m.length() > 2.0 { m.normalize() * 2.0 } else { m }
                };
                self.pts[i] + miter * d
            })
            .collect()
    }
}

// ─── Triangulation with holes ───

fn same(a: Vec2, b: Vec2) -> bool {
    a.distance_squared(b) < 1e-10
}

fn segments_cross(a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> bool {
    let o = |p: Vec2, q: Vec2, r: Vec2| (q - p).perp_dot(r - p);
    let (o1, o2, o3, o4) = (o(a, b, c), o(a, b, d), o(c, d, a), o(c, d, b));
    ((o1 > 1e-9 && o2 < -1e-9) || (o1 < -1e-9 && o2 > 1e-9))
        && ((o3 > 1e-9 && o4 < -1e-9) || (o3 < -1e-9 && o4 > 1e-9))
}

fn point_in_ring(p: Vec2, ring: &[Vec2]) -> bool {
    let mut inside = false;
    let mut j = ring.len() - 1;
    for i in 0..ring.len() {
        let (a, b) = (ring[i], ring[j]);
        if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Merged outer ring + holes: points, the `(ring, vertex)` each came from and
/// triangle indices into the points.
pub(crate) struct Triangulation {
    pub points: Vec<Vec2>,
    pub source: Vec<(usize, usize)>,
    pub indices: Vec<u32>,
}

/// Triangulates `outer` (counter-clockwise) with `holes` (clockwise) by
/// bridging each hole to the outer ring and ear clipping the result.
pub(crate) fn triangulate_with_holes(outer: &[Vec2], holes: &[&[Vec2]]) -> Triangulation {
    let mut ring: Vec<(Vec2, (usize, usize))> =
        outer.iter().enumerate().map(|(i, p)| (*p, (0, i))).collect();

    // Bridge right-most holes first so earlier bridges never block later ones.
    let mut order: Vec<usize> = (0..holes.len()).collect();
    let max_x = |h: &[Vec2]| h.iter().map(|p| p.x).fold(f32::MIN, f32::max);
    order.sort_by(|&a, &b| max_x(holes[b]).total_cmp(&max_x(holes[a])));

    for (done, &hi) in order.iter().enumerate() {
        let hole = holes[hi];
        let mi = (0..hole.len())
            .max_by(|&a, &b| hole[a].x.total_cmp(&hole[b].x))
            .unwrap_or(0);
        let m = hole[mi];
        let pending: Vec<&[Vec2]> = order[done + 1..].iter().map(|&k| holes[k]).collect();

        let mut candidates: Vec<usize> = (0..ring.len()).collect();
        candidates.sort_by(|&a, &b| {
            ring[a].0.distance_squared(m).total_cmp(&ring[b].0.distance_squared(m))
        });
        let clear = |v: Vec2| -> bool {
            let blocked = |a: Vec2, b: Vec2| {
                !(same(a, v) || same(b, v) || same(a, m) || same(b, m)) && segments_cross(m, v, a, b)
            };
            let n = ring.len();
            if (0..n).any(|i| blocked(ring[i].0, ring[(i + 1) % n].0)) {
                return false;
            }
            let own = hole.len();
            if (0..own).any(|i| blocked(hole[i], hole[(i + 1) % own])) {
                return false;
            }
            for h in &pending {
                if (0..h.len()).any(|i| blocked(h[i], h[(i + 1) % h.len()])) {
                    return false;
                }
            }
            let mid = (m + v) * 0.5;
            let ring_pts: Vec<Vec2> = ring.iter().map(|r| r.0).collect();
            point_in_ring(mid, &ring_pts) && !point_in_ring(mid, hole)
        };
        let ci = candidates
            .iter()
            .copied()
            .find(|&c| clear(ring[c].0))
            .unwrap_or(candidates[0]);

        let anchor = ring[ci];
        let mut merged: Vec<(Vec2, (usize, usize))> = ring[..=ci].to_vec();
        for k in 0..=hole.len() {
            let idx = (mi + k) % hole.len();
            merged.push((hole[idx], (hi + 1, idx)));
        }
        merged.push(anchor);
        merged.extend_from_slice(&ring[ci + 1..]);
        ring = merged;
    }

    let points: Vec<Vec2> = ring.iter().map(|r| r.0).collect();
    let source: Vec<(usize, usize)> = ring.iter().map(|r| r.1).collect();
    let indices = ear_clip(&points);
    Triangulation { points, source, indices }
}

/// Ear clipping for a counter-clockwise ring that may contain bridge
/// duplicates.
fn ear_clip(pts: &[Vec2]) -> Vec<u32> {
    let n = pts.len();
    let mut out = Vec::with_capacity(n.saturating_sub(2) * 3);
    if n < 3 {
        return out;
    }
    let (lo, hi) = pts
        .iter()
        .fold((Vec2::splat(f32::MAX), Vec2::splat(f32::MIN)), |(l, h), p| {
            (l.min(*p), h.max(*p))
        });
    let scale = (hi - lo).max_element().max(1e-6);
    let eps = 1e-7 * scale * scale;

    let mut idx: Vec<usize> = (0..n).collect();
    while idx.len() > 3 {
        let len = idx.len();
        let mut progressed = false;
        for i in 0..len {
            let (p, c, nx) = (idx[(i + len - 1) % len], idx[i], idx[(i + 1) % len]);
            let (a, b, d) = (pts[p], pts[c], pts[nx]);
            let cross = (b - a).perp_dot(d - b);
            if cross.abs() <= eps {
                // Zero-area spike or collinear vertex: drop without a triangle.
                idx.remove(i);
                progressed = true;
                break;
            }
            if cross < 0.0 {
                continue;
            }
            let blocked = idx.iter().any(|&j| {
                if j == p || j == c || j == nx {
                    return false;
                }
                let q = pts[j];
                if same(q, a) || same(q, b) || same(q, d) {
                    return false;
                }
                let s = |u: Vec2, v: Vec2| (v - u).perp_dot(q - u);
                s(a, b) >= -eps && s(b, d) >= -eps && s(d, a) >= -eps
            });
            if !blocked {
                out.extend_from_slice(&[p as u32, c as u32, nx as u32]);
                idx.remove(i);
                progressed = true;
                break;
            }
        }
        if !progressed {
            // Degenerate input: fan the remainder rather than loop forever.
            for k in 1..idx.len() - 1 {
                out.extend_from_slice(&[idx[0] as u32, idx[k] as u32, idx[k + 1] as u32]);
            }
            return out;
        }
    }
    if idx.len() == 3 {
        let (a, b, c) = (pts[idx[0]], pts[idx[1]], pts[idx[2]]);
        if (b - a).perp_dot(c - b).abs() > eps {
            out.extend_from_slice(&[idx[0] as u32, idx[1] as u32, idx[2] as u32]);
        }
    }
    out
}

// ─── Polygon / extruded polygon ───

fn empty_mesh() -> Mesh {
    MeshData::default().into_mesh(false)
}

/// Builds a flat (`depth <= 0`) or extruded polygon, optionally with holes.
///
/// The polygon lies in the XY plane, facing +Z, and extrusion is centred on
/// `z = 0`. `roundness` (`0..=1`) rounds the rim with a quarter-circle bevel up
/// to half the depth; `bevel_steps` is the number of pieces per bevel. Per-vertex
/// `colors` apply to the outer ring.
pub(crate) fn polygon_mesh(
    outer: &[Vec2],
    colors: Option<&[[f32; 4]]>,
    holes: &[Vec<Vec2>],
    depth: f32,
    roundness: f32,
    bevel_steps: u32,
) -> Mesh {
    let Some(outer) = Ring::new(outer, colors, true) else {
        return empty_mesh();
    };
    let has_colors = outer.colors.is_some();
    let holes: Vec<Ring> = holes
        .iter()
        .filter_map(|h| Ring::new(h, None, false))
        .collect();
    let rings: Vec<&Ring> = std::iter::once(&outer).chain(holes.iter()).collect();

    let (lo, hi) = outer
        .pts
        .iter()
        .fold((Vec2::splat(f32::MAX), Vec2::splat(f32::MIN)), |(l, h), p| {
            (l.min(*p), h.max(*p))
        });
    let extent = (hi - lo).max(Vec2::splat(1e-3));
    let cap_uv = |p: Vec2| (p - lo) / extent;

    let depth = if depth.is_nan() { 0.0 } else { depth.max(0.0) };
    let mut data = MeshData::default();

    let add_cap = |data: &mut MeshData, loops: &[Vec<Vec2>], z: f32, up: bool| {
        let hole_refs: Vec<&[Vec2]> = loops[1..].iter().map(Vec::as_slice).collect();
        let tri = triangulate_with_holes(&loops[0], &hole_refs);
        let normal = if up { Vec3::Z } else { Vec3::NEG_Z };
        let base = data.positions.len() as u32;
        for (p, &(r, v)) in tri.points.iter().zip(&tri.source) {
            data.push(Vec3::new(p.x, p.y, z), normal, cap_uv(*p), rings[r].color(v));
        }
        for t in tri.indices.chunks_exact(3) {
            if up {
                data.tri(base + t[0], base + t[1], base + t[2]);
            } else {
                data.tri(base + t[0], base + t[2], base + t[1]);
            }
        }
    };

    if depth <= 1e-6 {
        let flat: Vec<Vec<Vec2>> = rings.iter().map(|r| r.pts.clone()).collect();
        add_cap(&mut data, &flat, 0.0, true);
        return data.into_mesh(has_colors);
    }

    let half = depth * 0.5;
    let bevel = roundness.clamp(0.0, 1.0) * half.min(0.25 * extent.min_element());
    let bevel = if bevel.is_nan() { 0.0 } else { bevel };

    // Rim profile from the top cap edge down to the bottom cap edge.
    struct Profile {
        inset: f32,
        z: f32,
        nxy: f32,
        nz: f32,
    }
    let mut profile: Vec<Profile> = Vec::new();
    if bevel <= 1e-6 {
        profile.push(Profile { inset: 0.0, z: half, nxy: 1.0, nz: 0.0 });
        profile.push(Profile { inset: 0.0, z: -half, nxy: 1.0, nz: 0.0 });
    } else {
        let steps = bevel_steps.max(1);
        for k in 0..=steps {
            let phi = FRAC_PI_2 * (1.0 - k as f32 / steps as f32);
            profile.push(Profile {
                inset: bevel * (1.0 - phi.cos()),
                z: half - bevel + bevel * phi.sin(),
                nxy: phi.cos(),
                nz: phi.sin(),
            });
        }
        for k in 0..=steps {
            let phi = FRAC_PI_2 * k as f32 / steps as f32;
            profile.push(Profile {
                inset: bevel * (1.0 - phi.cos()),
                z: -(half - bevel + bevel * phi.sin()),
                nxy: phi.cos(),
                nz: -phi.sin(),
            });
        }
    }

    // Caps sit at the first / last profile point.
    for (up, p) in [(true, &profile[0]), (false, &profile[profile.len() - 1])] {
        let inset_rings: Vec<Vec<Vec2>> = rings.iter().map(|r| r.inset(p.inset)).collect();
        add_cap(&mut data, &inset_rings, p.z, up);
    }

    // Side walls: one quad strip per ring edge and profile segment.
    for ring in &rings {
        let n = ring.len();
        let offsets: Vec<Vec<Vec2>> = profile.iter().map(|p| ring.inset(p.inset)).collect();
        let lengths: Vec<f32> = (0..n)
            .map(|i| ring.pts[i].distance(ring.pts[(i + 1) % n]))
            .collect();
        let perimeter: f32 = lengths.iter().sum::<f32>().max(1e-6);
        let mut travelled = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            let (u0, u1) = (travelled / perimeter, (travelled + lengths[i]) / perimeter);
            travelled += lengths[i];
            let (na, nb) = (ring.vertex_normal(i, i), ring.vertex_normal(j, i));
            for pi in 0..profile.len() - 1 {
                let vertex = |level: usize, k: usize, normal: Vec2, u: f32, data: &mut MeshData| {
                    let p = &profile[level];
                    let xy = offsets[level][k];
                    let nrm = Vec3::new(normal.x * p.nxy, normal.y * p.nxy, p.nz).normalize_or_zero();
                    data.push(
                        Vec3::new(xy.x, xy.y, p.z),
                        nrm,
                        Vec2::new(u, (p.z + half) / depth),
                        ring.color(k),
                    )
                };
                let ta = vertex(pi, i, na, u0, &mut data);
                let tb = vertex(pi, j, nb, u1, &mut data);
                let ba = vertex(pi + 1, i, na, u0, &mut data);
                let bb = vertex(pi + 1, j, nb, u1, &mut data);
                data.tri(ta, bb, tb);
                data.tri(ta, ba, bb);
            }
        }
    }
    data.into_mesh(has_colors)
}


// ─── Lathe ───

/// Revolves `profile` (`x` = radius, `y` = height) around the Y axis.
///
/// The profile is reordered bottom-to-top if needed so normals face outward.
/// Hard corners keep a crisp edge; open ends with a radius above zero get flat
/// caps so the surface is closed.
pub(crate) fn lathe_mesh(profile: &[Vec2], segments: u32) -> Mesh {
    let mut pts: Vec<Vec2> = Vec::with_capacity(profile.len());
    for p in profile {
        let p = Vec2::new(p.x.max(0.0), p.y);
        if pts.last().is_none_or(|l| l.distance_squared(p) > 1e-10) {
            pts.push(p);
        }
    }
    if pts.len() < 2 {
        return empty_mesh();
    }
    if pts[0].y > pts[pts.len() - 1].y {
        pts.reverse();
    }
    let segments = segments.max(3);
    let n = pts.len();

    let edge_normal = |i: usize| -> Vec2 {
        let d = pts[i + 1] - pts[i];
        Vec2::new(d.y, -d.x).normalize_or_zero()
    };
    let vertex_normal = |k: usize, edge: usize| -> Vec2 {
        if k > 0 && k < n - 1 {
            let (a, b) = (edge_normal(k - 1), edge_normal(k));
            if a.dot(b) > SMOOTH_COS {
                return (a + b).normalize_or(edge_normal(edge));
            }
        }
        edge_normal(edge)
    };

    let mut lengths = vec![0.0f32];
    for i in 0..n - 1 {
        lengths.push(lengths[i] + pts[i].distance(pts[i + 1]));
    }
    let total = lengths[n - 1].max(1e-6);

    let mut data = MeshData::default();
    let ring = |data: &mut MeshData, k: usize, edge: usize| -> Vec<u32> {
        let nrm = vertex_normal(k, edge);
        (0..=segments)
            .map(|s| {
                let (sin, cos) = (TAU * s as f32 / segments as f32).sin_cos();
                data.push(
                    Vec3::new(pts[k].x * cos, pts[k].y, pts[k].x * sin),
                    Vec3::new(nrm.x * cos, nrm.y, nrm.x * sin).normalize_or_zero(),
                    Vec2::new(s as f32 / segments as f32, lengths[k] / total),
                    WHITE,
                )
            })
            .collect()
    };
    for i in 0..n - 1 {
        if (pts[i].distance_squared(pts[i + 1])) < 1e-10 {
            continue;
        }
        let a = ring(&mut data, i, i);
        let b = ring(&mut data, i + 1, i);
        for s in 0..segments as usize {
            data.tri(a[s], b[s], a[s + 1]);
            data.tri(a[s + 1], b[s], b[s + 1]);
        }
    }

    for (p, up) in [(pts[0], false), (pts[n - 1], true)] {
        if p.x <= 1e-6 {
            continue;
        }
        let normal = if up { Vec3::Y } else { Vec3::NEG_Y };
        let centre = data.push(Vec3::new(0.0, p.y, 0.0), normal, Vec2::splat(0.5), WHITE);
        let rim: Vec<u32> = (0..=segments)
            .map(|s| {
                let (sin, cos) = (TAU * s as f32 / segments as f32).sin_cos();
                data.push(
                    Vec3::new(p.x * cos, p.y, p.x * sin),
                    normal,
                    Vec2::new(0.5 + 0.5 * cos, 0.5 + 0.5 * sin),
                    WHITE,
                )
            })
            .collect();
        for s in 0..segments as usize {
            if up {
                data.tri(centre, rim[s + 1], rim[s]);
            } else {
                data.tri(centre, rim[s], rim[s + 1]);
            }
        }
    }
    data.into_mesh(false)
}

// ─── Flat curves ───

/// Thick flat polyline in the XY plane (facing +Z) with mitred joints.
///
/// `u` runs along the line (0 at the start, 1 at the end) and `v` across it, so
/// `Style::dual_directional` with direction `X` fades along the curve and `Y`
/// fades across it.
pub(crate) fn polyline_2d_mesh(points: &[Vec2], thickness: f32, closed: bool) -> Mesh {
    let mut pts: Vec<Vec2> = Vec::with_capacity(points.len());
    for p in points {
        if pts.last().is_none_or(|l| l.distance_squared(*p) > 1e-10) {
            pts.push(*p);
        }
    }
    if closed && pts.len() > 1 && pts[0].distance_squared(pts[pts.len() - 1]) <= 1e-10 {
        pts.pop();
    }
    let closed = closed && pts.len() >= 3;
    let n = pts.len();
    if n < 2 {
        return empty_mesh();
    }
    let half = thickness.abs() * 0.5;

    let segment = |i: usize| -> Vec2 {
        let d = (pts[(i + 1) % n] - pts[i]).normalize_or_zero();
        Vec2::new(-d.y, d.x)
    };
    let mut left = Vec::with_capacity(n);
    let mut right = Vec::with_capacity(n);
    for i in 0..n {
        let normal = if closed || (i > 0 && i < n - 1) {
            let (a, b) = (segment((i + n - 1) % n), segment(i));
            let sum = a + b;
            if sum.length_squared() < 1e-8 {
                a
            } else {
                let m = sum / (1.0 + a.dot(b)).max(1e-3);
                if m.length() > 2.0 { m.normalize() * 2.0 } else { m }
            }
        } else if i == 0 {
            segment(0)
        } else {
            segment(n - 2)
        };
        left.push(pts[i] + normal * half);
        right.push(pts[i] - normal * half);
    }

    let edges = if closed { n } else { n - 1 };
    let mut lengths = vec![0.0f32];
    for i in 0..edges {
        lengths.push(lengths[i] + pts[i].distance(pts[(i + 1) % n]));
    }
    let total = lengths[edges].max(1e-6);

    let mut data = MeshData::default();
    for i in 0..edges {
        let j = (i + 1) % n;
        let (u0, u1) = (lengths[i] / total, lengths[i + 1] / total);
        let v = |p: Vec2, u: f32, v: f32, data: &mut MeshData| {
            data.push(Vec3::new(p.x, p.y, 0.0), Vec3::Z, Vec2::new(u, v), WHITE)
        };
        let l0 = v(left[i], u0, 1.0, &mut data);
        let r0 = v(right[i], u0, 0.0, &mut data);
        let l1 = v(left[j], u1, 1.0, &mut data);
        let r1 = v(right[j], u1, 0.0, &mut data);
        data.tri(l0, r0, l1);
        data.tri(r0, r1, l1);
    }
    data.into_mesh(false)
}

/// Samples a Bézier curve of any degree (De Casteljau) into `segments + 1` points.
pub(crate) fn bezier_2d_points(control_points: &[Vec2], segments: u32) -> Vec<Vec2> {
    if control_points.len() < 2 {
        return Vec::new();
    }
    let segments = segments.max(1);
    let mut scratch = vec![Vec2::ZERO; control_points.len()];
    (0..=segments)
        .map(|i| {
            let t = i as f32 / segments as f32;
            scratch.copy_from_slice(control_points);
            for level in (1..scratch.len()).rev() {
                for k in 0..level {
                    scratch[k] = scratch[k].lerp(scratch[k + 1], t);
                }
            }
            scratch[0]
        })
        .collect()
}

// ─── Compound merge ───

fn read_f32x3(mesh: &Mesh, attr: bevy::mesh::MeshVertexAttribute) -> Option<Vec<Vec3>> {
    match mesh.attribute(attr)? {
        VertexAttributeValues::Float32x3(v) => Some(v.iter().map(|a| Vec3::from(*a)).collect()),
        _ => None,
    }
}

/// Merges `parts` into one mesh, applying each transform and multiplying each
/// part's vertex colours by its tint (linear RGBA).
pub(crate) fn merge_meshes(parts: &[(Mesh, Transform, Option<Color>)]) -> Mesh {
    let any_colors = parts.iter().any(|(m, _, tint)| {
        tint.is_some() || m.attribute(Mesh::ATTRIBUTE_COLOR).is_some()
    });
    let mut data = MeshData::default();
    for (mesh, transform, tint) in parts {
        let Some(positions) = read_f32x3(mesh, Mesh::ATTRIBUTE_POSITION) else {
            continue;
        };
        let normals = read_f32x3(mesh, Mesh::ATTRIBUTE_NORMAL);
        let uvs = match mesh.attribute(Mesh::ATTRIBUTE_UV_0) {
            Some(VertexAttributeValues::Float32x2(v)) => Some(v.clone()),
            _ => None,
        };
        let colors = match mesh.attribute(Mesh::ATTRIBUTE_COLOR) {
            Some(VertexAttributeValues::Float32x4(v)) => Some(v.clone()),
            _ => None,
        };
        let tint = tint.map_or(WHITE, |c| c.to_linear().to_f32_array());

        let affine = transform.compute_affine();
        let linear = Mat3::from(affine.matrix3);
        let det = linear.determinant();
        let normal_matrix = if det.abs() > 1e-12 {
            linear.inverse().transpose()
        } else {
            Mat3::from_quat(if transform.rotation.is_finite() { transform.rotation } else { Quat::IDENTITY })
        };

        let base = data.positions.len() as u32;
        for (i, p) in positions.iter().enumerate() {
            let n = normals.as_ref().map_or(Vec3::Y, |n| n[i]);
            let c = colors.as_ref().map_or(WHITE, |c| c[i]);
            data.push(
                affine.transform_point3(*p),
                (normal_matrix * n).normalize_or(Vec3::Y),
                uvs.as_ref().map_or(Vec2::ZERO, |u| Vec2::from(u[i])),
                [c[0] * tint[0], c[1] * tint[1], c[2] * tint[2], c[3] * tint[3]],
            );
        }
        let indices: Vec<u32> = match mesh.indices() {
            Some(Indices::U16(v)) => v.iter().map(|&i| u32::from(i)).collect(),
            Some(Indices::U32(v)) => v.clone(),
            None => (0..positions.len() as u32).collect(),
        };
        for t in indices.chunks_exact(3) {
            if det < 0.0 {
                data.tri(base + t[0], base + t[2], base + t[1]);
            } else {
                data.tri(base + t[0], base + t[1], base + t[2]);
            }
        }
    }
    data.into_mesh(any_colors)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(half: f32) -> Vec<Vec2> {
        vec![
            Vec2::new(-half, -half),
            Vec2::new(half, -half),
            Vec2::new(half, half),
            Vec2::new(-half, half),
        ]
    }

    fn tri_area(points: &[Vec2], indices: &[u32]) -> f32 {
        indices
            .chunks_exact(3)
            .map(|t| {
                let (a, b, c) = (points[t[0] as usize], points[t[1] as usize], points[t[2] as usize]);
                (b - a).perp_dot(c - a) * 0.5
            })
            .sum()
    }

    fn mesh_positions(mesh: &Mesh) -> Vec<Vec3> {
        read_f32x3(mesh, Mesh::ATTRIBUTE_POSITION).unwrap()
    }

    fn mesh_indices(mesh: &Mesh) -> Vec<u32> {
        match mesh.indices().unwrap() {
            Indices::U32(v) => v.clone(),
            Indices::U16(v) => v.iter().map(|&i| u32::from(i)).collect(),
        }
    }

    #[test]
    fn triangulation_covers_polygon_area() {
        let outer = square(2.0);
        let t = triangulate_with_holes(&outer, &[]);
        assert!((tri_area(&t.points, &t.indices) - 16.0).abs() < 1e-4);
    }

    #[test]
    fn triangulation_subtracts_a_hole() {
        let outer = square(2.0);
        let mut hole = square(0.5);
        hole.reverse();
        let t = triangulate_with_holes(&outer, &[&hole]);
        assert!((tri_area(&t.points, &t.indices) - 15.0).abs() < 1e-4, "{}", tri_area(&t.points, &t.indices));
        assert!(t.indices.iter().all(|&i| (i as usize) < t.points.len()));
    }

    #[test]
    fn triangulation_handles_two_holes_and_concave_outline() {
        let outer = vec![
            Vec2::new(-3.0, -2.0),
            Vec2::new(3.0, -2.0),
            Vec2::new(3.0, 2.0),
            Vec2::new(0.0, 0.5),
            Vec2::new(-3.0, 2.0),
        ];
        let mut h1: Vec<Vec2> = square(0.4).iter().map(|p| *p + Vec2::new(-1.5, -0.8)).collect();
        let mut h2: Vec<Vec2> = square(0.4).iter().map(|p| *p + Vec2::new(1.5, -0.8)).collect();
        h1.reverse();
        h2.reverse();
        let t = triangulate_with_holes(&outer, &[&h1, &h2]);
        let expected = signed_area(&outer) - 2.0 * 0.64;
        assert!((tri_area(&t.points, &t.indices) - expected).abs() < 1e-3);
    }

    #[test]
    fn flat_polygon_faces_plus_z() {
        let mesh = polygon_mesh(&square(1.0), None, &[], 0.0, 0.0, 3);
        let pos = mesh_positions(&mesh);
        assert!(pos.iter().all(|p| p.z == 0.0));
        let idx = mesh_indices(&mesh);
        assert_eq!(idx.len(), 6);
        for t in idx.chunks_exact(3) {
            let n = (pos[t[1] as usize] - pos[t[0] as usize]).cross(pos[t[2] as usize] - pos[t[0] as usize]);
            assert!(n.z > 0.0);
        }
    }

    #[test]
    fn polygon_winding_is_normalised() {
        let mut cw = square(1.0);
        cw.reverse();
        let mesh = polygon_mesh(&cw, None, &[], 0.0, 0.0, 3);
        let pos = mesh_positions(&mesh);
        let idx = mesh_indices(&mesh);
        let n = (pos[idx[1] as usize] - pos[idx[0] as usize]).cross(pos[idx[2] as usize] - pos[idx[0] as usize]);
        assert!(n.z > 0.0);
    }

    #[test]
    fn extrusion_spans_depth_and_normals_point_outward() {
        let mesh = polygon_mesh(&square(1.0), None, &[], 0.5, 0.0, 3);
        let pos = mesh_positions(&mesh);
        let (min_z, max_z) = pos.iter().fold((f32::MAX, f32::MIN), |(a, b), p| (a.min(p.z), b.max(p.z)));
        assert!((min_z + 0.25).abs() < 1e-6 && (max_z - 0.25).abs() < 1e-6);
        // Every triangle's geometric normal agrees with its vertex normals.
        let normals = read_f32x3(&mesh, Mesh::ATTRIBUTE_NORMAL).unwrap();
        for t in mesh_indices(&mesh).chunks_exact(3) {
            let g = (pos[t[1] as usize] - pos[t[0] as usize]).cross(pos[t[2] as usize] - pos[t[0] as usize]);
            if g.length() > 1e-9 {
                assert!(g.dot(normals[t[0] as usize]) > 0.0, "inward triangle {t:?}");
            }
        }
    }

    #[test]
    fn rounded_extrusion_has_bevel_normals() {
        let mesh = polygon_mesh(&square(1.0), None, &[], 0.5, 1.0, 4);
        let normals = read_f32x3(&mesh, Mesh::ATTRIBUTE_NORMAL).unwrap();
        assert!(normals.iter().any(|n| n.z > 0.1 && n.z < 0.9));
        let pos = mesh_positions(&mesh);
        for t in mesh_indices(&mesh).chunks_exact(3) {
            let g = (pos[t[1] as usize] - pos[t[0] as usize]).cross(pos[t[2] as usize] - pos[t[0] as usize]);
            if g.length() > 1e-9 {
                assert!(g.dot(normals[t[0] as usize]) > 0.0);
            }
        }
    }

    #[test]
    fn extrusion_with_hole_has_inner_wall_facing_the_hole() {
        let mut hole = square(0.25);
        hole.reverse();
        let mesh = polygon_mesh(&square(1.0), None, &[hole], 0.4, 0.0, 3);
        let pos = mesh_positions(&mesh);
        let normals = read_f32x3(&mesh, Mesh::ATTRIBUTE_NORMAL).unwrap();
        // Inner-wall vertices sit on the hole boundary and point towards the axis.
        let inner: Vec<usize> = (0..pos.len())
            .filter(|&i| normals[i].z.abs() < 1e-6 && pos[i].truncate().abs().max_element() < 0.3)
            .collect();
        assert!(!inner.is_empty());
        for i in inner {
            assert!(normals[i].truncate().dot(pos[i].truncate()) < 0.0);
        }
    }

    #[test]
    fn coloured_polygon_keeps_vertex_colors() {
        let colors = vec![[1.0, 0.0, 0.0, 1.0]; 4];
        let mesh = polygon_mesh(&square(1.0), Some(&colors), &[], 0.3, 0.5, 2);
        assert!(mesh.attribute(Mesh::ATTRIBUTE_COLOR).is_some());
        let plain = polygon_mesh(&square(1.0), None, &[], 0.3, 0.5, 2);
        assert!(plain.attribute(Mesh::ATTRIBUTE_COLOR).is_none());
    }

    #[test]
    fn degenerate_polygon_gives_empty_mesh() {
        let mesh = polygon_mesh(&[Vec2::ZERO, Vec2::X], None, &[], 1.0, 0.0, 2);
        assert!(mesh_positions(&mesh).is_empty());
    }

    #[test]
    fn lathe_builds_closed_surface_with_outward_normals() {
        let profile = [Vec2::new(1.0, 0.0), Vec2::new(1.0, 2.0)];
        let mesh = lathe_mesh(&profile, 12);
        let pos = mesh_positions(&mesh);
        let normals = read_f32x3(&mesh, Mesh::ATTRIBUTE_NORMAL).unwrap();
        for t in mesh_indices(&mesh).chunks_exact(3) {
            let g = (pos[t[1] as usize] - pos[t[0] as usize]).cross(pos[t[2] as usize] - pos[t[0] as usize]);
            if g.length() > 1e-9 {
                assert!(g.dot(normals[t[0] as usize]) > 0.0, "inward lathe triangle {t:?}");
            }
        }
        // Side wall plus two caps: normals include +Y and -Y.
        assert!(normals.iter().any(|n| n.y > 0.99));
        assert!(normals.iter().any(|n| n.y < -0.99));
    }

    #[test]
    fn lathe_accepts_top_to_bottom_profiles() {
        let down = lathe_mesh(&[Vec2::new(0.5, 2.0), Vec2::new(1.0, 0.0)], 8);
        let up = lathe_mesh(&[Vec2::new(1.0, 0.0), Vec2::new(0.5, 2.0)], 8);
        assert_eq!(mesh_positions(&down), mesh_positions(&up));
    }

    #[test]
    fn lathe_on_axis_needs_no_cap() {
        let cone = lathe_mesh(&[Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0)], 8);
        let normals = read_f32x3(&cone, Mesh::ATTRIBUTE_NORMAL).unwrap();
        assert!(normals.iter().any(|n| n.y < -0.99)); // bottom cap
        assert!(!normals.iter().any(|n| n.y > 0.99)); // apex, no top cap
    }

    #[test]
    fn merge_applies_transform_and_tint() {
        let a = polygon_mesh(&square(1.0), None, &[], 0.0, 0.0, 1);
        let b = polygon_mesh(&square(1.0), None, &[], 0.0, 0.0, 1);
        let merged = merge_meshes(&[
            (a, Transform::IDENTITY, None),
            (
                b,
                Transform::from_xyz(0.0, 0.0, 5.0).with_scale(Vec3::splat(2.0)),
                Some(Color::srgb(1.0, 0.0, 0.0)),
            ),
        ]);
        let pos = mesh_positions(&merged);
        assert_eq!(pos.len(), 8);
        assert_eq!(pos[4].z, 5.0);
        assert!((pos[4].x.abs() - 2.0).abs() < 1e-6);
        let Some(VertexAttributeValues::Float32x4(colors)) = merged.attribute(Mesh::ATTRIBUTE_COLOR) else {
            panic!("expected colours");
        };
        assert_eq!(colors[0], WHITE);
        assert!(colors[4][0] > 0.99 && colors[4][1] < 1e-6);
        assert_eq!(mesh_indices(&merged).len(), 12);
    }

    #[test]
    fn merge_mirrored_part_keeps_facing() {
        let a = polygon_mesh(&square(1.0), None, &[], 0.0, 0.0, 1);
        let merged = merge_meshes(&[(a, Transform::from_scale(Vec3::new(-1.0, 1.0, 1.0)), None)]);
        let pos = mesh_positions(&merged);
        let idx = mesh_indices(&merged);
        let g = (pos[idx[1] as usize] - pos[idx[0] as usize]).cross(pos[idx[2] as usize] - pos[idx[0] as usize]);
        assert!(g.z > 0.0);
    }

    fn mesh_area_z(mesh: &Mesh) -> f32 {
        let pos = mesh_positions(mesh);
        mesh_indices(mesh)
            .chunks_exact(3)
            .map(|t| {
                (pos[t[1] as usize] - pos[t[0] as usize])
                    .cross(pos[t[2] as usize] - pos[t[0] as usize])
                    .z
                    * 0.5
            })
            .sum()
    }

    #[test]
    fn polyline_2d_has_length_times_thickness_area_and_faces_up() {
        let mesh = polyline_2d_mesh(&[Vec2::ZERO, Vec2::new(3.0, 0.0), Vec2::new(6.0, 0.0)], 0.5, false);
        assert!((mesh_area_z(&mesh) - 3.0).abs() < 1e-5, "{}", mesh_area_z(&mesh));
        let pos = mesh_positions(&mesh);
        assert!(pos.iter().all(|p| p.z == 0.0 && p.y.abs() <= 0.25 + 1e-6));
        let normals = read_f32x3(&mesh, Mesh::ATTRIBUTE_NORMAL).unwrap();
        assert!(normals.iter().all(|n| *n == Vec3::Z));
    }

    #[test]
    fn polyline_2d_closed_loop_and_degenerate_input() {
        let square = polyline_2d_mesh(&square(1.0), 0.2, true);
        // Perimeter 8 x thickness 0.2 (mitred corners keep the width).
        assert!((mesh_area_z(&square) - 1.6).abs() < 1e-3, "{}", mesh_area_z(&square));
        assert!(mesh_positions(&polyline_2d_mesh(&[Vec2::ZERO], 1.0, false)).is_empty());
        assert!(mesh_positions(&polyline_2d_mesh(&[Vec2::ONE, Vec2::ONE], 1.0, false)).is_empty());
    }

    #[test]
    fn polyline_2d_uvs_run_along_and_across() {
        let mesh = polyline_2d_mesh(&[Vec2::ZERO, Vec2::new(2.0, 0.0)], 1.0, false);
        let Some(VertexAttributeValues::Float32x2(uv)) = mesh.attribute(Mesh::ATTRIBUTE_UV_0) else {
            panic!("uv");
        };
        let pos = mesh_positions(&mesh);
        for (p, uv) in pos.iter().zip(uv) {
            assert!((uv[0] - p.x / 2.0).abs() < 1e-6);
            assert!((uv[1] - (p.y + 0.5)).abs() < 1e-6);
        }
    }

    #[test]
    fn bezier_2d_points_hit_the_endpoints() {
        let pts = bezier_2d_points(&[Vec2::ZERO, Vec2::new(1.0, 2.0), Vec2::new(2.0, 0.0)], 8);
        assert_eq!(pts.len(), 9);
        assert_eq!(pts[0], Vec2::ZERO);
        assert!((pts[8] - Vec2::new(2.0, 0.0)).length() < 1e-6);
        assert!((pts[4] - Vec2::new(1.0, 1.0)).length() < 1e-6);
        assert!(bezier_2d_points(&[Vec2::ZERO], 4).is_empty());
    }
}
