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

// --- Heightfield ---

/// How far the skirts of a heightfield hang below the edge, as a fraction of
/// the larger extent.
const SKIRT_FRACTION: f32 = 0.1;

/// Depth of the skirts [`heightfield_mesh`] adds around the edge.
pub(crate) fn heightfield_skirt_depth(extent: Vec2) -> f32 {
    extent.x.abs().max(extent.y.abs()) * SKIRT_FRACTION
}

/// Checks the inputs of a heightfield; the error says what is wrong.
pub(crate) fn heightfield_check(
    size: bevy::math::UVec2,
    heights: usize,
    holes: Option<&[bool]>,
    border: Option<&crate::shape::HeightfieldBorder>,
) -> Result<(), String> {
    let (sx, sy) = (size.x as usize, size.y as usize);
    if sx < 2 || sy < 2 || heights != sx * sy {
        return Err(format!(
            "size {}x{} needs at least 2x2 samples and {} heights, got {}",
            size.x,
            size.y,
            sx * sy,
            heights
        ));
    }
    if let Some(h) = holes
        && h.len() != (sx - 1) * (sy - 1)
    {
        return Err(format!(
            "the hole mask needs one flag per cell, {}, got {}",
            (sx - 1) * (sy - 1),
            h.len()
        ));
    }
    if let Some(b) = border
        && (b.west.len() != sy || b.east.len() != sy || b.north.len() != sx || b.south.len() != sx)
    {
        return Err(format!(
            "the border needs {sy} samples on its west and east sides and {sx} on its north and south sides, got {}, {}, {} and {}",
            b.west.len(),
            b.east.len(),
            b.north.len(),
            b.south.len()
        ));
    }
    Ok(())
}

/// Which grid vertices a cell mask leaves in the mesh: those that touch at
/// least one cell that is not a hole.
fn heightfield_used(sx: usize, sy: usize, holes: Option<&[bool]>) -> Vec<bool> {
    let solid = |cx: usize, cy: usize| holes.is_none_or(|h| !h[cx + cy * (sx - 1)]);
    let mut used = vec![false; sx * sy];
    for cy in 0..sy - 1 {
        for cx in 0..sx - 1 {
            if solid(cx, cy) {
                for (x, y) in [(cx, cy), (cx + 1, cy), (cx, cy + 1), (cx + 1, cy + 1)] {
                    used[x + y * sx] = true;
                }
            }
        }
    }
    used
}

/// The four edges of the grid, each walked so a skirt faces outwards: grid
/// vertices in order, and for every step the cell it borders.
fn heightfield_edges(sx: usize, sy: usize) -> [(Vec<(usize, usize)>, Vec<(usize, usize)>); 4] {
    [
        (
            (0..sx).map(|x| (x, 0)).collect(),
            (0..sx - 1).map(|i| (i, 0)).collect(),
        ),
        (
            (0..sx).rev().map(|x| (x, sy - 1)).collect(),
            (0..sx - 1).map(|i| (sx - 2 - i, sy - 2)).collect(),
        ),
        (
            (0..sy).rev().map(|y| (0, y)).collect(),
            (0..sy - 1).map(|i| (0, sy - 2 - i)).collect(),
        ),
        (
            (0..sy).map(|y| (sx - 1, y)).collect(),
            (0..sy - 1).map(|i| (sx - 2, i)).collect(),
        ),
    ]
}

/// Bounds of a heightfield without and with its skirts, as `(surface, full)`.
/// Vertices that only touch holes, and skirts under holes, do not count.
pub(crate) fn heightfield_bounds(
    size: bevy::math::UVec2,
    extent: Vec2,
    heights: &[f32],
    holes: Option<&[bool]>,
) -> (crate::shape::ShapeBounds, crate::shape::ShapeBounds) {
    use crate::shape::ShapeBounds;
    let (sx, sy) = (size.x as usize, size.y as usize);
    let used = heightfield_used(sx, sy, holes);
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for (i, &h) in heights.iter().enumerate() {
        if used[i] {
            lo = lo.min(h);
            hi = hi.max(h);
        }
    }
    if lo > hi {
        return (ShapeBounds::default(), ShapeBounds::default());
    }
    let skirt = heightfield_skirt_depth(extent);
    let mut skirt_lo = lo;
    for (verts, cells) in heightfield_edges(sx, sy) {
        for (i, &(cx, cy)) in cells.iter().enumerate() {
            if holes.is_none_or(|h| !h[cx + cy * (sx - 1)]) {
                for (x, y) in [verts[i], verts[i + 1]] {
                    skirt_lo = skirt_lo.min(heights[x + y * sx] - skirt);
                }
            }
        }
    }
    let make = |lo: f32| ShapeBounds {
        min: Vec3::new(-extent.x * 0.5, lo, -extent.y * 0.5),
        max: Vec3::new(extent.x * 0.5, hi, extent.y * 0.5),
    };
    (make(lo), make(skirt_lo))
}

/// A grid of `size.x * size.y` height samples over `extent`, centred on the
/// origin in XZ (`heights[x + y * size.x]` is the sample at column `x`, row `y`,
/// rows running along +Z).
///
/// Normals come from central differences, UVs run 0 to 1 over the grid, and
/// skirts hang below the edges of the grid so cracks between neighbouring
/// chunks stay hidden. A `holes` cell (`holes[cx + cy * (size.x - 1)]`) has no
/// triangles, and no skirt below it; vertices only holes touch are left out.
/// With a `border`, the normals of the edge vertices use the neighbour heights
/// beyond the edge instead of one-sided differences. Invalid input returns an
/// empty mesh and logs a warning.
pub(crate) fn heightfield_mesh(
    size: bevy::math::UVec2,
    extent: Vec2,
    heights: &[f32],
    holes: Option<&[bool]>,
    hole_edge: crate::shape::HoleEdge,
    border: Option<&crate::shape::HeightfieldBorder>,
) -> Mesh {
    if let Err(why) = heightfield_check(size, heights.len(), holes, border) {
        bevy::log::warn!("heightfield: {why}; building an empty mesh");
        return MeshData::default().into_mesh(false);
    }
    if let Some(holes) = holes
        && hole_edge != crate::shape::HoleEdge::Cells
    {
        return heightfield_mesh_cut(size, extent, heights, holes, hole_edge, border);
    }
    let (sx, sy) = (size.x as usize, size.y as usize);
    let hx = extent.x / (sx - 1) as f32;
    let hz = extent.y / (sy - 1) as f32;
    let at = |x: usize, y: usize| heights[x + y * sx];
    let skirt = heightfield_skirt_depth(extent);
    let used = heightfield_used(sx, sy, holes);

    let mut data = MeshData::default();
    let mut id: Vec<Option<u32>> = vec![None; sx * sy];
    for y in 0..sy {
        for x in 0..sx {
            if !used[x + y * sx] {
                continue;
            }
            let c = at(x, y);
            let left = if x > 0 { Some(at(x - 1, y)) } else { border.map(|b| b.west[y]) };
            let right = if x + 1 < sx { Some(at(x + 1, y)) } else { border.map(|b| b.east[y]) };
            let back = if y > 0 { Some(at(x, y - 1)) } else { border.map(|b| b.north[x]) };
            let front = if y + 1 < sy { Some(at(x, y + 1)) } else { border.map(|b| b.south[x]) };
            let slope = |lo: Option<f32>, hi: Option<f32>, step: f32| match (lo, hi) {
                (Some(l), Some(r)) => (r - l) / (2.0 * step),
                (Some(l), None) => (c - l) / step,
                (None, Some(r)) => (r - c) / step,
                (None, None) => 0.0,
            };
            let normal = Vec3::new(-slope(left, right, hx), 1.0, -slope(back, front, hz)).normalize();
            let pos = Vec3::new(
                (x as f32 / (sx - 1) as f32 - 0.5) * extent.x,
                c,
                (y as f32 / (sy - 1) as f32 - 0.5) * extent.y,
            );
            let uv = Vec2::new(x as f32 / (sx - 1) as f32, y as f32 / (sy - 1) as f32);
            id[x + y * sx] = Some(data.push(pos, normal, uv, WHITE));
        }
    }
    let v = |x: usize, y: usize| id[x + y * sx].expect("a vertex of a solid cell is used");
    let solid = |cx: usize, cy: usize| holes.is_none_or(|h| !h[cx + cy * (sx - 1)]);
    for y in 0..sy - 1 {
        for x in 0..sx - 1 {
            if !solid(x, y) {
                continue;
            }
            let (a, b, c, d) = (v(x, y), v(x, y + 1), v(x + 1, y), v(x + 1, y + 1));
            data.tri(a, b, c);
            data.tri(c, b, d);
        }
    }

    // Skirts: only under edge cells that are not holes.
    let mut skirt_id: Vec<Option<u32>> = vec![None; sx * sy];
    for (verts, cells) in heightfield_edges(sx, sy) {
        for (i, &(cx, cy)) in cells.iter().enumerate() {
            if !solid(cx, cy) {
                continue;
            }
            let mut ends = [0u32; 2];
            let mut drops = [0u32; 2];
            for k in 0..2 {
                let (x, y) = verts[i + k];
                let e = v(x, y);
                let s = *skirt_id[x + y * sx].get_or_insert_with(|| {
                    let (p, n, uv) = (
                        data.positions[e as usize],
                        data.normals[e as usize],
                        data.uvs[e as usize],
                    );
                    data.push(p - Vec3::Y * skirt, n, uv, WHITE)
                });
                ends[k] = e;
                drops[k] = s;
            }
            data.tri(ends[0], ends[1], drops[0]);
            data.tri(ends[1], drops[1], drops[0]);
        }
    }
    data.into_mesh(false)
}

// --- Heightfield hole outlines ---

/// The coverage a point needs to count as land.
const HOLE_ISO: f32 = 0.5;

/// A cut this close to a vertex is moved onto it, so no sliver triangles remain.
const CUT_SNAP: f32 = 1e-4;

/// A point of the cut heightfield: a source vertex (a grid vertex, or the
/// centre vertex of a cell, which comes after all grid vertices) or the point
/// where the outline crosses the edge between two source vertices (lowest key
/// first, so that both triangles sharing the edge find the same point).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum CutPoint {
    Source(u32),
    Cut(u32, u32),
}

/// Source vertices of a cut heightfield with their land coverage: the outline
/// runs where the coverage crosses [`HOLE_ISO`], linear along every edge.
struct CutGrid {
    pos: Vec<Vec3>,
    nrm: Vec<Vec3>,
    uv: Vec<Vec2>,
    cover: Vec<f32>,
}

impl CutGrid {
    fn cut_t(&self, lo: u32, hi: u32) -> f32 {
        let (a, b) = (self.cover[lo as usize], self.cover[hi as usize]);
        (HOLE_ISO - a) / (b - a)
    }

    /// Where the outline crosses the edge between `a` and `b`, which must lie
    /// on different sides of it.
    fn cut(&self, a: u32, b: u32) -> CutPoint {
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let t = self.cut_t(lo, hi);
        if t <= CUT_SNAP {
            CutPoint::Source(lo)
        } else if t >= 1.0 - CUT_SNAP {
            CutPoint::Source(hi)
        } else {
            CutPoint::Cut(lo, hi)
        }
    }

    fn vertex(&self, p: CutPoint) -> (Vec3, Vec3, Vec2) {
        match p {
            CutPoint::Source(k) => (self.pos[k as usize], self.nrm[k as usize], self.uv[k as usize]),
            CutPoint::Cut(lo, hi) => {
                let (lo, hi) = (lo as usize, hi as usize);
                let t = self.cut_t(lo as u32, hi as u32);
                (
                    self.pos[lo].lerp(self.pos[hi], t),
                    self.nrm[lo].lerp(self.nrm[hi], t).normalize_or(Vec3::Y),
                    self.uv[lo].lerp(self.uv[hi], t),
                )
            }
        }
    }
}

/// Mesh under construction: vertices are created the first time a point is used.
struct CutMesh {
    data: MeshData,
    ids: std::collections::HashMap<CutPoint, u32>,
    skirt_ids: std::collections::HashMap<CutPoint, u32>,
    skirt: f32,
}

impl CutMesh {
    fn id(&mut self, grid: &CutGrid, p: CutPoint) -> u32 {
        if let Some(&i) = self.ids.get(&p) {
            return i;
        }
        let (pos, nrm, uv) = grid.vertex(p);
        let i = self.data.push(pos, nrm, uv, WHITE);
        self.ids.insert(p, i);
        i
    }

    /// A skirt under the chunk edge from `p` to `q`, walked with the land on
    /// the left as seen from above.
    fn skirt(&mut self, grid: &CutGrid, p: CutPoint, q: CutPoint) {
        if p == q {
            return;
        }
        let (pi, qi) = (self.id(grid, p), self.id(grid, q));
        let drop = |mesh: &mut Self, point: CutPoint, top: u32| {
            if let Some(&i) = mesh.skirt_ids.get(&point) {
                return i;
            }
            let i = mesh.data.push(
                mesh.data.positions[top as usize] - Vec3::Y * mesh.skirt,
                mesh.data.normals[top as usize],
                mesh.data.uvs[top as usize],
                WHITE,
            );
            mesh.skirt_ids.insert(point, i);
            i
        };
        let (pd, qd) = (drop(self, p, pi), drop(self, q, qi));
        self.data.tri(qi, pi, qd);
        self.data.tri(pi, pd, qd);
    }
}

/// Land coverage of every cell centre: 1 for land, 0 for a hole. With
/// [`HoleEdge::Smooth`](crate::shape::HoleEdge::Smooth) the mask is blurred over
/// the 3 x 3 cells around, then pushed back to at least 0.75 on land and at most
/// 0.25 on a hole, so that no hole and no island of land can vanish.
fn heightfield_cell_cover(
    sx: usize,
    sy: usize,
    holes: &[bool],
    edge: crate::shape::HoleEdge,
) -> Vec<f32> {
    let (cw, ch) = (sx - 1, sy - 1);
    let land = |x: usize, y: usize| if holes[x + y * cw] { 0.0 } else { 1.0 };
    let mut cover = Vec::with_capacity(cw * ch);
    for y in 0..ch {
        for x in 0..cw {
            let own = land(x, y);
            if edge != crate::shape::HoleEdge::Smooth {
                cover.push(own);
                continue;
            }
            const WEIGHTS: [[f32; 3]; 3] = [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]];
            let mut sum = 0.0;
            for (dy, row) in WEIGHTS.iter().enumerate() {
                for (dx, w) in row.iter().enumerate() {
                    let nx = (x + dx).saturating_sub(1).min(cw - 1);
                    let ny = (y + dy).saturating_sub(1).min(ch - 1);
                    sum += w * land(nx, ny);
                }
            }
            let blurred = sum / 16.0;
            cover.push(if own > 0.5 { blurred.max(0.75) } else { blurred.min(0.25) });
        }
    }
    cover
}

/// [`heightfield_mesh`] with the outline of the holes cut through the cells.
///
/// Every cell that touches the outline gets a centre vertex and is built as
/// four triangles around it, each clipped to the land. Cells that are land from
/// corner to corner keep the two triangles of the plain mesh, so the two meshes
/// agree away from the holes and share their edge vertices.
fn heightfield_mesh_cut(
    size: bevy::math::UVec2,
    extent: Vec2,
    heights: &[f32],
    holes: &[bool],
    edge: crate::shape::HoleEdge,
    border: Option<&crate::shape::HeightfieldBorder>,
) -> Mesh {
    let (sx, sy) = (size.x as usize, size.y as usize);
    let (cw, ch) = (sx - 1, sy - 1);
    let hx = extent.x / cw as f32;
    let hz = extent.y / ch as f32;
    let at = |x: usize, y: usize| heights[x + y * sx];
    let used = heightfield_used(sx, sy, Some(holes));
    let cell_cover = heightfield_cell_cover(sx, sy, holes, edge);

    let mut grid = CutGrid {
        pos: Vec::with_capacity(sx * sy + cw * ch),
        nrm: Vec::with_capacity(sx * sy + cw * ch),
        uv: Vec::with_capacity(sx * sy + cw * ch),
        cover: Vec::with_capacity(sx * sy + cw * ch),
    };
    for y in 0..sy {
        for x in 0..sx {
            let c = at(x, y);
            let left = if x > 0 { Some(at(x - 1, y)) } else { border.map(|b| b.west[y]) };
            let right = if x + 1 < sx { Some(at(x + 1, y)) } else { border.map(|b| b.east[y]) };
            let back = if y > 0 { Some(at(x, y - 1)) } else { border.map(|b| b.north[x]) };
            let front = if y + 1 < sy { Some(at(x, y + 1)) } else { border.map(|b| b.south[x]) };
            let slope = |lo: Option<f32>, hi: Option<f32>, step: f32| match (lo, hi) {
                (Some(l), Some(r)) => (r - l) / (2.0 * step),
                (Some(l), None) => (c - l) / step,
                (None, Some(r)) => (r - c) / step,
                (None, None) => 0.0,
            };
            grid.nrm.push(Vec3::new(-slope(left, right, hx), 1.0, -slope(back, front, hz)).normalize());
            grid.pos.push(Vec3::new(
                (x as f32 / cw as f32 - 0.5) * extent.x,
                c,
                (y as f32 / ch as f32 - 0.5) * extent.y,
            ));
            grid.uv.push(Vec2::new(x as f32 / cw as f32, y as f32 / ch as f32));
            // Coverage of a grid vertex: the mean over the cells around it.
            let (mut sum, mut count) = (0.0, 0.0);
            for cy in y.saturating_sub(1)..=y.min(ch - 1) {
                for cx in x.saturating_sub(1)..=x.min(cw - 1) {
                    sum += cell_cover[cx + cy * cw];
                    count += 1.0;
                }
            }
            grid.cover.push(sum / count);
        }
    }
    let key = |x: usize, y: usize| (x + y * sx) as u32;
    for cy in 0..ch {
        for cx in 0..cw {
            let corners = [key(cx, cy), key(cx, cy + 1), key(cx + 1, cy + 1), key(cx + 1, cy)];
            let on_land: Vec<u32> = corners.iter().copied().filter(|&k| used[k as usize]).collect();
            let from = if on_land.is_empty() { &corners[..] } else { &on_land[..] };
            let height = from.iter().map(|&k| grid.pos[k as usize].y).sum::<f32>() / from.len() as f32;
            let mean = |f: &dyn Fn(u32) -> Vec3| corners.iter().map(|&k| f(k)).sum::<Vec3>() / 4.0;
            let centre = mean(&|k| grid.pos[k as usize]);
            grid.pos.push(Vec3::new(centre.x, height, centre.z));
            grid.nrm.push(mean(&|k| grid.nrm[k as usize]).normalize_or(Vec3::Y));
            let uv = corners.iter().map(|&k| grid.uv[k as usize]).sum::<Vec2>() / 4.0;
            grid.uv.push(uv);
            grid.cover.push(cell_cover[cx + cy * cw]);
        }
    }

    let mut mesh = CutMesh {
        data: MeshData::default(),
        ids: std::collections::HashMap::new(),
        skirt_ids: std::collections::HashMap::new(),
        skirt: heightfield_skirt_depth(extent),
    };
    for cy in 0..ch {
        for cx in 0..cw {
            let centre = (sx * sy + cx + cy * cw) as u32;
            let corners = [key(cx, cy), key(cx, cy + 1), key(cx + 1, cy + 1), key(cx + 1, cy)];
            // Side `i` of the cell runs from corner `i` to corner `i + 1`.
            let on_chunk_edge = [cx == 0, cy + 1 == sy - 1, cx + 1 == sx - 1, cy == 0];
            let land = |k: u32| grid.cover[k as usize] >= HOLE_ISO;
            if land(centre) && corners.iter().all(|&k| land(k)) {
                let ids = corners.map(|k| mesh.id(&grid, CutPoint::Source(k)));
                mesh.data.tri(ids[0], ids[1], ids[3]);
                mesh.data.tri(ids[3], ids[1], ids[2]);
                for i in 0..4 {
                    if on_chunk_edge[i] {
                        let (p, q) = (corners[i], corners[(i + 1) % 4]);
                        mesh.skirt(&grid, CutPoint::Source(p), CutPoint::Source(q));
                    }
                }
                continue;
            }
            for i in 0..4 {
                let (p, q) = (corners[i], corners[(i + 1) % 4]);
                let triangle = [centre, p, q];
                // Clip the triangle to the land.
                let mut polygon: Vec<CutPoint> = Vec::with_capacity(4);
                for j in 0..3 {
                    let (a, b) = (triangle[j], triangle[(j + 1) % 3]);
                    let (a_in, b_in) = (land(a), land(b));
                    if a_in {
                        polygon.push(CutPoint::Source(a));
                    }
                    if a_in != b_in {
                        polygon.push(grid.cut(a, b));
                    }
                }
                polygon.dedup();
                while polygon.len() > 1 && polygon.first() == polygon.last() {
                    polygon.pop();
                }
                if polygon.len() < 3 {
                    continue;
                }
                let ids: Vec<u32> = polygon.iter().map(|&pt| mesh.id(&grid, pt)).collect();
                for j in 1..ids.len() - 1 {
                    mesh.data.tri(ids[0], ids[j], ids[j + 1]);
                }
                if on_chunk_edge[i] {
                    let on_side = |pt: CutPoint| {
                        pt == CutPoint::Source(p)
                            || pt == CutPoint::Source(q)
                            || pt == CutPoint::Cut(p.min(q), p.max(q))
                    };
                    for j in 0..polygon.len() {
                        let (a, b) = (polygon[j], polygon[(j + 1) % polygon.len()]);
                        if on_side(a) && on_side(b) {
                            mesh.skirt(&grid, a, b);
                        }
                    }
                }
            }
        }
    }
    mesh.data.into_mesh(false)
}

// --- Sweep ---

/// Profile corners sharper than this (cosine of the angle between the
/// adjacent edge normals, about 40 degrees) stay crisp when swept.
const SWEEP_SMOOTH_COS: f32 = 0.766;

/// Largest factor a mitred joint may widen the profile by.
const SWEEP_MAX_MITRE: f32 = 4.0;

/// Height tolerance when comparing opening outlines with profile heights.
const SWEEP_TOL: f32 = 1e-4;

/// Piecewise-linear lookup of `(ratio, value)` keyframes, clamped at the ends.
fn sample_keys<T: Copy>(keys: &[(f32, T)], r: f32, lerp: impl Fn(T, T, f32) -> T) -> Option<T> {
    let (first, last) = (keys.first()?, keys.last()?);
    if r <= first.0 {
        return Some(first.1);
    }
    for w in keys.windows(2) {
        if r <= w[1].0 {
            let span = w[1].0 - w[0].0;
            let t = if span > 1e-9 { (r - w[0].0) / span } else { 1.0 };
            return Some(lerp(w[0].1, w[1].1, t));
        }
    }
    Some(last.1)
}

/// An opening in `(distance along the path, height)` coordinates.
struct PreparedOpening {
    outline: Vec<Vec2>,
    s0: f32,
    s1: f32,
}

impl PreparedOpening {
    /// Lowest and highest outline height at distance `s` (clamped into the
    /// opening's range).
    fn span(&self, s: f32) -> (f32, f32) {
        let s = s.clamp(self.s0, self.s1);
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        let n = self.outline.len();
        for i in 0..n {
            let (a, b) = (self.outline[i], self.outline[(i + 1) % n]);
            if s < a.x.min(b.x) - 1e-6 || s > a.x.max(b.x) + 1e-6 {
                continue;
            }
            if (b.x - a.x).abs() < 1e-9 {
                lo = lo.min(a.y.min(b.y));
                hi = hi.max(a.y.max(b.y));
            } else {
                let t = ((s - a.x) / (b.x - a.x)).clamp(0.0, 1.0);
                let y = a.y + (b.y - a.y) * t;
                lo = lo.min(y);
                hi = hi.max(y);
            }
        }
        if lo <= hi { (lo, hi) } else { (0.0, 0.0) }
    }

    /// Distances where the opening's low or high chain bends or crosses one of
    /// `levels`, so that every ring cell sees straight chains.
    fn stations(&self, levels: &[f32]) -> Vec<f32> {
        let mut xs: Vec<f32> = self.outline.iter().map(|p| p.x).collect();
        xs.sort_by(f32::total_cmp);
        xs.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
        let mut out = xs.clone();
        for w in xs.windows(2) {
            let (a, b) = (w[0], w[1]);
            let (x1, x2) = (a + (b - a) * 0.25, a + (b - a) * 0.75);
            let (l1, h1) = self.span(x1);
            let (l2, h2) = self.span(x2);
            for (y1, y2) in [(l1, l2), (h1, h2)] {
                let slope = (y2 - y1) / (x2 - x1);
                if slope.abs() < 1e-9 {
                    continue;
                }
                for &v in levels {
                    let x = x1 + (v - y1) / slope;
                    if x > a + 1e-6 && x < b - 1e-6 {
                        out.push(x);
                    }
                }
            }
        }
        out
    }
}

/// Turns the user's openings into path-distance coordinates and drops the ones
/// that cannot be cut (too small, or overlapping an earlier opening).
fn prepare_openings(
    openings: &[crate::shape::SweepOpening],
    total: f32,
    closed: bool,
) -> Vec<PreparedOpening> {
    let margin = if closed { 0.0 } else { total * 1e-3 };
    let mut out: Vec<PreparedOpening> = Vec::new();
    for (k, op) in openings.iter().enumerate() {
        if op.outline.len() < 3 {
            bevy::log::warn!("sweep: opening {k} needs at least three outline points; ignored");
            continue;
        }
        let (ta, tb) = (op.t.start.min(op.t.end), op.t.start.max(op.t.end));
        let (sa, sb) = (ta.clamp(0.0, 1.0) * total, tb.clamp(0.0, 1.0) * total);
        if sb - sa <= total * 1e-4 {
            bevy::log::warn!("sweep: opening {k} covers no length of the path; ignored");
            continue;
        }
        let outline: Vec<Vec2> = op
            .outline
            .iter()
            .map(|p| {
                let s = sa + p.x.clamp(0.0, 1.0) * (sb - sa);
                Vec2::new(s.clamp(margin, total - margin), p.y)
            })
            .collect();
        let s0 = outline.iter().map(|p| p.x).fold(f32::MAX, f32::min);
        let s1 = outline.iter().map(|p| p.x).fold(f32::MIN, f32::max);
        if s1 - s0 <= total * 1e-4 {
            bevy::log::warn!("sweep: opening {k} outline has no width; ignored");
            continue;
        }
        if out.iter().any(|o| s0 < o.s1 - 1e-6 && s1 > o.s0 + 1e-6) {
            bevy::log::warn!("sweep: opening {k} overlaps an earlier opening; ignored");
            continue;
        }
        out.push(PreparedOpening { outline, s0, s1 });
    }
    out
}

fn path_length(pos: &[Vec3], closed: bool) -> f32 {
    let mut total: f32 = pos.windows(2).map(|w| w[0].distance(w[1])).sum();
    if closed && pos.len() > 1 {
        total += pos[pos.len() - 1].distance(pos[0]);
    }
    total
}

/// Adds a path point at every distance in `stations` that is not a ring yet.
fn insert_stations(pos: &mut Vec<Vec3>, closed: bool, stations: &mut [f32]) {
    if stations.is_empty() {
        return;
    }
    let m = pos.len();
    let mut d = vec![0.0_f32; m];
    for i in 1..m {
        d[i] = d[i - 1] + pos[i].distance(pos[i - 1]);
    }
    let total = path_length(pos, closed);
    let eps = total * 1e-5;
    stations.sort_by(f32::total_cmp);
    let mut out = Vec::with_capacity(m + stations.len());
    let mut si = 0;
    let mut last = f32::MIN;
    for i in 0..m {
        out.push(pos[i]);
        let seg_end = if i + 1 < m {
            d[i + 1]
        } else if closed {
            total
        } else {
            continue;
        };
        while si < stations.len() && stations[si] <= d[i] + eps {
            si += 1;
        }
        let (a, b) = (pos[i], pos[(i + 1) % m]);
        while si < stations.len() && stations[si] < seg_end - eps {
            let s = stations[si];
            si += 1;
            if s - last < eps {
                continue;
            }
            last = s;
            out.push(a.lerp(b, (s - d[i]) / (seg_end - d[i])));
        }
    }
    *pos = out;
}

/// Everything per ring that placing a profile point needs.
struct SweepRings {
    pos: Vec<Vec3>,
    tans: Vec<Vec3>,
    frames: Vec<(Vec3, Vec3)>,
    bends: Vec<Option<(Vec3, f32)>>,
    scale: Vec<Vec2>,
    spin: Vec<(f32, f32)>,
    ratio: Vec<f32>,
    color: Vec<[f32; 4]>,
    pts: Vec<Vec2>,
    edge_n: Vec<Vec2>,
    u: Vec<f32>,
    outs: Vec<Vec<u32>>,
    ins: Vec<Vec<u32>>,
}

impl SweepRings {
    fn rotate(&self, i: usize, v: Vec2) -> Vec2 {
        let (s, c) = self.spin[i];
        Vec2::new(v.x * c - v.y * s, v.x * s + v.y * c)
    }

    /// World position of profile point `p` on ring `i`.
    fn place(&self, i: usize, p: Vec2) -> Vec3 {
        let q = self.rotate(i, p * self.scale[i]);
        let (right, up) = self.frames[i];
        let mut off = right * q.x + up * q.y;
        if let Some((b, factor)) = self.bends[i] {
            off += b * (off.dot(b) * (factor - 1.0));
        }
        self.pos[i] + off
    }

    /// World normal for the profile normal `n2` on ring `i`.
    fn normal(&self, i: usize, n2: Vec2) -> Vec3 {
        let q = self.rotate(i, (n2 / self.scale[i]).normalize_or(n2));
        let (right, up) = self.frames[i];
        (right * q.x + up * q.y).normalize_or(up)
    }

    /// Vertex of ring `i` on profile edge `e` at height `y`, reusing the
    /// ring's own vertex when `y` is an end of the edge.
    fn cut_vertex(&self, data: &mut MeshData, i: usize, e: usize, y: f32) -> u32 {
        let np = self.pts.len();
        let e1 = (e + 1) % np;
        let (a, b) = (self.pts[e], self.pts[e1]);
        let lam = ((y - a.y) / (b.y - a.y)).clamp(0.0, 1.0);
        if lam <= 1e-4 {
            return self.outs[i][e];
        }
        if lam >= 1.0 - 1e-4 {
            return self.ins[i][e1];
        }
        let (start, end) = (self.outs[i][e] as usize, self.ins[i][e1] as usize);
        let normal = data.normals[start]
            .lerp(data.normals[end], lam)
            .normalize_or(data.normals[start]);
        let u_end = if e1 == 0 { 1.0 } else { self.u[e1] };
        let u = self.u[e] + (u_end - self.u[e]) * lam;
        data.push(
            self.place(i, a.lerp(b, lam)),
            normal,
            Vec2::new(u, self.ratio[i]),
            self.color[i],
        )
    }

    /// The outermost profile points at height `y`.
    fn cross_points(&self, y: f32) -> Option<(Vec2, Vec2)> {
        let n = self.pts.len();
        let mut hits: Vec<Vec2> = Vec::new();
        for k in 0..n {
            let (a, b) = (self.pts[k], self.pts[(k + 1) % n]);
            if (a.y <= y && b.y >= y) || (a.y >= y && b.y <= y) {
                if (b.y - a.y).abs() < 1e-9 {
                    hits.push(a);
                    hits.push(b);
                } else {
                    hits.push(a.lerp(b, (y - a.y) / (b.y - a.y)));
                }
            }
        }
        let left = hits.iter().copied().min_by(|p, q| p.x.total_cmp(&q.x))?;
        let right = hits.iter().copied().max_by(|p, q| p.x.total_cmp(&q.x))?;
        Some((left, right))
    }
}

/// Triangle fan over a convex polygon, skipping zero-area triangles.
fn fan(data: &mut MeshData, poly: &[u32]) {
    for k in 1..poly.len().saturating_sub(1) {
        let (a, b, c) = (poly[0], poly[k], poly[k + 1]);
        let (pa, pb, pc) = (
            data.positions[a as usize],
            data.positions[b as usize],
            data.positions[c as usize],
        );
        if (pb - pa).cross(pc - pa).length_squared() > 1e-14 {
            data.tri(a, b, c);
        }
    }
}

/// A flat quad facing `toward` (the side of the solid's free air).
fn flat_quad(data: &mut MeshData, corners: [Vec3; 4], toward: Vec3, color: [f32; 4]) {
    let mut n = (corners[2] - corners[0]).cross(corners[3] - corners[1]);
    if n.length_squared() < 1e-14 {
        return;
    }
    n = n.normalize();
    let mut c = corners;
    if n.dot(toward) < 0.0 {
        c.reverse();
        n = -n;
    }
    let uvs = [Vec2::ZERO, Vec2::X, Vec2::ONE, Vec2::Y];
    let base = data.positions.len() as u32;
    for k in 0..4 {
        data.push(c[k], n, uvs[k], color);
    }
    data.tri(base, base + 1, base + 2);
    data.tri(base, base + 2, base + 3);
}

/// Sweeps `profile` along `path` with parallel-transport frames and mitred
/// joints. See [`Shape::sweep`](crate::Shape::sweep) for the rules.
pub(crate) fn sweep_mesh(
    profile: &[Vec2],
    path: &[Vec3],
    given: Option<&[crate::shape::SweepFrame]>,
    o: &crate::shape::SweepOptions,
) -> Mesh {
    let empty = || MeshData::default().into_mesh(false);
    let Some(ring) = Ring::new(profile, None, true) else {
        bevy::log::warn!("sweep: the profile needs at least three distinct points; building an empty mesh");
        return empty();
    };

    // Distinct path points.
    let mut pts: Vec<Vec3> = Vec::with_capacity(path.len());
    let mut kept: Vec<usize> = Vec::with_capacity(path.len());
    for (k, &p) in path.iter().enumerate() {
        if pts.last().is_none_or(|l| l.distance_squared(p) > 1e-10) {
            pts.push(p);
            kept.push(k);
        }
    }
    let closed = o.closed;
    if closed {
        while pts.len() > 1 && pts[0].distance_squared(pts[pts.len() - 1]) <= 1e-10 {
            pts.pop();
            kept.pop();
        }
    }
    let n = pts.len();
    if n < if closed { 3 } else { 2 } {
        bevy::log::warn!("sweep: the path needs at least two distinct points (three when closed); building an empty mesh");
        return empty();
    }

    // Ring positions, split into pieces when something varies along the path.
    let tint_keys: Option<Vec<(f32, [f32; 4])>> = o
        .tint_by_length
        .as_ref()
        .filter(|t| !t.is_empty())
        .map(|t| t.iter().map(|(r, c)| (*r, c.to_linear().to_f32_array())).collect());
    let scale_keys = o.scale_along.as_deref().filter(|s| !s.is_empty());
    let roll_keys = o.roll_along.as_deref().filter(|r| !r.is_empty());
    // A closed sweep has no seam only if the twist is a whole number of turns
    // and the roll comes back to where it started: the nearest such twist is
    // used, and the roll's leftover is spread evenly along the loop.
    let twist = if closed { o.twist.round() } else { o.twist };
    let roll_drift = match roll_keys {
        Some(k) if closed => {
            let (a, b) = (
                sample_keys(k, 0.0, |a: f32, b: f32, t| a + (b - a) * t).unwrap_or(0.0),
                sample_keys(k, 1.0, |a: f32, b: f32, t| a + (b - a) * t).unwrap_or(0.0),
            );
            let d = b - a;
            d - (d / TAU).round() * TAU
        }
        _ => 0.0,
    };
    let varies = twist != 0.0 || roll_keys.is_some() || scale_keys.is_some() || tint_keys.is_some();
    let pieces = if varies {
        (o.detail.segments() / 4).max(1) as usize
    } else {
        1
    };
    let segs = if closed { n } else { n - 1 };
    let mut pos: Vec<Vec3> = Vec::with_capacity(segs * pieces + 1);
    for s in 0..segs {
        let (a, b) = (pts[s], pts[(s + 1) % n]);
        for k in 0..pieces {
            pos.push(a.lerp(b, k as f32 / pieces as f32));
        }
    }
    if !closed {
        pos.push(pts[n - 1]);
    }

    // Openings add rings wherever their outline or chains change slope.
    let mut levels: Vec<f32> = ring.pts.iter().map(|p| p.y).collect();
    levels.sort_by(f32::total_cmp);
    levels.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    let (ymin, ymax) = (levels[0], levels[levels.len() - 1]);
    let prepared = prepare_openings(&o.openings, path_length(&pos, closed), closed);
    let mut stations: Vec<f32> = prepared.iter().flat_map(|p| p.stations(&levels)).collect();
    insert_stations(&mut pos, closed, &mut stations);
    let m = pos.len();

    let dirs: Vec<Vec3> = (0..if closed { m } else { m - 1 })
        .map(|i| (pos[(i + 1) % m] - pos[i]).normalize())
        .collect();
    let mut dist = vec![0.0_f32; m];
    for i in 1..m {
        dist[i] = dist[i - 1] + pos[i].distance(pos[i - 1]);
    }
    let total = path_length(&pos, closed);
    let ratio: Vec<f32> = dist.iter().map(|d| d / total).collect();

    // Joint tangents (bisectors) and the bend axis with its mitre factor.
    let mut tans: Vec<Vec3> = Vec::with_capacity(m);
    let mut bends: Vec<Option<(Vec3, f32)>> = Vec::with_capacity(m);
    for i in 0..m {
        if !closed && (i == 0 || i == m - 1) {
            tans.push(if i == 0 { dirs[0] } else { dirs[m - 2] });
            bends.push(None);
            continue;
        }
        let prev = dirs[(i + dirs.len() - 1) % dirs.len()];
        let next = dirs[i % dirs.len()];
        let t = (prev + next).try_normalize().unwrap_or(next);
        let scale = (1.0 / t.dot(next).max(1.0 / SWEEP_MAX_MITRE)).min(SWEEP_MAX_MITRE);
        tans.push(t);
        bends.push((next - prev).try_normalize().map(|b| (b, scale)));
    }

    // Frames `(right, up)` with `right x up == forward`: the caller's, sampled
    // along the path, or parallel transport.
    let mut frames: Vec<(Vec3, Vec3)> = Vec::with_capacity(m);
    let mut frame_scale: Vec<Vec2> = vec![Vec2::ONE; m];
    if let Some(given) = given {
        let total_in = path_length(&pts, closed);
        let mut keys: Vec<(f32, (Vec3, Vec3, Vec2))> = Vec::with_capacity(pts.len() + 1);
        let mut d = 0.0;
        for (k, &idx) in kept.iter().enumerate() {
            if k > 0 {
                d += pts[k].distance(pts[k - 1]);
            }
            let f = &given[idx];
            keys.push((d / total_in, (f.right, f.up, f.scale)));
        }
        if closed {
            let first = keys[0].1;
            keys.push((1.0, first));
        }
        for i in 0..m {
            let (r, u, sc) = sample_keys(&keys, ratio[i], |a: (Vec3, Vec3, Vec2), b, t| {
                (a.0.lerp(b.0, t), a.1.lerp(b.1, t), a.2.lerp(b.2, t))
            })
            .unwrap_or((Vec3::X, Vec3::Y, Vec2::ONE));
            let tan = tans[i];
            let flat = |v: Vec3| (v - tan * v.dot(tan)).try_normalize();
            let (right, up) = match (flat(r), flat(u)) {
                (Some(r), Some(u)) if r.cross(u).dot(tan) >= 0.0 => (r, tan.cross(r)),
                // Left-handed frame: keep `up`, so a road never turns over.
                (_, Some(u)) => (u.cross(tan), u),
                (Some(r), None) => (r, tan.cross(r)),
                (None, None) => {
                    let r = tan.any_orthonormal_vector();
                    (r, tan.cross(r))
                }
            };
            frames.push((right, up));
            frame_scale[i] = sc;
        }
    } else {
        let up_hint = o.up.try_normalize().unwrap_or(Vec3::Y);
        let mut right = up_hint.cross(tans[0]);
        if right.length_squared() < 1e-6 {
            right = (if tans[0].y.abs() < 0.9 { Vec3::Y } else { Vec3::X }).cross(tans[0]);
        }
        let mut right = right.normalize();
        for i in 0..m {
            if i > 0 {
                right = Quat::from_rotation_arc(tans[i - 1], tans[i]) * right;
            }
            right = (right - tans[i] * right.dot(tans[i]))
                .try_normalize()
                .unwrap_or_else(|| tans[i].any_orthonormal_vector());
            frames.push((right, tans[i].cross(right)));
        }
        if closed {
            // Parallel transport around a loop returns twisted; spread the
            // difference over the loop so the last ring meets the first.
            let end = Quat::from_rotation_arc(tans[m - 1], tans[0]) * frames[m - 1].0;
            let start = frames[0].0;
            let angle = end.cross(start).dot(tans[0]).atan2(end.dot(start));
            for i in 0..m {
                let rot = Quat::from_axis_angle(tans[i], angle * ratio[i]);
                frames[i] = (rot * frames[i].0, rot * frames[i].1);
            }
        }
    }

    // Profile outline data.
    let np = ring.len();
    let mut u = vec![0.0_f32; np];
    for k in 1..np {
        u[k] = u[k - 1] + ring.pts[k].distance(ring.pts[k - 1]);
    }
    let perimeter = u[np - 1] + ring.pts[np - 1].distance(ring.pts[0]);
    for v in &mut u {
        *v /= perimeter;
    }

    let mut rings = SweepRings {
        pos,
        tans,
        frames,
        bends,
        scale: Vec::with_capacity(m),
        spin: Vec::with_capacity(m),
        color: Vec::with_capacity(m),
        ratio,
        pts: ring.pts.clone(),
        edge_n: (0..np).map(|k| ring.edge_normal(k)).collect(),
        u,
        outs: Vec::with_capacity(m),
        ins: Vec::with_capacity(m),
    };
    for i in 0..m {
        let r = rings.ratio[i];
        rings.scale.push(
            (scale_keys
                .and_then(|k| sample_keys(k, r, |a: Vec2, b: Vec2, t| a.lerp(b, t)))
                .unwrap_or(Vec2::ONE)
                * frame_scale[i])
                .max(Vec2::splat(1e-4)),
        );
        let roll = roll_keys
            .and_then(|k| sample_keys(k, r, |a: f32, b: f32, t| a + (b - a) * t))
            .unwrap_or(0.0)
            - roll_drift * r;
        rings.spin.push((TAU * twist * r + roll).sin_cos());
        rings.color.push(
            tint_keys
                .as_deref()
                .and_then(|k| {
                    sample_keys(k, r, |a: [f32; 4], b: [f32; 4], t| {
                        std::array::from_fn(|j| a[j] + (b[j] - a[j]) * t)
                    })
                })
                .unwrap_or(WHITE),
        );
    }

    let mut data = MeshData::default();
    for i in 0..m {
        let mut out = Vec::with_capacity(np);
        let mut inn = Vec::with_capacity(np);
        for k in 0..np {
            let position = rings.place(i, rings.pts[k]);
            let (prev_n, next_n) = (rings.edge_n[(k + np - 1) % np], rings.edge_n[k]);
            let split = prev_n.dot(next_n) <= SWEEP_SMOOTH_COS;
            let smooth = (prev_n + next_n).normalize_or(next_n);
            let (r, color) = (rings.ratio[i], rings.color[i]);
            if split || k == 0 {
                let (in_n, out_n) = if split { (prev_n, next_n) } else { (smooth, smooth) };
                let u_in = if k == 0 { 1.0 } else { rings.u[k] };
                inn.push(data.push(position, rings.normal(i, in_n), Vec2::new(u_in, r), color));
                out.push(data.push(position, rings.normal(i, out_n), Vec2::new(rings.u[k], r), color));
            } else {
                let v = data.push(position, rings.normal(i, smooth), Vec2::new(rings.u[k], r), color);
                inn.push(v);
                out.push(v);
            }
        }
        rings.outs.push(out);
        rings.ins.push(inn);
    }

    // Strips between consecutive rings, one quad per profile edge. Cells inside
    // an opening are rebuilt around the hole.
    let eps_s = total * 1e-4;
    for i in 0..if closed { m } else { m - 1 } {
        let j = (i + 1) % m;
        let (si, sj) = (dist[i], if j == 0 { total } else { dist[j] });
        let sm = 0.5 * (si + sj);
        let cut = prepared.iter().position(|op| sm > op.s0 && sm < op.s1);
        for e in 0..np {
            let e1 = (e + 1) % np;
            let (pa, pb) = (rings.pts[e], rings.pts[e1]);
            let horizontal = (pb.y - pa.y).abs() < 1e-6;
            let (ylo, yhi) = (pa.y.min(pb.y), pa.y.max(pb.y));
            let upward = pb.y > pa.y;
            // Region of the strip between heights `p` and `q` on each ring.
            let region = |data: &mut MeshData, p_i: f32, q_i: f32, p_j: f32, q_j: f32| {
                if q_i - p_i < SWEEP_TOL && q_j - p_j < SWEEP_TOL {
                    return;
                }
                let (a_i, b_i) = if upward { (p_i, q_i) } else { (q_i, p_i) };
                let (a_j, b_j) = if upward { (p_j, q_j) } else { (q_j, p_j) };
                let poly = [
                    rings.cut_vertex(data, i, e, a_i),
                    rings.cut_vertex(data, i, e, b_i),
                    rings.cut_vertex(data, j, e, b_j),
                    rings.cut_vertex(data, j, e, a_j),
                ];
                fan(data, &poly);
            };
            match cut {
                Some(k) => {
                    let op = &prepared[k];
                    if horizontal {
                        // A flat strip is either wholly inside the hole (gone)
                        // or untouched.
                        let (l, h) = op.span(sm);
                        if pa.y < l - SWEEP_TOL || pa.y > h + SWEEP_TOL {
                            let (a, b) = (rings.outs[i][e], rings.ins[i][e1]);
                            let (c, d) = (rings.outs[j][e], rings.ins[j][e1]);
                            data.tri(a, b, c);
                            data.tri(b, d, c);
                        }
                        continue;
                    }
                    let (l_i, h_i) = op.span(si);
                    let (l_j, h_j) = op.span(sj);
                    let cl = |v: f32| v.clamp(ylo, yhi);
                    region(&mut data, ylo, cl(l_i), ylo, cl(l_j));
                    region(&mut data, cl(h_i), yhi, cl(h_j), yhi);
                }
                None => {
                    // Rings next to an opening share their edge with the jamb,
                    // so the hole's corner heights join that edge.
                    let mut on_j: Vec<f32> = Vec::new();
                    let mut on_i: Vec<f32> = Vec::new();
                    if !horizontal {
                        for op in &prepared {
                            if (sj - op.s0).abs() <= eps_s {
                                let (l, h) = op.span(op.s0);
                                on_j.extend([l, h]);
                            }
                            if (si - op.s1).abs() <= eps_s {
                                let (l, h) = op.span(op.s1);
                                on_i.extend([l, h]);
                            }
                        }
                    }
                    let inside = |h: &f32| *h > ylo + SWEEP_TOL && *h < yhi - SWEEP_TOL;
                    on_j.retain(inside);
                    on_i.retain(inside);
                    let lam = |h: f32| (h - pa.y) / (pb.y - pa.y);
                    on_j.sort_by(|a, b| lam(*b).total_cmp(&lam(*a)));
                    on_i.sort_by(|a, b| lam(*a).total_cmp(&lam(*b)));
                    on_j.dedup_by(|a, b| (*a - *b).abs() < SWEEP_TOL);
                    on_i.dedup_by(|a, b| (*a - *b).abs() < SWEEP_TOL);
                    let (a, b) = (rings.outs[i][e], rings.ins[i][e1]);
                    let (c, d) = (rings.outs[j][e], rings.ins[j][e1]);
                    if on_i.is_empty() && on_j.is_empty() {
                        data.tri(a, b, c);
                        data.tri(b, d, c);
                    } else {
                        // Polygon a -> b -> d -> c, extras on the shared edge;
                        // fan from a corner that is not on that edge.
                        let mut poly = vec![a];
                        for h in &on_i {
                            poly.push(rings.cut_vertex(&mut data, i, e, *h));
                        }
                        poly.extend([b, d]);
                        for h in &on_j {
                            poly.push(rings.cut_vertex(&mut data, j, e, *h));
                        }
                        poly.push(c);
                        let start = if on_i.is_empty() { 0 } else { poly.len() - 2 };
                        poly.rotate_left(start);
                        fan(&mut data, &poly);
                    }
                }
            }
        }

        // Sill and lintel of the hole inside this cell.
        if let Some(k) = cut {
            let op = &prepared[k];
            let (l_i, h_i) = op.span(si);
            let (l_j, h_j) = op.span(sj);
            let clamp = |v: f32| v.clamp(ymin, ymax);
            for (a, b, toward) in [(l_i, l_j, 1.0), (h_i, h_j, -1.0)] {
                let (a, b) = (clamp(a), clamp(b));
                let at = |v: f32, edge: f32| (v - edge).abs() <= SWEEP_TOL;
                if (at(a, ymin) && at(b, ymin)) || (at(a, ymax) && at(b, ymax)) {
                    continue;
                }
                let (Some((l0, r0)), Some((l1, r1))) =
                    (rings.cross_points(a), rings.cross_points(b))
                else {
                    continue;
                };
                flat_quad(
                    &mut data,
                    [
                        rings.place(i, l0),
                        rings.place(i, r0),
                        rings.place(j, r1),
                        rings.place(j, l1),
                    ],
                    rings.frames[i].1 * toward,
                    rings.color[i],
                );
            }
        }
    }

    // Jambs: the hole's two ends.
    for op in &prepared {
        for (s, sign) in [(op.s0, 1.0), (op.s1, -1.0)] {
            let Some(r) = (0..m).find(|&r| (dist[r] - s).abs() <= eps_s) else {
                continue;
            };
            let (l, h) = op.span(s);
            let (l, h) = (l.clamp(ymin, ymax), h.clamp(ymin, ymax));
            if h - l <= SWEEP_TOL {
                continue;
            }
            let (Some((l0, r0)), Some((l1, r1))) = (rings.cross_points(l), rings.cross_points(h))
            else {
                continue;
            };
            flat_quad(
                &mut data,
                [
                    rings.place(r, l0),
                    rings.place(r, r0),
                    rings.place(r, r1),
                    rings.place(r, l1),
                ],
                rings.tans[r] * sign,
                rings.color[r],
            );
        }
    }

    // Flat caps over the profile polygon.
    if !closed && o.caps {
        let tris = ear_clip(&ring.pts);
        let (lo, hi) = ring
            .pts
            .iter()
            .fold((Vec2::splat(f32::MAX), Vec2::splat(f32::MIN)), |(l, h), p| {
                (l.min(*p), h.max(*p))
            });
        let size = (hi - lo).max(Vec2::splat(1e-6));
        for (i, start) in [(0, true), (m - 1, false)] {
            let normal = if start { -rings.tans[i] } else { rings.tans[i] };
            let base = data.positions.len() as u32;
            for k in 0..np {
                data.push(
                    rings.place(i, ring.pts[k]),
                    normal,
                    (ring.pts[k] - lo) / size,
                    rings.color[i],
                );
            }
            for t in tris.chunks_exact(3) {
                if start {
                    data.tri(base + t[0], base + t[2], base + t[1]);
                } else {
                    data.tri(base + t[0], base + t[1], base + t[2]);
                }
            }
        }
    }

    data.into_mesh(tint_keys.is_some())
}

// --- 2D ring helpers ---

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

// --- Triangulation with holes ---

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

// --- Polygon / extruded polygon ---

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


// --- Lathe ---

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

// --- Flat curves ---

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

// --- Compound merge ---

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
