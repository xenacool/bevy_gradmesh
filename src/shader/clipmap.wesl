// GPU geometry clipmap terrain.
//
// One square grid mesh is drawn once per level. The level (0 = finest) comes
// from the entity's `MeshTag`. Every level owns a layer of a toroidal height
// texture: `Rg32Float`, red = height, green = hole flag (1 = hole).
//
// Vertex stage: reads the height of its grid node, morphs it towards the
// height the next coarser level has at the same spot (geomorphing), and builds
// the normal the same way. The morph weight reaches 1 one cell inside the edge
// of the ring, so the edge lies exactly on the coarser surface and no crack
// opens.
//
// Fragment stage: discards holes, and discards the part of a ring that the
// next finer ring already covers. It works everything out from the world
// position and the ring's tag, because the depth and shadow passes carry no
// custom varyings.

import bevy_pbr::render::{
    mesh_functions,
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
    view_transformations::position_world_to_clip,
};

@if(PREPASS_PIPELINE)
import bevy_pbr::{
    prepass::io::{Vertex, VertexOutput, FragmentOutput, decompress_vertex},
    deferred::functions::deferred_output,
    render::pbr_prepass_functions,
};
@else
import bevy_pbr::render::{
    forward_io::{Vertex, VertexOutput, FragmentOutput, decompress_vertex},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
};

struct ClipmapUniform {
    // The terrain's local-from-world transform.
    local_from_world: mat4x4<f32>,
    // x: level-0 cell size, y: grid cells per ring side, z: texture size, w: level count.
    params: vec4<f32>,
    // x: fraction of the ring that morphs.
    morph: vec4<f32>,
    // Per level: (centre texel x, centre texel z, camera - centre x, camera - centre z),
    // all in that level's texel units.
    levels: array<vec4<f32>, 8>,
    // Height-band colours (linear rgba), lowest first.
    bands: array<vec4<f32>, 4>,
    // Height each band starts at; unused slots hold a huge value.
    band_heights: vec4<f32>,
    // x: number of bands (0 = use the base colour).
    band_info: vec4<f32>,
}

@group(3) @binding(100)
var<uniform> clipmap: ClipmapUniform;
@group(3) @binding(101)
var heights: texture_2d_array<f32>;

fn wrap_i(i: i32, n: i32) -> i32 {
    return ((i % n) + n) % n;
}

fn level_scale(level: i32) -> f32 {
    return clipmap.params.x * exp2(f32(level));
}

fn load_node(level: i32, t: vec2<i32>) -> vec2<f32> {
    let n = i32(clipmap.params.z);
    return textureLoad(heights, vec2<i32>(wrap_i(t.x, n), wrap_i(t.y, n)), level, 0).xy;
}

/// (height, hole) of a node. A hole node has no height of its own, so it
/// borrows the mean of its neighbours that are not holes; otherwise the land
/// beside a hole would slope towards nothing.
fn node_height(level: i32, t: vec2<i32>) -> vec2<f32> {
    let c = load_node(level, t);
    if (c.y < 0.5) {
        return c;
    }
    var sum = 0.0;
    var count = 0.0;
    let a = load_node(level, t + vec2<i32>(-1, 0));
    let b = load_node(level, t + vec2<i32>(1, 0));
    let d = load_node(level, t + vec2<i32>(0, -1));
    let e = load_node(level, t + vec2<i32>(0, 1));
    if (a.y < 0.5) { sum += a.x; count += 1.0; }
    if (b.y < 0.5) { sum += b.x; count += 1.0; }
    if (d.y < 0.5) { sum += d.x; count += 1.0; }
    if (e.y < 0.5) { sum += e.x; count += 1.0; }
    return vec2<f32>(select(c.x, sum / max(count, 1.0), count > 0.0), 1.0);
}

/// Bilinear height at a fractional texel position of `level`.
fn sample_bilinear(level: i32, f: vec2<f32>) -> f32 {
    let base = floor(f);
    let w = f - base;
    let t = vec2<i32>(base);
    let n00 = node_height(level, t).x;
    let n10 = node_height(level, t + vec2<i32>(1, 0)).x;
    let n01 = node_height(level, t + vec2<i32>(0, 1)).x;
    let n11 = node_height(level, t + vec2<i32>(1, 1)).x;
    return mix(mix(n00, n10, w.x), mix(n01, n11, w.x), w.y);
}

@vertex
fn vertex(vertex_in: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let instance = vertex_in.instance_index;
    let v = decompress_vertex(vertex_in, instance);

    let level = i32(mesh_functions::get_tag(instance));
    let level_count = i32(clipmap.params.w);
    let grid = clipmap.params.y;
    let lv = clipmap.levels[level];
    // Positions are integers; rounding undoes any compression of the mesh.
    let p = round(v.position.xz);
    let t = vec2<i32>(lv.xy) + vec2<i32>(p);
    let s = level_scale(level);

    var h = node_height(level, t).x;
    let hl = node_height(level, t + vec2<i32>(-1, 0)).x;
    let hr = node_height(level, t + vec2<i32>(1, 0)).x;
    let hd = node_height(level, t + vec2<i32>(0, -1)).x;
    let hu = node_height(level, t + vec2<i32>(0, 1)).x;
    var n = normalize(vec3<f32>(hl - hr, 2.0 * s, hd - hu));

    // Geomorph towards the next coarser level over the outer part of the ring.
    if (level < level_count - 1) {
        let radius = grid * 0.5;
        let finish = radius - 1.0;
        let band = clamp(clipmap.morph.x, 0.05, 0.9);
        let start = finish - band * radius;
        let d = max(abs(p.x - lv.z), abs(p.y - lv.w));
        let alpha = clamp((d - start) / (finish - start), 0.0, 1.0);
        if (alpha > 0.0) {
            let f = vec2<f32>(t) * 0.5;
            h = mix(h, sample_bilinear(level + 1, f), alpha);
            let cl = sample_bilinear(level + 1, f + vec2<f32>(-1.0, 0.0));
            let cr = sample_bilinear(level + 1, f + vec2<f32>(1.0, 0.0));
            let cd = sample_bilinear(level + 1, f + vec2<f32>(0.0, -1.0));
            let cu = sample_bilinear(level + 1, f + vec2<f32>(0.0, 1.0));
            let nc = normalize(vec3<f32>(cl - cr, 4.0 * s, cd - cu));
            n = normalize(mix(n, nc, alpha));
        }
    }

    let local = vec3<f32>(f32(t.x) * s, h, f32(t.y) * s);
    let world_from_local = mesh_functions::get_world_from_local(instance);
    out.world_position = mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(local, 1.0));
    out.position = position_world_to_clip(out.world_position.xyz);
@if(UNCLIPPED_DEPTH_ORTHO_EMULATION) {
    out.unclipped_depth = out.position.z;
    out.position.z = min(out.position.z, 1.0);
}
@if(!PREPASS_PIPELINE || NORMAL_PREPASS_OR_DEFERRED_PREPASS)
    out.world_normal = mesh_functions::mesh_normal_local_to_world(n, instance);
@if(PREPASS_PIPELINE && MOTION_VECTOR_PREPASS)
    out.previous_world_position = out.world_position;
@if(VERTEX_OUTPUT_INSTANCE_INDEX)
    out.instance_index = instance;
    return out;
}

/// Whether the fragment at `world_position` of the ring tagged `instance`
/// must not be drawn: it is a hole, or the next finer ring draws it.
fn cut_away(world_position: vec4<f32>, instance: u32) -> bool {
    let level = i32(mesh_functions::get_tag(instance));
    let local = (clipmap.local_from_world * world_position).xz;
    let g = local / level_scale(level);

    if (level > 0) {
        // The finer ring is centred on a grid point of this ring and reaches
        // grid / 4 of this ring's texels to every side.
        let c = clipmap.levels[level - 1].xy * 0.5;
        let q = abs(g - c);
        if (max(q.x, q.y) < clipmap.params.y * 0.25) {
            return true;
        }
    }

    // A cell with a hole at any corner has no triangles.
    let cell = vec2<i32>(floor(g));
    if (load_node(level, cell).y > 0.5
        || load_node(level, cell + vec2<i32>(1, 0)).y > 0.5
        || load_node(level, cell + vec2<i32>(0, 1)).y > 0.5
        || load_node(level, cell + vec2<i32>(1, 1)).y > 0.5) {
        return true;
    }
    return false;
}

/// Colour (linear rgba) of the height band a terrain-local `y` falls in.
fn height_band_color(y: f32) -> vec4<f32> {
    let h = clipmap.band_heights;
    var c = clipmap.bands[0];
    if (y >= h.y) {
        c = clipmap.bands[1];
    }
    if (y >= h.z) {
        c = clipmap.bands[2];
    }
    if (y >= h.w) {
        c = clipmap.bands[3];
    }
    return c;
}

@if(PREPASS_PIPELINE && !PREPASS_FRAGMENT)
@fragment
fn fragment(in: VertexOutput) {
    if (cut_away(in.world_position, in.instance_index)) {
        discard;
    }
}
@else
@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    if (cut_away(in.world_position, in.instance_index)) {
        discard;
    }

    var out: FragmentOutput;
@if(PREPASS_PIPELINE && !DEFERRED_PREPASS) {
@if(NORMAL_PREPASS)
    out.normal = vec4<f32>(normalize(in.world_normal) * 0.5 + vec3<f32>(0.5), 1.0);
@if(UNCLIPPED_DEPTH_ORTHO_EMULATION)
    out.frag_depth = in.unclipped_depth;
@if(MOTION_VECTOR_PREPASS)
    out.motion_vector = pbr_prepass_functions::calculate_motion_vector(in.world_position, in.previous_world_position);
} @else {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    var base_color = pbr_input.material.base_color;
    if (clipmap.band_info.x > 0.5) {
        let local_y = (clipmap.local_from_world * in.world_position).y;
        base_color = height_band_color(local_y);
    }
    pbr_input.material.base_color = alpha_discard(
        pbr_input.material.flags,
        pbr_input.material.alpha_cutoff,
        base_color,
    );

@if(PREPASS_PIPELINE) {
    out = deferred_output(in, pbr_input);
} @else {
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
}
}
    return out;
}
