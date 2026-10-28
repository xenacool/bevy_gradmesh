// BevyGradmesh unified shader — material extension + instanced rendering.
//
// This single .wesl file contains all shader entry points:
//
// 1. **Material extension** (fragment shader) — extends Bevy's StandardMaterial
//    with dual-directional gradient shading, OKLAB color interpolation, dash
//    patterns, and blend modes. Uses `@if(PREPASS_PIPELINE)` for deferred
//    rendering support.
//
// 2. **Instanced rendering** (vertex + fragment shaders) — GPU-instanced path
//    that reads per-instance transforms and properties from vertex attributes,
//    supporting deformation and LOD dither cross-fade.
//
// Note: Float comparisons use epsilon thresholds. Boolean flags stored as 0.0/1.0
// are compared with > 0.5 to avoid precision issues.

import bevy_pbr::render::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
};

@if(PREPASS_PIPELINE)
import bevy_pbr::{
    prepass::io::{VertexOutput, FragmentOutput},
    deferred::functions::deferred_output,
};
@else
import bevy_pbr::render::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    view_transformations::position_world_to_clip,
};

// ─── OKLAB color space conversion ───

fn oklab_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lms_in = vec3<f32>(
        c.x + 0.3963377774 * c.y + 0.2158037573 * c.z,
        c.x - 0.1055613458 * c.y - 0.0638541728 * c.z,
        c.x - 0.0894841775 * c.y - 1.2914855480 * c.z,
    );
    let lms_cubed = lms_in * lms_in * lms_in;
    return vec3<f32>(
        4.0767416621 * lms_cubed.x - 3.3077115913 * lms_cubed.y + 0.2309699292 * lms_cubed.z,
        -1.2684380046 * lms_cubed.x + 2.6097574011 * lms_cubed.y - 0.3413193965 * lms_cubed.z,
        -0.0041960863 * lms_cubed.x - 0.7034186147 * lms_cubed.y + 1.7076147010 * lms_cubed.z,
    );
}

fn linear_to_oklab(c: vec3<f32>) -> vec3<f32> {
    let lms_in = vec3<f32>(
        0.4122214708 * c.x + 0.5363325363 * c.y + 0.0614459929 * c.z,
        0.2119034982 * c.x + 0.6806995451 * c.y + 0.1073969566 * c.z,
        0.0883024619 * c.x + 0.2817188376 * c.y + 0.6299787005 * c.z,
    );
    let lms_cbrt = pow(lms_in, vec3<f32>(1.0 / 3.0));
    return vec3<f32>(
        0.2104542553 * lms_cbrt.x + 0.7936177850 * lms_cbrt.y - 0.0040720468 * lms_cbrt.z,
        1.9779984951 * lms_cbrt.x - 2.4285922050 * lms_cbrt.y + 0.4505937099 * lms_cbrt.z,
        0.0259040371 * lms_cbrt.x + 0.7827717662 * lms_cbrt.y - 0.8086757660 * lms_cbrt.z,
    );
}

fn oklab_mix(c1: vec3<f32>, c2: vec3<f32>, t: f32) -> vec3<f32> {
    let lab1 = linear_to_oklab(c1);
    let lab2 = linear_to_oklab(c2);
    let lab = mix(lab1, lab2, t);
    return oklab_to_linear(lab);
}

// ─── Dash pattern computation ───

/// Epsilon for float comparisons — avoids precision issues with near-1.0 values.
const EPSILON: f32 = 1e-4;

fn dash_segment_mask(progress: f32, across: f32, dash: vec4<f32>) -> f32 {
    let total = dash.x;
    if (total <= 0.0) { return 1.0; }

    let visible_ratio = 1.0 - clamp(dash.z, 0.0, 1.0);
    if (visible_ratio >= 1.0 - EPSILON) { return 1.0; }

    let type_code = dash.y;
    let modifier = clamp((type_code - 0.1) / 0.075, -1.0, 1.0);
    let type_id = i32(type_code);

    var coordinate = progress * total + dash.w + (1.0 - visible_ratio) * 0.5;

    if (type_id < 1) {
        coordinate += modifier * across * visible_ratio * 0.25;
    } else if (type_id < 2) {
        coordinate += abs(across) * modifier;
    }

    let phase = fract(coordinate);
    let half_visible = visible_ratio * 0.5;
    var signed_distance = half_visible - abs(phase - 0.5);

    if (type_id > 1) {
        let profile = sqrt(max(1.0 - across * across, 0.0));
        let cap = clamp(0.1, 0.0, half_visible);
        signed_distance = half_visible - cap * (1.0 - profile) - abs(phase - 0.5);
    }

    return clamp(signed_distance * 2.0 + 0.5, 0.0, 1.0);
}

// ─── Blend mode computation ───

fn apply_blend_mode(
    base_color: vec3<f32>,
    fragment_position: vec2<f32>,
    flags: vec4<f32>,
    color_w: f32,
) -> vec4<f32> {
    let blend_mode = i32(flags.y);
    var color = base_color;
    var alpha = color_w;

    if (blend_mode == 0) {
        alpha = 1.0;
    } else if (blend_mode == 1) {
        alpha = color_w;
    } else if (blend_mode == 2) {
        color = color * (1.0 + flags.z * 0.12);
        alpha = 1.0;
    } else if (blend_mode == 3) {
        color = color * 0.5;
        alpha = 1.0;
    } else if (blend_mode == 4) {
        color = color * 1.5;
        alpha = 1.0;
    } else if (blend_mode == 5) {
        let p = vec2<i32>(i32(fragment_position.x) % 16, i32(fragment_position.y) % 16);
        let rank = p.x * 2 + p.y * 3 - p.x * p.y * 4;
        let threshold = f32(rank + 1) / 257.0;
        if (alpha < threshold) {
            discard;
        }
        alpha = 1.0;
    }

    return vec4<f32>(color, alpha);
}

// ─── Perimeter progress for dash patterns on 2D shapes ───

fn compute_perimeter_progress(uv: vec2<f32>) -> f32 {
    let p = uv * 2.0 - 1.0;
    if (abs(p.x) < abs(p.y)) {
        return uv.x;
    }
    return uv.y;
}

fn compute_across(uv: vec2<f32>) -> f32 {
    let p = uv * 2.0 - 1.0;
    let dist_from_edge = min(abs(p.x), abs(p.y));
    return clamp((1.0 - dist_from_edge) * 2.0 - 1.0, -1.0, 1.0);
}

// ─── LOD Dither Cross-Fade ───

// Bayer 4x4 dither matrix for LOD cross-fade
const bayer_dither_4x4 = array<array<f32, 4>, 4>(
    array<f32, 4>(0.0/16.0,  8.0/16.0,  2.0/16.0, 10.0/16.0),
    array<f32, 4>(12.0/16.0, 4.0/16.0, 14.0/16.0,  6.0/16.0),
    array<f32, 4>( 3.0/16.0, 11.0/16.0, 1.0/16.0,  9.0/16.0),
    array<f32, 4>(15.0/16.0,  7.0/16.0, 13.0/16.0, 5.0/16.0),
);

/// Apply Bayer dither cross-fade between two LOD levels.
/// Returns 1.0 if the fragment should be visible, 0.0 if it should be discarded.
fn lod_dither_mask(frag_coord: vec2<f32>, transition_weight: f32) -> f32 {
    if (transition_weight <= 0.0 || transition_weight >= 1.0) {
        return 1.0;
    }
    
    let x = i32(frag_coord.x) % 4;
    let y = i32(frag_coord.y) % 4;
    let dither_value = bayer_dither_4x4[y][x];
    
    // If the dither value is less than the transition weight, show this fragment
    // This creates a smooth visual transition between LOD levels
    return select(1.0, 0.0, dither_value > transition_weight);
}

// ─── Vertex deformation ───

/// Apply per-instance deformation to a local-space vertex position.
/// Deformation params: x=type, y=amplitude, z=frequency, w=phase.
/// Returns the displaced local-space position.
fn apply_deformation(
    local_pos: vec3<f32>,
    local_normal: vec3<f32>,
    deformation: vec4<f32>,
    time: f32,
) -> vec3<f32> {
    let deform_type = deformation.x;
    let amplitude = deformation.y;
    let frequency = deformation.z;
    let phase = deformation.w;

    // No deformation
    if (deform_type <= 0.5) {
        return local_pos;
    }

    let t = sin(time * frequency * 6.28318530718 + phase * 6.28318530718);

    if (deform_type < 1.5) {
        // Wobble: sinusoidal displacement along normal
        return local_pos + t * amplitude * local_normal;
    } else if (deform_type < 2.5) {
        // Pulse: uniform scale oscillation
        return local_pos * (1.0 + t * amplitude);
    } else {
        // Morph: displacement toward/away from center based on normal
        return local_pos + t * amplitude * local_normal;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Material extension — fragment shader
// ═══════════════════════════════════════════════════════════════════════════════

/// GPU uniform for GradMesh material extension.
///
/// 6 Vec4 fields = 96 bytes total, bound at slot 100.
struct GradMeshUniform {
    color: vec4<f32>,
    secondary_color: vec4<f32>,
    direction: vec4<f32>,
    flags: vec4<f32>,
    dash: vec4<f32>,
    world_center: vec4<f32>,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(100)
var<uniform> gradmesh: GradMeshUniform;

/// Compute gradient progress for the material extension path.
/// Computes gradient in object space relative to each shape's center,
/// normalized by the shape's half-extent along the gradient direction.
fn compute_gradient_progress_material(in: VertexOutput) -> f32 {
    let color_mode = i32(gradmesh.flags.x);
    let dir = gradmesh.direction.xyz;

    if (color_mode == 0) {
        return 0.0;
    }

    // Object-space position: world position minus the shape's world center.
    let object_pos = in.world_position.xyz - gradmesh.world_center.xyz;

    if (color_mode == 1) {
        // Directional gradient: project object-space position onto the gradient
        // direction, normalized by the shape's half-extent (0.5 for unit shapes).
        let normalized_dir = normalize(dir);
        let half_extent = 0.5;
        let half_range = max(abs(dot(normalized_dir, vec3<f32>(half_extent, half_extent, half_extent))), 0.00001);
        let progress = dot(object_pos, normalized_dir) / (2.0 * half_range) + 0.5;
        return clamp(progress, 0.0, 1.0);
    } else if (color_mode == 2) {
        // Radial gradient: distance from shape center, normalized by half-extent.
        let dist = length(object_pos);
        return clamp(dist / 0.70710678, 0.0, 1.0);
    }

    return 0.5;
}

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    let progress = compute_gradient_progress_material(in);

    var base_color = gradmesh.color.rgb;
    if (gradmesh.flags.x > 0.5) {
        if (gradmesh.flags.w > 0.5) {
            base_color = oklab_mix(gradmesh.color.rgb, gradmesh.secondary_color.rgb, progress);
        } else {
            base_color = mix(gradmesh.color.rgb, gradmesh.secondary_color.rgb, progress);
        }
    }

    // Apply dash pattern if enabled
    let dash = gradmesh.dash;
    if (dash.x > 0.0) {
        let uv = in.uv;
        let perimeter_progress = compute_perimeter_progress(uv);
        let across = compute_across(uv);
        let dash_mask = dash_segment_mask(perimeter_progress, across, dash);
        base_color = mix(base_color, base_color * 0.0, 1.0 - dash_mask);
    }

    pbr_input.material.base_color = vec4<f32>(base_color, pbr_input.material.base_color.a);

    pbr_input.material.base_color = alpha_discard(
        pbr_input.material.flags,
        pbr_input.material.alpha_cutoff,
        pbr_input.material.base_color
    );

    var out: FragmentOutput;
@if(PREPASS_PIPELINE) {
    out = deferred_output(in, pbr_input);
} @else {
    out.color = apply_pbr_lighting(pbr_input);

    let blended = apply_blend_mode(out.color.rgb, in.position.xy, gradmesh.flags, gradmesh.color.w);
    out.color = blended;

    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
}

    return out;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Instanced rendering — vertex + fragment shaders
// ═══════════════════════════════════════════════════════════════════════════════

/// Per-instance gradmesh properties: 6×vec4 (color, secondary, direction, flags, dash, deformation).
struct InstanceProperties {
    color: vec4<f32>,
    secondary_color: vec4<f32>,
    direction: vec4<f32>,
    flags: vec4<f32>,
    dash: vec4<f32>,
    deformation: vec4<f32>,
}

// ─── Vertex input ───

struct Vertex {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    // Per-instance attributes
    @location(3) instance_col0: vec4<f32>,
    @location(4) instance_col1: vec4<f32>,
    @location(5) instance_col2: vec4<f32>,
    @location(6) instance_col3: vec4<f32>,
    @location(7) instance_color: vec4<f32>,
    @location(8) instance_secondary_color: vec4<f32>,
    @location(9) instance_direction: vec4<f32>,
    @location(10) instance_flags: vec4<f32>,
    @location(11) instance_dash: vec4<f32>,
    @location(12) instance_deformation: vec4<f32>,
}

/// Compute gradient progress for the instanced path.
fn compute_gradient_progress_instanced(world_pos: vec3<f32>, direction: vec4<f32>, color_mode: i32) -> f32 {
    let dir = direction.xyz;

    if (color_mode == 0) {
        return 0.0;
    } else if (color_mode == 1) {
        let progress = dot(world_pos, normalize(dir));
        return clamp(progress * 0.1 + 0.5, 0.0, 1.0);
    } else if (color_mode == 2) {
        let dist = length(world_pos);
        return clamp(dist * 0.5, 0.0, 1.0);
    }

    return 0.5;
}

@vertex
fn vertex(input: Vertex) -> VertexOutput {
    // Reconstruct instance transform matrix from per-instance attributes
    let instanceMatrix = mat4x4<f32>(
        input.instance_col0,
        input.instance_col1,
        input.instance_col2,
        input.instance_col3,
    );

    // Transform normal (use upper-left 3x3 of instance matrix)
    let normalMatrix = mat3x3<f32>(
        input.instance_col0.xyz,
        input.instance_col1.xyz,
        input.instance_col2.xyz,
    );
    let worldNormal = normalize(normalMatrix * input.normal);

    // Apply deformation in local space before world transform
    let localPos = apply_deformation(
        input.position.xyz,
        input.normal,
        input.instance_deformation,
        0.0 // Time will be passed via uniform in the future
    );

    // Transform deformed vertex position by instance matrix
    let worldPosition = instanceMatrix * vec4<f32>(localPos, 1.0);

    var out: VertexOutput;
    out.world_position = worldPosition;
    out.position = position_world_to_clip(worldPosition.xyz);
    out.world_normal = worldNormal;
    out.uv = input.uv;
    return out;
}

@fragment
fn instanced_fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    // The per-instance color/gradient is encoded in the material's base_color for now.
    // The full instanced path will read from storage buffers instead.

    pbr_input.material.base_color = alpha_discard(
        pbr_input.material.flags,
        pbr_input.material.alpha_cutoff,
        pbr_input.material.base_color
    );

    var out: FragmentOutput;
@if(PREPASS_PIPELINE) {
    out = deferred_output(in, pbr_input);
} @else {
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
}

    return out;
}
