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
    pbr_types,
    view_transformations::position_world_to_clip,
    mesh_view_bindings::{View, view},
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
    let type_id = i32(round((type_code - 0.1) / 0.3));
    // Center the modifier around each type's code so per-type variation is preserved.
    let center = 0.1 + f32(type_id) * 0.3;
    let modifier = clamp((type_code - center) / 0.075, -1.0, 1.0);

    var coordinate = progress * total + dash.w + (1.0 - visible_ratio) * 0.5;

    // Adaptive phase-per-across scaling: use fwidth to scale Chevron wave and Rounded cap
    // to screen resolution.
    let phase_per_across = fwidth(coordinate) / max(fwidth(across), 0.00001);

    if (type_id < 1) {
        coordinate += modifier * across * visible_ratio * 0.25;
    } else if (type_id < 2) {
        // Chevron: use abs(across) * phase_per_across * modifier for V-shaped end-caps
        coordinate += abs(across) * phase_per_across * modifier;
    }

    let phase = fract(coordinate);
    let half_visible = visible_ratio * 0.5;
    var signed_distance = half_visible - abs(phase - 0.5);

    if (type_id > 1) {
        // Rounded: use only signed_distance reshaping, no coordinate distortion
        let profile = sqrt(max(1.0 - across * across, 0.0));
        let cap = clamp(phase_per_across, 0.0, half_visible);
        signed_distance = half_visible - cap * (1.0 - profile) - abs(phase - 0.5);
    }

    // Adaptive AA: use screen-space derivative of signed_distance for anti-aliasing width
    let width = fwidth(signed_distance);
    return clamp(signed_distance / max(width, 0.00001) + 0.5, 0.0, 1.0);
}

// ─── Blend mode computation ───

/// 2×2 Bayer helper for 16×16 dither matrix.
fn bayer_2x2(px: i32, py: i32) -> i32 {
    return px * 2 + py * 3 - px * py * 4;
}

/// 16×16 Bayer dither threshold.
/// Returns a value in [0, 1) for alpha-to-coverage dithering.
fn dither_threshold(position: vec2<f32>) -> f32 {
    let px = i32(position.x) & 15;
    let py = i32(position.y) & 15;
    let rank = bayer_2x2(px & 1, py & 1) * 64 +
               bayer_2x2((px >> 1) & 1, (py >> 1) & 1) * 16 +
               bayer_2x2((px >> 2) & 1, (py >> 2) & 1) * 4 +
               bayer_2x2((px >> 3) & 1, (py >> 3) & 1);
    return f32(rank + 1) / 257.0;
}

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
        // Opaque
        alpha = 1.0;
    } else if (blend_mode == 1) {
        // Transparent
        alpha = color_w;
    } else if (blend_mode == 2) {
        // Additive
        color = color * (1.0 + flags.z * 0.12);
        alpha = 1.0;
    } else if (blend_mode == 3) {
        // Multiplicative
        color = color * 0.5;
        alpha = 1.0;
    } else if (blend_mode == 4) {
        // Screen (true screen blend: 1 - (1-src)*(1-dst), approximated as src*1.5)
        color = color * 1.5;
        alpha = 1.0;
    } else if (blend_mode == 5) {
        // Dither (alpha-to-coverage via 16×16 Bayer matrix)
        let threshold = dither_threshold(fragment_position);
        if (alpha < threshold) {
            discard;
        }
        alpha = 1.0;
    } else if (blend_mode == 6) {
        // Subtractive: subtract color from background
        color = color * 0.5;
        alpha = 1.0;
    } else if (blend_mode == 7) {
        // Lighten: max(src, dst) — approximated by boosting color
        color = color * 1.2;
        alpha = 1.0;
    } else if (blend_mode == 8) {
        // Color Dodge: dst / (1 - src) — approximated by stronger boost
        color = color * 1.8;
        alpha = 1.0;
    } else if (blend_mode == 9) {
        // Linear Burn: src + dst - 1 — approximated by darkening
        color = color * 0.3;
        alpha = 1.0;
    } else if (blend_mode == 10) {
        // Color Burn: 1 - (1 - dst) / src — approximated by strong darkening
        color = color * 0.2;
        alpha = 1.0;
    } else if (blend_mode == 11) {
        // Darken: min(src, dst) — approximated by reducing color
        color = color * 0.4;
        alpha = 1.0;
    }

    return vec4<f32>(color, alpha);
}

// ─── Perimeter progress for dash patterns on 2D shapes ───

/// Angular progress around center (for disc shapes).
/// Measures the angle from center and normalizes to [0, 1] over a full circle (start=0, span=TAU).
fn angular_progress(uv: vec2<f32>) -> f32 {
    let p = uv * 2.0 - 1.0;  // Map UV to [-1, 1]
    // Manual atan2: atan(y/x) with quadrant correction
    let eps = 0.00001;
    let a = atan(p.y / max(abs(p.x), eps));
    // a is in [-PI/2, PI/2]; add PI for x < 0 to get full [-PI, PI] range
    let angle = select(a + 3.14159265359, a, p.x >= 0.0);
    // angle is now in [-PI, PI]; normalize to [0, TAU) then to [0, 1)
    return fract(angle / 6.28318530718 + 0.5);
}

/// Rectangle perimeter progress (for cuboid/rectangle shapes).
/// Traces the actual perimeter centerline: right edge → top → left → bottom.
fn rectangle_perimeter_progress(uv: vec2<f32>) -> f32 {
    let p = uv * 2.0 - 1.0;  // Map UV to [-1, 1]
    let abs_p = abs(p);
    // Determine which edge the UV is closest to
    let dist_right = abs(p.x - 1.0);
    let dist_left = abs(p.x + 1.0);
    let dist_top = abs(p.y - 1.0);
    let dist_bottom = abs(p.y + 1.0);
    
    // Find minimum distance to any edge
    let min_h = min(dist_right, dist_left);
    let min_v = min(dist_top, dist_bottom);
    
    if (min_h < min_v) {
        // Closer to left/right edge — progress varies with Y
        if (dist_right < dist_left) {
            // Right edge: progress increases upward
            return fract(p.y * 0.5 + 0.75);
        } else {
            // Left edge: progress increases downward
            return fract(-p.y * 0.5 + 0.25);
        }
    } else {
        // Closer to top/bottom edge — progress varies with X
        if (dist_top < dist_bottom) {
            // Top edge: progress increases rightward
            return fract(p.x * 0.5 + 0.5);
        } else {
            // Bottom edge: progress increases leftward
            return fract(-p.x * 0.5);
        }
    }
}

/// Compute perimeter progress based on shape type.
/// shape_type: 0=generic (diagonal), 1=disc (angular), 2=rectangle (perimeter trace)
fn compute_perimeter_progress(uv: vec2<f32>, shape_type: f32) -> f32 {
    if (shape_type > 0.5 && shape_type < 1.5) {
        return angular_progress(uv);
    } else if (shape_type > 1.5) {
        return rectangle_perimeter_progress(uv);
    }
    // Generic: diagonal sweep
    return (uv.x + uv.y) * 0.5;
}

/// Compute across value based on shape type.
/// For discs: radial distance from center.
/// For rectangles: distance from nearest edge.
/// For generic: radial distance from center.
fn compute_across(uv: vec2<f32>, shape_type: f32) -> f32 {
    if (shape_type > 1.5) {
        // Rectangle: distance from nearest edge, normalized to [-1, 1]
        let p = uv * 2.0 - 1.0;
        let dist_from_edge = min(abs(p.x), abs(p.y));
        // Map: edge (dist=1) → +1, center (dist=0) → -1
        return clamp(dist_from_edge * 2.0 - 1.0, -1.0, 1.0);
    }
    // Disc/Generic: radial distance from center
    let center = uv - vec2<f32>(0.5, 0.5);
    let dist_from_center = length(center) * 2.0;
    return clamp(dist_from_center * 2.0 - 1.0, -1.0, 1.0);
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

@group(3) @binding(100)
var<uniform> gradmesh: GradMeshUniform;
// Note: MATERIAL_BIND_GROUP_INDEX = 3 in Bevy 0.20
// Using literal group number instead of constants::MATERIAL_BIND_GROUP
// because the shader def is injected at pipeline specialization time.

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

    // Apply dash pattern if enabled (requires UV attributes on the mesh)
@if(VERTEX_UVS_A) {
    let dash = gradmesh.dash;
    if (dash.x > 0.0) {
        let uv = in.uv;
        let shape_type = gradmesh.flags.z;
        let perimeter_progress = compute_perimeter_progress(uv, shape_type);
        let across = compute_across(uv, shape_type);
        let dash_mask = dash_segment_mask(perimeter_progress, across, dash);
        base_color = mix(base_color, base_color * 0.0, 1.0 - dash_mask);
    }
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

/// Per-instance gradmesh properties: 7×vec4 (color, secondary, direction, flags, dash, deformation, appearance).
struct InstanceProperties {
    color: vec4<f32>,
    secondary_color: vec4<f32>,
    direction: vec4<f32>,
    flags: vec4<f32>,
    dash: vec4<f32>,
    deformation: vec4<f32>,
    appearance: vec4<f32>,
}

/// Storage buffer for per-instance properties, accessible from the fragment shader.
@group(2) @binding(0)
var<storage, read> instanced_properties: array<InstanceProperties>;

/// Uniform block for instanced rendering: time and view center.
struct InstancedUniform {
    time: f32,
    _pad: vec3<f32>,
    view_center: vec4<f32>,
}
@group(2) @binding(1)
var<uniform> instanced_uniform: InstancedUniform;

// ─── Vertex input ───

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @if(VERTEX_UVS_A)
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
    @location(13) instance_appearance: vec4<f32>,
}

/// Custom vertex output for instanced rendering that includes per-instance color.
struct InstancedVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @if(VERTEX_UVS_A)
    @location(2) uv: vec2<f32>,
    // Per-instance color data passed to fragment shader
    @location(3) instance_color: vec4<f32>,
    @location(4) instance_secondary_color: vec4<f32>,
    @location(5) instance_direction: vec4<f32>,
    @location(6) instance_flags: vec4<f32>,
    @location(7) instance_dash: vec4<f32>,
    @location(8) instance_deformation: vec4<f32>,
    @location(9) instance_appearance: vec4<f32>,
}

/// Compute gradient progress for the instanced path.
fn compute_gradient_progress_instanced(world_pos: vec3<f32>, direction: vec4<f32>, color_mode: f32) -> f32 {
    let dir = direction.xyz;

    if (color_mode < 0.5) {
        return 0.0;
    } else if (color_mode < 1.5) {
        let progress = dot(world_pos, normalize(dir));
        return clamp(progress * 0.1 + 0.5, 0.0, 1.0);
    } else if (color_mode < 2.5) {
        let dist = length(world_pos);
        return clamp(dist * 0.5, 0.0, 1.0);
    }

    return 0.5;
}

/// Apply appearance animation to resolve animated color.
/// Returns the animated color based on time, speed, phase, and type.
fn apply_appearance_animation(
    base_color: vec4<f32>,
    secondary_color: vec4<f32>,
    appearance: vec4<f32>,
    time: f32,
) -> vec4<f32> {
    let speed = appearance.x;
    let phase = appearance.y;
    let anim_type = appearance.z;

    // No animation
    if (speed <= 0.0 || anim_type <= 0.5) {
        return base_color;
    }

    let t = sin(time * speed * 6.28318530718 + phase * 6.28318530718);
    let blend = t * 0.5 + 0.5; // Map sin(-1..1) to (0..1)

    if (anim_type < 1.5) {
        // ColorPulse: blend between primary and secondary
        return vec4<f32>(mix(base_color.rgb, secondary_color.rgb, blend), base_color.a);
    } else if (anim_type < 2.5) {
        // ColorCycle: shift hue by rotating RGB
        let shift = blend * 120.0 * 3.14159265359 / 180.0;
        let c = base_color.rgb;
        let p = 0.33333333333;
        let s = sin(shift);
        let cs = cos(shift);
        let rotated = vec3<f32>(
            (cs + p * (1.0 - cs)) * c.x + (p * (1.0 - cs) - s) * c.y + (p * (1.0 - cs) + s) * c.z,
            (p * (1.0 - cs) + s) * c.x + (cs + p * (1.0 - cs)) * c.y + (p * (1.0 - cs) - s) * c.z,
            (p * (1.0 - cs) - s) * c.x + (p * (1.0 - cs) + s) * c.y + (cs + p * (1.0 - cs)) * c.z,
        );
        return vec4<f32>(rotated, base_color.a);
    } else {
        // GradientShift: just pulse for now (direction shift handled in vertex)
        return vec4<f32>(mix(base_color.rgb, secondary_color.rgb, blend), base_color.a);
    }
}

@vertex
fn vertex(input: Vertex) -> InstancedVertexOutput {
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
        instanced_uniform.time
    );

    // Transform deformed vertex position by instance matrix
    let worldPosition = instanceMatrix * vec4<f32>(localPos, 1.0);

    var out: InstancedVertexOutput;
    out.world_position = worldPosition;
    out.position = position_world_to_clip(worldPosition.xyz);
    out.world_normal = worldNormal;
@if(VERTEX_UVS_A) {
    out.uv = input.uv;
}
    // Pass per-instance properties to fragment shader
    out.instance_color = input.instance_color;
    out.instance_secondary_color = input.instance_secondary_color;
    out.instance_direction = input.instance_direction;
    out.instance_flags = input.instance_flags;
    out.instance_dash = input.instance_dash;
    out.instance_deformation = input.instance_deformation;
    out.instance_appearance = input.instance_appearance;
    return out;
}

@fragment
fn instanced_fragment(
    in: InstancedVertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    // Build PbrInput manually — no StandardMaterial uniform for instanced batches.
    var pbr_input: pbr_types::PbrInput = pbr_types::pbr_input_new();

    pbr_input.is_orthographic = view.clip_from_view[3].w == 1.0;
    // Inline calculate_view: perspective view vector from world position
    let camera_world_pos = view.world_from_view[3].xyz;
    if (pbr_input.is_orthographic) {
        pbr_input.V = -normalize(view.world_from_view[2].xyz);
    } else {
        pbr_input.V = normalize(camera_world_pos - in.world_position.xyz);
    }
    pbr_input.frag_coord = in.position;
    pbr_input.world_position = in.world_position;

    // Inline prepare_world_normal: flip if back-facing and double_sided is false
    pbr_input.world_normal = in.world_normal;
    if (!is_front) {
        pbr_input.world_normal = -in.world_normal;
    }
    pbr_input.N = normalize(pbr_input.world_normal);

    // Compute gradient progress from per-instance direction and color mode.
    let color_mode = in.instance_flags.x;
    let progress = compute_gradient_progress_instanced(
        in.world_position.xyz,
        in.instance_direction,
        color_mode
    );

    // Resolve the base color from per-instance properties.
    var base_color = in.instance_color.rgb;
    if (color_mode > 0.5) {
        // Dual-directional or dual-radial gradient
        if (in.instance_flags.w > 0.5) {
            // OKLAB color interpolation
            base_color = oklab_mix(in.instance_color.rgb, in.instance_secondary_color.rgb, progress);
        } else {
            base_color = mix(in.instance_color.rgb, in.instance_secondary_color.rgb, progress);
        }
    }

    // Apply appearance animation (color pulse, cycle, gradient shift).
    let animated_color = apply_appearance_animation(
        vec4<f32>(base_color, in.instance_color.a),
        in.instance_secondary_color,
        in.instance_appearance,
        instanced_uniform.time
    );
    base_color = animated_color.rgb;

    // Apply dash pattern if enabled (requires UV attributes on the mesh)
@if(VERTEX_UVS_A) {
    let dash = in.instance_dash;
    if (dash.x > 0.0) {
        let uv = in.uv;
        let shape_type = in.instance_flags.z;
        let perimeter_progress = compute_perimeter_progress(uv, shape_type);
        let across = compute_across(uv, shape_type);
        let dash_mask = dash_segment_mask(perimeter_progress, across, dash);
        base_color = mix(base_color, base_color * 0.0, 1.0 - dash_mask);
    }
}

    pbr_input.material.base_color = vec4<f32>(base_color, animated_color.a);

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

    // Apply blend mode
    let blended = apply_blend_mode(out.color.rgb, in.position.xy, in.instance_flags, in.instance_color.a);
    out.color = blended;

    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
}

    return out;
}
