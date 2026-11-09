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
    mesh_functions,
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
    view_transformations::position_world_to_clip,
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
        0.4122214708 * c.x + 0.5363325363 * c.y + 0.0514459929 * c.z,
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

/// Convert an sRGB-encoded colour (as supplied by users / `Color::srgb`) to linear.
/// Material and instance colours are stored as sRGB components, while the render
/// targets and lighting expect linear values.
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

// ─── Gradient mixing in a selectable colour space ───
// `flags.w` carries the space: 0 = OKLAB (default), 1 = OKLCH, 2 = linear RGB, 3 = sRGB.
// Keep in sync with `MixSpace::gpu_code` in `src/mix_space.rs`.

const TAU: f32 = 6.28318530718;

fn oklch_mix(c1: vec3<f32>, c2: vec3<f32>, t: f32) -> vec3<f32> {
    let lab1 = linear_to_oklab(c1);
    let lab2 = linear_to_oklab(c2);
    let chroma1 = length(lab1.yz);
    let chroma2 = length(lab2.yz);
    var hue1 = atan2(lab1.z, lab1.y);
    var hue2 = atan2(lab2.z, lab2.y);
    // An achromatic end has no hue of its own; borrow the other end's hue.
    if (chroma1 < 0.0001) {
        hue1 = hue2;
    } else if (chroma2 < 0.0001) {
        hue2 = hue1;
    }
    // Shortest arc around the hue wheel.
    var delta = hue2 - hue1;
    delta = delta - TAU * round(delta / TAU);
    let hue = hue1 + delta * t;
    let lightness = mix(lab1.x, lab2.x, t);
    let chroma = mix(chroma1, chroma2, t);
    return oklab_to_linear(vec3<f32>(lightness, chroma * cos(hue), chroma * sin(hue)));
}

/// Mixes two sRGB-encoded colours in `space_code` and returns linear RGB.
fn mix_colors(a_srgb: vec3<f32>, b_srgb: vec3<f32>, t: f32, space_code: f32) -> vec3<f32> {
    let space = i32(round(space_code));
    if (space == 3) {
        return srgb_to_linear(mix(a_srgb, b_srgb, t));
    }
    let a = srgb_to_linear(a_srgb);
    let b = srgb_to_linear(b_srgb);
    if (space == 2) {
        return mix(a, b, t);
    }
    if (space == 1) {
        return oklch_mix(a, b, t);
    }
    return oklab_mix(a, b, t);
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
/// 15 Vec4 fields = 240 bytes total, bound at slot 100.
struct GradMeshUniform {
    color: vec4<f32>,
    secondary_color: vec4<f32>,
    direction: vec4<f32>,
    flags: vec4<f32>,
    dash: vec4<f32>,
    world_center: vec4<f32>,
    // Procedural noise: (scale, amount, 0, 0).
    noise: vec4<f32>,
    // Procedural stripes: (direction.xyz, frequency).
    stripes: vec4<f32>,
    // (stripe amount, 0, 0, 0).
    stripes_amount: vec4<f32>,
    // Rim light: (linear rgb, power).
    rim: vec4<f32>,
    // Height-band colours (linear rgba), lowest band first (colour mode 3).
    bands: array<vec4<f32>, 4>,
    // Height each band starts at; unused slots hold a huge value.
    band_heights: vec4<f32>,
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
    var dir = gradmesh.direction.xyz;

    if (color_mode == 0) {
        return 0.0;
    }

    // `direction.w` selects the direction space: 0 = world (fixed), 1 = local
    // (rotated by the entity's world-from-local transform, so the gradient
    // follows rotated meshes).
    if (gradmesh.direction.w > 0.5) {
        let world_from_local = mesh_functions::get_world_from_local(in.instance_index);
        let rotated = mat3x3<f32>(
            world_from_local[0].xyz,
            world_from_local[1].xyz,
            world_from_local[2].xyz,
        ) * dir;
        dir = normalize(rotated);
    }

    // Object-space position: world position minus the shape's world center.
    let object_pos = in.world_position.xyz - gradmesh.world_center.xyz;
    // `world_center.w` optionally carries the gradient half-extent (0 = default 0.5).
    let half_extent = select(0.5, gradmesh.world_center.w, gradmesh.world_center.w > 0.0);

    if (color_mode == 1) {
        // Directional gradient: project object-space position onto the gradient
        // direction, normalized by the shape's half-extent (0.5 for unit shapes).
        let normalized_dir = normalize(dir);
        let half_range = max(abs(dot(normalized_dir, vec3<f32>(half_extent, half_extent, half_extent))), 0.00001);
        let progress = dot(object_pos, normalized_dir) / (2.0 * half_range) + 0.5;
        return clamp(progress, 0.0, 1.0);
    } else if (color_mode == 2) {
        // Radial gradient: distance from shape center, normalized by half-extent.
        let dist = length(object_pos);
        return clamp(dist / (half_extent * 1.41421356), 0.0, 1.0);
    }

    return 0.5;
}

// ─── Procedural surface detail ───

/// Cheap 3D hash in [0, 1).
fn hash13(p3: vec3<f32>) -> f32 {
    var p = fract(p3 * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

/// Smooth trilinear value noise in [0, 1].
fn value_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let n000 = hash13(i);
    let n100 = hash13(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash13(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash13(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash13(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash13(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash13(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash13(i + vec3<f32>(1.0, 1.0, 1.0));
    return mix(
        mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
        mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y),
        u.z,
    );
}

/// Fragment position in the entity's local space (undoes rotation and scale),
/// so procedural patterns stick to the entity as it moves, turns and resizes.
fn local_position(in: VertexOutput) -> vec3<f32> {
    let m = mesh_functions::get_world_from_local(in.instance_index);
    let p = in.world_position.xyz - m[3].xyz;
    let cx = m[0].xyz;
    let cy = m[1].xyz;
    let cz = m[2].xyz;
    return vec3<f32>(
        dot(p, cx) / max(dot(cx, cx), 0.00000001),
        dot(p, cy) / max(dot(cy, cy), 0.00000001),
        dot(p, cz) / max(dot(cz, cz), 0.00000001),
    );
}

/// Colour (linear rgba) of the height band a world-space `y` falls in: a band
/// holds from its height up to the next one, and the first band holds below.
fn height_band_color(y: f32) -> vec4<f32> {
    let h = gradmesh.band_heights;
    var c = gradmesh.bands[0];
    if (y >= h.y) {
        c = gradmesh.bands[1];
    }
    if (y >= h.z) {
        c = gradmesh.bands[2];
    }
    if (y >= h.w) {
        c = gradmesh.bands[3];
    }
    return c;
}

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    let progress = compute_gradient_progress_material(in);

    // Colour mode 3 colours by world height instead of by gradient.
    let by_height = gradmesh.flags.x > 2.5;
    let band = height_band_color(in.world_position.y);

    var base_color = srgb_to_linear(gradmesh.color.rgb);
    if (by_height) {
        base_color = band.rgb;
    } else if (gradmesh.flags.x > 0.5) {
        base_color = mix_colors(
            gradmesh.color.rgb,
            gradmesh.secondary_color.rgb,
            progress,
            gradmesh.flags.w,
        );
    }

    // Per-vertex colours (compound part tints, coloured polygons) multiply the
    // gradient, so the gradient stays on top of the tint.
@if(VERTEX_COLORS) {
    base_color = base_color * in.color.rgb;
}

    // Procedural noise and stripes vary the tint in the entity's local space.
    if (gradmesh.noise.y > 0.0 || gradmesh.stripes_amount.x > 0.0) {
        let local_pos = local_position(in);
        var factor = 1.0;
        if (gradmesh.noise.y > 0.0) {
            let n = value_noise(local_pos * max(gradmesh.noise.x, 0.0001));
            factor = factor * (1.0 + gradmesh.noise.y * (n * 2.0 - 1.0));
        }
        if (gradmesh.stripes_amount.x > 0.0) {
            let phase = dot(local_pos, gradmesh.stripes.xyz) * gradmesh.stripes.w * TAU;
            factor = factor * (1.0 + gradmesh.stripes_amount.x * sin(phase));
        }
        base_color = base_color * max(factor, 0.0);
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
        // Gaps are cut out rather than painted black so dashes read as real dashes.
        if (dash_mask < 0.5) {
            discard;
        }
    }
}

    pbr_input.material.base_color = vec4<f32>(base_color, pbr_input.material.base_color.a);

    // Rim light: emission that grows towards the silhouette.
    if (gradmesh.rim.w > 0.0) {
        let n_dot_v = clamp(dot(normalize(pbr_input.N), normalize(pbr_input.V)), 0.0, 1.0);
        let rim_factor = pow(1.0 - n_dot_v, gradmesh.rim.w);
        pbr_input.material.emissive = vec4<f32>(
            pbr_input.material.emissive.rgb + gradmesh.rim.rgb * rim_factor,
            pbr_input.material.emissive.a,
        );
    }

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

    // Gradient alpha blends between both colours along the gradient.
    var alpha_w = gradmesh.color.w;
    if (by_height) {
        alpha_w = band.a;
    } else if (gradmesh.flags.x > 0.5) {
        alpha_w = mix(gradmesh.color.w, gradmesh.secondary_color.w, progress);
    }
    let blended = apply_blend_mode(out.color.rgb, in.position.xy, gradmesh.flags, alpha_w);
    out.color = blended;

    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
}

    return out;
}
