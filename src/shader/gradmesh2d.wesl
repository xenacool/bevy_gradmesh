// BevyGradmesh 2D shader - unlit gradient material for `Mesh2d` / `Material2d`.
//
// Evaluates the same dual-directional and dual-radial gradients, colour mix
// spaces and dash patterns as `gradmesh.wesl`, but from the mesh UVs instead of
// world positions and without any PBR lighting. Keep the shared helper sections
// below in sync with `gradmesh.wesl`.

import bevy_sprite_render::mesh2d::{
    vertex_output::VertexOutput,
    view_bindings::{view, globals},
};

@if(TONEMAP_IN_SHADER)
import bevy_core_pipeline::tonemapping;
@if(SRGB_OUTPUT)
import bevy_render::color_operations::linear_to_srgb;
@if(OKLAB_OUTPUT)
import bevy_render::color_operations::linear_rgb_to_oklab;

// --- OKLAB color space conversion ---

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

// --- Gradient mixing in a selectable colour space ---
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

// --- Dash pattern computation ---

/// Epsilon for float comparisons - avoids precision issues with near-1.0 values.
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

// --- Perimeter progress for dash patterns on 2D shapes ---

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
/// Traces the outline as one continuous loop: right edge -> top -> left -> bottom.
fn rectangle_perimeter_progress(uv: vec2<f32>) -> f32 {
    // One continuous loop around the outline, so dashes never restart at a corner:
    // right edge (upward) -> top (leftward) -> left (downward) -> bottom (rightward).
    // Each edge covers a quarter of the loop; `count` dashes span the whole outline.
    let p = uv * 2.0 - 1.0;  // Map UV to [-1, 1]
    let dist_right = abs(p.x - 1.0);
    let dist_left = abs(p.x + 1.0);
    let dist_top = abs(p.y - 1.0);
    let dist_bottom = abs(p.y + 1.0);

    let min_h = min(dist_right, dist_left);
    let min_v = min(dist_top, dist_bottom);

    if (min_h < min_v) {
        if (dist_right < dist_left) {
            return 0.25 * (p.y * 0.5 + 0.5);
        } else {
            return 0.5 + 0.25 * (0.5 - p.y * 0.5);
        }
    } else {
        if (dist_top < dist_bottom) {
            return 0.25 + 0.25 * (0.5 - p.x * 0.5);
        } else {
            return 0.75 + 0.25 * (p.x * 0.5 + 0.5);
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
        // Map: edge (dist=1) -> +1, center (dist=0) -> -1
        return clamp(dist_from_edge * 2.0 - 1.0, -1.0, 1.0);
    }
    // Disc/Generic: radial distance from center
    let center = uv - vec2<f32>(0.5, 0.5);
    let dist_from_center = length(center) * 2.0;
    return clamp(dist_from_center * 2.0 - 1.0, -1.0, 1.0);
}


// ===============================================================================
// 2D fragment shader - unlit gradients for `Mesh2d`
// ===============================================================================

/// Material uniform: same layout as the 3D `GradMeshUniformBuffer`.
///   color / secondary_color: sRGB components + alpha
///   direction.xy:            gradient direction in the mesh's UV frame (+Y = towards the top)
///   flags:                   x = colour mode (0 single, 1 directional, 2 radial),
///                            y = blend (0 opaque, otherwise alpha blended),
///                            z = shape type for dashes, w = mix space
///   dash:                    count, type code, gap ratio, offset
struct GradMesh2dUniform {
    color: vec4<f32>,
    secondary_color: vec4<f32>,
    direction: vec4<f32>,
    flags: vec4<f32>,
    dash: vec4<f32>,
    world_center: vec4<f32>,
}

// Material2d bind group is group 2 (view = 0, mesh = 1).
@group(2) @binding(0)
var<uniform> gradmesh: GradMesh2dUniform;

/// Appearance animation: x = speed, y = phase, z = type
/// (0 static, 1 colour pulse, 2 colour cycle, 3 gradient shift).
@group(2) @binding(1)
var<uniform> appearance: vec4<f32>;

fn hue_rotate(c: vec3<f32>, angle: f32) -> vec3<f32> {
    let lab = linear_to_oklab(c);
    let cs = cos(angle);
    let sn = sin(angle);
    return oklab_to_linear(vec3<f32>(lab.x, lab.y * cs - lab.z * sn, lab.y * sn + lab.z * cs));
}

/// Gradient progress from the mesh UVs, so it follows `Transform` scale and
/// rotation automatically.
fn gradient_progress_2d(uv: vec2<f32>, anim_angle: f32) -> f32 {
    let color_mode = i32(round(gradmesh.flags.x));
    if (color_mode == 0) {
        return 0.0;
    }
    let p = uv * 2.0 - 1.0;
    if (color_mode == 1) {
        var dir = gradmesh.direction.xy;
        if (i32(round(appearance.z)) == 3) {
            let cs = cos(anim_angle);
            let sn = sin(anim_angle);
            dir = vec2<f32>(dir.x * cs - dir.y * sn, dir.x * sn + dir.y * cs);
        }
        let len = length(dir);
        if (len < 0.00001) {
            return 0.5;
        }
        dir = dir / len;
        let half_range = max(abs(dir.x) + abs(dir.y), 0.00001);
        return clamp(dot(p, dir) / (2.0 * half_range) + 0.5, 0.0, 1.0);
    }
    return clamp(length(p), 0.0, 1.0);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let speed = appearance.x;
    let anim_angle = (globals.time * speed + appearance.y) * TAU;
    let anim_type = i32(round(appearance.z));
    let animated = speed > 0.0;

    let progress = gradient_progress_2d(in.uv, select(0.0, anim_angle, animated));

    var rgb = srgb_to_linear(gradmesh.color.rgb);
    var alpha = gradmesh.color.w;
    if (gradmesh.flags.x > 0.5) {
        rgb = mix_colors(gradmesh.color.rgb, gradmesh.secondary_color.rgb, progress, gradmesh.flags.w);
        alpha = mix(gradmesh.color.w, gradmesh.secondary_color.w, progress);
    }

    if (animated && anim_type == 1) {
        let k = 0.5 + 0.5 * sin(anim_angle);
        rgb = mix(rgb, srgb_to_linear(gradmesh.secondary_color.rgb), k);
    } else if (animated && anim_type == 2) {
        rgb = hue_rotate(rgb, anim_angle);
    }

@if(VERTEX_COLORS) {
    // Per-vertex colours multiply the gradient.
    rgb = rgb * in.color.rgb;
    alpha = alpha * in.color.a;
}

    let dash = gradmesh.dash;
    if (dash.x > 0.0) {
        let shape_type = gradmesh.flags.z;
        let perimeter_progress = compute_perimeter_progress(in.uv, shape_type);
        let across = compute_across(in.uv, shape_type);
        // Gaps are cut out so dashes read as real dashes.
        if (dash_segment_mask(perimeter_progress, across, dash) < 0.5) {
            discard;
        }
    }

    if (i32(round(gradmesh.flags.y)) == 0) {
        alpha = 1.0;
    }

    var output_color = vec4<f32>(rgb, alpha);
@if(TONEMAP_IN_SHADER)
    output_color = tonemapping::tone_mapping(output_color, view.color_grading);
@if(SRGB_OUTPUT)
    output_color = vec4<f32>(linear_to_srgb(output_color.rgb), output_color.a);
@if(OKLAB_OUTPUT)
    output_color = vec4<f32>(linear_rgb_to_oklab(output_color.rgb), output_color.a);
    return output_color;
}
