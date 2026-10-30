// BevyGradmesh standalone shader — no PBR dependency.
//
// This shader implements gradient shading without any Bevy PBR imports,
// avoiding descriptor pool pressure on mobile GPUs. All material variation
// comes from per-instance vertex attributes, not shader specialization.
//
// Entry points:
//   - vs_standalone: vertex shader with per-instance transforms and properties
//   - fs_standalone: fragment shader with gradient shading and blend modes

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

const EPSILON: f32 = 1e-4;

fn dash_segment_mask(progress: f32, across: f32, dash: vec4<f32>) -> f32 {
    let total = dash.x;
    if (total <= 0.0) { return 1.0; }

    let visible_ratio = 1.0 - clamp(dash.z, 0.0, 1.0);
    if (visible_ratio >= 1.0 - EPSILON) { return 1.0; }

    let type_code = dash.y;
    let type_id = i32(round((type_code - 0.1) / 0.3));
    let center = 0.1 + f32(type_id) * 0.3;
    let modifier = clamp((type_code - center) / 0.075, -1.0, 1.0);

    var coordinate = progress * total + dash.w + (1.0 - visible_ratio) * 0.5;

    let phase_per_across = fwidth(coordinate) / max(fwidth(across), 0.00001);

    if (type_id < 1) {
        coordinate += modifier * across * visible_ratio * 0.25;
    } else if (type_id < 2) {
        coordinate += abs(across) * phase_per_across * modifier;
    }

    let phase = fract(coordinate);
    let half_visible = visible_ratio * 0.5;
    var signed_distance = half_visible - abs(phase - 0.5);

    if (type_id > 1) {
        let profile = sqrt(max(1.0 - across * across, 0.0));
        let cap = clamp(phase_per_across, 0.0, half_visible);
        signed_distance = half_visible - cap * (1.0 - profile) - abs(phase - 0.5);
    }

    let width = fwidth(signed_distance);
    return clamp(signed_distance / max(width, 0.00001) + 0.5, 0.0, 1.0);
}

// ─── Blend mode computation ───

fn bayer_2x2(px: i32, py: i32) -> i32 {
    return px * 2 + py * 3 - px * py * 4;
}

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
        let threshold = dither_threshold(fragment_position);
        if (alpha < threshold) {
            discard;
        }
        alpha = 1.0;
    } else if (blend_mode == 6) {
        color = color * 0.5;
        alpha = 1.0;
    } else if (blend_mode == 7) {
        color = color * 1.2;
        alpha = 1.0;
    } else if (blend_mode == 8) {
        color = color * 1.8;
        alpha = 1.0;
    } else if (blend_mode == 9) {
        color = color * 0.3;
        alpha = 1.0;
    } else if (blend_mode == 10) {
        color = color * 0.2;
        alpha = 1.0;
    } else if (blend_mode == 11) {
        color = color * 0.4;
        alpha = 1.0;
    }

    return vec4<f32>(color, alpha);
}

// ─── Perimeter progress for dash patterns ───

fn angular_progress(uv: vec2<f32>) -> f32 {
    let p = uv * 2.0 - 1.0;
    let eps = 0.00001;
    let a = atan(p.y / max(abs(p.x), eps));
    let angle = select(a + 3.14159265359, a, p.x >= 0.0);
    return fract(angle / 6.28318530718 + 0.5);
}

fn rectangle_perimeter_progress(uv: vec2<f32>) -> f32 {
    let p = uv * 2.0 - 1.0;
    let abs_p = abs(p);
    let dist_right = abs(p.x - 1.0);
    let dist_left = abs(p.x + 1.0);
    let dist_top = abs(p.y - 1.0);
    let dist_bottom = abs(p.y + 1.0);
    
    let min_h = min(dist_right, dist_left);
    let min_v = min(dist_top, dist_bottom);
    
    if (min_h < min_v) {
        if (dist_right < dist_left) {
            return fract(p.y * 0.5 + 0.75);
        } else {
            return fract(-p.y * 0.5 + 0.25);
        }
    } else {
        if (dist_top < dist_bottom) {
            return fract(p.x * 0.5 + 0.5);
        } else {
            return fract(-p.x * 0.5);
        }
    }
}

fn compute_perimeter_progress(uv: vec2<f32>, shape_type: f32) -> f32 {
    if (shape_type > 0.5 && shape_type < 1.5) {
        return angular_progress(uv);
    } else if (shape_type > 1.5) {
        return rectangle_perimeter_progress(uv);
    }
    return (uv.x + uv.y) * 0.5;
}

fn compute_across(uv: vec2<f32>, shape_type: f32) -> f32 {
    if (shape_type > 1.5) {
        let p = uv * 2.0 - 1.0;
        let dist_from_edge = min(abs(p.x), abs(p.y));
        return clamp(dist_from_edge * 2.0 - 1.0, -1.0, 1.0);
    }
    let center = uv - vec2<f32>(0.5, 0.5);
    let dist_from_center = length(center) * 2.0;
    return clamp(dist_from_center * 2.0 - 1.0, -1.0, 1.0);
}

// ─── Vertex deformation ───

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

    if (deform_type <= 0.5) {
        return local_pos;
    }

    let t = sin(time * frequency * 6.28318530718 + phase * 6.28318530718);

    if (deform_type < 1.5) {
        return local_pos + t * amplitude * local_normal;
    } else if (deform_type < 2.5) {
        return local_pos * (1.0 + t * amplitude);
    } else {
        return local_pos + t * amplitude * local_normal;
    }
}

// ─── Vertex input ───

struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    // Per-instance transform (4×vec4)
    @location(3) instance_col0: vec4<f32>,
    @location(4) instance_col1: vec4<f32>,
    @location(5) instance_col2: vec4<f32>,
    @location(6) instance_col3: vec4<f32>,
    // Per-instance properties (7×vec4)
    @location(7) instance_color: vec4<f32>,
    @location(8) instance_secondary_color: vec4<f32>,
    @location(9) instance_direction: vec4<f32>,
    @location(10) instance_flags: vec4<f32>,
    @location(11) instance_dash: vec4<f32>,
    @location(12) instance_deformation: vec4<f32>,
    @location(13) instance_appearance: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) secondary_color: vec4<f32>,
    @location(5) direction: vec4<f32>,
    @location(6) flags: vec4<f32>,
    @location(7) dash: vec4<f32>,
    @location(8) appearance: vec4<f32>,
}

// View-projection matrix uniform
struct ViewUniform {
    view_proj: mat4x4<f32>,
    time: f32,
    _padding: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> view: ViewUniform;

@vertex
fn vs_standalone(input: VertexInput) -> VertexOutput {
    // Reconstruct instance transform matrix
    let instanceMatrix = mat4x4<f32>(
        input.instance_col0,
        input.instance_col1,
        input.instance_col2,
        input.instance_col3,
    );

    // Transform normal
    let normalMatrix = mat3x3<f32>(
        input.instance_col0.xyz,
        input.instance_col1.xyz,
        input.instance_col2.xyz,
    );
    let worldNormal = normalize(normalMatrix * input.normal);

    // Apply deformation in local space
    let localPos = apply_deformation(
        input.position.xyz,
        input.normal,
        input.instance_deformation,
        view.time
    );

    // Transform to world space
    let worldPosition = instanceMatrix * vec4<f32>(localPos, 1.0);

    var out: VertexOutput;
    out.clip_position = view.view_proj * worldPosition;
    out.world_position = worldPosition;
    out.world_normal = worldNormal;
    out.uv = input.uv;
    out.color = input.instance_color;
    out.secondary_color = input.instance_secondary_color;
    out.direction = input.instance_direction;
    out.flags = input.instance_flags;
    out.dash = input.instance_dash;
    out.appearance = input.instance_appearance;
    return out;
}

@fragment
fn fs_standalone(in: VertexOutput) -> @location(0) vec4<f32> {
    let color_mode = i32(in.flags.x);
    let blend_mode = i32(in.flags.y);
    let oklab_flag = in.flags.w;

    // Compute gradient progress
    var progress = 0.0;
    if (color_mode == 1) {
        // Directional gradient
        let dir = in.direction.xyz;
        let normalized_dir = normalize(dir);
        let world_pos = in.world_position.xyz;
        progress = dot(world_pos, normalized_dir) * 0.1 + 0.5;
        progress = clamp(progress, 0.0, 1.0);
    } else if (color_mode == 2) {
        // Radial gradient
        let dist = length(in.world_position.xyz);
        progress = clamp(dist * 0.5, 0.0, 1.0);
    }

    // Interpolate colors
    var base_color = in.color.rgb;
    if (color_mode > 0) {
        if (oklab_flag > 0.5) {
            base_color = oklab_mix(in.color.rgb, in.secondary_color.rgb, progress);
        } else {
            base_color = mix(in.color.rgb, in.secondary_color.rgb, progress);
        }
    }

    // Apply dash pattern
    let dash = in.dash;
    if (dash.x > 0.0) {
        let perimeter_progress = compute_perimeter_progress(in.uv, in.flags.z);
        let across = compute_across(in.uv, in.flags.z);
        let dash_mask = dash_segment_mask(perimeter_progress, across, dash);
        base_color = mix(base_color, vec3<f32>(0.0), 1.0 - dash_mask);
    }

    // Apply blend mode
    let result = apply_blend_mode(base_color, in.clip_position.xy, in.flags, in.color.w);

    return result;
}
