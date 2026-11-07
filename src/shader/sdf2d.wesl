// BevyGradmesh SDF 2D shader — shapes evaluated per pixel on one shared quad.
//
// The quad is a 1x1 centred `Mesh2d` scaled to (size + 2 * padding) by `Transform`.
// The fragment shader reconstructs the local position in world units from the UVs
// and evaluates a signed distance function (rounded rectangle or ellipse), an
// optional rim and an optional angular sector. Edges are anti-aliased from the
// screen-space size of one pixel, so no MSAA is required.
//
// The colour helpers below are shared with `gradmesh2d.wesl`; keep them in sync.

import bevy_sprite_render::mesh2d::{
    vertex_output::VertexOutput,
    view_bindings::view,
};

@if(TONEMAP_IN_SHADER)
import bevy_core_pipeline::tonemapping;
@if(SRGB_OUTPUT)
import bevy_render::color_operations::linear_to_srgb;
@if(OKLAB_OUTPUT)
import bevy_render::color_operations::linear_rgb_to_oklab;

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


// ═══════════════════════════════════════════════════════════════════════════════
// SDF shapes
// ═══════════════════════════════════════════════════════════════════════════════

struct SdfUniform {
    color: vec4<f32>,
    secondary_color: vec4<f32>,
    // x kind (0 rect, 1 ellipse), y rim width, z arc start, w arc span
    shape: vec4<f32>,
    // xy size, z corner radius, w padding per side
    size: vec4<f32>,
    // xy gradient direction, z colour mode (0 single, 1 linear), w mix-space code
    gradient: vec4<f32>,
}

// Material2d bind group is group 2 (view = 0, mesh = 1).
@group(2) @binding(0)
var<uniform> sdf: SdfUniform;

/// Signed distance to a rounded rectangle with half extents `half_size`.
fn sd_rounded_rect(p: vec2<f32>, half_size: vec2<f32>, radius: f32) -> f32 {
    let r = min(radius, min(half_size.x, half_size.y));
    let q = abs(p) - half_size + vec2<f32>(r, r);
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

/// Approximate signed distance to an ellipse with radii `radii` (exact for circles).
fn sd_ellipse(p: vec2<f32>, radii: vec2<f32>) -> f32 {
    let k0 = length(p / radii);
    let k1 = length(p / (radii * radii));
    if (k1 < 0.000001) {
        return -min(radii.x, radii.y);
    }
    return k0 * (k0 - 1.0) / k1;
}

/// Signed distance (positive inside) to the nearest edge of an angular sector,
/// measured in world units at the pixel's radius.
fn arc_inside_distance(p: vec2<f32>, start: f32, span: f32) -> f32 {
    var delta = atan2(p.y, p.x) - start;
    delta = delta - TAU * floor(delta / TAU);
    if (delta > 0.5 * (span + TAU)) {
        delta = delta - TAU;
    }
    return min(delta, span - delta) * length(p);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let size = sdf.size.xy;
    let extent = size + vec2<f32>(sdf.size.w * 2.0);
    // Local position in world units, centred, +Y up (mesh UV y points down).
    let p = vec2<f32>(in.uv.x - 0.5, 0.5 - in.uv.y) * extent;
    let half_size = size * 0.5;

    // World units per screen pixel at this fragment.
    let pixel = max(
        sqrt(0.5 * (dot(dpdx(p), dpdx(p)) + dot(dpdy(p), dpdy(p)))),
        0.00001,
    );

    var d: f32;
    if (i32(round(sdf.shape.x)) == 1) {
        d = sd_ellipse(p, half_size);
    } else {
        d = sd_rounded_rect(p, half_size, sdf.size.z);
    }

    let rim_width = sdf.shape.y;
    if (rim_width > 0.0) {
        // Hollow outline drawn inside the shape's edge.
        d = max(d, -(d + rim_width));
    }

    let span = sdf.shape.w;
    if (span < TAU - 0.0001) {
        d = max(d, -arc_inside_distance(p, sdf.shape.z, span));
    }

    let coverage = clamp(0.5 - d / pixel, 0.0, 1.0);
    if (coverage <= 0.0) {
        discard;
    }

    var progress = 0.0;
    if (sdf.gradient.z > 0.5) {
        var dir = sdf.gradient.xy;
        let len = length(dir);
        if (len < 0.00001) {
            progress = 0.5;
        } else {
            dir = dir / len;
            let pn = p / max(half_size, vec2<f32>(0.00001));
            let half_range = max(abs(dir.x) + abs(dir.y), 0.00001);
            progress = clamp(dot(pn, dir) / (2.0 * half_range) + 0.5, 0.0, 1.0);
        }
    }

    var rgb = srgb_to_linear(sdf.color.rgb);
    var alpha = sdf.color.w;
    if (sdf.gradient.z > 0.5) {
        rgb = mix_colors(sdf.color.rgb, sdf.secondary_color.rgb, progress, sdf.gradient.w);
        alpha = mix(sdf.color.w, sdf.secondary_color.w, progress);
    }

    var output_color = vec4<f32>(rgb, alpha * coverage);
@if(TONEMAP_IN_SHADER)
    output_color = tonemapping::tone_mapping(output_color, view.color_grading);
@if(SRGB_OUTPUT)
    output_color = vec4<f32>(linear_to_srgb(output_color.rgb), output_color.a);
@if(OKLAB_OUTPUT)
    output_color = vec4<f32>(linear_rgb_to_oklab(output_color.rgb), output_color.a);
    return output_color;
}
