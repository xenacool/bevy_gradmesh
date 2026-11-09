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
    // xy gradient direction (x = start angle for angular modes), z colour mode
    // (0 single, 1 linear, 2 radial, 3 angular mirrored, 4 angular sweep),
    // w mix-space code
    gradient: vec4<f32>,
    // x dash length (> 0) or minus the dash count (< 0), 0 = no dashes;
    // y gap (world units, or 0..1 ratio in count mode), z offset (0..1 of a
    // period), w snap code (0 off, 1 tiling, 2 end to end)
    dash: vec4<f32>,
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

// ─── Perimeter coordinate and dashes ───
// `s` runs counter-clockwise around the outline from +X, in world units, so a
// dash length means the same distance on every side. Disc and ellipse use the
// polar angle (exact for circles); the rounded rectangle traces its edges and
// corner arcs from the nearest outline point, so dashes stay square to long edges.

const QUARTER_PI: f32 = 0.78539816339;

fn wrap_to(x: f32, period: f32) -> f32 {
    return x - period * floor(x / period);
}

fn ellipse_perimeter(radii: vec2<f32>) -> f32 {
    let a = radii.x + radii.y;
    let h = (radii.x - radii.y) * (radii.x - radii.y) / max(a * a, 0.00001);
    return 3.14159265359 * a * (1.0 + 3.0 * h / (10.0 + sqrt(4.0 - 3.0 * h)));
}

/// Distance along a quarter of the outline (right-middle to top-middle) of the
/// point at `(ux, uy)` in the first quadrant.
fn quarter_coord(u: vec2<f32>, half_size: vec2<f32>, r: f32) -> f32 {
    let inner = half_size - vec2<f32>(r, r);
    let q = u - inner;
    if (q.x > 0.0 && q.y > 0.0) {
        return inner.y + r * atan2(q.y, q.x);
    }
    if (q.x > q.y) {
        return min(u.y, inner.y);
    }
    return inner.y + 1.57079632679 * r + (inner.x - min(u.x, inner.x));
}

fn rect_quarter_length(half_size: vec2<f32>, r: f32) -> f32 {
    return (half_size.x - r) + (half_size.y - r) + 1.57079632679 * r;
}

fn rect_perimeter_coord(p: vec2<f32>, half_size: vec2<f32>, r: f32) -> f32 {
    let quarter = rect_quarter_length(half_size, r);
    let sq = quarter_coord(abs(p), half_size, r);
    if (p.x >= 0.0 && p.y >= 0.0) {
        return sq;
    }
    if (p.x < 0.0 && p.y >= 0.0) {
        return 2.0 * quarter - sq;
    }
    if (p.x < 0.0) {
        return 2.0 * quarter + sq;
    }
    return 4.0 * quarter - sq;
}

/// Perimeter coordinate of the outline point in direction `theta`.
fn perimeter_at_angle(theta: f32, half_size: vec2<f32>, r: f32, kind: i32) -> f32 {
    if (kind == 1) {
        return ellipse_perimeter(half_size) * wrap_to(theta, TAU) / TAU;
    }
    let d = vec2<f32>(cos(theta), sin(theta));
    let t = min(
        half_size.x / max(abs(d.x), 0.000001),
        half_size.y / max(abs(d.y), 0.000001),
    );
    return rect_perimeter_coord(d * t, half_size, r);
}

/// `(s, perimeter)` of the outline point nearest to `p`.
fn perimeter_coord(p: vec2<f32>, half_size: vec2<f32>, r: f32, kind: i32) -> vec2<f32> {
    if (kind == 1) {
        let l = ellipse_perimeter(half_size);
        return vec2<f32>(l * wrap_to(atan2(p.y, p.x), TAU) / TAU, l);
    }
    let l = 4.0 * rect_quarter_length(half_size, r);
    return vec2<f32>(rect_perimeter_coord(p, half_size, r), l);
}

/// `(period, dash length)` after snapping, over `span` world units of outline.
/// Mirrors `Dash2d::pattern` in `src/sdf2d.rs`; keep both in sync.
fn dash_pattern(span: f32, is_arc: bool) -> vec2<f32> {
    let dash = sdf.dash;
    let snap = i32(round(dash.w));
    if (dash.x < 0.0) {
        let n = max(round(-dash.x), 1.0);
        let r = clamp(dash.y, 0.0, 0.95);
        var period = span / n;
        if (is_arc && snap == 2) {
            period = span / (n - r);
        }
        return vec2<f32>(period, period * (1.0 - r));
    }
    let len = dash.x;
    let gap = max(dash.y, 0.0);
    let t0 = len + gap;
    if (snap == 0) {
        return vec2<f32>(t0, len);
    }
    if (snap == 2 && is_arc) {
        let n = max(round((span + gap) / t0), 1.0);
        let k = span / (n * len + (n - 1.0) * gap);
        return vec2<f32>(t0 * k, len * k);
    }
    let n = max(round(span / t0), 1.0);
    let k = span / (n * t0);
    return vec2<f32>(t0 * k, len * k);
}

/// Anti-aliased dash mask (1 inside a dash) for the rim pixel at `p`.
fn dash_coverage(p: vec2<f32>, half_size: vec2<f32>, is_arc: bool) -> f32 {
    let kind = i32(round(sdf.shape.x));
    let r = min(sdf.size.z, min(half_size.x, half_size.y));
    let pc = perimeter_coord(p, half_size, r, kind);
    let perimeter = pc.y;
    var s = pc.x;
    var span = perimeter;
    if (is_arc) {
        let s_a = perimeter_at_angle(sdf.shape.z, half_size, r, kind);
        s = wrap_to(s - s_a, perimeter);
        if (kind == 1) {
            span = perimeter * sdf.shape.w / TAU;
        } else {
            let s_b = perimeter_at_angle(sdf.shape.z + sdf.shape.w, half_size, r, kind);
            span = wrap_to(s_b - s_a, perimeter);
        }
    }
    // Screen-space rate of change of `s`, with the wrap-around seam removed.
    var dsx = dpdx(s);
    var dsy = dpdy(s);
    dsx = dsx - perimeter * round(dsx / perimeter);
    dsy = dsy - perimeter * round(dsy / perimeter);
    let aa = max(sqrt(dsx * dsx + dsy * dsy), 0.00001);

    let pattern = dash_pattern(span, is_arc);
    let period = pattern.x;
    let dash_len = pattern.y;
    if (period <= 0.00001 || period - dash_len <= period * 0.001) {
        return 1.0;
    }
    let q = wrap_to(s + sdf.dash.z * period, period);
    var dist = -min(q - dash_len, period - q);
    if (q < dash_len) {
        dist = min(q, dash_len - q);
    }
    var cover = clamp(0.5 + dist / aa, 0.0, 1.0);
    // With tiling, the next period starts exactly where an arc ends; keep the
    // anti-aliased leading edge of that dash from bleeding in as a sliver.
    if (is_arc && i32(round(sdf.dash.w)) == 1) {
        cover = cover * clamp((span - s) / aa - 0.5, 0.0, 1.0);
    }
    return cover;
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
    let is_arc = span < TAU - 0.0001;
    if (is_arc) {
        d = max(d, -arc_inside_distance(p, sdf.shape.z, span));
    }

    // Dashes only cut rims; both conditions come from the uniform buffer.
    var dash_mask = 1.0;
    if (rim_width > 0.0 && sdf.dash.x != 0.0) {
        dash_mask = dash_coverage(p, half_size, is_arc);
    }

    let coverage = clamp(0.5 - d / pixel, 0.0, 1.0) * dash_mask;
    if (coverage <= 0.0) {
        discard;
    }

    let mode = i32(round(sdf.gradient.z));
    var progress = 0.0;
    let pn = p / max(half_size, vec2<f32>(0.00001));
    if (mode == 1) {
        var dir = sdf.gradient.xy;
        let len = length(dir);
        if (len < 0.00001) {
            progress = 0.5;
        } else {
            dir = dir / len;
            let half_range = max(abs(dir.x) + abs(dir.y), 0.00001);
            progress = clamp(dot(pn, dir) / (2.0 * half_range) + 0.5, 0.0, 1.0);
        }
    } else if (mode == 2) {
        // Radial, fitted to the bounds: 1 at the edge midpoints.
        progress = clamp(length(pn), 0.0, 1.0);
    } else if (mode >= 3) {
        // Angular, counter-clockwise from the start angle in `gradient.x`.
        let turn = wrap_to(atan2(p.y, p.x) - sdf.gradient.x, TAU) / TAU;
        if (mode == 3) {
            // Mirrored: back to the first colour at the end, so there is no seam.
            progress = abs(2.0 * turn - 1.0);
        } else {
            progress = turn;
        }
    }

    var rgb = srgb_to_linear(sdf.color.rgb);
    var alpha = sdf.color.w;
    if (mode >= 1) {
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
