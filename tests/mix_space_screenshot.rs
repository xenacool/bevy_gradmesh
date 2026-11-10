//! Checks the committed `screenshots/screenshot_mix_spaces.png` (the last example in the
//! README) against reference colours and against the blends the shader is meant to compute.
//!
//! Every strip of the example is a flat rectangle drawn unlit by a `Camera2d` (the `Mesh2d`
//! gradient material), blended along X from a start to an end colour in one `MixSpace`. There
//! is no lighting or tone mapping, so a pixel is the blend itself, up to the GPU and the
//! 8-bit output. Two kinds of check cover it:
//!
//! * `reference_pixels_match`: a table of pixels recorded from the verified screenshot. It
//!   fails when the picture changes, whether a space is mixed wrongly or the layout moved.
//! * the CPU tests recompute the blend of every space with the same maths as the shader and
//!   check which end a pixel is closer to, how the hue turns, and which space keeps the most
//!   colour in the middle.
//!
//! After changing the example, run `make screenshot-mix_spaces`, look at the picture and
//! update `REFERENCE` from it.

use bevy::asset::RenderAssetUsages;
use bevy::image::{CompressedImageFormats, Image, ImageSampler, ImageType};

const PNG: &[u8] = include_bytes!("../screenshots/screenshot_mix_spaces.png");

// The scene of `examples/mix_spaces.rs`. Its `Camera2d` and strips are scaled so each strip
// lands on the pixels a perspective camera at `CAMERA_Z` with this field of view would give.
const STRIP_WIDTH: f32 = 5.0;
const CAMERA_Z: f32 = 10.5;
const FOV_Y: f32 = std::f32::consts::FRAC_PI_4;
const PAIRS: [([f32; 3], [f32; 3]); 2] = [
    ([0.10, 0.25, 0.95], [1.00, 0.88, 0.15]),
    ([1.00, 0.20, 0.35], [0.15, 0.85, 0.65]),
];

#[derive(Clone, Copy, Debug, PartialEq)]
enum Space {
    Oklab,
    Oklch,
    LinearRgb,
    Srgb,
}

/// Rows of the example, top to bottom.
const SPACES: [Space; 4] = [Space::Oklab, Space::Oklch, Space::LinearRgb, Space::Srgb];

/// `(row, pair, t, [r, g, b])`: the mean of the 5 x 5 pixels around the strip at `t` along it,
/// in 0-255 sRGB, as recorded from the committed screenshot.
const REFERENCE: [(usize, usize, f32, [u8; 3]); 40] = [
    (0, 0, 0.1, [40, 92, 233]),
    (0, 0, 0.3, [84, 132, 211]),
    (0, 0, 0.5, [131, 163, 185]),
    (0, 0, 0.7, [180, 189, 151]),
    (0, 0, 0.9, [230, 213, 97]),
    (0, 1, 0.1, [246, 83, 97]),
    (0, 1, 0.3, [224, 124, 112]),
    (0, 1, 0.5, [197, 155, 127]),
    (0, 1, 0.7, [161, 181, 143]),
    (0, 1, 0.9, [103, 205, 158]),
    (1, 0, 0.1, [0, 102, 246]),
    (1, 0, 0.3, [0, 160, 221]),
    (1, 0, 0.5, [0, 198, 169]),
    (1, 0, 0.7, [50, 218, 104]),
    (1, 0, 0.9, [207, 224, 36]),
    (1, 1, 0.1, [255, 71, 39]),
    (1, 1, 0.3, [244, 116, 0]),
    (1, 1, 0.5, [208, 158, 0]),
    (1, 1, 0.7, [155, 190, 18]),
    (1, 1, 0.9, [84, 211, 129]),
    (2, 0, 0.1, [93, 97, 232]),
    (2, 0, 0.3, [151, 140, 208]),
    (2, 0, 0.5, [188, 170, 180]),
    (2, 0, 0.7, [218, 194, 144]),
    (2, 0, 0.9, [244, 215, 92]),
    (2, 1, 0.1, [244, 88, 100]),
    (2, 1, 0.3, [219, 132, 119]),
    (2, 1, 0.5, [189, 162, 134]),
    (2, 1, 0.7, [152, 186, 148]),
    (2, 1, 0.9, [96, 207, 160]),
    (3, 0, 0.1, [48, 80, 222]),
    (3, 0, 0.3, [95, 112, 181]),
    (3, 0, 0.5, [140, 144, 140]),
    (3, 0, 0.7, [186, 176, 100]),
    (3, 0, 0.9, [232, 208, 59]),
    (3, 1, 0.1, [234, 67, 97]),
    (3, 1, 0.3, [190, 101, 112]),
    (3, 1, 0.5, [147, 134, 128]),
    (3, 1, 0.7, [103, 167, 143]),
    (3, 1, 0.9, [60, 200, 158]),
];

/// Per channel, in 0-255. Re-capturing the same scene on another GPU moves pixels a little.
const TOLERANCE: f32 = 6.0;

fn screenshot() -> Image {
    Image::from_buffer(
        PNG,
        ImageType::Extension("png"),
        CompressedImageFormats::NONE,
        true,
        ImageSampler::default(),
        RenderAssetUsages::default(),
    )
    .expect("the screenshot decodes")
}

/// Pixel of the strip in `row` for the colour pair `pair` at `t` (0 to 1) along it, found by
/// projecting the strip's centre line through the camera of the example.
fn sample(image: &Image, row: usize, pair: usize, t: f32) -> [f32; 3] {
    let (width, height) = (image.width() as f32, image.height() as f32);
    let half_height = CAMERA_Z * (FOV_Y * 0.5).tan();
    let half_width = half_height * width / height;
    let y = 2.4 - row as f32 * 1.6;
    let x = if pair == 0 { -3.1 } else { 3.1 } + (t - 0.5) * STRIP_WIDTH;
    let px = ((x / half_width) * 0.5 + 0.5) * width;
    let py = (0.5 - (y / half_height) * 0.5) * height;
    let mut sum = [0.0; 3];
    for dy in -2..=2 {
        for dx in -2..=2 {
            let color = image
                .get_color_at((px as i32 + dx) as u32, (py as i32 + dy) as u32)
                .expect("the sample is inside the screenshot")
                .to_srgba();
            sum[0] += color.red;
            sum[1] += color.green;
            sum[2] += color.blue;
        }
    }
    sum.map(|c| c / 25.0 * 255.0)
}

// --- The blends of the shader, on the CPU ---

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_oklab(c: [f32; 3]) -> [f32; 3] {
    let l = (0.4122214708 * c[0] + 0.5363325363 * c[1] + 0.0514459929 * c[2]).cbrt();
    let m = (0.2119034982 * c[0] + 0.6806995451 * c[1] + 0.1073969566 * c[2]).cbrt();
    let s = (0.0883024619 * c[0] + 0.2817188376 * c[1] + 0.6299787005 * c[2]).cbrt();
    [
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ]
}

fn oklab_to_linear(c: [f32; 3]) -> [f32; 3] {
    let l = (c[0] + 0.3963377774 * c[1] + 0.2158037573 * c[2]).powi(3);
    let m = (c[0] - 0.1055613458 * c[1] - 0.0638541728 * c[2]).powi(3);
    let s = (c[0] - 0.0894841775 * c[1] - 1.2914855480 * c[2]).powi(3);
    [
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    ]
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn to_linear(srgb: [f32; 3]) -> [f32; 3] {
    srgb.map(srgb_to_linear)
}

/// The blended colour in linear RGB, the same as `mix_colors` in `gradmesh.wesl`.
fn blend(space: Space, from: [f32; 3], to: [f32; 3], t: f32) -> [f32; 3] {
    let (a, b) = (to_linear(from), to_linear(to));
    match space {
        Space::Srgb => to_linear([0, 1, 2].map(|i| lerp(from[i], to[i], t))),
        Space::LinearRgb => [0, 1, 2].map(|i| lerp(a[i], b[i], t)),
        Space::Oklab => {
            let (la, lb) = (linear_to_oklab(a), linear_to_oklab(b));
            oklab_to_linear([0, 1, 2].map(|i| lerp(la[i], lb[i], t)))
        }
        Space::Oklch => {
            let (la, lb) = (linear_to_oklab(a), linear_to_oklab(b));
            let (ca, cb) = (la[1].hypot(la[2]), lb[1].hypot(lb[2]));
            let (ha, hb) = (la[2].atan2(la[1]), lb[2].atan2(lb[1]));
            let tau = std::f32::consts::TAU;
            let delta = (hb - ha) - tau * ((hb - ha) / tau).round();
            let hue = ha + delta * t;
            let chroma = lerp(ca, cb, t);
            oklab_to_linear([lerp(la[0], lb[0], t), chroma * hue.cos(), chroma * hue.sin()])
        }
    }
}

/// Chroma and hue (radians) of an sRGB colour given in 0-255.
fn chroma_hue(srgb_255: [f32; 3]) -> (f32, f32) {
    let lab = linear_to_oklab(to_linear(srgb_255.map(|c| c / 255.0)));
    (lab[1].hypot(lab[2]), lab[2].atan2(lab[1]))
}

/// Chroma and hue (radians) of a linear colour.
fn chroma_hue_linear(linear: [f32; 3]) -> (f32, f32) {
    let lab = linear_to_oklab(linear);
    (lab[1].hypot(lab[2]), lab[2].atan2(lab[1]))
}

/// Smallest angle between two hues, in radians.
fn hue_distance(a: f32, b: f32) -> f32 {
    let tau = std::f32::consts::TAU;
    let d = (a - b) - tau * ((a - b) / tau).round();
    d.abs()
}

#[test]
fn reference_pixels_match() {
    let image = screenshot();
    for (row, pair, t, expected) in REFERENCE {
        let got = sample(&image, row, pair, t);
        for channel in 0..3 {
            assert!(
                (got[channel] - expected[channel] as f32).abs() <= TOLERANCE,
                "{:?} pair {pair} at t = {t}: got {got:?}, reference {expected:?}",
                SPACES[row],
            );
        }
    }
}

#[test]
fn the_strips_run_from_the_start_colour_to_the_end_colour() {
    let image = screenshot();
    for (row, space) in SPACES.into_iter().enumerate() {
        for (pair, (from, to)) in PAIRS.into_iter().enumerate() {
            let hue_of = |c: [f32; 3]| chroma_hue_linear(to_linear(c)).1;
            let (start, end) = (hue_of(from), hue_of(to));
            let near_start = chroma_hue(sample(&image, row, pair, 0.1)).1;
            let near_end = chroma_hue(sample(&image, row, pair, 0.9)).1;
            assert!(
                hue_distance(near_start, start) < hue_distance(near_start, end),
                "{space:?} pair {pair}: the start of the strip is not closer to the start colour"
            );
            assert!(
                hue_distance(near_end, end) < hue_distance(near_end, start),
                "{space:?} pair {pair}: the end of the strip is not closer to the end colour"
            );
        }
    }
}

#[test]
fn oklch_sweeps_the_hue_as_the_blend_predicts() {
    let image = screenshot();
    let row = SPACES.iter().position(|s| *s == Space::Oklch).unwrap();
    for (pair, (from, to)) in PAIRS.into_iter().enumerate() {
        for t in [0.3, 0.5, 0.7] {
            let (_, expected) = chroma_hue_linear(blend(Space::Oklch, from, to, t));
            let (_, got) = chroma_hue(sample(&image, row, pair, t));
            assert!(
                hue_distance(got, expected).to_degrees() < 25.0,
                "pair {pair} at t = {t}: hue {:.0} degrees, the blend predicts {:.0}",
                got.to_degrees(),
                expected.to_degrees()
            );
        }
    }
}

#[test]
fn srgb_goes_grey_in_the_middle_and_the_other_spaces_keep_colour() {
    let image = screenshot();
    let chroma_at = |space: Space, pair: usize| {
        let row = SPACES.iter().position(|s| *s == space).unwrap();
        chroma_hue(sample(&image, row, pair, 0.5)).0
    };
    for (pair, (from, to)) in PAIRS.into_iter().enumerate() {
        let predicted = |space| chroma_hue_linear(blend(space, from, to, 0.5)).0;
        // Wherever the blend says one space is clearly more colourful than another in the
        // middle, the picture agrees.
        for a in SPACES {
            for b in SPACES {
                if predicted(a) > predicted(b) + 0.04 {
                    assert!(
                        chroma_at(a, pair) > chroma_at(b, pair),
                        "pair {pair}: {a:?} should keep more colour than {b:?} at the middle \
                         (blend {:.3} vs {:.3}, picture {:.3} vs {:.3})",
                        predicted(a),
                        predicted(b),
                        chroma_at(a, pair),
                        chroma_at(b, pair)
                    );
                }
            }
        }
    }
}

#[test]
fn the_cpu_blend_agrees_with_itself() {
    // The helpers must hit both ends exactly, or the checks above would be meaningless.
    for (from, to) in PAIRS {
        for space in SPACES {
            for (t, end) in [(0.0, from), (1.0, to)] {
                let got = blend(space, from, to, t);
                let want = to_linear(end);
                for i in 0..3 {
                    assert!((got[i] - want[i]).abs() < 1e-3, "{space:?} at t = {t}: {got:?} vs {want:?}");
                }
            }
        }
    }
}
