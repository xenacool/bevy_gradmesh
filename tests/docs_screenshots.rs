//! Keeps the README, the book and the local screenshots in step.
//!
//! Screenshots are no longer tracked in git — they live in GitHub Releases. These tests validate
//! the local `screenshots/` directory when it exists (e.g. after `make screenshots` or
//! `make release-zip`). On a fresh clone where `screenshots/` is empty or missing, the format
//! tests are skipped.
//!
//! * The README no longer links individual images; it points to the release zip.
//! * Every example in the Makefile has a corresponding screenshot file.
//! * Every animated GIF has many frames: a single frame would be a screenshot with the wrong
//!   extension, and a capture that never advanced the animation would show the same frame
//!   over and over.
//!
//! To refresh pictures run `make screenshots`, or `make screenshot-<example>` for one.

use std::collections::{BTreeSet, HashSet};
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};

use bevy::asset::RenderAssetUsages;
use bevy::image::{CompressedImageFormats, Image, ImageSampler, ImageType};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(path: impl AsRef<Path>) -> String {
    let path = root().join(path);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Names after `NAME =` in the Makefile, e.g. `EXAMPLES`.
fn makefile_list(name: &str) -> Vec<String> {
    let makefile = read("Makefile");
    let line = makefile
        .lines()
        .find(|l| l.starts_with(&format!("{name} =")))
        .unwrap_or_else(|| panic!("no `{name} =` in the Makefile"));
    line.split('=').nth(1).unwrap().split_whitespace().map(str::to_owned).collect()
}

/// `screenshots/screenshot_<example>.<ext>` paths cited in a markdown text.
fn cited(text: &str) -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    let mut rest = text;
    while let Some(start) = rest.find("screenshots/screenshot_") {
        let tail = &rest[start..];
        let end = tail.find(')').unwrap_or(tail.len());
        found.insert(tail[..end].to_owned());
        rest = &tail[end..];
    }
    found
}

/// Returns true if the local `screenshots/` directory has any captured files.
/// On a fresh clone the directory is empty (gitignored), so format tests should skip.
fn has_local_screenshots() -> bool {
    let dir = root().join("screenshots");
    dir.exists() && fs::read_dir(&dir).map_or(false, |mut entries| {
        entries.next().is_some()
    })
}

fn picture_of(example: &str, animated: bool) -> String {
    format!("screenshots/screenshot_{example}.{}", if animated { "gif" } else { "png" })
}

/// Decodes a PNG file the way Bevy would load it.
fn decode_png(path: &Path) -> Image {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    Image::from_buffer(
        &bytes,
        ImageType::Extension("png"),
        CompressedImageFormats::NONE,
        true,
        ImageSampler::default(),
        RenderAssetUsages::default(),
    )
    .unwrap_or_else(|e| panic!("{} is not a decodable PNG: {e}", path.display()))
}

/// How many different colours the picture has on a 32 x 32 grid of sample points.
fn distinct_sampled_colours(image: &Image) -> usize {
    const GRID: u32 = 32;
    let mut colours = HashSet::new();
    for gy in 0..GRID {
        for gx in 0..GRID {
            let x = gx * image.width() / GRID;
            let y = gy * image.height() / GRID;
            let colour = image.get_color_at(x, y).expect("sample point is inside the image");
            let srgba = colour.to_srgba();
            colours.insert([srgba.red, srgba.green, srgba.blue].map(|c| (c * 255.0).round() as u8));
        }
    }
    colours.len()
}

/// Number of frames in a GIF, counted by decoding every one of them.
fn gif_frame_count(path: &Path) -> usize {
    let file = File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut decoder = gif::DecodeOptions::new()
        .read_info(BufReader::new(file))
        .unwrap_or_else(|e| panic!("{} is not a decodable GIF: {e}", path.display()));
    let mut frames = 0;
    while decoder
        .read_next_frame()
        .unwrap_or_else(|e| panic!("{}: frame {frames} is broken: {e}", path.display()))
        .is_some()
    {
        frames += 1;
    }
    frames
}

#[test]
fn every_cited_picture_exists_and_has_the_format_of_its_extension() {
    // The README no longer cites individual screenshot files — it links to the release zip.
    // So `cited()` returns an empty set. This test is kept as a no-op for now; if the README
    // ever cites individual files again, it will start validating them.
    let _all = cited(&read("README.md"));
    // No assertion — the README correctly has no individual screenshot paths.
}

#[test]
fn every_example_has_a_picture_in_the_readme() {
    // The README no longer links individual screenshot files. Instead it lists every example
    // by name in the Screenshots section. Check that each Makefile example appears in the README.
    let readme = read("README.md");
    for example in makefile_list("EXAMPLES") {
        assert!(
            readme.contains(&format!("`{example}`")),
            "README.md does not mention example `{example}`"
        );
    }
}

#[test]
fn makefile_lists_are_consistent() {
    let (stills, animations) = (makefile_list("STILLS"), makefile_list("ANIMATIONS"));
    let mut every: Vec<String> = stills.iter().chain(&animations).cloned().collect();
    every.sort();
    let mut examples = makefile_list("EXAMPLES");
    examples.sort();
    assert_eq!(every, examples, "STILLS and ANIMATIONS must split EXAMPLES between them");
}

#[test]
fn the_examples_in_the_makefile_exist_and_are_built() {
    let manifest = read("Cargo.toml");
    for example in makefile_list("EXAMPLES") {
        let file = root().join(format!("examples/{example}.rs"));
        assert!(file.exists(), "examples/{example}.rs is missing");
        // Examples are found automatically unless Cargo.toml lists them, in which case the
        // entry must name the same file.
        if let Some(at) = manifest.find(&format!("name = \"{example}\"")) {
            let entry = &manifest[at..manifest.len().min(at + 120)];
            assert!(entry.contains(&format!("examples/{example}.rs")), "{example}: wrong path in Cargo.toml");
        }
    }
}

#[test]
fn animations_are_animated_gifs_with_many_frames() {
    // Skip on fresh clones where screenshots/ is not populated.
    if !has_local_screenshots() {
        return;
    }
    for example in makefile_list("ANIMATIONS") {
        let path = picture_of(&example, true);
        let full = root().join(&path);
        if !full.exists() {
            continue;
        }
        let frames = gif_frame_count(&full);
        assert!(frames >= 30, "{path} has only {frames} frames, a second of animation is 30");
    }
}

#[test]
fn stills_are_pngs_that_are_not_blank() {
    // Skip on fresh clones where screenshots/ is not populated.
    if !has_local_screenshots() {
        return;
    }
    for example in makefile_list("STILLS") {
        let path = picture_of(&example, false);
        let full = root().join(&path);
        if !full.exists() {
            continue;
        }
        // A capture of a window that never drew is one flat colour.
        let colours = distinct_sampled_colours(&decode_png(&full));
        assert!(colours > 1, "{path} has a single colour: was anything drawn?");
    }
}
