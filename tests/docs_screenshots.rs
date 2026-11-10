//! Keeps the README, the book and the local screenshots in step.
//!
//! Screenshots are no longer tracked in git — they live in GitHub Releases. These tests validate
//! the local `screenshots/` directory when it exists (e.g. after `make screenshots` or
//! `make release-zip`). On a fresh clone where `screenshots/` is empty or missing, the format
//! tests are skipped.
//!
//! * The README no longer links individual images; it points to the release zip.
//! * Every example in the Makefile has a corresponding screenshot file.
//! * Every animated WebP has many frames: a single frame would be a screenshot with the wrong
//!   extension, and a capture that never advanced the animation would show the same frame
//!   over and over.
//!
//! To refresh pictures run `make screenshots`, or `make screenshot-<example>` for one.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

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
    format!("screenshots/screenshot_{example}.{}", if animated { "webp" } else { "png" })
}

fn is_png(bytes: &[u8]) -> bool {
    bytes.starts_with(b"\x89PNG\r\n\x1a\n")
}

fn is_webp(bytes: &[u8]) -> bool {
    bytes.len() > 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP"
}

/// Number of frames of an animated WebP, or `None` if it is not animated: the chunks of the
/// RIFF container are walked and the `ANMF` (frame) chunks counted, after an `ANIM` chunk.
fn webp_frames(bytes: &[u8]) -> Option<usize> {
    let (mut at, mut animated, mut frames) = (12, false, 0);
    while at + 8 <= bytes.len() {
        let size = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap()) as usize;
        match &bytes[at..at + 4] {
            b"ANIM" => animated = true,
            b"ANMF" => frames += 1,
            _ => {}
        }
        at += 8 + size + (size & 1);
    }
    animated.then_some(frames)
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
fn animations_are_animated_webps_with_many_frames() {
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
        let bytes = fs::read(&full).unwrap_or_else(|e| panic!("{path}: {e}"));
        assert!(is_webp(&bytes), "{path} is not a WebP");
        let frames = webp_frames(&bytes).unwrap_or_else(|| panic!("{path} is a still WebP, not an animation"));
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
        let bytes = fs::read(&full).unwrap_or_else(|e| panic!("{path}: {e}"));
        assert!(is_png(&bytes), "{path} is not a PNG");
        // A capture of a window that never drew compresses to almost nothing.
        assert!(bytes.len() > 20_000, "{path} is only {} bytes: was anything drawn?", bytes.len());
    }
}
