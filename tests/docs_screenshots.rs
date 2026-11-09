//! Keeps the README, the book and the screenshots in step.
//!
//! * Every image the README or the book cites exists and is a real PNG or WebP.
//! * Every example in the Makefile has a picture, cited by the README and by the book, as a PNG
//!   if it is still and as an animated WebP if it changes over time.
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

fn book_text() -> String {
    let mut all = String::new();
    for entry in fs::read_dir(root().join("book/src")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "md") {
            all.push_str(&fs::read_to_string(path).unwrap());
        }
    }
    all
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
    let mut all = cited(&read("README.md"));
    all.extend(cited(&book_text()));
    assert!(!all.is_empty());
    for path in all {
        let bytes = fs::read(root().join(&path)).unwrap_or_else(|e| panic!("{path}: {e}"));
        if path.ends_with(".png") {
            assert!(is_png(&bytes), "{path} is not a PNG");
        } else if path.ends_with(".webp") {
            assert!(is_webp(&bytes), "{path} is not a WebP");
        } else {
            panic!("{path}: only PNG and WebP are used");
        }
    }
}

#[test]
fn every_example_has_a_picture_in_the_readme_and_in_the_book() {
    let (stills, animations) = (makefile_list("STILLS"), makefile_list("ANIMATIONS"));
    let mut every: Vec<String> = stills.iter().chain(&animations).cloned().collect();
    every.sort();
    let mut examples = makefile_list("EXAMPLES");
    examples.sort();
    assert_eq!(every, examples, "STILLS and ANIMATIONS must split EXAMPLES between them");

    let (readme, book) = (cited(&read("README.md")), cited(&book_text()));
    for (list, animated) in [(&stills, false), (&animations, true)] {
        for example in list {
            let picture = picture_of(example, animated);
            assert!(readme.contains(&picture), "README.md does not show {picture}");
            assert!(book.contains(&picture), "no book chapter shows {picture}");
        }
    }
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
    for example in makefile_list("ANIMATIONS") {
        let path = picture_of(&example, true);
        let bytes = fs::read(root().join(&path)).unwrap_or_else(|e| panic!("{path}: {e}"));
        assert!(is_webp(&bytes), "{path} is not a WebP");
        let frames = webp_frames(&bytes).unwrap_or_else(|| panic!("{path} is a still WebP, not an animation"));
        assert!(frames >= 30, "{path} has only {frames} frames, a second of animation is 30");
    }
}

#[test]
fn stills_are_pngs_that_are_not_blank() {
    for example in makefile_list("STILLS") {
        let path = picture_of(&example, false);
        let bytes = fs::read(root().join(&path)).unwrap_or_else(|e| panic!("{path}: {e}"));
        assert!(is_png(&bytes), "{path} is not a PNG");
        // A capture of a window that never drew compresses to almost nothing.
        assert!(bytes.len() > 20_000, "{path} is only {} bytes: was anything drawn?", bytes.len());
    }
}

#[test]
fn the_book_reaches_the_screenshots() {
    // mdBook publishes what is under `book/src`, so the folder is linked in.
    let link = root().join("book/src/screenshots");
    assert!(link.join("screenshot_shapes.png").exists(), "book/src/screenshots must lead to ../../screenshots");
}
