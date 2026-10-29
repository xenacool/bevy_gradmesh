//! Automatic screenshot capture utility for headless rendering.
//!
//! When the `DSD_SCREENSHOT` environment variable is set, this module configures
//! the render plugin for synchronous shader compilation and adds a system that
//! waits for the scene to fully render before capturing a screenshot and exiting.
//! This is used by the `make screenshots` target to generate README images without
//! manual intervention.
//!
//! # Timing
//!
//! The Bevy screenshot pipeline spans multiple frames:
//! 1. Spawn `Screenshot` entity in Update
//! 2. Next ExtractSchedule: entity is extracted to the render world
//! 3. Next Render: screenshot is rendered to a texture and copied to a GPU buffer
//! 4. Async GPU readback completes
//! 5. Next Update: `ScreenshotCaptured` event fires
//!
//! This plugin detects when the first frame has actually rendered by observing
//! `FrameCount` (which increments during the `Last` schedule after rendering).
//! It then spawns the screenshot entity and waits for the `ScreenshotCaptured`
//! observer before exiting — no frame counters or sleep needed.
//!
//! Configure `RenderPlugin` with `synchronous_pipeline_compilation: true` so all
//! shaders compile before the first frame. Without this, the first frame may render
//! black while shaders are still compiling asynchronously.
//!
//! ```rust,no_run
//! use bevy::prelude::*;
//! use bevy::render::RenderPlugin;
//!
//! fn setup() {}
//!
//! fn main() {
//!     App::new()
//!         .add_plugins(DefaultPlugins.set(RenderPlugin {
//!             synchronous_pipeline_compilation: true,
//!             ..default()
//!         }))
//!         .add_plugins(screenshot::AutoScreenshotPlugin::new("shapes.png"))
//!         .add_systems(Startup, setup)
//!         .run();
//! }
//! ```
//!
//! The screenshot will only be taken when `DSD_SCREENSHOT=1` is set in the environment.

use bevy::diagnostic::FrameCount;
use bevy::prelude::*;
use bevy::render::view::screenshot::{save_to_disk, Screenshot, ScreenshotCaptured};

/// Plugin that adds automatic screenshot capture when `DSD_SCREENSHOT` is set.
///
/// When enabled, the app will wait for the first frame to render, capture a
/// screenshot of the primary window, save it to the given filename, and then exit.
pub struct AutoScreenshotPlugin {
    filename: String,
    title: Option<String>,
    description: Option<String>,
    /// Minimum frame count to wait before taking the screenshot (default 1).
    /// Complex 3D scenes may need more frames for shaders to compile.
    wait_frames: u32,
}

impl AutoScreenshotPlugin {
    /// Create a new screenshot plugin that saves to the given filename.
    ///
    /// The screenshot is only taken when the `DSD_SCREENSHOT` environment variable
    /// is set to a non-empty value (e.g., `DSD_SCREENSHOT=1`).
    pub fn new(filename: impl Into<String>) -> Self {
        Self {
            filename: filename.into(),
            title: None,
            description: None,
            wait_frames: 1,
        }
    }

    /// Set the title displayed in the screenshot overlay banner.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set the description displayed in the screenshot overlay banner.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the minimum frame count to wait before taking the screenshot.
    ///
    /// Complex 3D scenes with many materials may need more frames for all
    /// shaders to compile and render correctly. Default is 1.
    pub fn wait_frames(mut self, frames: u32) -> Self {
        self.wait_frames = frames.max(1);
        self
    }
}

impl Plugin for AutoScreenshotPlugin {
    fn build(&self, app: &mut App) {
        // Only enable when DSD_SCREENSHOT is set
        if std::env::var("DSD_SCREENSHOT").is_err() {
            return;
        }

        app.insert_resource(AutoScreenshotFilename(self.filename.clone()))
            .insert_resource(ScreenshotState::default())
            .insert_resource(WaitFrames(self.wait_frames));

        // Add title overlay if configured
        if self.title.is_some() || self.description.is_some() {
            app.insert_resource(ScreenshotTitle {
                title: self.title.clone(),
                description: self.description.clone(),
            })
            .add_systems(Startup, spawn_title_overlay);
        }

        app.add_systems(
            Update,
            (wait_for_first_frame, quit_after_screenshot_saved).chain(),
        );
    }
}

/// Title and description for the screenshot overlay banner.
#[derive(Resource)]
struct ScreenshotTitle {
    title: Option<String>,
    description: Option<String>,
}

/// Component marking the title overlay entity so it can be removed after screenshot.
#[derive(Component)]
struct TitleOverlay;

/// Spawn a semi-transparent banner at the top of the window showing the title and description.
fn spawn_title_overlay(mut commands: Commands, title: Res<ScreenshotTitle>, window: Query<&Window, With<Window>>) {
    let window = window.single().expect("no window found");
    let width = window.resolution.width() as f32;
    let height = window.resolution.height() as f32;
    let banner_height = 44.0_f32;

    // Title text using Text2d
    if let Some(ref t) = title.title {
        commands.spawn((
            Text2d(t.clone()),
            TextColor(Color::srgb(1.0, 1.0, 1.0)),
            TextFont {
                font_size: FontSize::Px(18.0),
                ..default()
            },
            Transform::from_xyz(width / 2.0 - 100.0, height / 2.0 - banner_height / 2.0 - 10.0, 1000.0),
            TitleOverlay,
            Name::new("Title Overlay Text"),
        ));
    }
    // Description text using Text2d
    if let Some(ref d) = title.description {
        commands.spawn((
            Text2d(d.clone()),
            TextColor(Color::srgba(1.0, 1.0, 1.0, 0.7)),
            TextFont {
                font_size: FontSize::Px(14.0),
                ..default()
            },
            Transform::from_xyz(width / 2.0 - 100.0, height / 2.0 - banner_height / 2.0 + 10.0, 1000.0),
            TitleOverlay,
            Name::new("Title Overlay Description"),
        ));
    }
}

/// Target filename stored as a resource so the observer can access it.
#[derive(Resource)]
struct AutoScreenshotFilename(String);

/// Minimum frame count to wait before taking the screenshot.
#[derive(Resource)]
struct WaitFrames(u32);

/// Tracks the screenshot lifecycle: pending → triggered → saved → quit.
#[derive(Resource, Default)]
struct ScreenshotState {
    /// `true` after we spawned the `Screenshot` entity.
    triggered: bool,
    /// `true` after the `ScreenshotCaptured` observer ran and the file is saved.
    saved: bool,
}

/// Wait for the first rendered frame, then spawn the screenshot entity.
///
/// Instead of guessing with a frame counter, we use `FrameCount` which Bevy
/// increments during the `Last` schedule — after the render pipeline has
/// completed. When `FrameCount.0 > 0`, at least one frame has fully rendered.
///
/// The Bevy screenshot pipeline then needs 2 more frames:
/// - This frame: spawn `Screenshot` entity in Update
/// - Next frame: ExtractSchedule extracts it, Render captures it
/// - Next frame: GPU readback completes, `ScreenshotCaptured` fires in Update
///
/// Observers handle the rest:
/// 1. `save_to_disk` writes the PNG file.
/// 2. `mark_screenshot_done` sets the resource flag so the quit system can exit.
fn wait_for_first_frame(
    mut state: ResMut<ScreenshotState>,
    filename: Res<AutoScreenshotFilename>,
    mut commands: Commands,
    frame_count: Res<FrameCount>,
    window_query: Query<Entity, With<Window>>,
    wait_frames: Res<WaitFrames>,
) {
    if state.triggered {
        return;
    }

    // The window must exist before we can screenshot it.
    if window_query.is_empty() {
        return;
    }

    // `FrameCount` increments during the `Last` schedule, after rendering completes.
    // Wait for the configured number of frames before taking the screenshot.
    // Complex 3D scenes may need more frames for all shaders to compile.
    if frame_count.0 >= wait_frames.0 {
        state.triggered = true;
        info!(
            "Taking screenshot after {} frames rendered (frame {}) -> {}",
            wait_frames.0, frame_count.0, filename.0
        );
        let path = filename.0.clone();
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path.clone()))
            .observe(move |_trigger: On<ScreenshotCaptured>| {
                info!("Screenshot saved to {}", path);
            })
            .observe(mark_screenshot_done);
    }
}

/// Observer on the screenshot entity: marks the screenshot as saved so the
/// quit system can exit on the next frame.
fn mark_screenshot_done(_trigger: On<ScreenshotCaptured>, mut state: ResMut<ScreenshotState>) {
    state.saved = true;
    info!("Screenshot capture complete");
}

/// After the screenshot file is written, exit the app.
fn quit_after_screenshot_saved(
    state: Res<ScreenshotState>,
    mut app_exit: bevy::ecs::message::MessageWriter<AppExit>,
) {
    if state.saved {
        info!("Exiting after screenshot");
        app_exit.write(AppExit::Success);
    }
}
