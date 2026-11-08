//! GPU clipmap terrain: an endless heightfield drawn around the camera from a
//! height callback. Detail halves with every ring, the rings blend into each
//! other without cracks, a hole is see-through, and the land is coloured by
//! height. The camera drifts for a while before the screenshot, so the height
//! texture scrolls: only the strips that came into view are recomputed.
//!
//! Run with: `cargo run --example clipmap --features clipmap`

mod common;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{Clipmap, ClipmapPlugin, ClipmapSettings, GradMeshPlugin, HeightSource};
use common::screenshot::{AutoScreenshotPlugin, spawn_world_label};

/// Centre and radius of the hole, in the terrain's local space.
const HOLE_CENTER: Vec2 = Vec2::new(14.0, 96.0);
const HOLE_RADIUS: f32 = 7.0;

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "BevyGradmesh Clipmap".into(),
                        resolution: (1280, 720).into(),
                        ..default()
                    }),
                    ..default()
                })
                .set(RenderPlugin {
                    synchronous_pipeline_compilation: true,
                    ..default()
                }),
        )
        .add_plugins((GradMeshPlugin, ClipmapPlugin))
        .add_plugins(
            AutoScreenshotPlugin::new("screenshot_clipmap.png")
                .with_title("GPU clipmap terrain")
                .with_description("Six rings from one callback, geomorphed, with a hole and height bands")
                .wait_frames(100),
        )
        .add_systems(Startup, setup)
        .add_systems(Update, drift)
        .run();
}

/// The land: a pure function of the position, with `None` for the hole.
fn land(p: Vec2) -> Option<f32> {
    if p.distance(HOLE_CENTER) < HOLE_RADIUS {
        return None;
    }
    let hills = 11.0 * (p.x * 0.021).sin() * (p.y * 0.017).cos();
    let ridges = 5.0 * (p.x * 0.07 + p.y * 0.05).sin();
    let rocks = 1.4 * (p.x * 0.23).sin() * (p.y * 0.19).sin();
    Some(hills + ridges + rocks)
}

#[derive(Component)]
struct Drifting;

fn setup(mut commands: Commands) {
    let camera = commands
        .spawn((
            Camera3d::default(),
            Transform::from_xyz(14.0, 45.0, 60.0).looking_at(Vec3::new(14.0, 0.0, 96.0), Vec3::Y),
            Drifting,
        ))
        .id();

    commands.spawn((
        DirectionalLight {
            illuminance: 12_000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(-60.0, 80.0, 30.0).looking_at(Vec3::new(0.0, 0.0, 70.0), Vec3::Y),
    ));
    commands.insert_resource(GlobalAmbientLight {
        brightness: 250.0,
        ..default()
    });

    commands.spawn(Clipmap::new(
        camera,
        HeightSource::callback(land),
        ClipmapSettings::default()
            .with_levels(6)
            .with_grid(64)
            .with_cell_size(1.0)
            .with_height_bands(&[
                (-30.0, Color::srgb(0.42, 0.30, 0.18)),
                (-7.0, Color::srgb(0.22, 0.45, 0.20)),
                (4.0, Color::srgb(0.45, 0.44, 0.42)),
                (13.0, Color::srgb(0.95, 0.96, 0.98)),
            ]),
    ));

    spawn_world_label(&mut commands, "hole", Vec3::new(HOLE_CENTER.x, 9.0, HOLE_CENTER.y));
}

/// The camera drifts along the land, so the rings follow it.
fn drift(mut cameras: Query<&mut Transform, With<Drifting>>) {
    for mut transform in &mut cameras {
        // Drifts for a while, then holds still so the hole is in view.
        transform.translation.x = (transform.translation.x + 0.12).min(26.0);
        transform.translation.z = (transform.translation.z + 0.1).min(70.0);
    }
}
