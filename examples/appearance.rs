//! Appearance animation demo - per-instance colour animation resolved on the GPU.
//!
//! Four rows of batched spheres, one per `AppearanceSnapshot` kind. Every row spreads
//! its phase over the instances, so the staggering is visible even in a still frame.
//! The CPU only uploads the snapshot once per instance; the shader animates it.
//!
//! Run with: `cargo run --example appearance`

mod common;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{
    AppearanceSnapshot, BatchRecorder, GradMeshMaterial, GradMeshPlugin, MixSpace, ShapeDetail,
    opaque,
};
use common::screenshot::{AutoScreenshotPlugin, spawn_world_label};

const COLUMNS: usize = 12;
const SPACING: f32 = 1.3;
const ROW_SPACING: f32 = 1.9;
/// Index of the `gradient_shift` row. It turns the gradient direction around the Y axis, so
/// that row needs a sideways gradient; a vertical one would not change.
const GRADIENT_ROW: usize = 3;


/// Run this example. Called by the WASM runner.
pub fn run(app: &mut App) {
    app
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "BevyGradmesh Appearance Animation".into(),
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
        .add_plugins(GradMeshPlugin)
        .add_plugins(
            AutoScreenshotPlugin::new("screenshot_appearance.png")
                .animation(2.0)
                .with_title("Appearance Animation")
                .with_description(
                    "Static, colour pulse, colour cycle and gradient shift - phase staggered per instance",
                )
                .wait_frames(30),
        )
        .insert_resource(ClearColor(Color::srgb(0.07, 0.07, 0.09)))
        .insert_resource(Recorder(BatchRecorder::new()))
        .add_systems(Startup, setup)
        .add_systems(PostUpdate, draw)
        ;
}

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "BevyGradmesh Appearance Animation".into(),
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
        .add_plugins(GradMeshPlugin)
        .add_plugins(
            AutoScreenshotPlugin::new("screenshot_appearance.png")
                .animation(2.0)
                .with_title("Appearance Animation")
                .with_description(
                    "Static, colour pulse, colour cycle and gradient shift - phase staggered per instance",
                )
                .wait_frames(30),
        )
        .insert_resource(ClearColor(Color::srgb(0.07, 0.07, 0.09)))
        .insert_resource(Recorder(BatchRecorder::new()))
        .add_systems(Startup, setup)
        .add_systems(PostUpdate, draw)
        .run();
}

#[derive(Resource)]
struct Recorder(BatchRecorder);

#[derive(Component)]
struct FrameEntity;

/// Row label and the snapshot for instance `i` of `n`.
const ROWS: [(&str, fn(f32) -> AppearanceSnapshot); 4] = [
    ("static_appearance", |_| AppearanceSnapshot::static_appearance()),
    ("color_pulse(1.0, phase)", |phase| AppearanceSnapshot::color_pulse(1.0, phase)),
    ("color_cycle(0.5, phase)", |phase| AppearanceSnapshot::color_cycle(0.5, phase)),
    ("gradient_shift(0.5, phase)", |phase| AppearanceSnapshot::gradient_shift(0.5, phase)),
];

fn row_z(row: usize) -> f32 {
    (row as f32 - (ROWS.len() as f32 - 1.0) * 0.5) * ROW_SPACING
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
) {
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::new(Vec3::Y, Vec2::new(14.0, 5.0)))),
        MeshMaterial3d(materials.add(opaque(0.1, 0.1, 0.13))),
        Transform::from_translation(Vec3::new(0.0, -0.55, 0.0)),
    ));
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 11.0, 10.5).looking_at(Vec3::new(0.0, 0.0, 0.2), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 15_000.0,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.6, 0.6, 0.7),
        brightness: 600.0,
        affects_lightmapped_meshes: false,
    });

    let left = -(COLUMNS as f32 - 1.0) * SPACING * 0.5;
    for (row, (label, _)) in ROWS.iter().enumerate() {
        spawn_world_label(&mut commands, label, Vec3::new(left + 1.5, 0.0, row_z(row) - 0.95));
    }
}

fn draw(
    mut commands: Commands,
    meshes: ResMut<Assets<Mesh>>,
    mut recorder: ResMut<Recorder>,
    previous: Query<Entity, With<FrameEntity>>,
) {
    for entity in &previous {
        commands.entity(entity).despawn();
    }

    let recorder = &mut recorder.0;
    recorder.begin();
    let left = -(COLUMNS as f32 - 1.0) * SPACING * 0.5;
    for (row, (_, appearance)) in ROWS.iter().enumerate() {
        for column in 0..COLUMNS {
            let phase = column as f32 / COLUMNS as f32;
            recorder.set_position(Vec3::new(left + column as f32 * SPACING, 0.0, row_z(row)));
            recorder.set_appearance(appearance(phase));
            let direction = if row == GRADIENT_ROW {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            recorder.sphere_dual_directional(
                0.5,
                ShapeDetail::Medium,
                [0.9, 0.15, 0.2],
                [0.2, 0.75, 0.95],
                direction,
                MixSpace::default(),
            );
        }
    }
    recorder.end();

    for entity in recorder.render_batches(&mut commands, meshes, None) {
        commands.entity(entity).insert(FrameEntity);
    }
}
