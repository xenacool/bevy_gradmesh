//! Standalone pipeline demo — renders gradient shapes without any PBR material.
//!
//! Every shape goes through `StandaloneRenderPlugin`: one pipeline, one view
//! bind group, no `StandardMaterial`, no lights. This is the mobile-friendly path.
//!
//! Run with: `cargo run --example standalone`

mod common;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{BatchRecorder, GradMeshPlugin, ShapeDetail, render_standalone_batches};
use common::screenshot::AutoScreenshotPlugin;

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "BevyGradmesh Standalone Pipeline".into(),
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
            AutoScreenshotPlugin::new("screenshot_standalone.png")
                .with_title("Standalone Pipeline")
                .with_description("Spheres and boxes drawn without any PBR material"),
        )
        .insert_resource(ClearColor(Color::srgb(0.08, 0.08, 0.1)))
        .add_systems(Startup, setup)
        .add_systems(PostUpdate, draw_standalone)
        .run();
}

#[derive(Resource)]
struct StandaloneRecorder(BatchRecorder);

#[derive(Component)]
struct StandaloneMarker;

fn setup(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 9.0, 14.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.spawn((
        Text::new("Standalone pipeline: no StandardMaterial, no lights"),
        TextFont {
            font_size: 24.0.into(),
            ..default()
        },
        TextColor(Color::srgb(1.0, 1.0, 1.0)),
        Node {
            position_type: PositionType::Absolute,
            top: px(16),
            left: px(16),
            ..default()
        },
    ));

    commands.insert_resource(StandaloneRecorder(BatchRecorder::new()));
}

fn draw_standalone(
    time: Res<Time>,
    mut commands: Commands,
    meshes: ResMut<Assets<Mesh>>,
    mut recorder: ResMut<StandaloneRecorder>,
    query: Query<Entity, With<StandaloneMarker>>,
) {
    for entity in &query {
        commands.entity(entity).despawn();
    }

    recorder.0.begin();

    let grid = 12;
    let spacing = 1.4;
    for x in 0..grid {
        for z in 0..grid {
            let px = (x as f32 - grid as f32 * 0.5) * spacing;
            let pz = (z as f32 - grid as f32 * 0.5) * spacing;
            let py = (px * px + pz * pz) * 0.01 + (time.elapsed_secs() * 2.0 + px).sin() * 0.15;
            recorder.0.set_position(Vec3::new(px, py, pz));

            let r = ((px / 8.0 + 1.0) * 0.5).clamp(0.0, 1.0);
            let g = ((pz / 8.0 + 1.0) * 0.5).clamp(0.0, 1.0);
            let b = 0.4;

            if (x + z) % 2 == 0 {
                recorder.0.sphere(0.45, ShapeDetail::Medium, [r, g, b, 1.0]);
            } else {
                recorder
                    .0
                    .cuboid(Vec3::splat(0.7), 0.1, ShapeDetail::Low, [b, g, r, 1.0]);
            }
        }
    }

    let _ = recorder.0.end();

    let spawned = render_standalone_batches(&mut recorder.0, &mut commands, meshes, None);
    for entity in spawned {
        commands.entity(entity).insert(StandaloneMarker);
    }
}
