//! Deformation example — demonstrates GPU-side vertex shader deformation.
//!
//! Shows all three deformation types side by side: wobble (jelly-like),
//! pulse (breathing), and morph (directional displacement). All animation
//! is resolved on the GPU with zero per-frame CPU overhead.
//!
//! Run with: `cargo run --example deformation`

mod common;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{
    dual_directional, ShapeDetail, GradMeshMaterial, GradMeshPlugin,
    DeformationParams, BatchRecorder,
};
use common::labels::{spawn_label, position_labels};
use common::screenshot::AutoScreenshotPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "BevyGradmesh Deformation".into(),
                resolution: (1280, 720).into(),
                ..default()
            }),
            ..default()
        }).set(RenderPlugin {
            synchronous_pipeline_compilation: true,
            ..default()
        }))
        .add_plugins(GradMeshPlugin::default())
        .add_plugins(
            AutoScreenshotPlugin::new("screenshot_deformation.png")
                .with_title("GPU Deformation")
                .with_description("Wobble, Pulse, and Morph — zero CPU overhead")
        )
        .insert_resource(ClearColor(Color::srgb(0.1, 0.1, 0.12)))
        .add_systems(Startup, setup)
        .add_systems(PostUpdate, (update_deformations, position_labels).chain())
        .run();
}

/// Tracks the current time for deformation animation.
#[derive(Resource, Default)]
struct DeformTime(f32);

#[derive(Component)]
struct DeformationBatch;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
) {
    commands.insert_resource(DeformTime(0.0));

    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 3.0, 10.0).looking_at(Vec3::new(0.0, 1.5, 0.0), Vec3::Y),
    ));

    // Lighting
    commands.spawn((
        DirectionalLight {
            illuminance: 15_000.0,
            ..default()
        },
        Transform::from_xyz(5.0, 8.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn(AmbientLight {
        color: Color::srgb(0.6, 0.6, 0.7),
        brightness: 300.0,
        affects_lightmapped_meshes: false,
    });

    // Ground plane
    let ground = meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(20.0)));
    commands.spawn((
        Mesh3d(ground),
        MeshMaterial3d(materials.add(dual_directional(
            [0.07, 0.07, 0.09],
            [0.12, 0.12, 0.15],
            [0.0, 1.0, 0.0],
        ))),
        Transform::from_translation(Vec3::new(0.0, -0.01, 0.0)),
    ));

    // Title labels
    spawn_label(&mut commands, "Wobble (jelly)", Vec3::new(-4.0, 3.5, 0.0));
    spawn_label(&mut commands, "Pulse (breathe)", Vec3::new(0.0, 3.5, 0.0));
    spawn_label(&mut commands, "Morph (shift)", Vec3::new(4.0, 3.5, 0.0));

    // Subtitle labels
    spawn_label(&mut commands, "sinusoidal normal displacement", Vec3::new(-4.0, -0.5, 0.0));
    spawn_label(&mut commands, "uniform scale oscillation", Vec3::new(0.0, -0.5, 0.0));
    spawn_label(&mut commands, "directional vertex push", Vec3::new(4.0, -0.5, 0.0));
}

fn update_deformations(
    time: Res<Time>,
    mut deform_time: ResMut<DeformTime>,
    mut commands: Commands,
    meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
    previous_batches: Query<Entity, With<DeformationBatch>>,
) {
    for entity in &previous_batches {
        commands.entity(entity).despawn();
    }
    deform_time.0 += time.delta_secs();
    let t = deform_time.0;

    // Use a batch recorder to render deformed shapes each frame
    let mut recorder = BatchRecorder::new();
    recorder.begin();

    // Wobble: jelly-like sinusoidal displacement
    recorder.set_position(Vec3::new(-4.0, 1.5, 0.0));
    recorder.set_deformation(DeformationParams::wobble(0.15, 3.0, 0.0));
    recorder.sphere(0.6, ShapeDetail::High, [0.9, 0.3, 0.3, 1.0]);

    // Pulse: breathing scale oscillation
    recorder.set_position(Vec3::new(0.0, 1.5, 0.0));
    recorder.set_deformation(DeformationParams::pulse(0.2, 1.5, 0.0));
    recorder.sphere(0.6, ShapeDetail::High, [0.3, 0.9, 0.3, 1.0]);

    // Morph: directional displacement
    recorder.set_position(Vec3::new(4.0, 1.5, 0.0));
    recorder.set_deformation(DeformationParams::morph(0.12, 2.0, 0.0));
    recorder.sphere(0.6, ShapeDetail::High, [0.3, 0.3, 0.9, 1.0]);

    // Additional row: wobble with different phase offsets
    for i in 0..5 {
        let x = -2.0 + i as f32 * 1.0;
        recorder.set_position(Vec3::new(x, 0.5, -2.0));
        recorder.set_deformation(DeformationParams::wobble(0.08, 2.5, i as f32 * 0.3));
        recorder.sphere(0.25, ShapeDetail::Medium, [0.95, 0.6, 0.2, 1.0]);
    }

    // Pulse row with staggered phases
    for i in 0..5 {
        let x = -2.0 + i as f32 * 1.0;
        recorder.set_position(Vec3::new(x, 0.5, 2.0));
        recorder.set_deformation(DeformationParams::pulse(0.15, 1.0 + i as f32 * 0.2, i as f32 * 0.2));
        recorder.sphere(0.25, ShapeDetail::Medium, [0.2, 0.7, 0.95, 1.0]);
    }

    recorder.end();
    for entity in recorder.render_batches(&mut commands, meshes, materials, None) {
        commands.entity(entity).insert(DeformationBatch);
    }

    let _ = t; // time tracked for potential UI display
}
