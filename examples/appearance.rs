//! Appearance animation example — demonstrates per-instance color animation.
//!
//! Shows a grid of shapes, each with a different appearance animation type
//! (ColorPulse, ColorCycle, GradientShift) and staggered phases. All color
//! animation is resolved in the vertex shader with no per-frame CPU updates.
//!
//! Run with: `cargo run --example appearance`

mod common;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{
    dual_directional, ShapeDetail, GradMeshMaterial, GradMeshPlugin,
    AppearanceSnapshot, BatchRecorder,
};
use common::labels::{spawn_label, position_labels};
use common::screenshot::AutoScreenshotPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "BevyGradmesh Appearance Animation".into(),
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
            AutoScreenshotPlugin::new("screenshot_appearance.png")
                .with_title("Appearance Animation")
                .with_description("ColorPulse, ColorCycle, GradientShift — no CPU updates")
        )
        .insert_resource(ClearColor(Color::srgb(0.08, 0.08, 0.1)))
        .add_systems(Startup, setup)
        .add_systems(PostUpdate, (render_appearance_grid, position_labels).chain())
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
) {
    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 5.0, 14.0).looking_at(Vec3::new(0.0, 1.5, 0.0), Vec3::Y),
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
    let ground = meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(30.0)));
    commands.spawn((
        Mesh3d(ground),
        MeshMaterial3d(materials.add(dual_directional(
            [0.05, 0.05, 0.07],
            [0.1, 0.1, 0.12],
            [0.0, 1.0, 0.0],
        ))),
        Transform::from_translation(Vec3::new(0.0, -0.01, 0.0)),
    ));

    // Section labels
    spawn_label(&mut commands, "ColorPulse (blend primary/secondary)", Vec3::new(-5.0, 4.2, 0.0));
    spawn_label(&mut commands, "ColorCycle (hue shift)", Vec3::new(0.0, 4.2, 0.0));
    spawn_label(&mut commands, "GradientShift (rotate direction)", Vec3::new(5.0, 4.2, 0.0));
}

#[derive(Component)]
struct AppearanceBatch;

fn render_appearance_grid(
    mut commands: Commands,
    meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
    previous_batches: Query<Entity, With<AppearanceBatch>>,
) {
    for entity in &previous_batches {
        commands.entity(entity).despawn();
    }
    let mut recorder = BatchRecorder::new();
    recorder.begin();

    // ColorPulse grid: 5x5 spheres pulsing between colors
    for row in 0..5 {
        for col in 0..5 {
            let x = -7.0 + col as f32 * 1.0;
            let z = -2.0 + row as f32 * 0.8;
            let phase = (row * 5 + col) as f32 / 25.0;
            let speed = 1.0 + (row % 3) as f32 * 0.5;

            recorder.set_position(Vec3::new(x, 0.5, z));
            recorder.set_appearance(AppearanceSnapshot::color_pulse(speed, phase));
            recorder.sphere(0.2, ShapeDetail::Medium, [0.9, 0.3, 0.3, 1.0]);
        }
    }

    // ColorCycle grid: 5x5 spheres cycling through hues
    for row in 0..5 {
        for col in 0..5 {
            let x = -2.5 + col as f32 * 1.0;
            let z = -2.0 + row as f32 * 0.8;
            let phase = (row * 5 + col) as f32 / 25.0;
            let speed = 0.5 + (col % 3) as f32 * 0.3;

            recorder.set_position(Vec3::new(x, 0.5, z));
            recorder.set_appearance(AppearanceSnapshot::color_cycle(speed, phase));
            recorder.sphere(0.2, ShapeDetail::Medium, [0.3, 0.9, 0.3, 1.0]);
        }
    }

    // GradientShift grid: 5x5 spheres with rotating gradient direction
    for row in 0..5 {
        for col in 0..5 {
            let x = 3.0 + col as f32 * 1.0;
            let z = -2.0 + row as f32 * 0.8;
            let phase = (row * 5 + col) as f32 / 25.0;
            let speed = 0.8 + (row % 2) as f32 * 0.4;

            recorder.set_position(Vec3::new(x, 0.5, z));
            recorder.set_appearance(AppearanceSnapshot::gradient_shift(speed, phase));
            recorder.sphere(0.2, ShapeDetail::Medium, [0.3, 0.3, 0.9, 1.0]);
        }
    }

    // Bottom row: mixed animation types for comparison
    for i in 0..15 {
        let x = -7.0 + i as f32 * 1.0;
        let phase = i as f32 / 15.0;

        recorder.set_position(Vec3::new(x, 0.5, 3.0));
        let anim_type = i % 3;
        match anim_type {
            0 => recorder.set_appearance(AppearanceSnapshot::color_pulse(1.5, phase)),
            1 => recorder.set_appearance(AppearanceSnapshot::color_cycle(1.5, phase)),
            2 => recorder.set_appearance(AppearanceSnapshot::gradient_shift(1.5, phase)),
            _ => {}
        }
        recorder.sphere(0.18, ShapeDetail::Low, [0.9, 0.8, 0.2, 1.0]);
    }

    recorder.end();
    for entity in recorder.render_batches(&mut commands, meshes, materials, None) {
        commands.entity(entity).insert(AppearanceBatch);
    }
}
