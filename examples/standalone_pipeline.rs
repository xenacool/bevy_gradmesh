//! Standalone pipeline example — demonstrates the PBR-free render pipeline.
//!
//! Shows a scene with many different materials that would cause descriptor pool
//! pressure in PBR mode, but works efficiently with the standalone pipeline.
//! One pipeline, one bind group layout, one descriptor set.
//!
//! Run with: `cargo run --example standalone_pipeline`

mod common;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{
    ShapeDetail, GradMeshPlugin,
    BatchRecorder, StandaloneRenderPlugin, render_standalone_batches,
};
use common::labels::{spawn_label, position_labels};
use common::screenshot::AutoScreenshotPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "BevyGradmesh Standalone Pipeline".into(),
                resolution: (1280, 720).into(),
                ..default()
            }),
            ..default()
        }).set(RenderPlugin {
            synchronous_pipeline_compilation: true,
            ..default()
        }))
        .add_plugins(GradMeshPlugin::default())
        .add_plugins(StandaloneRenderPlugin)
        .add_plugins(
            AutoScreenshotPlugin::new("screenshot_standalone_pipeline.png")
                .with_title("Standalone Pipeline")
                .with_description("PBR-free rendering — 0 descriptor sets vs N specializations")
        )
        .insert_resource(ClearColor(Color::srgb(0.08, 0.08, 0.1)))
        .add_systems(Startup, setup)
        .add_systems(PostUpdate, (render_standalone_scene, position_labels).chain())
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
) {
    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 4.0, 14.0).looking_at(Vec3::new(0.0, 1.5, 0.0), Vec3::Y),
    ));

    // Ground plane (using standalone batch)
    let ground = meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(30.0)));
    // Note: ground is rendered via the standalone batch system below

    spawn_label(&mut commands, "Standalone Pipeline", Vec3::new(0.0, 5.0, 0.0));
    spawn_label(&mut commands, "1 pipeline, 1 bind group, 0 PBR overhead", Vec3::new(0.0, 4.5, 0.0));
    spawn_label(&mut commands, "Mobile-friendly: Mali-G715, Adreno", Vec3::new(0.0, 4.1, 0.0));
}

fn render_standalone_scene(
    time: Res<Time>,
    mut commands: Commands,
    meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
) {
    let t = time.elapsed_secs();
    let mut recorder = BatchRecorder::new();
    recorder.begin();

    // Ground plane
    recorder.set_position(Vec3::new(0.0, -0.01, 0.0));
    recorder.set_scale(Vec3::new(15.0, 1.0, 15.0));
    recorder.sphere(0.5, ShapeDetail::Low, [0.08, 0.08, 0.1, 1.0]);

    // Grid of many different colored spheres — each would be a separate
    // material specialization in PBR mode, but here they're all one batch
    for row in 0..6 {
        for col in 0..10 {
            let x = -5.0 + col as f32 * 1.1;
            let z = -2.0 + row as f32 * 1.0;
            let y = 0.5 + ((t + col as f32 + row as f32) * 0.5).sin() * 0.2;

            // Vary colors to demonstrate per-instance color attributes
            let r = ((col * 37 + row * 53) % 255) as f32 / 255.0;
            let g = ((col * 71 + row * 89) % 255) as f32 / 255.0;
            let b = ((col * 113 + row * 131) % 255) as f32 / 255.0;

            recorder.set_position(Vec3::new(x, y, z));
            recorder.sphere(0.2, ShapeDetail::Medium, [r, g, b, 1.0]);
        }
    }

    // Different shapes to show mesh variety
    recorder.set_position(Vec3::new(-6.0, 0.5, 5.0));
    recorder.cuboid(Vec3::splat(0.6), 0.0, ShapeDetail::Medium, [0.9, 0.3, 0.3, 1.0]);

    recorder.set_position(Vec3::new(-4.0, 0.6, 5.0));
    recorder.cylinder(0.3, 1.2, ShapeDetail::Medium, [0.3, 0.9, 0.3, 1.0]);

    recorder.set_position(Vec3::new(-2.0, 0.5, 5.0));
    recorder.cone(0.3, 1.0, ShapeDetail::Medium, [0.3, 0.3, 0.9, 1.0]);

    recorder.set_position(Vec3::new(0.0, 0.4, 5.0));
    recorder.capsule(0.2, 0.8, ShapeDetail::Medium, [0.9, 0.8, 0.2, 1.0]);

    recorder.set_position(Vec3::new(2.0, 0.5, 5.0));
    recorder.torus(0.3, 0.1, 0.0, std::f32::consts::TAU, ShapeDetail::Medium, [0.9, 0.4, 0.6, 1.0]);

    recorder.set_position(Vec3::new(4.0, 0.5, 5.0));
    recorder.pyramid(4, 0.3, 1.0, ShapeDetail::Medium, [0.4, 0.8, 0.9, 1.0]);

    recorder.set_position(Vec3::new(6.0, 0.5, 5.0));
    recorder.hemisphere(0.35, true, ShapeDetail::Medium, [0.6, 0.4, 0.9, 1.0]);

    recorder.end();
    render_standalone_batches(&mut recorder, &mut commands, meshes, None);
}
