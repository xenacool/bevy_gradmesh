//! Minimal getting-started example matching the "Your First Shape" snippet in the book.
//!
//! Spawns a single dual-directional gradient sphere lit by a directional light and gently
//! rotates it so both ends of the OKLAB gradient are visible.
//!
//! Run with: `cargo run --example hello_sphere`

mod common;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{
    GradMeshMaterial, GradMeshPlugin, Shape, ShapeDetail, dual_directional, opaque,
};
use common::screenshot::AutoScreenshotPlugin;

/// Run this example. Called by the WASM runner.
pub fn run(app: &mut App) {
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "BevyGradmesh Hello Sphere".into(),
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
        AutoScreenshotPlugin::new("screenshot_hello_sphere.png")
            .with_title("Hello Sphere")
            .with_description("Your first procedural shape with a dual-directional gradient"),
    )
    .insert_resource(ClearColor(Color::srgb(0.08, 0.09, 0.13)))
    .add_systems(Startup, setup)
    .add_systems(Update, spin);
}

fn main() {
    let mut app = App::new();
    run(&mut app);
    app.run();
}

#[derive(Component)]
struct HeroSphere;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 1.4, 4.5).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 12_000.0,
            ..default()
        },
        Transform::from_xyz(3.0, 5.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.55, 0.58, 0.7),
        brightness: 450.0,
        affects_lightmapped_meshes: false,
    });

    // Subtle ground disc beneath the sphere.
    let floor = meshes.add(Shape::cylinder(2.6, 0.06, ShapeDetail::Medium).build());
    commands.spawn((
        Mesh3d(floor),
        MeshMaterial3d(materials.add(opaque(0.14, 0.16, 0.21))),
        Transform::from_xyz(0.0, -1.25, 0.0),
    ));

    // Your first shape: a procedural sphere with a dual-directional gradient.
    let sphere = Shape::sphere(1.0, ShapeDetail::Medium).build();
    let material = dual_directional(
        [0.6, 0.4, 0.3],
        [0.96, 0.8, 0.69],
        [0.0, 1.0, 0.0],
    );
    commands.spawn((
        Mesh3d(meshes.add(sphere)),
        MeshMaterial3d(materials.add(material)),
        Transform::from_xyz(0.0, 0.1, 0.0),
        HeroSphere,
    ));
}

#[allow(clippy::needless_pass_by_value)]
fn spin(time: Res<Time>, mut query: Query<&mut Transform, With<HeroSphere>>) {
    let t = time.elapsed_secs();
    for mut transform in &mut query {
        transform.rotation = Quat::from_euler(EulerRot::YXZ, t * 0.65, (t * 0.45).sin() * 0.28, 0.0);
    }
}
