//! Animated 3D Bézier curves, `Shape::catmull_rom_3d`, and open & closed `Shape::polyline3d` tubes.
//!
//! Directly matches the animated Bézier and 3D polyline code snippets in `curves.md`:
//! - Top-left: animated cubic `Shape::bezier_curve` wave rebuilt in place each frame.
//! - Top-right: animated closed `Shape::catmull_rom_3d` ribbon loop.
//! - Bottom-left: open `Shape::polyline3d` 3D zigzag tube.
//! - Bottom-right: closed `Shape::polyline3d` 3D torus-knot loop.
//!
//! Run with: `cargo run --example animated_curves_3d`

mod common;

use std::f32::consts::TAU;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{GradMeshMaterial, GradMeshPlugin, Shape, ShapeDetail, Style};
use common::screenshot::{AutoScreenshotPlugin, spawn_world_label};

/// Run this example. Called by the WASM runner.
pub fn run(app: &mut App) {
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "BevyGradmesh Animated 3D Curves & Polylines".into(),
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
        AutoScreenshotPlugin::new("screenshot_animated_curves_3d.png")
            .animation(3.0)
            .with_title("Animated 3D Curves & Polylines")
            .with_description("Live bezier_curve, catmull_rom_3d, and open/closed polyline3d tubes"),
    )
    .insert_resource(ClearColor(Color::srgb(0.07, 0.08, 0.12)))
    .add_systems(Startup, setup)
    .add_systems(Update, animate_curves);
}

fn main() {
    let mut app = App::new();
    run(&mut app);
    app.run();
}

#[derive(Resource)]
struct AnimatedMeshes {
    bezier: Handle<Mesh>,
    catmull_rom: Handle<Mesh>,
}

#[derive(Component)]
struct SlowSpin;

fn bezier_points(t: f32) -> Vec<Vec3> {
    vec![
        Vec3::new(-2.4, 0.0, 0.0),
        Vec3::new(-0.8, (t * 1.7).sin() * 1.4, (t * 1.1).cos() * 0.7),
        Vec3::new(0.8, (t * 1.7).cos() * 1.4, -(t * 1.1).sin() * 0.7),
        Vec3::new(2.4, 0.0, 0.0),
    ]
}

fn catmull_points(t: f32) -> Vec<Vec3> {
    (0u16..8)
        .map(|i| {
            let a = f32::from(i) / 8.0 * TAU;
            let r = 1.45 + 0.35 * (a * 2.0 + t * 1.5).sin();
            let y = 0.55 * (a * 3.0 - t * 1.2).cos();
            Vec3::new(a.cos() * r, y, a.sin() * r * 0.75)
        })
        .collect()
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 1.5, 11.5).looking_at(Vec3::new(0.0, 0.1, 0.0), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 13_000.0,
            ..default()
        },
        Transform::from_xyz(4.0, 7.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.6, 0.62, 0.72),
        brightness: 500.0,
        affects_lightmapped_meshes: false,
    });

    // 1. Top-left: Animated cubic Bézier tube
    let bezier_handle = meshes.add(
        Shape::bezier_curve(bezier_points(0.0), 0.14, 16, ShapeDetail::Medium).build(),
    );
    commands.spawn((
        Mesh3d(bezier_handle.clone()),
        MeshMaterial3d(
            materials.add(
                Style::dual_directional(
                    Color::srgb(0.98, 0.36, 0.25),
                    Color::srgb(1.0, 0.86, 0.28),
                    Vec3::X,
                )
                .build(),
            ),
        ),
        Transform::from_xyz(-3.1, 1.65, 0.0),
    ));
    spawn_world_label(
        &mut commands,
        "Animated bezier_curve",
        Vec3::new(-3.1, 3.1, 0.0),
    );

    // 2. Top-right: Animated closed Catmull-Rom 3D tube
    let catmull_handle = meshes.add(
        Shape::catmull_rom_3d(catmull_points(0.0), 0.13, true, 10, ShapeDetail::Medium).build(),
    );
    commands.spawn((
        Mesh3d(catmull_handle.clone()),
        MeshMaterial3d(
            materials.add(
                Style::dual_directional(
                    Color::srgb(0.22, 0.42, 0.96),
                    Color::srgb(0.35, 0.94, 0.88),
                    Vec3::Y,
                )
                .build(),
            ),
        ),
        Transform::from_xyz(3.1, 1.65, 0.0),
    ));
    spawn_world_label(
        &mut commands,
        "Animated catmull_rom_3d",
        Vec3::new(3.1, 3.1, 0.0),
    );

    // 3. Bottom-left: Open 3D polyline zigzag
    let open_pts = vec![
        Vec3::new(-2.2, -0.5, 0.3),
        Vec3::new(-1.3, 0.7, -0.4),
        Vec3::new(-0.4, -0.6, 0.5),
        Vec3::new(0.5, 0.8, -0.3),
        Vec3::new(1.4, -0.5, 0.4),
        Vec3::new(2.2, 0.6, -0.2),
    ];
    commands.spawn((
        Mesh3d(meshes.add(Shape::polyline3d(open_pts, 0.13, false, 8, ShapeDetail::Medium).build())),
        MeshMaterial3d(
            materials.add(
                Style::dual_directional(
                    Color::srgb(0.18, 0.72, 0.42),
                    Color::srgb(0.68, 0.96, 0.35),
                    Vec3::X,
                )
                .build(),
            ),
        ),
        Transform::from_xyz(-3.1, -1.65, 0.0),
        SlowSpin,
    ));
    spawn_world_label(
        &mut commands,
        "polyline3d (open)",
        Vec3::new(-3.1, -0.25, 0.0),
    );

    // 4. Bottom-right: Closed 3D polyline trefoil loop
    let closed_pts: Vec<Vec3> = (0u16..36)
        .map(|i| {
            let a = f32::from(i) / 36.0 * TAU;
            let x = (a.sin() + 2.0 * (2.0 * a).sin()) * 0.48;
            let y = (a.cos() - 2.0 * (2.0 * a).cos()) * 0.42;
            let z = -(3.0 * a).sin() * 0.55;
            Vec3::new(x, y, z)
        })
        .collect();
    commands.spawn((
        Mesh3d(meshes.add(Shape::polyline3d(closed_pts, 0.12, true, 8, ShapeDetail::Medium).build())),
        MeshMaterial3d(
            materials.add(
                Style::dual_directional(
                    Color::srgb(0.65, 0.25, 0.92),
                    Color::srgb(0.98, 0.42, 0.76),
                    Vec3::Y,
                )
                .build(),
            ),
        ),
        Transform::from_xyz(3.1, -1.65, 0.0),
        SlowSpin,
    ));
    spawn_world_label(
        &mut commands,
        "polyline3d (closed)",
        Vec3::new(3.1, -0.25, 0.0),
    );

    commands.insert_resource(AnimatedMeshes {
        bezier: bezier_handle,
        catmull_rom: catmull_handle,
    });
}

#[allow(clippy::needless_pass_by_value)]
fn animate_curves(
    time: Res<Time>,
    handles: Res<AnimatedMeshes>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut spinners: Query<&mut Transform, With<SlowSpin>>,
) {
    let t = time.elapsed_secs();
    if let Some(mut mesh) = meshes.get_mut(&handles.bezier) {
        *mesh = Shape::bezier_curve(bezier_points(t), 0.14, 16, ShapeDetail::Medium).build();
    }
    if let Some(mut mesh) = meshes.get_mut(&handles.catmull_rom) {
        *mesh = Shape::catmull_rom_3d(catmull_points(t), 0.13, true, 10, ShapeDetail::Medium).build();
    }
    for mut tf in &mut spinners {
        tf.rotation = Quat::from_rotation_y(t * 0.45);
    }
}
