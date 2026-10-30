//! Culling demo — demonstrates per-instance frustum culling.
//!
//! Shows a large grid of instances with real-time stats: total instances,
//! culled instances, and culling percentage. Demonstrates how instances
//! outside the camera's view frustum are automatically excluded from rendering.
//!
//! Run with: `cargo run --example culling_demo`

mod common;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{
    dual_directional, ShapeDetail, GradMeshMaterial, GradMeshPlugin,
    BatchRecorder, InstancedStats,
};
use common::labels::{spawn_label, position_labels};
use common::screenshot::AutoScreenshotPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "BevyGradmesh Culling Demo".into(),
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
            AutoScreenshotPlugin::new("screenshot_culling_demo.png")
                .with_title("Frustum Culling")
                .with_description("Per-instance AABB culling against camera frustum")
        )
        .insert_resource(ClearColor(Color::srgb(0.1, 0.1, 0.12)))
        .add_systems(Startup, setup)
        .add_systems(PostUpdate, (render_culling_grid, print_culling_stats, position_labels).chain())
        .run();
}

/// Tracks whether stats have been printed (to avoid spam).
#[derive(Resource, Default)]
struct StatsPrinted(bool);

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
) {
    commands.insert_resource(StatsPrinted(false));

    // Camera — positioned to see only part of the grid
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 8.0, 0.0).looking_at(Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
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
    let ground = meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(40.0)));
    commands.spawn((
        Mesh3d(ground),
        MeshMaterial3d(materials.add(dual_directional(
            [0.07, 0.07, 0.09],
            [0.12, 0.12, 0.15],
            [0.0, 1.0, 0.0],
        ))),
        Transform::from_translation(Vec3::new(0.0, -0.01, 0.0)),
    ));

    spawn_label(&mut commands, "Frustum Culling Demo", Vec3::new(0.0, 4.0, -8.0));
    spawn_label(&mut commands, "100 instances in a 10x10 grid", Vec3::new(0.0, 3.5, -8.0));
    spawn_label(&mut commands, "Only visible instances are rendered", Vec3::new(0.0, 3.1, -8.0));
}

fn render_culling_grid(
    mut commands: Commands,
    meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
) {
    let mut recorder = BatchRecorder::new();
    recorder.begin();

    // 10x10 grid of spheres spread across a large area
    for row in 0..10 {
        for col in 0..10 {
            let x = -4.5 + col as f32 * 1.0;
            let z = -4.5 + row as f32 * 1.0;

            // Color by position to show which are visible
            let r = col as f32 / 9.0;
            let b = row as f32 / 9.0;

            recorder.set_position(Vec3::new(x, 0.3, z));
            recorder.sphere(0.2, ShapeDetail::Medium, [r, 0.3, b, 1.0]);
        }
    }

    recorder.end();
    recorder.render_batches(&mut commands, meshes, materials, None);
}

fn print_culling_stats(
    stats: Option<Res<InstancedStats>>,
    mut printed: ResMut<StatsPrinted>,
    time: Res<Time>,
) {
    if let Some(stats) = stats {
        if stats.total_instances_before_cull > 0 && !printed.0 {
            let culled_pct = if stats.total_instances_before_cull > 0 {
                (stats.culled_instances as f32 / stats.total_instances_before_cull as f32) * 100.0
            } else {
                0.0
            };
            info!(
                "Culling: {} total, {} visible, {} culled ({:.1}%)",
                stats.total_instances_before_cull,
                stats.total_instances,
                stats.culled_instances,
                culled_pct
            );
            printed.0 = true;
        }
    }
    let _ = time;
}
