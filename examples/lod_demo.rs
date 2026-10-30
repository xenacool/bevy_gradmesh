//! LOD demo — demonstrates distance-based Level of Detail with dither cross-fade.
//!
//! Shows a row of identical shapes at increasing distances from the camera,
//! with LOD level labels. Demonstrates the dither cross-fade transition zone
//! that eliminates pop-in.
//!
//! Run with: `cargo run --example lod_demo`

mod common;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{
    dual_directional, Shape, ShapeDetail, GradMeshMaterial, GradMeshPlugin,
    LodConfig, InstancedStats,
};
use common::labels::{spawn_label, position_labels};
use common::screenshot::AutoScreenshotPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "BevyGradmesh LOD Demo".into(),
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
            AutoScreenshotPlugin::new("screenshot_lod_demo.png")
                .with_title("LOD Demo")
                .with_description("Distance-based LOD with dither cross-fade")
        )
        .insert_resource(ClearColor(Color::srgb(0.1, 0.1, 0.12)))
        .insert_resource(LodConfig {
            thresholds: [2.0, 6.0, 14.0, 30.0],
            transition_width: 2.0,
            hysteresis: 0.1,
        })
        .add_systems(Startup, setup)
        .add_systems(PostUpdate, (print_lod_stats, position_labels).chain())
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
) {
    // Camera — positioned to see the LOD progression
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 3.0, 0.0).looking_at(Vec3::new(0.0, 1.0, 15.0), Vec3::Y),
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
    let ground = meshes.add(Plane3d::new(Vec3::Y, Vec2::new(10.0, 40.0)));
    commands.spawn((
        Mesh3d(ground),
        MeshMaterial3d(materials.add(dual_directional(
            [0.07, 0.07, 0.09],
            [0.12, 0.12, 0.15],
            [0.0, 1.0, 0.0],
        ))),
        Transform::from_translation(Vec3::new(0.0, -0.01, 20.0)),
    ));

    let mut mat = |r, g, b| -> Handle<GradMeshMaterial> {
        materials.add(dual_directional(
            [r * 0.65, g * 0.65, b * 0.65],
            [r, g, b],
            [0.0, 1.0, 0.0],
        ))
    };

    // Title
    spawn_label(&mut commands, "LOD Levels by Distance", Vec3::new(0.0, 5.0, 0.0));

    // Row of spheres at increasing distances, each with a different LOD level
    let details = [
        (ShapeDetail::Extreme, "Extreme (32 seg)", 0.9, 0.3, 0.3),
        (ShapeDetail::High, "High (24 seg)", 0.9, 0.6, 0.2),
        (ShapeDetail::Medium, "Medium (16 seg)", 0.3, 0.9, 0.3),
        (ShapeDetail::Low, "Low (12 seg)", 0.3, 0.6, 0.9),
        (ShapeDetail::Minimal, "Minimal (8 seg)", 0.6, 0.3, 0.9),
    ];

    for (i, (detail, label, r, g, b)) in details.iter().enumerate() {
        let z = 5.0 + i as f32 * 6.0;
        let mesh = meshes.add(Shape::sphere(0.5, *detail).build());
        commands.spawn((
            Mesh3d(mesh),
            MeshMaterial3d(mat(*r, *g, *b)),
            Transform::from_translation(Vec3::new(0.0, 0.8, z)),
        ));
        spawn_label(&mut commands, label, Vec3::new(0.0, 1.8, z));

        // Distance marker
        let dist_label = format!("~{:.0}m from camera", z);
        spawn_label(&mut commands, &dist_label, Vec3::new(0.0, 0.1, z));
    }

    // Side-by-side comparison row: same shape, different LODs, same distance
    for (i, (detail, _label, r, g, b)) in details.iter().enumerate() {
        let x = -2.0 + i as f32 * 1.0;
        let mesh = meshes.add(Shape::sphere(0.3, *detail).build());
        commands.spawn((
            Mesh3d(mesh),
            MeshMaterial3d(mat(*r, *g, *b)),
            Transform::from_translation(Vec3::new(x, 0.5, 35.0)),
        ));
    }
    spawn_label(&mut commands, "Comparison at same distance", Vec3::new(0.0, 1.5, 35.0));
}

fn print_lod_stats(stats: Option<Res<InstancedStats>>) {
    if let Some(stats) = stats {
        if stats.batch_count > 0 {
            info!(
                "LOD: {} batches, {} instances ({} culled)",
                stats.batch_count, stats.total_instances, stats.culled_instances
            );
        }
    }
}
