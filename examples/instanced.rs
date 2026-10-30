//! Instanced rendering demo — showcases the `BatchRecorder` API with many shapes.
//!
//! Demonstrates how the batch recorder groups thousands of shapes into batches
//! for reduced draw calls. Uses the GPU instanced render pipeline for single
//! `draw_indexed` calls per batch.
//!
//! Run with: `cargo run --example instanced`

mod common;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{
    ShapeDetail,
    BatchRecorder, GradMeshPlugin,
};
use common::screenshot::AutoScreenshotPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "BevyGradmesh Instanced Rendering".into(),
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
            AutoScreenshotPlugin::new("screenshot_instanced.png")
                .with_title("Instanced Rendering")
                .with_description("500 spheres rendered in a single GPU batch")
        )
        .insert_resource(ClearColor(Color::srgb(0.08, 0.08, 0.1)))
        .add_systems(Startup, setup)
        .add_systems(PostUpdate, draw_instanced)
        .run();
}

/// Stores the recorder for instanced mode drawing.
#[derive(Resource)]
struct InstancedRecorder(BatchRecorder);

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
    mut materials: ResMut<Assets<bevy_gradmesh::GradMeshMaterial>>,
) {
    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 15.0, 20.0).looking_at(Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
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
        color: Color::srgb(0.5, 0.5, 0.6),
        brightness: 200.0,
        affects_lightmapped_meshes: false,
    });

    // Bowl-shaped ground — a large sphere centered below the scene so the
    // smaller spheres appear to rest on a curved surface rather than floating
    // above a flat plane.
    let ground = meshes.add(
        bevy_gradmesh::Shape::sphere(14.0, ShapeDetail::Medium).build()
    );
    commands.spawn((
        Mesh3d(ground),
        MeshMaterial3d(materials.add(bevy_gradmesh::opaque(0.1, 0.1, 0.12))),
        Transform::from_translation(Vec3::new(0.0, -18.0, 0.0)),
    ));

    // Label
    commands.spawn((
        Text::new("Instanced Rendering: 500 spheres in 1 batch"),
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

    // Label for stats
    commands.spawn((
        Text::new(""),
        TextFont {
            font_size: 18.0.into(),
            ..default()
        },
        TextColor(Color::srgb(0.8, 0.8, 0.8)),
        Node {
            position_type: PositionType::Absolute,
            top: px(48),
            left: px(16),
            ..default()
        },
    ));

    // Initialize recorder
    commands.insert_resource(InstancedRecorder(BatchRecorder::new()));
}

/// Marker for instanced entities so we can despawn them each frame.
#[derive(Component)]
struct InstancedMarker;

/// Per-frame instanced mode drawing.
fn draw_instanced(
    time: Res<Time>,
    mut commands: Commands,
    meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
    mut materials: ResMut<Assets<bevy_gradmesh::GradMeshMaterial>>,
    mut recorder: ResMut<InstancedRecorder>,
    query: Query<Entity, With<InstancedMarker>>,
) {
    // Remove previous frame's entities
    for entity in query.iter() {
        commands.entity(entity).despawn();
    }

    recorder.0.begin();

    // Draw 500 spheres in a grid pattern
    let grid_size = 20;
    let spacing = 1.5;
    let mut count = 0;
    for x in 0..grid_size {
        for z in 0..(grid_size - x) {
            if count >= 500 {
                break;
            }

            let px = (x as f32 - grid_size as f32 * 0.5) * spacing;
            let pz = (z as f32 - grid_size as f32 * 0.5) * spacing;
            let py = (px * px + pz * pz) * 0.01 + 0.5 + (time.elapsed_secs() * 2.0).sin() * 0.3;

            recorder.0.set_position(Vec3::new(px, py, pz));

            // Color based on position
            let r = ((px / 15.0 + 1.0) * 0.5).clamp(0.0, 1.0);
            let g = ((pz / 15.0 + 1.0) * 0.5).clamp(0.0, 1.0);
            let b = (py / 3.0).clamp(0.0, 1.0);

            recorder.0.sphere(0.3, ShapeDetail::Low, [r, g, b, 1.0]);
            count += 1;
        }
        if count >= 500 {
            break;
        }
    }

    let _batches = recorder.0.end();

    // Render batches as native Bevy entities — Bevy handles instancing automatically
    let spawned_entities = recorder.0.render_batches(&mut commands, meshes, materials, None);

    // Tag spawned entities for next frame's cleanup
    for entity_id in spawned_entities {
        commands.entity(entity_id).insert(InstancedMarker);
    }

    println!(
        "Frame: {} instances in {} batches ({} dirty)",
        recorder.0.total_instances(),
        recorder.0.batch_count(),
        recorder.0.dirty_batch_count(),
    );
}
