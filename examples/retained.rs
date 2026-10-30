//! Retained mode example — demonstrates persistent shapes managed by GradMeshHandle.
//!
//! Shows how to create shapes once, then update positions/rotations/scales
//! per-frame via handles. Uses batched instancing internally, eliminating
//! per-shape entities and material assets.
//!
//! Run with: `cargo run --example retained`

mod common;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{
    dual_directional, ShapeDetail, GradMeshMaterial, GradMeshPlugin,
    RetainedWorld, GradMeshHandle,
};
use common::labels::{spawn_label, position_labels};
use common::screenshot::AutoScreenshotPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "BevyGradmesh Retained Mode".into(),
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
            AutoScreenshotPlugin::new("screenshot_retained.png")
                .with_title("Retained Mode")
                .with_description("Persistent shapes via GradMeshHandle — batched instancing")
                .wait_frames(10)
        )
        .insert_resource(ClearColor(Color::srgb(0.1, 0.1, 0.12)))
        .add_systems(Startup, setup)
        .add_systems(PostUpdate, (update_retained_shapes, position_labels).chain())
        .run();
}

/// Stores the retained world and handles for our animated shapes.
#[derive(Resource)]
struct RetainedScene {
    world: RetainedWorld,
    handles: Vec<GradMeshHandle>,
}

#[derive(Component)]
struct RetainedBatch;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
) {
    // Camera — pulled back to capture cuboid at x=-5 and cylinder at x=5
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 5.0, 16.0).looking_at(Vec3::new(0.0, 0.8, 0.0), Vec3::Y),
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
    let ground = meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(24.0)));
    commands.spawn((
        Mesh3d(ground),
        MeshMaterial3d(materials.add(dual_directional(
            [0.07, 0.07, 0.09],
            [0.12, 0.12, 0.15],
            [0.0, 1.0, 0.0],
        ))),
        Transform::from_translation(Vec3::new(0.0, -0.01, 0.0)),
    ));

    // Create retained world and populate with shapes
    let mut world = RetainedWorld::new();

    // Orbiting spheres
    let mut handles = Vec::new();
    for i in 0..8 {
        let angle = i as f32 * std::f32::consts::TAU / 8.0;
        let x = angle.cos() * 3.0;
        let z = angle.sin() * 3.0;
        let handle = world.sphere(
            &mut commands,
            &mut meshes,
            Vec3::new(x, 1.0, z),
            0.25,
            ShapeDetail::Medium,
            [0.9, 0.4, 0.3, 1.0],
        );
        handles.push(handle);
    }

    // Static reference shapes
    let _cuboid = world.cuboid(
        &mut commands,
        &mut meshes,
        Vec3::new(-5.0, 0.5, 0.0),
        Vec3::splat(0.8),
        0.0,
        ShapeDetail::Medium,
        [0.3, 0.6, 0.9, 1.0],
    );

    let _cylinder = world.cylinder(
        &mut commands,
        &mut meshes,
        Vec3::new(5.0, 0.6, 0.0),
        0.3,
        1.2,
        ShapeDetail::Medium,
        [0.3, 0.9, 0.5, 1.0],
    );

    let _capsule = world.capsule(
        &mut commands,
        &mut meshes,
        Vec3::new(0.0, 0.4, -4.0),
        0.2,
        0.8,
        ShapeDetail::Medium,
        [0.8, 0.4, 0.9, 1.0],
    );

    // Flush initial state
    for entity in world.flush(&mut commands, meshes, materials) {
        commands.entity(entity).insert(RetainedBatch);
    }

    spawn_label(&mut commands, "Orbiting spheres (8 handles)", Vec3::new(0.0, 3.5, 0.0));
    spawn_label(&mut commands, "Cuboid", Vec3::new(-5.0, 1.5, 0.0));
    spawn_label(&mut commands, "Cylinder", Vec3::new(5.0, 1.7, 0.0));
    spawn_label(&mut commands, "Capsule", Vec3::new(0.0, 1.3, -4.0));

    commands.insert_resource(RetainedScene { world, handles });
}

fn update_retained_shapes(
    time: Res<Time>,
    mut scene: ResMut<RetainedScene>,
    mut commands: Commands,
    meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
    previous_batches: Query<Entity, With<RetainedBatch>>,
) {
    for entity in &previous_batches {
        commands.entity(entity).despawn();
    }
    let t = time.elapsed_secs();

    // Collect handles first to avoid borrow conflict
    let handles: Vec<_> = scene.handles.clone();

    // Update orbiting spheres
    for (i, handle) in handles.iter().enumerate() {
        let base_angle = i as f32 * std::f32::consts::TAU / 8.0;
        let angle = base_angle + t * 0.5;
        let x = angle.cos() * 3.0;
        let z = angle.sin() * 3.0;
        let y = 1.0 + (t + i as f32 * 0.5).sin() * 0.3;

        scene.world.set_position(*handle, Vec3::new(x, y, z));
    }

    // Flush updates to GPU
    for entity in scene.world.flush(&mut commands, meshes, materials) {
        commands.entity(entity).insert(RetainedBatch);
    }
}
