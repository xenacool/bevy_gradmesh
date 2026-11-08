//! Retained world example - demonstrates the `RetainedWorld` API for persistent, updatable shapes.
//!
//! Shows how to create shapes that persist across frames and can be updated
//! efficiently via handles, without recreating entities each frame.
//!
//! Run with: `cargo run --example retained`

mod common;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{GradMeshHandle, GradMeshMaterial, GradMeshPlugin, RetainedWorld, ShapeDetail};
use common::screenshot::AutoScreenshotPlugin;

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "BevyGradmesh Retained World".into(),
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
            AutoScreenshotPlugin::new("screenshot_retained.png")
                .with_title("Retained World")
                .with_description("Persistent shapes updated via handles each frame"),
        )
        .insert_resource(ClearColor(Color::srgb(0.08, 0.08, 0.1)))
        .add_systems(Startup, setup)
        .add_systems(PostUpdate, (update_retained_shapes, position_labels).chain())
        .run();
}

/// Retained world and handles for animated shapes.
#[derive(Resource)]
struct RetainedScene {
    world: RetainedWorld,
    /// Handles for the orbiting spheres
    spheres: Vec<GradMeshHandle>,
    /// Handle for the pulsing central shape
    center: GradMeshHandle,
}

#[derive(Component)]
struct WorldLabel(Vec3);

#[derive(Component)]
struct RetainedMarker;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
) {
    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 5.5, 6.5).looking_at(Vec3::new(0.0, 0.7, 0.0), Vec3::Y),
    ));

    // Lighting
    commands.spawn((
        DirectionalLight {
            illuminance: 15_000.0,
            ..default()
        },
        Transform::from_xyz(5.0, 8.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.5, 0.5, 0.6),
        brightness: 900.0,
        affects_lightmapped_meshes: false,
    });

    // Ground plane
    let ground = meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(22.0)));
    commands.spawn((
        Mesh3d(ground),
        MeshMaterial3d(materials.add(bevy_gradmesh::opaque(0.2, 0.19, 0.26))),
        Transform::from_translation(Vec3::new(0.0, -0.01, 0.0)),
    ));

    // Labels
    spawn_label(&mut commands, "Orbiting Spheres", Vec3::new(0.0, 2.3, -2.0));
    spawn_label(&mut commands, "Pulsing Center", Vec3::new(0.0, 0.0, 0.6));

    // Create retained world and add shapes
    let mut world = RetainedWorld::new();

    // Central pulsing sphere
    let center = world.sphere(
        &mut commands,
        &mut meshes,
        Vec3::new(0.0, 0.9, 0.0),
        0.7,
        ShapeDetail::High,
        [1.0, 0.82, 0.5, 1.0],
    );

    // Orbiting spheres
    let sphere_count = 12;
    let mut spheres = Vec::with_capacity(sphere_count);
    for i in 0..sphere_count {
        let hue = i as f32 / sphere_count as f32;
        let color = hsl_to_rgb(hue, 0.7, 0.78);
        let handle = world.sphere(
            &mut commands,
            &mut meshes,
            Vec3::ZERO,
            0.3,
            ShapeDetail::Medium,
            [color[0], color[1], color[2], 1.0],
        );
        spheres.push(handle);
    }

    commands.insert_resource(RetainedScene {
        world,
        spheres,
        center,
    });
}

/// Update retained shapes each frame: orbit spheres and pulse center.
fn update_retained_shapes(
    time: Res<Time>,
    mut scene: ResMut<RetainedScene>,
    mut commands: Commands,
    meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
    query: Query<Entity, With<RetainedMarker>>,
) {
    let t = time.elapsed_secs();

    // Collect sphere data first to avoid borrow conflicts
    let sphere_count = scene.spheres.len();
    let spheres: Vec<_> = scene.spheres.clone();
    let center = scene.center;

    // Orbiting spheres
    for (i, handle) in spheres.iter().enumerate() {
        let angle = (i as f32 / sphere_count as f32) * std::f32::consts::TAU + t * 0.8;
        let radius = 3.0 + (t * 0.5 + i as f32 * 0.3).sin() * 0.5;
        let x = angle.cos() * radius;
        let z = angle.sin() * radius;
        let y = 0.9 + (t + i as f32 * 0.5).sin() * 0.4;

        scene.world.set_transform(
            *handle,
            Vec3::new(x, y, z),
            Quat::IDENTITY,
            Vec3::ONE,
        );
    }

    // Pulsing center
    let scale = 1.0 + t.sin() * 0.3;
    scene.world.set_transform(
        center,
        Vec3::new(0.0, 0.9, 0.0),
        Quat::IDENTITY,
        Vec3::splat(scale),
    );

    // Flush retained world to batches
    // Remove previous entities
    for entity in query.iter() {
        commands.entity(entity).despawn();
    }
    let spawned = scene.world.flush(&mut commands, meshes);
    for entity_id in spawned {
        commands.entity(entity_id).insert(RetainedMarker);
    }
}

/// Simple HSL to RGB conversion.
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> [f32; 3] {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;

    let (r1, g1, b1) = if h < 1.0 / 6.0 {
        (c, x, 0.0)
    } else if h < 2.0 / 6.0 {
        (x, c, 0.0)
    } else if h < 3.0 / 6.0 {
        (0.0, c, x)
    } else if h < 4.0 / 6.0 {
        (0.0, x, c)
    } else if h < 5.0 / 6.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    [r1 + m, g1 + m, b1 + m]
}

fn spawn_label(commands: &mut Commands, text: &str, position: Vec3) {
    commands.spawn((
        Text::new(text.to_string()),
        TextFont {
            font_size: FontSize::Px(15.0),
            ..default()
        },
        TextColor(Color::srgb(0.88, 0.91, 0.98)),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(0.0),
            top: Val::Px(0.0),
            width: Val::Px(160.0),
            height: Val::Px(24.0),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        },
        WorldLabel(position),
    ));
}

fn position_labels(
    camera: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    mut labels: Query<(&WorldLabel, &mut Node)>,
) {
    let Ok((camera, camera_transform)) = camera.single() else {
        return;
    };
    for (label, mut node) in &mut labels {
        if let Ok(viewport) = camera.world_to_viewport(camera_transform, label.0) {
            node.left = Val::Px(viewport.x - 80.0);
            node.top = Val::Px(viewport.y - 12.0);
            node.display = Display::Flex;
        } else {
            node.display = Display::None;
        }
    }
}
