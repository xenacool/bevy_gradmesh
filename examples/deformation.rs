//! Deformation showcase - demonstrates vertex shader deformations: wobble, pulse, and morph.
//!
//! Shows animated shapes with different deformation types side-by-side,
//! comparing deformed shapes against their static originals.
//! Uses `BatchRecorder` for GPU-instanced rendering with per-instance deformation.
//!
//! Run with: `cargo run --example deformation`

mod common;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{
    BatchRecorder, Deformation, DeformationParams, GradMeshMaterial, GradMeshPlugin, ShapeDetail,
};
use common::screenshot::AutoScreenshotPlugin;


/// Run this example. Called by the WASM runner.
pub fn run(app: &mut App) {
    app
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "BevyGradmesh Deformation".into(),
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
            AutoScreenshotPlugin::new("screenshot_deformation.png")
                .animation(3.0)
                .with_title("Deformation")
                .with_description("Wobble, pulse, and morph vertex shader animations")
                .wait_frames(10),
        )
        .insert_resource(ClearColor(Color::srgb(0.08, 0.08, 0.1)))
        .insert_resource(Deformation {
            time: 0.0,
            enabled: true,
        })
        .insert_resource(DeformRecorder(BatchRecorder::new()))
        .add_systems(Startup, setup)
        .add_systems(Update, tick_deformation)
        .add_systems(PostUpdate, (draw_deformations, position_labels).chain())
        ;
}

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "BevyGradmesh Deformation".into(),
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
            AutoScreenshotPlugin::new("screenshot_deformation.png")
                .animation(3.0)
                .with_title("Deformation")
                .with_description("Wobble, pulse, and morph vertex shader animations")
                .wait_frames(10),
        )
        .insert_resource(ClearColor(Color::srgb(0.08, 0.08, 0.1)))
        .insert_resource(Deformation {
            time: 0.0,
            enabled: true,
        })
        .insert_resource(DeformRecorder(BatchRecorder::new()))
        .add_systems(Startup, setup)
        .add_systems(Update, tick_deformation)
        .add_systems(PostUpdate, (draw_deformations, position_labels).chain())
        .run();
}

#[derive(Resource)]
struct DeformRecorder(BatchRecorder);

#[derive(Component)]
struct WorldLabel(Vec3);

#[derive(Component)]
struct DeformMarker;

fn tick_deformation(time: Res<Time>, mut deformation: ResMut<Deformation>) {
    deformation.tick(time.delta_secs());
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
) {
    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 3.0, 8.0).looking_at(Vec3::new(0.0, 2.3, 0.0), Vec3::Y),
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
    let ground = meshes.add(bevy::prelude::Plane3d::new(Vec3::Y, Vec2::splat(22.0)));
    commands.spawn((
        Mesh3d(ground),
        MeshMaterial3d(materials.add(bevy_gradmesh::opaque(0.16, 0.15, 0.2))),
        Transform::from_translation(Vec3::new(0.0, -0.01, 0.0)),
    ));

    // Labels
    spawn_label(&mut commands, "Static", Vec3::new(-3.5, 1.6, 0.0));
    spawn_label(&mut commands, "Wobble", Vec3::new(0.0, 1.6, 0.0));
    spawn_label(&mut commands, "Strong Wobble", Vec3::new(3.5, 1.6, 0.0));
    spawn_label(&mut commands, "Static", Vec3::new(-3.5, 3.3, 0.0));
    spawn_label(&mut commands, "Pulse", Vec3::new(0.0, 3.3, 0.0));
    spawn_label(&mut commands, "Slow Pulse", Vec3::new(3.5, 3.3, 0.0));
    spawn_label(&mut commands, "Static", Vec3::new(-3.5, 5.0, 0.0));
    spawn_label(&mut commands, "Morph", Vec3::new(0.0, 5.0, 0.0));
    spawn_label(&mut commands, "Organic Morph", Vec3::new(3.5, 5.0, 0.0));
}

/// Per-frame deformation drawing via BatchRecorder.
fn draw_deformations(
    time: Res<Time>,
    mut commands: Commands,
    meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
    mut recorder: ResMut<DeformRecorder>,
    query: Query<Entity, With<DeformMarker>>,
) {
    // Remove previous frame's entities
    for entity in query.iter() {
        commands.entity(entity).despawn();
    }

    let _t = time.elapsed_secs();
    let rec = &mut recorder.0;

    rec.begin();

    // Pastel palette with a few deeper tones for contrast.
    let static_grey = [0.62, 0.64, 0.72, 1.0];
    let pink = [1.0, 0.70, 0.80, 1.0];
    let butter = [1.0, 0.92, 0.55, 1.0];
    let mint = [0.62, 0.92, 0.78, 1.0];
    let sky = [0.62, 0.78, 1.0, 1.0];
    let lilac = [0.80, 0.70, 0.98, 1.0];
    let coral = [0.92, 0.38, 0.45, 1.0];
    let deep_teal = [0.12, 0.52, 0.58, 1.0];
    let _ = sky;

    let xs = [-3.5_f32, 0.0, 3.5];

    // --- Row 1: Wobble (y = 0.7) ---
    let rows = [0.7_f32, 2.4, 4.1];
    let row1 = [
        (DeformationParams::none(), static_grey),
        (DeformationParams::wobble(0.15, 3.0, 0.0), pink),
        (DeformationParams::wobble(0.3, 2.0, 1.0), coral),
    ];
    for (x, (deform, color)) in xs.iter().zip(row1) {
        rec.set_position(Vec3::new(*x, rows[0], 0.0));
        rec.set_deformation(deform);
        rec.sphere(0.6, ShapeDetail::High, color);
    }

    // --- Row 2: Pulse (y = 2.4) ---
    let row2 = [
        (DeformationParams::none(), static_grey),
        (DeformationParams::pulse(0.2, 2.5, 0.0), butter),
        (DeformationParams::pulse(0.15, 1.0, 2.0), deep_teal),
    ];
    for (x, (deform, color)) in xs.iter().zip(row2) {
        rec.set_position(Vec3::new(*x, rows[1], 0.0));
        rec.set_deformation(deform);
        rec.cuboid(Vec3::splat(0.9), 0.15, ShapeDetail::High, color);
    }

    // --- Row 3: Morph (y = 4.1) ---
    let row3 = [
        (DeformationParams::none(), static_grey),
        (DeformationParams::morph(0.2, 2.0, 0.0), mint),
        (DeformationParams::morph(0.25, 1.5, 3.0), lilac),
    ];
    for (x, (deform, color)) in xs.iter().zip(row3) {
        rec.set_position(Vec3::new(*x, rows[2], 0.0));
        rec.set_deformation(deform);
        rec.capsule(0.35, 0.8, ShapeDetail::High, color);
    }

    rec.end();

    // Render batches
    let spawned = rec.render_batches(&mut commands, meshes, None);
    for entity_id in spawned {
        commands.entity(entity_id).insert(DeformMarker);
    }
}

fn spawn_label(commands: &mut Commands, text: &str, position: Vec3) {
    commands.spawn((
        Text::new(text.to_string()),
        TextFont {
            font_size: FontSize::Px(14.0),
            ..default()
        },
        TextColor(Color::srgb(0.88, 0.91, 0.98)),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(0.0),
            top: Val::Px(0.0),
            width: Val::Px(120.0),
            height: Val::Px(22.0),
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
            node.left = Val::Px(viewport.x - 60.0);
            node.top = Val::Px(viewport.y - 11.0);
            node.display = Display::Flex;
        } else {
            node.display = Display::None;
        }
    }
}
