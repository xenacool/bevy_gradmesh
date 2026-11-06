//! Deformation showcase — demonstrates vertex shader deformations: wobble, pulse, and morph.
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
                .with_title("Deformation")
                .with_description("Wobble, pulse, and morph vertex shader animations"),
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
        Transform::from_xyz(0.0, 4.0, 10.0).looking_at(Vec3::new(0.0, 1.5, 0.0), Vec3::Y),
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

    // Ground plane
    let ground = meshes.add(bevy::prelude::Plane3d::new(Vec3::Y, Vec2::splat(22.0)));
    commands.spawn((
        Mesh3d(ground),
        MeshMaterial3d(materials.add(bevy_gradmesh::opaque(0.1, 0.1, 0.12))),
        Transform::from_translation(Vec3::new(0.0, -0.01, 0.0)),
    ));

    // Labels
    spawn_label(&mut commands, "Static", Vec3::new(-3.0, 1.2, 0.0));
    spawn_label(&mut commands, "Wobble", Vec3::new(0.0, 1.2, 0.0));
    spawn_label(&mut commands, "Strong Wobble", Vec3::new(3.0, 1.2, 0.0));
    spawn_label(&mut commands, "Static", Vec3::new(-3.0, 2.9, 0.0));
    spawn_label(&mut commands, "Pulse", Vec3::new(0.0, 2.9, 0.0));
    spawn_label(&mut commands, "Slow Pulse", Vec3::new(3.0, 2.9, 0.0));
    spawn_label(&mut commands, "Static", Vec3::new(-3.0, 4.6, 0.0));
    spawn_label(&mut commands, "Morph", Vec3::new(0.0, 4.6, 0.0));
    spawn_label(&mut commands, "Organic Morph", Vec3::new(3.0, 4.6, 0.0));
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

    // ─── Row 1: Wobble (y = 0.5) ───
    // Static sphere
    rec.set_position(Vec3::new(-3.0, 0.5, 0.0));
    rec.set_scale(Vec3::splat(0.4));
    rec.set_deformation(DeformationParams::none());
    rec.sphere(0.4, ShapeDetail::High, [0.5, 0.5, 0.5, 1.0]);

    // Wobbling sphere
    rec.set_position(Vec3::new(0.0, 0.5, 0.0));
    rec.set_scale(Vec3::splat(0.4));
    rec.set_deformation(DeformationParams::wobble(0.15, 3.0, 0.0));
    rec.sphere(0.4, ShapeDetail::High, [0.9, 0.3, 0.3, 1.0]);

    // Strong wobble
    rec.set_position(Vec3::new(3.0, 0.5, 0.0));
    rec.set_scale(Vec3::splat(0.4));
    rec.set_deformation(DeformationParams::wobble(0.3, 2.0, 1.0));
    rec.sphere(0.4, ShapeDetail::High, [0.9, 0.6, 0.2, 1.0]);

    // ─── Row 2: Pulse (y = 2.2) ───
    // Static cuboid
    rec.set_position(Vec3::new(-3.0, 2.2, 0.0));
    rec.set_scale(Vec3::splat(0.6));
    rec.set_deformation(DeformationParams::none());
    rec.cuboid(Vec3::splat(0.6), 0.15, ShapeDetail::High, [0.5, 0.5, 0.5, 1.0]);

    // Pulsing cuboid
    rec.set_position(Vec3::new(0.0, 2.2, 0.0));
    rec.set_scale(Vec3::splat(0.6));
    rec.set_deformation(DeformationParams::pulse(0.2, 2.5, 0.0));
    rec.cuboid(Vec3::splat(0.6), 0.15, ShapeDetail::High, [0.3, 0.85, 0.4, 1.0]);

    // Slow pulse
    rec.set_position(Vec3::new(3.0, 2.2, 0.0));
    rec.set_scale(Vec3::splat(0.6));
    rec.set_deformation(DeformationParams::pulse(0.15, 1.0, 2.0));
    rec.cuboid(Vec3::splat(0.6), 0.15, ShapeDetail::High, [0.3, 0.4, 0.95, 1.0]);

    // ─── Row 3: Morph (y = 3.9) ───
    // Static capsule
    rec.set_position(Vec3::new(-3.0, 3.9, 0.0));
    rec.set_scale(Vec3::ONE);
    rec.set_deformation(DeformationParams::none());
    rec.capsule(0.2, 0.6, ShapeDetail::High, [0.5, 0.5, 0.5, 1.0]);

    // Morphing capsule
    rec.set_position(Vec3::new(0.0, 3.9, 0.0));
    rec.set_scale(Vec3::ONE);
    rec.set_deformation(DeformationParams::morph(0.2, 2.0, 0.0));
    rec.capsule(0.2, 0.6, ShapeDetail::High, [0.95, 0.45, 0.6, 1.0]);

    // Organic morph
    rec.set_position(Vec3::new(3.0, 3.9, 0.0));
    rec.set_scale(Vec3::ONE);
    rec.set_deformation(DeformationParams::morph(0.25, 1.5, 3.0));
    rec.capsule(0.2, 0.6, ShapeDetail::High, [1.0, 0.85, 0.1, 1.0]);

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
