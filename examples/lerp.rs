//! Shape lerp (morphing) demo — showcases `Shape::lerp` with animated morphing.
//!
//! Six shapes animate continuously, demonstrating cross-kind morphing:\n
//! 1. **Disc ↔ Star**: filled flat shapes morph as outlines.\n
//! 2. **Rectangle ↔ Ring**: a solid rectangle morphs into a ring (outer edge only).\n
//! 3. **Polygon ↔ Disc rim**: a hexagon morphs into a disc rim.\n
//! 4. **Ring ↔ Disc**: ring morph — the outer edge blends, inner hole is ignored.\n
//! 5. **Disc rim ↔ Rectangle**: cross-kind rim morphing.\n
//! 6. **Wedge ↔ Disc**: a semicircular wedge morphs into a full disc.\n
//!
//! Run with: `cargo run --example lerp`

mod common;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{GradMeshMaterial, GradMeshPlugin, Shape, ShapeDetail, opaque};
use common::screenshot::AutoScreenshotPlugin;


/// Run this example. Called by the WASM runner.
pub fn run(app: &mut App) {
    app
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "BevyGradmesh Shape Lerp".into(),
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
            AutoScreenshotPlugin::new("screenshot_lerp.png")
                .animation(4.0)
                .with_title("Shape Lerp")
                .with_description(
                    "Morphing shapes: disc↔star, rect↔ring, polygon↔rim, ring↔disc, wedge↔disc",
                ),
        )
        .insert_resource(ClearColor(Color::srgb(0.08, 0.08, 0.1)))
        .add_systems(Startup, setup)
        .add_systems(PostUpdate, draw_lerp)
        ;
}

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "BevyGradmesh Shape Lerp".into(),
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
            AutoScreenshotPlugin::new("screenshot_lerp.png")
                .animation(4.0)
                .with_title("Shape Lerp")
                .with_description(
                    "Morphing shapes: disc↔star, rect↔ring, polygon↔rim, ring↔disc, wedge↔disc",
                ),
        )
        .insert_resource(ClearColor(Color::srgb(0.08, 0.08, 0.1)))
        .add_systems(Startup, setup)
        .add_systems(PostUpdate, draw_lerp)
        .run();
}

#[derive(Component)]
struct LerpMarker;

const D: ShapeDetail = ShapeDetail::Medium;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
) {
    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 10.0, 14.0).looking_at(Vec3::new(0.0, 0.5, -1.0), Vec3::Y),
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
        ..default()
    });

    // Backdrop
    let backdrop_mesh = meshes.add(Plane3d::new(Vec3::Z, Vec2::new(16.0, 8.0)));
    commands.spawn((
        Mesh3d(backdrop_mesh),
        MeshMaterial3d(materials.add(opaque(0.12, 0.12, 0.15))),
        Transform::from_translation(Vec3::new(0.0, 0.5, -1.5)),
        Name::new("Backdrop"),
    ));

    // Title
    commands.spawn((
        Text::new("Shape Lerp Demo"),
        TextFont { font_size: 22.0.into(), ..default() },
        TextColor(Color::srgb(1.0, 1.0, 1.0)),
        Node {
            position_type: PositionType::Absolute,
            top: px(12),
            left: px(16),
            ..default()
        },
    ));
}

fn spawn_morphed(
    commands: &mut Commands,
    meshes: &mut Assets<bevy::render::mesh::Mesh>,
    materials: &mut Assets<GradMeshMaterial>,
    shape: Shape,
    position: Vec3,
    color: [f32; 3],
) {
    let mesh = meshes.add(shape.build());
    commands.spawn((
        Mesh3d(mesh),
        MeshMaterial3d(materials.add(opaque(color[0], color[1], color[2]))),
        Transform::from_translation(position),
        LerpMarker,
        Name::new("Morphed shape"),
    ));
}

fn draw_lerp(
    time: Res<Time>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
    query: Query<Entity, With<LerpMarker>>,
) {
    // Clear previous frame
    for entity in &query {
        commands.entity(entity).despawn();
    }

    let t = (time.elapsed_secs() * 0.4).sin() * 0.5 + 0.5; // oscillates 0..1

    // --- Row 1: Cross-kind flat shape morphing ---

    // Disc ↔ Star
    spawn_morphed(
        &mut commands, &mut meshes, &mut materials,
        Shape::disc(1.2, D).lerp(&Shape::star(5, 1.2, 0.5, D), t),
        Vec3::new(-5.0, 1.5, 0.0),
        [0.3, 0.7, 0.9],
    );

    // Rectangle ↔ Ring
    spawn_morphed(
        &mut commands, &mut meshes, &mut materials,
        Shape::rectangle(Vec2::new(2.8, 1.8), D).lerp(&Shape::ring(1.5, 0.6, D), t),
        Vec3::new(0.0, 1.5, 0.0),
        [0.9, 0.5, 0.3],
    );

    // Polygon ↔ Disc rim
    spawn_morphed(
        &mut commands, &mut meshes, &mut materials,
        Shape::polygon(6, 1.2, D).lerp(&Shape::disc_rim(1.0, 0.3, D), t),
        Vec3::new(5.0, 1.5, 0.0),
        [0.4, 0.9, 0.5],
    );

    // --- Row 2: Ring morph examples ---

    // Ring ↔ Disc
    spawn_morphed(
        &mut commands, &mut meshes, &mut materials,
        Shape::ring(1.2, 0.5, D).lerp(&Shape::disc(1.2, D), t),
        Vec3::new(-5.0, -1.2, 0.0),
        [0.8, 0.4, 0.9],
    );

    // Disc rim ↔ Rectangle (cross-kind)
    spawn_morphed(
        &mut commands, &mut meshes, &mut materials,
        Shape::disc_rim(1.0, 0.3, D).lerp(&Shape::rectangle(Vec2::splat(2.6), D), t),
        Vec3::new(0.0, -1.2, 0.0),
        [0.9, 0.7, 0.3],
    );

    // Wedge ↔ Disc
    spawn_morphed(
        &mut commands, &mut meshes, &mut materials,
        Shape::wedge(1.2, std::f32::consts::PI, D).lerp(&Shape::disc(1.2, D), t),
        Vec3::new(5.0, -1.2, 0.0),
        [0.3, 0.9, 0.7],
    );
}
