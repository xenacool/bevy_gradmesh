//! Advanced shapes example — showcases shapes not in the main shapes showcase.
//!
//! Demonstrates: star, ring, wedge, arbitrary polygon (with colored vertices),
//! tapered cylinder, torus arc, and hemisphere variants.
//!
//! Run with: `cargo run --example advanced_shapes`

mod common;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{
    dual_directional, Shape, ShapeDetail, GradMeshMaterial, GradMeshPlugin,
};
use common::labels::{spawn_label, position_labels};
use common::screenshot::AutoScreenshotPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "BevyGradmesh Advanced Shapes".into(),
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
            AutoScreenshotPlugin::new("screenshot_advanced_shapes.png")
                .with_title("Advanced Shapes")
                .with_description("Star, Ring, Wedge, Polygon, Tapered Cylinder, Torus Arc")
        )
        .insert_resource(ClearColor(Color::srgb(0.1, 0.1, 0.12)))
        .add_systems(Startup, setup)
        .add_systems(PostUpdate, position_labels)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
) {
    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 4.0, 12.0).looking_at(Vec3::new(0.0, 2.0, 0.0), Vec3::Y),
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

    let mut mat = |r, g, b| -> Handle<GradMeshMaterial> {
        materials.add(dual_directional(
            [r * 0.65, g * 0.65, b * 0.65],
            [r, g, b],
            [0.0, 1.0, 0.0],
        ))
    };

    let tilt = Quat::from_euler(EulerRot::XYZ, 0.35, 0.4, 0.0);

    // ─── Row 1: 2D advanced shapes (y = 0.5) ───

    // Star (5-pointed)
    let star_mesh = meshes.add(Shape::star(5, 0.45, 0.2, ShapeDetail::High).build());
    commands.spawn((
        Mesh3d(star_mesh),
        MeshMaterial3d(mat(0.95, 0.75, 0.2)),
        Transform::from_translation(Vec3::new(-5.0, 0.5, 0.0)),
    ));
    spawn_label(&mut commands, "Star (5-pointed)", Vec3::new(-5.0, -0.3, 0.0));

    // Star (8-pointed)
    let star8_mesh = meshes.add(Shape::star(8, 0.4, 0.18, ShapeDetail::High).build());
    commands.spawn((
        Mesh3d(star8_mesh),
        MeshMaterial3d(mat(0.95, 0.4, 0.5)),
        Transform::from_translation(Vec3::new(-2.5, 0.5, 0.0)),
    ));
    spawn_label(&mut commands, "Star (8-pointed)", Vec3::new(-2.5, -0.3, 0.0));

    // Ring (annulus)
    let ring_mesh = meshes.add(Shape::ring(0.45, 0.2, ShapeDetail::High).build());
    commands.spawn((
        Mesh3d(ring_mesh),
        MeshMaterial3d(mat(0.3, 0.8, 0.95)),
        Transform::from_translation(Vec3::new(0.0, 0.5, 0.0)),
    ));
    spawn_label(&mut commands, "Ring (annulus)", Vec3::new(0.0, -0.3, 0.0));

    // Wedge (pie slice, 90 degrees)
    let wedge_mesh = meshes.add(Shape::wedge(0.45, std::f32::consts::FRAC_PI_2, ShapeDetail::High).build());
    commands.spawn((
        Mesh3d(wedge_mesh),
        MeshMaterial3d(mat(0.4, 0.95, 0.4)),
        Transform::from_translation(Vec3::new(2.5, 0.5, 0.0)),
    ));
    spawn_label(&mut commands, "Wedge (90°)", Vec3::new(2.5, -0.3, 0.0));

    // Arbitrary polygon (hexagon with colored vertices)
    let hex_vertices: Vec<Vec2> = (0..6)
        .map(|i| {
            let a = i as f32 * std::f32::consts::TAU / 6.0;
            Vec2::new(a.cos() * 0.4, a.sin() * 0.4)
        })
        .collect();
    let hex_colors: Vec<[f32; 4]> = vec![
        [1.0, 0.2, 0.2, 1.0],
        [1.0, 0.6, 0.2, 1.0],
        [1.0, 1.0, 0.2, 1.0],
        [0.2, 1.0, 0.4, 1.0],
        [0.2, 0.4, 1.0, 1.0],
        [0.8, 0.2, 1.0, 1.0],
    ];
    let poly_mesh = meshes.add(Shape::arbitrary_polygon_colored(hex_vertices, hex_colors, ShapeDetail::High).build());
    commands.spawn((
        Mesh3d(poly_mesh),
        MeshMaterial3d(mat(0.7, 0.5, 0.95)),
        Transform::from_translation(Vec3::new(5.0, 0.5, 0.0)),
    ));
    spawn_label(&mut commands, "Colored polygon", Vec3::new(5.0, -0.3, 0.0));

    // ─── Row 2: 3D advanced shapes (y = 2.5) ───

    // Tapered cylinder (cone-frustum)
    let tapered_mesh = meshes.add(
        Shape::tapered_cylinder(0.15, 0.35, 0.9, ShapeDetail::High).build()
    );
    commands.spawn((
        Mesh3d(tapered_mesh),
        MeshMaterial3d(mat(0.9, 0.5, 0.2)),
        Transform::from_translation(Vec3::new(-4.0, 2.5, 0.0)).with_rotation(tilt),
    ));
    spawn_label(&mut commands, "Tapered cylinder", Vec3::new(-4.0, 1.7, 0.0));

    // Torus arc (270° partial torus)
    let torus_arc_mesh = meshes.add(
        Shape::torus(0.35, 0.1, 0.0, std::f32::consts::PI * 1.5).build()
    );
    commands.spawn((
        Mesh3d(torus_arc_mesh),
        MeshMaterial3d(mat(0.3, 0.7, 0.95)),
        Transform::from_translation(Vec3::new(-1.5, 2.5, 0.0)).with_rotation(tilt),
    ));
    spawn_label(&mut commands, "Torus arc (270°)", Vec3::new(-1.5, 1.7, 0.0));

    // Torus arc (180° semicircle)
    let torus_half_mesh = meshes.add(
        Shape::torus(0.35, 0.1, -std::f32::consts::FRAC_PI_2, std::f32::consts::PI).build()
    );
    commands.spawn((
        Mesh3d(torus_half_mesh),
        MeshMaterial3d(mat(0.95, 0.3, 0.6)),
        Transform::from_translation(Vec3::new(1.0, 2.5, 0.0)).with_rotation(tilt),
    ));
    spawn_label(&mut commands, "Torus arc (180°)", Vec3::new(1.0, 1.7, 0.0));

    // Hemisphere (capped)
    let hemi_mesh = meshes.add(Shape::hemisphere(0.38, true, ShapeDetail::High).build());
    commands.spawn((
        Mesh3d(hemi_mesh),
        MeshMaterial3d(mat(0.4, 0.95, 0.5)),
        Transform::from_translation(Vec3::new(3.5, 2.5, 0.0)).with_rotation(tilt),
    ));
    spawn_label(&mut commands, "Hemisphere (capped)", Vec3::new(3.5, 1.7, 0.0));

    // Hemisphere (open, no cap)
    let hemi_open_mesh = meshes.add(Shape::hemisphere(0.38, false, ShapeDetail::High).build());
    commands.spawn((
        Mesh3d(hemi_open_mesh),
        MeshMaterial3d(mat(0.6, 0.5, 0.95)),
        Transform::from_translation(Vec3::new(5.5, 2.5, 0.0)).with_rotation(tilt),
    ));
    spawn_label(&mut commands, "Hemisphere (open)", Vec3::new(5.5, 1.7, 0.0));

    // ─── Row 3: Wedge variations (y = 4.5) ───

    // Full circle wedge
    let full_wedge = meshes.add(
        Shape::wedge(0.35, std::f32::consts::TAU, ShapeDetail::High).build()
    );
    commands.spawn((
        Mesh3d(full_wedge),
        MeshMaterial3d(mat(0.95, 0.9, 0.3)),
        Transform::from_translation(Vec3::new(-3.0, 4.5, 0.0)),
    ));
    spawn_label(&mut commands, "Wedge (360° = disc)", Vec3::new(-3.0, 3.7, 0.0));

    // Small wedge (45°)
    let small_wedge = meshes.add(
        Shape::wedge(0.35, std::f32::consts::FRAC_PI_4, ShapeDetail::High).build()
    );
    commands.spawn((
        Mesh3d(small_wedge),
        MeshMaterial3d(mat(0.95, 0.5, 0.3)),
        Transform::from_translation(Vec3::new(0.0, 4.5, 0.0)),
    ));
    spawn_label(&mut commands, "Wedge (45°)", Vec3::new(0.0, 3.7, 0.0));

    // Arbitrary concave polygon (arrow shape)
    let arrow_vertices = vec![
        Vec2::new(0.0, 0.45),   // tip
        Vec2::new(0.15, 0.15),  // right inner
        Vec2::new(0.4, 0.15),   // right wing
        Vec2::new(0.2, 0.0),    // right notch
        Vec2::new(0.2, -0.45),  // right tail
        Vec2::new(0.0, -0.3),   // center bottom
        Vec2::new(-0.2, -0.45), // left tail
        Vec2::new(-0.2, 0.0),   // left notch
        Vec2::new(-0.4, 0.15),  // left wing
        Vec2::new(-0.15, 0.15), // left inner
    ];
    let arrow_mesh = meshes.add(Shape::arbitrary_polygon(arrow_vertices, ShapeDetail::High).build());
    commands.spawn((
        Mesh3d(arrow_mesh),
        MeshMaterial3d(mat(0.95, 0.3, 0.3)),
        Transform::from_translation(Vec3::new(3.0, 4.5, 0.0)),
    ));
    spawn_label(&mut commands, "Concave polygon", Vec3::new(3.0, 3.7, 0.0));
}
