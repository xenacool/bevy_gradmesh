//! Arbitrary shapes example - demonstrates custom polygon, star, ring, wedge, and tapered cylinder.
//!
//! Shows shapes that go beyond the standard primitives: user-defined polygons
//! with per-vertex colors, star shapes, annular rings, pie-slice wedges,
//! and tapered cylinders (frustums).
//!
//! Run with: `cargo run --example arbitrary_shapes`

mod common;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{GradMeshMaterial, GradMeshPlugin, Shape, ShapeDetail, dual_directional, opaque};
use common::screenshot::AutoScreenshotPlugin;

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "BevyGradmesh Arbitrary Shapes".into(),
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
            AutoScreenshotPlugin::new("screenshot_arbitrary_shapes.png")
                .with_title("Arbitrary Shapes")
                .with_description("Custom polygons, stars, rings, wedges, and tapered cylinders"),
        )
        .insert_resource(ClearColor(Color::srgb(0.08, 0.08, 0.1)))
        .add_systems(Startup, setup)
        .add_systems(PostUpdate, position_labels)
        .run();
}

#[derive(Component)]
struct WorldLabel(Vec3);

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
) {
    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 5.0, 10.0).looking_at(Vec3::new(0.0, 1.5, 0.0), Vec3::Y),
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
        brightness: 200.0,
        affects_lightmapped_meshes: false,
    });

    // Ground plane
    let ground = meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(22.0)));
    commands.spawn((
        Mesh3d(ground),
        MeshMaterial3d(materials.add(opaque(0.1, 0.1, 0.12))),
        Transform::from_translation(Vec3::new(0.0, -0.01, 0.0)),
    ));

    let mut mat = |r: f32, g: f32, b: f32| -> Handle<GradMeshMaterial> {
        materials.add(dual_directional(
            [r * 0.6, g * 0.6, b * 0.6],
            [r, g, b],
            [0.0, 1.0, 0.0],
        ))
    };

    // --- Row 1: Star shapes (y = 0.5) ---
    // 5-pointed star
    let star5 = meshes.add(
        Shape::star(5, 0.45, 0.18, ShapeDetail::High)
            .build(),
    );
    commands.spawn((
        Mesh3d(star5),
        MeshMaterial3d(mat(0.95, 0.85, 0.1)),
        Transform::from_translation(Vec3::new(-3.0, 0.5, 0.0))
            .with_rotation(Quat::from_rotation_x(-std::f32::consts::PI / 2.0)),
    ));
    spawn_label(&mut commands, "5-Point Star", Vec3::new(-3.0, 1.2, 0.0));

    // 8-pointed star
    let star8 = meshes.add(
        Shape::star(8, 0.45, 0.22, ShapeDetail::High)
            .build(),
    );
    commands.spawn((
        Mesh3d(star8),
        MeshMaterial3d(mat(0.9, 0.3, 0.3)),
        Transform::from_translation(Vec3::new(0.0, 0.5, 0.0))
            .with_rotation(Quat::from_rotation_x(-std::f32::consts::PI / 2.0)),
    ));
    spawn_label(&mut commands, "8-Point Star", Vec3::new(0.0, 1.2, 0.0));

    // Deep-indent star
    let star_deep = meshes.add(
        Shape::star(6, 0.45, 0.08, ShapeDetail::High)
            .build(),
    );
    commands.spawn((
        Mesh3d(star_deep),
        MeshMaterial3d(mat(0.3, 0.85, 0.4)),
        Transform::from_translation(Vec3::new(3.0, 0.5, 0.0))
            .with_rotation(Quat::from_rotation_x(-std::f32::consts::PI / 2.0)),
    ));
    spawn_label(&mut commands, "Deep Star", Vec3::new(3.0, 1.2, 0.0));

    // --- Row 2: Ring & Wedge (y = 2.2) ---
    // Full ring
    let ring = meshes.add(
        Shape::ring(0.45, 0.25, ShapeDetail::High)
            .build(),
    );
    commands.spawn((
        Mesh3d(ring),
        MeshMaterial3d(mat(0.3, 0.4, 0.95)),
        Transform::from_translation(Vec3::new(-3.0, 2.2, 0.0))
            .with_rotation(Quat::from_rotation_x(-std::f32::consts::PI / 2.0)),
    ));
    spawn_label(&mut commands, "Ring", Vec3::new(-3.0, 2.9, 0.0));

    // Thin ring
    let thin_ring = meshes.add(
        Shape::ring(0.45, 0.38, ShapeDetail::High)
            .build(),
    );
    commands.spawn((
        Mesh3d(thin_ring),
        MeshMaterial3d(mat(0.95, 0.45, 0.6)),
        Transform::from_translation(Vec3::new(0.0, 2.2, 0.0))
            .with_rotation(Quat::from_rotation_x(-std::f32::consts::PI / 2.0)),
    ));
    spawn_label(&mut commands, "Thin Ring", Vec3::new(0.0, 2.9, 0.0));

    // Wedge (quarter circle)
    let wedge = meshes.add(
        Shape::wedge(0.45, std::f32::consts::FRAC_PI_2, ShapeDetail::High)
            .build(),
    );
    commands.spawn((
        Mesh3d(wedge),
        MeshMaterial3d(mat(1.0, 0.6, 0.2)),
        Transform::from_translation(Vec3::new(3.0, 2.2, 0.0))
            .with_rotation(Quat::from_rotation_x(-std::f32::consts::PI / 2.0)),
    ));
    spawn_label(&mut commands, "Quarter Wedge", Vec3::new(3.0, 2.9, 0.0));

    // --- Row 3: Tapered Cylinder (y = 3.9) ---
    // Cup shape (narrow top, wide bottom)
    let cup = meshes.add(
        Shape::tapered_cylinder(0.2, 0.4, 0.7, ShapeDetail::High)
            .build(),
    );
    commands.spawn((
        Mesh3d(cup),
        MeshMaterial3d(mat(0.4, 0.7, 0.9)),
        Transform::from_translation(Vec3::new(-3.0, 3.9, 0.0))
            .with_rotation(Quat::from_rotation_y(0.3)),
    ));
    spawn_label(&mut commands, "Cup", Vec3::new(-3.0, 4.6, 0.0));

    // Inverted cone shape (wide top, narrow bottom)
    let inverted = meshes.add(
        Shape::tapered_cylinder(0.4, 0.15, 0.7, ShapeDetail::High)
            .build(),
    );
    commands.spawn((
        Mesh3d(inverted),
        MeshMaterial3d(mat(0.7, 0.4, 0.9)),
        Transform::from_translation(Vec3::new(0.0, 3.9, 0.0))
            .with_rotation(Quat::from_rotation_y(0.3)),
    ));
    spawn_label(&mut commands, "Inverted", Vec3::new(0.0, 4.6, 0.0));

    // Barrel shape (narrow middle via two tapered cylinders)
    let barrel_top = meshes.add(
        Shape::tapered_cylinder(0.35, 0.25, 0.35, ShapeDetail::High)
            .build(),
    );
    let barrel_bot = meshes.add(
        Shape::tapered_cylinder(0.25, 0.35, 0.35, ShapeDetail::High)
            .build(),
    );
    commands.spawn((
        Mesh3d(barrel_top),
        MeshMaterial3d(mat(0.6, 0.4, 0.2)),
        Transform::from_translation(Vec3::new(3.0, 4.0, 0.0))
            .with_rotation(Quat::from_rotation_y(0.3)),
    ));
    commands.spawn((
        Mesh3d(barrel_bot),
        MeshMaterial3d(mat(0.6, 0.4, 0.2)),
        Transform::from_translation(Vec3::new(3.0, 3.65, 0.0))
            .with_rotation(Quat::from_rotation_y(0.3)),
    ));
    spawn_label(&mut commands, "Barrel", Vec3::new(3.0, 4.6, 0.0));

    // --- Row 4: Arbitrary Polygons (y = 5.6) ---
    // Arrow shape
    let arrow_verts = vec![
        Vec2::new(0.0, 0.5),   // tip
        Vec2::new(0.15, 0.15), // inner right
        Vec2::new(0.4, 0.15),  // right wing
        Vec2::new(0.2, 0.0),   // inner notch
        Vec2::new(0.2, -0.5),  // tail
        Vec2::new(-0.2, -0.5), // tail
        Vec2::new(-0.2, 0.0),  // inner notch
        Vec2::new(-0.4, 0.15), // left wing
        Vec2::new(-0.15, 0.15),// inner left
    ];
    let arrow = meshes.add(
        Shape::arbitrary_polygon(arrow_verts, ShapeDetail::Minimal)
            .build(),
    );
    commands.spawn((
        Mesh3d(arrow),
        MeshMaterial3d(mat(0.9, 0.5, 0.2)),
        Transform::from_translation(Vec3::new(-3.0, 5.6, 0.0))
            .with_rotation(Quat::from_rotation_x(-std::f32::consts::PI / 2.0)),
    ));
    spawn_label(&mut commands, "Arrow", Vec3::new(-3.0, 6.3, 0.0));

    // Colored diamond with per-vertex colors
    let diamond_verts = vec![
        Vec2::new(0.0, 0.5),
        Vec2::new(0.4, 0.0),
        Vec2::new(0.0, -0.5),
        Vec2::new(-0.4, 0.0),
    ];
    let diamond_colors = vec![
        [1.0, 0.2, 0.2, 1.0], // red top
        [0.2, 1.0, 0.2, 1.0], // green right
        [0.2, 0.2, 1.0, 1.0], // blue bottom
        [1.0, 1.0, 0.2, 1.0], // yellow left
    ];
    let diamond = meshes.add(
        Shape::arbitrary_polygon_colored(diamond_verts, diamond_colors, ShapeDetail::Minimal)
            .build(),
    );
    commands.spawn((
        Mesh3d(diamond),
        MeshMaterial3d(mat(0.8, 0.8, 0.8)),
        Transform::from_translation(Vec3::new(0.0, 5.6, 0.0))
            .with_rotation(Quat::from_rotation_x(-std::f32::consts::PI / 2.0)),
    ));
    spawn_label(&mut commands, "Colored Diamond", Vec3::new(0.0, 6.3, 0.0));

    // L-shape polygon
    let l_verts = vec![
        Vec2::new(-0.3, 0.5),
        Vec2::new(0.0, 0.5),
        Vec2::new(0.0, 0.1),
        Vec2::new(0.4, 0.1),
        Vec2::new(0.4, -0.5),
        Vec2::new(-0.3, -0.5),
    ];
    let l_shape = meshes.add(
        Shape::arbitrary_polygon(l_verts, ShapeDetail::Minimal)
            .build(),
    );
    commands.spawn((
        Mesh3d(l_shape),
        MeshMaterial3d(mat(0.3, 0.9, 0.6)),
        Transform::from_translation(Vec3::new(3.0, 5.6, 0.0))
            .with_rotation(Quat::from_rotation_x(-std::f32::consts::PI / 2.0)),
    ));
    spawn_label(&mut commands, "L-Shape", Vec3::new(3.0, 6.3, 0.0));
}

fn spawn_label(commands: &mut Commands, text: &str, position: Vec3) {
    commands.spawn((
        Text::new(text.to_string()),
        TextFont {
            font_size: FontSize::Px(13.0),
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
