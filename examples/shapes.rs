//! Shape showcase example — displays all procedural primitives with different detail levels.
//!
//! Run with: `cargo run --example shapes`

mod common;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{dual_directional, opaque, Shape, ShapeDetail, GradMeshMaterial, GradMeshPlugin};
use common::screenshot::AutoScreenshotPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "BevyGradmesh Shape Showcase".into(),
                resolution: (1280, 720).into(),
                ..default()
            }),
            ..default()
        }).set(RenderPlugin {
            synchronous_pipeline_compilation: true,
            ..default()
        }))
        .add_plugins(GradMeshPlugin)
        .add_plugins(
            AutoScreenshotPlugin::new("screenshot_shapes.png")
                .with_title("Shape Showcase")
                .with_description("16 procedural primitives with detail levels")
        )
        .add_systems(Startup, setup)
        .add_systems(PostUpdate, (rotate_shapes, position_labels).chain())
        .run();
}

#[derive(Component)]
struct Spin(Quat);

#[derive(Component)]
struct WorldLabel(Vec3);

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
) {
    // Camera — centered on the shape showcase wall
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 3.9, 11.2).looking_at(Vec3::new(0.0, 3.9, 0.0), Vec3::Y),
    ));

    // Lighting — side-on angle so each shape shows roughly half lit, half shadowed
    commands.spawn((
        DirectionalLight {
            illuminance: 15_000.0,
            ..default()
        },
        Transform::from_xyz(8.0, 5.0, 2.0).looking_at(Vec3::new(0.0, 4.0, 0.0), Vec3::Y),
    ));

    // Ambient light to fill shadows
    commands.spawn(AmbientLight {
        color: Color::srgb(0.6, 0.6, 0.7),
        brightness: 300.0,
        affects_lightmapped_meshes: false,
    });

    // Backdrop plane
    let backdrop_mesh = meshes.add(
        Plane3d::new(Vec3::Z, Vec2::new(24.0, 16.0))
    );
    commands.spawn((
        Mesh3d(backdrop_mesh),
        MeshMaterial3d(materials.add(opaque(0.12, 0.12, 0.15))),
        Transform::from_translation(Vec3::new(0.0, 3.9, -0.8)),
        Name::new("Backdrop"),
    ));

    // 2D shapes are shown face-on in their own row.
    // Build these materials first, before the `mat` closure borrows `materials`.
    let flat_shapes = [
        (Shape::rectangle(Vec2::new(0.85, 0.58), ShapeDetail::High), "Rectangle", [0.95, 0.38, 0.35]),
        (Shape::rectangle_rim(Vec2::new(0.85, 0.58), 0.09, ShapeDetail::High), "Rect rim", [0.95, 0.65, 0.3]),
        (Shape::disc(0.34, ShapeDetail::High), "Disc", [0.45, 0.8, 0.95]),
        (Shape::disc_rim(0.34, 0.09, ShapeDetail::High), "Disc rim", [0.45, 0.95, 0.7]),
        (Shape::polygon(6, 0.36, ShapeDetail::High), "Polygon", [0.75, 0.55, 0.95]),
        (Shape::polygon_rim(6, 0.36, 0.09, ShapeDetail::High), "Polygon rim", [0.95, 0.45, 0.7]),
        (Shape::ellipse(Vec2::new(0.42, 0.28), ShapeDetail::High), "Ellipse", [0.95, 0.8, 0.35]),
        (Shape::ellipse_rim(Vec2::new(0.42, 0.28), 0.08, ShapeDetail::High), "Ellipse rim", [0.55, 0.9, 0.55]),
    ];
    let flat_mats: Vec<_> = flat_shapes
        .iter()
        .map(|(_, _, color)| {
            materials.add(dual_directional(
                [color[0] * 0.65, color[1] * 0.65, color[2] * 0.65],
                *color,
                [0.0, 1.0, 0.0],
            ))
        })
        .collect();

    // Helper to create a material with dual-directional gradient shading
    let mut mat = |r, g, b| -> Handle<GradMeshMaterial> {
        materials.add(dual_directional(
            [r * 0.7, g * 0.7, b * 0.7],
            [r, g, b],
            [0.0, 1.0, 0.0],
        ))
    };

    // Row 1: Basic shapes (y = 0.5)
    let y1 = 0.5;
    let r1_xs = [-4.0, -2.0, 0.0, 2.0, 4.0];

    // Sphere
    add_shape(
        &mut commands, &mut meshes, &mat(0.9, 0.3, 0.3),
        Shape::sphere(0.45, ShapeDetail::High),
        Vec3::new(r1_xs[0], y1, 0.0),
        "Sphere".to_string(),
    );

    // Ellipsoid
    add_shape(
        &mut commands, &mut meshes, &mat(0.3, 0.9, 0.3),
        Shape::ellipsoid(Vec3::new(0.3, 0.5, 0.45), ShapeDetail::High),
        Vec3::new(r1_xs[1], y1, 0.0),
        "Ellipsoid".to_string(),
    );

    // Cuboid
    add_shape(
        &mut commands, &mut meshes, &mat(0.3, 0.3, 0.9),
        Shape::cuboid(Vec3::new(0.55, 0.55, 0.55), 0.0, ShapeDetail::High),
        Vec3::new(r1_xs[2], y1, 0.0),
        "Cuboid".to_string(),
    );

    // Rounded Cuboid
    add_shape(
        &mut commands, &mut meshes, &mat(0.9, 0.6, 0.3),
        Shape::cuboid(Vec3::new(0.55, 0.55, 0.55), 0.25, ShapeDetail::High),
        Vec3::new(r1_xs[3], y1, 0.0),
        "Rounded Cuboid".to_string(),
    );

    // Cylinder
    add_shape(
        &mut commands, &mut meshes, &mat(0.6, 0.3, 0.9),
        Shape::cylinder(0.28, 0.6, ShapeDetail::High),
        Vec3::new(r1_xs[4], y1, 0.0),
        "Cylinder".to_string(),
    );

    // Row 2: Complex shapes (y = 2.2)
    let y2 = 2.2;
    let r2_xs = [-4.0, -2.0, 0.0, 2.0, 4.0];

    // Cone
    add_shape(
        &mut commands, &mut meshes, &mat(0.9, 0.5, 0.1),
        Shape::cone(0.28, 0.6, ShapeDetail::High),
        Vec3::new(r2_xs[0], y2, 0.0),
        "Cone".to_string(),
    );

    // Capsule
    add_shape(
        &mut commands, &mut meshes, &mat(0.1, 0.8, 0.8),
        Shape::capsule(0.2, 0.6, ShapeDetail::High),
        Vec3::new(r2_xs[1], y2, 0.0),
        "Capsule".to_string(),
    );

    // Hemisphere
    add_shape(
        &mut commands, &mut meshes, &mat(0.8, 0.2, 0.5),
        Shape::hemisphere(0.38, true, ShapeDetail::High),
        Vec3::new(r2_xs[2], y2, 0.0),
        "Hemisphere".to_string(),
    );

    // Torus — tilted to show the doughnut shape clearly
    add_shape_rotated(
        &mut commands, &mut meshes, &mat(0.5, 0.8, 0.2),
        Shape::torus(0.35, 0.12, 0.0, std::f32::consts::TAU),
        Vec3::new(r2_xs[3], y2, 0.0),
        Quat::from_euler(EulerRot::XYZ, 0.5, 0.4, 0.0),
        "Torus".to_string(),
    );

    // Torus Arc (smile) — tilted
    add_shape_rotated(
        &mut commands, &mut meshes, &mat(0.9, 0.3, 0.6),
        Shape::torus(0.35, 0.12, -std::f32::consts::FRAC_PI_4, std::f32::consts::PI * 1.2),
        Vec3::new(r2_xs[4], y2, 0.0),
        Quat::from_euler(EulerRot::XYZ, 0.4, 0.3, 0.0),
        "Torus Arc".to_string(),
    );

    // Row 3: Polyhedral shapes (y = 3.9)
    let y3 = 3.9;
    let r3_xs = [-4.5, -2.7, -0.9, 0.9, 2.7, 4.5];

    // Regular Prism (hexagon)
    add_shape(
        &mut commands, &mut meshes, &mat(0.4, 0.6, 0.9),
        Shape::regular_prism(6, 0.32, 0.55, 0.0, ShapeDetail::High),
        Vec3::new(r3_xs[0], y3, 0.0),
        "Hex Prism".to_string(),
    );

    // Pyramid (square)
    add_shape(
        &mut commands, &mut meshes, &mat(0.9, 0.7, 0.2),
        Shape::pyramid(4, 0.32, 0.6, ShapeDetail::High),
        Vec3::new(r3_xs[1], y3, 0.0),
        "Pyramid".to_string(),
    );

    // Regular Prism (octagon)
    add_shape(
        &mut commands, &mut meshes, &mat(0.7, 0.4, 0.8),
        Shape::regular_prism(8, 0.32, 0.55, 0.0, ShapeDetail::High),
        Vec3::new(r3_xs[2], y3, 0.0),
        "Oct Prism".to_string(),
    );

    // Pyramid (triangle)
    add_shape(
        &mut commands, &mut meshes, &mat(0.3, 0.9, 0.6),
        Shape::pyramid(3, 0.32, 0.6, ShapeDetail::High),
        Vec3::new(r3_xs[3], y3, 0.0),
        "Tri Pyramid".to_string(),
    );

    // 3-sided prism
    add_shape(
        &mut commands, &mut meshes, &mat(0.3, 0.7, 0.9),
        Shape::regular_prism(3, 0.32, 0.55, 0.0, ShapeDetail::High),
        Vec3::new(r3_xs[4], y3, 0.0),
        "3-sided prism".to_string(),
    );

    // Open hemisphere
    add_shape(
        &mut commands, &mut meshes, &mat(0.9, 0.45, 0.65),
        Shape::hemisphere(0.38, false, ShapeDetail::High),
        Vec3::new(r3_xs[5], y3, 0.0),
        "Open hemisphere".to_string(),
    );

    // Row 4: 2D shapes (y = 5.6)
    let y4 = 5.6;
    for (index, ((shape, label, _), material)) in flat_shapes.iter().zip(flat_mats.into_iter()).enumerate() {
        let x = (index as f32 - 3.5) * 1.35;
        add_flat_shape(
            &mut commands,
            &mut meshes,
            material,
            shape.clone(),
            Vec3::new(x, y4, 0.1),
            label,
        );
    }

    // Row 5: Detail-level comparison (y = 7.3)
    let y5 = 7.3;
    let r5_xs = [-4.0, -2.0, 0.0, 2.0, 4.0];

    add_shape(
        &mut commands, &mut meshes, &mat(0.5, 0.5, 0.5),
        Shape::sphere(0.38, ShapeDetail::Minimal),
        Vec3::new(r5_xs[0], y5, 0.0),
        "Minimal".to_string(),
    );

    add_shape(
        &mut commands, &mut meshes, &mat(0.6, 0.6, 0.6),
        Shape::sphere(0.38, ShapeDetail::Low),
        Vec3::new(r5_xs[1], y5, 0.0),
        "Low".to_string(),
    );

    add_shape(
        &mut commands, &mut meshes, &mat(0.7, 0.7, 0.7),
        Shape::sphere(0.38, ShapeDetail::Medium),
        Vec3::new(r5_xs[2], y5, 0.0),
        "Medium".to_string(),
    );

    add_shape(
        &mut commands, &mut meshes, &mat(0.8, 0.8, 0.8),
        Shape::sphere(0.38, ShapeDetail::High),
        Vec3::new(r5_xs[3], y5, 0.0),
        "High".to_string(),
    );

    add_shape(
        &mut commands, &mut meshes, &mat(0.9, 0.9, 0.9),
        Shape::sphere(0.38, ShapeDetail::Extreme),
        Vec3::new(r5_xs[4], y5, 0.0),
        "Extreme".to_string(),
    );
}

fn add_shape(
    commands: &mut Commands,
    meshes: &mut Assets<bevy::render::mesh::Mesh>,
    material: &Handle<GradMeshMaterial>,
    shape: Shape,
    position: Vec3,
    label: String,
) {
    let tilt = Quat::from_euler(EulerRot::XYZ, 0.35, 0.4, 0.0);
    add_shape_rotated(commands, meshes, material, shape, position, tilt, label);
}

fn add_shape_rotated(
    commands: &mut Commands,
    meshes: &mut Assets<bevy::render::mesh::Mesh>,
    material: &Handle<GradMeshMaterial>,
    shape: Shape,
    position: Vec3,
    rotation: Quat,
    label: String,
) {
    let mesh_handle = meshes.add(shape.build());
    commands.spawn((
        Mesh3d(mesh_handle),
        MeshMaterial3d(material.clone()),
        Transform::from_translation(position).with_rotation(rotation),
        Spin(rotation),
    ));
    spawn_label(commands, &label, position - Vec3::Y * 0.52);
}

fn add_flat_shape(
    commands: &mut Commands,
    meshes: &mut Assets<bevy::render::mesh::Mesh>,
    material: Handle<GradMeshMaterial>,
    shape: Shape,
    position: Vec3,
    label: &str,
) {
    commands.spawn((
        Mesh3d(meshes.add(shape.build())),
        MeshMaterial3d(material),
        Transform::from_translation(position),
    ));
    spawn_label(commands, label, position - Vec3::Y * 0.48);
}

fn spawn_label(commands: &mut Commands, text: &str, position: Vec3) {
    commands.spawn((
        Text::new(text.to_string()),
        TextFont { font_size: FontSize::Px(13.0), ..default() },
        TextColor(Color::srgb(0.88, 0.91, 0.98)),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(0.0),
            top: Val::Px(0.0),
            width: Val::Px(110.0),
            height: Val::Px(20.0),
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
    let Ok((camera, camera_transform)) = camera.single() else { return; };
    for (label, mut node) in &mut labels {
        if let Ok(viewport) = camera.world_to_viewport(camera_transform, label.0) {
            node.left = Val::Px(viewport.x - 55.0);
            node.top = Val::Px(viewport.y - 10.0);
            node.display = Display::Flex;
        } else {
            node.display = Display::None;
        }
    }
}

fn rotate_shapes(time: Res<Time>, mut query: Query<(&Spin, &mut Transform)>) {
    for (spin, mut transform) in &mut query {
        transform.rotation = Quat::from_rotation_y(time.elapsed_secs() * 0.25) * spin.0;
    }
}
