//! Bézier curves example — demonstrates curve primitives as tubes.
//!
//! Shows linear, quadratic, and cubic Bézier curves rendered as tube geometry.
//!
//! Run with: `cargo run --example curves`

mod common;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{
    dual_directional, Shape, ShapeDetail,
    GradMeshMaterial, GradMeshPlugin,
};
use common::screenshot::AutoScreenshotPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "BevyGradmesh Curves".into(),
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
            AutoScreenshotPlugin::new("screenshot_curves.png")
                .with_title("Curves")
                .with_description("Bézier curve primitives rendered as tubes")
        )
        .insert_resource(ClearColor(Color::srgb(0.1, 0.1, 0.12)))
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
    // Camera — top-down angled view
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 12.0, 10.0).looking_at(Vec3::new(0.0, 0.5, 0.0), Vec3::Y),
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
    let ground = meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(22.0)));
    commands.spawn((
        Mesh3d(ground),
        MeshMaterial3d(materials.add(dual_directional(
            [0.07, 0.07, 0.09],
            [0.12, 0.12, 0.15],
            [0.0, 1.0, 0.0],
        ))),
        Transform::from_translation(Vec3::new(0.0, -0.01, 0.0)),
    ));

    let mut mat = |r: f32, g: f32, b: f32| -> Handle<GradMeshMaterial> {
        materials.add(dual_directional(
            [r * 0.6, g * 0.6, b * 0.6],
            [r, g, b],
            [0.0, 1.0, 0.0],
        ))
    };

    // ─── Row 1: Linear curve (straight line tube) ───
    let linear = vec![
        Vec3::new(-3.0, 0.0, 0.0),
        Vec3::new(3.0, 0.0, 0.0),
    ];
    let linear_mesh = meshes.add(Shape::bezier_curve(linear, 0.08, 8, ShapeDetail::Medium).build());
    commands.spawn((
        Mesh3d(linear_mesh),
        MeshMaterial3d(mat(0.9, 0.3, 0.3)),
        Transform::from_translation(Vec3::new(-4.0, 1.0, -3.0)),
    ));
    spawn_label(&mut commands, "Linear", Vec3::new(-4.0, 1.8, -3.0));

    // ─── Row 2: Quadratic Bézier (arc) ───
    let quadratic = vec![
        Vec3::new(-3.0, 0.0, 0.0),
        Vec3::new(0.0, 2.5, 0.0),
        Vec3::new(3.0, 0.0, 0.0),
    ];
    let quad_mesh = meshes.add(Shape::bezier_curve(quadratic, 0.08, 8, ShapeDetail::Medium).build());
    commands.spawn((
        Mesh3d(quad_mesh),
        MeshMaterial3d(mat(0.3, 0.85, 0.4)),
        Transform::from_translation(Vec3::new(4.0, 1.0, -3.0)),
    ));
    spawn_label(&mut commands, "Quadratic Arc", Vec3::new(4.0, 3.0, -3.0));

    // ─── Row 3: Cubic S-curve ───
    let scurve = vec![
        Vec3::new(-3.0, 0.0, 0.0),
        Vec3::new(-1.5, 2.0, 0.0),
        Vec3::new(1.5, -2.0, 0.0),
        Vec3::new(3.0, 0.0, 0.0),
    ];
    let scurve_mesh = meshes.add(Shape::bezier_curve(scurve, 0.08, 8, ShapeDetail::Medium).build());
    commands.spawn((
        Mesh3d(scurve_mesh),
        MeshMaterial3d(mat(0.3, 0.4, 0.95)),
        Transform::from_translation(Vec3::new(-4.0, 1.0, 3.0)),
    ));
    spawn_label(&mut commands, "Cubic S-Curve", Vec3::new(-4.0, 2.5, 3.0));

    // ─── Row 4: Cubic loop ───
    let loop_curve = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(2.5, 2.0, 0.0),
        Vec3::new(2.5, -2.0, 0.0),
        Vec3::new(0.0, 0.0, 0.0),
    ];
    let loop_mesh = meshes.add(Shape::bezier_curve(loop_curve, 0.08, 8, ShapeDetail::Medium).build());
    commands.spawn((
        Mesh3d(loop_mesh),
        MeshMaterial3d(mat(0.95, 0.45, 0.6)),
        Transform::from_translation(Vec3::new(4.0, 1.0, 3.0)),
    ));
    spawn_label(&mut commands, "Cubic Loop", Vec3::new(4.0, 2.5, 3.0));

    // ─── 3D helix-like curve ───
    let helix = vec![
        Vec3::new(-2.0, 0.0, -2.0),
        Vec3::new(2.0, 1.5, -2.0),
        Vec3::new(2.0, 3.0, 2.0),
        Vec3::new(-2.0, 4.5, 2.0),
    ];
    let helix_mesh = meshes.add(Shape::bezier_curve(helix, 0.1, 10, ShapeDetail::Medium).build());
    commands.spawn((
        Mesh3d(helix_mesh),
        MeshMaterial3d(mat(1.0, 0.85, 0.1)),
        Transform::from_translation(Vec3::new(0.0, 0.0, 0.0)),
    ));
    spawn_label(&mut commands, "3D Helix", Vec3::new(0.0, 5.5, 0.0));
}

fn spawn_label(commands: &mut Commands, text: &str, position: Vec3) {
    commands.spawn((
        Text::new(text.to_string()),
        TextFont { font_size: FontSize::Px(15.0), ..default() },
        TextColor(Color::srgb(0.88, 0.91, 0.98)),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(0.0),
            top: Val::Px(0.0),
            width: Val::Px(150.0),
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
    let Ok((camera, camera_transform)) = camera.single() else { return; };
    for (label, mut node) in &mut labels {
        if let Ok(viewport) = camera.world_to_viewport(camera_transform, label.0) {
            node.left = Val::Px(viewport.x - 75.0);
            node.top = Val::Px(viewport.y - 12.0);
            node.display = Display::Flex;
        } else {
            node.display = Display::None;
        }
    }
}
