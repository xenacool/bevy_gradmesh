//! Polyline example — demonstrates 3D polylines as first-class primitives.
//!
//! Shows open and closed polylines with different thicknesses and styles.
//!
//! Run with: `cargo run --example polyline`

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
                title: "BevyGradmesh Polyline".into(),
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
            AutoScreenshotPlugin::new("screenshot_polyline.png")
                .with_title("Polyline")
                .with_description("3D polyline primitives with tube geometry")
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
        Transform::from_xyz(0.0, 10.0, 10.0).looking_at(Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
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
    let ground = meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(20.0)));
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

    // ─── Row 1: Open zigzag polyline ───
    let pts = zigzag_points(-3.0, 3.0, 6, 0.8);
    let polyline_mesh = meshes.add(Shape::polyline3d(pts.clone(), 0.1, false, 8, ShapeDetail::Medium).build());
    let miter_mat = mat(0.9, 0.3, 0.3);
    commands.spawn((
        Mesh3d(polyline_mesh),
        MeshMaterial3d(miter_mat),
        Transform::from_translation(Vec3::new(-4.0, 1.0, -3.0)),
    ));
    spawn_label(&mut commands, "Open Zigzag", Vec3::new(-4.0, 2.0, -3.0));

    // ─── Row 2: Closed hexagon loop ───
    let pts = hexagon_points(1.5);
    let closed_mesh = meshes.add(Shape::polyline3d(pts.clone(), 0.1, true, 8, ShapeDetail::Medium).build());
    let closed_mat = mat(0.95, 0.45, 0.6);
    commands.spawn((
        Mesh3d(closed_mesh),
        MeshMaterial3d(closed_mat),
        Transform::from_translation(Vec3::new(4.0, 1.0, -3.0)),
    ));
    spawn_label(&mut commands, "Closed Loop", Vec3::new(4.0, 2.0, -3.0));

    // ─── Row 3: Thick polyline ───
    let pts = zigzag_points(-3.0, 3.0, 4, 1.0);
    let thick_mesh = meshes.add(Shape::polyline3d(pts.clone(), 0.2, false, 12, ShapeDetail::Medium).build());
    let thick_mat = mat(0.3, 0.85, 0.4);
    commands.spawn((
        Mesh3d(thick_mesh),
        MeshMaterial3d(thick_mat),
        Transform::from_translation(Vec3::new(-4.0, 1.0, 3.0)),
    ));
    spawn_label(&mut commands, "Thick Line", Vec3::new(-4.0, 2.0, 3.0));

    // ─── Row 4: Thin detailed polyline ───
    let pts = zigzag_points(-3.0, 3.0, 8, 0.5);
    let thin_mesh = meshes.add(Shape::polyline3d(pts.clone(), 0.04, false, 6, ShapeDetail::Medium).build());
    let thin_mat = mat(0.3, 0.4, 0.95);
    commands.spawn((
        Mesh3d(thin_mesh),
        MeshMaterial3d(thin_mat),
        Transform::from_translation(Vec3::new(4.0, 1.0, 3.0)),
    ));
    spawn_label(&mut commands, "Thin Line", Vec3::new(4.0, 2.0, 3.0));

    // ─── 3D spiral polyline ───
    let pts = spiral_points(10, 2.0, 3.0);
    let spiral_mesh = meshes.add(Shape::polyline3d(pts.clone(), 0.06, false, 8, ShapeDetail::Medium).build());
    let spiral_mat = mat(1.0, 0.85, 0.1);
    commands.spawn((
        Mesh3d(spiral_mesh),
        MeshMaterial3d(spiral_mat),
        Transform::from_translation(Vec3::new(0.0, 0.0, 0.0)),
    ));
    spawn_label(&mut commands, "3D Spiral", Vec3::new(0.0, 3.5, 0.0));
}

/// Generate zigzag points for polyline demos.
fn zigzag_points(x_min: f32, x_max: f32, peaks: usize, amplitude: f32) -> Vec<Vec3> {
    let mut pts = Vec::new();
    for i in 0..=peaks {
        let t = i as f32 / peaks as f32;
        let x = x_min + t * (x_max - x_min);
        let y = if i % 2 == 0 { 0.0 } else { amplitude };
        pts.push(Vec3::new(x, y, 0.0));
    }
    pts
}

/// Generate hexagon points for closed loop demo.
fn hexagon_points(radius: f32) -> Vec<Vec3> {
    (0..6)
        .map(|i| {
            let a = i as f32 * std::f32::consts::TAU / 6.0;
            Vec3::new(a.cos() * radius, a.sin() * radius, 0.0)
        })
        .collect()
}

/// Generate a 3D spiral (helix) path.
fn spiral_points(turns: u32, radius: f32, height: f32) -> Vec<Vec3> {
    let steps = (turns * 20) as usize;
    (0..=steps)
        .map(|i| {
            let t = i as f32 / steps as f32;
            let angle = t * turns as f32 * std::f32::consts::TAU;
            Vec3::new(
                angle.cos() * radius,
                t * height - height * 0.5,
                angle.sin() * radius,
            )
        })
        .collect()
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
