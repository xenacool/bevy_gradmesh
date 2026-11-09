//! Polyline example - demonstrates 3D polylines as first-class primitives.
//!
//! Shows open and closed polylines with different thicknesses and styles.
//!
//! Run with: `cargo run --example polyline`

mod common;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{
    GradMeshMaterial, GradMeshPlugin, Shape, ShapeDetail, dual_directional, opaque,
    with_gradient_bounds,
};
use common::screenshot::AutoScreenshotPlugin;


/// Run this example. Called by the WASM runner.
pub fn run(app: &mut App) {
    app
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "BevyGradmesh Polyline".into(),
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
            AutoScreenshotPlugin::new("screenshot_polyline.png")
                .animation(4.0)
                .with_title("Polyline")
                .with_description("3D polyline primitives with tube geometry"),
        )
        .insert_resource(ClearColor(Color::srgb(0.1, 0.1, 0.12)))
        .add_systems(Startup, setup)
        .add_systems(PostUpdate, position_labels)
        ;
}

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "BevyGradmesh Polyline".into(),
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
            AutoScreenshotPlugin::new("screenshot_polyline.png")
                .animation(4.0)
                .with_title("Polyline")
                .with_description("3D polyline primitives with tube geometry"),
        )
        .insert_resource(ClearColor(Color::srgb(0.1, 0.1, 0.12)))
        .add_systems(Startup, setup)
        .add_systems(PostUpdate, position_labels)
        .run();
}

#[derive(Component)]
struct WorldLabel(Vec3);

/// Pastel and deep colour stops used across the scene.
const BUTTER: [f32; 3] = [1.0, 0.93, 0.58];
const PINK: [f32; 3] = [1.0, 0.70, 0.82];
const MINT: [f32; 3] = [0.62, 0.93, 0.80];
const SKY: [f32; 3] = [0.62, 0.80, 1.0];
const LILAC: [f32; 3] = [0.82, 0.72, 0.98];
const ROSE: [f32; 3] = [0.80, 0.25, 0.42];
const TEAL: [f32; 3] = [0.10, 0.50, 0.56];
const PLUM: [f32; 3] = [0.42, 0.22, 0.58];

/// Gradient material that blends from `from` to `to` along `direction` across the
/// shape centred at `center` (half the gradient length is `half_extent`).
fn gradient(
    from: [f32; 3],
    to: [f32; 3],
    direction: [f32; 3],
    center: Vec3,
    half_extent: f32,
) -> GradMeshMaterial {
    let mut mat = dual_directional(from, to, direction);
    mat.base.perceptual_roughness = 0.45;
    with_gradient_bounds(mat, center, half_extent)
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
) {
    // Camera - angled view that fills the frame
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 7.5, 11.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
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
        color: Color::srgb(0.6, 0.6, 0.7),
        brightness: 300.0,
        affects_lightmapped_meshes: false,
    });

    // Ground plane
    let ground = meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(20.0)));
    commands.spawn((
        Mesh3d(ground),
        MeshMaterial3d(materials.add(opaque(0.17, 0.16, 0.22))),
        Transform::from_translation(Vec3::new(0.0, -0.01, 0.0)),
    ));

    // --- Open zigzag: butter → pink along the path (X), mitred joints ---
    let origin = Vec3::new(-4.0, 0.9, -2.6);
    let pts = zigzag_points(-2.6, 2.6, 6, 0.9);
    let mesh = meshes.add(Shape::polyline3d(pts, 0.11, false, 12, ShapeDetail::Medium).build());
    commands.spawn((
        Mesh3d(mesh),
        MeshMaterial3d(materials.add(gradient(BUTTER, PINK, [1.0, 0.0, 0.0], origin, 2.6))),
        Transform::from_translation(origin),
    ));
    spawn_label(&mut commands, "Open Zigzag", origin + Vec3::new(0.0, 1.4, 0.0));

    // --- Closed hexagon: lifted well clear of the ground, rose → pink vertically ---
    let origin = Vec3::new(4.0, 1.9, -2.6);
    let pts = hexagon_points(1.3);
    let mesh = meshes.add(Shape::polyline3d(pts, 0.1, true, 12, ShapeDetail::Medium).build());
    commands.spawn((
        Mesh3d(mesh),
        MeshMaterial3d(materials.add(gradient(ROSE, PINK, [0.0, 1.0, 0.0], origin, 1.3))),
        Transform::from_translation(origin),
    ));
    spawn_label(&mut commands, "Closed Loop", origin + Vec3::new(0.0, 1.8, 0.0));

    // --- Thick zigzag: lilac → sky ---
    let origin = Vec3::new(-4.0, 0.9, 2.4);
    let pts = zigzag_points(-2.6, 2.6, 4, 1.0);
    let mesh = meshes.add(Shape::polyline3d(pts, 0.22, false, 16, ShapeDetail::Medium).build());
    commands.spawn((
        Mesh3d(mesh),
        MeshMaterial3d(materials.add(gradient(LILAC, SKY, [1.0, 0.0, 0.0], origin, 2.6))),
        Transform::from_translation(origin),
    ));
    spawn_label(&mut commands, "Thick Line", origin + Vec3::new(0.0, 1.5, 0.0));

    // --- Thin detailed zigzag: deep plum → pink ---
    let origin = Vec3::new(4.0, 0.9, 2.4);
    let pts = zigzag_points(-2.6, 2.6, 8, 0.55);
    let mesh = meshes.add(Shape::polyline3d(pts, 0.05, false, 8, ShapeDetail::Medium).build());
    commands.spawn((
        Mesh3d(mesh),
        MeshMaterial3d(materials.add(gradient(PLUM, PINK, [1.0, 0.0, 0.0], origin, 2.6))),
        Transform::from_translation(origin),
    ));
    spawn_label(&mut commands, "Thin Line", origin + Vec3::new(0.0, 1.2, 0.0));

    // --- 3D spiral: teal → butter bottom to top ---
    let origin = Vec3::new(0.0, 1.8, 0.0);
    let pts = spiral_points(4, 1.2, 3.2);
    let mesh = meshes.add(Shape::polyline3d(pts, 0.07, false, 10, ShapeDetail::Medium).build());
    commands.spawn((
        Mesh3d(mesh),
        MeshMaterial3d(materials.add(gradient(TEAL, BUTTER, [0.0, 1.0, 0.0], origin, 1.6))),
        Transform::from_translation(origin),
    ));
    spawn_label(&mut commands, "3D Spiral", origin + Vec3::new(0.0, 2.2, 0.0));
    let _ = MINT;
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
        TextFont {
            font_size: FontSize::Px(15.0),
            ..default()
        },
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
    let Ok((camera, camera_transform)) = camera.single() else {
        return;
    };
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
