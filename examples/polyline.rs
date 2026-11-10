//! Polyline example - demonstrates polylines as first-class primitives.
//!
//! Shows open and closed flat polylines with different thicknesses and styles, drawn with
//! `Camera2d`, `Mesh2d` and the unlit `GradMeshMaterial2d`, so it needs no PBR feature. Units
//! are pixels and the gradient fades along each line. The 3D tube version (`Shape::polyline3d`)
//! needs a lit material or the standalone pipeline, whose gradient is a fixed world-space
//! one, so the 3D spiral became a flat spiral; lights and the ground plane are gone.
//!
//! Run with: `cargo run --example polyline`

mod common;

use bevy::mesh::Mesh2d;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::sprite_render::MeshMaterial2d;
use bevy_gradmesh::{GradMeshMaterial2d, GradMeshPlugin, Shape, Style};
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
                .with_description("Flat polyline primitives with the unlit 2D gradient material"),
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
                .with_description("Flat polyline primitives with the unlit 2D gradient material"),
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

/// Unlit gradient material that blends from `from` to `to` along the line.
fn gradient(from: [f32; 3], to: [f32; 3]) -> GradMeshMaterial2d {
    Style::dual_directional(
        Color::srgb(from[0], from[1], from[2]),
        Color::srgb(to[0], to[1], to[2]),
        Vec3::X,
    )
    .build_2d()
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial2d>>,
) {
    commands.spawn((Camera2d, Msaa::Off));

    let mut spawn_line = |commands: &mut Commands,
                          shape: Shape,
                          material: GradMeshMaterial2d,
                          origin: Vec3,
                          label: &str,
                          label_offset: Vec3| {
        commands.spawn((
            Mesh2d(meshes.add(shape.build_2d())),
            MeshMaterial2d(materials.add(material)),
            Transform::from_translation(origin),
        ));
        spawn_label(commands, label, origin + label_offset);
    };

    // --- Open zigzag: butter -> pink along the path, mitred joints ---
    spawn_line(
        &mut commands,
        Shape::polyline_2d(zigzag_points(-156.0, 156.0, 6, 54.0), 14.0, false),
        gradient(BUTTER, PINK),
        Vec3::new(-260.0, 110.0, 0.0),
        "Open Zigzag",
        Vec3::new(0.0, 100.0, 0.0),
    );

    // --- Closed hexagon: rose -> pink around the loop ---
    spawn_line(
        &mut commands,
        Shape::polyline_2d(hexagon_points(78.0), 12.0, true),
        gradient(ROSE, PINK),
        Vec3::new(260.0, 120.0, 0.0),
        "Closed Loop",
        Vec3::new(0.0, 110.0, 0.0),
    );

    // --- Thick zigzag: lilac -> sky ---
    spawn_line(
        &mut commands,
        Shape::polyline_2d(zigzag_points(-156.0, 156.0, 4, 60.0), 26.0, false),
        gradient(LILAC, SKY),
        Vec3::new(-260.0, -150.0, 0.0),
        "Thick Line",
        Vec3::new(0.0, 100.0, 0.0),
    );

    // --- Thin detailed zigzag: deep plum -> pink ---
    spawn_line(
        &mut commands,
        Shape::polyline_2d(zigzag_points(-156.0, 156.0, 8, 33.0), 5.0, false),
        gradient(PLUM, PINK),
        Vec3::new(260.0, -150.0, 0.0),
        "Thin Line",
        Vec3::new(0.0, 70.0, 0.0),
    );

    // --- Spiral: teal -> butter from the centre outwards ---
    spawn_line(
        &mut commands,
        Shape::polyline_2d(spiral_points(4, 90.0), 7.0, false),
        gradient(TEAL, BUTTER),
        Vec3::new(0.0, -10.0, 0.0),
        "Spiral",
        Vec3::new(0.0, 120.0, 0.0),
    );
    let _ = MINT;
}

/// Generate zigzag points for polyline demos.
fn zigzag_points(x_min: f32, x_max: f32, peaks: usize, amplitude: f32) -> Vec<Vec2> {
    let mut pts = Vec::new();
    for i in 0..=peaks {
        let t = i as f32 / peaks as f32;
        let x = x_min + t * (x_max - x_min);
        let y = if i % 2 == 0 { -amplitude * 0.5 } else { amplitude * 0.5 };
        pts.push(Vec2::new(x, y));
    }
    pts
}

/// Generate hexagon points for closed loop demo.
fn hexagon_points(radius: f32) -> Vec<Vec2> {
    (0..6)
        .map(|i| {
            let a = i as f32 * std::f32::consts::TAU / 6.0;
            Vec2::new(a.cos() * radius, a.sin() * radius)
        })
        .collect()
}

/// Generate a flat spiral path whose radius grows from the centre to `radius`.
fn spiral_points(turns: u32, radius: f32) -> Vec<Vec2> {
    let steps = (turns * 20) as usize;
    (0..=steps)
        .map(|i| {
            let t = i as f32 / steps as f32;
            let angle = t * turns as f32 * std::f32::consts::TAU;
            Vec2::new(angle.cos(), angle.sin()) * (t * radius)
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
    camera: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
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
