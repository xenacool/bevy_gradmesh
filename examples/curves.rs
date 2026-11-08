//! Bézier curves example - demonstrates curve primitives as tubes.
//!
//! Shows linear, quadratic, and cubic Bézier curves rendered as tube geometry.
//!
//! Run with: `cargo run --example curves`

mod common;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{
    GradMeshMaterial, GradMeshPlugin, Shape, ShapeDetail, dual_directional, opaque,
    with_gradient_bounds,
};
use common::screenshot::AutoScreenshotPlugin;

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "BevyGradmesh Curves".into(),
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
            AutoScreenshotPlugin::new("screenshot_curves.png")
                .with_title("Curves")
                .with_description("Bézier curve primitives rendered as tubes"),
        )
        .insert_resource(ClearColor(Color::srgb(0.1, 0.1, 0.12)))
        .add_systems(Startup, setup)
        .add_systems(PostUpdate, position_labels)
        .run();
}

#[derive(Component)]
struct WorldLabel(Vec3);

const BUTTER: [f32; 3] = [1.0, 0.93, 0.58];
const PINK: [f32; 3] = [1.0, 0.70, 0.82];
const MINT: [f32; 3] = [0.62, 0.93, 0.80];
const SKY: [f32; 3] = [0.62, 0.80, 1.0];
const LILAC: [f32; 3] = [0.82, 0.72, 0.98];
const ROSE: [f32; 3] = [0.80, 0.25, 0.42];
const TEAL: [f32; 3] = [0.10, 0.50, 0.56];
const PLUM: [f32; 3] = [0.42, 0.22, 0.58];

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
        Transform::from_xyz(0.0, 9.5, 13.0).looking_at(Vec3::new(0.0, 1.2, 0.0), Vec3::Y),
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
    let ground = meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(22.0)));
    commands.spawn((
        Mesh3d(ground),
        MeshMaterial3d(materials.add(opaque(0.17, 0.16, 0.22))),
        Transform::from_translation(Vec3::new(0.0, -0.01, 0.0)),
    ));

    // Helper: spawn a Bézier tube at `origin` with a gradient along X.
    let mut spawn_curve = |commands: &mut Commands,
                           points: Vec<Vec3>,
                           thickness: f32,
                           origin: Vec3,
                           colors: ([f32; 3], [f32; 3]),
                           label: &str,
                           label_offset: Vec3| {
        let mesh = meshes.add(Shape::bezier_curve(points, thickness, 12, ShapeDetail::Medium).build());
        commands.spawn((
            Mesh3d(mesh),
            MeshMaterial3d(materials.add(gradient(colors.0, colors.1, [1.0, 0.0, 0.0], origin, 3.0))),
            Transform::from_translation(origin),
        ));
        spawn_label(commands, label, origin + label_offset);
    };

    // Linear
    spawn_curve(
        &mut commands,
        vec![Vec3::new(-3.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.0)],
        0.09,
        Vec3::new(-4.3, 0.9, -3.0),
        (ROSE, PINK),
        "Linear",
        Vec3::new(0.0, 0.8, 0.0),
    );

    // Quadratic arc
    spawn_curve(
        &mut commands,
        vec![
            Vec3::new(-3.0, 0.0, 0.0),
            Vec3::new(0.0, 2.5, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        ],
        0.09,
        Vec3::new(4.3, 0.9, -3.0),
        (MINT, BUTTER),
        "Quadratic Arc",
        Vec3::new(0.0, 2.2, 0.0),
    );

    // Cubic S-curve
    spawn_curve(
        &mut commands,
        vec![
            Vec3::new(-3.0, 0.0, 0.0),
            Vec3::new(-1.5, 2.0, 0.0),
            Vec3::new(1.5, -2.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        ],
        0.09,
        Vec3::new(-4.3, 1.4, 3.0),
        (TEAL, SKY),
        "Cubic S-Curve",
        Vec3::new(0.0, 1.6, 0.0),
    );

    // Five-point (quartic) Bézier: higher-degree curves use every control point
    spawn_curve(
        &mut commands,
        vec![
            Vec3::new(-3.0, 0.0, 0.0),
            Vec3::new(-2.0, 3.0, 0.0),
            Vec3::new(0.0, -1.5, 0.0),
            Vec3::new(2.0, 3.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        ],
        0.09,
        Vec3::new(4.3, 1.2, 3.0),
        (PLUM, PINK),
        "Quartic (5 points)",
        Vec3::new(0.0, 2.0, 0.0),
    );

    // --- 3D helix: sampled from cos/sin so it is a true helix ---
    let turns = 3.0_f32;
    let radius = 1.0_f32;
    let height = 3.6_f32;
    let steps = 96;
    let helix: Vec<Vec3> = (0..=steps)
        .map(|i| {
            let t = i as f32 / steps as f32;
            let angle = t * turns * std::f32::consts::TAU;
            Vec3::new(angle.cos() * radius, t * height, angle.sin() * radius)
        })
        .collect();
    let origin = Vec3::new(0.0, 0.2, 0.0);
    let helix_mesh = meshes.add(Shape::polyline3d(helix, 0.1, false, 12, ShapeDetail::Medium).build());
    commands.spawn((
        Mesh3d(helix_mesh),
        MeshMaterial3d(materials.add(gradient(
            LILAC,
            BUTTER,
            [0.0, 1.0, 0.0],
            origin + Vec3::new(0.0, height * 0.5, 0.0),
            height * 0.5,
        ))),
        Transform::from_translation(origin),
    ));
    spawn_label(&mut commands, "3D Helix", Vec3::new(0.0, 4.5, 0.0));
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
