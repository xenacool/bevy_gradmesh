//! Bezier curves example - demonstrates curve primitives as tubes.
//!
//! Shows linear, quadratic, quartic and cubic Bezier curves and a helix rendered as tube
//! geometry through the standalone pipeline (`BatchRecorder` + `render_standalone_batches`),
//! so it needs no PBR feature. There are no lights: tubes are unlit, and the gradient runs
//! along a fixed world-space axis (about ten world units from end to end, centred on the
//! origin) instead of being fitted to each tube, so the colours near the origin show the full
//! blend.
//!
//! Run with: `cargo run --example curves`

mod common;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{
    BatchRecorder, BlendMode, ColorMode, GradMeshPlugin, InstanceProperties, MixSpace, Shape,
    ShapeDetail, render_standalone_batches,
};
use common::screenshot::AutoScreenshotPlugin;


/// Run this example. Called by the WASM runner.
pub fn run(app: &mut App) {
    app
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
                .with_description("Bezier curve primitives rendered as tubes"),
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
                .with_description("Bezier curve primitives rendered as tubes"),
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

/// Records `shape` at `origin` with a gradient from `from` to `to` along the world-space
/// `direction`.
fn add_gradient(
    recorder: &mut BatchRecorder,
    shape: &Shape,
    origin: Vec3,
    from: [f32; 3],
    to: [f32; 3],
    direction: Vec3,
) {
    recorder.set_position(origin);
    recorder.add_instance(
        shape,
        InstanceProperties::from_style(
            [from[0], from[1], from[2], 1.0],
            [to[0], to[1], to[2], 1.0],
            [direction.x, direction.y, direction.z, 0.0],
            ColorMode::DualDirectional,
            BlendMode::Opaque,
            MixSpace::default(),
            None,
        ),
    );
}

fn setup(mut commands: Commands, meshes: ResMut<Assets<Mesh>>) {
    // Camera - angled view that fills the frame
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 9.5, 13.0).looking_at(Vec3::new(0.0, 1.2, 0.0), Vec3::Y),
    ));

    let mut recorder = BatchRecorder::new();
    recorder.begin();

    // Ground slab
    recorder.set_position(Vec3::new(0.0, -0.01, 0.0));
    recorder.cuboid(
        Vec3::new(44.0, 0.02, 44.0),
        0.0,
        ShapeDetail::Low,
        [0.17, 0.16, 0.22, 1.0],
    );

    // Helper: record a Bezier tube at `origin` with a gradient along X.
    let add_curve = |commands: &mut Commands,
                         recorder: &mut BatchRecorder,
                         points: Vec<Vec3>,
                         thickness: f32,
                         origin: Vec3,
                         colors: ([f32; 3], [f32; 3]),
                         label: &str,
                         label_offset: Vec3| {
        let shape = Shape::bezier_curve(points, thickness, 12, ShapeDetail::Medium);
        add_gradient(recorder, &shape, origin, colors.0, colors.1, Vec3::X);
        spawn_label(commands, label, origin + label_offset);
    };

    // Linear
    add_curve(
        &mut commands,
        &mut recorder,
        vec![Vec3::new(-3.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.0)],
        0.09,
        Vec3::new(-4.3, 0.9, -3.0),
        (ROSE, PINK),
        "Linear",
        Vec3::new(0.0, 0.8, 0.0),
    );

    // Quadratic arc
    add_curve(
        &mut commands,
        &mut recorder,
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
    add_curve(
        &mut commands,
        &mut recorder,
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

    // Five-point (quartic) Bezier: higher-degree curves use every control point
    add_curve(
        &mut commands,
        &mut recorder,
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
    let helix_shape = Shape::polyline3d(helix, 0.1, false, 12, ShapeDetail::Medium);
    add_gradient(&mut recorder, &helix_shape, origin, LILAC, BUTTER, Vec3::Y);
    spawn_label(&mut commands, "3D Helix", Vec3::new(0.0, 4.5, 0.0));

    let _ = recorder.end();
    render_standalone_batches(&mut recorder, &mut commands, meshes, None);
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
