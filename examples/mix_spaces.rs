//! Colour-space example - the same two colours blended in each `MixSpace`.
//!
//! Each row blends in one space; each column uses one colour pair. Blue to
//! yellow turns grey in the middle in sRGB and linear RGB; OKLAB keeps the
//! lightness steady and OKLCH sweeps through the hues in between.
//!
//! The strips are unlit `Mesh2d` rectangles with `GradMeshMaterial2d`, so the
//! example needs no PBR. The 2D camera is scaled so each strip lands on the
//! screen pixels that a 3D camera 10.5 units away with a 45 degree field of view
//! would give (`tests/mix_space_screenshot.rs` samples those pixels).
//!
//! Run with: `cargo run --example mix_spaces`

mod common;

use bevy::mesh::Mesh2d;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::sprite_render::MeshMaterial2d;
use bevy_gradmesh::{
    GradMeshMaterial2d, GradMeshPlugin, MixSpace, Shape, ShapeDetail, Style,
};
use common::screenshot::AutoScreenshotPlugin;

const STRIP: Vec2 = Vec2::new(5.0, 1.2);
const WINDOW_HEIGHT: f32 = 720.0;
const CAMERA_Z: f32 = 10.5;
const FOV_Y: f32 = std::f32::consts::FRAC_PI_4;
const PAIRS: [([f32; 3], [f32; 3]); 2] = [
    ([0.10, 0.25, 0.95], [1.00, 0.88, 0.15]),
    ([1.00, 0.20, 0.35], [0.15, 0.85, 0.65]),
];
const SPACES: [(MixSpace, &str); 4] = [
    (MixSpace::Oklab, "OKLAB (default)"),
    (MixSpace::Oklch, "OKLCH"),
    (MixSpace::LinearRgb, "Linear RGB"),
    (MixSpace::Srgb, "sRGB"),
];


/// Run this example. Called by the WASM runner.
pub fn run(app: &mut App) {
    app
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "BevyGradmesh Mix Spaces".into(),
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
            AutoScreenshotPlugin::new("screenshot_mix_spaces.png")
                .with_title("Mix Spaces")
                .with_description("One gradient, four colour spaces"),
        )
        .insert_resource(ClearColor(Color::srgb(0.1, 0.1, 0.12)))
        .add_systems(Startup, setup)
        ;
}

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "BevyGradmesh Mix Spaces".into(),
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
            AutoScreenshotPlugin::new("screenshot_mix_spaces.png")
                .with_title("Mix Spaces")
                .with_description("One gradient, four colour spaces"),
        )
        .insert_resource(ClearColor(Color::srgb(0.1, 0.1, 0.12)))
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial2d>>,
) {
    commands.spawn((Camera2d, Msaa::Off));

    // Window pixels per scene unit of the 3D view this layout was designed for.
    let unit = WINDOW_HEIGHT * 0.5 / (CAMERA_Z * (FOV_Y * 0.5).tan());
    let strip = meshes.add(Shape::rectangle(Vec2::ONE, ShapeDetail::Low).build_2d());
    for (row, (space, label)) in SPACES.into_iter().enumerate() {
        let y = 2.4 - row as f32 * 1.6;
        for (col, (from, to)) in PAIRS.into_iter().enumerate() {
            let x = if col == 0 { -3.1 } else { 3.1 };
            let style = Style::dual_directional(
                Color::srgb(from[0], from[1], from[2]),
                Color::srgb(to[0], to[1], to[2]),
                Vec3::X,
            )
            .with_mix_space(space);
            commands.spawn((
                Mesh2d(strip.clone()),
                MeshMaterial2d(materials.add(style.build_2d())),
                Transform::from_xyz(x * unit, y * unit, 0.0)
                    .with_scale((STRIP * unit).extend(1.0)),
            ));
        }
        commands.spawn((
            Text::new(label),
            TextFont {
                font_size: FontSize::Px(16.0),
                ..default()
            },
            TextColor(Color::srgb(0.88, 0.91, 0.98)),
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(14.0),
                top: Val::Px(WINDOW_HEIGHT * 0.5 - y * unit - 10.0),
                ..default()
            },
        ));
    }
}
