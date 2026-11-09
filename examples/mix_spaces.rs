//! Colour-space example - the same two colours blended in each `MixSpace`.
//!
//! Each row blends in one space; each column uses one colour pair. Blue to
//! yellow turns grey in the middle in sRGB and linear RGB; OKLAB keeps the
//! lightness steady and OKLCH sweeps through the hues in between.
//!
//! Run with: `cargo run --example mix_spaces`

mod common;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{
    GradMeshMaterial, GradMeshPlugin, MixSpace, Shape, ShapeDetail, dual_directional,
    with_gradient_bounds, with_mix_space,
};
use common::screenshot::AutoScreenshotPlugin;

const STRIP: Vec2 = Vec2::new(5.0, 1.2);
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
    mut materials: ResMut<Assets<GradMeshMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.0, 10.5).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 12_000.0,
            ..default()
        },
        Transform::from_xyz(0.0, 2.0, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.insert_resource(GlobalAmbientLight {
        color: Color::WHITE,
        brightness: 400.0,
        affects_lightmapped_meshes: false,
    });

    let strip = meshes.add(Shape::rectangle(STRIP, ShapeDetail::Low).build());
    for (row, (space, label)) in SPACES.into_iter().enumerate() {
        let y = 2.4 - row as f32 * 1.6;
        for (col, (from, to)) in PAIRS.into_iter().enumerate() {
            let center = Vec3::new(if col == 0 { -3.1 } else { 3.1 }, y, 0.0);
            let material = with_mix_space(dual_directional(from, to, [1.0, 0.0, 0.0]), space);
            commands.spawn((
                Mesh3d(strip.clone()),
                MeshMaterial3d(materials.add(with_gradient_bounds(
                    material,
                    center,
                    STRIP.x * 0.5,
                ))),
                Transform::from_translation(center),
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
                top: Val::Px(360.0 - y * 70.0 - 10.0),
                ..default()
            },
        ));
    }
}
