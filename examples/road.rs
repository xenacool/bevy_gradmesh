//! Banked roads with `Shape::sweep_frames`: an open `CatmullRomSpline` carries a
//! width and a roll channel, `BakedSpline::sweep_frames` turns the baked frames
//! into sweep frames, and the road leans into its bends and widens and narrows
//! along the way. A flat ribbon on a straight path shows `SweepOptions::roll_with`
//! turning once around its own axis.
//!
//! Run with: `cargo run --example road`

mod common;

use bevy::light::NotShadowCaster;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{
    CatmullRomSpline, GradMeshMaterial, GradMeshMaterialPool, GradMeshPlugin, MeshCache, Shape,
    ShapeDetail, Style, SweepOptions,
};
use common::screenshot::{AutoScreenshotPlugin, spawn_world_label};
use std::f32::consts::TAU;


/// Run this example. Called by the WASM runner.
pub fn run(app: &mut App) {
    app
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "BevyGradmesh Road".into(),
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
            AutoScreenshotPlugin::new("screenshot_road.png")
                .with_title("Banked road")
                .with_description("Spline channels for width and roll, caller-supplied sweep frames, roll along a path")
                .wait_frames(3),
        )
        .add_systems(Startup, setup)
        ;
}

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "BevyGradmesh Road".into(),
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
            AutoScreenshotPlugin::new("screenshot_road.png")
                .with_title("Banked road")
                .with_description("Spline channels for width and roll, caller-supplied sweep frames, roll along a path")
                .wait_frames(3),
        )
        .add_systems(Startup, setup)
        .run();
}

/// Road cross-section, one unit wide: the width channel scales it across.
fn road_profile() -> Vec<Vec2> {
    vec![
        Vec2::new(-0.5, 0.0),
        Vec2::new(0.5, 0.0),
        Vec2::new(0.5, 0.12),
        Vec2::new(0.45, 0.12),
        Vec2::new(0.45, 0.04),
        Vec2::new(-0.45, 0.04),
        Vec2::new(-0.45, 0.12),
        Vec2::new(-0.5, 0.12),
    ]
}

/// A thin flat ribbon.
fn ribbon_profile() -> Vec<Vec2> {
    vec![
        Vec2::new(-0.9, -0.05),
        Vec2::new(0.9, -0.05),
        Vec2::new(0.9, 0.05),
        Vec2::new(-0.9, 0.05),
    ]
}

fn setup(
    mut commands: Commands,
    mut cache: ResMut<MeshCache>,
    mut pool: ResMut<GradMeshMaterialPool>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 13.0, 12.5).looking_at(Vec3::new(0.0, 0.0, -0.8), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 14_000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(-4.0, 9.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.65, 0.65, 0.8),
        brightness: 350.0,
        affects_lightmapped_meshes: false,
    });

    let d = ShapeDetail::High;
    let mut piece =
        |commands: &mut Commands, shape: &Shape, style: &Style, transform: Transform| {
            let mut entity = commands.spawn((
                Mesh3d(cache.get_or_build_handle(shape, &mut meshes)),
                MeshMaterial3d(pool.get_or_add(style, &mut materials)),
                transform,
            ));
            if style.is_translucent() {
                entity.insert(NotShadowCaster);
            }
        };

    piece(
        &mut commands,
        &Shape::cuboid(Vec3::new(22.0, 0.2, 15.0), 0.0, d),
        &Style::shaded(Color::srgb(0.62, 0.78, 0.58), 0.6).with_perceptual_roughness(0.95),
        // Well below the road: a banked road tilts one of its edges under its
        // own centre line, and it must never cut into the ground.
        Transform::from_xyz(0.0, -1.1, 0.0),
    );

    // The road: an open spline, so it has two ends instead of a loop. Channel 0
    // is the width, channel 1 the roll in radians; both are interpolated with
    // the position's own weights, so they stay in step with it.
    // A sine wave of 12 units wavelength and 1.4 amplitude, with eight control
    // points per wave. A road folds over itself on the inside of a bend whose
    // radius is smaller than its half-width, so the example checks the tightest
    // turn of the curve against the widest road.
    let k = std::f32::consts::PI / 6.0;
    let xs: Vec<f32> = (0..13).map(|i| -9.0 + i as f32 * 1.5).collect();
    let spline = CatmullRomSpline::open(
        xs.iter()
            .map(|&x| Vec3::new(x, 0.5 + 0.5 * (x / 9.0 * std::f32::consts::FRAC_PI_2).sin(), -2.4 + 1.4 * (x * k).sin()))
            .collect::<Vec<_>>(),
    ).unwrap()
    .with_channel(&xs.iter().map(|&x| 1.6 + 0.3 * (x * k * 0.5).cos()).collect::<Vec<_>>())
    .with_channel(&xs.iter().map(|&x| 0.3 * (x * k).sin()).collect::<Vec<_>>());
    let widest = 1.9;
    debug_assert!(
        spline.min_turn_radius() > widest,
        "the road is wider than its tightest bend allows"
    );
    let baked = spline.bake(0.25);
    let frames = baked.sweep_frames(Some(0), Some(1));
    let road = Shape::sweep_frames(road_profile(), &frames, SweepOptions::default().with_detail(d));
    piece(
        &mut commands,
        &road,
        &Style::shaded(Color::srgb(0.45, 0.45, 0.5), 0.5).with_perceptual_roughness(0.9),
        Transform::IDENTITY,
    );

    // The ribbon: a straight path, one full turn of roll spread over its length.
    let ribbon_path: Vec<Vec3> = (0..=30)
        .map(|i| Vec3::new(-6.0 + i as f32 * 0.4, 1.4, 4.2))
        .collect();
    let ribbon = Shape::sweep(
        ribbon_profile(),
        ribbon_path,
        SweepOptions::default()
            .with_detail(d)
            .with_tint_by_length(vec![
                (0.0, Color::srgb(0.95, 0.45, 0.35)),
                (1.0, Color::srgb(0.35, 0.6, 0.95)),
            ])
            .roll_with(32, |t| t * TAU),
    );
    piece(
        &mut commands,
        &ribbon,
        &Style::shaded(Color::WHITE, 0.4).with_perceptual_roughness(0.5),
        Transform::IDENTITY,
    );

    spawn_world_label(&mut commands, "Width and roll channels on an open spline", Vec3::new(0.0, 1.8, -5.6));
    spawn_world_label(&mut commands, "Ribbon: roll_with, one full turn", Vec3::new(0.0, 1.4, 5.6));
}
