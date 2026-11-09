//! Profiles swept along paths with `Shape::sweep`: a wall that follows a baked
//! spline loop with an arched gate and windows cut through it, a kerbed road on
//! an S-curve with a length tint, and a tapering pipe winding up a helix.
//!
//! Run with: `cargo run --example sweep`

mod common;

use bevy::light::NotShadowCaster;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{
    CatmullRomSpline, GradMeshMaterial, GradMeshMaterialPool, GradMeshPlugin, MeshCache, Shape,
    ShapeDetail, Style, SweepOpening, SweepOptions,
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
                        title: "BevyGradmesh Sweep".into(),
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
            AutoScreenshotPlugin::new("screenshot_sweep.png")
                .with_title("Sweep")
                .with_description("A wall on a spline loop, a kerbed road and a tapering pipe")
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
                        title: "BevyGradmesh Sweep".into(),
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
            AutoScreenshotPlugin::new("screenshot_sweep.png")
                .with_title("Sweep")
                .with_description("A wall on a spline loop, a kerbed road and a tapering pipe")
                .wait_frames(3),
        )
        .add_systems(Startup, setup)
        .run();
}

/// Wall cross-section: a tapered body with a flat top, `x` across and `y` up.
fn wall_profile() -> Vec<Vec2> {
    vec![
        Vec2::new(-0.35, 0.0),
        Vec2::new(0.35, 0.0),
        Vec2::new(0.22, 1.4),
        Vec2::new(0.32, 1.4),
        Vec2::new(0.32, 1.7),
        Vec2::new(-0.32, 1.7),
        Vec2::new(-0.32, 1.4),
        Vec2::new(-0.22, 1.4),
    ]
}

/// Road cross-section: a shallow trough between two raised kerbs.
fn road_profile() -> Vec<Vec2> {
    vec![
        Vec2::new(-1.5, 0.0),
        Vec2::new(1.5, 0.0),
        Vec2::new(1.5, 0.16),
        Vec2::new(1.28, 0.16),
        Vec2::new(1.28, 0.05),
        Vec2::new(-1.28, 0.05),
        Vec2::new(-1.28, 0.16),
        Vec2::new(-1.5, 0.16),
    ]
}

fn circle_profile(radius: f32, points: usize) -> Vec<Vec2> {
    (0..points)
        .map(|i| {
            let a = i as f32 / points as f32 * TAU;
            Vec2::new(a.cos(), a.sin()) * radius
        })
        .collect()
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
        Transform::from_xyz(0.8, 9.0, 17.5).looking_at(Vec3::new(0.8, 1.6, 0.8), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 14_000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(5.0, 9.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
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

    // Ground.
    piece(
        &mut commands,
        &Shape::cuboid(Vec3::new(26.0, 0.2, 18.0), 0.0, d),
        &Style::shaded(Color::srgb(0.62, 0.78, 0.58), 0.6).with_perceptual_roughness(0.95),
        Transform::from_xyz(0.0, -0.1, 0.0),
    );

    // Wall: a closed sweep along a baked spline loop. `openings` cut a real gate
    // and windows through it; `t` is the stretch of the loop and `y` the height.
    // The loop starts on its +X side and runs towards +Z, so 0.25 is the front.
    let centre = Vec3::new(-5.0, 0.0, -0.5);
    let ring = CatmullRomSpline::closed(
        (0..7)
            .map(|i| {
                let a = i as f32 / 7.0 * TAU;
                let r = 3.4 + 0.7 * (a * 2.0).sin();
                centre + Vec3::new(a.cos() * r, 0.0, a.sin() * r)
            })
            .collect::<Vec<_>>(),
    ).unwrap();
    let baked = ring.bake(0.5);
    let wall = Shape::sweep(
        wall_profile(),
        baked.frames().iter().map(|f| f.position).collect(),
        SweepOptions::default()
            .with_closed(true)
            .with_detail(d)
            .with_openings(vec![
                SweepOpening::arch(0.205..0.295, 0.0, 0.9, 1.35, 10),
                SweepOpening::rect(0.455..0.485, 0.6, 1.1),
                SweepOpening::rect(0.715..0.745, 0.6, 1.1),
            ]),
    );
    piece(
        &mut commands,
        &wall,
        &Style::shaded(Color::srgb(0.93, 0.85, 0.72), 0.5)
            .with_perceptual_roughness(0.85)
            .with_noise(4.0, 0.15),
        Transform::IDENTITY,
    );

    // Road: an S-curve that climbs a little, tinted dark to light along its length.
    let road_path: Vec<Vec3> = (0..=40)
        .map(|i| {
            let t = i as f32 / 40.0;
            Vec3::new(-2.0 + t * 12.0, t * 0.8, 4.2 + (t * TAU).sin() * 1.8)
        })
        .collect();
    let road = Shape::sweep(
        road_profile(),
        road_path,
        SweepOptions::default()
            .with_tint_by_length(vec![
                (0.0, Color::srgb(0.55, 0.55, 0.6)),
                (1.0, Color::srgb(1.0, 0.95, 0.85)),
            ])
            .with_detail(d),
    );
    piece(
        &mut commands,
        &road,
        &Style::shaded(Color::srgb(0.45, 0.45, 0.5), 0.5).with_perceptual_roughness(0.9),
        Transform::IDENTITY,
    );

    spawn_world_label(&mut commands, "Wall: closed sweep + openings", Vec3::new(-5.0, 3.3, -0.5));
    spawn_world_label(&mut commands, "Road: tint by length", Vec3::new(4.0, 1.4, 6.6));
    spawn_world_label(&mut commands, "Pipe: scale along a helix", Vec3::new(5.0, 5.4, -1.5));

    // Pipe: a helix that tapers towards the top.
    let helix: Vec<Vec3> = (0..=90)
        .map(|i| {
            let t = i as f32 / 90.0;
            let a = t * TAU * 1.75;
            Vec3::new(5.0 + a.cos() * 1.7, 0.4 + t * 4.2, -1.5 + a.sin() * 1.7)
        })
        .collect();
    let pipe = Shape::sweep(
        circle_profile(0.38, 14),
        helix,
        SweepOptions::default()
            .with_detail(d)
            .scale_with(8, |t| Vec2::splat(1.0 - 0.55 * t)),
    );
    piece(
        &mut commands,
        &pipe,
        &Style::shaded(Color::srgb(0.85, 0.5, 0.25), 0.6)
            .with_metallic(0.8)
            .with_perceptual_roughness(0.35),
        Transform::IDENTITY,
    );
}
