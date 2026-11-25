//! Spline relaxation (`CatmullRomSpline::relaxed`), `BakedSpline::sample_at_distance`, and
//! twisted capped sweeps (`SweepOptions::with_twist`, `with_caps`).
//!
//! - **Left**: a tightly pinched spline (thin crimson preview) compared against its
//!   `CatmullRomSpline::relaxed(2.2)` smoothed track (kerbed teal-gold road), with a glowing
//!   marker gliding along it via `BakedSpline::sample_at_distance`.
//! - **Right**: a rectangular beam swept with `SweepOptions::with_twist(1.5)` and
//!   `with_caps(true)` plus a closed twisted torus-like Möbius ribbon.
//!
//! Run with: `cargo run --example spline_relax_and_twist`

mod common;

use std::f32::consts::TAU;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{
    BakedSpline, CatmullRomSpline, GradMeshMaterial, GradMeshPlugin, Shape, ShapeDetail, Style,
    SweepOptions,
};
use common::screenshot::{AutoScreenshotPlugin, spawn_world_label};

/// Run this example. Called by the WASM runner.
pub fn run(app: &mut App) {
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "BevyGradmesh Spline Relaxation & Twist Sweeps".into(),
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
        AutoScreenshotPlugin::new("screenshot_spline_relax_and_twist.png")
            .animation(4.0)
            .with_title("Spline Relaxation & Twist Sweeps")
            .with_description("CatmullRomSpline::relaxed, sample_at_distance, and SweepOptions::with_twist"),
    )
    .insert_resource(ClearColor(Color::srgb(0.07, 0.08, 0.12)))
    .add_systems(Startup, setup)
    .add_systems(Update, (drive_marker, rotate_twists));
}

fn main() {
    let mut app = App::new();
    run(&mut app);
    app.run();
}

#[derive(Resource)]
struct TrackSpline {
    baked: BakedSpline,
    offset: Vec3,
}

#[derive(Component)]
struct TrackRunner;

#[derive(Component)]
struct SpinTwist;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 7.2, 11.5).looking_at(Vec3::new(0.0, 0.4, 0.0), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 14_000.0,
            ..default()
        },
        Transform::from_xyz(5.0, 9.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.6, 0.62, 0.72),
        brightness: 480.0,
        affects_lightmapped_meshes: false,
    });

    // --- Left: Pinched spline vs CatmullRomSpline::relaxed(2.0) ---
    let left_offset = Vec3::new(-3.4, 0.0, 0.0);
    let pinched = CatmullRomSpline::closed(vec![
        Vec3::new(-2.4, 0.0, -2.0),
        Vec3::new(0.0, 0.0, -0.35), // sharp pinch inward
        Vec3::new(2.4, 0.0, -2.0),
        Vec3::new(2.2, 0.0, 2.0),
        Vec3::new(0.0, 0.0, 0.35), // sharp pinch inward
        Vec3::new(-2.2, 0.0, 2.0),
    ])
    .expect("six control points form a valid closed spline");
    let relaxed = pinched.clone().relaxed(2.0);

    let pinched_baked = pinched.bake(0.15);
    let relaxed_baked = relaxed.bake(0.15);

    let thin_profile = vec![
        Vec2::new(-0.08, 0.0),
        Vec2::new(0.08, 0.0),
        Vec2::new(0.08, 0.14),
        Vec2::new(-0.08, 0.14),
    ];
    let road_profile = vec![
        Vec2::new(-0.42, 0.0),
        Vec2::new(0.42, 0.0),
        Vec2::new(0.42, 0.12),
        Vec2::new(0.32, 0.12),
        Vec2::new(0.32, 0.04),
        Vec2::new(-0.32, 0.04),
        Vec2::new(-0.32, 0.12),
        Vec2::new(-0.42, 0.12),
    ];

    // Original pinched outline hovered slightly above in crimson.
    let pinched_pts: Vec<Vec3> = pinched_baked.frames().iter().map(|f| f.position).collect();
    let pinched_shape = Shape::sweep(
        thin_profile,
        pinched_pts,
        SweepOptions::default()
            .with_closed(true)
            .with_detail(ShapeDetail::Medium),
    );
    commands.spawn((
        Mesh3d(meshes.add(pinched_shape.build())),
        MeshMaterial3d(
            materials.add(
                Style::transparent(Color::srgba(0.96, 0.28, 0.32, 0.65)).build(),
            ),
        ),
        Transform::from_translation(left_offset + Vec3::Y * 0.25),
    ));

    // Relaxed smooth track below in teal-to-gold gradient.
    let relaxed_pts: Vec<Vec3> = relaxed_baked.frames().iter().map(|f| f.position).collect();
    let relaxed_shape = Shape::sweep(
        road_profile,
        relaxed_pts,
        SweepOptions::default()
            .with_closed(true)
            .with_detail(ShapeDetail::Medium),
    );
    commands.spawn((
        Mesh3d(meshes.add(relaxed_shape.build())),
        MeshMaterial3d(
            materials.add(
                Style::dual_directional_up(
                    Color::srgb(0.14, 0.48, 0.68),
                    Color::srgb(0.42, 0.92, 0.78),
                )
                .build(),
            ),
        ),
        Transform::from_translation(left_offset),
    ));

    // Marker gliding along relaxed_baked via sample_at_distance.
    commands.spawn((
        Mesh3d(meshes.add(Shape::sphere(0.24, ShapeDetail::Medium).build())),
        MeshMaterial3d(
            materials.add(
                Style::dual_directional_up(
                    Color::srgb(0.98, 0.52, 0.16),
                    Color::srgb(1.0, 0.92, 0.36),
                )
                .build(),
            ),
        ),
        Transform::from_translation(left_offset),
        TrackRunner,
    ));

    spawn_world_label(
        &mut commands,
        "Pinched (red) vs relaxed(2.0) + sample_at_distance",
        left_offset + Vec3::new(0.0, 2.0, -2.2),
    );

    commands.insert_resource(TrackSpline {
        baked: relaxed_baked,
        offset: left_offset,
    });

    // --- Right: SweepOptions::with_twist + with_caps ---
    let right_offset = Vec3::new(3.4, 0.4, 0.0);

    // 1. Open capped beam twisted 1.5 full turns.
    let bar_path: Vec<Vec3> = (0u16..=40)
        .map(|i| {
            let u = f32::from(i) / 40.0;
            Vec3::new((u - 0.5) * 4.4, (u * TAU).sin() * 0.35, -1.2)
        })
        .collect();
    let bar_profile = vec![
        Vec2::new(-0.48, -0.14),
        Vec2::new(0.48, -0.14),
        Vec2::new(0.48, 0.14),
        Vec2::new(-0.48, 0.14),
    ];
    let twisted_bar = Shape::sweep(
        bar_profile.clone(),
        bar_path,
        SweepOptions::default()
            .with_twist(1.5)
            .with_caps(true)
            .with_tint_by_length(vec![
                (0.0, Color::srgb(0.95, 0.32, 0.42)),
                (0.5, Color::srgb(1.0, 0.82, 0.28)),
                (1.0, Color::srgb(0.32, 0.88, 0.96)),
            ])
            .with_detail(ShapeDetail::Medium),
    );
    commands.spawn((
        Mesh3d(meshes.add(twisted_bar.build())),
        MeshMaterial3d(materials.add(Style::opaque(Color::WHITE).build())),
        Transform::from_translation(right_offset),
    ));

    // 2. Closed ring with 1 full twist.
    let ring_path: Vec<Vec3> = (0u16..48)
        .map(|i| {
            let a = f32::from(i) / 48.0 * TAU;
            Vec3::new(a.cos() * 1.45, 0.0, a.sin() * 1.45)
        })
        .collect();
    let twisted_ring = Shape::sweep(
        bar_profile,
        ring_path,
        SweepOptions::default()
            .with_closed(true)
            .with_twist(1.0)
            .with_detail(ShapeDetail::Medium),
    );
    commands.spawn((
        Mesh3d(meshes.add(twisted_ring.build())),
        MeshMaterial3d(
            materials.add(
                Style::dual_directional_up(
                    Color::srgb(0.48, 0.22, 0.86),
                    Color::srgb(0.98, 0.46, 0.8),
                )
                .build(),
            ),
        ),
        Transform::from_translation(right_offset + Vec3::new(0.0, 0.0, 1.5)),
        SpinTwist,
    ));

    spawn_world_label(
        &mut commands,
        "SweepOptions::with_twist + with_caps",
        right_offset + Vec3::new(0.0, 1.8, -1.6),
    );
}

#[allow(clippy::needless_pass_by_value)]
fn drive_marker(
    time: Res<Time>,
    track: Res<TrackSpline>,
    mut query: Query<&mut Transform, With<TrackRunner>>,
) {
    let dist = (time.elapsed_secs() * 2.8) % track.baked.length().max(0.01);
    let frame = track.baked.sample_at_distance(dist);
    for mut tf in &mut query {
        tf.translation = track.offset + frame.position + Vec3::Y * 0.34;
    }
}

#[allow(clippy::needless_pass_by_value)]
fn rotate_twists(time: Res<Time>, mut query: Query<&mut Transform, With<SpinTwist>>) {
    let t = time.elapsed_secs();
    for mut tf in &mut query {
        tf.rotation = Quat::from_euler(EulerRot::YXZ, t * 0.5, 0.35, 0.0);
    }
}
