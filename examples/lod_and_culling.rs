//! Distance-based LOD, Bayer dither cross-fade, frustum culling, and live `InstancedStats`.
//!
//! A field of batched spheres and capsules stretches deep into the scene while the camera
//! glides forward, backward, and pans sideways so instances continuously cross `LodConfig`
//! thresholds and move in and out of the camera frustum. The HUD displays live telemetry from
//! `Res<InstancedStats>`: visible vs culled instances, draw calls, triangles, LOD distribution,
//! and active dither transitions.
//!
//! Run with: `cargo run --example lod_and_culling`

mod common;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{
    BatchRecorder, GradMeshMaterial, GradMeshPlugin, InstancedStats, LodConfig, ShapeDetail, opaque,
};
use common::screenshot::AutoScreenshotPlugin;

const COLS: i16 = 11;
const ROWS: i16 = 16;
const SPACING_X: f32 = 2.6;
const SPACING_Z: f32 = 3.0;

/// Run this example. Called by the WASM runner.
pub fn run(app: &mut App) {
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "BevyGradmesh LOD & Frustum Culling".into(),
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
        AutoScreenshotPlugin::new("screenshot_lod_and_culling.png")
            .animation(4.0)
            .with_title("LOD & Frustum Culling")
            .with_description("Distance-based LOD cross-fade, frustum culling, and InstancedStats")
            .wait_frames(20),
    )
    .insert_resource(ClearColor(Color::srgb(0.06, 0.08, 0.12)))
    .insert_resource(LodConfig {
        thresholds: [7.0, 15.0, 26.0, 40.0],
        transition_width: 3.0,
        hysteresis: 0.2,
    })
    .insert_resource(FieldRecorder(BatchRecorder::new()))
    .add_systems(Startup, setup)
    .add_systems(Update, (animate_camera, update_stats_hud))
    .add_systems(PostUpdate, draw_field);
}

fn main() {
    let mut app = App::new();
    run(&mut app);
    app.run();
}

#[derive(Resource)]
struct FieldRecorder(BatchRecorder);

#[derive(Component)]
struct RoamingCamera;

#[derive(Component)]
struct StatsHud;

#[derive(Component)]
struct BatchEntity;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 5.5, 10.0).looking_at(Vec3::new(0.0, 0.5, -18.0), Vec3::Y),
        RoamingCamera,
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 14_000.0,
            ..default()
        },
        Transform::from_xyz(8.0, 14.0, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.55, 0.6, 0.72),
        brightness: 650.0,
        affects_lightmapped_meshes: false,
    });

    // Ground plane extending down the field.
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::new(Vec3::Y, Vec2::new(26.0, 34.0)))),
        MeshMaterial3d(materials.add(opaque(0.09, 0.11, 0.15))),
        Transform::from_xyz(0.0, -0.65, -20.0),
    ));

    commands.spawn((
        Text::new("LOD & Frustum Culling"),
        TextFont {
            font_size: 20.0.into(),
            ..default()
        },
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            top: px(12),
            left: px(16),
            ..default()
        },
    ));

    commands.spawn((
        Text::new("Collecting InstancedStats..."),
        TextFont {
            font_size: 14.0.into(),
            ..default()
        },
        TextColor(Color::srgb(0.82, 0.9, 1.0)),
        Node {
            position_type: PositionType::Absolute,
            top: px(42),
            left: px(16),
            ..default()
        },
        StatsHud,
    ));
}

#[allow(clippy::needless_pass_by_value)]
fn animate_camera(time: Res<Time>, mut query: Query<&mut Transform, With<RoamingCamera>>) {
    let t = time.elapsed_secs();
    let z = 11.0 + (t * 0.55).sin() * 14.0;
    let x = (t * 0.35).sin() * 7.5;
    let look_x = (t * 0.35).sin() * 4.0;
    for mut tf in &mut query {
        *tf = Transform::from_xyz(x, 5.2, z).looking_at(Vec3::new(look_x, 0.4, z - 22.0), Vec3::Y);
    }
}

#[allow(clippy::needless_pass_by_value)]
fn draw_field(
    time: Res<Time>,
    mut commands: Commands,
    meshes: ResMut<Assets<Mesh>>,
    mut recorder: ResMut<FieldRecorder>,
    old_batches: Query<Entity, With<BatchEntity>>,
) {
    for entity in &old_batches {
        commands.entity(entity).despawn();
    }

    let t = time.elapsed_secs();
    recorder.0.begin();

    let half_cols = f32::from(COLS - 1) * 0.5;
    for row in 0..ROWS {
        for col in 0..COLS {
            let x = (f32::from(col) - half_cols) * SPACING_X;
            let z = 2.0 - f32::from(row) * SPACING_Z;
            let wave = (t * 1.4 + x * 0.35 + z * 0.2).sin() * 0.3;
            let y = 0.55 + wave;

            let row_frac = f32::from(row) / f32::from(ROWS - 1);
            let col_frac = f32::from(col) / f32::from(COLS - 1);
            let color = [
                0.22 + 0.72 * (1.0 - row_frac),
                0.35 + 0.55 * col_frac,
                0.3 + 0.68 * row_frac,
                1.0,
            ];

            recorder.0.set_position(Vec3::new(x, y, z));
            if (row + col) % 2 == 0 {
                recorder.0.sphere(0.62, ShapeDetail::High, color);
            } else {
                recorder.0.capsule(0.4, 0.75, ShapeDetail::High, color);
            }
        }
    }

    let _ = recorder.0.end();
    for entity in recorder.0.render_batches(&mut commands, meshes, None) {
        commands.entity(entity).insert(BatchEntity);
    }
}

#[allow(clippy::needless_pass_by_value)]
fn update_stats_hud(stats: Res<InstancedStats>, mut query: Query<&mut Text, With<StatsHud>>) {
    let dist = stats.lod_distribution;
    let text = format!(
        "Visible: {} / {}  (Culled: {})  |  Draw calls: {}  |  Tris: {}\n\
         LOD [Min..Ext]: [{}, {}, {}, {}, {}]  |  Cross-fading: {}",
        stats.total_instances,
        stats.total_instances_before_cull,
        stats.culled_instances,
        stats.draw_calls,
        stats.visible_triangles,
        dist[0],
        dist[1],
        dist[2],
        dist[3],
        dist[4],
        stats.lod_transitioning_instances,
    );
    for mut hud in &mut query {
        (**hud).clone_from(&text);
    }
}
