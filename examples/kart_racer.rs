//! Kart Racer — playable demo showcasing Bézier curves, LOD, PBR, and deformation.
//!
//! Controls: W/↑ Accel, S/↓ Brake, A/← Steer left, D/→ Steer right, R Reset

#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod common;

use std::sync::LazyLock;

use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy::render::RenderPlugin;
use bevy_asset::RenderAssetUsages;
use bevy_gradmesh::{
    additive, dual_directional_oklab, dual_radial_oklab,
    opaque, Shape, ShapeDetail,
    GradMeshMaterial, GradMeshPlugin, transparent,
    BatchRecorder, LodConfig, DeformationParams,
};
use common::screenshot::AutoScreenshotPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "BevyGradmesh Kart Racer".into(),
                resolution: (1920, 1080).into(),
                ..default()
            }),
            ..default()
        }).set(RenderPlugin {
            synchronous_pipeline_compilation: true,
            ..default()
        }))
        .add_plugins(GradMeshPlugin)
        .add_plugins(
            AutoScreenshotPlugin::new("screenshot_kart_racer.png")
                .with_title("Kart Racer")
                .with_description("Bézier track, PBR kart, collectibles, HUD")
                .wait_frames(5)
        )
        .insert_resource(ClearColor(Color::srgb(0.1, 0.15, 0.3)))
        .insert_resource(LapTimer::default())
        .insert_resource(KartState::default())
        .insert_resource(DecorationsRecorder(BatchRecorder::new()))
        .add_systems(Startup, setup)
        .add_systems(Update, (
            handle_input,
            update_physics,
            update_kart_transform,
            (update_camera, update_hud_follow, update_speed_hud).chain(),
            check_checkpoints,
            update_hud_text,
            (check_coin_collection, animate_coins).chain(),
        ))
        .add_systems(PostUpdate, draw_decorations)
        .run();
}

// ─── Components ───

#[derive(Component)]
struct Kart;
#[derive(Component)]
struct TrackBarrier;
#[derive(Component)]
struct TrackDecoration;
#[derive(Component)]
struct Coin {
    id: u32,
    collected: bool,
}
#[derive(Component)]
struct SpeedBarFill;
#[derive(Component)]
struct DecorationMarker;

// ─── Resources ───

#[derive(Resource, Default)]
struct KartState {
    position: Vec3,
    velocity: Vec3,
    heading: f32,
    speed: f32,
    coins: u32,
    boost: f32,
    track_t: f32,
}

#[derive(Resource, Default)]
struct LapTimer {
    current_lap: u32,
    elapsed: f32,
    started: bool,
    race_start: f32,
}

#[derive(Resource)]
struct DecorationsRecorder(BatchRecorder);

// ─── Track Definition ───

/// Closed Catmull-Rom spline control points for the race track.
/// First 3 points repeated at end for C1 continuity at the seam.
static TRACK_CP: LazyLock<Vec<Vec3>> = LazyLock::new(|| {
    vec![
        Vec3::new(0.0, 0.5, 100.0),
        Vec3::new(80.0, 0.8, 100.0),
        Vec3::new(160.0, 1.2, 100.0),
        Vec3::new(240.0, 1.5, 80.0),
        Vec3::new(280.0, 1.8, 20.0),
        Vec3::new(260.0, 1.5, -60.0),
        Vec3::new(200.0, 0.8, -100.0),
        Vec3::new(120.0, 0.3, -80.0),
        Vec3::new(80.0, -0.2, -120.0),
        Vec3::new(40.0, 0.2, -180.0),
        Vec3::new(-20.0, 0.8, -200.0),
        Vec3::new(-100.0, 1.3, -180.0),
        Vec3::new(-160.0, 1.6, -220.0),
        Vec3::new(-200.0, 1.0, -160.0),
        Vec3::new(-240.0, 0.3, -200.0),
        Vec3::new(-280.0, -0.3, -140.0),
        Vec3::new(-300.0, -0.5, -60.0),
        Vec3::new(-280.0, -0.2, 20.0),
        Vec3::new(-240.0, 0.3, 80.0),
        Vec3::new(-180.0, 0.8, 120.0),
        Vec3::new(-100.0, 1.0, 100.0),
        Vec3::new(-40.0, 0.8, 140.0),
        Vec3::new(20.0, 0.6, 120.0),
        // Close loop
        Vec3::new(0.0, 0.5, 100.0),
        Vec3::new(80.0, 0.8, 100.0),
        Vec3::new(160.0, 1.2, 100.0),
    ]
});

const NUM_UNIQUE_CPS: usize = 23;

/// Point on the track curve at parameter t ∈ [0, 1).
fn track_point(t: f32) -> Vec3 {
    let t = t.rem_euclid(1.0);
    let cps = &*TRACK_CP;
    let n = NUM_UNIQUE_CPS;
    let t_scaled = t * n as f32;
    let seg = t_scaled.floor() as usize;
    let lt = t_scaled - seg as f32;
    let prev_idx = if seg > 0 { seg - 1 } else { n - 1 };
    let p0 = cps[prev_idx];
    let p1 = cps[seg];
    let p2 = cps[seg + 1];
    let p3 = cps[seg + 2];
    // Catmull-Rom → cubic Bézier
    let b0 = p1;
    let b1 = p1 + (p2 - p0) / 3.0;
    let b2 = p2 - (p3 - p1) / 3.0;
    let b3 = p2;
    let u = 1.0 - lt;
    b0 * u * u * u + b1 * 3.0 * u * u * lt + b2 * 3.0 * u * lt * lt + b3 * lt * lt * lt
}

/// Horizontal heading direction at parameter t.
fn track_heading(t: f32) -> Vec3 {
    let dt = 0.001;
    let p1 = track_point(t);
    let p2 = track_point((t + dt).rem_euclid(1.0));
    let dir = (p2.xz() - p1.xz()).normalize();
    Vec3::new(dir.x, 0.0, dir.y)
}

/// Right vector perpendicular to heading (horizontal).
fn track_right(t: f32) -> Vec3 {
    let h = track_heading(t);
    Vec3::new(-h.z, 0.0, h.x).normalize()
}

// ─── Checkpoints ───

const CHECKPOINT_COUNT: usize = 8;

#[derive(Resource, Default)]
struct CheckpointState {
    passed: Vec<bool>,
    progress: f32,
}

// ─── Setup ───

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
) {
    let start_pos = track_point(0.0);
    let start_heading = track_heading(0.0).angle_between(Vec3::Z);

    // 3D Camera
    let cam_offset = Vec3::new(start_heading.sin() * -20.0, 12.0, start_heading.cos() * -20.0);
    commands.spawn((
        Camera3d::default(),
        IsDefaultUiCamera,
        Transform::from_translation(start_pos + cam_offset)
            .looking_at(start_pos + Vec3::Y * 0.5, Vec3::Y),
        Name::new("Camera3d"),
    ));

    // 2D Camera for HUD text overlay
    commands.spawn((
        Camera2d::default(),
        Camera {
            order: 1,
            ..default()
        },
        Name::new("Camera2d"),
    ));


    // Lighting
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(1.0, 0.92, 0.78),
            illuminance: 15_000.0,
            ..default()
        },
        Transform::from_xyz(200.0, 300.0, 100.0)
            .looking_at(Vec3::new(-100.0, 0.0, -50.0), Vec3::Y),
        Name::new("Sun"),
    ));
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(0.25, 0.35, 0.7),
            illuminance: 3_000.0,
            ..default()
        },
        Transform::from_xyz(-200.0, 200.0, -100.0)
            .looking_at(Vec3::new(-100.0, 0.0, -50.0), Vec3::Y),
        Name::new("Fill Light"),
    ));
    commands.spawn(AmbientLight {
        color: Color::srgb(0.2, 0.25, 0.4),
        brightness: 400.0,
        affects_lightmapped_meshes: false,
    });

    // Ground
    let ground = meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(1000.0)));
    commands.spawn((
        Mesh3d(ground),
        MeshMaterial3d(materials.add(dual_directional_oklab(
            [0.08, 0.15, 0.06], [0.12, 0.22, 0.10], [0.2, 0.6, 0.15],
        ))),
        Transform::from_translation(Vec3::new(-120.0, -0.05, -50.0)),
        Name::new("Ground"),
    ));

    // Track, kart, coins, decorations
    build_track(&mut commands, &mut meshes, &mut materials);
    build_kart(&mut commands, &mut meshes, &mut materials);
    commands.insert_resource(LodConfig::default());
    build_coins(&mut commands, &mut meshes, &mut materials);
    commands.insert_resource(CheckpointState {
        passed: vec![false; CHECKPOINT_COUNT],
        progress: 0.0,
    });

    // HUD: 3D billboard follows camera for mesh elements
    let hud_root = commands.spawn((
        Transform::default(),
        InheritedVisibility::default(),
        Name::new("HUD Root"),
    )).id();
    commands.entity(hud_root).insert(HudRootMarker);

    let hud_rot = Quat::from_rotation_x(-std::f32::consts::PI / 2.0);
    const HUD_Z_BACK: f32 = -7.0;
    const HUD_Z_MID: f32  = -6.5;
    const HUD_TOP_Y: f32 = 2.0;
    const HUD_BOTTOM_Y: f32 = -2.0;

    // Speed bar background
    let speed_bg = meshes.add(
        Shape::rectangle(Vec2::new(4.0, 0.35), ShapeDetail::Low).build()
    );
    commands.spawn((
        Mesh3d(speed_bg),
        MeshMaterial3d(materials.add(opaque(0.05, 0.05, 0.15))),
        Transform::from_xyz(0.0, HUD_BOTTOM_Y, HUD_Z_BACK).with_rotation(hud_rot),
        ChildOf(hud_root),
        Name::new("Speed Bar BG"),
    ));

    // Speed bar fill
    let speed_fill = meshes.add(
        Shape::rectangle(Vec2::new(3.9, 0.25), ShapeDetail::Low).build()
    );
    let speed_fill_mat = materials.add(dual_directional_oklab(
        [0.0, 0.5, 1.0], [0.0, 1.0, 0.5], [1.0, 0.0, 0.0],
    ));
    commands.spawn((
        Mesh3d(speed_fill),
        MeshMaterial3d(speed_fill_mat),
        Transform::from_xyz(-1.95, HUD_BOTTOM_Y, HUD_Z_MID)
            .with_rotation(hud_rot)
            .with_scale(Vec3::new(0.001, 1.0, 1.0)),
        ChildOf(hud_root),
        SpeedBarFill,
        Name::new("Speed Bar Fill"),
    ));

    // Coin icon — star shape (top-right)
    let coin_star = meshes.add(Shape::star(5, 0.35, 0.14, ShapeDetail::Medium).build());
    let coin_star_mat = materials.add(dual_radial_oklab(
        [1.0, 0.85, 0.2, 1.0], [0.8, 0.5, 0.1, 1.0], 0.7,
    ));
    commands.spawn((
        Mesh3d(coin_star),
        MeshMaterial3d(coin_star_mat),
        Transform::from_xyz(2.3, HUD_TOP_Y, HUD_Z_MID)
            .with_rotation(hud_rot)
            .with_scale(Vec3::splat(1.0)),
        ChildOf(hud_root),
        Name::new("Coin Icon"),
    ));

    // Lap indicator — hexagon (top-left)
    let lap_hex = meshes.add(Shape::polygon(6, 0.3, ShapeDetail::Low).build());
    let lap_hex_mat = materials.add(dual_directional_oklab(
        [0.1, 0.4, 0.8], [0.3, 0.8, 1.0], [0.0, 1.0, 0.0],
    ));
    commands.spawn((
        Mesh3d(lap_hex),
        MeshMaterial3d(lap_hex_mat),
        Transform::from_xyz(-3.0, HUD_TOP_Y, HUD_Z_MID)
            .with_rotation(hud_rot)
            .with_scale(Vec3::splat(1.0)),
        ChildOf(hud_root),
        Name::new("Lap Indicator"),
    ));

    // 2D Camera overlay for HUD text
    commands.spawn((
        Camera2d,
        Camera {
            order: 1,
            clear_color: ClearColorConfig::None,
            ..default()
        },
        Name::new("Camera2d HUD"),
    ));

    // HUD text elements (rendered by Camera2d overlay)
    // Speed label
    commands.spawn((
        Text2d("SPEED".into()),
        TextColor(Color::srgb(0.0, 0.9, 1.0)),
        TextFont {
            font_size: FontSize::Px(24.0),
            ..default()
        },
        Transform::from_xyz(0.0, 440.0, 0.0),
        Name::new("Speed Label"),
    ));

    // Coin count
    commands.spawn((
        Text2d("0/24".into()),
        TextColor(Color::srgb(1.0, 0.85, 0.2)),
        TextFont {
            font_size: FontSize::Px(28.0),
            ..default()
        },
        Transform::from_xyz(1700.0, 40.0, 0.0),
        Name::new("Coin Count"),
    ));

    // Lap/timer text
    commands.spawn((
        Text2d("Lap 0 / 3  Ready!".into()),
        TextColor(Color::srgb(0.3, 0.8, 1.0)),
        TextFont {
            font_size: FontSize::Px(28.0),
            ..default()
        },
        Transform::from_xyz(40.0, 40.0, 0.0),
        Name::new("Timer Text"),
    ));

    // Coordinates text
    commands.spawn((
        Text2d("Pos: 0.0, 0.0, 0.0".into()),
        TextColor(Color::srgba(0.6, 0.6, 0.7, 0.9)),
        TextFont {
            font_size: FontSize::Px(18.0),
            ..default()
        },
        Transform::from_xyz(40.0, -480.0, 0.0),
        Name::new("Coordinates Text"),
    ));

    // Controls text
    commands.spawn((
        Text2d("W/↑ Accel  S/↓ Brake  A/D Steer  R Reset".into()),
        TextColor(Color::srgba(1.0, 1.0, 1.0, 0.5)),
        TextFont {
            font_size: FontSize::Px(18.0),
            ..default()
        },
        Transform::from_xyz(40.0, -510.0, 0.0),
        Name::new("Controls Text"),
    ));
}

// ─── Track Building ───

fn build_track(
    commands: &mut Commands,
    meshes: &mut Assets<bevy::render::mesh::Mesh>,
    materials: &mut Assets<GradMeshMaterial>,
) {
    let road_width = 16.0;

    // Road surface
    let track_mesh = meshes.add(build_flat_road_mesh(road_width / 2.0, 800));
    let road_mat = materials.add({
        let mut mat = dual_directional_oklab(
            [0.45, 0.45, 0.47], [0.55, 0.55, 0.57], [0.0, 1.0, 0.0],
        );
        mat.base.metallic = 0.0;
        mat.base.perceptual_roughness = 0.85;
        mat.base.cull_mode = None;
        mat
    });
    commands.spawn((
        Mesh3d(track_mesh),
        MeshMaterial3d(road_mat),
        Transform::default(),
        Name::new("Track Road"),
    ));

    // Curbs
    let curb_count = 60;
    let curb_mesh = meshes.add(
        Shape::cuboid(Vec3::new(1.5, 0.12, 0.3), 0.02, ShapeDetail::Low).build()
    );
    let curb_red = materials.add(dual_directional_oklab(
        [0.7, 0.05, 0.05], [1.0, 0.2, 0.2], [0.0, 1.0, 0.0],
    ));
    let curb_white = materials.add(dual_directional_oklab(
        [0.8, 0.8, 0.8], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0],
    ));

    for i in 0..curb_count {
        let t = i as f32 / curb_count as f32;
        let center = track_point(t);
        let right = track_right(t);
        let rot = Quat::from_rotation_y(track_heading(t).angle_between(Vec3::Z));
        let mat = if i % 2 == 0 { &curb_red } else { &curb_white };
        let offset = road_width / 2.0 + 0.15;

        for side in [-1, 1] {
            let pos = center + right * offset * side as f32 + Vec3::Y * 0.06;
            commands.spawn((
                Mesh3d(curb_mesh.clone()),
                MeshMaterial3d(mat.clone()),
                Transform::from_translation(pos).with_rotation(rot),
                TrackBarrier,
                Name::new(format!("Curb {}{}", i, if side > 0 { "R" } else { "L" })),
            ));
        }
    }

    // Guardrail posts
    let post_mesh = meshes.add(
        Shape::tapered_cylinder(0.08, 0.15, 2.0, ShapeDetail::Low).build()
    );
    let post_mat = materials.add({
        let mut mat = dual_directional_oklab(
            [0.4, 0.4, 0.45], [0.7, 0.7, 0.75], [0.0, 1.0, 0.0],
        );
        mat.base.metallic = 0.5;
        mat.base.perceptual_roughness = 0.3;
        mat
    });

    for i in 0..30 {
        let t = i as f32 / 30.0;
        let center = track_point(t);
        let right = track_right(t);
        let offset = road_width / 2.0 + 1.0;
        for side in [-1, 1] {
            let pos = center + right * offset * side as f32 + Vec3::Y * 1.0;
            commands.spawn((
                Mesh3d(post_mesh.clone()),
                MeshMaterial3d(post_mat.clone()),
                Transform::from_translation(pos),
                TrackBarrier,
                Name::new(format!("Post {}{}", i, if side > 0 { "R" } else { "L" })),
            ));
        }
    }

    // Checkpoint arches
    let arch_pillar = meshes.add(
        Shape::tapered_cylinder(0.15, 0.25, 4.0, ShapeDetail::Medium).build()
    );
    let arch_beam = meshes.add(
        Shape::cuboid(Vec3::new(road_width + 2.0, 0.2, 0.3), 0.03, ShapeDetail::Low).build()
    );
    let arch_mat = materials.add({
        let mut mat = dual_directional_oklab(
            [0.1, 0.5, 0.15], [0.3, 1.0, 0.4], [0.0, 1.0, 0.0],
        );
        mat.base.metallic = 0.3;
        mat.base.perceptual_roughness = 0.4;
        mat
    });
    let arch_top_mat = materials.add({
        let mut mat = additive(0.2, 1.0, 0.3);
        mat.base.emissive = bevy::color::LinearRgba::rgb(0.2, 1.0, 0.3) * 20_000.0;
        mat
    });

    for i in 0..CHECKPOINT_COUNT {
        let t = i as f32 / CHECKPOINT_COUNT as f32;
        let center = track_point(t);
        let right = track_right(t);
        let rot = Quat::from_rotation_y(track_heading(t).angle_between(Vec3::Z));
        let pillar_offset = road_width / 2.0 + 0.5;

        for side in [-1, 1] {
            commands.spawn((
                Mesh3d(arch_pillar.clone()),
                MeshMaterial3d(arch_mat.clone()),
                Transform::from_translation(center + right * pillar_offset * side as f32 + Vec3::Y * 2.0),
                Name::new(format!("Checkpoint {} Pillar", i)),
            ));
        }
        commands.spawn((
            Mesh3d(arch_beam.clone()),
            MeshMaterial3d(arch_mat.clone()),
            Transform::from_translation(center + Vec3::Y * 4.0).with_rotation(rot),
            Name::new(format!("Checkpoint {} Beam", i)),
        ));
        let glow = meshes.add(Shape::sphere(0.3, ShapeDetail::Low).build());
        commands.spawn((
            Mesh3d(glow),
            MeshMaterial3d(arch_top_mat.clone()),
            Transform::from_translation(center + Vec3::Y * 4.5),
            Name::new(format!("Checkpoint {} Glow", i)),
        ));
    }

    // Light posts
    let light_post = meshes.add(
        Shape::tapered_cylinder(0.06, 0.12, 4.0, ShapeDetail::Low).build()
    );
    let light_post_mat = materials.add({
        let mut mat = dual_directional_oklab(
            [0.3, 0.3, 0.35], [0.6, 0.6, 0.65], [0.0, 1.0, 0.0],
        );
        mat.base.metallic = 0.7;
        mat.base.perceptual_roughness = 0.2;
        mat
    });
    let light_glow = meshes.add(Shape::sphere(0.25, ShapeDetail::Low).build());
    let light_glow_mat = materials.add({
        let mut mat = additive(1.0, 0.9, 0.5);
        mat.base.emissive = bevy::color::LinearRgba::rgb(1.0, 0.9, 0.5) * 40_000.0;
        mat
    });

    for i in 0..20 {
        let t = i as f32 / 20.0;
        let center = track_point(t);
        let right = track_right(t);
        let side = if i % 2 == 0 { 1.0 } else { -1.0 };
        let offset = road_width / 2.0 + 2.5;
        let base = center + right * offset * side;
        commands.spawn((
            Mesh3d(light_post.clone()),
            MeshMaterial3d(light_post_mat.clone()),
            Transform::from_translation(base + Vec3::Y * 2.0),
            TrackDecoration,
            Name::new(format!("Light post {}", i)),
        ));
        commands.spawn((
            Mesh3d(light_glow.clone()),
            MeshMaterial3d(light_glow_mat.clone()),
            Transform::from_translation(base + Vec3::Y * 4.1),
            TrackDecoration,
            Name::new(format!("Light glow {}", i)),
        ));
    }
}

/// Builds a flat road ribbon mesh along the track curve (closed loop).
fn build_flat_road_mesh(half_width: f32, samples: u32) -> bevy::render::mesh::Mesh {
    let n = (samples as usize).max(2);
    let up = Vec3::Y;

    let mut positions = Vec::with_capacity(n * 2);
    let mut normals = Vec::with_capacity(n * 2);
    let mut uvs = Vec::with_capacity(n * 2);
    let mut indices: Vec<u32> = Vec::with_capacity(n * 6);

    for i in 0..n {
        let t = i as f32 / n as f32;
        let center = track_point(t);
        let right = track_right(t);
        let uv_v = i as f32 / n as f32;

        positions.push(center + right * half_width);
        normals.push(up);
        uvs.push(bevy::math::Vec2::new(1.0, uv_v));
        positions.push(center - right * half_width);
        normals.push(up);
        uvs.push(bevy::math::Vec2::new(0.0, uv_v));
    }

    // Closed loop: last segment wraps to first
    // Each road segment forms a quad with 4 vertices:
    //   a = right edge of segment i
    //   b = left edge of segment i
    //   c = right edge of segment next
    //   d = left edge of segment next
    // Triangulate as (a,b,c) and (b,d,c) for CCW winding (viewed from +Y).
    for i in 0..n {
        let next = (i + 1) % n;
        let a = (i * 2) as u32;
        let b = (i * 2 + 1) as u32;
        let c = (next * 2) as u32;
        let d = (next * 2 + 1) as u32;
        indices.extend_from_slice(&[a, b, c, b, d, c]);
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

// ─── Kart Building ───

fn build_kart(
    commands: &mut Commands,
    meshes: &mut Assets<bevy::render::mesh::Mesh>,
    materials: &mut Assets<GradMeshMaterial>,
) {
    let start_pos = track_point(0.0);
    let start_yaw = track_heading(0.0).angle_between(Vec3::Z);

    let kart_root = commands.spawn((
        Transform::from_translation(start_pos)
            .with_rotation(Quat::from_rotation_y(start_yaw)),
        Kart,
        InheritedVisibility::default(),
        Name::new("Kart"),
    )).id();

    // Body
    let body = meshes.add(
        Shape::cuboid(Vec3::new(1.0, 0.4, 2.0), 0.12, ShapeDetail::Medium).build()
    );
    let body_mat = materials.add({
        let mut mat = dual_directional_oklab(
            [0.6, 0.05, 0.05], [1.0, 0.25, 0.2], [0.0, 1.0, 0.0],
        );
        mat.base.metallic = 0.8;
        mat.base.perceptual_roughness = 0.2;
        mat
    });
    commands.spawn((
        Mesh3d(body), MeshMaterial3d(body_mat),
        Transform::from_translation(Vec3::Y * 0.5),
        ChildOf(kart_root), Name::new("Body"),
    ));

    // Cabin
    let cabin = meshes.add(
        Shape::cuboid(Vec3::new(0.7, 0.35, 0.9), 0.15, ShapeDetail::Medium).build()
    );
    let cabin_mat = materials.add({
        let mut mat = dual_directional_oklab(
            [0.15, 0.05, 0.08], [0.35, 0.12, 0.15], [0.0, 1.0, 0.0],
        );
        mat.base.metallic = 0.6;
        mat.base.perceptual_roughness = 0.15;
        mat
    });
    commands.spawn((
        Mesh3d(cabin), MeshMaterial3d(cabin_mat),
        Transform::from_translation(Vec3::new(0.0, 0.88, -0.15)),
        ChildOf(kart_root), Name::new("Cabin"),
    ));

    // Windshield
    let windshield = meshes.add(
        Shape::cuboid(Vec3::new(0.65, 0.3, 0.05), 0.0, ShapeDetail::Low).build()
    );
    let windshield_mat = materials.add({
        let mut mat = transparent(0.4, 0.6, 0.8, 0.3);
        mat.base.metallic = 0.0;
        mat.base.perceptual_roughness = 0.05;
        mat
    });
    commands.spawn((
        Mesh3d(windshield), MeshMaterial3d(windshield_mat),
        Transform::from_translation(Vec3::new(0.0, 0.85, 0.32))
            .with_rotation(Quat::from_rotation_x(-0.3)),
        ChildOf(kart_root), Name::new("Windshield"),
    ));

    // Wheels
    let wheel = meshes.add(
        Shape::cylinder(0.22, 0.18, ShapeDetail::Medium).build()
    );
    let wheel_mat = materials.add({
        let mut mat = opaque(0.12, 0.12, 0.14);
        mat.base.perceptual_roughness = 0.8;
        mat
    });
    let rim = meshes.add(Shape::ring(0.18, 0.08, ShapeDetail::Medium).build());
    let rim_mat = materials.add({
        let mut mat = dual_directional_oklab(
            [0.5, 0.5, 0.55], [0.9, 0.9, 0.95], [0.0, 1.0, 0.0],
        );
        mat.base.metallic = 0.95;
        mat.base.perceptual_roughness = 0.1;
        mat
    });

    for (i, &pos) in [
        Vec3::new(0.6, 0.22, 0.7), Vec3::new(-0.6, 0.22, 0.7),
        Vec3::new(0.6, 0.22, -0.7), Vec3::new(-0.6, 0.22, -0.7),
    ].iter().enumerate() {
        let wheel_root = commands.spawn((
            Transform::from_translation(pos)
                .with_rotation(Quat::from_rotation_z(std::f32::consts::PI / 2.0)),
            ChildOf(kart_root),
            InheritedVisibility::default(),
            Name::new(format!("Wheel {}", i)),
        )).id();
        commands.entity(wheel_root).with_children(|p| {
            p.spawn((Mesh3d(wheel.clone()), MeshMaterial3d(wheel_mat.clone()), Transform::default()));
            for side in [-1.0, 1.0] {
                p.spawn((
                    Mesh3d(rim.clone()), MeshMaterial3d(rim_mat.clone()),
                    Transform::from_translation(Vec3::new(0.0, 0.0, side * 0.09)),
                ));
            }
        });
    }

    // Spoiler
    let spoiler = meshes.add(
        Shape::cuboid(Vec3::new(1.2, 0.06, 0.35), 0.02, ShapeDetail::Low).build()
    );
    let spoiler_mat = materials.add({
        let mut mat = dual_directional_oklab(
            [0.8, 0.25, 0.0], [1.0, 0.55, 0.1], [0.0, 1.0, 0.0],
        );
        mat.base.metallic = 0.7;
        mat.base.perceptual_roughness = 0.25;
        mat
    });
    commands.spawn((
        Mesh3d(spoiler), MeshMaterial3d(spoiler_mat.clone()),
        Transform::from_translation(Vec3::new(0.0, 0.75, -0.95)),
        ChildOf(kart_root), Name::new("Spoiler"),
    ));
    for side in [-0.4, 0.4] {
        let support = meshes.add(Shape::cylinder(0.03, 0.3, ShapeDetail::Low).build());
        commands.spawn((
            Mesh3d(support), MeshMaterial3d(spoiler_mat.clone()),
            Transform::from_translation(Vec3::new(side, 0.6, -0.95)),
            ChildOf(kart_root), Name::new("Spoiler Support"),
        ));
    }

    // Headlights
    let light = meshes.add(Shape::sphere(0.08, ShapeDetail::Low).build());
    let headlight_mat = materials.add({
        let mut mat = additive(1.0, 1.0, 0.4);
        mat.base.emissive = bevy::color::LinearRgba::rgb(1.0, 1.0, 0.4) * 50_000.0;
        mat
    });
    for side in [-0.3, 0.3] {
        commands.spawn((
            Mesh3d(light.clone()), MeshMaterial3d(headlight_mat.clone()),
            Transform::from_translation(Vec3::new(side, 0.25, 1.02)),
            ChildOf(kart_root), Name::new("Headlight"),
        ));
    }

    // Tail lights
    let tail_mat = materials.add({
        let mut mat = additive(0.9, 0.1, 0.1);
        mat.base.emissive = bevy::color::LinearRgba::rgb(0.9, 0.1, 0.1) * 30_000.0;
        mat
    });
    for side in [-0.35, 0.35] {
        commands.spawn((
            Mesh3d(light.clone()), MeshMaterial3d(tail_mat.clone()),
            Transform::from_translation(Vec3::new(side, 0.25, -1.02)),
            ChildOf(kart_root), Name::new("Tail Light"),
        ));
    }
}

// ─── Decorations ───

fn populate_decorations(recorder: &mut BatchRecorder, _time: f32) {
    recorder.begin();

    let trunk_light = [0.4, 0.25, 0.12];
    let leaf_colors = [
        ([0.05, 0.25, 0.05], [0.15, 0.55, 0.15]),
        ([0.02, 0.15, 0.04], [0.10, 0.35, 0.10]),
        ([0.10, 0.30, 0.05], [0.25, 0.60, 0.12]),
    ];
    let tree_positions = [
        Vec3::new(-250.0, 0.0, -250.0), Vec3::new(-280.0, 0.0, -230.0),
        Vec3::new(-230.0, 0.0, -270.0), Vec3::new(-100.0, 0.0, -260.0),
        Vec3::new(50.0, 0.0, -270.0), Vec3::new(200.0, 0.0, -250.0),
        Vec3::new(250.0, 0.0, -200.0), Vec3::new(320.0, 0.0, -50.0),
        Vec3::new(340.0, 0.0, 30.0), Vec3::new(250.0, 0.0, 180.0),
        Vec3::new(300.0, 0.0, 200.0), Vec3::new(100.0, 0.0, 200.0),
        Vec3::new(150.0, 0.0, 220.0), Vec3::new(-150.0, 0.0, 200.0),
        Vec3::new(-180.0, 0.0, 180.0), Vec3::new(-340.0, 0.0, -30.0),
        Vec3::new(-330.0, 0.0, 50.0), Vec3::new(100.0, 0.0, -150.0),
        Vec3::new(-80.0, 0.0, -130.0), Vec3::new(150.0, 0.0, 50.0),
        Vec3::new(-120.0, 0.0, 50.0), Vec3::new(-200.0, 0.0, -100.0),
        Vec3::new(80.0, 0.0, -200.0), Vec3::new(180.0, 0.0, 150.0),
    ];

    for (i, &pos) in tree_positions.iter().enumerate() {
        let scale = 0.7 + (i % 3) as f32 * 0.3;
        let (dark, light) = leaf_colors[i % leaf_colors.len()];
        let canopy_r = if i % 2 == 0 { 2.5 } else { 1.8 };

        recorder.set_position(pos + Vec3::Y * 2.0 * scale);
        recorder.set_scale(Vec3::splat(scale));
        recorder.set_rotation(Quat::IDENTITY);
        recorder.cone(0.3, 4.0, ShapeDetail::Low, [trunk_light[0], trunk_light[1], trunk_light[2], 1.0]);

        recorder.set_position(pos + Vec3::Y * 5.5 * scale);
        recorder.set_scale(Vec3::splat(scale));
        recorder.set_rotation(Quat::IDENTITY);
        recorder.set_deformation(DeformationParams::wobble(0.15, 1.0, i as f32 * 0.3));
        recorder.sphere_dual_directional(canopy_r, ShapeDetail::Medium, dark, light, [0.0, 1.0, 0.0], true);
    }

    let bush_colors = [
        ([0.03, 0.18, 0.04], [0.12, 0.40, 0.10]),
        ([0.04, 0.20, 0.06], [0.14, 0.45, 0.12]),
    ];
    let bush_positions = [
        Vec3::new(-220.0, 0.0, -220.0), Vec3::new(-80.0, 0.0, -240.0),
        Vec3::new(180.0, 0.0, -230.0), Vec3::new(300.0, 0.0, -30.0),
        Vec3::new(220.0, 0.0, 180.0), Vec3::new(80.0, 0.0, 220.0),
        Vec3::new(-120.0, 0.0, 160.0), Vec3::new(-260.0, 0.0, -10.0),
    ];

    for (i, &pos) in bush_positions.iter().enumerate() {
        let (dark, light) = bush_colors[i % bush_colors.len()];
        recorder.set_position(pos + Vec3::Y * 0.5);
        recorder.set_scale(Vec3::splat(1.0));
        recorder.set_rotation(Quat::IDENTITY);
        recorder.set_deformation(DeformationParams::pulse(0.1, 1.5, i as f32 * 0.5));
        recorder.sphere_dual_directional(0.8, ShapeDetail::Low, dark, light, [0.0, 1.0, 0.0], true);
    }

    recorder.end();
}

fn draw_decorations(
    time: Res<Time>,
    mut commands: Commands,
    meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
    mut recorder: ResMut<DecorationsRecorder>,
    query: Query<Entity, With<DecorationMarker>>,
) {
    for entity in query.iter() {
        commands.entity(entity).despawn();
    }
    populate_decorations(&mut recorder.0, time.elapsed_secs());
    let spawned = recorder.0.render_batches(&mut commands, meshes, None);
    for eid in spawned {
        commands.entity(eid).insert(DecorationMarker);
    }
}

// ─── Coins ───

fn build_coins(
    commands: &mut Commands,
    meshes: &mut Assets<bevy::render::mesh::Mesh>,
    materials: &mut Assets<GradMeshMaterial>,
) {
    let coin_mesh = meshes.add(
        Shape::cylinder(0.3, 0.08, ShapeDetail::Medium).build()
    );
    let coin_mat = materials.add({
        let mut mat = dual_radial_oklab(
            [1.0, 0.85, 0.2, 1.0], [0.8, 0.5, 0.1, 1.0], 0.7,
        );
        mat.base.metallic = 0.9;
        mat.base.perceptual_roughness = 0.1;
        mat.base.emissive = bevy::color::LinearRgba::rgb(1.0, 0.85, 0.2) * 5_000.0;
        mat
    });

    for i in 0..24 {
        let t = (i as f32 + 0.5) / 24.0;
        let center = track_point(t);
        let right = track_right(t);
        let side = if i % 2 == 0 { 1.0 } else { -1.0 };
        let pos = center + right * 2.0 * side + Vec3::Y * 0.8;

        commands.spawn((
            Mesh3d(coin_mesh.clone()),
            MeshMaterial3d(coin_mat.clone()),
            Transform::from_translation(pos)
                .with_rotation(Quat::from_rotation_x(std::f32::consts::PI / 2.0)),
            Coin { id: i as u32, collected: false },
            Name::new(format!("Coin {}", i)),
        ));
    }
}

// ─── Input ───

fn handle_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut state: ResMut<KartState>,
) {
    let dt = time.delta_secs();
    let mut throttle = 0.0f32;
    let mut steer = 0.0f32;

    if keyboard.pressed(KeyCode::ArrowUp) || keyboard.pressed(KeyCode::KeyW) {
        throttle = 1.0;
    }
    if keyboard.pressed(KeyCode::ArrowDown) || keyboard.pressed(KeyCode::KeyS) {
        throttle = -0.6;
    }
    if keyboard.pressed(KeyCode::ArrowLeft) || keyboard.pressed(KeyCode::KeyA) {
        steer = 1.0;
    }
    if keyboard.pressed(KeyCode::ArrowRight) || keyboard.pressed(KeyCode::KeyD) {
        steer = -1.0;
    }
    if keyboard.just_pressed(KeyCode::KeyR) {
        state.position = track_point(0.0);
        state.heading = track_heading(0.0).angle_between(Vec3::Z);
        state.speed = 0.0;
        state.boost = 0.0;
    }

    let speed_factor = (state.speed.abs() / 20.0).min(1.0);
    state.heading += steer * 2.5 * dt * (0.3 + 0.7 * (1.0 - speed_factor));

    let boost_mult = 1.0 + state.boost;
    if throttle > 0.0 {
        state.speed += throttle * 50.0 * dt * boost_mult;
    } else if throttle < 0.0 {
        state.speed += throttle * 60.0 * dt;
    }

    state.speed *= if throttle > 0.0 { 0.998 } else { 0.99 };
    state.speed = state.speed.clamp(-24.0, 80.0);
    state.boost *= 0.99;
    if state.boost < 0.01 { state.boost = 0.0; }

    let forward = Vec3::new(state.heading.sin(), 0.0, state.heading.cos());
    state.velocity = forward * state.speed;
}

// ─── Physics ───

fn update_physics(mut state: ResMut<KartState>, time: Res<Time>) {
    let dt = time.delta_secs().min(0.033);
    let forward = Vec3::new(state.heading.sin(), 0.0, state.heading.cos());
    let move_xz = forward.xz() * state.speed * dt;
    state.position.x += move_xz.x;
    state.position.z += move_xz.y;

    // Find nearest track point for elevation
    let mut min_dist_sq = f32::MAX;
    let mut best_t = state.track_t;
    let search_range = 0.10;
    for i in 0..31 {
        let t = (state.track_t + (i as f32 - 15.5) * search_range / 15.5).rem_euclid(1.0);
        let tp = track_point(t);
        let d = (state.position.xz() - tp.xz()).length_squared();
        if d < min_dist_sq { min_dist_sq = d; best_t = t; }
    }
    for i in 0..11 {
        let t = (best_t + (i as f32 - 5.5) * 0.01 / 5.5).rem_euclid(1.0);
        let tp = track_point(t);
        let d = (state.position.xz() - tp.xz()).length_squared();
        if d < min_dist_sq { min_dist_sq = d; best_t = t; }
    }

    state.position.y = track_point(best_t).y;
    state.track_t = best_t;

    // Off-track slowdown
    if min_dist_sq > 324.0 {
        let dist = min_dist_sq.sqrt();
        let off = (dist - 18.0).min(8.0) / 8.0;
        state.speed *= 1.0 - off * 0.05;
    }
}

// ─── Kart Transform ───

fn update_kart_transform(
    state: Res<KartState>,
    mut kart_query: Query<&mut Transform, With<Kart>>,
) {
    if let Ok(mut transform) = kart_query.single_mut() {
        transform.translation = state.position;
        transform.rotation = Quat::from_rotation_y(state.heading);
    }
}

// ─── Camera ───

#[derive(Component)]
struct HudRootMarker;

fn update_camera(
    state: Res<KartState>,
    time: Res<Time>,
    mut camera_query: Query<&mut Transform, With<Camera3d>>,
    mut smooth: Local<Vec3>,
) {
    if let Ok(mut cam) = camera_query.single_mut() {
        let offset = Vec3::new(
            state.heading.sin() * -20.0, 12.0, state.heading.cos() * -20.0,
        );
        let desired = state.position + offset;
        let lerp = (1.0 - 2.0_f32.powf(-4.0 * time.delta_secs())).min(1.0);
        cam.translation = cam.translation.lerp(desired, lerp);

        let forward = Vec3::new(state.heading.sin(), 0.0, state.heading.cos());
        let target = state.position + Vec3::Y * 0.5 + forward * 16.0;
        let look_lerp = (1.0 - 2.0_f32.powf(-6.0 * time.delta_secs())).min(1.0);
        *smooth = smooth.lerp(target, look_lerp);
        cam.look_at(*smooth, Vec3::Y);
    }
}

fn update_hud_follow(
    camera_query: Query<&Transform, (With<Camera3d>, Without<HudRootMarker>)>,
    mut hud_query: Query<&mut Transform, With<HudRootMarker>>,
) {
    if let Ok(cam) = camera_query.single() {
        if let Ok(mut hud) = hud_query.single_mut() {
            hud.translation = cam.translation;
            hud.rotation = cam.rotation;
        }
    }
}

// ─── Checkpoints ───

fn check_checkpoints(
    state: Res<KartState>,
    mut timer: ResMut<LapTimer>,
    mut cp: ResMut<CheckpointState>,
    time: Res<Time>,
) {
    let kart_pos = state.position;
    let check_dist_sq = 36.0;

    for i in 0..CHECKPOINT_COUNT {
        let t = i as f32 / CHECKPOINT_COUNT as f32;
        let cp_pos = track_point(t);
        if (kart_pos - cp_pos).length_squared() < check_dist_sq && !cp.passed[i] {
            let next = cp.progress as usize % CHECKPOINT_COUNT;
            if i == next {
                cp.passed[i] = true;
                cp.progress = (i as f32 + 1.0) / CHECKPOINT_COUNT as f32;
                if !timer.started {
                    timer.started = true;
                    timer.race_start = time.elapsed_secs();
                }
                timer.elapsed = time.elapsed_secs() - timer.race_start;
                if cp.passed.iter().all(|&p| p) {
                    timer.current_lap += 1;
                    cp.passed = vec![false; CHECKPOINT_COUNT];
                    cp.progress = 0.0;
                }
            }
        }
    }
}


// ─── Coin Collection ───

fn check_coin_collection(
    mut state: ResMut<KartState>,
    mut coin_query: Query<(&mut Transform, &mut Coin)>,
) {
    let kart_pos = state.position;
    for (mut transform, mut coin) in &mut coin_query {
        if coin.collected { continue; }
        if (kart_pos - transform.translation).length_squared() < 2.25 {
            coin.collected = true;
            transform.translation.y -= 100.0;
        }
    }
    state.coins = coin_query.iter().filter(|(_, c)| c.collected).count() as u32;
    if state.coins > 0 {
        state.boost = state.boost.max(0.3);
    }
}

// ─── Speed HUD (3D bar) ───

fn update_speed_hud(
    state: Res<KartState>,
    mut speed_bar: Query<&mut Transform, With<SpeedBarFill>>,
) {
    if let Ok(mut transform) = speed_bar.single_mut() {
        let ratio = (state.speed.abs() / 80.0).clamp(0.0, 1.0);
        transform.scale.x = ratio.max(0.001);
        transform.translation.x = -2.0 + 3.8 * transform.scale.x * 0.5;
    }
}

// ─── Animate Coins ───

fn animate_coins(
    time: Res<Time>,
    mut coin_query: Query<(&Coin, &mut Transform)>,
) {
    let t = time.elapsed_secs();
    for (coin, mut transform) in &mut coin_query {
        if coin.collected { continue; }
        transform.rotation = Quat::from_rotation_x(std::f32::consts::PI / 2.0)
            * Quat::from_rotation_y(t * 2.0 + coin.id as f32);
        transform.translation.y = 0.8 + (t * 1.5 + coin.id as f32 * 0.5).sin() * 0.15;
    }
}

// ─── HUD Text Updates (2D overlay) ───

fn update_hud_text(
    state: Res<KartState>,
    timer: Res<LapTimer>,
    mut text_query: Query<(&mut Text2d, &Name)>,
) {
    for (mut text, name) in &mut text_query {
        match name.as_str() {
            "Coin Count" => {
                *text = Text2d(format!("{}/24", state.coins).into());
            }
            "Timer Text" => {
                let lap = timer.current_lap.min(3);
                if timer.current_lap >= 3 {
                    *text = Text2d(format!("🏆 Race Complete!").into());
                } else {
                    let time_str = if timer.started {
                        format!("{:.1}s", timer.elapsed)
                    } else {
                        "Ready!".into()
                    };
                    *text = Text2d(format!("Lap {} / 3  {}", lap, time_str).into());
                }
            }
            "Coordinates Text" => {
                let p = state.position;
                *text = Text2d(format!("Pos: {:.1}, {:.1}, {:.1}", p.x, p.y, p.z).into());
            }
            _ => {}
        }
    }
}
