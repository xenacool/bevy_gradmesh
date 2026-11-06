//! Kart Racer — playable demo showcasing Bézier curves, LOD, PBR, and deformation.
//!
//! Controls: W/↑ Accel, S/↓ Brake, A/← Steer left, D/→ Steer right, R Reset

#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod common;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy_asset::RenderAssetUsages;
use bevy_gradmesh::{
    BakedSpline, BatchRecorder, CatmullRomSpline, DeformationParams, GradMeshMaterial, GradMeshPlugin, LodConfig, MixSpace, Shape,
    ShapeDetail, additive, dual_directional, dual_radial_alpha, opaque, transparent,
};
use common::screenshot::AutoScreenshotPlugin;

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "BevyGradmesh Kart Racer".into(),
                        resolution: (1920, 1080).into(),
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
            AutoScreenshotPlugin::new("screenshot_kart_racer.png")
                .with_title("Kart Racer")
                .with_description("Bézier track, PBR kart, collectibles, HUD")
                .wait_frames(5),
        )
        .insert_resource(ClearColor(Color::srgb(0.1, 0.15, 0.3)))
        .insert_resource(LapTimer::default())
        .insert_resource(KartState::default())
        .insert_resource(DecorationsRecorder(BatchRecorder::new()))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                handle_input,
                update_physics,
                update_kart_transform,
                (update_camera, update_hud_follow, update_speed_hud).chain(),
                check_checkpoints,
                update_hud_text,
                (check_coin_collection, animate_coins).chain(),
            ),
        )
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
    /// Resting position on the track centerline (bobbing is applied relative to this).
    base: Vec3,
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

// ─── Track ───

/// Tightest turn (world units) the road may take; the spline relaxes anything sharper.
const MIN_TURN_RADIUS: f32 = 14.0;

/// Spacing of the baked frames the road, barriers and physics are built from.
const TRACK_FRAME_STEP: f32 = 0.5;

/// Hand-placed control points of a closed loop; tight corners are straightened for us.
fn track_spline() -> CatmullRomSpline {
    CatmullRomSpline::closed(vec![
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
    ])
    .relaxed(MIN_TURN_RADIUS)
}

/// Roads are wider on straights and narrower in corners.
fn road_width(curvature: f32) -> f32 {
    match curvature.abs() {
        c if c < 1.0 / 150.0 => 16.0,
        c if c < 1.0 / 50.0 => 14.0,
        _ => 12.0,
    }
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
    let track = track_spline().bake(TRACK_FRAME_STEP);
    let start = track.sample(0.0);
    let start_pos = start.position;
    let start_heading = start.forward.x.atan2(start.forward.z);

    // 3D Camera
    let cam_offset = Vec3::new(
        start_heading.sin() * -20.0,
        12.0,
        start_heading.cos() * -20.0,
    );
    commands.spawn((
        Camera3d::default(),
        IsDefaultUiCamera,
        Transform::from_translation(start_pos + cam_offset)
            .looking_at(start_pos + Vec3::Y * 0.5, Vec3::Y),
        Name::new("Camera3d"),
    ));

    // Lighting
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(1.0, 0.92, 0.78),
            illuminance: 15_000.0,
            ..default()
        },
        Transform::from_xyz(200.0, 300.0, 100.0).looking_at(Vec3::new(-100.0, 0.0, -50.0), Vec3::Y),
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
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.2, 0.25, 0.4),
        brightness: 400.0,
        affects_lightmapped_meshes: false,
    });

    // Ground
    let ground = meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(1000.0)));
    commands.spawn((
        Mesh3d(ground),
        MeshMaterial3d(materials.add(dual_directional(
            [0.08, 0.15, 0.06],
            [0.12, 0.22, 0.10],
            [0.2, 0.6, 0.15],
        ))),
        Transform::from_translation(Vec3::new(-120.0, -1.0, -50.0)),
        Name::new("Ground"),
    ));

    let sampler = track.clone();
    commands.insert_resource(track);

    // Track, kart, coins, decorations
    build_track(&mut commands, &mut meshes, &mut materials, &sampler);
    build_kart(&mut commands, &mut meshes, &mut materials, &sampler);
    commands.insert_resource(LodConfig::default());
    build_coins(&mut commands, &mut meshes, &mut materials, &sampler);
    commands.insert_resource(CheckpointState {
        passed: vec![false; CHECKPOINT_COUNT],
        progress: 0.0,
    });

    // HUD: 3D billboard follows camera for mesh elements
    let hud_root = commands
        .spawn((
            Transform::default(),
            InheritedVisibility::default(),
            Name::new("HUD Root"),
        ))
        .id();
    commands.entity(hud_root).insert(HudRootMarker);

    let hud_rot = Quat::from_rotation_x(-std::f32::consts::PI / 2.0);
    const HUD_Z_BACK: f32 = -7.0;
    const HUD_Z_MID: f32 = -6.5;
    const HUD_TOP_Y: f32 = 2.0;
    const HUD_BOTTOM_Y: f32 = -2.0;

    // Speed bar background
    let speed_bg = meshes.add(Shape::rectangle(Vec2::new(4.0, 0.35), ShapeDetail::Low).build());
    commands.spawn((
        Mesh3d(speed_bg),
        MeshMaterial3d(materials.add(opaque(0.05, 0.05, 0.15))),
        Transform::from_xyz(0.0, HUD_BOTTOM_Y, HUD_Z_BACK).with_rotation(hud_rot),
        ChildOf(hud_root),
        Name::new("Speed Bar BG"),
    ));

    // Speed bar fill
    let speed_fill = meshes.add(Shape::rectangle(Vec2::new(3.9, 0.25), ShapeDetail::Low).build());
    let speed_fill_mat = materials.add(dual_directional(
        [0.0, 0.5, 1.0],
        [0.0, 1.0, 0.5],
        [1.0, 0.0, 0.0],
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
    let coin_star_mat = materials.add(dual_radial_alpha([1.0, 0.85, 0.2, 1.0], [0.8, 0.5, 0.1, 1.0]));
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
    let lap_hex_mat = materials.add(dual_directional(
        [0.1, 0.4, 0.8],
        [0.3, 0.8, 1.0],
        [0.0, 1.0, 0.0],
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
        Text2d(format!("0/{COIN_COUNT}").into()),
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
    sampler: &BakedSpline,
) {
    // Road surface
    let track_mesh = meshes.add(build_road_mesh_from_sampler(sampler));
    let road_mat = materials.add({
        let mut mat =
            dual_directional([0.45, 0.45, 0.47], [0.55, 0.55, 0.57], [0.0, 1.0, 0.0]);
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

    // Curbs — use sampler for position, right, and width
    let curb_count = 60;
    let curb_mesh =
        meshes.add(Shape::cuboid(Vec3::new(1.5, 0.12, 0.3), 0.02, ShapeDetail::Low).build());
    let curb_red = materials.add(dual_directional(
        [0.7, 0.05, 0.05],
        [1.0, 0.2, 0.2],
        [0.0, 1.0, 0.0],
    ));
    let curb_white = materials.add(dual_directional(
        [0.8, 0.8, 0.8],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
    ));

    for i in 0..curb_count {
        let t = i as f32 / curb_count as f32;
        let frame = sampler.sample(t);
        let (center, forward, right) = (frame.position, frame.forward, frame.right);
        let width = road_width(frame.curvature);
        let rot = Quat::from_rotation_arc(Vec3::Z, forward);
        let mat = if i % 2 == 0 { &curb_red } else { &curb_white };
        let offset = width / 2.0 + 0.15;

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

    // Guardrail posts — use sampler
    let post_mesh = meshes.add(Shape::tapered_cylinder(0.08, 0.15, 2.0, ShapeDetail::Low).build());
    let post_mat = materials.add({
        let mut mat = dual_directional([0.4, 0.4, 0.45], [0.7, 0.7, 0.75], [0.0, 1.0, 0.0]);
        mat.base.metallic = 0.5;
        mat.base.perceptual_roughness = 0.3;
        mat
    });

    for i in 0..30 {
        let t = i as f32 / 30.0;
        let frame = sampler.sample(t);
        let (center, _forward, right) = (frame.position, frame.forward, frame.right);
        let width = road_width(frame.curvature);
        let offset = width / 2.0 + 1.0;
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

    // Checkpoint arches — use sampler for variable width
    let arch_pillar =
        meshes.add(Shape::tapered_cylinder(0.15, 0.25, 4.0, ShapeDetail::Medium).build());
    let arch_mat = materials.add({
        let mut mat = dual_directional([0.1, 0.5, 0.15], [0.3, 1.0, 0.4], [0.0, 1.0, 0.0]);
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
        let frame = sampler.sample(t);
        let (center, forward, right) = (frame.position, frame.forward, frame.right);
        let width = road_width(frame.curvature);
        let rot = Quat::from_rotation_arc(Vec3::Z, forward);
        let pillar_offset = width / 2.0 + 0.5;

        for side in [-1, 1] {
            commands.spawn((
                Mesh3d(arch_pillar.clone()),
                MeshMaterial3d(arch_mat.clone()),
                Transform::from_translation(
                    center + right * pillar_offset * side as f32 + Vec3::Y * 2.0,
                ),
                Name::new(format!("Checkpoint {} Pillar", i)),
            ));
        }
        let arch_beam = meshes
            .add(Shape::cuboid(Vec3::new(width + 2.0, 0.2, 0.3), 0.03, ShapeDetail::Low).build());
        commands.spawn((
            Mesh3d(arch_beam),
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

    // Light posts — use sampler
    let light_post = meshes.add(Shape::tapered_cylinder(0.06, 0.12, 4.0, ShapeDetail::Low).build());
    let light_post_mat = materials.add({
        let mut mat = dual_directional([0.3, 0.3, 0.35], [0.6, 0.6, 0.65], [0.0, 1.0, 0.0]);
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
        let frame = sampler.sample(t);
        let (center, _forward, right) = (frame.position, frame.forward, frame.right);
        let width = road_width(frame.curvature);
        let side = if i % 2 == 0 { 1.0 } else { -1.0 };
        let offset = width / 2.0 + 2.5;
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

/// Builds a road ribbon mesh from the baked track, widening on straights.
fn build_road_mesh_from_sampler(sampler: &BakedSpline) -> bevy::render::mesh::Mesh {
    let frames = sampler.frames();
    let n = frames.len();

    let mut positions = Vec::with_capacity(n * 2);
    let mut normals = Vec::with_capacity(n * 2);
    let mut uvs = Vec::with_capacity(n * 2);
    let mut indices: Vec<u32> = Vec::with_capacity(n * 6);

    for i in 0..n {
        let f = frames[i];
        let half_w = road_width(f.curvature) / 2.0;

        let uv_v = i as f32 / n as f32;

        positions.push(f.position + f.right * half_w);
        normals.push(f.up);
        uvs.push(bevy::math::Vec2::new(1.0, uv_v));
        positions.push(f.position - f.right * half_w);
        normals.push(f.up);
        uvs.push(bevy::math::Vec2::new(0.0, uv_v));
    }

    // Closed loop: last segment wraps to first
    // Each road segment forms a quad with 4 vertices:
    //   a = right edge of segment i  (vertex 2*i)
    //   b = left edge of segment i   (vertex 2*i+1)
    //   c = right edge of segment next (vertex 2*next)
    //   d = left edge of segment next  (vertex 2*next+1)
    // Triangulate as (a,c,b) and (b,c,d) for CCW winding (viewed from +Y).
    for i in 0..n {
        let next = (i + 1) % n;
        let a = (i * 2) as u32;
        let b = (i * 2 + 1) as u32;
        let c = (next * 2) as u32;
        let d = (next * 2 + 1) as u32;
        indices.extend_from_slice(&[a, c, b, b, c, d]);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
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
    sampler: &BakedSpline,
) {
    let start = sampler.sample(0.0);
    let (start_pos, start_fwd) = (start.position, start.forward);
    let start_yaw = start_fwd.x.atan2(start_fwd.z);
    // Start the simulation on the track, facing along it.
    commands.insert_resource(KartState {
        position: start_pos,
        heading: start_yaw,
        ..default()
    });

    let kart_root = commands
        .spawn((
            Transform::from_translation(start_pos).with_rotation(Quat::from_rotation_y(start_yaw)),
            Kart,
            InheritedVisibility::default(),
            Name::new("Kart"),
        ))
        .id();

    // Body
    let body =
        meshes.add(Shape::cuboid(Vec3::new(1.0, 0.4, 2.0), 0.12, ShapeDetail::Medium).build());
    let body_mat = materials.add({
        let mut mat = dual_directional([0.6, 0.05, 0.05], [1.0, 0.25, 0.2], [0.0, 1.0, 0.0]);
        mat.base.metallic = 0.8;
        mat.base.perceptual_roughness = 0.2;
        mat
    });
    commands.spawn((
        Mesh3d(body),
        MeshMaterial3d(body_mat),
        Transform::from_translation(Vec3::Y * 0.5),
        ChildOf(kart_root),
        Name::new("Body"),
    ));

    // Cabin
    let cabin =
        meshes.add(Shape::cuboid(Vec3::new(0.7, 0.35, 0.9), 0.15, ShapeDetail::Medium).build());
    let cabin_mat = materials.add({
        let mut mat =
            dual_directional([0.15, 0.05, 0.08], [0.35, 0.12, 0.15], [0.0, 1.0, 0.0]);
        mat.base.metallic = 0.6;
        mat.base.perceptual_roughness = 0.15;
        mat
    });
    commands.spawn((
        Mesh3d(cabin),
        MeshMaterial3d(cabin_mat),
        Transform::from_translation(Vec3::new(0.0, 0.88, -0.15)),
        ChildOf(kart_root),
        Name::new("Cabin"),
    ));

    // Windshield
    let windshield =
        meshes.add(Shape::cuboid(Vec3::new(0.65, 0.3, 0.05), 0.0, ShapeDetail::Low).build());
    let windshield_mat = materials.add({
        let mut mat = transparent(0.4, 0.6, 0.8, 0.3);
        mat.base.metallic = 0.0;
        mat.base.perceptual_roughness = 0.05;
        mat
    });
    commands.spawn((
        Mesh3d(windshield),
        MeshMaterial3d(windshield_mat),
        Transform::from_translation(Vec3::new(0.0, 0.85, 0.32))
            .with_rotation(Quat::from_rotation_x(-0.3)),
        ChildOf(kart_root),
        Name::new("Windshield"),
    ));

    // Wheels
    let wheel = meshes.add(Shape::cylinder(0.22, 0.18, ShapeDetail::Medium).build());
    let wheel_mat = materials.add({
        let mut mat = opaque(0.12, 0.12, 0.14);
        mat.base.perceptual_roughness = 0.8;
        mat
    });
    let rim = meshes.add(Shape::ring(0.18, 0.08, ShapeDetail::Medium).build());
    let rim_mat = materials.add({
        let mut mat = dual_directional([0.5, 0.5, 0.55], [0.9, 0.9, 0.95], [0.0, 1.0, 0.0]);
        mat.base.metallic = 0.95;
        mat.base.perceptual_roughness = 0.1;
        mat
    });

    for (i, &pos) in [
        Vec3::new(0.6, 0.22, 0.7),
        Vec3::new(-0.6, 0.22, 0.7),
        Vec3::new(0.6, 0.22, -0.7),
        Vec3::new(-0.6, 0.22, -0.7),
    ]
    .iter()
    .enumerate()
    {
        let wheel_root = commands
            .spawn((
                Transform::from_translation(pos)
                    .with_rotation(Quat::from_rotation_z(std::f32::consts::PI / 2.0)),
                ChildOf(kart_root),
                InheritedVisibility::default(),
                Name::new(format!("Wheel {}", i)),
            ))
            .id();
        commands.entity(wheel_root).with_children(|p| {
            p.spawn((
                Mesh3d(wheel.clone()),
                MeshMaterial3d(wheel_mat.clone()),
                Transform::default(),
            ));
            for side in [-1.0, 1.0] {
                p.spawn((
                    Mesh3d(rim.clone()),
                    MeshMaterial3d(rim_mat.clone()),
                    Transform::from_translation(Vec3::new(0.0, 0.0, side * 0.09)),
                ));
            }
        });
    }

    // Spoiler
    let spoiler =
        meshes.add(Shape::cuboid(Vec3::new(1.2, 0.06, 0.35), 0.02, ShapeDetail::Low).build());
    let spoiler_mat = materials.add({
        let mut mat = dual_directional([0.8, 0.25, 0.0], [1.0, 0.55, 0.1], [0.0, 1.0, 0.0]);
        mat.base.metallic = 0.7;
        mat.base.perceptual_roughness = 0.25;
        mat
    });
    commands.spawn((
        Mesh3d(spoiler),
        MeshMaterial3d(spoiler_mat.clone()),
        Transform::from_translation(Vec3::new(0.0, 0.75, -0.95)),
        ChildOf(kart_root),
        Name::new("Spoiler"),
    ));
    for side in [-0.4, 0.4] {
        let support = meshes.add(Shape::cylinder(0.03, 0.3, ShapeDetail::Low).build());
        commands.spawn((
            Mesh3d(support),
            MeshMaterial3d(spoiler_mat.clone()),
            Transform::from_translation(Vec3::new(side, 0.6, -0.95)),
            ChildOf(kart_root),
            Name::new("Spoiler Support"),
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
            Mesh3d(light.clone()),
            MeshMaterial3d(headlight_mat.clone()),
            Transform::from_translation(Vec3::new(side, 0.25, 1.02)),
            ChildOf(kart_root),
            Name::new("Headlight"),
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
            Mesh3d(light.clone()),
            MeshMaterial3d(tail_mat.clone()),
            Transform::from_translation(Vec3::new(side, 0.25, -1.02)),
            ChildOf(kart_root),
            Name::new("Tail Light"),
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
        Vec3::new(-250.0, 0.0, -250.0),
        Vec3::new(-280.0, 0.0, -230.0),
        Vec3::new(-230.0, 0.0, -270.0),
        Vec3::new(-100.0, 0.0, -260.0),
        Vec3::new(50.0, 0.0, -270.0),
        Vec3::new(200.0, 0.0, -250.0),
        Vec3::new(250.0, 0.0, -200.0),
        Vec3::new(320.0, 0.0, -50.0),
        Vec3::new(340.0, 0.0, 30.0),
        Vec3::new(250.0, 0.0, 180.0),
        Vec3::new(300.0, 0.0, 200.0),
        Vec3::new(100.0, 0.0, 200.0),
        Vec3::new(150.0, 0.0, 220.0),
        Vec3::new(-150.0, 0.0, 200.0),
        Vec3::new(-180.0, 0.0, 180.0),
        Vec3::new(-340.0, 0.0, -30.0),
        Vec3::new(-330.0, 0.0, 50.0),
        Vec3::new(100.0, 0.0, -150.0),
        Vec3::new(-80.0, 0.0, -130.0),
        Vec3::new(150.0, 0.0, 50.0),
        Vec3::new(-120.0, 0.0, 50.0),
        Vec3::new(-200.0, 0.0, -100.0),
        Vec3::new(80.0, 0.0, -200.0),
        Vec3::new(180.0, 0.0, 150.0),
    ];

    for (i, &pos) in tree_positions.iter().enumerate() {
        let scale = 0.7 + (i % 3) as f32 * 0.3;
        let (dark, light) = leaf_colors[i % leaf_colors.len()];
        let canopy_r = if i % 2 == 0 { 2.5 } else { 1.8 };

        recorder.set_position(pos + Vec3::Y * 2.0 * scale);
        recorder.set_scale(Vec3::splat(scale));
        recorder.set_rotation(Quat::IDENTITY);
        recorder.cone(
            0.3,
            4.0,
            ShapeDetail::Low,
            [trunk_light[0], trunk_light[1], trunk_light[2], 1.0],
        );

        recorder.set_position(pos + Vec3::Y * 5.5 * scale);
        recorder.set_scale(Vec3::splat(scale));
        recorder.set_rotation(Quat::IDENTITY);
        recorder.set_deformation(DeformationParams::wobble(0.15, 1.0, i as f32 * 0.3));
        recorder.sphere_dual_directional(
            canopy_r,
            ShapeDetail::Medium,
            dark,
            light,
            [0.0, 1.0, 0.0],
            MixSpace::default(),
        );
    }

    let bush_colors = [
        ([0.03, 0.18, 0.04], [0.12, 0.40, 0.10]),
        ([0.04, 0.20, 0.06], [0.14, 0.45, 0.12]),
    ];
    let bush_positions = [
        Vec3::new(-220.0, 0.0, -220.0),
        Vec3::new(-80.0, 0.0, -240.0),
        Vec3::new(180.0, 0.0, -230.0),
        Vec3::new(300.0, 0.0, -30.0),
        Vec3::new(220.0, 0.0, 180.0),
        Vec3::new(80.0, 0.0, 220.0),
        Vec3::new(-120.0, 0.0, 160.0),
        Vec3::new(-260.0, 0.0, -10.0),
    ];

    for (i, &pos) in bush_positions.iter().enumerate() {
        let (dark, light) = bush_colors[i % bush_colors.len()];
        recorder.set_position(pos + Vec3::Y * 0.5);
        recorder.set_scale(Vec3::splat(1.0));
        recorder.set_rotation(Quat::IDENTITY);
        recorder.set_deformation(DeformationParams::pulse(0.1, 1.5, i as f32 * 0.5));
        recorder.sphere_dual_directional(0.8, ShapeDetail::Low, dark, light, [0.0, 1.0, 0.0], MixSpace::default());
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

const COIN_COUNT: usize = 24;
/// Height of a coin's resting position above the road centerline.
const COIN_HOVER: f32 = 0.8;

fn build_coins(
    commands: &mut Commands,
    meshes: &mut Assets<bevy::render::mesh::Mesh>,
    materials: &mut Assets<GradMeshMaterial>,
    sampler: &BakedSpline,
) {
    let coin_mesh = meshes.add(Shape::cylinder(0.3, 0.08, ShapeDetail::Medium).build());
    let coin_mat = materials.add({
        let mut mat = dual_radial_alpha([1.0, 0.85, 0.2, 1.0], [0.8, 0.5, 0.1, 1.0]);
        mat.base.metallic = 0.9;
        mat.base.perceptual_roughness = 0.1;
        mat.base.emissive = bevy::color::LinearRgba::rgb(1.0, 0.85, 0.2) * 5_000.0;
        mat
    });

    // Walk the baked curve at even arc-length intervals so coins are evenly spaced,
    // and sit exactly on the centerline, hovering just above the road surface.
    for i in 0..COIN_COUNT {
        let center = sampler.sample((i as f32 + 0.5) / COIN_COUNT as f32).position;
        let pos = center + Vec3::Y * COIN_HOVER;

        commands.spawn((
            Mesh3d(coin_mesh.clone()),
            MeshMaterial3d(coin_mat.clone()),
            Transform::from_translation(pos)
                .with_rotation(Quat::from_rotation_x(std::f32::consts::PI / 2.0)),
            Coin {
                id: i as u32,
                collected: false,
                base: pos,
            },
            Name::new(format!("Coin {}", i)),
        ));
    }
}

// ─── Input ───

fn handle_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut state: ResMut<KartState>,
    sampler: Res<BakedSpline>,
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
        let start = sampler.sample(0.0);
        let (pos, fwd) = (start.position, start.forward);
        state.position = pos;
        state.heading = fwd.x.atan2(fwd.z);
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
    if state.boost < 0.01 {
        state.boost = 0.0;
    }

    let forward = Vec3::new(state.heading.sin(), 0.0, state.heading.cos());
    state.velocity = forward * state.speed;
}

// ─── Physics ───

fn update_physics(mut state: ResMut<KartState>, time: Res<Time>, sampler: Res<BakedSpline>) {
    let dt = time.delta_secs().min(0.033);
    let forward = Vec3::new(state.heading.sin(), 0.0, state.heading.cos());
    let move_xz = forward.xz() * state.speed * dt;
    state.position.x += move_xz.x;
    state.position.z += move_xz.y;

    // Find nearest track point using the sampler's baked frame table
    let best_t = sampler.nearest_t(state.position);
    let track_pos = sampler.sample(best_t).position;
    let min_dist_sq = (state.position.xz() - track_pos.xz()).length_squared();

    state.position.y = track_pos.y;
    state.track_t = best_t;

    // Off-track slowdown
    if min_dist_sq > 324.0 {
        let dist = min_dist_sq.sqrt();
        let off = (dist - 18.0).min(8.0) / 8.0;
        state.speed *= 1.0 - off * 0.05;
    }
}

// ─── Kart Transform ───

fn update_kart_transform(state: Res<KartState>, mut kart_query: Query<&mut Transform, With<Kart>>) {
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
            state.heading.sin() * -20.0,
            12.0,
            state.heading.cos() * -20.0,
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
    sampler: Res<BakedSpline>,
) {
    let kart_pos = state.position;
    let check_dist_sq = 36.0;

    for i in 0..CHECKPOINT_COUNT {
        let t = i as f32 / CHECKPOINT_COUNT as f32;
        let cp_pos = sampler.sample(t).position;
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
        if coin.collected {
            continue;
        }
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

fn animate_coins(time: Res<Time>, mut coin_query: Query<(&Coin, &mut Transform)>) {
    let t = time.elapsed_secs();
    for (coin, mut transform) in &mut coin_query {
        if coin.collected {
            continue;
        }
        transform.rotation = Quat::from_rotation_x(std::f32::consts::PI / 2.0)
            * Quat::from_rotation_y(t * 2.0 + coin.id as f32);
        transform.translation = coin.base
            + Vec3::Y * ((t * 1.5 + coin.id as f32 * 0.5).sin() * 0.15);
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
                *text = Text2d(format!("{}/{}", state.coins, COIN_COUNT).into());
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
