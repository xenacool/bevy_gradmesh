//! Immediate mode drawing example — showcases the `ImmediateRecorder` API.
//!
//! Demonstrates per-frame drawing with animated particles, billboarding,
//! push/pop state transform stacking, render layer ordering, and line drawing.
//!
//! Run with: `cargo run --example immediate`

mod common;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{
    opaque, ShapeDetail,
    ImmediateRecorder, BillboardMode, GradMeshPlugin,
    imm_opaque, imm_transparent, imm_additive, imm_dual_directional,
    GradMeshMaterial,
};
use common::screenshot::AutoScreenshotPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "BevyGradmesh Immediate Mode".into(),
                resolution: (1280, 720).into(),
                ..default()
            }),
            ..default()
        }).set(RenderPlugin {
            synchronous_pipeline_compilation: true,
            ..default()
        }))
        .add_plugins(GradMeshPlugin)
        .add_plugins(
            AutoScreenshotPlugin::new("screenshot_immediate.png")
                .with_title("Immediate Mode")
                .with_description("Per-frame drawing with particles, billboarding, and line drawing")
        )
        .insert_resource(ClearColor(Color::srgb(0.08, 0.08, 0.1)))
        .add_systems(Startup, setup)
        .add_systems(PostUpdate, (draw_immediate, position_labels).chain())
        .run();
}

#[derive(Component)]
struct WorldLabel(Vec3);

/// Stores the recorder for immediate mode drawing.
#[derive(Resource)]
struct Recorder(ImmediateRecorder);

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
) {
    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 10.0, 10.0).looking_at(Vec3::new(0.0, 0.8, -1.0), Vec3::Y),
    ));

    // Lighting
    commands.spawn((
        DirectionalLight {
            illuminance: 15_000.0,
            ..default()
        },
        Transform::from_xyz(5.0, 8.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn(AmbientLight {
        color: Color::srgb(0.5, 0.5, 0.6),
        brightness: 200.0,
        affects_lightmapped_meshes: false,
    });

    // Ground plane
    let ground = meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(26.0)));
    commands.spawn((
        Mesh3d(ground),
        MeshMaterial3d(materials.add(opaque(0.1, 0.1, 0.12))),
        Transform::from_translation(Vec3::new(0.0, -0.01, 0.0)),
    ));

    // Static reference objects (not immediate mode)
    // A wireframe-like cube made of lines
    let cube_center = Vec3::new(-4.0, 1.0, -3.0);
    spawn_label(&mut commands, "Animated Particles", cube_center + Vec3::new(0.0, 1.4, 0.0));

    // Push/pop state demo area
    let stack_center = Vec3::new(0.0, 1.0, -3.0);
    spawn_label(&mut commands, "Push/Pop State Stack", stack_center + Vec3::new(0.0, 1.4, 0.0));

    // Billboarding demo area
    let bill_center = Vec3::new(4.0, 1.0, -3.0);
    spawn_label(&mut commands, "Billboarding", bill_center + Vec3::new(0.0, 1.4, 0.0));

    // Line drawing demo
    let line_center = Vec3::new(-4.0, 1.0, 1.0);
    spawn_label(&mut commands, "Line Drawing", line_center + Vec3::new(0.0, 1.4, 0.0));

    // Render layer demo
    let layer_center = Vec3::new(0.0, 1.0, 1.0);
    spawn_label(&mut commands, "Render Layers", layer_center + Vec3::new(0.0, 1.4, 0.0));

    // Mixed shapes demo
    let mixed_center = Vec3::new(4.0, 1.0, 1.0);
    spawn_label(&mut commands, "Mixed Shapes", mixed_center + Vec3::new(0.0, 1.4, 0.0));

    // Initialize recorder
    commands.insert_resource(Recorder(ImmediateRecorder::new()));
}

/// Per-frame immediate mode drawing.
fn draw_immediate(
    time: Res<Time>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
    mut recorder: ResMut<Recorder>,
) {
    let t = time.elapsed_secs();
    let rec = &mut recorder.0;

    rec.begin(&mut commands);

    // ─── Animated Particles (orbiting spheres) ───
    let particle_count = 20;
    for i in 0..particle_count {
        let angle = (i as f32 / particle_count as f32) * std::f32::consts::TAU + t * 0.5;
        let radius = 1.2 + (i as f32 * 0.1).sin() * 0.3;
        let x = angle.cos() * radius - 4.0;
        let z = angle.sin() * radius - 3.0;
        let y = 0.5 + (t + i as f32 * 0.3).sin() * 0.4;

        let hue = i as f32 / particle_count as f32;
        let color = hsl_to_rgb(hue, 0.8, 0.6);
        let mat = imm_opaque(color[0], color[1], color[2]);

        rec.set_position(Vec3::new(x, y, z));
        rec.set_scale(Vec3::splat(0.08 + (t.sin() * 0.5 + 0.5) * 0.04));
        rec.sphere(&mut commands, &mut meshes, &mut materials, 0.5, ShapeDetail::Low, mat);
    }

    // ─── Push/Pop State Stack (nested transforms) ───
    let base = Vec3::new(0.0, 0.3, -3.0);
    rec.set_position(base);
    rec.set_rotation(Quat::IDENTITY);
    rec.set_scale(Vec3::ONE);

    // Base cuboid
    rec.cuboid(
        &mut commands, &mut meshes, &mut materials,
        Vec3::splat(0.6), 0.0, ShapeDetail::Low,
        imm_opaque(0.3, 0.5, 0.9),
    );

    // Push and offset for child
    rec.push_state();
    rec.set_position(base + Vec3::new(0.0, 0.5, 0.0));
    rec.set_rotation(Quat::from_rotation_y(t * 0.8));
    rec.cuboid(
        &mut commands, &mut meshes, &mut materials,
        Vec3::splat(0.4), 0.1, ShapeDetail::Low,
        imm_opaque(0.9, 0.5, 0.3),
    );

    // Push again for grandchild
    rec.push_state();
    rec.set_position(base + Vec3::new(0.0, 0.9, 0.0));
    rec.set_rotation(Quat::from_rotation_x(t * 1.2));
    rec.sphere(
        &mut commands, &mut meshes, &mut materials,
        0.2, ShapeDetail::Low,
        imm_opaque(0.3, 0.9, 0.5),
    );
    rec.pop_state(); // Back to child level

    // Another child at same level
    rec.set_position(base + Vec3::new(0.5, 0.5, 0.0));
    rec.set_rotation(Quat::from_rotation_z(t * 0.6));
    rec.cone(
        &mut commands, &mut meshes, &mut materials,
        0.15, 0.4, ShapeDetail::Low,
        imm_opaque(0.9, 0.3, 0.6),
    );
    rec.pop_state(); // Back to base level

    // ─── Billboarding (2D shapes facing camera) ───
    let bill_pos = Vec3::new(4.0, 1.0, -3.0);

    // FaceCameraY disc
    rec.set_position(bill_pos + Vec3::new(-0.8, 0.0, 0.0));
    rec.disc(
        &mut commands, &mut meshes, &mut materials,
        0.4, imm_opaque(0.3, 0.8, 0.9),
        BillboardMode::FaceCameraY,
    );

    // FaceCamera disc
    rec.set_position(bill_pos + Vec3::new(0.0, 0.0, 0.0));
    rec.disc(
        &mut commands, &mut meshes, &mut materials,
        0.4, imm_opaque(0.9, 0.4, 0.7),
        BillboardMode::FaceCamera,
    );

    // No billboard disc (flat, facing +Z)
    rec.set_position(bill_pos + Vec3::new(0.8, 0.0, 0.0));
    rec.disc(
        &mut commands, &mut meshes, &mut materials,
        0.4, imm_opaque(0.4, 0.9, 0.4),
        BillboardMode::None,
    );

    // Animated billboard ring
    rec.set_position(bill_pos + Vec3::new(0.0, 1.2, 0.0));
    rec.set_scale(Vec3::splat(1.0 + t.sin() * 0.2));
    rec.disc_rim(
        &mut commands, &mut meshes, &mut materials,
        0.3, 0.05, imm_opaque(0.9, 0.8, 0.2),
        BillboardMode::FaceCamera,
    );

    // ─── Line Drawing ───
    let line_base = Vec3::new(-4.0, 0.3, 1.0);

    // Static triangle
    let tri_pts = [
        line_base + Vec3::new(0.0, 1.2, 0.0),
        line_base + Vec3::new(-0.6, 0.0, 0.0),
        line_base + Vec3::new(0.6, 0.0, 0.0),
    ];
    let line_mat = || imm_opaque(0.3, 0.7, 0.9);
    rec.line_3d(&mut commands, &mut meshes, &mut materials, tri_pts[0], tri_pts[1], 0.02, line_mat());
    rec.line_3d(&mut commands, &mut meshes, &mut materials, tri_pts[1], tri_pts[2], 0.02, line_mat());
    rec.line_3d(&mut commands, &mut meshes, &mut materials, tri_pts[2], tri_pts[0], 0.02, line_mat());

    // Animated spiral lines
    let spiral_center = line_base + Vec3::new(2.0, 0.6, 0.0);
    let spiral_count = 12;
    for i in 0..spiral_count {
        let a1 = (i as f32 / spiral_count as f32) * std::f32::consts::TAU + t;
        let a2 = ((i + 1) as f32 / spiral_count as f32) * std::f32::consts::TAU + t;
        let r1 = 0.8 - i as f32 * 0.05;
        let r2 = 0.8 - (i + 1) as f32 * 0.05;
        let p1 = spiral_center + Vec3::new(a1.cos() * r1, a1.sin() * r1, 0.0);
        let p2 = spiral_center + Vec3::new(a2.cos() * r2, a2.sin() * r2, 0.0);
        let spiral_mat = imm_opaque(0.9, 0.5, 0.2);
        rec.line_3d(&mut commands, &mut meshes, &mut materials, p1, p2, 0.02, spiral_mat);
    }

    // ─── Render Layers (depth ordering) ───
    let layer_base = Vec3::new(0.0, 0.5, 1.0);

    // Background layer — large transparent disc
    rec.set_position(layer_base + Vec3::new(0.0, 0.3, 0.0));
    rec.set_scale(Vec3::splat(1.5));
    rec.disc(
        &mut commands, &mut meshes, &mut materials,
        0.5, imm_transparent(0.2, 0.3, 0.6, 0.3),
        BillboardMode::FaceCamera,
    );

    // Middle layer — sphere
    rec.set_position(layer_base + Vec3::new(0.0, 0.3, 0.0));
    rec.set_scale(Vec3::ONE);
    rec.sphere(
        &mut commands, &mut meshes, &mut materials,
        0.3, ShapeDetail::High,
        imm_dual_directional([0.5, 0.2, 0.15], [0.96, 0.8, 0.69], [0.0, 1.0, 0.0]),
    );

    // Overlay layer — small additive glow
    rec.set_position(layer_base + Vec3::new(0.0, 0.8, 0.0));
    rec.set_scale(Vec3::splat(0.15 + t.sin() * 0.05));
    rec.sphere(
        &mut commands, &mut meshes, &mut materials,
        0.5, ShapeDetail::Low,
        imm_additive(1.0, 0.9, 0.3),
    );

    // ─── Mixed Shapes (various primitives) ───
    let mixed_base = Vec3::new(4.0, 0.3, 1.0);

    // Torus
    rec.set_position(mixed_base + Vec3::new(-1.0, 0.5, 0.0));
    rec.set_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2));
    rec.set_scale(Vec3::ONE);
    rec.torus(
        &mut commands, &mut meshes, &mut materials,
        0.3, 0.1, 0.0, std::f32::consts::TAU, ShapeDetail::High,
        imm_opaque(0.5, 0.8, 0.3),
    );

    // Pyramid
    rec.set_position(mixed_base + Vec3::new(0.0, 0.3, 0.0));
    rec.set_rotation(Quat::from_rotation_y(t * 0.5));
    rec.pyramid(
        &mut commands, &mut meshes, &mut materials,
        4, 0.3, 0.6, ShapeDetail::High,
        imm_opaque(0.9, 0.4, 0.3),
    );

    // Capsule
    rec.set_position(mixed_base + Vec3::new(1.0, 0.4, 0.0));
    rec.set_rotation(Quat::from_rotation_z(t * 0.7));
    rec.capsule(
        &mut commands, &mut meshes, &mut materials,
        0.12, 0.5, ShapeDetail::High,
        imm_opaque(0.3, 0.5, 0.9),
    );

    // Regular polygon (2D)
    rec.set_position(mixed_base + Vec3::new(0.0, 1.2, 0.0));
    rec.set_scale(Vec3::splat(1.0 + t.sin() * 0.1));
    rec.regular_polygon(
        &mut commands, &mut meshes, &mut materials,
        6, 0.25, imm_opaque(0.8, 0.3, 0.7),
        BillboardMode::FaceCamera,
    );

    // Polygon rim
    rec.set_position(mixed_base + Vec3::new(1.0, 1.2, 0.0));
    rec.set_scale(Vec3::splat(1.0 + t.cos() * 0.1));
    rec.polygon_rim(
        &mut commands, &mut meshes, &mut materials,
        8, 0.25, 0.04, imm_opaque(0.3, 0.9, 0.6),
        BillboardMode::FaceCamera,
    );
}

/// Simple HSL to RGB conversion.
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> [f32; 3] {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;

    let (r1, g1, b1) = if h < 1.0 / 6.0 {
        (c, x, 0.0)
    } else if h < 2.0 / 6.0 {
        (x, c, 0.0)
    } else if h < 3.0 / 6.0 {
        (0.0, c, x)
    } else if h < 4.0 / 6.0 {
        (0.0, x, c)
    } else if h < 5.0 / 6.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    [r1 + m, g1 + m, b1 + m]
}

fn spawn_label(commands: &mut Commands, text: &str, position: Vec3) {
    commands.spawn((
        Text::new(text.to_string()),
        TextFont { font_size: FontSize::Px(15.0), ..default() },
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
    let Ok((camera, camera_transform)) = camera.single() else { return; };
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
