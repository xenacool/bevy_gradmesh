//! Animated `Style::lerp` transitions on 3D primitives (including `Shape::tapered_cylinder`).
//!
//! Three pairs of contrasting `Style` descriptors interpolate continuously via `Style::lerp`:
//! 1. **Gradient & roughness/metallic** on a `Shape::tapered_cylinder` (warm matte sunset ↔
//!    cool metallic ocean).
//! 2. **Noise ↔ Stripes** surface modulation on a `Shape::rounded_cuboid`.
//! 3. **Dual directional ↔ Dual radial + Rim glow** on a `Shape::sphere`.
//!
//! Run with: `cargo run --example style_lerp`

mod common;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{GradMeshMaterial, GradMeshPlugin, Shape, ShapeDetail, Style};
use common::screenshot::{AutoScreenshotPlugin, spawn_world_label};

/// Run this example. Called by the WASM runner.
pub fn run(app: &mut App) {
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "BevyGradmesh Style::lerp".into(),
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
        AutoScreenshotPlugin::new("screenshot_style_lerp.png")
            .animation(4.0)
            .with_title("Style::lerp")
            .with_description("Smoothly interpolating gradient styles, PBR parameters, and effects"),
    )
    .insert_resource(ClearColor(Color::srgb(0.08, 0.09, 0.13)))
    .add_systems(Startup, setup)
    .add_systems(Update, animate_styles);
}

fn main() {
    let mut app = App::new();
    run(&mut app);
    app.run();
}

#[derive(Component)]
struct LerpStyle {
    from: Style,
    to: Style,
    material: Handle<GradMeshMaterial>,
    phase: f32,
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 2.6, 9.2).looking_at(Vec3::new(0.0, 0.2, 0.0), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 14_000.0,
            ..default()
        },
        Transform::from_xyz(4.0, 7.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.6, 0.62, 0.72),
        brightness: 520.0,
        affects_lightmapped_meshes: false,
    });

    let pairs: [(&str, Shape, Vec3, Style, Style, f32); 3] = [
        (
            "tapered_cylinder: Matte Sunset <-> Metallic Ocean",
            Shape::tapered_cylinder(0.65, 1.15, 2.2, ShapeDetail::High),
            Vec3::new(-3.4, 0.0, 0.0),
            Style::dual_directional_up(
                Color::srgb(0.88, 0.24, 0.2),
                Color::srgb(1.0, 0.84, 0.3),
            )
            .with_perceptual_roughness(0.85)
            .with_metallic(0.0),
            Style::dual_directional_up(
                Color::srgb(0.1, 0.26, 0.72),
                Color::srgb(0.32, 0.92, 0.96),
            )
            .with_perceptual_roughness(0.18)
            .with_metallic(0.85),
            0.0,
        ),
        (
            "cuboid (rounded): Noise <-> Stripes",
            Shape::cuboid(Vec3::splat(1.85), 0.28, ShapeDetail::High),
            Vec3::new(0.0, 0.0, 0.0),
            Style::dual_directional_up(
                Color::srgb(0.14, 0.52, 0.34),
                Color::srgb(0.55, 0.94, 0.58),
            )
            .with_noise(4.0, 0.35),
            Style::dual_directional_up(
                Color::srgb(0.46, 0.22, 0.8),
                Color::srgb(0.95, 0.45, 0.78),
            )
            .with_stripes(Vec3::Y, 4.5, 0.28),
            1.2,
        ),
        (
            "sphere: Warm Core <-> Radial Rim",
            Shape::sphere(1.15, ShapeDetail::High),
            Vec3::new(3.4, 0.0, 0.0),
            Style::dual_directional_up(
                Color::srgb(0.25, 0.22, 0.32),
                Color::srgb(0.96, 0.62, 0.24),
            ),
            Style::dual_radial(
                Color::srgb(0.12, 0.16, 0.34),
                Color::srgb(0.36, 0.85, 1.0),
            )
            .with_rim(Color::srgb(1.0, 0.55, 0.85), 2.4),
            2.4,
        ),
    ];

    for (label, shape, pos, from, to, phase) in pairs {
        let mat_handle = materials.add(from.build());
        commands.spawn((
            Mesh3d(meshes.add(shape.build())),
            MeshMaterial3d(mat_handle.clone()),
            Transform::from_translation(pos),
            LerpStyle {
                from,
                to,
                material: mat_handle,
                phase,
            },
        ));
        spawn_world_label(&mut commands, label, pos + Vec3::new(0.0, 1.75, 0.0));
    }
}

#[allow(clippy::needless_pass_by_value)]
fn animate_styles(
    time: Res<Time>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
    mut query: Query<(&LerpStyle, &mut Transform)>,
) {
    let elapsed = time.elapsed_secs();
    for (entry, mut tf) in &mut query {
        let t = ((elapsed * 0.75 + entry.phase).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
        let blended = entry.from.lerp(&entry.to, t);
        if let Some(mut mat) = materials.get_mut(&entry.material) {
            *mat = blended.build();
        }
        tf.rotation = Quat::from_euler(EulerRot::YXZ, elapsed * 0.45, 0.2, 0.0);
    }
}
