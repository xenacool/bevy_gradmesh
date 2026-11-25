//! `SdfMaterial2d::lerp` and `FlatShape::sdf_2d()` cross-backend conversion (PBR-free).
//!
//! All four shapes share one `unit_quad()` mesh and run with `--no-default-features`:
//! - **Disc ↔ Rounded Rect**: converts `Shape::disc` and `Shape::rectangle` through
//!   `Shape::as_flat().unwrap().sdf_2d()`, then morphs cleanly between them via
//!   `SdfMaterial2d::lerp` (using corner radius).
//! - **Thin Rim ↔ Thick Dashed Rim**: interpolates rim width, colors, and dash density.
//! - **Spiky Star ↔ Plump Star**: converts `Shape::star` via `FlatShape::sdf_2d()` and
//!   interpolates inner radius + radial gradient colors via `SdfMaterial2d::lerp`.
//! - **Hexagon ↔ Dodecagon**: converts `Shape::polygon` via `FlatShape::sdf_2d()` and
//!   interpolates side count + gradient colors.
//!
//! Run with: `cargo run --example sdf_lerp --no-default-features`

mod common;

use bevy::mesh::Mesh2d;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::sprite_render::MeshMaterial2d;
use bevy_gradmesh::{Dash2d, GradMeshPlugin, SdfMaterial2d, Shape, ShapeDetail, unit_quad};
use common::screenshot::AutoScreenshotPlugin;

/// Run this example. Called by the WASM runner.
pub fn run(app: &mut App) {
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "BevyGradmesh SdfMaterial2d::lerp & FlatShape::sdf_2d".into(),
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
        AutoScreenshotPlugin::new("screenshot_sdf_lerp.png")
            .animation(4.0)
            .with_title("SdfMaterial2d::lerp & FlatShape::sdf_2d")
            .with_description("Cross-backend FlatShape::sdf_2d() and smooth SdfMaterial2d::lerp"),
    )
    .insert_resource(ClearColor(Color::srgb(0.08, 0.09, 0.13)))
    .add_systems(Startup, setup)
    .add_systems(Update, animate);
}

fn main() {
    let mut app = App::new();
    run(&mut app);
    app.run();
}

#[derive(Component)]
struct SdfLerp {
    from: SdfMaterial2d,
    to: SdfMaterial2d,
    material: Handle<SdfMaterial2d>,
    phase: f32,
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<SdfMaterial2d>>,
) {
    commands.spawn((Camera2d, Msaa::Off));
    let quad = meshes.add(unit_quad());
    let d = ShapeDetail::Medium;

    // 1. Disc <-> Rounded Rectangle via FlatShape::sdf_2d() + SdfMaterial2d::lerp
    let disc = Shape::disc(85.0, d)
        .as_flat()
        .expect("disc is flat")
        .sdf_2d()
        .expect("disc converts to SDF")
        .with_colors(Color::srgb(0.18, 0.45, 0.96), Color::srgb(0.32, 0.92, 0.92));
    let rect = Shape::rectangle(Vec2::new(210.0, 130.0), d)
        .as_flat()
        .expect("rectangle is flat")
        .sdf_2d()
        .expect("rectangle converts to SDF")
        .with_corner_radius(16.0)
        .with_colors(Color::srgb(0.95, 0.34, 0.28), Color::srgb(1.0, 0.82, 0.28));

    // 2. Thin solid rim <-> Thick dashed rim
    let thin_rim = SdfMaterial2d::disc(86.0)
        .with_rim(6.0)
        .with_colors(Color::srgb(0.25, 0.85, 0.6), Color::srgb(0.45, 0.96, 0.88));
    let thick_dash = SdfMaterial2d::disc(96.0)
        .with_rim(22.0)
        .with_dash(Dash2d::by_count(12, 0.55))
        .with_colors(Color::srgb(0.96, 0.42, 0.72), Color::srgb(0.56, 0.3, 0.96));

    // 3. Spiky star <-> Plump star via FlatShape::sdf_2d()
    let spiky = Shape::star(5, 95.0, 32.0, d)
        .as_flat()
        .expect("star is flat")
        .sdf_2d()
        .expect("star converts to SDF")
        .with_colors(Color::srgb(1.0, 0.65, 0.18), Color::srgb(0.96, 0.24, 0.34));
    let plump = Shape::star(5, 95.0, 68.0, d)
        .as_flat()
        .expect("star is flat")
        .sdf_2d()
        .expect("star converts to SDF")
        .with_colors(Color::srgb(0.35, 0.9, 0.48), Color::srgb(0.2, 0.52, 0.92));

    // 4. Hexagon <-> Dodecagon via FlatShape::sdf_2d()
    let hex = Shape::polygon(6, 85.0, d)
        .as_flat()
        .expect("polygon is flat")
        .sdf_2d()
        .expect("polygon converts to SDF")
        .with_colors(Color::srgb(0.58, 0.3, 0.95), Color::srgb(0.95, 0.48, 0.8));
    let dodec = Shape::polygon(12, 92.0, d)
        .as_flat()
        .expect("polygon is flat")
        .sdf_2d()
        .expect("polygon converts to SDF")
        .with_colors(Color::srgb(0.22, 0.72, 0.96), Color::srgb(0.72, 0.96, 0.42));

    let items: [(Vec2, SdfMaterial2d, SdfMaterial2d, f32); 4] = [
        (Vec2::new(-320.0, 120.0), disc, rect, 0.0),
        (Vec2::new(320.0, 120.0), thin_rim, thick_dash, 0.9),
        (Vec2::new(-320.0, -140.0), spiky, plump, 1.8),
        (Vec2::new(320.0, -140.0), hex, dodec, 2.7),
    ];

    for (pos, from, to, phase) in items {
        let scale = from.quad_scale();
        let handle = materials.add(from.clone());
        commands.spawn((
            Mesh2d(quad.clone()),
            MeshMaterial2d(handle.clone()),
            Transform::from_translation(pos.extend(0.0)).with_scale(scale),
            SdfLerp {
                from,
                to,
                material: handle,
                phase,
            },
        ));
    }

    commands.spawn((
        Text::new("SdfMaterial2d::lerp + FlatShape::sdf_2d()  —  disc↔rect, rim↔dash, star, polygon"),
        TextFont {
            font_size: 18.0.into(),
            ..default()
        },
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            top: px(14),
            left: px(18),
            ..default()
        },
    ));
}

#[allow(clippy::needless_pass_by_value)]
fn animate(
    time: Res<Time>,
    mut materials: ResMut<Assets<SdfMaterial2d>>,
    mut query: Query<(&SdfLerp, &mut Transform)>,
) {
    let elapsed = time.elapsed_secs();
    for (entry, mut tf) in &mut query {
        let t = (elapsed * 0.7 + entry.phase).sin() * 0.5 + 0.5;
        let lerped = entry.from.lerp(&entry.to, t);
        tf.scale = lerped.quad_scale();
        if let Some(mut mat) = materials.get_mut(&entry.material) {
            *mat = lerped;
        }
    }
}
