//! Dash gallery for the SDF 2D rim: every snap mode on discs, an ellipse, rounded
//! rectangles and a partial arc, at rim widths from 1.5 to 18 pixels.
//!
//! Columns are `Off`, `Tiling` and `EndToEnd` (dashes given by length) and a
//! count-based pattern. Look at the start/end of each outline (the +X side) and
//! at the rectangle corners for seams, jumps and uneven dashes. The patterns
//! slide slowly around the outline so a seam would flicker.
//!
//! Run with: `cargo run --example sdf_dash_gallery`

mod common;

use std::f32::consts::PI;

use bevy::mesh::Mesh2d;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::sprite_render::MeshMaterial2d;
use bevy_gradmesh::{Dash2d, DashSnap, GradMeshPlugin, SdfMaterial2d, unit_quad};
use common::screenshot::AutoScreenshotPlugin;

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "BevyGradmesh SDF Dash Gallery".into(),
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
            AutoScreenshotPlugin::new("screenshot_sdf_dash_gallery.png")
                .with_title("SDF dashes")
                .with_description("Off / Tiling / EndToEnd / count, on disc, ellipse, rect, arc"),
        )
        .insert_resource(ClearColor(Color::srgb(0.1, 0.1, 0.13)))
        .add_systems(Startup, setup)
        .add_systems(Update, slide)
        .run();
}

/// Marks a dashed material whose offset slides over time.
#[derive(Component)]
struct Sliding(Handle<SdfMaterial2d>);

/// One row of the gallery: a shape and its rim, plus the dash size for it.
struct Row {
    material: fn() -> SdfMaterial2d,
    length: f32,
    gap: f32,
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<SdfMaterial2d>>,
) {
    commands.spawn((Camera2d, Msaa::Off));
    let quad = meshes.add(unit_quad());

    let rows = [
        // Thick rim on a disc.
        Row {
            material: || SdfMaterial2d::disc(48.0).with_rim(14.0),
            length: 16.0,
            gap: 9.0,
        },
        // 1.5 px rim: anti-aliasing at the smallest width.
        Row {
            material: || SdfMaterial2d::disc(48.0).with_rim(1.5),
            length: 9.0,
            gap: 6.0,
        },
        // Ellipse.
        Row {
            material: || SdfMaterial2d::ellipse(Vec2::new(72.0, 40.0)).with_rim(8.0),
            length: 14.0,
            gap: 8.0,
        },
        // Rounded rectangle with a large corner radius.
        Row {
            material: || {
                SdfMaterial2d::rect(Vec2::new(150.0, 90.0))
                    .with_corner_radius(32.0)
                    .with_rim(8.0)
            },
            length: 14.0,
            gap: 8.0,
        },
        // Nearly square corners with a wide rim.
        Row {
            material: || {
                SdfMaterial2d::rect(Vec2::new(150.0, 90.0))
                    .with_corner_radius(2.0)
                    .with_rim(18.0)
            },
            length: 20.0,
            gap: 10.0,
        },
        // Partial arc: the first and last dash meet the ends in `EndToEnd`.
        Row {
            material: || {
                SdfMaterial2d::disc(48.0)
                    .with_rim(10.0)
                    .with_arc(-PI * 0.25, PI * 1.5)
            },
            length: 14.0,
            gap: 9.0,
        },
    ];

    let columns = [
        ("Off", Some(DashSnap::Off), false),
        ("Tiling", Some(DashSnap::Tiling), false),
        ("EndToEnd", Some(DashSnap::EndToEnd), false),
        ("Count 12, Tiling", None, true),
    ];

    let colors = [
        Color::srgb(0.98, 0.78, 0.35),
        Color::srgb(0.45, 0.85, 0.95),
        Color::srgb(0.95, 0.5, 0.65),
        Color::srgb(0.6, 0.95, 0.55),
    ];

    for (c, (label, snap, by_count)) in columns.iter().enumerate() {
        let x = -480.0 + c as f32 * 320.0;
        commands.spawn((
            Text2d(label.to_string()),
            TextColor(Color::srgba(1.0, 1.0, 1.0, 0.8)),
            TextFont {
                font_size: FontSize::Px(15.0),
                ..default()
            },
            Transform::from_xyz(x, -348.0, 1.0),
        ));
        for (r, row) in rows.iter().enumerate() {
            let y = 260.0 - r as f32 * 108.0;
            let dash = if *by_count {
                Dash2d::by_count(12, 0.4)
            } else {
                Dash2d::by_length(row.length, row.gap).with_snap(snap.unwrap_or_default())
            };
            let material = (row.material)().with_dash(dash).with_color(colors[c]);
            let scale = material.quad_scale();
            let handle = materials.add(material);
            commands.spawn((
                Mesh2d(quad.clone()),
                MeshMaterial2d(handle.clone()),
                Transform::from_xyz(x, y, 0.0).with_scale(scale),
                Sliding(handle),
            ));
        }
    }
}

/// Slides every pattern along its outline by editing only the offset float.
fn slide(time: Res<Time>, mut materials: ResMut<Assets<SdfMaterial2d>>, query: Query<&Sliding>) {
    let offset = (time.elapsed_secs() * 0.15).fract();
    for Sliding(handle) in &query {
        if let Some(mut m) = materials.get_mut(handle) {
            m.uniform.dash.z = offset;
        }
    }
}
