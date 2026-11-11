//! SDF 2D parameter morphing demo — showcases smooth animations of SDF material uniforms.
//!
//! Unlike `lerp.rs` (which morphs between shape kinds using mesh outlines), the SDF shader
//! evaluates a discrete shape kind per pixel, so cross-kind morphing is impossible. However,
//! within the same kind, all parameters are smooth floats: size, rim width, corner radius,
//! arc span, and colours. Every shape here uses the *same* unit quad mesh.
//!
//! Six shapes animate continuously:\n
//! 1. **Disc size pulse**: radius oscillates between 40 and 100.\n
//! 2. **Rect corner radius**: sharp rectangle morphs to a pill shape and back.\n
//! 3. **Ellipse rim width**: a thin ring thickens and thins.\n
//! 4. **Disc arc span**: a partial arc sweeps from a sliver to a full circle.\n
//! 5. **Polygon side count**: a hexagon morphs to a near-circle (12 sides) and back.\n
//! 6. **Star inner ratio**: a star's inner radius ratio oscillates, changing its spikiness.\n
//!
//! Run with: `cargo run --example lerp_2d`

mod common;

use std::f32::consts::{PI, TAU};

use bevy::mesh::Mesh2d;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::sprite_render::MeshMaterial2d;
use bevy_gradmesh::{GradMeshPlugin, SdfMaterial2d, unit_quad};
use common::screenshot::AutoScreenshotPlugin;


/// Run this example. Called by the WASM runner.
pub fn run(app: &mut App) {
    app
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "BevyGradmesh SDF 2D Lerp".into(),
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
            AutoScreenshotPlugin::new("screenshot_lerp_2d.png")
                .animation(4.0)
                .with_title("SDF 2D Lerp")
                .with_description(
                    "Smooth SDF parameter morphs: size, corners, rim, arc, sides, star ratio",
                ),
        )
        .insert_resource(ClearColor(Color::srgb(0.1, 0.1, 0.13)))
        .add_systems(Startup, setup)
        .add_systems(Update, animate)
        ;
}

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "BevyGradmesh SDF 2D Lerp".into(),
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
            AutoScreenshotPlugin::new("screenshot_lerp_2d.png")
                .animation(4.0)
                .with_title("SDF 2D Lerp")
                .with_description(
                    "Smooth SDF parameter morphs: size, corners, rim, arc, sides, star ratio",
                ),
        )
        .insert_resource(ClearColor(Color::srgb(0.1, 0.1, 0.13)))
        .add_systems(Startup, setup)
        .add_systems(Update, animate)
        .run();
}

/// Which animation this entity performs.
#[derive(Component)]
struct AnimationType {
    handle: Handle<SdfMaterial2d>,
    kind: AnimKind,
}

#[derive(Clone, Copy)]
enum AnimKind {
    /// Disc: pulse size between 40 and 100
    DiscSize,
    /// Rect: morph corner radius from 0 to half shortest side
    RectCorner,
    /// Ellipse: morph rim width between 2 and 20
    EllipseRim,
    /// Disc with arc: sweep arc span from sliver to full circle
    DiscArc,
    /// Polygon: morph side count from 6 to 12
    PolygonSides,
    /// Star: morph inner radius ratio from 0.2 to 0.7
    StarRatio,
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<SdfMaterial2d>>,
) {
    commands.spawn((Camera2d, Msaa::Off));
    let quad = meshes.add(unit_quad());

    let mut spawn =
        |commands: &mut Commands, material: SdfMaterial2d, pos: Vec2, z: f32, kind: AnimKind| {
            let scale = material.quad_scale();
            let handle = materials.add(material);
            commands.spawn((
                Mesh2d(quad.clone()),
                MeshMaterial2d(handle.clone()),
                Transform::from_translation(pos.extend(z)).with_scale(scale),
                AnimationType { handle, kind },
            ));
        };

    let warm = (Color::srgb(1.0, 0.35, 0.25), Color::srgb(1.0, 0.9, 0.35));
    let cool = (Color::srgb(0.25, 0.4, 1.0), Color::srgb(0.4, 1.0, 0.85));

    // Row 1: Disc size pulse, Rect corner radius, Ellipse rim width
    spawn(
        &mut commands,
        SdfMaterial2d::disc(70.0).with_colors(warm.0, warm.1),
        Vec2::new(-450.0, 180.0),
        0.0,
        AnimKind::DiscSize,
    );
    spawn(
        &mut commands,
        SdfMaterial2d::rect(Vec2::new(160.0, 120.0))
            .with_corner_radius(0.0)
            .with_colors(cool.0, cool.1),
        Vec2::new(-180.0, 180.0),
        0.0,
        AnimKind::RectCorner,
    );
    spawn(
        &mut commands,
        SdfMaterial2d::ellipse(Vec2::new(90.0, 60.0))
            .with_rim(6.0)
            .with_color(Color::srgb(0.9, 0.5, 0.9)),
        Vec2::new(90.0, 180.0),
        0.0,
        AnimKind::EllipseRim,
    );

    // Row 2: Disc arc span, Polygon side count, Star inner ratio
    spawn(
        &mut commands,
        SdfMaterial2d::disc(70.0)
            .with_rim(10.0)
            .with_arc(0.0, PI)
            .with_colors(warm.0, warm.1),
        Vec2::new(-450.0, -30.0),
        0.0,
        AnimKind::DiscArc,
    );
    spawn(
        &mut commands,
        SdfMaterial2d::polygon(6, 70.0).with_colors(cool.0, cool.1),
        Vec2::new(-180.0, -30.0),
        0.0,
        AnimKind::PolygonSides,
    );
    spawn(
        &mut commands,
        SdfMaterial2d::star(5, 70.0, 28.0)
            .with_colors(Color::srgb(1.0, 0.7, 0.2), Color::srgb(0.3, 0.9, 0.5)),
        Vec2::new(90.0, -30.0),
        0.0,
        AnimKind::StarRatio,
    );
}

/// Animates SDF material parameters by editing uniform floats each frame.
///
/// Each shape oscillates between two extremes using a sine wave mapped to 0..1.
fn animate(time: Res<Time>, mut materials: ResMut<Assets<SdfMaterial2d>>, query: Query<&AnimationType>) {
    let t = (time.elapsed_secs() * 0.5).sin() * 0.5 + 0.5; // oscillates 0..1

    for anim in &query {
        let Some(mut mat) = materials.get_mut(&anim.handle) else {
            continue;
        };
        let u = &mut mat.uniform;

        match anim.kind {
            AnimKind::DiscSize => {
                // Size pulse: small (40) ↔ large (100)
                let radius = 40.0 + 60.0 * t;
                u.size.x = radius * 2.0;
                u.size.y = radius * 2.0;
            }
            AnimKind::RectCorner => {
                // Morph corner radius: sharp (0) ↔ pill (half shortest side)
                let max_radius = (u.size.x.min(u.size.y) * 0.5).max(0.0);
                u.size.z = max_radius * t;
            }
            AnimKind::EllipseRim => {
                // Morph rim width: thin (2) ↔ thick (20)
                u.shape.y = 2.0 + 18.0 * t;
            }
            AnimKind::DiscArc => {
                // Arc span: sliver (0.1) ↔ full circle (TAU)
                u.shape.w = 0.1 + (TAU - 0.1) * t;
            }
            AnimKind::PolygonSides => {
                // Morph side count: hexagon (6) ↔ dodecagon (12)
                u.params.x = 6.0 + 6.0 * t;
            }
            AnimKind::StarRatio => {
                // Morph inner radius ratio: spiky (0.2) ↔ rounded (0.7)
                u.params.y = 0.2 + 0.5 * t;
            }
        }
    }
}
