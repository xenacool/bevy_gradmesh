//! SDF 2D shapes: one shared quad mesh, shape evaluated per pixel.
//!
//! Every shape below uses the *same* mesh handle. Rims, partial arcs, rounded
//! corners and edge anti-aliasing come from the fragment shader, and the
//! animated shapes change only material floats each frame.
//!
//! Run with: `cargo run --example sdf_2d`

mod common;

use std::f32::consts::{PI, TAU};

use bevy::mesh::Mesh2d;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::sprite_render::MeshMaterial2d;
use bevy_gradmesh::{GradMeshPlugin, MixSpace, SdfMaterial2d, unit_quad};
use common::screenshot::AutoScreenshotPlugin;

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "BevyGradmesh SDF 2D".into(),
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
            AutoScreenshotPlugin::new("screenshot_sdf_2d.png")
                .with_title("SDF 2D")
                .with_description("One shared quad: discs, rims, arcs, rounded rects, no MSAA"),
        )
        .insert_resource(ClearColor(Color::srgb(0.1, 0.1, 0.13)))
        .add_systems(Startup, setup)
        .add_systems(Update, animate)
        .run();
}

#[derive(Component)]
struct Animated(Handle<SdfMaterial2d>);

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<SdfMaterial2d>>,
) {
    commands.spawn((Camera2d, Msaa::Off));
    let quad = meshes.add(unit_quad());

    let mut spawn = |commands: &mut Commands, material: SdfMaterial2d, pos: Vec2, z: f32| {
        let scale = material.quad_scale();
        let handle = materials.add(material);
        commands.spawn((
            Mesh2d(quad.clone()),
            MeshMaterial2d(handle.clone()),
            Transform::from_translation(pos.extend(z)).with_scale(scale),
            Animated(handle),
        ));
    };

    let warm = (Color::srgb(1.0, 0.35, 0.25), Color::srgb(1.0, 0.9, 0.35));
    let cool = (Color::srgb(0.25, 0.4, 1.0), Color::srgb(0.4, 1.0, 0.85));

    // Row 1: filled disc, ellipse, rounded rect, circle with a gradient in Oklch.
    spawn(
        &mut commands,
        SdfMaterial2d::disc(80.0).with_colors(warm.0, warm.1),
        Vec2::new(-450.0, 255.0),
        0.0,
    );
    spawn(
        &mut commands,
        SdfMaterial2d::ellipse(Vec2::new(110.0, 65.0)).with_colors(cool.0, cool.1),
        Vec2::new(-210.0, 255.0),
        0.0,
    );
    spawn(
        &mut commands,
        SdfMaterial2d::rect(Vec2::new(200.0, 140.0))
            .with_corner_radius(36.0)
            .with_colors(warm.0, cool.1)
            .with_gradient_direction(Vec2::new(1.0, -1.0)),
        Vec2::new(60.0, 255.0),
        0.0,
    );
    spawn(
        &mut commands,
        SdfMaterial2d::disc(80.0)
            .with_colors(Color::srgb(1.0, 0.1, 0.3), Color::srgb(0.1, 0.9, 1.0))
            .with_mix_space(MixSpace::Oklch)
            .with_gradient_direction(Vec2::Y),
        Vec2::new(330.0, 255.0),
        0.0,
    );

    // Row 2: rims.
    spawn(
        &mut commands,
        SdfMaterial2d::disc(80.0)
            .with_rim(12.0)
            .with_color(Color::srgb(0.5, 0.9, 0.6)),
        Vec2::new(-450.0, 90.0),
        0.0,
    );
    spawn(
        &mut commands,
        SdfMaterial2d::rect(Vec2::new(200.0, 140.0))
            .with_corner_radius(20.0)
            .with_rim(8.0)
            .with_color(Color::srgb(0.95, 0.65, 0.3)),
        Vec2::new(-210.0, 90.0),
        0.0,
    );
    spawn(
        &mut commands,
        SdfMaterial2d::ellipse(Vec2::new(110.0, 65.0))
            .with_rim(6.0)
            .with_colors(cool.0, cool.1),
        Vec2::new(60.0, 90.0),
        0.0,
    );
    // Thin rim: sub-pixel-ish AA check.
    spawn(
        &mut commands,
        SdfMaterial2d::disc(80.0)
            .with_rim(1.5)
            .with_color(Color::WHITE),
        Vec2::new(330.0, 90.0),
        0.0,
    );

    // Row 3: arcs (animated by `animate`).
    spawn(
        &mut commands,
        SdfMaterial2d::disc(80.0)
            .with_rim(14.0)
            .with_arc(0.0, PI * 1.2)
            .with_colors(warm.0, warm.1),
        Vec2::new(-450.0, -75.0),
        0.0,
    );
    spawn(
        &mut commands,
        SdfMaterial2d::disc(80.0)
            .with_arc(PI * 0.25, PI * 0.5)
            .with_colors(cool.0, cool.1),
        Vec2::new(-210.0, -75.0),
        0.0,
    );
    spawn(
        &mut commands,
        SdfMaterial2d::disc(80.0)
            .with_rim(18.0)
            .with_arc(0.0, TAU * 0.35)
            .with_color(Color::srgb(0.9, 0.5, 0.9)),
        Vec2::new(60.0, -75.0),
        0.0,
    );
    spawn(
        &mut commands,
        SdfMaterial2d::rect(Vec2::new(200.0, 140.0))
            .with_corner_radius(30.0)
            .with_color(Color::srgba(0.4, 0.8, 1.0, 0.6)),
        Vec2::new(330.0, -75.0),
        1.0,
    );

    // Row 4: gradients fitted to the bounds (radial) and around the centre (angular).
    let sunset = (Color::srgb(1.0, 0.85, 0.2), Color::srgb(0.7, 0.05, 0.4));
    spawn(
        &mut commands,
        SdfMaterial2d::disc(80.0)
            .with_colors(sunset.0, sunset.1)
            .with_radial_gradient(),
        Vec2::new(-450.0, -240.0),
        0.0,
    );
    spawn(
        &mut commands,
        SdfMaterial2d::rect(Vec2::new(200.0, 140.0))
            .with_corner_radius(30.0)
            .with_colors(cool.1, cool.0)
            .with_radial_gradient(),
        Vec2::new(-210.0, -240.0),
        0.0,
    );
    spawn(
        &mut commands,
        SdfMaterial2d::disc(80.0)
            .with_colors(Color::srgb(1.0, 0.1, 0.3), Color::srgb(0.1, 0.9, 1.0))
            .with_angular_gradient(0.0),
        Vec2::new(60.0, -240.0),
        0.0,
    );
    spawn(
        &mut commands,
        SdfMaterial2d::ellipse(Vec2::new(110.0, 65.0))
            .with_colors(Color::srgb(1.0, 0.1, 0.3), Color::srgb(0.1, 0.9, 1.0))
            .with_angular_sweep(PI * 0.5)
            .with_mix_space(MixSpace::Oklch),
        Vec2::new(330.0, -240.0),
        0.0,
    );
}

/// Animates the rim arc, span and corner radius by editing material floats only.
fn animate(time: Res<Time>, mut materials: ResMut<Assets<SdfMaterial2d>>, query: Query<&Animated>) {
    let t = time.elapsed_secs();
    for Animated(handle) in &query {
        let Some(mut m) = materials.get_mut(handle) else {
            continue;
        };
        // Only shapes that already have a partial arc keep rotating.
        if m.uniform.shape.w < TAU - 0.01 {
            m.uniform.shape.z = t;
        }
    }
}
