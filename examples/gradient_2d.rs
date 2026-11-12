//! 2D gradient shapes with `Camera2d`, `Mesh2d` and `GradMeshMaterial2d`.
//!
//! No 3D camera is needed: shapes are flat `Shape::build_2d()` meshes drawn with
//! an unlit gradient material. `Transform` scale resizes them and
//! `Transform.translation.z` layers them.
//!
//! Run with: `cargo run --example gradient_2d`

mod common;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::mesh::Mesh2d;
use bevy::sprite_render::MeshMaterial2d;
use bevy_gradmesh::{
    AppearanceSnapshot, Dash, GradMeshMaterial2d, GradMeshPlugin, MeshCache, Shape, ShapeDetail,
    ShapeType,
    Style,
};
use common::screenshot::AutoScreenshotPlugin;


/// Run this example. Called by the WASM runner.
pub fn run(app: &mut App) {
    app
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "BevyGradmesh 2D".into(),
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
            AutoScreenshotPlugin::new("screenshot_gradient_2d.png")
                .animation(3.0)
                .with_title("2D Gradients")
                .with_description("Mesh2d shapes with the unlit gradient material, no 3D camera"),
        )
        .add_systems(Startup, setup)
        ;
}

fn main() {
    let mut app = App::new();
    run(&mut app);
    app.run();
}

fn setup(
    mut commands: Commands,
    mut cache: ResMut<MeshCache>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial2d>>,
) {
    commands.spawn((Camera2d, Msaa::Off));

    let mut spawn = |commands: &mut Commands,
                     shape: Shape,
                     material: GradMeshMaterial2d,
                     translation: Vec3,
                     scale: Vec2| {
        commands.spawn((
            Mesh2d(cache.get_or_build_handle(&shape.clone(), &mut meshes)),
            MeshMaterial2d(materials.add(material)),
            Transform::from_translation(translation).with_scale(scale.extend(1.0)),
        ));
    };

    let card = Shape::arbitrary_polygon(
        Shape::rounded_rectangle(Vec2::new(1.0, 1.4), 0.12, 6),
        ShapeDetail::High,
    );
    let size = Vec2::new(200.0, 280.0);
    let pastel = [
        (Color::srgb(0.98, 0.76, 0.82), -420.0),
        (Color::srgb(0.99, 0.93, 0.62), -190.0),
        (Color::srgb(0.78, 0.7, 0.96), 40.0),
    ];
    for (color, x) in pastel {
        spawn(
            &mut commands,
            card.clone(),
            Style::shaded(color, 0.55).build_2d(),
            Vec3::new(x, 60.0, 0.0),
            size,
        );
    }

    // A card being dragged: larger, higher z, translucent, over the others.
    spawn(
        &mut commands,
        card.clone(),
        Style::shaded(Color::srgb(0.6, 0.9, 0.8), 0.5)
            .with_alpha(0.6)
            .build_2d(),
        Vec3::new(-110.0, 20.0, 5.0),
        size * 1.1,
    );

    // Radial gradient, dashed outline and an animated pulse.
    spawn(
        &mut commands,
        Shape::disc(0.5, ShapeDetail::Extreme),
        Style::dual_radial(Color::srgb(1.0, 0.95, 0.8), Color::srgb(0.95, 0.5, 0.6)).build_2d(),
        Vec3::new(330.0, 150.0, 0.0),
        Vec2::splat(220.0),
    );
    spawn(
        &mut commands,
        Shape::star(5, 0.5, 0.22, ShapeDetail::High),
        Style::dual_directional(Color::srgb(0.3, 0.45, 0.9), Color::srgb(0.7, 0.95, 1.0), Vec3::X)
            .build_2d()
            .with_appearance(AppearanceSnapshot::color_pulse(0.5, 0.0)),
        Vec3::new(330.0, -110.0, 0.0),
        Vec2::splat(200.0),
    );
    spawn(
        &mut commands,
        Shape::disc_rim(0.5, 0.06, ShapeDetail::Extreme),
        Style::opaque(Color::srgb(0.95, 0.6, 0.3))
            .with_shape_type(ShapeType::Disc)
            .with_dash(Dash::fixed(16, 0.4))
            .build_2d(),
        Vec3::new(-420.0, -220.0, 0.0),
        Vec2::splat(150.0),
    );
    spawn(
        &mut commands,
        Shape::arbitrary_polygon_with_holes(
            Shape::rounded_rectangle(Vec2::new(1.4, 0.8), 0.1, 4),
            vec![Shape::blob(0.22, 0.2, 5)],
            ShapeDetail::High,
        ),
        Style::shaded(Color::srgb(0.65, 0.85, 0.65), 0.5).build_2d(),
        Vec3::new(-150.0, -230.0, 0.0),
        Vec2::splat(260.0),
    );
}
