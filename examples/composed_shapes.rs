//! Composed 3D pieces: compound meshes, extruded polygons with holes, lathes,
//! PBR styles, rounded roofs and gems, procedural noise and stripes, a rim glow,
//! a rotated piece with a local gradient and a translucent ghost.
//!
//! Run with: `cargo run --example composed_shapes`

mod common;

use bevy::light::NotShadowCaster;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{
    DirectionSpace, GradMeshMaterial, GradMeshMaterialPool, GradMeshPlugin, MeshCache, Shape,
    ShapeDetail, ShapePart, Style, arch, blob, rounded_rectangle,
};
use common::screenshot::AutoScreenshotPlugin;

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "BevyGradmesh Composed Shapes".into(),
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
            AutoScreenshotPlugin::new("screenshot_composed_shapes.png")
                .with_title("Composed Shapes")
                .with_description("Compound, extruded, lathe, PBR styles and a ghost preview")
                .wait_frames(3),
        )
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut cache: ResMut<MeshCache>,
    mut pool: ResMut<GradMeshMaterialPool>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 4.5, 11.0).looking_at(Vec3::new(0.0, 1.4, 0.0), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 14_000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(5.0, 9.0, 6.0).looking_at(Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
    ));
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.65, 0.65, 0.8),
        brightness: 350.0,
        affects_lightmapped_meshes: false,
    });

    let d = ShapeDetail::High;
    let mut piece = |commands: &mut Commands,
                     shape: &Shape,
                     style: &Style,
                     transform: Transform,
                     shadow: bool| {
        let mut entity = commands.spawn((
            Mesh3d(cache.get_or_build_handle(shape, &mut meshes)),
            MeshMaterial3d(pool.get_or_add(style, &mut materials)),
            transform,
        ));
        if !shadow {
            entity.insert(NotShadowCaster);
        }
    };

    // Ground.
    piece(
        &mut commands,
        &Shape::cuboid(Vec3::new(18.0, 0.2, 9.0), 0.0, d),
        &Style::shaded(Color::srgb(0.62, 0.78, 0.58), 0.6).with_perceptual_roughness(0.95),
        Transform::from_xyz(0.0, -0.1, 0.0),
        true,
    );

    // Tower: lathe body with a flared top plus a cone roof, one mesh.
    let tower = Shape::compound(vec![
        ShapePart::new(Shape::lathe(
            vec![
                Vec2::new(0.9, 0.0),
                Vec2::new(0.8, 2.2),
                Vec2::new(0.8, 2.6),
                Vec2::new(1.1, 2.9),
                Vec2::new(1.1, 3.2),
            ],
            24,
            d,
        )),
        ShapePart::new(Shape::rounded_cone(1.3, 1.4, 0.45, d))
            .with_transform(Transform::from_xyz(0.0, 3.9, 0.0))
            .with_tint(Color::srgb(1.0, 0.7, 0.7)),
    ]);
    piece(
        &mut commands,
        &tower,
        &Style::shaded(Color::srgb(0.88, 0.86, 0.82), 0.45)
            .with_perceptual_roughness(0.9)
            .with_stripes(Vec3::Y, 2.5, 0.08),
        Transform::from_xyz(-5.0, 0.0, 0.0),
        true,
    );

    // Wall with an arched door opening, extruded from one polygon with a hole.
    let wall = Shape::extruded_polygon_with_holes(
        rounded_rectangle(Vec2::new(4.0, 3.4), 0.15, 3),
        vec![arch(1.2, 2.0, 10).into_iter().map(|p| p - Vec2::new(0.0, 0.7)).collect()],
        0.5,
        0.35,
        d,
    );
    piece(
        &mut commands,
        &wall,
        &Style::shaded(Color::srgb(0.93, 0.85, 0.72), 0.5)
            .with_perceptual_roughness(0.85)
            .with_noise(4.0, 0.18),
        Transform::from_xyz(-0.8, 1.7, 0.0),
        true,
    );

    // Cobbles: flat extruded blobs with tinted tops, lying on the ground.
    for i in 0..5_u64 {
        let outline = blob(0.32, 0.25, i);
        let n = outline.len();
        let tint = 0.8 + 0.04 * i as f32;
        let cobble = Shape::extruded_polygon_colored(
            outline,
            vec![[tint, tint * 0.95, tint * 0.9, 1.0]; n],
            0.18,
            1.0,
            d,
        );
        piece(
            &mut commands,
            &cobble,
            &Style::shaded(Color::srgb(0.7, 0.7, 0.72), 0.5)
                .with_direction_space(DirectionSpace::Local)
                .with_direction(Vec3::Z)
                .with_perceptual_roughness(1.0),
            Transform::from_xyz(-1.8 + 0.75 * i as f32, 0.09, 2.6)
                .with_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
            true,
        );
    }

    // Glowing torch (lathe + emissive) and a metallic gem.
    let torch = Shape::lathe(
        vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(0.08, 0.0),
            Vec2::new(0.07, 1.0),
            Vec2::new(0.18, 1.15),
            Vec2::new(0.0, 1.5),
        ],
        16,
        d,
    );
    piece(
        &mut commands,
        &torch,
        &Style::shaded(Color::srgb(1.0, 0.65, 0.25), 0.7)
            .with_emissive(LinearRgba::rgb(3.0, 1.2, 0.3)),
        Transform::from_xyz(3.2, 0.0, 1.0),
        true,
    );
    piece(
        &mut commands,
        &Shape::rounded_pyramid(6, 0.5, 0.7, 0.5, ShapeDetail::High),
        &Style::shaded(Color::srgb(0.6, 0.8, 1.0), 0.3)
            .with_metallic(1.0)
            .with_perceptual_roughness(0.2)
            .with_rim(Color::srgb(0.4, 0.7, 1.0), 3.0),
        Transform::from_xyz(4.4, 0.7, 1.0).with_rotation(Quat::from_rotation_x(std::f32::consts::PI)),
        true,
    );

    // Terrain: two neighbouring heightfield chunks over the same hills. The right
    // chunk is sampled more coarsely, so its edge does not meet the left one's and
    // the skirts have a real crack to hide.
    let hills = |x: f32, z: f32| 0.55 + 0.45 * (x * 1.3).sin() * (z * 1.1).cos();
    let chunk = |centre: Vec2, samples: u32, key: u64| {
        let extent = Vec2::splat(2.5);
        let heights: Vec<f32> = (0..samples)
            .flat_map(|j| (0..samples).map(move |i| (i, j)))
            .map(|(i, j)| {
                let u = i as f32 / (samples - 1) as f32 - 0.5;
                let v = j as f32 / (samples - 1) as f32 - 0.5;
                hills(centre.x + u * extent.x, centre.y + v * extent.y)
            })
            .collect();
        Shape::heightfield(UVec2::splat(samples), extent, heights.into(), key, d)
    };
    // Grass, rock and snow by world height: the bands follow the terrain, so the
    // two chunks agree on colour at their shared edge.
    let terrain_style = Style::opaque(Color::WHITE)
        .with_height_bands(&[
            (-1.0, Color::srgb(0.30, 0.55, 0.25)),
            (0.7, Color::srgb(0.50, 0.46, 0.42)),
            (0.98, Color::srgb(0.95, 0.96, 1.0)),
        ])
        .with_perceptual_roughness(0.95);
    for (centre, samples, key) in [
        (Vec2::new(5.0, -2.0), 17, 1),
        (Vec2::new(7.5, -2.0), 6, 2),
    ] {
        piece(
            &mut commands,
            &chunk(centre, samples, key),
            &terrain_style,
            Transform::from_xyz(centre.x, 0.0, centre.y),
            true,
        );
    }

    // Ghost preview: translucent and casting no shadow.
    let ghost_style = Style::shaded(Color::srgb(0.5, 0.9, 0.7), 0.5).with_alpha(0.45);
    piece(
        &mut commands,
        &Shape::cuboid(Vec3::new(1.4, 1.4, 1.4), 0.2, d),
        &ghost_style,
        Transform::from_xyz(3.0, 0.7, 3.2),
        ghost_style.casts_shadows(),
    );
}
