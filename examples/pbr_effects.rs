//! PBR effects showcase — transparency, emission, roughness, metallicity, and more.
//!
//! Demonstrates how BevyGradmesh shapes interact with Bevy's full PBR material
//! system, including transparency, emissive glow, metallic surfaces, and
//! various roughness levels.
//!
//! Run with: `cargo run --example pbr_effects`

mod common;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{
    GradMeshMaterial, GradMeshPlugin, Shape, ShapeDetail, additive, dual_directional,
    dual_radial_alpha, opaque, screen, transparent,
};
use common::screenshot::AutoScreenshotPlugin;

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "BevyGradmesh PBR Effects".into(),
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
            AutoScreenshotPlugin::new("screenshot_pbr_effects.png")
                .with_title("PBR Effects")
                .with_description("Transparency, emission, roughness, and metallicity"),
        )
        .add_systems(Startup, setup)
        .add_systems(PostUpdate, (rotate_shapes, position_labels).chain())
        .run();
}

#[derive(Component)]
struct Spin(Quat);

#[derive(Component)]
struct WorldLabel(Vec3);

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
) {
    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 3.0, 8.8).looking_at(Vec3::new(0.0, 3.05, 0.0), Vec3::Y),
    ));

    // Lighting — warm key light for dramatic PBR
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(1.0, 0.95, 0.85),
            illuminance: 20_000.0,
            ..default()
        },
        Transform::from_xyz(8.0, 5.0, 2.0).looking_at(Vec3::new(0.0, 4.0, 0.0), Vec3::Y),
    ));

    // Cool fill light from the opposite side
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(0.6, 0.7, 1.0),
            illuminance: 5_000.0,
            ..default()
        },
        Transform::from_xyz(-6.0, 3.0, -3.0).looking_at(Vec3::new(0.0, 3.0, 0.0), Vec3::Y),
    ));

    // Ambient fill
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.5, 0.5, 0.6),
        brightness: 200.0,
        affects_lightmapped_meshes: false,
    });

    // Dark backdrop
    let backdrop_mesh = meshes.add(Plane3d::new(Vec3::Z, Vec2::new(24.0, 16.0)));
    commands.spawn((
        Mesh3d(backdrop_mesh),
        MeshMaterial3d(materials.add(opaque(0.04, 0.04, 0.06))),
        Transform::from_translation(Vec3::new(0.0, 3.5, -0.8)),
        Name::new("Backdrop"),
    ));

    // ─── Row 1: Transparency variants (y = 0.5) ───
    let y1 = 0.5;
    let xs1 = [-4.0, -2.0, 0.0, 2.0, 4.0];

    // Semi-transparent sphere
    add_shape(
        &mut commands,
        &mut meshes,
        &materials.add(transparent(0.6, 0.8, 1.0, 0.5)),
        Shape::sphere(0.4, ShapeDetail::High),
        Vec3::new(xs1[0], y1, 0.0),
        "Transparent\n(alpha 0.4)",
    );

    // Glass-like sphere (high transparency, low roughness)
    {
        let mut mat = transparent(0.8, 0.95, 1.0, 0.2);
        mat.base.perceptual_roughness = 0.05;
        add_shape(
            &mut commands,
            &mut meshes,
            &materials.add(mat),
            Shape::sphere(0.4, ShapeDetail::High),
            Vec3::new(xs1[1], y1, 0.0),
            "Glass-like\n(alpha 0.2)",
        );
    }

    // Gradient transparent
    add_shape(
        &mut commands,
        &mut meshes,
        &materials.add(transparent(1.0, 0.68, 0.8, 0.55)),
        Shape::cuboid(Vec3::new(0.55, 0.55, 0.55), 0.2, ShapeDetail::High),
        Vec3::new(xs1[2], y1, 0.0),
        "Transparent\nrounded cube",
    );

    // Dual radial with alpha (inner glow fading to transparent)
    add_shape(
        &mut commands,
        &mut meshes,
        &materials.add(dual_radial_alpha([1.0, 0.8, 0.3, 0.8], [0.2, 0.1, 0.5, 0.0])),
        Shape::disc(0.4, ShapeDetail::High),
        Vec3::new(xs1[3], y1, 0.0),
        "Radial fade\nto transparent",
    );

    // Dither blend
    add_shape(
        &mut commands,
        &mut meshes,
        &materials.add(bevy_gradmesh::dither(0.85, 0.7, 1.0, 0.5)),
        Shape::disc(0.4, ShapeDetail::High),
        Vec3::new(xs1[4], y1, 0.0),
        "Dither blend\n(alpha 0.5)",
    );

    // ─── Row 2: Emission / Additive (y = 2.2) ───
    let y2 = 2.2;
    let xs2 = [-4.0, -2.0, 0.0, 2.0, 4.0];

    // Additive glow sphere
    add_shape(
        &mut commands,
        &mut meshes,
        &materials.add(additive(1.0, 0.3, 0.2)),
        Shape::sphere(0.35, ShapeDetail::High),
        Vec3::new(xs2[0], y2, 0.0),
        "Additive\nglow",
    );

    // Additive cyan
    add_shape(
        &mut commands,
        &mut meshes,
        &materials.add(additive(0.2, 1.0, 0.9)),
        Shape::sphere(0.35, ShapeDetail::High),
        Vec3::new(xs2[1], y2, 0.0),
        "Additive\ncyan",
    );

    // Screen blend
    add_shape(
        &mut commands,
        &mut meshes,
        &materials.add(screen(1.0, 0.88, 0.45)),
        Shape::cuboid(Vec3::new(0.5, 0.5, 0.5), 0.15, ShapeDetail::High),
        Vec3::new(xs2[2], y2, 0.0),
        "Screen blend",
    );

    // Emissive gradient (manual: dual_directional + emissive)
    {
        let mut mat = dual_directional([0.2, 0.0, 0.4], [1.0, 0.3, 0.1], [0.0, 1.0, 0.0]);
        mat.base.emissive = bevy::color::LinearRgba::rgb(0.8, 0.2, 0.5) * 2.0;
        add_shape(
            &mut commands,
            &mut meshes,
            &materials.add(mat),
            Shape::capsule(0.2, 0.55, ShapeDetail::High),
            Vec3::new(xs2[3], y2, 0.0),
            "Emissive\ngradient",
        );
    }

    // Strong emissive torus
    {
        let mut mat = dual_directional([0.1, 0.5, 0.2], [0.9, 1.0, 0.3], [1.0, 0.0, 0.0]);
        mat.base.emissive = bevy::color::LinearRgba::rgb(0.3, 1.0, 0.4) * 3.0;
        add_shape_rotated(
            &mut commands,
            &mut meshes,
            &materials.add(mat),
            Shape::torus(0.3, 0.1, 0.0, std::f32::consts::TAU),
            Vec3::new(xs2[4], y2, 0.0),
            Quat::from_euler(EulerRot::XYZ, 0.5, 0.4, 0.0),
            "Emissive\ntorus",
        );
    }

    // ─── Row 3: Roughness & Metallicity (y = 3.9) ───
    let y3 = 3.9;
    let xs3 = [-4.0, -2.0, 0.0, 2.0, 4.0];

    // Polished metal (low roughness, high metallicity)
    {
        let mut mat = opaque(0.7, 0.75, 0.85);
        mat.base.perceptual_roughness = 0.05;
        mat.base.metallic = 1.0;
        add_shape(
            &mut commands,
            &mut meshes,
            &materials.add(mat),
            Shape::sphere(0.4, ShapeDetail::High),
            Vec3::new(xs3[0], y3, 0.0),
            "Polished\nmetal",
        );
    }

    // Matte (high roughness, no metal)
    {
        let mut mat = opaque(0.97, 0.72, 0.62);
        mat.base.perceptual_roughness = 1.0;
        mat.base.metallic = 0.0;
        add_shape(
            &mut commands,
            &mut meshes,
            &materials.add(mat),
            Shape::sphere(0.4, ShapeDetail::High),
            Vec3::new(xs3[1], y3, 0.0),
            "Matte\n(rough 1.0)",
        );
    }

    // Gold-like (metallic, warm color, medium roughness)
    {
        let mut mat = opaque(0.9, 0.7, 0.2);
        mat.base.perceptual_roughness = 0.3;
        mat.base.metallic = 1.0;
        add_shape(
            &mut commands,
            &mut meshes,
            &materials.add(mat),
            Shape::cuboid(Vec3::new(0.5, 0.5, 0.5), 0.1, ShapeDetail::High),
            Vec3::new(xs3[2], y3, 0.0),
            "Gold-like\nmetal",
        );
    }

    // Gradient on metallic surface
    {
        let mut mat = dual_directional([0.2, 0.2, 0.8], [0.8, 0.2, 0.2], [0.0, 1.0, 0.0]);
        mat.base.perceptual_roughness = 0.15;
        mat.base.metallic = 0.8;
        add_shape(
            &mut commands,
            &mut meshes,
            &materials.add(mat),
            Shape::cylinder(0.28, 0.55, ShapeDetail::High),
            Vec3::new(xs3[3], y3, 0.0),
            "Gradient on\nmetallic",
        );
    }

    // Gradient on matte surface
    {
        let mut mat = dual_directional([0.1, 0.6, 0.3], [0.9, 0.8, 0.2], [0.0, 1.0, 0.0]);
        mat.base.perceptual_roughness = 0.9;
        mat.base.metallic = 0.0;
        add_shape(
            &mut commands,
            &mut meshes,
            &materials.add(mat),
            Shape::cylinder(0.28, 0.55, ShapeDetail::High),
            Vec3::new(xs3[4], y3, 0.0),
            "Gradient on\nmatte",
        );
    };

    // ─── Row 4: Combined effects (y = 5.6) ───
    let y4 = 5.6;
    let xs4 = [-4.0, -2.0, 0.0, 2.0, 4.0];

    // Transparent + emissive (neon glass)
    {
        let mut mat = transparent(0.2, 1.0, 0.5, 0.3);
        mat.base.perceptual_roughness = 0.05;
        mat.base.emissive = bevy::color::LinearRgba::rgb(0.1, 1.0, 0.4) * 2.0;
        add_shape(
            &mut commands,
            &mut meshes,
            &materials.add(mat),
            Shape::capsule(0.2, 0.55, ShapeDetail::High),
            Vec3::new(xs4[0], y4, 0.0),
            "Neon glass\n(trans + emissive)",
        );
    }

    // Metallic + emissive (glowing metal)
    {
        let mut mat = opaque(0.3, 0.3, 0.5);
        mat.base.perceptual_roughness = 0.1;
        mat.base.metallic = 1.0;
        mat.base.emissive = bevy::color::LinearRgba::rgb(0.5, 0.3, 1.0) * 1.0;
        add_shape(
            &mut commands,
            &mut meshes,
            &materials.add(mat),
            Shape::pyramid(4, 0.35, 0.6, ShapeDetail::High),
            Vec3::new(xs4[1], y4, 0.0),
            "Glowing\nmetal",
        );
    }

    // Dual radial + transparent + emissive (orb)
    {
        let mut mat = dual_radial_alpha([1.0, 0.9, 0.5, 0.9], [0.3, 0.1, 0.8, 0.1]);
        mat.base.perceptual_roughness = 0.0;
        mat.base.emissive = bevy::color::LinearRgba::rgb(1.0, 0.8, 0.3) * 1.2;
        add_shape(
            &mut commands,
            &mut meshes,
            &materials.add(mat),
            Shape::sphere(0.4, ShapeDetail::High),
            Vec3::new(xs4[2], y4, 0.0),
            "Energy orb\n(radial + emissive)",
        );
    }

    // Dashed outline: a ring (disc rim) with disc-style perimeter dashes
    {
        let mut mat = bevy_gradmesh::dashed(1.0, 0.72, 0.84, 10, 0.4, 0.0, 0, 0.0, 1);
        mat.base.perceptual_roughness = 0.3;
        mat.base.metallic = 0.6;
        add_shape_rotated(
            &mut commands,
            &mut meshes,
            &materials.add(mat),
            Shape::disc_rim(0.45, 0.14, ShapeDetail::High),
            Vec3::new(xs4[3], y4, 0.0),
            Quat::IDENTITY,
            "Dashed\noutline",
        );
    }

    // Multiplicative blend
    add_shape(
        &mut commands,
        &mut meshes,
        &materials.add(bevy_gradmesh::multiplicative(0.5, 0.2, 0.8)),
        Shape::hemisphere(0.35, true, ShapeDetail::High),
        Vec3::new(xs4[4], y4, 0.0),
        "Multiplicative\nblend",
    );

    // A small square pad behind every shape so transparency, additive blends
    // and emissive colour have a nearby surface to read against.
    let pad_mesh = meshes.add(Plane3d::new(Vec3::Z, Vec2::splat(0.62)));
    let pad_tones = [
        [0.20, 0.17, 0.26],
        [0.16, 0.20, 0.26],
        [0.24, 0.17, 0.21],
        [0.17, 0.22, 0.22],
    ];
    for (row, y) in [y1, y2, y3, y4].into_iter().enumerate() {
        let tone = pad_tones[row % pad_tones.len()];
        let pad_material = materials.add(opaque(tone[0], tone[1], tone[2]));
        for x in xs1 {
            commands.spawn((
                Mesh3d(pad_mesh.clone()),
                MeshMaterial3d(pad_material.clone()),
                Transform::from_translation(Vec3::new(x, y, -0.45)),
                Name::new("Pad"),
            ));
        }
    }
}

fn add_shape(
    commands: &mut Commands,
    meshes: &mut Assets<bevy::render::mesh::Mesh>,
    material: &Handle<GradMeshMaterial>,
    shape: Shape,
    position: Vec3,
    label: &str,
) {
    let tilt = Quat::from_euler(EulerRot::XYZ, 0.35, 0.4, 0.0);
    add_shape_rotated(commands, meshes, material, shape, position, tilt, label);
}

fn add_shape_rotated(
    commands: &mut Commands,
    meshes: &mut Assets<bevy::render::mesh::Mesh>,
    material: &Handle<GradMeshMaterial>,
    shape: Shape,
    position: Vec3,
    rotation: Quat,
    label: &str,
) {
    let mesh_handle = meshes.add(shape.build());
    commands.spawn((
        Mesh3d(mesh_handle),
        MeshMaterial3d(material.clone()),
        Transform::from_translation(position).with_rotation(rotation),
        Spin(rotation),
    ));
    spawn_label(commands, label, position - Vec3::Y * 0.55);
}

fn spawn_label(commands: &mut Commands, text: &str, position: Vec3) {
    commands.spawn((
        Text::new(text.to_string()),
        TextFont {
            font_size: FontSize::Px(12.0),
            ..default()
        },
        TextColor(Color::srgb(0.88, 0.91, 0.98)),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(0.0),
            top: Val::Px(0.0),
            width: Val::Px(110.0),
            height: Val::Px(36.0),
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
    let Ok((camera, camera_transform)) = camera.single() else {
        return;
    };
    for (label, mut node) in &mut labels {
        if let Ok(viewport) = camera.world_to_viewport(camera_transform, label.0) {
            node.left = Val::Px(viewport.x - 55.0);
            node.top = Val::Px(viewport.y - 18.0);
            node.display = Display::Flex;
        } else {
            node.display = Display::None;
        }
    }
}

fn rotate_shapes(time: Res<Time>, mut query: Query<(&Spin, &mut Transform)>) {
    for (spin, mut transform) in &mut query {
        transform.rotation = Quat::from_rotation_y(time.elapsed_secs() * 0.25) * spin.0;
    }
}
