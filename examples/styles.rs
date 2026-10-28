//! Style showcase example — demonstrates all blend modes, color modes, and dash patterns.
//!
//! Uses the `Style` builder API to create materials, showcasing varied shapes
//! (spheres, discs, cuboids) against a backdrop plane.
//!
//! Run with: `cargo run --example styles`

mod common;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{
    Style, Dash, GradMeshMaterial, GradMeshPlugin, Shape, ShapeDetail, opaque,
};
use common::screenshot::AutoScreenshotPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "BevyGradmesh Style Showcase".into(),
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
            AutoScreenshotPlugin::new("screenshot_styles.png")
                .with_title("Style Showcase")
                .with_description("Blend modes, color modes, dash patterns, and material presets")
        )
        .insert_resource(ClearColor(Color::srgb(0.15, 0.15, 0.18)))
        .add_systems(Startup, setup)
        .add_systems(PostUpdate, position_labels)
        .run();
}

#[derive(Component)]
struct WorldLabel(Vec3);

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
) {
    // Camera — angled top-down view to see all rows clearly
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 13.0, 12.0).looking_at(Vec3::new(0.0, 0.6, 0.0), Vec3::Y),
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
        color: Color::srgb(0.6, 0.6, 0.7),
        brightness: 300.0,
        affects_lightmapped_meshes: false,
    });

    // Ground plane
    let backdrop = meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(28.0)));
    commands.spawn((
        Mesh3d(backdrop),
        MeshMaterial3d(materials.add(opaque(0.12, 0.12, 0.15))),
        Transform::from_translation(Vec3::new(0.0, -0.01, 0.0)),
    ));

    // Pre-built meshes for varied shapes
    let sphere = meshes.add(Shape::sphere(0.42, ShapeDetail::High).build());
    let disc = meshes.add(Shape::disc(0.42, ShapeDetail::High).build());
    let cuboid = meshes.add(Shape::cuboid(Vec3::splat(0.6), 0.0, ShapeDetail::High).build());

    let row_spacing = 3.2;
    let col_spacing = 2.6;
    let origin_y = 0.6;

    let z0 = -row_spacing * 1.5;
    let z1 = -row_spacing * 0.5;
    let z2 = row_spacing * 0.5;
    let z3 = row_spacing * 1.5;

    // ─── Row 1: Color Modes ───
    let r1_xs = [-col_spacing * 2.2, -col_spacing * 1.1, 0.0, col_spacing * 1.1, col_spacing * 2.2];

    // Single (opaque)
    add_shape(
        &mut commands, &mut materials, &sphere,
        Style::opaque(Color::srgb(0.9, 0.3, 0.3)),
        Vec3::new(r1_xs[0], origin_y, z0),
        "Single",
    );

    // Dual Directional
    add_shape(
        &mut commands, &mut materials, &sphere,
        Style::dual_directional(
            Color::srgb(0.5, 0.2, 0.15),
            Color::srgb(0.96, 0.8, 0.69),
            Vec3::Y,
        ),
        Vec3::new(r1_xs[1], origin_y, z0),
        "Dual Dir (RGB)",
    );

    // Dual Directional OKLAB
    add_shape(
        &mut commands, &mut materials, &sphere,
        Style::dual_directional_oklab(
            Color::srgb(0.5, 0.2, 0.15),
            Color::srgb(0.96, 0.8, 0.69),
            Vec3::Y,
        ),
        Vec3::new(r1_xs[2], origin_y, z0),
        "Dual Dir (OKLAB)",
    );

    // Dual Radial
    add_shape(
        &mut commands, &mut materials, &disc,
        Style::dual_radial(
            Color::srgb(0.1, 0.25, 0.5),
            Color::srgb(0.3, 0.6, 0.9),
        ),
        Vec3::new(r1_xs[3], origin_y, z0),
        "Dual Radial",
    );

    // Dual Radial OKLAB
    add_shape(
        &mut commands, &mut materials, &disc,
        Style::dual_radial(
            Color::srgb(0.1, 0.25, 0.5),
            Color::srgb(0.3, 0.6, 0.9),
        ).with_oklab(),
        Vec3::new(r1_xs[4], origin_y, z0),
        "Dual Radial OKLAB",
    );

    // ─── Row 2: Blend Modes ───
    let r2_xs = [-col_spacing * 2.5, -col_spacing * 1.5, -col_spacing * 0.5, col_spacing * 0.5, col_spacing * 1.5, col_spacing * 2.5];

    // Opaque
    add_shape(
        &mut commands, &mut materials, &cuboid,
        Style::opaque(Color::srgb(0.3, 0.8, 0.3)),
        Vec3::new(r2_xs[0], origin_y, z1),
        "Opaque",
    );

    // Transparent
    add_shape(
        &mut commands, &mut materials, &sphere,
        Style::transparent(Color::srgba(1.0, 0.4, 0.4, 0.4)),
        Vec3::new(r2_xs[1], origin_y, z1),
        "Transparent",
    );

    // Additive
    let additive_mat = {
        let mut mat = Style::opaque(Color::srgb(1.0, 0.9, 0.2)).build();
        mat.base.emissive = Color::srgb(1.0, 0.9, 0.2).to_linear().into();
        materials.add(mat)
    };
    commands.spawn((
        Mesh3d(sphere.clone()),
        MeshMaterial3d(additive_mat),
        Transform::from_translation(Vec3::new(r2_xs[2], origin_y, z1)),
    ));
    spawn_label(&mut commands, "Additive", Vec3::new(r2_xs[2], origin_y + 0.8, z1));

    // Multiplicative
    add_shape(
        &mut commands, &mut materials, &cuboid,
        Style::multiplicative(Color::srgb(0.2, 0.4, 0.8)),
        Vec3::new(r2_xs[3], origin_y, z1),
        "Multiplicative",
    );

    // Screen
    add_shape(
        &mut commands, &mut materials, &sphere,
        Style::screen(Color::srgb(0.9, 0.3, 0.6)),
        Vec3::new(r2_xs[4], origin_y, z1),
        "Screen",
    );

    // Dither
    add_shape(
        &mut commands, &mut materials, &disc,
        Style::dither(Color::srgba(0.3, 0.9, 0.5, 0.5)),
        Vec3::new(r2_xs[5], origin_y, z1),
        "Dither",
    );

    // ─── Row 3: Dash Patterns ───
    let r3_xs = r2_xs;

    // Normal dash on disc
    add_shape(
        &mut commands, &mut materials, &disc,
        Style::opaque(Color::srgb(0.9, 0.5, 0.2)).with_dash(Dash::fixed(8, 0.3)),
        Vec3::new(r3_xs[0], origin_y, z2),
        "Dash Normal",
    );

    // Chevron dash on disc
    add_shape(
        &mut commands, &mut materials, &disc,
        Style::opaque(Color::srgb(0.3, 0.7, 0.9)).with_dash(Dash::chevron(6, 0.4)),
        Vec3::new(r3_xs[1], origin_y, z2),
        "Dash Chevron",
    );

    // Rounded dash on disc
    add_shape(
        &mut commands, &mut materials, &disc,
        Style::opaque(Color::srgb(0.8, 0.3, 0.7)).with_dash(Dash::rounded(10, 0.2)),
        Vec3::new(r3_xs[2], origin_y, z2),
        "Dash Rounded",
    );

    // Normal dash on rect
    add_shape(
        &mut commands, &mut materials, &cuboid,
        Style::opaque(Color::srgb(0.4, 0.9, 0.4)).with_dash(Dash::fixed(6, 0.35)),
        Vec3::new(r3_xs[3], origin_y, z2),
        "Rect Dash",
    );

    // Chevron dash on rect
    add_shape(
        &mut commands, &mut materials, &cuboid,
        Style::opaque(Color::srgb(0.9, 0.7, 0.3)).with_dash(Dash::chevron(8, 0.25)),
        Vec3::new(r3_xs[4], origin_y, z2),
        "Rect Chevron",
    );

    // Rounded dash on rect
    add_shape(
        &mut commands, &mut materials, &cuboid,
        Style::opaque(Color::srgb(0.5, 0.5, 0.95)).with_dash(Dash::rounded(12, 0.15)),
        Vec3::new(r3_xs[5], origin_y, z2),
        "Rect Rounded",
    );

    // ─── Row 4: Material Presets ───
    let r4_xs = r2_xs;

    // Skin tone
    add_shape(
        &mut commands, &mut materials, &sphere,
        Style::dual_directional_up(
            Color::srgb(0.74, 0.62, 0.53),
            Color::srgb(0.96, 0.80, 0.69),
        ),
        Vec3::new(r4_xs[0], origin_y, z3),
        "Skin",
    );

    // Hair
    add_shape(
        &mut commands, &mut materials, &sphere,
        Style::dual_directional_up(
            Color::srgb(0.1, 0.07, 0.05),
            Color::srgb(0.2, 0.14, 0.1),
        ),
        Vec3::new(r4_xs[1], origin_y, z3),
        "Hair",
    );

    // Eye iris
    add_shape(
        &mut commands, &mut materials, &disc,
        Style::dual_radial(
            Color::srgb(0.1, 0.25, 0.5),
            Color::srgb(0.18, 0.38, 0.68),
        ),
        Vec3::new(r4_xs[2], origin_y, z3),
        "Eye Iris",
    );

    // Blush
    add_shape(
        &mut commands, &mut materials, &sphere,
        Style::transparent(Color::srgba(1.0, 0.5, 0.5, 0.35)),
        Vec3::new(r4_xs[3], origin_y, z3),
        "Blush",
    );

    // Metal
    add_shape(
        &mut commands, &mut materials, &cuboid,
        Style::dual_directional_up(
            Color::srgb(0.3, 0.32, 0.35),
            Color::srgb(0.85, 0.87, 0.9),
        ),
        Vec3::new(r4_xs[4], origin_y, z3),
        "Metal",
    );

    // Neon glow
    add_shape(
        &mut commands, &mut materials, &sphere,
        Style::dual_directional_oklab(
            Color::srgb(0.0, 0.8, 0.2),
            Color::srgb(0.3, 1.0, 0.5),
            Vec3::Y,
        ),
        Vec3::new(r4_xs[5], origin_y, z3),
        "Neon",
    );
}

fn add_shape(
    commands: &mut Commands,
    materials: &mut Assets<GradMeshMaterial>,
    mesh: &Handle<bevy::render::mesh::Mesh>,
    style: Style,
    position: Vec3,
    label: &str,
) {
    let mat = style.build_material(materials);
    commands.spawn((
        Mesh3d(mesh.clone()),
        MeshMaterial3d(mat),
        Transform::from_translation(position),
    ));
    spawn_label(commands, label, position + Vec3::new(0.0, 0.8, 0.0));
}

fn spawn_label(commands: &mut Commands, text: &str, position: Vec3) {
    commands.spawn((
        Text::new(text.to_string()),
        TextFont { font_size: FontSize::Px(12.5), ..default() },
        TextColor(Color::srgb(0.88, 0.91, 0.98)),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(0.0),
            top: Val::Px(0.0),
            width: Val::Px(120.0),
            height: Val::Px(20.0),
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
            node.left = Val::Px(viewport.x - 60.0);
            node.top = Val::Px(viewport.y - 10.0);
            node.display = Display::Flex;
        } else {
            node.display = Display::None;
        }
    }
}
