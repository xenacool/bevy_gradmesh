//! Arbitrary shapes example - demonstrates custom polygon, star, ring, wedge, and tapered cylinder.
//!
//! Shows shapes that go beyond the standard primitives: user-defined polygons
//! with per-vertex colors, star shapes, annular rings and pie-slice wedges, all
//! as unlit `Mesh2d` shapes with `GradMeshMaterial2d`, so no PBR is needed.
//!
//! Dropped from the lit 3D version: the ground plane and the shading from
//! lights (each shape keeps a dark-to-light gradient instead). Tapered
//! cylinders are solids with no flat form (`Shape::as_flat` refuses them), so
//! they appear as their side profile, a trapezoid polygon.
//!
//! Run with: `cargo run --example arbitrary_shapes`

mod common;

use bevy::mesh::Mesh2d;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::sprite_render::MeshMaterial2d;
use bevy_gradmesh::{GradMeshMaterial2d, GradMeshPlugin, Shape, ShapeDetail, Style};
use common::screenshot::AutoScreenshotPlugin;


/// Run this example. Called by the WASM runner.
pub fn run(app: &mut App) {
    app
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "BevyGradmesh Arbitrary Shapes".into(),
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
            AutoScreenshotPlugin::new("screenshot_arbitrary_shapes.png")
                .with_title("Arbitrary Shapes")
                .with_description("Custom polygons, stars, rings, wedges, and tapered cylinders"),
        )
        .insert_resource(ClearColor(Color::srgb(0.08, 0.08, 0.1)))
        .add_systems(Startup, setup)
        .add_systems(PostUpdate, position_labels)
        ;
}

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "BevyGradmesh Arbitrary Shapes".into(),
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
            AutoScreenshotPlugin::new("screenshot_arbitrary_shapes.png")
                .with_title("Arbitrary Shapes")
                .with_description("Custom polygons, stars, rings, wedges, and tapered cylinders"),
        )
        .insert_resource(ClearColor(Color::srgb(0.08, 0.08, 0.1)))
        .add_systems(Startup, setup)
        .add_systems(PostUpdate, position_labels)
        .run();
}

/// Window pixels per scene unit; the shape sizes below are in scene units.
const UNIT: f32 = 100.0;
/// Scene height of the middle of the four rows.
const CENTER_Y: f32 = 3.4;

#[derive(Component)]
struct WorldLabel(Vec3);

/// Side profile of a tapered cylinder: a trapezoid of the given top and bottom radius.
fn profile(top: f32, bottom: f32, height: f32) -> Shape {
    let half = height * 0.5;
    Shape::arbitrary_polygon(
        vec![
            Vec2::new(-bottom, -half),
            Vec2::new(bottom, -half),
            Vec2::new(top, half),
            Vec2::new(-top, half),
        ],
        ShapeDetail::Minimal,
    )
}

/// A puzzle piece as one single polygon: a knob on the right and a blank on
/// top, with every edge cut into many points that are almost, but not quite,
/// on a straight line.
fn jigsaw_piece(size: f32) -> Shape {
    let half = size * 0.5;
    let corners = [
        Vec2::new(-half, -half),
        Vec2::new(half, -half),
        Vec2::new(half, half),
        Vec2::new(-half, half),
    ];
    let knobs = [0.0, 1.0, -1.0, 0.0];
    let mut outline = Vec::new();
    for e in 0..4 {
        let (a, b) = (corners[e], corners[(e + 1) % 4]);
        let dir = (b - a) / size;
        let outward = Vec2::new(dir.y, -dir.x);
        let knob = knobs[e];

        // The edge in (along, outward) coordinates, both as a fraction of `size`.
        let mut edge: Vec<Vec2> = Vec::new();
        let mut k = 0usize;
        let run = |edge: &mut Vec<Vec2>, from: f32, to: f32, k: &mut usize| {
            let steps = ((24.0 * (to - from)).ceil() as usize).max(1);
            for i in 0..steps {
                let wobble = 1e-5 * ((*k + e * 7) as f32 * 12.9898).sin();
                *k += 1;
                edge.push(Vec2::new(from + (to - from) * i as f32 / steps as f32, wobble));
            }
        };
        if knob == 0.0 {
            run(&mut edge, 0.0, 1.0, &mut k);
        } else {
            run(&mut edge, 0.0, 0.42, &mut k);
            edge.push(Vec2::new(0.42, 0.0));
            edge.push(Vec2::new(0.42, knob * 0.055));
            let (from, to) = (216.87f32.to_radians(), -36.87f32.to_radians());
            for j in 0..=24 {
                let angle = from + (to - from) * j as f32 / 24.0;
                edge.push(Vec2::new(
                    0.5 + 0.1 * angle.cos(),
                    knob * (0.17 + 0.1 * angle.sin()),
                ));
            }
            edge.push(Vec2::new(0.58, knob * 0.055));
            edge.push(Vec2::new(0.58, 0.0));
            run(&mut edge, 0.58, 1.0, &mut k);
        }
        outline.extend(edge.iter().map(|p| a + dir * (p.x * size) + outward * (p.y * size)));
    }
    Shape::arbitrary_polygon(outline, ShapeDetail::Minimal)
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial2d>>,
) {
    commands.spawn((Camera2d, Msaa::Off));

    let mut piece = |commands: &mut Commands, shape: Shape, color: [f32; 3], at: Vec2| {
        let [r, g, b] = color;
        let style = Style::dual_directional(
            Color::srgb(r * 0.6, g * 0.6, b * 0.6),
            Color::srgb(r, g, b),
            Vec3::Y,
        );
        let flat = shape.as_flat().expect("every shape in this example is flat");
        commands.spawn((
            Mesh2d(meshes.add(flat.build_2d())),
            MeshMaterial2d(materials.add(style.build_2d())),
            Transform::from_translation(Vec3::new(
                at.x * UNIT,
                (at.y - CENTER_Y) * UNIT,
                0.0,
            ))
            .with_scale(Vec3::new(UNIT, UNIT, 1.0)),
        ));
    };

    // --- Row 1: Star shapes (y = 0.5) ---
    piece(
        &mut commands,
        Shape::star(5, 0.45, 0.18, ShapeDetail::High),
        [0.95, 0.85, 0.1],
        Vec2::new(-3.0, 0.5),
    );
    spawn_label(&mut commands, "5-Point Star", Vec3::new(-3.0, 1.2, 0.0));

    piece(
        &mut commands,
        Shape::star(8, 0.45, 0.22, ShapeDetail::High),
        [0.9, 0.3, 0.3],
        Vec2::new(0.0, 0.5),
    );
    spawn_label(&mut commands, "8-Point Star", Vec3::new(0.0, 1.2, 0.0));

    piece(
        &mut commands,
        Shape::star(6, 0.45, 0.08, ShapeDetail::High),
        [0.3, 0.85, 0.4],
        Vec2::new(3.0, 0.5),
    );
    spawn_label(&mut commands, "Deep Star", Vec3::new(3.0, 1.2, 0.0));

    // --- Row 2: Ring & Wedge (y = 2.2) ---
    piece(
        &mut commands,
        Shape::ring(0.45, 0.25, ShapeDetail::High),
        [0.3, 0.4, 0.95],
        Vec2::new(-3.0, 2.2),
    );
    spawn_label(&mut commands, "Ring", Vec3::new(-3.0, 2.9, 0.0));

    piece(
        &mut commands,
        Shape::ring(0.45, 0.38, ShapeDetail::High),
        [0.95, 0.45, 0.6],
        Vec2::new(0.0, 2.2),
    );
    spawn_label(&mut commands, "Thin Ring", Vec3::new(0.0, 2.9, 0.0));

    // Wedge (quarter circle)
    piece(
        &mut commands,
        Shape::wedge(0.45, std::f32::consts::FRAC_PI_2, ShapeDetail::High),
        [1.0, 0.6, 0.2],
        Vec2::new(3.0, 2.2),
    );
    spawn_label(&mut commands, "Quarter Wedge", Vec3::new(3.0, 2.9, 0.0));

    // --- Row 3: Tapered cylinders, as side profiles (y = 3.9) ---
    // Cup shape (narrow top, wide bottom)
    piece(
        &mut commands,
        profile(0.2, 0.4, 0.7),
        [0.4, 0.7, 0.9],
        Vec2::new(-3.0, 3.9),
    );
    spawn_label(&mut commands, "Cup", Vec3::new(-3.0, 4.6, 0.0));

    // Inverted cone shape (wide top, narrow bottom)
    piece(
        &mut commands,
        profile(0.4, 0.15, 0.7),
        [0.7, 0.4, 0.9],
        Vec2::new(0.0, 3.9),
    );
    spawn_label(&mut commands, "Inverted", Vec3::new(0.0, 4.6, 0.0));

    // Barrel shape (narrow middle via two tapered cylinders)
    piece(
        &mut commands,
        profile(0.35, 0.25, 0.35),
        [0.6, 0.4, 0.2],
        Vec2::new(3.0, 4.0),
    );
    piece(
        &mut commands,
        profile(0.25, 0.35, 0.35),
        [0.6, 0.4, 0.2],
        Vec2::new(3.0, 3.65),
    );
    spawn_label(&mut commands, "Barrel", Vec3::new(3.0, 4.6, 0.0));

    // --- Row 4: Arbitrary Polygons (y = 5.6) ---
    // Arrow shape
    let arrow_verts = vec![
        Vec2::new(0.0, 0.5),   // tip
        Vec2::new(0.15, 0.15), // inner right
        Vec2::new(0.4, 0.15),  // right wing
        Vec2::new(0.2, 0.0),   // inner notch
        Vec2::new(0.2, -0.5),  // tail
        Vec2::new(-0.2, -0.5), // tail
        Vec2::new(-0.2, 0.0),  // inner notch
        Vec2::new(-0.4, 0.15), // left wing
        Vec2::new(-0.15, 0.15),// inner left
    ];
    piece(
        &mut commands,
        Shape::arbitrary_polygon(arrow_verts, ShapeDetail::Minimal),
        [0.9, 0.5, 0.2],
        Vec2::new(-4.5, 5.6),
    );
    spawn_label(&mut commands, "Arrow", Vec3::new(-4.5, 6.3, 0.0));

    // Colored diamond with per-vertex colors
    let diamond_verts = vec![
        Vec2::new(0.0, 0.5),
        Vec2::new(0.4, 0.0),
        Vec2::new(0.0, -0.5),
        Vec2::new(-0.4, 0.0),
    ];
    let diamond_colors = vec![
        [1.0, 0.2, 0.2, 1.0], // red top
        [0.2, 1.0, 0.2, 1.0], // green right
        [0.2, 0.2, 1.0, 1.0], // blue bottom
        [1.0, 1.0, 0.2, 1.0], // yellow left
    ];
    piece(
        &mut commands,
        Shape::arbitrary_polygon_colored(diamond_verts, diamond_colors, ShapeDetail::Minimal),
        [0.8, 0.8, 0.8],
        Vec2::new(-1.5, 5.6),
    );
    spawn_label(&mut commands, "Colored Diamond", Vec3::new(-1.5, 6.3, 0.0));

    // L-shape polygon
    let l_verts = vec![
        Vec2::new(-0.3, 0.5),
        Vec2::new(0.0, 0.5),
        Vec2::new(0.0, 0.1),
        Vec2::new(0.4, 0.1),
        Vec2::new(0.4, -0.5),
        Vec2::new(-0.3, -0.5),
    ];
    piece(
        &mut commands,
        Shape::arbitrary_polygon(l_verts, ShapeDetail::Minimal),
        [0.3, 0.9, 0.6],
        Vec2::new(1.5, 5.6),
    );
    spawn_label(&mut commands, "L-Shape", Vec3::new(1.5, 6.3, 0.0));

    // Jigsaw piece: one concave polygon with hundreds of near-collinear points
    piece(
        &mut commands,
        jigsaw_piece(0.8),
        [0.85, 0.65, 0.95],
        Vec2::new(4.5, 5.6),
    );
    spawn_label(&mut commands, "Jigsaw Piece", Vec3::new(4.5, 6.3, 0.0));
}

fn spawn_label(commands: &mut Commands, text: &str, position: Vec3) {
    commands.spawn((
        Text::new(text.to_string()),
        TextFont {
            font_size: FontSize::Px(13.0),
            ..default()
        },
        TextColor(Color::srgb(0.88, 0.91, 0.98)),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(0.0),
            top: Val::Px(0.0),
            width: Val::Px(120.0),
            height: Val::Px(22.0),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        },
        WorldLabel(Vec3::new(position.x, position.y - CENTER_Y, position.z) * UNIT),
    ));
}

fn position_labels(
    camera: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    mut labels: Query<(&WorldLabel, &mut Node)>,
) {
    let Ok((camera, camera_transform)) = camera.single() else {
        return;
    };
    for (label, mut node) in &mut labels {
        if let Ok(viewport) = camera.world_to_viewport(camera_transform, label.0) {
            node.left = Val::Px(viewport.x - 60.0);
            node.top = Val::Px(viewport.y - 11.0);
            node.display = Display::Flex;
        } else {
            node.display = Display::None;
        }
    }
}
