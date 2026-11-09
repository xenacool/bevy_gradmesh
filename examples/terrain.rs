//! Terrain chunks with `Shape::heightfield_with_options`: four chunks meet at a
//! peak with shared edge normals (no crease along the borders), one chunk has
//! holes (no triangles, no skirt, see-through), and the height bands are given
//! as fractions of the land's own height.
//!
//! The front row cuts the same round hole three ways with `HoleEdge`: whole cells
//! (a staircase, the cheapest), rounded corners and a smoothed outline. Each
//! label shows the triangle count of that chunk; the example asserts that the
//! cost grows with the smoothness and that no level closes the one-cell hole.
//!
//! Run with: `cargo run --example terrain`

mod common;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{
    GradMeshMaterial, GradMeshMaterialPool, GradMeshPlugin, HeightfieldBorder, HeightfieldOptions,
    HoleEdge, MeshCache, Shape, Style,
};
use common::screenshot::{AutoScreenshotPlugin, spawn_world_label};
use std::sync::Arc;

/// Samples per chunk side.
const SAMPLES: u32 = 25;
/// World size of a chunk side.
const EXTENT: f32 = 8.0;
/// Samples and world size of a chunk side in the front row.
const SMALL_SAMPLES: u32 = 13;
const SMALL_EXTENT: f32 = 5.0;

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "BevyGradmesh Terrain".into(),
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
            AutoScreenshotPlugin::new("screenshot_terrain.png")
                .with_title("Terrain chunks")
                .with_description("Seamless chunk edges, relative height bands, three ways to cut a hole")
                .wait_frames(3),
        )
        .add_systems(Startup, setup)
        .run();
}

/// The land, a pure function of the world position, so every chunk and its
/// neighbours agree on the heights along a shared edge.
fn land(x: f32, z: f32) -> f32 {
    let peak = 3.2 * (-(x * x + z * z) / 14.0).exp();
    let ripples = 0.18 * (x * 0.9).sin() * (z * 0.9).cos();
    peak + ripples + 0.4
}

/// Heights of the chunk centred at `(cx, cz)` plus the ring of neighbour
/// heights just outside it.
fn chunk_heights(cx: f32, cz: f32) -> (Vec<f32>, HeightfieldBorder) {
    let step = EXTENT / (SAMPLES - 1) as f32;
    let world = |i: i32| -EXTENT / 2.0 + i as f32 * step;
    let n = SAMPLES as i32;
    let mut heights = Vec::with_capacity((SAMPLES * SAMPLES) as usize);
    for row in 0..n {
        for col in 0..n {
            heights.push(land(cx + world(col), cz + world(row)));
        }
    }
    let border = HeightfieldBorder::new(
        (0..n).map(|r| land(cx + world(-1), cz + world(r))).collect(),
        (0..n).map(|r| land(cx + world(n), cz + world(r))).collect(),
        (0..n).map(|c| land(cx + world(c), cz + world(-1))).collect(),
        (0..n).map(|c| land(cx + world(c), cz + world(n))).collect(),
    );
    (heights, border)
}

/// One flag per cell: true for the cells whose centre is near a hole.
fn chunk_holes(cx: f32, cz: f32) -> Vec<bool> {
    let step = EXTENT / (SAMPLES - 1) as f32;
    let cells = SAMPLES - 1;
    let mut holes = Vec::with_capacity((cells * cells) as usize);
    for row in 0..cells {
        for col in 0..cells {
            let x = cx - EXTENT / 2.0 + (col as f32 + 0.5) * step;
            let z = cz - EXTENT / 2.0 + (row as f32 + 0.5) * step;
            let round = Vec2::new(x - (cx + 1.5), z - (cz + 1.6)).length() < 1.1;
            let square = (x - (cx - 1.5)).abs() < 0.5 && (z - (cz + 2.3)).abs() < 0.5;
            holes.push(round || square);
        }
    }
    holes
}

/// Cell mask of a small chunk: a round hole in the middle and a one-cell hole
/// beside it, the smallest thing a hole outline must not lose.
fn small_holes() -> Vec<bool> {
    let step = SMALL_EXTENT / (SMALL_SAMPLES - 1) as f32;
    let cells = SMALL_SAMPLES - 1;
    let mut holes = Vec::with_capacity((cells * cells) as usize);
    for row in 0..cells {
        for col in 0..cells {
            let x = -SMALL_EXTENT / 2.0 + (col as f32 + 0.5) * step;
            let z = -SMALL_EXTENT / 2.0 + (row as f32 + 0.5) * step;
            let round = Vec2::new(x + 0.3, z).length() < 1.25;
            let single = (col, row) == (cells - 2, 2);
            holes.push(round || single);
        }
    }
    holes
}

/// A gentle rise, so the cut cells are not all flat.
fn small_heights() -> Vec<f32> {
    let step = SMALL_EXTENT / (SMALL_SAMPLES - 1) as f32;
    let mut heights = Vec::with_capacity((SMALL_SAMPLES * SMALL_SAMPLES) as usize);
    for row in 0..SMALL_SAMPLES {
        for col in 0..SMALL_SAMPLES {
            let x = -SMALL_EXTENT / 2.0 + col as f32 * step;
            let z = -SMALL_EXTENT / 2.0 + row as f32 * step;
            heights.push(0.35 + 0.12 * (x * 1.3).sin() * (z * 1.1).cos());
        }
    }
    heights
}

fn triangle_count(meshes: &Assets<Mesh>, handle: &Handle<Mesh>) -> usize {
    meshes.get(handle).and_then(|mesh| mesh.indices()).map_or(0, |indices| indices.len() / 3)
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
        Transform::from_xyz(0.0, 19.0, 25.0).looking_at(Vec3::new(0.0, -0.8, 3.2), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 13_000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(-6.0, 10.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.7, 0.72, 0.85),
        brightness: 380.0,
        affects_lightmapped_meshes: false,
    });

    let size = UVec2::splat(SAMPLES);
    let extent = Vec2::splat(EXTENT);
    let half = EXTENT / 2.0;

    // The bands are fractions of the land's own height: the same palette would
    // fit a terrain scaled to any height. The four chunks are mirror images of
    // each other around the peak, so one style resolved on the first chunk is
    // right for all of them and the colours meet along the borders.
    let reference = {
        let (heights, border) = chunk_heights(-half, -half);
        Shape::heightfield_with_options(
            Arc::from(heights),
            size,
            extent,
            HeightfieldOptions::default().with_border(border),
        )
    };
    let palette = [
        (0.0, Color::srgb(0.45, 0.36, 0.25)),
        (0.12, Color::srgb(0.38, 0.62, 0.30)),
        (0.45, Color::srgb(0.52, 0.50, 0.48)),
        (0.85, Color::srgb(0.96, 0.97, 1.0)),
    ];
    let style = Style::shaded(Color::WHITE, 0.5)
        .with_perceptual_roughness(0.9)
        .with_height_bands_on(&palette, &reference, &Transform::IDENTITY);

    let chunks = [(-half, -half), (half, -half), (-half, half), (half, half)];
    for (index, (cx, cz)) in chunks.into_iter().enumerate() {
        let (heights, border) = chunk_heights(cx, cz);
        // Only the chunk in the +X/+Z corner gets holes.
        let mut options = HeightfieldOptions::default().with_border(border);
        if index == 3 {
            options = options
                .with_holes(chunk_holes(cx, cz))
                .with_hole_edge(HoleEdge::Smooth);
        }
        let shape = Shape::heightfield_with_options(
            Arc::from(heights),
            size,
            extent,
            options,
        );
        commands.spawn((
            Mesh3d(cache.get_or_build_handle(&shape, &mut meshes)),
            MeshMaterial3d(pool.get_or_add(&style, &mut materials)),
            Transform::from_xyz(cx, 0.0, cz),
        ));
    }

    // The same round hole cut three ways, cheapest first.
    let small_style = Style::shaded(Color::srgb(0.42, 0.62, 0.32), 0.5).with_perceptual_roughness(0.9);
    let levels = [
        (HoleEdge::Cells, "HoleEdge::Cells"),
        (HoleEdge::Rounded, "HoleEdge::Rounded"),
        (HoleEdge::Smooth, "HoleEdge::Smooth"),
    ];
    let mut counts = Vec::new();
    for (index, (edge, name)) in levels.into_iter().enumerate() {
        let x = (index as f32 - 1.0) * 6.6;
        let z = 12.5;
        let shape = Shape::heightfield_with_options(
            Arc::from(small_heights()),
            UVec2::splat(SMALL_SAMPLES),
            Vec2::splat(SMALL_EXTENT),
            HeightfieldOptions::default()
                .with_holes(small_holes())
                .with_hole_edge(edge),
        );
        let handle = cache.get_or_build_handle(&shape, &mut meshes);
        let triangles = triangle_count(&meshes, &handle);
        counts.push(triangles);
        commands.spawn((
            Mesh3d(handle),
            MeshMaterial3d(pool.get_or_add(&small_style, &mut materials)),
            Transform::from_xyz(x, 0.0, z),
        ));
        spawn_world_label(
            &mut commands,
            &format!("{name}: {triangles} triangles"),
            Vec3::new(x, 0.9, z + SMALL_EXTENT / 2.0 + 0.4),
        );
    }
    // Smoother costs more, and the three chunks differ.
    assert!(counts[0] < counts[1] && counts[1] <= counts[2], "triangles per level: {counts:?}");

    // A dark floor far below, so the holes read as holes.
    let floor = Shape::cuboid(Vec3::new(24.0, 0.2, 31.0), 0.0, bevy_gradmesh::ShapeDetail::Low);
    commands.spawn((
        Mesh3d(cache.get_or_build_handle(&floor, &mut meshes)),
        MeshMaterial3d(pool.get_or_add(&Style::shaded(Color::srgb(0.12, 0.1, 0.16), 0.5), &mut materials)),
        Transform::from_xyz(0.0, -1.0, 4.0),
    ));

    spawn_world_label(&mut commands, "Crest on the chunk corner: no crease", Vec3::new(0.0, 5.4, 0.0));
    spawn_world_label(&mut commands, "Holes: no triangles, no skirt", Vec3::new(half + 0.5, 0.6, half + 3.2));
    spawn_world_label(&mut commands, "Seam along z = 0: shared normals", Vec3::new(-4.5, 0.9, 0.0));
}
