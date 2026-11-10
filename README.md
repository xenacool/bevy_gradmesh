# BevyGradmesh

[![Crates.io](https://img.shields.io/crates/v/bevy_gradmesh.svg)](https://crates.io/crates/bevy_gradmesh)
[![docs.rs](https://img.shields.io/docsrs/bevy_gradmesh)](https://docs.rs/bevy_gradmesh)
[![Bevy](https://img.shields.io/badge/Bevy-0.20-232326)](https://bevyengine.org/)
[![Rust](https://img.shields.io/badge/rust-1.87%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/crates/l/bevy_gradmesh.svg)](LICENSE-MIT)
[![Last commit](https://img.shields.io/github/last-commit/xenacool/bevy_gradmesh)](https://github.com/xenacool/bevy_gradmesh/commits)

Procedural shapes, mesh generation and gradient materials for [Bevy](https://bevyengine.org/).

BevyGradmesh provides a clean API for generating procedural 3D primitives with proper geometry, detail levels, and gradient shading inspired by Freya Holmér's work.

## Quick Start

```toml
[dependencies]
bevy_gradmesh = "0.3"
```

```rust
use bevy::prelude::*;
use bevy_gradmesh::{GradMeshMaterial, Shape, ShapeDetail, Style};

fn setup(mut meshes: ResMut<Assets<Mesh>>, mut materials: ResMut<Assets<GradMeshMaterial>>) {
    // Generate a smooth sphere mesh
    let mesh = meshes.add(Shape::sphere(1.0, ShapeDetail::Medium).build());

    // Create a dual-directional gradient material
    let material = Style::dual_directional(
        Color::srgb(0.6, 0.4, 0.3),   // dark color
        Color::srgb(0.96, 0.8, 0.69), // light color
        Vec3::Y,                      // light direction
    )
    .build_material(&mut materials);
}
```

### Features

The `pbr` feature (on by default) enables the lit `GradMeshMaterial` and the instanced PBR pipeline.
Disable it (`default-features = false`) for PBR-free builds: the `Mesh2d`, SDF 2D and standalone backends
need no lights and no `StandardMaterial`. These examples run without `pbr`: `shapes`, `styles`,
`mix_spaces`, `curves`, `polyline`, `composed_shapes`, `arbitrary_shapes`, `gradient_2d`, `sdf_2d`,
`sdf_dash_gallery` and `standalone`; the rest need it.

## Documentation

Read the [book](https://xenacool.github.io/bevy_gradmesh/) for a guided tour of the API,
or the [API reference on docs.rs](https://docs.rs/bevy_gradmesh).

## Screenshots

All example screenshots are packaged in a zip archive attached to each [GitHub Release](https://github.com/xenacool/bevy_gradmesh/releases). Download the latest `screenshots-v*.zip` for still PNGs and animated WebPs.

### Examples

**PBR (requires `--features pbr`):**

- `shapes` — Procedural primitives at every detail level, shaded by dual-directional gradients
- `styles` — Colour modes, blend modes, dash patterns and style presets
- `pbr_effects` — Transparency, emissive glow, metallic surfaces, and combined effects
- `character_creator` — Character parts built from gradient-shaded primitives
- `immediate` — Per-frame animated particles, push/pop state stacking, billboarding, and line drawing
- `retained` — Persistent shapes updated each frame via handles without recreating entities
- `instanced` — 500 spheres rendered in a single GPU batch via `BatchRecorder`
- `sweep` — Profiles swept along 3D paths: a wall with an arched gate and windows, a kerbed road, and a tapering pipe on a helix
- `road` — Banked road using `Shape::sweep_frames` with width and roll channels
- `terrain` — Heightfield chunks with shared edge normals, relative height bands, and hole edges
- `deformation` — Vertex shader animations: wobble, pulse, and morph effects
- `appearance` — Per-instance colour animation: static, pulse, cycle, and gradient shift

**PBR-free (run with `--no-default-features`):**

- `mix_spaces` — The same gradient in each `MixSpace` (OKLAB, OKLCH, linear RGB, sRGB)
- `curves` — Bezier curves of every degree and a 3D helix as smooth tubes
- `polyline` — Flat 2D polylines with gradient colors, drawn as `Mesh2d`
- `composed_shapes` — Compound tower, wall, cobbles, torch, gem and heightfield terrain
- `arbitrary_shapes` — Custom polygons, stars, rings, wedges, and tapered cylinder profiles
- `gradient_2d` — `Camera2d` and `Mesh2d` with gradient cards, a radial disc, a pulsing star, a dashed ring and a polygon with a hole
- `sdf_2d` — One shared unit quad mesh: filled discs, ellipses, rounded rectangles, rims, animated partial arcs, radial and angular gradients
- `sdf_dash_gallery` — Dashed rim outlines comparing `Off`, `Tiling`, `EndToEnd` and count-based snap modes
- `standalone` — Gradient shapes drawn without any PBR material or lights

Run any example with `cargo run --example <name>` (add `--no-default-features` for the PBR-free ones listed above).

## License

Dual-licensed under MIT or Apache-2.0.

