# BevyGradmesh

[![Crates.io](https://img.shields.io/crates/v/bevy_gradmesh.svg)](https://crates.io/crates/bevy_gradmesh)
[![docs.rs](https://img.shields.io/docsrs/bevy_gradmesh)](https://docs.rs/bevy_gradmesh)
[![Bevy](https://img.shields.io/badge/Bevy-0.20-232326)](https://bevyengine.org/)
[![Rust](https://img.shields.io/badge/rust-1.87%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/crates/l/bevy_gradmesh.svg)](LICENSE-MIT)
[![Last commit](https://img.shields.io/github/last-commit/xenacool/bevy_gradmesh)](https://github.com/xenacool/bevy_gradmesh/commits)

Procedural shapes, mesh generation and gradient materials for [Bevy](https://bevyengine.org/).

BevyGradmesh provides a clean API for generating procedural 2D3D primitives with proper geometry, detail levels, and gradient shading inspired by Freya Holmér's work.

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
need no lights and no `StandardMaterial`. These examples run without `pbr`: `mix_spaces`, `curves`,
`polyline`, `arbitrary_shapes`, `gradient_2d`, `sdf_2d`, `sdf_dash_gallery`, `lerp_2d` and `standalone`; the rest need it.

## Documentation

Read the [book](https://xenacool.github.io/bevy_gradmesh/) for a guided tour of the API,
or the [API reference on docs.rs](https://docs.rs/bevy_gradmesh).

## Screenshots

All example screenshots are packaged in a zip archive attached to each [GitHub Release](https://github.com/xenacool/bevy_gradmesh/releases). Download the latest `screenshots-v*.zip` for still PNGs and animated GIFs.


## License

Dual-licensed under MIT or Apache-2.0.

