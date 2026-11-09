# BevyGradmesh

[![Crates.io](https://img.shields.io/crates/v/bevy_gradmesh.svg)](https://crates.io/crates/bevy_gradmesh)
[![docs.rs](https://img.shields.io/docsrs/bevy_gradmesh)](https://docs.rs/bevy_gradmesh)
[![Bevy](https://img.shields.io/badge/Bevy-0.20-232326)](https://bevyengine.org/)
[![Rust](https://img.shields.io/badge/rust-1.87%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/crates/l/bevy_gradmesh.svg)](LICENSE-MIT)
[![Last commit](https://img.shields.io/github/last-commit/xenacool/bevy_gradmesh)](https://github.com/xenacool/bevy_gradmesh/commits)

Procedural mesh generation and gradient materials for [Bevy](https://bevyengine.org/).

BevyGradmesh provides a clean API for generating procedural 3D primitives with proper geometry, detail levels, and gradient shading — inspired by Freya Holmér's work.

## Quick Start

```toml
[dependencies]
bevy_gradmesh = "0.3"
```

```rust
use bevy_gradmesh::Shape;
use bevy_gradmesh::Style;

// Generate a smooth sphere mesh
let mesh = Shape::sphere(1.0, ShapeDetail::Medium).build();

// Create a dual-directional gradient material
let material = Style::dual_directional(
    Color::srgb(0.6, 0.4, 0.3), // dark color
    Color::srgb(0.96, 0.8, 0.69), // light color
    Vec3::UP, // light direction
).build_material(&mut materials);
```

## Documentation

Read the [book](https://xenacool.github.io/bevy_gradmesh/) for a guided tour of the API,
or the [API reference on docs.rs](https://docs.rs/bevy_gradmesh).

## Gallery

| Screenshot | Example | Description |
|---|---|---|
| ![shapes](https://raw.githubusercontent.com/xenacool/bevy_gradmesh/main/screenshots/screenshot_shapes.png) | `shapes` | All procedural primitives with dual-directional gradient shading |
| ![character_creator](https://raw.githubusercontent.com/xenacool/bevy_gradmesh/main/screenshots/screenshot_character_creator.webp) | `character_creator` | Character parts built from gradient-shaded primitives |
| ![styles](https://raw.githubusercontent.com/xenacool/bevy_gradmesh/main/screenshots/screenshot_styles.png) | `styles` | Opaque, transparent, additive, multiplicative, screen, dither, and dashed blend modes |
| ![mix_spaces](https://raw.githubusercontent.com/xenacool/bevy_gradmesh/main/screenshots/screenshot_mix_spaces.png) | `mix_spaces` | The same gradient in each `MixSpace` (OKLAB, OKLCH, linear RGB, sRGB) |
| ![pbr_effects](https://raw.githubusercontent.com/xenacool/bevy_gradmesh/main/screenshots/screenshot_pbr_effects.png) | `pbr_effects` | Transparency, emissive glow, metallic surfaces, and combined effects |
| ![gradient_2d](https://raw.githubusercontent.com/xenacool/bevy_gradmesh/main/screenshots/screenshot_gradient_2d.webp) | `gradient_2d` | `Camera2d` and `Mesh2d` with gradient cards, a radial disc, a pulsing star, a dashed ring and a polygon with a hole |
| ![sdf_2d](https://raw.githubusercontent.com/xenacool/bevy_gradmesh/main/screenshots/screenshot_sdf_2d.webp) | `sdf_2d` | One shared unit quad mesh: filled discs, ellipses, rounded rectangles, rims, animated partial arcs, radial and angular gradients |
| ![sdf_dash_gallery](https://raw.githubusercontent.com/xenacool/bevy_gradmesh/main/screenshots/screenshot_sdf_dash_gallery.png) | `sdf_dash_gallery` | Dashed rim outlines comparing `Off`, `Tiling`, `EndToEnd` and count-based snap modes |
| ![immediate](https://raw.githubusercontent.com/xenacool/bevy_gradmesh/main/screenshots/screenshot_immediate.webp) | `immediate` | Per-frame animated particles, push/pop state stacking, billboarding, and line drawing |
| ![retained](https://raw.githubusercontent.com/xenacool/bevy_gradmesh/main/screenshots/screenshot_retained.webp) | `retained` | Persistent shapes updated each frame via handles without recreating entities |
| ![instanced](https://raw.githubusercontent.com/xenacool/bevy_gradmesh/main/screenshots/screenshot_instanced.webp) | `instanced` | 500 spheres rendered in a single GPU batch via `BatchRecorder` |
| ![standalone](https://raw.githubusercontent.com/xenacool/bevy_gradmesh/main/screenshots/screenshot_standalone.webp) | `standalone` | Gradient shapes drawn without any PBR material or lights — one shader pipeline, one bind group |
| ![curves](https://raw.githubusercontent.com/xenacool/bevy_gradmesh/main/screenshots/screenshot_curves.png) | `curves` | Spline and parametric curves rendered as smooth tubes |
| ![polyline](https://raw.githubusercontent.com/xenacool/bevy_gradmesh/main/screenshots/screenshot_polyline.webp) | `polyline` | Polylines with joint styles, width variation, gradient colors, and animated reveal |
| ![sweep](https://raw.githubusercontent.com/xenacool/bevy_gradmesh/main/screenshots/screenshot_sweep.png) | `sweep` | Profiles swept along 3D paths: a wall with an arched gate and windows, a kerbed road, and a tapering pipe on a helix |
| ![road](https://raw.githubusercontent.com/xenacool/bevy_gradmesh/main/screenshots/screenshot_road.png) | `road` | Banked road using `Shape::sweep_frames` with width and roll channels |
| ![composed_shapes](https://raw.githubusercontent.com/xenacool/bevy_gradmesh/main/screenshots/screenshot_composed_shapes.png) | `composed_shapes` | Compound tower, wall with arched opening, extruded cobbles, emissive torch, metallic gem, and translucent ghost piece |
| ![terrain](https://raw.githubusercontent.com/xenacool/bevy_gradmesh/main/screenshots/screenshot_terrain.png) | `terrain` | Heightfield chunks with shared edge normals, relative height bands, and hole edges |
| ![deformation](https://raw.githubusercontent.com/xenacool/bevy_gradmesh/main/screenshots/screenshot_deformation.webp) | `deformation` | Vertex shader animations: wobble, pulse, and morph effects |
| ![appearance](https://raw.githubusercontent.com/xenacool/bevy_gradmesh/main/screenshots/screenshot_appearance.webp) | `appearance` | Per-instance colour animation: static, pulse, cycle, and gradient shift |
| ![arbitrary_shapes](https://raw.githubusercontent.com/xenacool/bevy_gradmesh/main/screenshots/screenshot_arbitrary_shapes.png) | `arbitrary_shapes` | Custom polygons with per-vertex colors, star shapes, annular rings, pie-slice wedges, and tapered cylinders |

Run any example with `cargo run --example <name>`.

## License

Dual-licensed under MIT or Apache-2.0.

