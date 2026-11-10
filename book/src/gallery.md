# Gallery

Screenshots for every example are packaged in a zip archive attached to each [GitHub Release](https://github.com/xenacool/bevy_gradmesh/releases).

| Example | Description |
|---|---|
| `shapes` | Procedural primitives at every detail level, shaded by dual-directional gradients |
| `character_creator` | Character parts built from gradient-shaded primitives |
| `styles` | Colour modes, blend modes (opaque, transparent, additive, multiplicative, screen, dither), dash patterns and style presets |
| `mix_spaces` | The same gradient in each `MixSpace` (OKLAB, OKLCH, linear RGB, sRGB) |
| `pbr_effects` | Transparency, emissive glow, metallic surfaces, and combined effects |
| `gradient_2d` | `Camera2d` and `Mesh2d` with gradient cards, a radial disc, a pulsing star, a dashed ring and a polygon with a hole |
| `sdf_2d` | One shared unit quad mesh: filled discs, ellipses, rounded rectangles, rims, animated partial arcs, radial and angular gradients |
| `sdf_dash_gallery` | Dashed rim outlines comparing `Off`, `Tiling`, `EndToEnd` and count-based snap modes |
| `immediate` | Per-frame animated particles, push/pop state stacking, billboarding, and line drawing |
| `retained` | Persistent shapes updated each frame via handles without recreating entities |
| `instanced` | 500 spheres rendered in a single GPU batch via `BatchRecorder` |
| `standalone` | Gradient shapes drawn without any PBR material or lights: one shader pipeline, one bind group |
| `curves` | Bezier curves of every degree and a 3D helix rendered as smooth tubes by the standalone pipeline |
| `polyline` | Flat 2D polylines (open, closed, thick and thin) with gradient colors, drawn as `Mesh2d` without PBR |
| `sweep` | Profiles swept along 3D paths: a wall with an arched gate and windows, a kerbed road, and a tapering pipe on a helix |
| `road` | Banked road using `Shape::sweep_frames` with width and roll channels |
| `composed_shapes` | Compound tower, wall with arched opening, extruded cobbles, torch, gem and heightfield terrain, drawn unlit by the standalone pipeline |
| `terrain` | Heightfield chunks with shared edge normals, relative height bands, and hole edges |
| `deformation` | Vertex shader animations: wobble, pulse, and morph effects |
| `appearance` | Per-instance colour animation: static, pulse, cycle, and gradient shift |
| `arbitrary_shapes` | Custom polygons with per-vertex colors, star shapes, annular rings, pie-slice wedges, and the side profiles of tapered cylinders |

Run any example with `cargo run --example <name>`. The examples listed under Features in [Getting Started](./getting-started.md) also run with `--no-default-features`.
