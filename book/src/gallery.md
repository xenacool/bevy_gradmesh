# Gallery

| Example | Description |
|---|---|
| `shapes` | All procedural primitives with dual-directional gradient shading |
| `character_creator` | Character parts built from gradient-shaded primitives |
| `styles` | Opaque, transparent, additive, multiplicative, screen, dither, and dashed blend modes |
| `mix_spaces` | The same gradient in each `MixSpace` (OKLAB, OKLCH, linear RGB, sRGB) |
| `pbr_effects` | Transparency, emissive glow, metallic surfaces, and combined effects |
| `gradient_2d` | `Camera2d` and `Mesh2d` with gradient cards, a radial disc, a pulsing star, a dashed ring and a polygon with a hole |
| `sdf_2d` | One shared unit quad mesh: filled discs, ellipses, rounded rectangles, rims, animated partial arcs, radial and angular gradients |
| `sdf_dash_gallery` | Dashed rim outlines comparing `Off`, `Tiling`, `EndToEnd` and count-based snap modes |
| `immediate` | Per-frame animated particles, push/pop state stacking, billboarding, and line drawing |
| `retained` | Persistent shapes updated each frame via handles without recreating entities |
| `instanced` | 500 spheres rendered in a single GPU batch via `BatchRecorder` |
| `standalone` | Gradient shapes drawn without any PBR material or lights — one shader pipeline, one bind group |
| `curves` | Spline and parametric curves rendered as smooth tubes |
| `polyline` | Polylines with joint styles, width variation, gradient colors, and animated reveal |
| `sweep` | Profiles swept along 3D paths: a wall with an arched gate and windows, a kerbed road, and a tapering pipe on a helix |
| `road` | Banked road using `Shape::sweep_frames` with width and roll channels |
| `composed_shapes` | Compound tower, wall with arched opening, extruded cobbles, emissive torch, metallic gem, and translucent ghost piece |
| `terrain` | Heightfield chunks with shared edge normals, relative height bands, and hole edges |
| `deformation` | Vertex shader animations: wobble, pulse, and morph effects |
| `appearance` | Per-instance colour animation: static, pulse, cycle, and gradient shift |
| `arbitrary_shapes` | Custom polygons with per-vertex colors, star shapes, annular rings, pie-slice wedges, and tapered cylinders |

Run any example with `cargo run --example <name>`.
