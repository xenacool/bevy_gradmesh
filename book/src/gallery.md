# Gallery

Screenshots for every example are packaged in a zip archive attached to each [GitHub Release](https://github.com/xenacool/bevy_gradmesh/releases).

| Example | Description |
|---|---|
| `hello_sphere` | Minimal getting-started sphere with a dual-directional gradient matching the book's starter snippet |
| `shapes` | Procedural primitives at every detail level, shaded by dual-directional gradients |
| `character_creator` | Character parts built from gradient-shaded primitives |
| `styles` | Colour modes, blend modes (opaque, transparent, additive, multiplicative, screen, dither), dash patterns and style presets |
| `style_lerp` | Animated `Style::lerp` transitions between matte and metallic gradients, noise and stripes, and directional and radial rim styles |
| `mix_spaces` | The same gradient in each `MixSpace` (OKLAB, OKLCH, linear RGB, sRGB) |
| `pbr_effects` | Transparency, emissive glow, metallic surfaces, and combined effects |
| `gradient_2d` | `Camera2d` and `Mesh2d` with gradient cards, a radial disc, a pulsing star, a dashed ring and a polygon with a hole |
| `sdf_2d` | One shared unit quad mesh: filled discs, ellipses, rounded rectangles, rims, animated partial arcs, radial and angular gradients |
| `sdf_dash_gallery` | Dashed rim outlines comparing `Off`, `Tiling`, `EndToEnd` and count-based snap modes |
| `sdf_lerp` | `FlatShape::sdf_2d()` conversion and `SdfMaterial2d::lerp` morphing discs, rounded rectangles, dashed rims, stars and polygons |
| `immediate` | Per-frame animated particles, push/pop state stacking, billboarding, and line drawing |
| `retained` | Persistent shapes updated each frame via handles without recreating entities |
| `instanced` | 210 spheres rendered in a single GPU batch via `BatchRecorder` |
| `lod_and_culling` | Distance-based `LodConfig` Bayer dither cross-fades, frustum culling, and live `InstancedStats` HUD |
| `standalone` | Gradient shapes drawn without any PBR material or lights: one shader pipeline, one bind group |
| `curves` | Bezier curves of every degree and a 3D helix rendered as smooth tubes by the standalone pipeline |
| `animated_curves_3d` | Animated 3D `bezier_curve` and `catmull_rom_3d` tubes rebuilt per frame alongside open and closed `polyline3d` tubes |
| `polyline` | Flat 2D polylines (open, closed, thick and thin) with gradient colors, drawn as `Mesh2d` without PBR |
| `sweep` | Profiles swept along 3D paths: a wall with an arched gate and windows, a kerbed road, and a tapering pipe on a helix |
| `road` | Banked road using `Shape::sweep_frames` with width and roll channels |
| `spline_relax_and_twist` | `CatmullRomSpline::relaxed`, arc-length `sample_at_distance` tracking, and `SweepOptions::with_twist` + `with_caps` |
| `composed_shapes` | Compound tower, wall with arched opening, extruded cobbles, emissive torch, metallic rim gem, ghost preview and heightfield terrain |
| `terrain` | Heightfield chunks with shared edge normals, relative height bands, and hole edges |
| `deformation` | Vertex shader animations: wobble, pulse, and morph effects |
| `appearance` | Per-instance colour animation: static, pulse, cycle, and gradient shift |
| `arbitrary_shapes` | Custom polygons with per-vertex colors, star shapes, annular rings, pie-slice wedges, and the side profiles of tapered cylinders |
| `lerp` | Six animated shape morphs: disc↔star, rectangle↔ring, polygon↔disc rim, ring↔disc, disc rim↔rectangle, wedge↔disc |
| `lerp_2d` | 2D flat shape morphing via `Shape::lerp` and `Mesh2d` without PBR |

Run any example with `cargo run --example <name>`. The examples listed under Features in [Getting Started](./getting-started.md) also run with `--no-default-features`.
