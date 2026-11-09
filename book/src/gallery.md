# Gallery

| Screenshot | Example | Description |
|---|---|---|
| ![shapes](screenshots/screenshot_shapes.png) | `shapes` | All procedural primitives with dual-directional gradient shading |
| ![character_creator](screenshots/screenshot_character_creator.webp) | `character_creator` | Character parts built from gradient-shaded primitives |
| ![styles](screenshots/screenshot_styles.png) | `styles` | Opaque, transparent, additive, multiplicative, screen, dither, and dashed blend modes |
| ![mix_spaces](screenshots/screenshot_mix_spaces.png) | `mix_spaces` | The same gradient in each `MixSpace` (OKLAB, OKLCH, linear RGB, sRGB) |
| ![pbr_effects](screenshots/screenshot_pbr_effects.png) | `pbr_effects` | Transparency, emissive glow, metallic surfaces, and combined effects |
| ![gradient_2d](screenshots/screenshot_gradient_2d.webp) | `gradient_2d` | `Camera2d` and `Mesh2d` with gradient cards, a radial disc, a pulsing star, a dashed ring and a polygon with a hole |
| ![sdf_2d](screenshots/screenshot_sdf_2d.webp) | `sdf_2d` | One shared unit quad mesh: filled discs, ellipses, rounded rectangles, rims, animated partial arcs, radial and angular gradients |
| ![sdf_dash_gallery](screenshots/screenshot_sdf_dash_gallery.png) | `sdf_dash_gallery` | Dashed rim outlines comparing `Off`, `Tiling`, `EndToEnd` and count-based snap modes |
| ![immediate](screenshots/screenshot_immediate.webp) | `immediate` | Per-frame animated particles, push/pop state stacking, billboarding, and line drawing |
| ![retained](screenshots/screenshot_retained.webp) | `retained` | Persistent shapes updated each frame via handles without recreating entities |
| ![instanced](screenshots/screenshot_instanced.webp) | `instanced` | 500 spheres rendered in a single GPU batch via `BatchRecorder` |
| ![standalone](screenshots/screenshot_standalone.webp) | `standalone` | Gradient shapes drawn without any PBR material or lights — one shader pipeline, one bind group |
| ![curves](screenshots/screenshot_curves.png) | `curves` | Spline and parametric curves rendered as smooth tubes |
| ![polyline](screenshots/screenshot_polyline.webp) | `polyline` | Polylines with joint styles, width variation, gradient colors, and animated reveal |
| ![sweep](screenshots/screenshot_sweep.png) | `sweep` | Profiles swept along 3D paths: a wall with an arched gate and windows, a kerbed road, and a tapering pipe on a helix |
| ![road](screenshots/screenshot_road.png) | `road` | Banked road using `Shape::sweep_frames` with width and roll channels |
| ![composed_shapes](screenshots/screenshot_composed_shapes.png) | `composed_shapes` | Compound tower, wall with arched opening, extruded cobbles, emissive torch, metallic gem, and translucent ghost piece |
| ![terrain](screenshots/screenshot_terrain.png) | `terrain` | Heightfield chunks with shared edge normals, relative height bands, and hole edges |
| ![deformation](screenshots/screenshot_deformation.webp) | `deformation` | Vertex shader animations: wobble, pulse, and morph effects |
| ![appearance](screenshots/screenshot_appearance.webp) | `appearance` | Per-instance colour animation: static, pulse, cycle, and gradient shift |
| ![arbitrary_shapes](screenshots/screenshot_arbitrary_shapes.png) | `arbitrary_shapes` | Custom polygons with per-vertex colors, star shapes, annular rings, pie-slice wedges, and tapered cylinders |

Run any example with `cargo run --example <name>`.

`make screenshots` regenerates every image (headless with `xvfb-run` when no display is available). A PNG is captured for a still scene, and an animated WebP (lossy, downscaled to 480 px wide) for one that changes over time. Use `make screenshot-<example>` to refresh a single one, for example `make screenshot-appearance`.

`cargo test` checks the pictures: `tests/docs_screenshots.rs` fails if an example has no picture in the README or the book, or if an animation is not a multi-frame WebP, and `tests/mix_space_screenshot.rs` compares the Mix Spaces picture with reference pixel values.
