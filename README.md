# BevyGradmesh

[![Crates.io](https://img.shields.io/crates/v/bevy_gradmesh.svg)](https://crates.io/crates/bevy_gradmesh)

Procedural mesh generation and gradient materials for [Bevy](https://bevyengine.org/).

BevyGradmesh provides a clean API for generating procedural 3D primitives with proper geometry, detail levels, and gradient shading — inspired by Freya Holmer's work.

## Features

### Shapes

- **3D primitives**: Sphere, ellipsoid, cuboid (with rounded edges), cylinder, cone, capsule, hemisphere, torus (with arc support), regular prism, pyramid, tapered cylinder
- **2D primitives**: Disc, polygon, ellipse, rectangle, star (N-pointed), ring/annulus, wedge/pie-slice
- **Advanced geometry**: Arbitrary polygons (convex & concave), 3D polylines as tubes, Bézier curves (linear, quadratic, cubic)
- **Extruded polygons and holes**: `Shape::extruded_polygon` (rounded edges, per-vertex colours) and `Shape::arbitrary_polygon_with_holes` / `extruded_polygon_with_holes` for walls with openings
- **Outline helpers**: `arch`, `rounded_rectangle` and `blob` return counter-clockwise outlines centred on the origin
- **Compound shapes**: `Shape::compound` merges several shapes with transforms and tints into one mesh and one draw call
- **Lathe**: `Shape::lathe` revolves a `(radius, height)` profile around Y
- **Flat curves**: `Shape::polyline_2d` and `Shape::bezier_2d` with thickness
- **Sweep**: `Shape::sweep` extrudes a 2D profile along a 3D path with optional openings (arched gates, windows), length-based tints, and per-segment scaling
- **Rounded roofs and gems**: `Shape::rounded_cone` and `Shape::rounded_pyramid` (and `regular_prism`) take a `roundness`
- **Bounds**: `Shape::bounds`, `Shape::bounds_2d` and `Shape::is_round` read a shape's size and kind
- **Detail levels**: Control mesh resolution from minimal (8 segments) to extreme (32 segments)

### 2D

- **True `Mesh2d` output**: `Shape::build_2d()` returns flat XY-plane geometry facing +Z with gradient UVs — no billboarding and no 3D camera
- **`GradMeshMaterial2d`**: an unlit `Material2d` with the same dual-directional and dual-radial gradients, mix spaces, alpha and dashes; `Style::build_2d()` builds it from a `Style`
- **SDF 2D shapes**: `SdfMaterial2d` evaluates disc, ellipse, rounded rectangle and partial arcs per-pixel on a shared unit quad — rims, edge anti-aliasing, and animated arcs change only material floats each frame
- **SDF dash modes**: `Dash2d` with `DashSnap::Off`, `Tiling` and `EndToEnd` (length-based or count-based), plus sliding offset animation
- **SDF gradient variants**: radial gradients fitted to bounds and angular (conic) gradients with sweep control
- **Scale and layering**: `Transform` scale resizes shapes on the XY plane and `Transform.translation.z` orders them
- **No extra setup**: `GradMeshPlugin` registers the material when Bevy's 2D mesh renderer is present (add `GradMesh2dPlugin` otherwise)

### Rendering Modes

- **Direct mode**: Call `Shape::build()` and spawn `Mesh3d` entities manually
- **Immediate mode**: Per-frame draw/dispose with `ImmediateRecorder` — supports push/pop state stacking, billboarding, and relative transforms
- **Retained mode**: `RetainedWorld` + `GradMeshHandle` for persistent shapes updated per-frame
- **Batched GPU instancing**: `BatchRecorder` groups instances by mesh key, blend mode, and render state into single `draw_indexed` calls — thousands of shapes in far fewer draw calls

### Materials & Shading

- **Gradient materials**: Dual-directional and dual-radial shading for 3D-looking gradients
- **Selectable mix space**: OKLAB (default), OKLCH, linear RGB or sRGB for gradients and CPU-side `Style::lerp` / `Shape::lerp` blending
- **Blend modes**: Opaque, transparent, additive, multiplicative, screen, dither, and dashed outlines
- **PBR integration**: Full compatibility with Bevy's physically-based rendering (roughness, metallicity, emissive); `Style::with_perceptual_roughness`, `with_metallic` and `with_emissive` are written to the material and blended by `Style::lerp`
- **Style helpers**: `Style::shaded(color, dark_factor)` for a one-colour gradient, `Style::with_alpha` for ghost previews, and `PartialEq` on `Style`, `Dash` and `AppearanceSnapshot`
- **Gradient direction space**: world space by default; `Style::with_direction_space(DirectionSpace::Local)` makes the gradient follow the entity's rotation (for example Y-up meshes placed in a Z-up scene)
- **Deduplication**: `MeshCache::get_or_build_handle` returns one `Handle<Mesh>` per equal `Shape`, and `GradMeshMaterialPool::get_or_add` one material per equal `Style`
- **Surface detail**: `Style::with_noise` and `Style::with_stripes` vary the tint in local space; `Style::with_rim` adds a view-dependent rim glow
- **Vertex tints**: per-vertex colours (compound part tints, `extruded_polygon_colored`) multiply the gradient
- **Custom shader passthrough**: Attach any Bevy shader via `BlendMode::Custom`

### Performance

- **Batched GPU instancing**: Single draw call per batch with per-instance transforms and properties on the GPU
- **Dirty tracking**: Only modified batches re-upload GPU data each frame
- **Mesh caching**: `MeshCache` resource avoids regenerating identical geometry
- **Automatic LOD**: Distance-based level-of-detail with Bayer dither cross-fade eliminating pop-in
- **Frustum culling**: Per-instance AABB culling against camera frustum planes
- **Statistics API**: `InstancedStats` tracks instances, batches, cache hits, GPU upload bytes, and culling ratios

### Custom Pipeline (Mobile-Friendly)

- **Standalone pipeline**: `StandaloneRenderPlugin` and `render_standalone_batches()` bypass Bevy's PBR system entirely, using a simple vertex+fragment shader with no `StandardMaterial`, no lights, and no PBR imports
- **Zero descriptor pool pressure**: All material variation via per-instance vertex attributes — one pipeline, one bind group layout, one descriptor set
- **Faster compilation**: One shader pipeline instead of N specializations
- **`GradMeshMobilePlugin`**: lightweight plugin variant for minimal dependency setups

### Animation

- **Vertex shader deformation**: Wobble, pulse, and morph effects without CPU mesh regeneration
- **Appearance snapshots**: Per-instance color animation (pulse, cycle, gradient shift) resolved in the shader — no per-frame CPU updates

## What is Dual-Directional Shading?

Dual-directional shading is a technique that colors a surface with a smooth gradient between two colors along a given direction, instead of using a single flat color. One side of the shape takes on the "dark" color and the opposite side takes on the "light" color, with a smooth transition in between. This mimics the look of a single light source illuminating the shape, but is computed entirely in the shader — no actual lights are needed.

It is commonly used in stylized games to give characters a painted, cel-shaded appearance: skin tones shift from warm highlights to cool shadows, and hair shows a bright rim where light catches it. The gradient direction can be any axis (e.g., `Vec3::UP` for top-down lighting), and the two colors are interpolated in [OKLAB](https://bottosson.github.io/posts/oklab/) by default (a perceptually uniform color space that avoids muddy midpoints), or in OKLCH, linear RGB or sRGB via `MixSpace`.

### Key Terms

- **Gradient direction**: A 3D vector defining which axis the gradient runs along. Vertices on one side of the shape get the dark color; vertices on the opposite side get the light color.
- **OKLAB**: A color space designed so that equal distances correspond to equal perceived color differences. Gradients in OKLAB look smoother and more natural than linear RGB interpolation. OKLCH is the same space in polar form (lightness, chroma, hue), so blends sweep around the hue wheel.
- **Blend mode**: How a shape's color combines with what's behind it. Common modes include *opaque* (solid, no transparency), *transparent* (alpha-blended), *additive* (colors add together, creating a glow effect), *multiplicative* (colors multiply, darkening the result), *screen* (the opposite of multiply, lightening the result), and *dither* (uses a pattern to simulate intermediate alpha levels).
- **PBR (Physically-Based Rendering)**: A shading model that simulates how light interacts with real-world materials. Parameters include *roughness* (how smooth or rough a surface is), *metallicity* (whether a surface is metallic or dielectric), and *emissive* (light the material emits on its own, like a glowing screen).
- **Detail level**: Controls the number of vertices and triangles in a generated mesh. Higher detail produces smoother curves but uses more GPU memory and processing.

## Compatibility

| bevy_gradmesh | Bevy | Rust |
|---------------|------|------|
| 0.2.x         | 0.20 | 1.87+|

## Quick Start

```toml
[dependencies]
bevy_gradmesh = "0.2"
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

## Screenshots

### Shape Showcase

All procedural primitives with dual-directional gradient shading:

![Shape Showcase](screenshots/screenshot_shapes.png)

### Character Creator

Character parts built from gradient-shaded primitives:

![Character Creator](screenshots/screenshot_character_creator.png)

### Style Variants

Opaque, transparent, additive, multiplicative, screen, dither, and dashed blend modes:

![Style Variants](screenshots/screenshot_styles.png)

### Immediate Mode

Per-frame animated particles, push/pop state stacking, billboarding, and line drawing:

![Immediate Mode](screenshots/screenshot_immediate.png)

### Curves

Spline and parametric curves rendered as smooth tubes:

![Curves](screenshots/screenshot_curves.png)

### Polyline

Polylines with joint styles, width variation, gradient colors, and animated reveal:

![Polyline](screenshots/screenshot_polyline.png)

### PBR Effects

Transparency, emissive glow, metallic surfaces, and combined effects:

![PBR Effects](screenshots/screenshot_pbr_effects.png)

### 2D Gradients

`Camera2d` and `Mesh2d` with `GradMeshMaterial2d`: gradient cards, a translucent dragged card layered by z, a radial disc, a pulsing star, a dashed ring and a polygon with a hole:

![2D Gradients](screenshots/screenshot_gradient_2d.png)

### SDF 2D

One shared unit quad mesh: filled discs, ellipses, rounded rectangles, rims, animated partial arcs, radial and angular gradients — all shape changes are material floats, no mesh regeneration:

![SDF 2D](screenshots/screenshot_sdf_2d.png)

### SDF Dash Gallery

Dashed rim outlines on discs, ellipses, rounded rectangles and partial arcs, comparing `Off`, `Tiling`, `EndToEnd` and count-based snap modes at rim widths from 1.5 to 18 pixels:

![SDF Dash Gallery](screenshots/screenshot_sdf_dash_gallery.png)

### Sweep

Profiles swept along 3D paths: a wall following a baked spline loop with an arched gate and windows, a kerbed road on an S-curve with length tint, and a tapering pipe winding up a helix:

![Sweep](screenshots/screenshot_sweep.png)

### Instanced Rendering

500 spheres rendered in a single GPU batch via `BatchRecorder`:

![Instanced Rendering](screenshots/screenshot_instanced.png)

### Standalone Pipeline

Gradient shapes drawn without any PBR material or lights — one shader pipeline, one bind group:

![Standalone Pipeline](screenshots/screenshot_standalone.png)

### Composed Shapes

Compound tower (lathe plus roof), a wall with an arched opening, extruded cobbles, an emissive torch, a metallic gem and a translucent ghost piece that casts no shadow:

![Composed Shapes](screenshots/screenshot_composed_shapes.png)

### Deformation

Vertex shader animations: wobble, pulse, and morph effects compared against static originals:

![Deformation](screenshots/screenshot_deformation.png)

### Retained World

Persistent shapes updated each frame via handles without recreating entities:

![Retained World](screenshots/screenshot_retained.png)

### Arbitrary Shapes

Custom polygons with per-vertex colors, star shapes, annular rings, pie-slice wedges, and tapered cylinders:

![Arbitrary Shapes](screenshots/screenshot_arbitrary_shapes.png)

### Mix Spaces
Mix color spaces:
![Mix Space](screenshots/screenshot_mix_spaces.png)

### Capturing Screenshots

To automatically capture screenshots of all examples, run:

```bash
make screenshots
```

This will render each example for a few frames, save a PNG to the `screenshots/` directory, and exit. On headless systems, `xvfb-run` is auto-detected and used automatically.

## License

Dual-licensed under MIT or Apache-2.0.
