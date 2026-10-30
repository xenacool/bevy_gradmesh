# BevyGradmesh

Procedural mesh generation and gradient materials for [Bevy](https://bevyengine.org/).

BevyGradmesh provides a clean API for generating procedural 3D primitives with proper geometry, detail levels, and gradient shading — inspired by Freya Holmer's work.

## Features

### Shapes

- **3D primitives**: Sphere, ellipsoid, cuboid (with rounded edges), cylinder, cone, capsule, hemisphere, torus (with arc support), regular prism, pyramid, tapered cylinder
- **2D primitives**: Disc, polygon, ellipse, rectangle, star (N-pointed), ring/annulus, wedge/pie-slice
- **Advanced geometry**: Arbitrary polygons (convex & concave), 3D polylines as tubes, Bézier curves (linear, quadratic, cubic)
- **Detail levels**: Control mesh resolution from minimal (8 segments) to extreme (32 segments)

### Rendering Modes

- **Direct mode**: Call `Shape::build()` and spawn `Mesh3d` entities manually
- **Immediate mode**: Per-frame draw/dispose with `ImmediateRecorder` — supports push/pop state stacking, billboarding, and relative transforms
- **Retained mode**: `RetainedWorld` + `GradMeshHandle` for persistent shapes updated per-frame
- **Batched GPU instancing**: `BatchRecorder` groups instances by mesh key, blend mode, and render state into single `draw_indexed` calls — thousands of shapes in far fewer draw calls

### Materials & Shading

- **Gradient materials**: Dual-directional and dual-radial shading for 3D-looking gradients
- **OKLAB color space**: Perceptually uniform color interpolation avoiding muddy midpoints
- **Blend modes**: Opaque, transparent, additive, multiplicative, screen, dither, and dashed outlines
- **PBR integration**: Full compatibility with Bevy's physically-based rendering (roughness, metallicity, emissive)
- **Custom shader passthrough**: Attach any Bevy shader via `BlendMode::Custom`

### Performance

- **Batched GPU instancing**: Single draw call per batch with per-instance transforms and properties on the GPU
- **Dirty tracking**: Only modified batches re-upload GPU data each frame
- **Mesh caching**: `MeshCache` resource avoids regenerating identical geometry
- **Automatic LOD**: Distance-based level-of-detail with Bayer dither cross-fade eliminating pop-in
- **Frustum culling**: Per-instance AABB culling against camera frustum planes
- **Statistics API**: `InstancedStats` tracks instances, batches, cache hits, GPU upload bytes, and culling ratios

### Custom Pipeline (Mobile-Friendly)

- **Standalone pipeline**: `StandaloneRenderPlugin` bypasses Bevy's PBR system entirely, using a simple vertex+fragment shader with no PBR imports
- **Zero descriptor pool pressure**: All material variation via per-instance vertex attributes — one pipeline, one bind group layout, one descriptor set
- **Faster compilation**: One shader pipeline instead of N specializations

### Animation

- **Vertex shader deformation**: Wobble, pulse, and morph effects without CPU mesh regeneration
- **Appearance snapshots**: Per-instance color animation (pulse, cycle, gradient shift) resolved in the shader — no per-frame CPU updates

## What is Dual-Directional Shading?

Dual-directional shading is a technique that colors a surface with a smooth gradient between two colors along a given direction, instead of using a single flat color. One side of the shape takes on the "dark" color and the opposite side takes on the "light" color, with a smooth transition in between. This mimics the look of a single light source illuminating the shape, but is computed entirely in the shader — no actual lights are needed.

It is commonly used in stylized games to give characters a painted, cel-shaded appearance: skin tones shift from warm highlights to cool shadows, and hair shows a bright rim where light catches it. The gradient direction can be any axis (e.g., `Vec3::UP` for top-down lighting), and the two colors can be interpolated in either standard RGB or [OKLAB](https://bottosson.github.io/posts/oklab/) (a perceptually uniform color space that avoids muddy midpoints).

### Key Terms

- **Gradient direction**: A 3D vector defining which axis the gradient runs along. Vertices on one side of the shape get the dark color; vertices on the opposite side get the light color.
- **OKLAB**: A color space designed so that equal distances correspond to equal perceived color differences. Gradients in OKLAB look smoother and more natural than linear RGB interpolation.
- **Blend mode**: How a shape's color combines with what's behind it. Common modes include *opaque* (solid, no transparency), *transparent* (alpha-blended), *additive* (colors add together, creating a glow effect), *multiplicative* (colors multiply, darkening the result), *screen* (the opposite of multiply, lightening the result), and *dither* (uses a pattern to simulate intermediate alpha levels).
- **PBR (Physically-Based Rendering)**: A shading model that simulates how light interacts with real-world materials. Parameters include *roughness* (how smooth or rough a surface is), *metallicity* (whether a surface is metallic or dielectric), and *emissive* (light the material emits on its own, like a glowing screen).
- **Detail level**: Controls the number of vertices and triangles in a generated mesh. Higher detail produces smoother curves but uses more GPU memory and processing.

## Compatibility

| bevy_gradmesh | Bevy | Rust |
|---------------|------|------|
| 0.1.x         | 0.20 | 1.87+|

## Quick Start

```toml
[dependencies]
bevy_gradmesh = "0.1"
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

### Kart Racer

Playable racing demo with a Bézier-curve track, PBR kart, collectible coins, and HUD:

![Kart Racer](screenshots/screenshot_kart_racer.png)

### GPU Deformation

Vertex shader wobble, pulse, and morph effects with zero CPU overhead:

![GPU Deformation](screenshots/screenshot_deformation.png)

### Appearance Animation

Per-instance color animation (ColorPulse, ColorCycle, GradientShift) resolved in the shader:

![Appearance Animation](screenshots/screenshot_appearance.png)

### Retained Mode

Persistent shapes via `GradMeshHandle` with batched instancing:

![Retained Mode](screenshots/screenshot_retained.png)

### Standalone Pipeline

PBR-free rendering with zero descriptor pool pressure — one pipeline, one descriptor set:

![Standalone Pipeline](screenshots/screenshot_standalone_pipeline.png)

### Advanced Shapes

Star, ring, wedge, colored polygon, tapered cylinder, torus arc, and hemisphere:

![Advanced Shapes](screenshots/screenshot_advanced_shapes.png)

### LOD Demo

Distance-based level-of-detail with dither cross-fade eliminating pop-in:

![LOD Demo](screenshots/screenshot_lod_demo.png)

### Frustum Culling

Per-instance AABB culling against camera frustum planes with real-time stats:

![Frustum Culling](screenshots/screenshot_culling_demo.png)

### Instanced Rendering

500 spheres rendered in a single GPU batch:

![Instanced Rendering](screenshots/screenshot_instanced.png)

### Capturing Screenshots

To automatically capture screenshots of all examples, run:

```bash
make screenshots
```

This will render each example for a few frames, save a PNG to the `screenshots/` directory, and exit. On headless systems, `xvfb-run` is auto-detected and used automatically.

## License

Dual-licensed under MIT or Apache-2.0.
