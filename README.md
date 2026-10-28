# BevyGradmesh

Procedural mesh generation and gradient materials for [Bevy](https://bevyengine.org/).

BevyGradmesh provides a clean API for generating procedural 3D primitives with proper geometry, detail levels, and gradient shading — inspired by Freya Holmer's work.

## Features

- **Procedural primitives**: Sphere, ellipsoid, cuboid, cylinder, cone, capsule, hemisphere, torus (with arc support), regular prism, pyramid, and 2D shapes (disc, polygon, ellipse, rectangle)
- **Detail levels**: Control mesh resolution from minimal to extreme
- **Gradient materials**: Dual-directional shading for 3D-looking skin and hair
- **Blend modes**: Transparent, additive, multiplicative, screen, dither, and dashed outlines
- **PBR integration**: Full compatibility with Bevy's physically-based rendering (roughness, metallicity, emissive)
- **Bevy integration**: Ready-to-use `Mesh` and `StandardMaterial` outputs

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

### Capturing Screenshots

To automatically capture screenshots of all examples, run:

```bash
make screenshots
```

This will render each example for a few frames, save a PNG to the `screenshots/` directory, and exit. On headless systems, `xvfb-run` is auto-detected and used automatically.

## License

Dual-licensed under MIT or Apache-2.0.
