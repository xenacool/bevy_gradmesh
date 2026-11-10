# Introduction

BevyGradmesh is a procedural mesh generation library for [Bevy](https://bevyengine.org/) that provides clean APIs for creating 3D primitives with gradient shading, inspired by Freya Holmer's work.

## What It Solves

Instead of manually creating mesh assets for every shape, BevyGradmesh generates watertight geometry at runtime with proper normals and UVs. Combined with dual-directional gradient materials, you get stylized 3D-looking characters and objects without any external assets.

*Every procedural primitive with dual-directional gradient shading. Run `cargo run --example shapes` to reproduce it.*

## Key Concepts

- **Shapes**: Procedural primitives (spheres, cuboids, cylinders, etc.) generated at runtime
- **Styles**: Gradient materials with dual-directional or dual-radial shading
- **Modes**: Four rendering modes (direct, immediate, retained, batched) for different use cases
- **Performance**: GPU instancing, LOD, frustum culling, and dirty tracking for rendering thousands of shapes

## Design Philosophy

BevyGradmesh follows these principles:

1. **Zero external assets**: Everything is generated procedurally
2. **Bevy-native**: Outputs standard `Mesh` and `StandardMaterial` types
3. **Performance-first**: GPU instancing and dirty tracking minimize draw calls
4. **Flexible**: Multiple rendering modes for different patterns

## Key Terms

- **Gradient direction**: A 3D vector defining which axis the gradient runs along. Vertices on one side of the shape get the dark color; vertices on the opposite side get the light color.
- **OKLAB**: A color space designed so that equal distances correspond to equal perceived color differences. Gradients in OKLAB look smoother and more natural than linear RGB interpolation. OKLCH is the same space in polar form (lightness, chroma, hue), so blends sweep around the hue wheel.
- **Blend mode**: How a shape's color combines with what's behind it. Common modes include *opaque* (solid, no transparency), *transparent* (alpha-blended), *additive* (colors add together, creating a glow effect), *multiplicative* (colors multiply, darkening the result), *screen* (the opposite of multiply, lightening the result), and *dither* (uses a pattern to simulate intermediate alpha levels).
- **PBR (Physically-Based Rendering)**: A shading model that simulates how light interacts with real-world materials. Parameters include *roughness* (how smooth or rough a surface is), *metallicity* (whether a surface is metallic or dielectric), and *emissive* (light the material emits on its own, like a glowing screen).
- **Detail level**: Controls the number of vertices and triangles in a generated mesh. Higher detail produces smoother curves but uses more GPU memory and processing.

## Compatibility

| bevy_gradmesh | Bevy | Rust |
|---------------|------|------|
| 0.3.x         | 0.20 | 1.87+|
