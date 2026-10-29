# Introduction

BevyGradmesh is a procedural mesh generation library for [Bevy](https://bevyengine.org/) that provides clean APIs for creating 3D primitives with gradient shading, inspired by Freya Holmer's work.

## What It Solves

Instead of manually creating mesh assets for every shape, BevyGradmesh generates watertight geometry at runtime with proper normals and UVs. Combined with dual-directional gradient materials, you get stylized 3D-looking characters and objects without any external assets.

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

## Compatibility

| bevy_gradmesh | Bevy | Rust |
|---------------|------|------|
| 0.1.x         | 0.20 | 1.87+|
