# Getting Started

## Installation

Add `bevy_gradmesh` to your `Cargo.toml`:

```toml
[dependencies]
bevy = "0.20"
bevy_gradmesh = "0.1"
```

## Your First Shape

```rust
use bevy::prelude::*;
use bevy_gradmesh::{GradMeshPlugin, Shape, ShapeDetail, dual_directional};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(GradMeshPlugin)
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Camera
    commands.spawn(Camera3d::from_transform(
        Transform::from_xyz(0.0, 5.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y)
    ));

    // Light
    commands.spawn(DirectionalLight::new_white());

    // Generate a sphere mesh
    let mesh = meshes.add(Shape::sphere(1.0, ShapeDetail::Medium).build());

    // Create a dual-directional gradient material
    let mat = materials.add(dual_directional(
        [0.6, 0.4, 0.3],  // dark color
        [0.96, 0.8, 0.69], // light color
        [0.0, 1.0, 0.0],   // gradient direction (up)
    ));

    // Spawn the shape
    commands.spawn((Mesh3d(mesh), MeshMaterial3d(mat)));
}
```

## Understanding the API

The core API has three main types:

1. **`Shape`** — Describes a procedural primitive. Call `.build()` to get a Bevy `Mesh`.
2. **`Style`** — Describes gradient material properties. Call `.build_material()` to get a `StandardMaterial`.
3. **`GradMeshPlugin`** — Registers the custom material and mesh cache with Bevy.

## Detail Levels

Control mesh resolution with `ShapeDetail`:

```rust
Shape::sphere(1.0, ShapeDetail::Minimal)  // 8 segments
Shape::sphere(1.0, ShapeDetail::Low)      // 12 segments
Shape::sphere(1.0, ShapeDetail::Medium)   // 16 segments (default)
Shape::sphere(1.0, ShapeDetail::High)     // 24 segments
Shape::sphere(1.0, ShapeDetail::Extreme)  // 32 segments
```

## Next Steps

- Learn about all available [Shapes](./shapes.md)
- Explore [Styles & Materials](./styles.md)
- Choose a [Rendering Mode](./modes.md) for your use case