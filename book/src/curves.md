# Curves & Polylines

## Bézier Curves

Render smooth curves as tube geometry:

```rust
# use bevy::prelude::*;
# use bevy_gradmesh::{Shape, ShapeDetail};
# fn main() {}
fn curve_mesh() -> Mesh {
    let control_points = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 2.0, 0.0),
        Vec3::new(2.0, -2.0, 0.0),
        Vec3::new(3.0, 0.0, 0.0),
    ];
    Shape::bezier_curve(control_points, 0.1, 8, ShapeDetail::Medium).build()
}
```

### Curve Types

- **Linear** (2 control points): Straight line tube
- **Quadratic** (3 control points): Single curve
- **Cubic** (4+ control points): Complex S-curves

### Animated Curves

Update control points each frame for animated paths:

```rust
# use bevy::prelude::*;
# use bevy_gradmesh::{Shape, ShapeDetail};
# fn main() {}
#[derive(Resource)]
struct CurveMesh(Handle<Mesh>);

fn animate_curve(
    time: Res<Time>,
    curve: Res<CurveMesh>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let t = time.elapsed_secs();
    let points = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, t.sin() * 2.0, 0.0),
        Vec3::new(2.0, t.cos() * 2.0, 0.0),
        Vec3::new(3.0, 0.0, 0.0),
    ];
    // Rebuild curve with new control points
    let rebuilt = Shape::bezier_curve(points, 0.1, 8, ShapeDetail::Medium).build();
    if let Some(mut mesh) = meshes.get_mut(&curve.0) {
        *mesh = rebuilt;
    }
}
```

*Bezier curves of every degree and a 3D helix, drawn as tubes by the standalone pipeline without PBR. The standalone gradient is measured from each tube's own origin along the world axis, so every tube shows its full blend wherever it sits. Run `cargo run --example curves` to reproduce it.*

## 3D Polylines

Connect points with tube geometry:

```rust
# use bevy::prelude::*;
# use bevy_gradmesh::{Shape, ShapeDetail};
# fn main() {}
fn polyline_mesh() -> Mesh {
    let points = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(3.0, 1.0, 0.0),
    ];
    Shape::polyline3d(points, 0.05, false, 8, ShapeDetail::Medium).build()
}
```

### Closed Polylines

```rust
# use bevy::prelude::*;
# use bevy_gradmesh::{Shape, ShapeDetail};
# fn main() {}
fn closed_polyline_mesh() -> Mesh {
    let points = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(3.0, 1.0, 0.0),
    ];
    Shape::polyline3d(points, 0.05, true, 8, ShapeDetail::Medium).build()
}
```

*Flat 2D polylines drawn as `Mesh2d` without PBR: open, closed, thick and thin, with a gradient along each line. The example is 2D only; `Shape::polyline3d` (above) builds tubes for a 3D camera. Run `cargo run --example polyline` to reproduce it.*

## Use Cases

- **Track paths**: Define racing tracks with Bézier curves
- **Cables and wires**: Connect objects with polylines
- **Animated trails**: Update control points each frame
- **Data visualization**: Plot 3D data as connected tubes
