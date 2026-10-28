# Curves & Polylines

## Bézier Curves

Render smooth curves as tube geometry:

```rust
use bevy_gradmesh::Shape;

let control_points = vec![
    Vec3::new(0.0, 0.0, 0.0),
    Vec3::new(1.0, 2.0, 0.0),
    Vec3::new(2.0, -2.0, 0.0),
    Vec3::new(3.0, 0.0, 0.0),
];
let mesh = Shape::bezier_curve(control_points, 0.1, 8, ShapeDetail::Medium).build();
```

### Curve Types

- **Linear** (2 control points): Straight line tube
- **Quadratic** (3 control points): Single curve
- **Cubic** (4+ control points): Complex S-curves

### Animated Curves

Update control points each frame for animated paths:

```rust
fn animate_curve(mut time: Res<Time>) {
    let t = time.elapsed_secs();
    let points = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, t.sin() * 2.0, 0.0),
        Vec3::new(2.0, t.cos() * 2.0, 0.0),
        Vec3::new(3.0, 0.0, 0.0),
    ];
    // Rebuild curve with new control points
}
```

## 3D Polylines

Connect points with tube geometry:

```rust
use bevy_gradmesh::Shape;

let points = vec![
    Vec3::new(0.0, 0.0, 0.0),
    Vec3::new(1.0, 1.0, 0.0),
    Vec3::new(2.0, 0.0, 0.0),
    Vec3::new(3.0, 1.0, 0.0),
];
let mesh = Shape::polyline3d(points, 0.05, false, 8, ShapeDetail::Medium).build();
```

### Closed Polylines

```rust
let mesh = Shape::polyline3d(points, 0.05, true, 8, ShapeDetail::Medium).build();
```

## Use Cases

- **Track paths**: Define racing tracks with Bézier curves
- **Cables and wires**: Connect objects with polylines
- **Animated trails**: Update control points each frame
- **Data visualization**: Plot 3D data as connected tubes
