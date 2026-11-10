# Shapes

BevyGradmesh provides a wide variety of procedural primitives, all generated at runtime with proper normals and UVs.

## 3D Primitives

### Sphere
```rust
# use bevy_gradmesh::{Shape, ShapeDetail};
# fn main() {}
fn sphere() -> Shape {
    Shape::sphere(1.0, ShapeDetail::Medium)
}
```

### Ellipsoid
```rust
# use bevy::prelude::*;
# use bevy_gradmesh::{Shape, ShapeDetail};
# fn main() {}
fn ellipsoid() -> Shape {
    Shape::ellipsoid(Vec3::new(2.0, 1.0, 1.5), ShapeDetail::Medium)
}
```

### Cuboid
```rust
# use bevy::prelude::*;
# use bevy_gradmesh::{Shape, ShapeDetail};
# fn main() {}
fn cuboid() -> Shape {
    let roundness = 0.2;
    Shape::cuboid(Vec3::new(1.0, 2.0, 1.0), roundness, ShapeDetail::Medium)
}
```
The `roundness` parameter (0.0 to 1.0) creates rounded edges. At 0.0, produces a sharp box.

### Cylinder
```rust
# use bevy_gradmesh::{Shape, ShapeDetail};
# fn main() {}
fn cylinder() -> Shape {
    Shape::cylinder(0.5, 2.0, ShapeDetail::Medium)
}
```

### Cone
```rust
# use bevy_gradmesh::{Shape, ShapeDetail};
# fn main() {}
fn cone() -> Shape {
    Shape::cone(0.5, 2.0, ShapeDetail::Medium)
}
```

### Capsule
```rust
# use bevy_gradmesh::{Shape, ShapeDetail};
# fn main() {}
fn capsule() -> Shape {
    Shape::capsule(0.5, 2.0, ShapeDetail::Medium)
}
```

### Hemisphere
```rust
# use bevy_gradmesh::{Shape, ShapeDetail};
# fn main() {}
fn hemisphere() -> Shape {
    let capped = true;
    Shape::hemisphere(1.0, capped, ShapeDetail::Medium)
}
```
When `capped` is true, the flat face is filled with geometry.

### Torus
```rust
# use bevy_gradmesh::Shape;
# fn main() {}
fn torus() -> Shape {
    let start_angle = 0.0;
    let span = std::f32::consts::TAU;
    Shape::torus(1.0, 0.25, start_angle, span)
}
```
Supports partial torus arcs. Use `start_angle = 0.0` and `span = TAU` for a full torus.

### Regular Prism
```rust
# use bevy_gradmesh::{Shape, ShapeDetail};
# fn main() {}
fn regular_prism() -> Shape {
    let sides = 6;
    let roundness = 0.1;
    Shape::regular_prism(sides, 1.0, 2.0, roundness, ShapeDetail::Medium)
}
```

### Pyramid
```rust
# use bevy_gradmesh::{Shape, ShapeDetail};
# fn main() {}
fn pyramid() -> Shape {
    let sides = 4;
    Shape::pyramid(sides, 1.0, 1.5, ShapeDetail::Medium)
}
```

### Tapered Cylinder
```rust
# use bevy_gradmesh::{Shape, ShapeDetail};
# fn main() {}
fn tapered_cylinder() -> Shape {
    let top_radius = 0.3;
    let bottom_radius = 0.6;
    Shape::tapered_cylinder(top_radius, bottom_radius, 2.0, ShapeDetail::Medium)
}
```
A cone-frustum with flat top and bottom. When radii are equal, equivalent to a regular cylinder.

*The 3D primitives, shaded with dual-directional gradients. Run `cargo run --example shapes` to reproduce it.*

## 2D Primitives

All 2D shapes are flat quads facing the camera (billboarded).

### Rectangle
```rust
# use bevy::prelude::*;
# use bevy_gradmesh::{Shape, ShapeDetail};
# fn main() {}
fn rectangle() -> Shape {
    Shape::rectangle(Vec2::new(2.0, 1.0), ShapeDetail::Medium)
}
```

### Disc
```rust
# use bevy_gradmesh::{Shape, ShapeDetail};
# fn main() {}
fn disc() -> Shape {
    Shape::disc(1.0, ShapeDetail::Medium)
}
```

### Polygon
```rust
# use bevy_gradmesh::{Shape, ShapeDetail};
# fn main() {}
fn polygon() -> Shape {
    let sides = 5;
    Shape::polygon(sides, 1.0, ShapeDetail::Medium)
}
```

### Ellipse
```rust
# use bevy::prelude::*;
# use bevy_gradmesh::{Shape, ShapeDetail};
# fn main() {}
fn ellipse() -> Shape {
    Shape::ellipse(Vec2::new(1.5, 0.75), ShapeDetail::Medium)
}
```

### Star
```rust
# use bevy_gradmesh::{Shape, ShapeDetail};
# fn main() {}
fn star() -> Shape {
    let points = 5;
    let outer_radius = 1.0;
    let inner_radius = 0.4;
    Shape::star(points, outer_radius, inner_radius, ShapeDetail::Medium)
}
```
N-pointed star. The `inner_radius` controls how deep the indentations are.

### Ring
```rust
# use bevy_gradmesh::{Shape, ShapeDetail};
# fn main() {}
fn ring() -> Shape {
    let outer_radius = 1.0;
    let inner_radius = 0.6;
    Shape::ring(outer_radius, inner_radius, ShapeDetail::Medium)
}
```
Annulus (2D donut with a hole).

### Wedge
```rust
# use bevy_gradmesh::{Shape, ShapeDetail};
# fn main() {}
fn wedge() -> Shape {
    let angle = std::f32::consts::FRAC_PI_2;
    Shape::wedge(1.0, angle, ShapeDetail::Medium)
}
```
Pie-slice shape. Angle is in radians (0 to 2 pi).

*Custom polygons with per-vertex colours, stars, rings, wedges and a tapered cylinder. Run `cargo run --example arbitrary_shapes` to reproduce it.*

### Arbitrary Polygon
```rust
# use bevy::prelude::*;
# use bevy_gradmesh::{Shape, ShapeDetail};
# fn main() {}
fn arbitrary_polygon() -> Shape {
    let vertices = vec![
        Vec2::new(-1.0, -1.0),
        Vec2::new(1.0, -1.0),
        Vec2::new(0.0, 1.0),
    ];
    Shape::arbitrary_polygon(vertices, ShapeDetail::Medium)
}

fn arbitrary_polygon_colored() -> Shape {
    let vertices = vec![
        Vec2::new(-1.0, -1.0),
        Vec2::new(1.0, -1.0),
        Vec2::new(0.0, 1.0),
    ];
    let colors = vec![
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ];
    Shape::arbitrary_polygon_colored(vertices, colors, ShapeDetail::Medium)
}
```
Accepts any set of 2D vertices. Triangulated using ear-clipping algorithm. Supports optional per-vertex RGBA colors.

## Advanced Geometry

### 3D Polyline
```rust
# use bevy::prelude::*;
# use bevy_gradmesh::{Shape, ShapeDetail};
# fn main() {}
fn polyline3d() -> Shape {
    let points = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(2.0, 0.0, 1.0),
    ];
    let thickness = 0.1;
    let closed = false;
    let segments = 8;
    Shape::polyline3d(points, thickness, closed, segments, ShapeDetail::Medium)
}
```
Renders as tube geometry along the point path.

### Bezier Curve
```rust
# use bevy::prelude::*;
# use bevy_gradmesh::{Shape, ShapeDetail};
# fn main() {}
fn bezier_curve() -> Shape {
    let control_points = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 2.0, 0.0),
        Vec3::new(2.0, 2.0, 0.0),
        Vec3::new(3.0, 0.0, 0.0),
    ];
    let thickness = 0.1;
    let segments = 32;
    Shape::bezier_curve(control_points, thickness, segments, ShapeDetail::Medium)
}
```
Supports linear (2 points), quadratic (3 points), and cubic (4+ points) Bezier curves rendered as tubes.
