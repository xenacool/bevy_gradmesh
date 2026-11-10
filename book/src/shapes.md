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

*Every primitive at several detail levels, shaded by dual-directional gradients instead of lights. Solids go through the standalone pipeline and flat shapes are `Mesh2d` meshes, so it runs without PBR. Run `cargo run --example shapes` to reproduce it.*

## 2D Primitives

Rectangles, discs, ellipses, rings, wedges, polygons, stars and arbitrary polygons use the SDF
backend (`SdfMaterial2d`). Each one is a material drawn on the shared `unit_quad()` mesh. Sizes are
in world units (pixels with the default `Camera2d`), and the quad's `Transform` scale must be
`SdfMaterial2d::quad_scale()`. See [Choosing a Backend](./backends.md) for the full SDF API.

Polygons with holes, per-vertex colours and outlines of more than 16 vertices go through the mesh
backend with `Shape::as_flat()?.build_2d()`. See [Arbitrary Polygon on the mesh backend](#arbitrary-polygon-on-the-mesh-backend).

### Rectangle
```rust
# use bevy::prelude::*;
# use bevy_gradmesh::SdfMaterial2d;
# fn main() {}
fn rectangle() -> SdfMaterial2d {
    let corner_radius = 8.0;
    SdfMaterial2d::rect(Vec2::new(200.0, 100.0)).with_corner_radius(corner_radius)
}
```
The corner radius is optional. Leave it out for sharp corners.

### Disc
```rust
# use bevy_gradmesh::SdfMaterial2d;
# fn main() {}
fn disc() -> SdfMaterial2d {
    SdfMaterial2d::disc(100.0)
}
```

### Ellipse
```rust
# use bevy::prelude::*;
# use bevy_gradmesh::SdfMaterial2d;
# fn main() {}
fn ellipse() -> SdfMaterial2d {
    SdfMaterial2d::ellipse(Vec2::new(150.0, 75.0))
}
```

### Ring
```rust
# use bevy_gradmesh::SdfMaterial2d;
# fn main() {}
fn ring() -> SdfMaterial2d {
    let outer_radius = 100.0;
    let inner_radius = 60.0;
    let rim_width = outer_radius - inner_radius;
    SdfMaterial2d::disc(outer_radius).with_rim(rim_width)
}
```
Annulus (2D donut with a hole). The rim is drawn inside the outline of the disc.

### Wedge
```rust
# use bevy_gradmesh::SdfMaterial2d;
# fn main() {}
fn wedge() -> SdfMaterial2d {
    let start_angle = 0.0;
    let span = std::f32::consts::FRAC_PI_2;
    SdfMaterial2d::disc(100.0).with_arc(start_angle, span)
}
```
Pie-slice shape. The arc starts at `start_angle` radians counter-clockwise from +X and covers
`span` radians.

### Spawning an SDF primitive
```rust
# use bevy::prelude::*;
# use bevy_gradmesh::{SdfMaterial2d, unit_quad};
# fn main() {}
fn spawn_primitive(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<SdfMaterial2d>>,
) {
    let material = SdfMaterial2d::disc(100.0)
        .with_colors(Color::srgb(0.9, 0.4, 0.5), Color::WHITE);
    commands.spawn((
        Mesh2d(meshes.add(unit_quad())),
        Transform::from_scale(material.quad_scale()),
        MeshMaterial2d(materials.add(material)),
    ));
}
```
Add all primitives to the same `unit_quad()` handle to share one mesh asset.

### Polygon
```rust
# use bevy_gradmesh::SdfMaterial2d;
# fn main() {}
fn polygon() -> SdfMaterial2d {
    let sides = 5;
    let radius = 100.0;
    SdfMaterial2d::polygon(sides, radius)
}
```
Regular polygon with 3 to 64 sides and a vertex pointing up. `radius` is the distance from the
centre to a vertex.

### Star
```rust
# use bevy_gradmesh::SdfMaterial2d;
# fn main() {}
fn star() -> SdfMaterial2d {
    let points = 5;
    let outer_radius = 100.0;
    let inner_radius = 40.0;
    SdfMaterial2d::star(points, outer_radius, inner_radius)
}
```
N-pointed star with 3 to 64 points. The `inner_radius` controls how deep the indentations are. An
inner radius at or above the outer radius gives a polygon with twice as many sides.

### Arbitrary Polygon
```rust
# use bevy::prelude::*;
# use bevy_gradmesh::{SdfMaterial2d, SdfPolygonError};
# fn main() {}
fn arrow() -> Result<SdfMaterial2d, SdfPolygonError> {
    let points = [
        Vec2::new(-70.0, -20.0),
        Vec2::new(10.0, -20.0),
        Vec2::new(10.0, -60.0),
        Vec2::new(75.0, 0.0),
        Vec2::new(10.0, 60.0),
        Vec2::new(10.0, 20.0),
        Vec2::new(-70.0, 20.0),
    ];
    SdfMaterial2d::polygon_from_points(&points)
}
```
Any outline of 3 to 16 points, concave outlines included, wound either way. Fewer than 3 or more
than 16 points return a `SdfPolygonError` that names the count. The shape is centred on the middle
of its bounding box.

Polygons and stars support rims, arcs, gradients and colour modes like every other SDF shape. A
rim is drawn inside the outline, so thin features of a concave shape can disappear under a wide
rim. Dashes are not supported on these kinds: a dashed rim draws solid.

*Discs, ellipses, rounded rectangles, rims, arcs, polygons and stars on one shared quad. Run `cargo run --example sdf_2d` to reproduce it.*

*Custom polygons with per-vertex colours, stars, rings, wedges and the side profiles of tapered cylinders (`Shape::as_flat` refuses the solid itself). Run `cargo run --example arbitrary_shapes` to reproduce it.*

### Arbitrary Polygon on the mesh backend
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
Use this form for outlines of more than 16 vertices, polygons with holes and per-vertex colours.

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

### Bézier Curve
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
Supports linear (2 points), quadratic (3 points), and cubic (4+ points) Bézier curves rendered as tubes.

## Shape Morphing

`Shape::lerp` blends two shapes by interpolating their parameters. When the shapes share the same
kind, every continuous parameter (radius, height, size, roundness) is interpolated. Integer counts
(sides, segments, points) and boolean flags (capped, closed) snap at `t = 0.5` — a prism cannot
have 6.3 sides.

### Cross-kind flat shape morphing

Filled flat shapes of **different kinds** morph as outlines: a polygon can turn into a star, a
disc rim can turn into a rectangle, and a ring can turn into a disc. The blend produces an
arbitrary polygon. Supported kinds: rectangle, disc, ellipse, polygon, star, arbitrary polygon,
rectangle rim, disc rim, ellipse rim, polygon rim, ring and wedge. For rims and rings only the
outer edge participates; a ring's inner hole is ignored.

```rust
# use bevy_gradmesh::{Shape, ShapeDetail};
# fn main() {}
let a = Shape::disc(1.2, ShapeDetail::Medium);
let b = Shape::star(5, 1.2, 0.5, ShapeDetail::Medium);
let mid = a.lerp(&b, 0.5); // an ArbitraryPolygon between disc and star
```

### Ring morph example

A ring morphs into a disc by blending the ring's outer edge with the disc's outline:

```rust
# use bevy_gradmesh::{Shape, ShapeDetail};
# fn main() {}
let ring = Shape::ring(1.2, 0.5, ShapeDetail::Medium);
let disc = Shape::disc(1.2, ShapeDetail::Medium);
let mid = ring.lerp(&disc, 0.5); // fills the hole gradually
```

A wedge morphs into a disc similarly:

```rust
# use bevy_gradmesh::{Shape, ShapeDetail};
# fn main() {}
let wedge = Shape::wedge(1.2, std::f32::consts::PI, ShapeDetail::Medium);
let disc = Shape::disc(1.2, ShapeDetail::Medium);
let mid = wedge.lerp(&disc, 0.5);
```

### Performance: pre-computing keyframe tables

Calling `lerp` with a different `t` each frame produces a new mesh each frame. At 60 Hz this is
60 new meshes per second. To avoid this cost:

1. **Animate a continuous proxy**: Keep a fixed mesh and animate its `Transform` (scale,
   position, rotation) instead of morphing the shape.

2. **Pre-compute a keyframe table**: Build intermediate shapes once at fixed `t` values and
   switch between them each frame:

   ```rust
   # use bevy_gradmesh::{Shape, ShapeDetail};
   # let a = Shape::disc(1.0, ShapeDetail::Medium);
   # let b = Shape::star(5, 1.0, 0.4, ShapeDetail::Medium);
   // Build once (e.g. in Startup system)
   let keyframes: Vec<Shape> = (0..=10)
       .map(|i| a.lerp(&b, i as f32 / 10.0))
       .collect();
   // Each frame, pick the nearest keyframe
   let idx = (t * 10.0).round() as usize;
   let shape = &keyframes[idx.min(keyframes.len() - 1)];
   ```

   This limits mesh generation to the number of keyframes (11 in the example) and makes each
   frame a cheap array lookup.

3. **Quantise `t`**: Round `t` to a small set of steps before calling `lerp`, so repeated frames
   reuse the same `Shape` value.

*Six morphing shapes: disc↔star, rect↔ring, polygon↔disc rim, ring↔disc, disc rim↔rect, wedge↔disc. Run `cargo run --example lerp` to reproduce it.*
