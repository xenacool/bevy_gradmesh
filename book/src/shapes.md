# Shapes

BevyGradmesh provides a wide variety of procedural primitives, all generated at runtime with proper normals and UVs.

## 3D Primitives

### Sphere
```rust
Shape::sphere(radius, detail)
```

### Ellipsoid
```rust
Shape::ellipsoid(size, detail)
```

### Cuboid
```rust
Shape::cuboid(size, roundness, detail)
```
The `roundness` parameter (0.0–1.0) creates rounded edges. At 0.0, produces a sharp box.

### Cylinder
```rust
Shape::cylinder(radius, height, detail)
```

### Cone
```rust
Shape::cone(radius, height, detail)
```

### Capsule
```rust
Shape::capsule(radius, height, detail)
```

### Hemisphere
```rust
Shape::hemisphere(radius, capped, detail)
```
When `capped` is true, the flat face is filled with geometry.

### Torus
```rust
Shape::torus(radius, thickness, start_angle, span)
```
Supports partial torus arcs. Use `start_angle = 0.0` and `span = TAU` for a full torus.

### Regular Prism
```rust
Shape::regular_prism(sides, radius, height, roundness, detail)
```

### Pyramid
```rust
Shape::pyramid(sides, radius, height, detail)
```

### Tapered Cylinder
```rust
Shape::tapered_cylinder(top_radius, bottom_radius, height, detail)
```
A cone-frustum with flat top and bottom. When radii are equal, equivalent to a regular cylinder.

## 2D Primitives

All 2D shapes are flat quads facing the camera (billboarded).

### Rectangle
```rust
Shape::rectangle(size, detail)
```

### Disc
```rust
Shape::disc(radius, detail)
```

### Polygon
```rust
Shape::polygon(sides, radius, detail)
```

### Ellipse
```rust
Shape::ellipse(radius, detail)
```

### Star
```rust
Shape::star(points, outer_radius, inner_radius, detail)
```
N-pointed star. The `inner_radius` controls how deep the indentations are.

### Ring
```rust
Shape::ring(outer_radius, inner_radius, detail)
```
Annulus (2D donut with a hole).

### Wedge
```rust
Shape::wedge(radius, angle, detail)
```
Pie-slice shape. Angle is in radians (0 to 2π).

### Arbitrary Polygon
```rust
Shape::arbitrary_polygon(vertices, detail)
Shape::arbitrary_polygon_colored(vertices, colors, detail)
```
Accepts any set of 2D vertices. Triangulated using ear-clipping algorithm. Supports optional per-vertex RGBA colors.

## Advanced Geometry

### 3D Polyline
```rust
Shape::polyline3d(points, thickness, closed, segments, detail)
```
Renders as tube geometry along the point path.

### Bézier Curve
```rust
Shape::bezier_curve(control_points, thickness, segments, detail)
```
Supports linear (2 points), quadratic (3 points), and cubic (4+ points) Bézier curves rendered as tubes.
