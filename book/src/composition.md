# Composing Shapes, 2D and Ghost Previews

## Flat shapes with `Mesh2d`

`Shape::build_2d()` returns geometry in the XY plane facing +Z with gradient UVs. Pair it with
`GradMeshMaterial2d` (built from a `Style`) and a `Camera2d`:

```rust
use bevy::prelude::*;
use bevy_gradmesh::{Shape, ShapeDetail, Style};

let mesh = Shape::disc(0.5, ShapeDetail::High).build_2d();
let material = Style::dual_radial(Color::WHITE, Color::srgb(0.9, 0.4, 0.5)).build_2d();
// commands.spawn((Mesh2d(meshes.add(mesh)), MeshMaterial2d(materials.add(material)),
//     Transform::from_xyz(0.0, 0.0, 5.0).with_scale(Vec3::splat(120.0))));
```

- `Transform` scale resizes the shape on the XY plane; gradients and dashes follow because they are
  computed from the mesh UVs.
- `Transform.translation.z` orders shapes against each other.
- `GradMeshPlugin` registers the material when Bevy's 2D mesh renderer is present. Add
  `GradMesh2dPlugin` yourself if you add `GradMeshPlugin` before the render plugins.
- The material is unlit; roughness, metallic and emissive have no effect in 2D.
- In 2D the gradient direction is expressed in the shape's own UV frame (+Y towards the top).
- Animate a material with `GradMeshMaterial2d::with_appearance(AppearanceSnapshot::color_pulse(..))`.
- `Shape::polyline_2d` and `Shape::bezier_2d` build flat curves with thickness (`u` runs along the
  curve, `v` across it).

*Gradient cards, a translucent dragged card layered by z, a radial disc, a pulsing star, a dashed ring and a polygon with a hole. Run `cargo run --example gradient_2d` to reproduce it.*

## Outlines, extrusion, holes

`arch`, `rounded_rectangle` and `blob` return counter-clockwise outlines centred on the origin.

```rust
use bevy::prelude::*;
use bevy_gradmesh::{Shape, ShapeDetail, arch, rounded_rectangle};

let wall = Shape::extruded_polygon_with_holes(
    rounded_rectangle(Vec2::new(4.0, 3.4), 0.15, 3),
    vec![arch(1.2, 2.0, 10)],
    0.5,  // depth
    0.35, // roundness of the rim
    ShapeDetail::High,
);
```

`Shape::extruded_polygon` and `extruded_polygon_colored` handle single outlines (the colours are
per vertex and multiply the gradient), `arbitrary_polygon_with_holes` builds the flat version. Either
winding is accepted; extrusion is centred on `z = 0`.

## Compound shapes and lathe

`Shape::compound` merges parts into one mesh (one draw call). Each `ShapePart` has a transform and an
optional tint that multiplies the gradient through vertex colours, so parts keep their own gradients.
Compounds are `Hash + Eq` and work with `MeshCache`.

`Shape::lathe(profile, segments, detail)` revolves `(radius, height)` points around Y. Crisp corners stay
crisp, gentle ones are smoothed, and open ends away from the axis get caps.

*A compound tower, a wall with an arched opening, extruded cobbles, an emissive torch, a metallic gem and a ghost piece that casts no shadow. Run `cargo run --example composed_shapes` to reproduce it.*

## Sharing meshes and materials

```rust
// let mesh_handle = cache.get_or_build_handle(&shape, &mut meshes);
// let material_handle = pool.get_or_add(&style, &mut materials);
```

Equal shapes return the same `Handle<Mesh>`, equal styles (`Style: PartialEq`) the same material, so a
per-frame sync of an unchanged node allocates nothing.

## Style additions

- `Style::shaded(color, dark_factor)`: lit `color`, shadow side `color * dark_factor` (linear RGB).
- `with_perceptual_roughness`, `with_metallic`, `with_emissive`: written to the `StandardMaterial`
  base and blended by `Style::lerp`.
- `with_alpha`: multiplies both gradient colours' alpha; `Style::lerp` fades it.

### Gradient direction space

The gradient direction is a **world-space** vector by default: rotating the entity does not rotate the
gradient. `Style::with_direction_space(DirectionSpace::Local)` rotates the direction with the entity's
`GlobalTransform`, which keeps the intended lit side for pieces built Y-up and placed in a Z-up scene by
a rotation.

### Ghost previews and shadows

Translucent gradient materials still render in Bevy's shadow pass (alpha-tested only), so they cast a
full shadow. Bevy decides shadow casting per entity: when `style.casts_shadows()` is `false`, add
`NotShadowCaster` to the entity (see the `composed_shapes` example).

### Surface detail and rim glow

`GradMeshMaterial` can vary the tint without textures:

- `Style::with_noise(scale, amount)`: smooth value noise, `scale` cells per local unit.
- `Style::with_stripes(direction, frequency, amount)`: bands along a local direction.
- `Style::with_rim(color, power)`: view-dependent emission towards the silhouette, for example to
  highlight the nearest petal of a radial menu.

Noise and stripes live in the entity's local space, so they stay attached as it moves, turns and
scales. All three are blended by `Style::lerp`.

### Rounded roofs and gems

`Shape::rounded_cone` and `Shape::rounded_pyramid` take a `roundness` from `0.0` (identical to `cone` and
`pyramid`) to `1.0`. `Shape::regular_prism` already takes one. The rounded solid stays inside the sharp
one, so footprints and heights are kept.

### Bounds and kind

`Shape::bounds()` and `Shape::bounds_2d()` return a `ShapeBounds` measured from the built vertices, and
`Shape::is_round()` tells spheres, ellipsoids, discs, ellipses and rings from other shapes, so hit tests
can approximate them with an ellipsoid.

### Picking

Every shape builds positions, normals, `UV_0` and indices, so Bevy's `MeshPickingPlugin` and the
automatically computed `Aabb` work without custom backends. 2D shapes pick through the standard `Mesh2d`
path.
