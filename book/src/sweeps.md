# Sweeps, Meshes, Heightfields and Height Bands

These shape kinds build geometry from your data. Like every `Shape` they are
`Clone + Debug + Eq + Hash`, work with `MeshCache`, and can be a part of
`Shape::compound`.

## `Shape::from_mesh`

```rust
let shape = Shape::from_mesh(Arc::new(my_mesh), 42);
```

Wraps a mesh you generated yourself. The shape is `Eq` and `Hash` by `key`
only; the mesh is never compared. **You own key uniqueness**: two different
meshes with the same key share one cache entry. Bounds and `is_round()` are
measured from the positions once, in the constructor, so `bounds()` does not
build anything. `build()` and `build_2d()` return a copy of the mesh, and
`Shape::lerp` takes the nearer shape.

## `Shape::heightfield`

```rust
let chunk = Shape::heightfield(
    UVec2::new(17, 17),      // samples
    Vec2::splat(16.0),       // world extent
    heights.into(),          // Arc<[f32]>, heights[x + y * size.x]
    chunk_key,
    ShapeDetail::Medium,
);
```

A terrain chunk centred on the origin in XZ, with smooth normals from central
differences and UVs from 0 to 1. Skirts hang below the four edges (a tenth of the
larger extent) to hide cracks between neighbouring chunks of different detail.
Equality and hashing use `key`, `size` and `extent`, not the heights, so change
`key` when the heights change. A size below 2 in either axis, or a wrong
`heights` length, gives an empty mesh and a warning.

## `Shape::sweep`

```rust
let wall = Shape::sweep(
    profile,                      // Vec<Vec2>: x across the path, y up
    path,                         // Vec<Vec3>: a polyline or baked spline positions
    SweepOptions { closed: true, ..default() },
);
```

Sweeps a closed profile along a path as one mesh: walls, roads, rails and pipes.
Frames follow the path by parallel transport from `SweepOptions::up`, joints are
mitred (the sides keep a constant width; the mitre is clamped at four times the
width), and normals are smooth along the path. A corner of the profile sharper
than about 40 degrees stays crisp, gentler ones are smoothed. For a baked spline
pass `baked.frames().iter().map(|f| f.position)`.

Options:

* `closed` joins the last point to the first (no caps, no seam).
* `caps` closes the ends of an open sweep.
* `twist` turns the profile, in whole turns over the path.
* `scale_along` and `tint_by_length` are keyframes along the path.
* `detail` sets how finely long segments are split when something varies along them.
* `openings` cuts gates and windows (below).

### Keyframes instead of a closure

A `Shape` has to be `Clone + Eq + Hash`, which a boxed closure cannot be. So
`scale_along` is `Option<Vec<(f32, Vec2)>>`: `(ratio along the path, scale)`
pairs with linear steps in between. To start from a function, bake it:

```rust
let options = SweepOptions::default()
    .scale_with(8, |t| Vec2::splat(1.0 - 0.55 * t));
```

`tint_by_length` is a list of `(ratio, Color)` pairs and multiplies the style's
colour through vertex colours.

### Openings

```rust
let options = SweepOptions {
    openings: vec![
        SweepOpening::arch(0.30..0.38, 0.0, 1.2, 1.9, 12),  // a gate
        SweepOpening::rect(0.60..0.64, 0.9, 1.4),            // a window
    ],
    ..default()
};
```

An opening is a real hole through the wall, with sill, lintel and jamb faces on
the inside. Its `outline` is in **wall-face coordinates**, not in the profile
plane: `x` runs `0..1` over the opening's `t` range (a ratio of the path length)
and `y` is the height above the profile origin. An outline that reaches the
profile's lowest height cuts the wall open at the bottom (a gate); one that stays
inside makes a window. Outlines should be convex or cross every vertical line at
most twice. Openings must not overlap along the path (a later one is ignored with
a warning), and they suit profiles with one horizontal interval, such as walls,
rails and kerbs, swept without twist.

Degenerate input never panics: fewer than two distinct path points (three when
closed) or fewer than three distinct profile points gives an empty mesh and a
warning. Repeated path points are dropped. On the inside of a tight bend a wide
profile can overlap itself; that is not corrected.

## Height bands

```rust
let terrain = Style::shaded(Color::WHITE, 0.5).with_height_bands(&[
    (0.0, Color::srgb(0.35, 0.6, 0.3)),   // grass
    (1.2, Color::srgb(0.5, 0.48, 0.45)),  // rock
    (2.4, Color::srgb(0.95, 0.95, 1.0)),  // snow
]);
```

`GradMeshMaterial` colours each fragment by its world height: a band's colour
holds from its height up to the next band, with a hard step and no blending. At
most four bands are used (more trigger a `debug_assert!` and are cut off); they
are sorted by height, and a fragment below the first band takes the first
colour. The bands are material floats, so raising terrain does not mean
re-baking vertex colours. They work on the material path only, not on the
instanced, standalone or Mesh2d paths.

See `examples/sweep.rs` and `examples/composed_shapes.rs`.
