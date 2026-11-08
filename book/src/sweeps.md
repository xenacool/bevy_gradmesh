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
    heights.into(),          // Arc<[f32]>, heights[x + y * size.x]
    UVec2::new(17, 17),      // samples
    Vec2::splat(16.0),       // world extent
    chunk_key,
);
```

A terrain chunk centred on the origin in XZ, with smooth normals from central
differences and UVs from 0 to 1. Skirts hang below the edges (a tenth of the
larger extent) to hide cracks between neighbouring chunks of different detail;
the depth is not an option, because a chunk without skirts shows cracks.
Equality and hashing use `key`, `size`, `extent` (and the holes and border
below), not the heights, so change `key` when the heights change. A size below 2
in either axis, or a wrong `heights` length, gives an empty mesh and a warning.

`Shape::bounds()` includes the skirts, as it measures the mesh;
`Shape::surface_bounds()` is the land alone. Neither builds anything.

### Holes and seamless chunk edges

```rust
let chunk = Shape::heightfield_with_options(
    heights.into(), size, extent, key,
    HeightfieldOptions::default()
        .with_holes(holes)          // Vec<bool>, (size.x - 1) * (size.y - 1) cells
        .with_border(HeightfieldBorder::new(west, east, north, south)),
);
```

A hole cell has no triangles and no skirt below it (a skirt hides seams between
chunks, and a hole is meant to be see-through); vertices that only holes touch
are left out of the mesh and the bounds. The border is the ring of neighbour
heights just outside the chunk (`west` and `east` hold `size.y` heights, `north`
and `south` hold `size.x`; corners are not needed). It is used for the normals of
the edge vertices only and is never meshed, so two chunks that share an edge
share its normals exactly and a crest on the border shows no crease. The mask and
the border are part of the shape's identity, so changing only them needs no new
key (they are walked when hashing; the heights are not). A mask or border of the
wrong length gives an empty mesh and a warning, like wrong heights.

## `Shape::sweep`

```rust
let wall = Shape::sweep(
    profile,                      // Vec<Vec2>: x across the path, y up
    path,                         // Vec<Vec3>: a polyline or baked spline positions
    SweepOptions::default().with_closed(true),
);
```

Sweeps a closed profile along a path as one mesh: walls, roads, rails and pipes.
Frames follow the path by parallel transport from `SweepOptions::up`, joints are
mitred (the sides keep a constant width; the mitre is clamped at four times the
width), and normals are smooth along the path. A corner of the profile sharper
than about 40 degrees stays crisp, gentler ones are smoothed.

`SweepOptions` and `SweepOpening` are `#[non_exhaustive]`: start from
`SweepOptions::default()` and use the `with_*` methods, so new options never break
your code.

Options:

* `closed` joins the last point to the first (no caps, no seam). A closed sweep
  never shows a seam: a fractional `twist` is rounded to the nearest whole turn,
  and a `roll_along` that does not return to its start (modulo a whole turn) has
  the difference spread evenly along the loop.
* `caps` closes the ends of an open sweep.
* `twist` turns the profile, in whole turns over the path.
* `roll_along`, `scale_along` and `tint_by_length` are keyframes along the path.
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

### Roll and banked roads

`roll_along` is `(ratio, radians)` keyframes with the same rules, and
`SweepOptions::roll_with(samples, f)` fills it from a closure. Roll turns the
(already scaled) profile about the path tangent, on top of the
parallel-transported frame, and adds to `twist`.

To give the frames yourself, for example from a baked spline with its own
banking, use `Shape::sweep_frames`:

```rust
let road = Shape::sweep_frames(
    profile,
    &baked.sweep_frames(Some(0), Some(1)),   // width channel 0, roll channel 1
    SweepOptions::default(),
);
```

Each `SweepFrame` has a `position`, `right`, `up` and `scale`. The sweep keeps them
as given (no parallel transport) and blends them where it needs extra rings.
`right × up` should point along the path; a left-handed frame keeps its `up` and
gets its `right` flipped, so the surface never turns inside out. A closed sweep
joins the last frame to the first without correcting the frames at the seam.

### Openings

```rust
let options = SweepOptions::default().with_openings(vec![
    SweepOpening::arch(0.30..0.38, 0.0, 1.2, 1.9, 12),  // a gate
    SweepOpening::rect(0.60..0.64, 0.9, 1.4),            // a window
]);
```

An opening is a real hole through the wall, with sill, lintel and jamb faces on
the inside. Its `outline` is in **wall-face coordinates**, not in the profile
plane: `x` runs `0..1` over the opening's `t` range (a ratio of the path length)
and `y` is the height above the profile origin. An outline that reaches the
profile's lowest height cuts the wall open at the bottom (a gate); one that stays
inside makes a window. Outlines should be convex or cross every vertical line at
most twice. Openings must not overlap along the path (a later one is ignored with
a warning), and they suit profiles with one horizontal interval, such as walls,
rails and kerbs, swept without twist. `SweepOpening::new(t, outline)` takes a
custom outline.

Degenerate input never panics: fewer than two distinct path points (three when
closed) or fewer than three distinct profile points gives an empty mesh and a
warning. Repeated path points are dropped. On the inside of a tight bend a wide
profile can overlap itself; that is not corrected.

## Splines

`CatmullRomSpline::closed` and `CatmullRomSpline::open` build a curve through
control points; an open one has one-sided end tangents and its bake puts a frame
exactly on the last point. `with_channel(&[f32])` attaches a number per control
point (width, roll, tilt...) that is interpolated like the position and wraps
around a closed loop. `bake` samples every channel at every frame;
`BakedSpline::sweep_frames(width, roll)` turns the baked frames into sweep frames.

`BakedSpline::nearest_t` searches the whole curve, so a point under a bridge may
match the bridge. `nearest_distance_near(point, around, window)` and
`nearest_t_in(point, range)` search only a stretch of the curve, so a kart stays on
its part of the road.

`relaxed(min_radius)` straightens tight turns but never collapses the loop: it
keeps at least half the length of the control polygon.
`relaxed_within(min_radius, keep)` lets you choose the share and returns
`Err(Relaxed { best, radius })` when no curve that keeps enough length is gentle
enough, so you can reject the design.

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
most four bands fit (given more, the four lowest are kept and a warning is
logged, the same in every build); they are sorted by height, and a fragment below
the first band takes the first colour. The bands are material floats, so raising terrain does not mean
re-baking vertex colours. They work on the material path only, not on the
instanced, standalone or Mesh2d paths.

To state the heights as fractions of a shape's own height, so that re-scaling the
land keeps the look, use `with_height_bands_on`:

```rust
let terrain = Style::shaded(Color::WHITE, 0.5).with_height_bands_on(
    &[(0.0, grass), (0.55, rock), (0.9, snow)],   // 0 = lowest land, 1 = highest
    &chunk,
    &transform,                                   // the placement the chunk is shown with
);
```

The fractions are resolved on the spot against `Shape::surface_bounds()` (the
land, without skirts) and the transform into ordinary world heights, so an
unresolved band can never reach the shader. Build the style again when the
heights or the placement change. Neighbouring chunks only agree on colour at
their shared edge with the absolute form, because each chunk has its own range.

## Morphing

`Shape::lerp` also morphs filled flat shapes of different kinds (rectangle, disc,
ellipse, regular polygon, star and arbitrary polygon) as outlines, and
`SdfMaterial2d::lerp` blends two SDF shapes (a circle becomes a rectangle through
its corner radius).

## Materials and animated colours

`GradMeshMaterialPool` does not keep its materials alive: it remembers their ids
and hands out strong handles. A material that nothing refers to any more is freed
by Bevy and forgotten by the pool, so a screen that animates colours holds only
the materials it shows now, and the "live `GradMeshMaterial` instances" warning
counts only those. Hold the handle for as long as the material is shown.
`MaterialWarning { threshold }` changes the warning threshold.

See `examples/sweep.rs` and `examples/composed_shapes.rs`.
