# Choosing a Backend

BevyGradmesh can draw its shapes through five render paths. They differ in what
they need from Bevy, what they can draw, and how they scale. Pick by the target
first (mobile or desktop, 2D or 3D), then by the number of shapes.

| Backend | Camera | Needs `bevy_pbr` | Lighting | Best for |
|---|---|---|---|---|
| Material (`GradMeshMaterial`) | 3D | yes | full PBR | A few to a few hundred lit shapes, picking, shadows, height bands |
| Instanced (`BatchRecorder`, `RetainedWorld`) | 3D | yes | PBR | Thousands of similar shapes, animated appearance |
| Standalone (`StandaloneRenderPlugin`) | 3D | no | none | Mobile GPUs, flat-shaded crowds, anything that must avoid PBR |
| Mesh2d gradient (`GradMeshMaterial2d`) | 2D | no | none | Any 2D shape that comes from `Shape::build_2d` (stars, polygons, outlines) |
| SDF 2D (`SdfMaterial2d`) | 2D | no | none | Discs, ellipses and rounded rectangles with rims, arcs and dashes |

## Material

`Shape::build()` plus `Style::build_material()` gives an ordinary Bevy
`Mesh3d` and `MeshMaterial3d`. The material extends `StandardMaterial`, so
shadows, fog, tonemapping and mesh picking all work. `MeshCache` and
`GradMeshMaterialPool` deduplicate meshes and materials.

Only this path has `Style::with_height_bands`.

## Instanced

`BatchRecorder` and `RetainedWorld` group instances by shape and draw each group
with one instanced draw call. Per-instance colour, appearance animation and
deformation live in the instance buffer. Lighting comes from Bevy's PBR
functions, so it needs the `pbr` feature.

## Standalone

`StandaloneRenderPlugin` draws the same batches without `bevy_pbr`: it uses its
own camera bind group and pipeline, and finds meshes through `MeshAllocator`.
There is no lighting, no alpha blending and no culling. It is the path to use on
GPUs where the PBR pipeline is a problem (it was verified on a Mali-G715 with
Bevy's `PbrPlugin` disabled).

*Gradient shapes drawn with one shader pipeline, no PBR material and no lights. Run `cargo run --example standalone` to reproduce it.*

## Mesh2d gradient

`Shape::build_2d()` returns flat geometry in the XY plane. Pair it with
`Style::build_2d()` (a `GradMeshMaterial2d`) and `Camera2d`. Every shape kind
works, but each distinct parameter set is a distinct mesh.

## SDF 2D

`SdfMaterial2d` draws a shape from a signed distance field on one shared quad
(`unit_quad()`). Edges are anti-aliased from screen-space derivatives, so no MSAA
is needed. Radius, rim width, arc, gradient and dash are material floats: changing
them creates no mesh and no pipeline.

```rust
let quad = meshes.add(unit_quad());
let material = SdfMaterial2d::disc(80.0)
    .with_rim(10.0)
    .with_arc(0.0, std::f32::consts::PI * 1.5)
    .with_dash(Dash2d::by_length(14.0, 8.0).with_snap(DashSnap::EndToEnd))
    .with_colors(Color::srgb(1.0, 0.35, 0.25), Color::srgb(1.0, 0.9, 0.35));
commands.spawn((
    Mesh2d(quad.clone()),
    Transform::from_scale(material.quad_scale()),
    MeshMaterial2d(materials.add(material)),
));
```

Gradients: `with_colors` (linear, along `with_gradient_direction`),
`with_radial_gradient()` (fitted to the shape bounds), `with_angular_gradient(start)`
(mirrored, no seam) and `with_angular_sweep(start)` (one turn, hard seam). All of
them blend in the chosen `MixSpace`.

Dashes (`Dash2d`) apply to rims. Give a length and a gap in world units, or a
count and a gap ratio. The snap mode decides how the pattern fits the outline:

* `Off` keeps the pattern as given and cuts it where the outline starts and ends.
* `Tiling` rounds the number of periods to a whole number and rescales the period,
  so a closed outline has no seam. It is the default.
* `EndToEnd` is like `Tiling` on a closed outline; on a partial arc the first dash
  starts at the arc start and the last dash ends at the arc end.

`examples/sdf_dash_gallery.rs` shows every mode on discs, an ellipse, rounded
rectangles and a partial arc. Two things to know: on an eccentric ellipse dashes
are spaced by angle, not by arc length, and at a sharp rectangle corner the
dashes meet in a mitre. `Transform` scale must be `SdfMaterial2d::quad_scale()`.

*Discs, ellipses, rounded rectangles, rims and animated arcs on one shared quad. Run `cargo run --example sdf_2d` to reproduce it.*

*Dashed rims compared across `Off`, `Tiling`, `EndToEnd` and count-based snap modes. Run `cargo run --example sdf_dash_gallery` to reproduce it.*

## Without PBR: the `pbr` feature and `for_mobile()`

The crate has a `pbr` cargo feature that is on by default. It gates `material.rs`,
`instanced_render.rs`, `GradMeshExtension` and the PBR parts of `GradMeshPlugin`.
For a PBR-free app, switch default features off and add the mobile plugin:

```toml
bevy_gradmesh = { version = "0.3", default-features = false }
```

```rust
app.add_plugins(GradMeshPlugin::for_mobile());
```

`for_mobile()` registers the standalone pipeline, the Mesh2d gradient material and
`SdfMaterial2d`. Without `pbr` the Material and Instanced backends and
`RetainedWorld::flush` are not available. Check that none of your other
dependencies enables `bevy_pbr` again (`cargo tree -e features -i bevy_pbr`).
