# Styles & Materials

## Dual-Directional Shading

The signature feature of BevyGradmesh. Colors a surface with a smooth gradient between two colors along a given direction.

```rust
# use bevy_gradmesh::{GradMeshMaterial, dual_directional};
# fn main() {}
fn directional_material() -> GradMeshMaterial {
    dual_directional(
        [0.6, 0.4, 0.3],   // dark color (RGB)
        [0.96, 0.8, 0.69], // light color (RGB)
        [0.0, 1.0, 0.0],   // gradient direction
    )
}
```

## Color Spaces

Gradients blend in OKLAB by default, which avoids the muddy midpoints that
RGB interpolation produces. Pick another space with `with_mix_space`:

```rust
# use bevy_gradmesh::{GradMeshMaterial, MixSpace, dual_directional, with_mix_space};
# fn main() {}
fn oklch_material() -> GradMeshMaterial {
    with_mix_space(
        dual_directional([0.6, 0.4, 0.3], [0.96, 0.8, 0.69], [0.0, 1.0, 0.0]),
        MixSpace::Oklch,
    )
}
```

| `MixSpace`  | Behaviour |
|-------------|-----------|
| `Oklab`     | Default. Perceptually uniform; steady lightness. |
| `Oklch`     | Sweeps hue along the shortest arc; stays vivid. |
| `LinearRgb` | Physically correct light mixing. |
| `Srgb`      | Mixes the picker values directly; cheapest, greyest midpoints. |

`Style` takes the same setting (`Style::with_mix_space`), and
`MixSpace::mix` / `Style::lerp` blend on the CPU in the same space, so an
animated blend matches the shader gradient. See `examples/mix_spaces.rs`.

## Blending Styles and Shapes

`Style::lerp(&other, t)` mixes colors in the style's space, interpolates
continuous values and snaps discrete ones (blend mode, dash count) to the
nearer end. `Shape::lerp(&other, t)` interpolates the parameters of two shapes
of the same kind and returns the nearer shape otherwise. Every distinct blended
shape is a distinct mesh, so for smooth animation prefer scaling a fixed shape.

Closed polygons (`arbitrary_polygon`, `extruded_polygon`, holes included) also
morph when their vertex counts differ: both outlines are grown to the larger
count by splitting their longest edges, the second is flipped to the first's
winding and rotated to the start that moves points the least, and per-vertex
colours blend along (a missing list counts as white). Open point lists such as
polylines and curves of different lengths still snap to the nearer shape.

## Reading Shape Bounds

`Shape::bounds()` returns a `ShapeBounds` (`min`, `max`, `center()`, `size()`,
`half_extents()`, `rect_xy()`) measured from the generated vertices, so it is
exact for every shape kind. `Shape::bounds_2d()` does the same for
`Shape::build_2d()`. Each call builds the mesh once; cache the result if you
read it every frame.

## Dual-Radial Shading

Gradient radiates from the center outward:

```rust
# use bevy_gradmesh::{GradMeshMaterial, dual_radial};
# fn main() {}
fn radial_material() -> GradMeshMaterial {
    dual_radial(
        [0.6, 0.4, 0.3],   // inner color
        [0.96, 0.8, 0.69], // outer color
    )
}
```

## Blend Modes

*Colour modes, blend modes, dash patterns and style presets. Blends and solids use the standalone pipeline and dashed rims are `Mesh2d`, so the example runs without PBR. Run `cargo run --example styles` to reproduce it.*

### Opaque
```rust
# use bevy_gradmesh::{GradMeshMaterial, opaque};
# fn main() {}
fn opaque_material() -> GradMeshMaterial {
    opaque(1.0, 0.0, 0.0)
}
```

### Transparent
```rust
# use bevy_gradmesh::{GradMeshMaterial, transparent};
# fn main() {}
fn transparent_material() -> GradMeshMaterial {
    transparent(1.0, 0.0, 0.0, 0.5)
}
```

### Additive (Glow Effect)
```rust
# use bevy_gradmesh::{GradMeshMaterial, additive};
# fn main() {}
fn additive_material() -> GradMeshMaterial {
    additive(1.0, 0.5, 0.0)
}
```

### Multiplicative
```rust
# use bevy_gradmesh::{GradMeshMaterial, multiplicative};
# fn main() {}
fn multiplicative_material() -> GradMeshMaterial {
    multiplicative(0.5, 0.5, 0.5)
}
```

### Screen
```rust
# use bevy_gradmesh::{GradMeshMaterial, screen};
# fn main() {}
fn screen_material() -> GradMeshMaterial {
    screen(1.0, 1.0, 0.0)
}
```

### Dither
```rust
# use bevy_gradmesh::{GradMeshMaterial, dither};
# fn main() {}
fn dither_material() -> GradMeshMaterial {
    dither(1.0, 0.0, 0.0, 0.5)
}
```

### Dashed Outlines

Use `Style::with_dash` to add a dash pattern to any style:

```rust
# use bevy::prelude::*;
# use bevy_gradmesh::{Dash, Style};
# fn main() {}
fn dashed_style() -> Style {
    Style::opaque(Color::WHITE).with_dash(Dash::fixed(4, 0.2))
}
```

## Custom Shaders

Attach any Bevy shader:

```rust
# use bevy::prelude::*;
# use bevy_gradmesh::Style;
# fn main() {}
fn custom_style(my_shader_handle: Handle<Shader>) -> Style {
    Style::custom(my_shader_handle, Color::srgb(1.0, 0.0, 0.0))
}
```

## PBR Properties

All materials support Bevy's PBR properties:

```rust
# use bevy::prelude::*;
# use bevy_gradmesh::{GradMeshMaterial, dual_directional};
# fn main() {}
fn pbr_material() -> GradMeshMaterial {
    let mut mat = dual_directional([0.2, 0.2, 0.8], [0.8, 0.2, 0.2], [0.0, 1.0, 0.0]);
    mat.base.perceptual_roughness = 0.1; // Polished surface
    mat.base.metallic = 1.0; // Metallic
    mat.base.emissive = Color::srgb(1.0, 0.0, 0.0).to_linear(); // Glow
    mat
}
```

*Transparency, emissive glow, metallic surfaces and combinations. Run `cargo run --example pbr_effects` to reproduce it.*
