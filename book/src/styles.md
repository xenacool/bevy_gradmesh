# Styles & Materials

## Dual-Directional Shading

The signature feature of BevyGradmesh. Colors a surface with a smooth gradient between two colors along a given direction.

```rust
use bevy_gradmesh::dual_directional;

let mat = dual_directional(
    [0.6, 0.4, 0.3],  // dark color (RGB)
    [0.96, 0.8, 0.69], // light color (RGB)
    [0.0, 1.0, 0.0],   // gradient direction
);
```

## Color Spaces

Gradients blend in OKLAB by default, which avoids the muddy midpoints that
RGB interpolation produces. Pick another space with `with_mix_space`:

```rust
use bevy_gradmesh::{MixSpace, dual_directional, with_mix_space};

let mat = with_mix_space(
    dual_directional([0.6, 0.4, 0.3], [0.96, 0.8, 0.69], [0.0, 1.0, 0.0]),
    MixSpace::Oklch,
);
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

## Dual-Radial Shading

Gradient radiates from the center outward:

```rust
use bevy_gradmesh::dual_radial;

let mat = dual_radial(
    [0.6, 0.4, 0.3],
    [0.96, 0.8, 0.69],
);
```

## Blend Modes

### Opaque
```rust
use bevy_gradmesh::opaque;
let mat = opaque([1.0, 0.0, 0.0, 1.0]);
```

### Transparent
```rust
use bevy_gradmesh::transparent;
let mat = transparent([1.0, 0.0, 0.0, 0.5]);
```

### Additive (Glow Effect)
```rust
use bevy_gradmesh::additive;
let mat = additive([1.0, 0.5, 0.0, 1.0]);
```

### Multiplicative
```rust
use bevy_gradmesh::multiplicative;
let mat = multiplicative([0.5, 0.5, 0.5, 1.0]);
```

### Screen
```rust
use bevy_gradmesh::screen;
let mat = screen([1.0, 1.0, 0.0, 1.0]);
```

### Dither
```rust
use bevy_gradmesh::dither;
let mat = dither([1.0, 0.0, 0.0, 0.5]);
```

### Dashed Outlines
```rust
use bevy_gradmesh::{dashed, Dash, DashType};
let mat = dashed(
    [1.0, 1.0, 1.0, 1.0],
    Dash::new(4, DashType::Normal, 0.2, 0.0),
);
```

## Custom Shaders

Attach any Bevy shader:

```rust
use bevy_gradmesh::BlendMode;
use bevy::asset::Handle;

let style = Style::custom(my_shader_handle, [1.0, 0.0, 0.0, 1.0]);
```

## PBR Properties

All materials support Bevy's PBR properties:

```rust
let mut mat = dual_directional([...], [...], [...]);
mat.roughness = 0.1;    // Polished surface
mat.metallic = 1.0;     // Metallic
mat.emissive = Color::srgb(1.0, 0.0, 0.0).to_linear(); // Glow
```
