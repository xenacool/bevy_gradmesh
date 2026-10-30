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

## OKLAB Color Space

For perceptually uniform color interpolation:

```rust
use bevy_gradmesh::dual_directional_oklab;

let mat = dual_directional_oklab(
    [0.6, 0.4, 0.3],
    [0.96, 0.8, 0.69],
    [0.0, 1.0, 0.0],
);
```

OKLAB avoids muddy midpoints that occur with linear RGB interpolation.

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

See the [Dash Patterns](#dash-patterns) section below for full details.

## Style Builder API

The `Style` struct provides a fluent builder API for creating materials:

```rust
use bevy_gradmesh::{Style, Dash, ShapeType};
use bevy::color::Color;

// Opaque single color
let style = Style::opaque(Color::srgb(1.0, 0.0, 0.0));

// Transparent with alpha
let style = Style::transparent(Color::srgba(1.0, 0.0, 0.0, 0.5));

// Dual-directional gradient
let style = Style::dual_directional(
    Color::srgb(0.6, 0.4, 0.3),
    Color::srgb(0.96, 0.8, 0.69),
    Vec3::Y,
);

// Dual-directional with default up direction
let style = Style::dual_directional_up(
    Color::srgb(0.6, 0.4, 0.3),
    Color::srgb(0.96, 0.8, 0.69),
);

// Dual-directional with OKLAB interpolation
let style = Style::dual_directional_oklab(
    Color::srgb(0.6, 0.4, 0.3),
    Color::srgb(0.96, 0.8, 0.69),
    Vec3::Y,
);

// Dual-radial gradient
let style = Style::dual_radial(
    Color::srgb(1.0, 1.0, 1.0),
    Color::srgb(0.0, 0.0, 0.0),
);

// Blend modes
let style = Style::multiplicative(Color::srgb(0.5, 0.5, 0.5));
let style = Style::screen(Color::srgb(1.0, 1.0, 0.0));
let style = Style::dither(Color::srgba(1.0, 0.0, 0.0, 0.5));
let style = Style::subtractive(Color::srgb(0.8, 0.2, 0.2));
let style = Style::lighten(Color::srgb(0.2, 0.8, 0.2));
let style = Style::color_dodge(Color::srgb(0.9, 0.6, 0.1));
let style = Style::linear_burn(Color::srgb(0.2, 0.2, 0.8));
let style = Style::color_burn(Color::srgb(0.8, 0.2, 0.6));
let style = Style::darken(Color::srgb(0.15, 0.15, 0.3));

// Custom shader passthrough
let style = Style::custom(my_shader_handle, Color::srgb(1.0, 0.0, 0.0));

// Chaining modifiers
let style = Style::opaque(Color::srgb(0.9, 0.5, 0.2))
    .with_dash(Dash::fixed(8, 0.35))
    .with_shape_type(ShapeType::Disc)
    .with_oklab();

// Build the material
let material = style.build();
// Or register it directly in the asset server
let handle = style.build_material(&mut materials);
```

## Dash Patterns

Dash patterns create outlined/dashed shapes. The `Dash` struct controls the
pattern, while `ShapeType` tells the shader how to compute perimeter progress
for shape-aware dashes.

### Creating Dash Patterns

```rust
use bevy_gradmesh::{Dash, DashType};

// Fixed dashes (normal style)
let dash = Dash::fixed(8, 0.35); // 8 dashes, 35% gap

// Chevron-style V-shaped dashes
let dash = Dash::chevron(8, 0.35);

// Rounded end-cap dashes
let dash = Dash::rounded(8, 0.35);
```

### Shape Types

```rust
use bevy_gradmesh::ShapeType;

// Generic: diagonal UV sweep (default)
let style = Style::opaque(Color::WHITE).with_shape_type(ShapeType::Generic);

// Disc: angular progress around center
let style = Style::opaque(Color::WHITE)
    .with_shape_type(ShapeType::Disc)
    .with_dash(Dash::fixed(8, 0.35));

// Rectangle: perimeter tracing around edges
let style = Style::opaque(Color::WHITE)
    .with_shape_type(ShapeType::Rectangle)
    .with_dash(Dash::fixed(8, 0.35));
```

### Dash Properties

| Property | Type | Description |
|----------|------|-------------|
| `count` | `u32` | Number of dashes (1-4096), 0 disables |
| `spacing` | `f32` | Gap ratio between dashes (0.0-1.0) |
| `offset` | `f32` | Pattern offset |
| `style` | `DashType` | Normal, Chevron, or Rounded |
| `modifier` | `f32` | Effect intensity (-1.0 to 1.0) |

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
