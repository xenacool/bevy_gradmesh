# Deformation

Vertex shader deformation enables runtime shape animation without CPU mesh regeneration.

## Deformation Types

### Wobble
Sinusoidal displacement along the vertex normal:

```rust
use bevy_gradmesh::DeformationParams;

let deform = DeformationParams::wobble(
    0.2,  // amplitude
    2.0,  // frequency (cycles per second)
    0.0,  // phase offset
);
```

### Pulse
Uniform scale oscillation (breathing effect):

```rust
let deform = DeformationParams::pulse(
    0.1,  // ±10% scale oscillation
    1.0,  // 1 cycle per second
    0.0,  // phase
);
```

### Morph
Displacement toward/away from center:

```rust
let deform = DeformationParams::morph(
    0.3,  // displacement magnitude
    1.5,  // frequency
    0.5,  // phase
);
```

## Using with BatchRecorder

```rust
use bevy_gradmesh::{BatchRecorder, DeformationParams};

fn draw(mut recorder: ResMut<BatchRecorder>) {
    recorder.begin();
    
    recorder.set_deformation(
        DeformationParams::wobble(0.15, 3.0, 0.0)
    );
    recorder.set_position(Vec3::ZERO);
    recorder.sphere(0.5, ShapeDetail::Medium, [1.0, 0.0, 0.0, 1.0]);
    
    recorder.render_batches(&mut commands, &mut meshes, None);
}
```

## How It Works

1. Each instance carries a `DeformationParams` vec4 in its GPU buffer
2. The vertex shader reads this attribute and displaces vertices in local space
3. The global `Deformation` resource provides elapsed time for animation
4. All computation happens on the GPU — no CPU overhead per frame

## Performance

Deformation is free for batched instances since it runs entirely in the vertex shader. No additional draw calls or CPU work is needed.
