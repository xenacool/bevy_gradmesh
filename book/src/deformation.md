# Deformation

Vertex shader deformation enables runtime shape animation without CPU mesh regeneration.

## Deformation Types

### Wobble
Sinusoidal displacement along the vertex normal:

```rust
# use bevy_gradmesh::DeformationParams;
# fn main() {}
fn wobble() -> DeformationParams {
    DeformationParams::wobble(
        0.2,  // amplitude
        2.0,  // frequency (cycles per second)
        0.0,  // phase offset
    )
}
```

### Pulse
Uniform scale oscillation (breathing effect):

```rust
# use bevy_gradmesh::DeformationParams;
# fn main() {}
fn pulse() -> DeformationParams {
    DeformationParams::pulse(
        0.1,  // 10% scale oscillation in each direction
        1.0,  // 1 cycle per second
        0.0,  // phase
    )
}
```

### Morph
Displacement toward/away from center:

```rust
# use bevy_gradmesh::DeformationParams;
# fn main() {}
fn morph() -> DeformationParams {
    DeformationParams::morph(
        0.3,  // displacement magnitude
        1.5,  // frequency
        0.5,  // phase
    )
}
```

*Wobble, pulse and morph compared against static originals. Run `cargo run --example deformation` to reproduce it.*

## Using with BatchRecorder

```rust
# use bevy::prelude::*;
# use bevy_gradmesh::{BatchRecorder, DeformationParams, ShapeDetail};
# fn main() {}
#[derive(Resource)]
struct Recorder(BatchRecorder);

fn draw(
    mut commands: Commands,
    meshes: ResMut<Assets<Mesh>>,
    mut recorder: ResMut<Recorder>,
) {
    let recorder = &mut recorder.0;
    recorder.begin();

    recorder.set_deformation(
        DeformationParams::wobble(0.15, 3.0, 0.0)
    );
    recorder.set_position(Vec3::ZERO);
    recorder.sphere(0.5, ShapeDetail::Medium, [1.0, 0.0, 0.0, 1.0]);

    recorder.end();
    recorder.render_batches(&mut commands, meshes, None);
}
```

## How It Works

1. Each instance carries a `DeformationParams` vec4 in its GPU buffer
2. The vertex shader reads this attribute and displaces vertices in local space
3. The global `Deformation` resource provides elapsed time for animation
4. All computation happens on the GPU - no CPU overhead per frame

## Performance

Deformation is free for batched instances since it runs entirely in the vertex shader. No additional draw calls or CPU work is needed.
