# Appearance Animation

Per-instance color animation resolved in the vertex shader without per-frame CPU updates.

## Animation Types

### Color Pulse
Blend between primary and secondary color over time:

```rust
# use bevy_gradmesh::AppearanceSnapshot;
# fn main() {}
fn pulse() -> AppearanceSnapshot {
    AppearanceSnapshot::color_pulse(
        1.0,  // 1 cycle per second
        0.0,  // phase offset (0.0 to 1.0)
    )
}
```

### Color Cycle
Shift through color hues continuously:

```rust
# use bevy_gradmesh::AppearanceSnapshot;
# fn main() {}
fn cycle() -> AppearanceSnapshot {
    AppearanceSnapshot::color_cycle(
        0.5,  // 0.5 cycles per second
        0.25, // starting hue offset
    )
}
```

### Gradient Shift
Shift the gradient direction over time:

```rust
# use bevy_gradmesh::AppearanceSnapshot;
# fn main() {}
fn shift() -> AppearanceSnapshot {
    AppearanceSnapshot::gradient_shift(
        0.3,  // rotation speed
        0.0,  // starting angle
    )
}
```

### Static (No Animation)
```rust
# use bevy_gradmesh::AppearanceSnapshot;
# fn main() {}
fn still() -> AppearanceSnapshot {
    AppearanceSnapshot::static_appearance()
}
```

*Static, colour pulse, colour cycle and gradient shift rows; the phase is staggered over the instances, and the shader animates what the CPU uploaded once. Run `cargo run --example appearance` to reproduce it.*

## Using with BatchRecorder

```rust
# use bevy::prelude::*;
# use bevy_gradmesh::{AppearanceSnapshot, BatchRecorder, ShapeDetail};
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

    for i in 0..50 {
        recorder.set_position(Vec3::new(
            (i % 10) as f32 * 2.0,
            0.0,
            (i / 10) as f32 * 2.0,
        ));

        // Each instance animates with a different phase
        recorder.set_appearance(
            AppearanceSnapshot::color_pulse(2.0, i as f32 / 50.0)
        );

        recorder.sphere(0.3, ShapeDetail::Medium, [1.0, 0.0, 0.0, 1.0]);
    }

    recorder.end();
    recorder.render_batches(&mut commands, meshes, None);
}
```

## How It Works

1. Each instance carries an `AppearanceSnapshot` vec4 encoding speed, phase, and type
2. The vertex shader resolves the animated color at render time using global time
3. Different phase values create staggered animation across instances
4. No per-frame CPU updates needed - the GPU handles all animation

## Performance

Appearance animation is free for batched instances. The animation parameters are uploaded once with the instance data, and the vertex shader computes the animated color each frame.
