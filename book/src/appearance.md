# Appearance Animation

Per-instance color animation resolved in the vertex shader without per-frame CPU updates.

## Animation Types

### Color Pulse
Blend between primary and secondary color over time:

```rust
use bevy_gradmesh::AppearanceSnapshot;

let appearance = AppearanceSnapshot::color_pulse(
    1.0,  // 1 cycle per second
    0.0,  // phase offset (0.0–1.0)
);
```

### Color Cycle
Shift through color hues continuously:

```rust
let appearance = AppearanceSnapshot::color_cycle(
    0.5,  // 0.5 cycles per second
    0.25, // starting hue offset
);
```

### Gradient Shift
Shift the gradient direction over time:

```rust
let appearance = AppearanceSnapshot::gradient_shift(
    0.3,  // rotation speed
    0.0,  // starting angle
);
```

### Static (No Animation)
```rust
let appearance = AppearanceSnapshot::static_appearance();
```

## Using with BatchRecorder

```rust
use bevy_gradmesh::{BatchRecorder, AppearanceSnapshot};

fn draw(mut recorder: ResMut<BatchRecorder>) {
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
    
    recorder.render_batches(&mut commands, &mut meshes, None);
}
```

## How It Works

1. Each instance carries an `AppearanceSnapshot` vec4 encoding speed, phase, and type
2. The vertex shader resolves the animated color at render time using global time
3. Different phase values create staggered animation across instances
4. No per-frame CPU updates needed — the GPU handles all animation

## Performance

Appearance animation is free for batched instances. The animation parameters are uploaded once with the instance data, and the vertex shader computes the animated color each frame.
