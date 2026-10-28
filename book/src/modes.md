# Rendering Modes

BevyGradmesh provides four rendering modes for different use cases.

## Direct Mode

The simplest approach: build a mesh and spawn it as a Bevy entity.

```rust
let mesh = meshes.add(Shape::sphere(1.0, ShapeDetail::Medium).build());
let mat = materials.add(dual_directional(
    [0.6, 0.4, 0.3], [0.96, 0.8, 0.69], [0.0, 1.0, 0.0]
));
commands.spawn((Mesh3d(mesh), MeshMaterial3d(mat)));
```

**Best for**: Static scenes with a few shapes.

## Immediate Mode

Per-frame draw/dispose pattern using `ImmediateRecorder`. Supports push/pop state stacking and billboarding.

```rust
use bevy_gradmesh::{ImmediateRecorder, imm_opaque};

fn draw_frame(mut recorder: ResMut<ImmediateRecorder>) {
    recorder.begin();
    
    recorder.push_state();
    recorder.set_position(Vec3::new(1.0, 0.0, 0.0));
    recorder.sphere(0.5, ShapeDetail::Medium, imm_opaque([1.0, 0.0, 0.0, 1.0]));
    recorder.pop_state();
    
    recorder.render(&mut commands, &mut meshes);
}
```

**Best for**: Particle systems, animated effects, debug visualization.

## Retained Mode

Persistent shapes managed by handle. Create once, update per-frame.

```rust
use bevy_gradmesh::{RetainedWorld, GradMeshHandle};

fn setup(mut retained: ResMut<RetainedWorld>) {
    let handle = retained.create(
        Shape::sphere(1.0, ShapeDetail::Medium),
        imm_opaque([1.0, 0.0, 0.0, 1.0]),
    );
}

fn update(mut retained: ResMut<RetainedWorld>, handle: Res<GradMeshHandle>) {
    retained.set_position(*handle, Vec3::new(0.0, time.sin(), 0.0));
}
```

**Best for**: Scenes with persistent shapes that change position or properties each frame.

## Batched Mode

Groups instances into batches for GPU instancing — thousands of shapes in far fewer draw calls.

```rust
use bevy_gradmesh::{BatchRecorder, AppearanceSnapshot};

fn draw_frame(mut recorder: ResMut<BatchRecorder>) {
    recorder.begin();
    
    for i in 0..100 {
        recorder.set_position(Vec3::new(
            (i % 10) as f32,
            0.0,
            (i / 10) as f32,
        ));
        
        // Optional: add color animation
        recorder.set_appearance(
            AppearanceSnapshot::color_pulse(1.0, i as f32 / 100.0)
        );
        
        recorder.sphere(0.3, ShapeDetail::Medium, [1.0, 0.0, 0.0, 1.0]);
    }
    
    recorder.render_batches(&mut commands, &mut meshes, None);
}
```

**Best for**: Large numbers of similar shapes (terrain, crowds, particles).

### Dirty Tracking

Batched mode tracks which batches changed each frame, uploading only dirty data:

```rust
// Check statistics
let stats = stats.read();
info!("Batches: {}, Dirty: {}, Instances: {}",
    stats.batch_count, stats.dirty_batch_count, stats.total_instances
);
```

### Choosing a Mode

| Scenario | Recommended Mode |
|----------|-----------------|
| Few static shapes | Direct |
| Per-frame particles | Immediate |
| Persistent animated objects | Retained |
| Thousands of instances | Batched |
