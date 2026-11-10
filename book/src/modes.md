# Rendering Modes

BevyGradmesh provides four rendering modes for different use cases.

## Direct Mode

The simplest approach: build a mesh and spawn it as a Bevy entity.

```rust
# use bevy::prelude::*;
# use bevy_gradmesh::{GradMeshMaterial, Shape, ShapeDetail, dual_directional};
# fn main() {}
fn spawn_sphere(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
) {
    let mesh = meshes.add(Shape::sphere(1.0, ShapeDetail::Medium).build());
    let mat = materials.add(dual_directional(
        [0.6, 0.4, 0.3], [0.96, 0.8, 0.69], [0.0, 1.0, 0.0]
    ));
    commands.spawn((Mesh3d(mesh), MeshMaterial3d(mat)));
}
```

**Best for**: Static scenes with a few shapes.

## Immediate Mode

Per-frame draw/dispose pattern using `ImmediateRecorder`. Supports push/pop state stacking and billboarding.

```rust
# use bevy::prelude::*;
# use bevy_gradmesh::{GradMeshMaterial, ImmediateRecorder, ShapeDetail, imm_opaque};
# fn main() {}
#[derive(Resource)]
struct Recorder(ImmediateRecorder);

fn draw_frame(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
    mut recorder: ResMut<Recorder>,
) {
    let recorder = &mut recorder.0;
    recorder.begin(&mut commands);

    recorder.push_state();
    recorder.set_position(Vec3::new(1.0, 0.0, 0.0));
    recorder.sphere(
        &mut commands,
        &mut meshes,
        &mut materials,
        0.5,
        ShapeDetail::Medium,
        imm_opaque(1.0, 0.0, 0.0),
    );
    recorder.pop_state();
}
```

**Best for**: Particle systems, animated effects, debug visualization.

*Per-frame particles, push/pop state, billboarding and line drawing. Run `cargo run --example immediate` to reproduce it.*

## Retained Mode

Persistent shapes managed by handle. Create once, update per-frame.

```rust
# use bevy::prelude::*;
# use bevy_gradmesh::{GradMeshHandle, RetainedWorld, ShapeDetail};
# fn main() {}
#[derive(Resource)]
struct Scene {
    world: RetainedWorld,
    ball: GradMeshHandle,
}

fn setup(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
    let mut world = RetainedWorld::new();
    let ball = world.sphere(
        &mut commands, &mut meshes, Vec3::ZERO, 1.0, ShapeDetail::Medium,
        [1.0, 0.0, 0.0, 1.0],
    );
    commands.insert_resource(Scene { world, ball });
}

fn update(mut scene: ResMut<Scene>, time: Res<Time>) {
    let ball = scene.ball;
    scene.world.set_position(ball, Vec3::new(0.0, time.elapsed_secs().sin(), 0.0));
}
```

**Best for**: Scenes with persistent shapes that change position or properties each frame.

*Persistent shapes updated each frame through handles. Run `cargo run --example retained` to reproduce it.*

## Batched Mode

Groups instances into batches for GPU instancing - thousands of shapes in far fewer draw calls.

```rust
# use bevy::prelude::*;
# use bevy_gradmesh::{AppearanceSnapshot, BatchRecorder, ShapeDetail};
# fn main() {}
#[derive(Resource)]
struct Recorder(BatchRecorder);

fn draw_frame(
    mut commands: Commands,
    meshes: ResMut<Assets<Mesh>>,
    mut recorder: ResMut<Recorder>,
) {
    let recorder = &mut recorder.0;
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

    recorder.end();
    recorder.render_batches(&mut commands, meshes, None);
}
```

**Best for**: Large numbers of similar shapes (terrain, crowds, particles).

*500 spheres drawn in one GPU batch with `BatchRecorder`. Run `cargo run --example instanced` to reproduce it.*

### Dirty Tracking

Batched mode tracks which batches changed each frame, uploading only dirty data:

```rust
# use bevy::prelude::*;
# use bevy_gradmesh::InstancedStats;
# fn main() {}
fn check_statistics(stats: Res<InstancedStats>) {
    info!("Batches: {}, Dirty: {}, Instances: {}",
        stats.batch_count, stats.dirty_batch_count, stats.total_instances
    );
}
```

### Choosing a Mode

| Scenario | Recommended Mode |
|----------|-----------------|
| Few static shapes | Direct |
| Per-frame particles | Immediate |
| Persistent animated objects | Retained |
| Thousands of instances | Batched |
