# Custom Pipeline

## The Problem with ExtendedMaterial

By default, BevyGradmesh uses `GradMeshMaterial`, which is implemented as `ExtendedMaterial<StandardMaterial, GradMeshExtension>`. This means every unique material instance triggers a new shader specialization via Bevy's PBR pipeline. Each specialization allocates new descriptor sets.

On mobile GPUs (e.g., Android Mali-G715), the Vulkan descriptor pool fragments after ~20-30 such allocations, causing `VK_ERROR_FRAGMENTED_POOL` and a wgpu panic. Even with `GradMeshMaterialPool` deduplication, adding many materials can push past the limit.

The root issue: **ExtendedMaterial inherits the entire PBR pipeline**, including lighting, shadows, prepass, deferred rendering, and meshlet support — none of which GradMesh uses. Gradient shading is a simple two-color vertex shader that doesn't need PBR at all.

## The Standalone Pipeline

The standalone pipeline (`StandaloneRenderPlugin`) bypasses Bevy's PBR system entirely. It uses a pure WGSL shader (`standalone.wgsl`) with no Bevy PBR imports, where all material variation comes from per-instance vertex attributes rather than shader specialization.

### Key Benefits

- **Zero descriptor pool pressure** — One pipeline, one bind group layout, one descriptor set
- **No PBR overhead** — No prepass, no shadows, no deferred rendering
- **Mobile-friendly** — Works on Mali-G715, Adreno, and any GPU with limited descriptor pool capacity
- **Faster compilation** — One shader pipeline instead of N specializations

### Architecture

```
StandalonePipeline
├── Single render pipeline (no specialization per material)
├── Reuses Opaque3d binned render phase
├── Per-batch vertex buffer (transforms + properties)
└── Standalone WGSL shader (no PBR imports)
```

### Usage

The standalone pipeline is automatically enabled when you add `GradMeshPlugin`:

```rust
use bevy_gradmesh::{GradMeshPlugin, BatchRecorder, render_standalone_batches};

// GradMeshPlugin includes StandaloneRenderPlugin by default
app.add_plugins(GradMeshPlugin);
```

To use the standalone pipeline for rendering, use `render_standalone_batches()` instead of `BatchRecorder::render_batches()`:

```rust
use bevy_gradmesh::{BatchRecorder, ShapeDetail, render_standalone_batches};

fn render(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
    let mut recorder = BatchRecorder::new();
    recorder.begin();

    recorder.set_position(Vec3::ZERO);
    recorder.sphere(0.5, ShapeDetail::Medium, [1.0, 0.0, 0.0, 1.0]);

    recorder.set_position(Vec3::X);
    recorder.sphere(0.5, ShapeDetail::Medium, [0.0, 1.0, 0.0, 1.0]);

    // Use standalone pipeline — no PBR, no descriptor pool pressure
    let _entities = render_standalone_batches(&mut recorder, &mut commands, meshes, None);
}
```

### Standalone vs PBR Pipeline

| Feature | PBR Pipeline (`InstancedBatchData`) | Standalone Pipeline (`StandaloneBatchData`) |
|---|---|---|
| Shader | `gradmesh.wesl` (PBR imports) | `standalone.wgsl` (no imports) |
| Descriptor sets | One per material specialization | One shared across all batches |
| Lighting | Full PBR lighting | None (gradient shading only) |
| Shadows | Supported | Not supported |
| Mobile GPU safe | No (descriptor pool pressure) | Yes |
| Draw calls | One per batch | One per batch |

### When to Use Which

- **Use the PBR pipeline** (`InstancedBatchData`) when you need lighting, shadows, or deferred rendering integration
- **Use the standalone pipeline** (`StandaloneBatchData`) for mobile targets, large scenes with many materials, or when you only need gradient shading without PBR

### Limitations

- No PBR lighting or shadows
- No deferred rendering support
- No meshlet support
- Gradient shading computed in world space (not object space)

![Standalone Pipeline](../../screenshots/screenshot_standalone_pipeline.png)
