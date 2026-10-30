# Advanced Topics

## Frustum Culling

BevyGradmesh automatically culls instances outside the camera's view frustum.
Each batch stores a mesh AABB (axis-aligned bounding box), and per-instance
transforms are tested against 6 frustum planes extracted from the view-projection
matrix. Only visible instances contribute to draw calls.

Culling works for both the standard instanced pipeline and the standalone pipeline.

```rust
// Culling is automatic — no configuration needed
// Check stats to see how many instances were culled
fn log_culling(stats: Res<InstancedStats>) {
    info!("Culled: {}/{} ({:.1}%)",
        stats.culled_instances,
        stats.total_instances_before_cull,
        stats.culled_instances as f32 / stats.total_instances_before_cull as f32 * 100.0
    );
}
```

### How It Works

1. Each mesh has a precomputed AABB in local space
2. Per-instance transforms are applied to the AABB to get world-space bounds
3. The camera's 6 frustum planes are extracted from the clip matrix
4. Instances whose AABBs fall entirely outside any plane are excluded
5. Only visible instances are counted in `stats.total_instances`

See the `culling_demo` example for a visual demonstration with a 10×10 grid.

![Frustum Culling](../../screenshots/screenshot_culling_demo.png)

## Voxel Batch Recorder

`VoxelBatchRecorder` extends `BatchRecorder` with spatial partitioning. Instances
are grouped into 3D voxel grid cells based on their position, enabling spatial
coherence for culling and rendering. Each voxel cell maintains its own batch
recorder, so instances in the same cell share draw calls.

```rust
use bevy_gradmesh::{VoxelBatchRecorder, Shape, ShapeDetail};

let mut voxel_rec = VoxelBatchRecorder::new(1.0); // 1 unit per voxel cell

voxel_rec.begin();

// Instances near each other share a voxel
voxel_rec.set_position(Vec3::new(0.5, 0.5, 0.5));
voxel_rec.sphere(0.1, ShapeDetail::Low, [1.0, 0.0, 0.0, 1.0]);

voxel_rec.set_position(Vec3::new(1.5, 0.5, 0.5));
voxel_rec.sphere(0.1, ShapeDetail::Low, [0.0, 1.0, 0.0, 1.0]);

// Render all voxels
let entities = voxel_rec.render_all(&mut commands, &mut meshes, None);
```

### When to Use

| Scenario | Use |
|----------|-----|
| Simple batched rendering | `BatchRecorder` |
| Spatial queries + batched rendering | `VoxelBatchRecorder` |
| Per-cell culling or LOD | `VoxelBatchRecorder` |

### API

- `VoxelBatchRecorder::new(voxel_size)` — create with cell size in world units
- `voxel_count()` — number of non-empty cells
- `has_voxel(coords)` — check if a cell has instances
- `coords_for_position(pos)` — get voxel coordinates for a world position

## Mesh Caching

The `MeshCache` resource avoids regenerating identical geometry:

```rust
use bevy_gradmesh::MeshCache;

fn check_cache(cache: Res<MeshCache>) {
    info!("Cache hit rate: {:.1}%", cache.hit_rate() * 100.0);
}
```

## Render State Control

Control depth test, depth write, and render layer per-batch:

```rust
use bevy_gradmesh::{RenderState, DepthTest, DepthWrite};

recorder.set_render_state(RenderState {
    layer: 0,
    depth_test: DepthTest::LessEqual,
    depth_write: DepthWrite::On,
});
```

## Custom Shader Passthrough

Use any Bevy shader with gradmesh shapes:

```rust
use bevy_gradmesh::{BlendMode, Style};

let custom_style = Style::custom(my_shader_handle, [1.0, 0.0, 0.0, 1.0]);
```

## Physics Integration

BevyGradmesh meshes work with any Bevy physics plugin:

```rust
use avian3d::prelude::*;

// Exact triangle-mesh collision
let collider = Collider::trimesh_from_mesh(&mesh);

// Convex approximation
let collider = Collider::convex_hull_from_mesh(&mesh);
```

## Performance Tuning

1. **Use batched mode** for large numbers of similar shapes
2. **Enable LOD** for scenes with varying camera distances
3. **Check dirty tracking** to minimize GPU uploads
4. **Use appropriate detail levels** — don't use Extreme for distant objects
5. **Monitor InstancedStats** each frame during development

## Limitations

- No Order-Independent Transparency (OIT)
- No planar reflections
- No visual editor integration (Bevy has no editor)
- NURBS curves not supported (only cubic Bézier)
- No secondary LOD membership during transitions
