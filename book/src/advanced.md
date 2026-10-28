# Advanced Topics

## Frustum Culling

BevyGradmesh automatically culls instances outside the camera's view frustum. Each batch stores a mesh AABB, and per-instance transforms are tested against 6 frustum planes extracted from the view-projection matrix.

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
