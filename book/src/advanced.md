# Advanced Topics

## Frustum Culling

BevyGradmesh automatically culls instances outside the camera's view frustum. Each batch stores a mesh AABB, and per-instance transforms are tested against 6 frustum planes extracted from the view-projection matrix.

```rust
# use bevy::prelude::*;
# use bevy_gradmesh::InstancedStats;
# fn main() {}
// Culling is automatic - no configuration needed
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
# use bevy::prelude::*;
# use bevy_gradmesh::MeshCache;
# fn main() {}
fn check_cache(cache: Res<MeshCache>) {
    info!("Cache hit rate: {:.1}%", cache.hit_rate() * 100.0);
}
```

## Render State Control

Control depth test, depth write, and render layer per-batch:

```rust
# use bevy_gradmesh::{BatchRecorder, DepthTest, DepthWrite, RenderState};
# fn main() {}
fn set_state(recorder: &mut BatchRecorder) {
    recorder.set_render_state(RenderState {
        layer: 0,
        depth_test: DepthTest::LessEqual,
        depth_write: DepthWrite::Enabled,
    });
}
```

## Custom Shader Passthrough

Use any Bevy shader with gradmesh shapes:

```rust
# use bevy::prelude::*;
# use bevy_gradmesh::Style;
# fn main() {}
fn custom_style(my_shader_handle: Handle<Shader>) -> Style {
    Style::custom(my_shader_handle, Color::srgb(1.0, 0.0, 0.0))
}
```

## Physics Integration

BevyGradmesh meshes work with any Bevy physics plugin. Read the vertices and
triangles from the built mesh and pass them to the plugin's trimesh or convex
hull constructor:

```rust
# use bevy::prelude::*;
# use bevy_gradmesh::Shape;
# fn main() {}
fn collider_data(shape: &Shape) -> Option<(Vec<Vec3>, Vec<[u32; 3]>)> {
    let mesh = shape.build();
    let positions = mesh.attribute(Mesh::ATTRIBUTE_POSITION)?.as_float3()?;
    let vertices: Vec<Vec3> = positions.iter().map(|p| Vec3::from(*p)).collect();
    let indices: Vec<u32> = mesh.indices()?.iter().map(|i| i as u32).collect();
    let triangles = indices.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
    Some((vertices, triangles))
}
```

## Performance Tuning

1. **Use batched mode** for large numbers of similar shapes
2. **Enable LOD** for scenes with varying camera distances
3. **Check dirty tracking** to minimize GPU uploads
4. **Use appropriate detail levels** - don't use Extreme for distant objects
5. **Monitor InstancedStats** each frame during development

## Limitations

- No Order-Independent Transparency (OIT)
- No planar reflections
- No visual editor integration (Bevy has no editor)
- NURBS curves not supported (only cubic Bezier)
- No secondary LOD membership during transitions

## License

Dual-licensed under MIT or Apache-2.0.
