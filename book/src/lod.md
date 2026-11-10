# Level of Detail

Automatic distance-based LOD with dither cross-fade eliminates pop-in when shapes change detail level.

## How It Works

1. Each frame, the camera distance to each batch is computed
2. Based on distance, an LOD level (0 to 4) is selected
3. During transitions, both LOD levels render simultaneously with a Bayer dither mask
4. The dither pattern smoothly fades between levels, eliminating visual pop-in

## Configuration

```rust
# use bevy::prelude::*;
# use bevy_gradmesh::LodConfig;
# fn main() {}
fn configure_lod(mut commands: Commands) {
    let lod_config = LodConfig {
        // Distances where each level hands over to the next, nearest first
        thresholds: [5.0, 20.0, 50.0, 100.0],
        transition_width: 3.0, // Distance range for cross-fade
        hysteresis: 0.2,       // Prevents LOD flickering (0.0 to 1.0)
    };
    commands.insert_resource(lod_config);
}
```

## LOD Levels

| Level | Detail | Segments | Distance Range |
|-------|--------|----------|---------------|
| 4 | Extreme | 32 | 0 to `thresholds[0]` |
| 3 | High | 24 | `thresholds[0]` to `thresholds[1]` |
| 2 | Medium | 16 | `thresholds[1]` to `thresholds[2]` |
| 1 | Low | 12 | `thresholds[2]` to `thresholds[3]` |
| 0 | Minimal | 8 | beyond `thresholds[3]` |

## Statistics

Monitor LOD distribution each frame:

```rust
# use bevy::prelude::*;
# use bevy_gradmesh::InstancedStats;
# fn main() {}
fn log_stats(stats: Res<InstancedStats>) {
    info!("LOD distribution: {:?}", stats.lod_distribution);
    // Output: "LOD distribution: [12, 45, 230, 88, 125]"
}
```

## Performance

LOD reduces GPU workload for distant objects automatically. The dither cross-fade adds minimal overhead during transitions, and hysteresis prevents rapid LOD switching.
