# Level of Detail

Automatic distance-based LOD with dither cross-fade eliminates pop-in when shapes change detail level.

## How It Works

1. Each frame, the camera distance to each batch is computed
2. Based on distance, an LOD level (0–4) is selected
3. During transitions, both LOD levels render simultaneously with a Bayer dither mask
4. The dither pattern smoothly fades between levels, eliminating visual pop-in

## Configuration

```rust
use bevy_gradmesh::LodConfig;

let lod_config = LodConfig {
    near_distance: 5.0,    // Extreme detail (level 4)
    mid_distance: 20.0,    // Medium detail (level 2)
    far_distance: 50.0,    // Minimal detail (level 0)
    transition_width: 3.0, // Distance range for cross-fade
    hysteresis: 1.0,       // Prevents LOD flickering
};
commands.insert_resource(lod_config);
```

## LOD Levels

| Level | Detail | Segments | Distance Range |
|-------|--------|----------|---------------|
| 4 | Extreme | 32 | 0–near |
| 3 | High | 24 | near–mid |
| 2 | Medium | 16 | mid–far |
| 1 | Low | 12 | far–very far |
| 0 | Minimal | 8 | very far |

## Statistics

Monitor LOD distribution each frame:

```rust
use bevy_gradmesh::InstancedStats;

fn log_stats(stats: Res<InstancedStats>) {
    stats.log();
    // Output: "LOD distribution: [12, 45, 230, 88, 125]"
}
```

## Performance

LOD reduces GPU workload for distant objects automatically. The dither cross-fade adds minimal overhead during transitions, and hysteresis prevents rapid LOD switching.
