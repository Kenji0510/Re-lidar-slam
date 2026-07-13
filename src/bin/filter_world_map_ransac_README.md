# filter_world_map_ransac.rs

Place the file at:

```text
src/bin/filter_world_map_ransac.rs
```

The code reuses the existing project modules:

```rust
use re_lidar_slam::{
    file_handler::save_pcd_xyz,
    types::PointXYZ,
};
```

No new crate dependency is required beyond crates already used by the uploaded project:
`anyhow`, `nalgebra`, `pcd-rs`, and `rayon`.

## Run

```bash
cargo run --release --bin filter_world_map_ransac -- \
  data/output/debug/07122026/voxel-0.1_world_map.pcd \
  data/output/debug/07122026/voxel-0.1_world_map_ransac.pcd \
  --removed-output data/output/debug/07122026/voxel-0.1_world_map_removed.pcd
```

## Recommended initial values

The defaults are intended for an input map downsampled to approximately 0.1 m:

```text
filter voxel size       0.20 m
neighbor range          1 cell (3 x 3 x 3)
RANSAC threshold        0.035 m
iterations              64
minimum neighborhood    12 points
minimum inliers         8 points
minimum inlier ratio    0.45
maximum planes          2
```

The default keeps regions where no reliable plane is found. This protects trees, poles,
edges, curved surfaces, and sparse areas.

To output only reliable planar inliers, add:

```bash
--remove-nonplanar
```

For stronger wall cleanup, first try:

```bash
--distance-threshold 0.025 --max-planarity-ratio 0.10
```

For a map that becomes too sparse, first try:

```bash
--distance-threshold 0.05 --min-inlier-ratio 0.35
```
