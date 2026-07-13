# filter_world_map_ransac_v3

v3 addresses two artifacts observed with the strict v2 configuration:

1. Small recesses were absorbed into the broad RANSAC/PCA wall plane.
2. Independent per-cell planes caused local density holes.

## Main changes

- Two-pass processing:
  - Pass 1 estimates a plane for every occupied filter cell.
  - Pass 2 classifies points using compatible planes from neighboring cells.
- Dominant residual-layer re-fit:
  - RANSAC first detects a broad wall using `fit-distance-threshold`.
  - Signed point-to-plane residuals are histogrammed.
  - Only the densest surface layer is used for the final PCA plane.
- Hole prevention:
  - If strict filtering leaves an occupied cell empty, the closest point is
    retained up to `min-output-points`.
- Parallel duplicate suppression:
  - A neighboring nearly-parallel plane is accepted only when its separation
    from the current cell plane is within `parallel-plane-tolerance`.

## Placement

```text
src/bin/filter_world_map_ransac_v3.rs
```

## Recommended first run for the reported map

```bash
cargo run --release --bin filter_world_map_ransac_v3 -- \
  data/output/debug/07122026/voxel-0.1_world_map.pcd \
  data/output/debug/07122026/debug/voxel-0.1_world_map_ransac_v3.pcd \
  --removed-output data/output/debug/07122026/debug/voxel-0.1_world_map_removed_v3.pcd \
  --voxel-size 0.15 \
  --neighbor-range 1 \
  --fit-distance-threshold 0.060 \
  --refit-distance-threshold 0.025 \
  --keep-distance-threshold 0.015 \
  --residual-bin-size 0.010 \
  --plane-vote-range 1 \
  --parallel-plane-tolerance 0.035 \
  --parallel-normal-angle 15 \
  --min-output-points 1 \
  --fallback-keep-distance 0.050 \
  --iterations 96 \
  --min-neighbors 16 \
  --min-inliers 8 \
  --min-inlier-ratio 0.20 \
  --max-planarity-ratio 0.30 \
  --min-planar-spread 0.04 \
  --max-planes 1
```

## Tuning

More detail preservation:

```text
--fallback-keep-distance 0.060
--min-output-points 1
--refit-distance-threshold 0.020
```

Fewer holes but slightly more retained noise:

```text
--plane-vote-range 2
--parallel-plane-tolerance 0.040
```

Stronger duplicate-wall removal:

```text
--parallel-plane-tolerance 0.020
--fallback-keep-distance 0.040
```

Thinner final walls:

```text
--keep-distance-threshold 0.010
```

The fallback does not restore all rejected points. It restores only the closest
finite number of points in a filter voxel, so a coherent small recess can remain
visible without restoring the entire thick wall layer.
