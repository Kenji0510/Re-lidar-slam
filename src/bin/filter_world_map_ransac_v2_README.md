# RANSAC global-map filter v2

This version separates plane detection thickness from final retained thickness.

- `--fit-distance-threshold`: broad threshold used to detect a rough/thick wall.
- `--keep-distance-threshold`: narrow threshold used to retain points around the refined plane.

Recommended first run:

```bash
cargo run --release --bin filter_world_map_ransac_v2 -- \
  data/output/debug/07122026/voxel-0.1_world_map.pcd \
  data/output/debug/07122026/debug/voxel-0.1_world_map_ransac_v2.pcd \
  --removed-output data/output/debug/07122026/debug/voxel-0.1_world_map_removed_v2.pcd \
  --voxel-size 0.30 \
  --neighbor-range 1 \
  --fit-distance-threshold 0.060 \
  --keep-distance-threshold 0.015 \
  --iterations 96 \
  --min-neighbors 16 \
  --min-inliers 8 \
  --min-inlier-ratio 0.20 \
  --max-planarity-ratio 0.30 \
  --min-planar-spread 0.04 \
  --max-planes 1
```

If too many wall regions still remain unmodeled:

```text
--fit-distance-threshold 0.08
--min-inlier-ratio 0.15
--max-planarity-ratio 0.40
```

If the retained wall is still too thick:

```text
--keep-distance-threshold 0.010
```

Do not add `--remove-nonplanar` initially. With the supplied run, more than 4.4 million
points were classified as non-planar; removing all of them would delete most trees,
edges, poles, and sparse structures.
