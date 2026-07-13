# filter_world_map_ransac_v4

v4 fixes the wall-thickening introduced by v3 neighbor-plane voting.

## Cause in v3

v3 retained a point when it was close to **any** compatible plane in surrounding
cells. If neighboring cells estimated slightly shifted parallel planes, the
union of all narrow slabs became one thick band.

## v4 processing

1. Estimate one raw RANSAC/PCA plane per occupied filter cell.
2. Cluster neighboring raw planes by normal direction.
3. Within the dominant normal cluster, find the densest plane-offset layer.
4. Build one consensus surface plane per cell.
5. For each occupied cell, score nearby consensus planes against the cell's own
   points.
6. Select exactly one reference plane.
7. Keep only points near that one plane.
8. If no strict point survives, retain a finite fallback point to avoid a hole.

## Placement

```text
src/bin/filter_world_map_ransac_v4.rs
```

## Recommended command

```bash
cargo run --release --bin filter_world_map_ransac_v4 -- \
  data/output/debug/07122026/voxel-0.1_world_map.pcd \
  data/output/debug/07122026/debug/voxel-0.1_world_map_ransac_v4.pcd \
  --removed-output data/output/debug/07122026/debug/voxel-0.1_world_map_removed_v4.pcd \
  --voxel-size 0.15 \
  --neighbor-range 1 \
  --fit-distance-threshold 0.060 \
  --refit-distance-threshold 0.025 \
  --keep-distance-threshold 0.015 \
  --residual-bin-size 0.010 \
  --plane-vote-range 1 \
  --parallel-plane-tolerance 0.020 \
  --consensus-normal-angle 12 \
  --min-consensus-planes 1 \
  --reference-plane-range 1 \
  --reference-support-distance 0.060 \
  --min-output-points 1 \
  --fallback-keep-distance 0.040 \
  --iterations 96 \
  --min-neighbors 16 \
  --min-inliers 8 \
  --min-inlier-ratio 0.20 \
  --max-planarity-ratio 0.30 \
  --min-planar-spread 0.04 \
  --max-planes 1
```

## Tuning for thinner walls

```text
--keep-distance-threshold 0.010
--parallel-plane-tolerance 0.015
--consensus-normal-angle 8
```

Use one change at a time. The most direct wall-thickness control is
`keep-distance-threshold`.

## Tuning if density holes return

```text
--min-output-points 1
--fallback-keep-distance 0.050
--reference-plane-range 2
```

Unlike v3, increasing `reference-plane-range` does not retain the union of all
neighbor planes. It only gives the cell more candidates, after which exactly one
reference plane is selected.

## Useful v4 log values

- `Consensus surface planes`
- `Own-reference cells`
- `Neighbor-reference cells`
- `No-reference cells`
- `Kept fallback points`

A very large `Neighbor-reference cells` count is acceptable because only one
neighbor plane is selected per cell. A large `No-reference cells` count means
RANSAC or consensus requirements are too strict.
