use crate::voxel_map::{VoxelCell, VoxelMap, voxel_key};
use nalgebra::{Point3, Vector3};
use nohash_hasher::IntMap;
use rayon::prelude::*;

type FastMap<V> = IntMap<u64, V>;

#[derive(Clone, Debug)]
struct VoxelStat {
    sum: Vector3<f32>,
    count: usize,
}

pub struct DownsampledPointCloud {
    pub points: Vec<Point3<f32>>,
    pub voxel_map: VoxelMap,
}

impl Default for VoxelStat {
    fn default() -> Self {
        Self {
            sum: Vector3::zeros(),
            count: 0,
        }
    }
}

impl VoxelStat {
    #[inline]
    fn add_point(&mut self, p: &Point3<f32>) {
        self.sum.x += p.x;
        self.sum.y += p.y;
        self.sum.z += p.z;
        self.count += 1;
    }

    #[inline]
    fn merge(&mut self, other: &VoxelStat) {
        self.sum += other.sum;
        self.count += other.count;
    }

    #[inline]
    fn centroid(&self) -> Point3<f32> {
        let inv_count = 1.0 / self.count as f32;
        Point3::from(self.sum * inv_count)
    }
}

#[inline]
fn morton3d(ix: u32, iy: u32, iz: u32) -> u64 {
    #[inline]
    fn part1by2(n: u32) -> u64 {
        let mut x = n as u64 & 0x1f_ffff; // 21 bit
        x = (x | (x << 32)) & 0x001f_0000_0000_ffff;
        x = (x | (x << 16)) & 0x001f_0000_ff00_00ff;
        x = (x | (x << 8)) & 0x100f_00f0_0f00_f00f;
        x = (x | (x << 4)) & 0x10c3_0c30_c30c_30c3;
        x = (x | (x << 2)) & 0x1249_2492_4924_9249;
        x
    }

    part1by2(ix) | (part1by2(iy) << 1) | (part1by2(iz) << 2)
}

pub fn voxel_downsample_points(points: &[Point3<f32>], voxel_size: f32) -> Vec<Point3<f32>> {
    if points.is_empty() {
        return Vec::new();
    }

    assert!(voxel_size > 0.0, "voxel_size must be positive");

    let inv_voxel = 1.0 / voxel_size;
    let (min_corner, max_corner) = point_cloud_bounds(points);
    warn_if_morton_extent_exceeded(&min_corner, &max_corner, inv_voxel);

    let global_map = build_stat_map(points, &min_corner, inv_voxel);
    stat_map_into_centroids(global_map)
}

/// Downsample the same point cloud at two resolutions in one parallel pass.
///
/// Each resolution keeps an independent accumulator, so voxel membership and
/// centroid calculation are identical to two calls to [`voxel_downsample_points`].
/// The point-cloud bounds calculation and Rayon traversal are shared.
pub fn voxel_downsample_points_dual(
    points: &[Point3<f32>],
    first_voxel_size: f32,
    second_voxel_size: f32,
) -> (Vec<Point3<f32>>, Vec<Point3<f32>>) {
    if points.is_empty() {
        return (Vec::new(), Vec::new());
    }

    assert!(first_voxel_size > 0.0, "voxel_size must be positive");
    assert!(second_voxel_size > 0.0, "voxel_size must be positive");

    let first_inv_voxel = 1.0 / first_voxel_size;
    let second_inv_voxel = 1.0 / second_voxel_size;
    let (min_corner, max_corner) = point_cloud_bounds(points);

    warn_if_morton_extent_exceeded(&min_corner, &max_corner, first_inv_voxel);
    warn_if_morton_extent_exceeded(&min_corner, &max_corner, second_inv_voxel);

    let (first_map, second_map) =
        build_dual_stat_maps(points, &min_corner, first_inv_voxel, second_inv_voxel);

    (
        stat_map_into_centroids(first_map),
        stat_map_into_centroids(second_map),
    )
}

/// Downsample at two resolutions and build the corresponding source voxel maps.
///
/// The maps are populated while the centroid vectors are emitted, preserving
/// the insertion and representative-point selection performed by
/// `build_voxel_map(points, voxel_size, _, false)` without hashing the vectors
/// in a separate pass.
pub fn voxel_downsample_points_and_maps_dual(
    points: &[Point3<f32>],
    first_voxel_size: f32,
    second_voxel_size: f32,
) -> (DownsampledPointCloud, DownsampledPointCloud) {
    if points.is_empty() {
        return (
            DownsampledPointCloud {
                points: Vec::new(),
                voxel_map: VoxelMap::default(),
            },
            DownsampledPointCloud {
                points: Vec::new(),
                voxel_map: VoxelMap::default(),
            },
        );
    }

    assert!(first_voxel_size > 0.0, "voxel_size must be positive");
    assert!(second_voxel_size > 0.0, "voxel_size must be positive");

    let first_inv_voxel = 1.0 / first_voxel_size;
    let second_inv_voxel = 1.0 / second_voxel_size;
    let (min_corner, max_corner) = point_cloud_bounds(points);

    warn_if_morton_extent_exceeded(&min_corner, &max_corner, first_inv_voxel);
    warn_if_morton_extent_exceeded(&min_corner, &max_corner, second_inv_voxel);

    let (first_map, second_map) =
        build_dual_stat_maps(points, &min_corner, first_inv_voxel, second_inv_voxel);

    (
        stat_map_into_point_cloud(first_map, first_voxel_size),
        stat_map_into_point_cloud(second_map, second_voxel_size),
    )
}

fn point_cloud_bounds(points: &[Point3<f32>]) -> (Vector3<f32>, Vector3<f32>) {
    points
        .par_iter()
        .fold(
            || {
                (
                    Vector3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
                    Vector3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
                )
            },
            |(mut min_v, mut max_v), p| {
                min_v.x = min_v.x.min(p.x);
                min_v.y = min_v.y.min(p.y);
                min_v.z = min_v.z.min(p.z);

                max_v.x = max_v.x.max(p.x);
                max_v.y = max_v.y.max(p.y);
                max_v.z = max_v.z.max(p.z);

                (min_v, max_v)
            },
        )
        .reduce(
            || {
                (
                    Vector3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
                    Vector3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
                )
            },
            |(min_a, max_a), (min_b, max_b)| {
                (
                    Vector3::new(
                        min_a.x.min(min_b.x),
                        min_a.y.min(min_b.y),
                        min_a.z.min(min_b.z),
                    ),
                    Vector3::new(
                        max_a.x.max(max_b.x),
                        max_a.y.max(max_b.y),
                        max_a.z.max(max_b.z),
                    ),
                )
            },
        )
}

fn warn_if_morton_extent_exceeded(
    min_corner: &Vector3<f32>,
    max_corner: &Vector3<f32>,
    inv_voxel: f32,
) {
    let max_idx_x = ((max_corner.x - min_corner.x) * inv_voxel).floor() as u64;
    let max_idx_y = ((max_corner.y - min_corner.y) * inv_voxel).floor() as u64;
    let max_idx_z = ((max_corner.z - min_corner.z) * inv_voxel).floor() as u64;

    if max_idx_x > 0x1f_ffff || max_idx_y > 0x1f_ffff || max_idx_z > 0x1f_ffff {
        eprintln!(
            "Warning: point cloud extent exceeds Morton code 21-bit limit. Key collision may occur."
        );
    }
}

fn build_stat_map(
    points: &[Point3<f32>],
    min_corner: &Vector3<f32>,
    inv_voxel: f32,
) -> FastMap<VoxelStat> {
    points
        .par_iter()
        .fold(FastMap::<VoxelStat>::default, |mut local_map, point| {
            add_point_to_map(&mut local_map, point, min_corner, inv_voxel);
            local_map
        })
        .reduce(FastMap::<VoxelStat>::default, |mut map_a, map_b| {
            merge_stat_maps(&mut map_a, map_b);
            map_a
        })
}

fn build_dual_stat_maps(
    points: &[Point3<f32>],
    min_corner: &Vector3<f32>,
    first_inv_voxel: f32,
    second_inv_voxel: f32,
) -> (FastMap<VoxelStat>, FastMap<VoxelStat>) {
    points
        .par_iter()
        .fold(
            || {
                (
                    FastMap::<VoxelStat>::default(),
                    FastMap::<VoxelStat>::default(),
                )
            },
            |(mut first_map, mut second_map), point| {
                add_point_to_map(&mut first_map, point, min_corner, first_inv_voxel);
                add_point_to_map(&mut second_map, point, min_corner, second_inv_voxel);
                (first_map, second_map)
            },
        )
        .reduce(
            || {
                (
                    FastMap::<VoxelStat>::default(),
                    FastMap::<VoxelStat>::default(),
                )
            },
            |(mut first_a, mut second_a), (first_b, second_b)| {
                merge_stat_maps(&mut first_a, first_b);
                merge_stat_maps(&mut second_a, second_b);
                (first_a, second_a)
            },
        )
}

#[inline]
fn add_point_to_map(
    map: &mut FastMap<VoxelStat>,
    point: &Point3<f32>,
    min_corner: &Vector3<f32>,
    inv_voxel: f32,
) {
    let ix = ((point.x - min_corner.x) * inv_voxel).floor().max(0.0) as u32;
    let iy = ((point.y - min_corner.y) * inv_voxel).floor().max(0.0) as u32;
    let iz = ((point.z - min_corner.z) * inv_voxel).floor().max(0.0) as u32;

    let key = morton3d(ix, iy, iz);
    map.entry(key).or_default().add_point(point);
}

fn merge_stat_maps(map_a: &mut FastMap<VoxelStat>, map_b: FastMap<VoxelStat>) {
    for (key, stat_b) in map_b {
        map_a.entry(key).or_default().merge(&stat_b);
    }
}

fn stat_map_into_centroids(global_map: FastMap<VoxelStat>) -> Vec<Point3<f32>> {
    let mut result = Vec::with_capacity(global_map.len());

    for stat in global_map.into_values() {
        if stat.count > 0 {
            result.push(stat.centroid());
        }
    }

    result
}

fn stat_map_into_point_cloud(
    global_map: FastMap<VoxelStat>,
    voxel_size: f32,
) -> DownsampledPointCloud {
    let mut points = Vec::with_capacity(global_map.len());
    let mut voxel_map = VoxelMap::default();
    voxel_map.reserve(global_map.len());

    for stat in global_map.into_values() {
        if stat.count == 0 {
            continue;
        }

        let point = stat.centroid();
        points.push(point);

        let key = voxel_key(&point, voxel_size);
        voxel_map
            .entry(key)
            .or_insert_with(|| VoxelCell::from_key(&key, voxel_size, point, 0));
    }

    DownsampledPointCloud { points, voxel_map }
}

#[cfg(test)]
mod tests {
    use super::{
        voxel_downsample_points, voxel_downsample_points_and_maps_dual,
        voxel_downsample_points_dual,
    };
    use crate::voxel_map::build_voxel_map;
    use nalgebra::Point3;

    fn sort_points(points: &mut [Point3<f32>]) {
        points.sort_unstable_by(|left, right| {
            left.x
                .total_cmp(&right.x)
                .then_with(|| left.y.total_cmp(&right.y))
                .then_with(|| left.z.total_cmp(&right.z))
        });
    }

    fn assert_same_points(mut actual: Vec<Point3<f32>>, mut expected: Vec<Point3<f32>>) {
        sort_points(&mut actual);
        sort_points(&mut expected);
        assert_eq!(actual.len(), expected.len());

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual.x - expected.x).abs() <= 1.0e-6);
            assert!((actual.y - expected.y).abs() <= 1.0e-6);
            assert!((actual.z - expected.z).abs() <= 1.0e-6);
        }
    }

    #[test]
    fn dual_downsample_matches_independent_downsamples() {
        let points: Vec<Point3<f32>> = (0..10_000)
            .map(|index| {
                let index = index as f32;
                Point3::new(
                    (index * 0.137).sin() * 40.0,
                    (index * 0.071).cos() * 25.0,
                    (index % 97.0) * 0.031 - 1.5,
                )
            })
            .collect();

        let expected_first = voxel_downsample_points(&points, 0.25);
        let expected_second = voxel_downsample_points(&points, 0.05);
        let (actual_first, actual_second) = voxel_downsample_points_dual(&points, 0.25, 0.05);

        assert_same_points(actual_first, expected_first);
        assert_same_points(actual_second, expected_second);
    }

    #[test]
    fn dual_downsample_handles_empty_input() {
        let (first, second) = voxel_downsample_points_dual(&[], 0.25, 0.05);
        assert!(first.is_empty());
        assert!(second.is_empty());
    }

    #[test]
    fn dual_downsample_maps_match_separate_map_builds() {
        let points: Vec<Point3<f32>> = (0..10_000)
            .map(|index| {
                let index = index as f32;
                Point3::new(
                    (index * 0.137).sin() * 40.0,
                    (index * 0.071).cos() * 25.0,
                    (index % 97.0) * 0.031 - 1.5,
                )
            })
            .collect();

        let (first, second) = voxel_downsample_points_and_maps_dual(&points, 0.25, 0.05);
        let expected_first_map = build_voxel_map(&first.points, 0.25, 0, false);
        let expected_second_map = build_voxel_map(&second.points, 0.05, 0, false);

        assert_eq!(first.voxel_map.len(), expected_first_map.len());
        assert_eq!(second.voxel_map.len(), expected_second_map.len());

        for (key, expected) in expected_first_map {
            let actual = first.voxel_map.get(&key).unwrap();
            assert_eq!(actual.point.0, expected.point.0);
            assert_eq!(actual.mean, expected.mean);
        }
        for (key, expected) in expected_second_map {
            let actual = second.voxel_map.get(&key).unwrap();
            assert_eq!(actual.point.0, expected.point.0);
            assert_eq!(actual.mean, expected.mean);
        }
    }

    #[test]
    fn dual_downsample_maps_handle_empty_input() {
        let (first, second) = voxel_downsample_points_and_maps_dual(&[], 0.25, 0.05);
        assert!(first.points.is_empty());
        assert!(first.voxel_map.is_empty());
        assert!(second.points.is_empty());
        assert!(second.voxel_map.is_empty());
    }
}
