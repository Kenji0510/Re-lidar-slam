use nalgebra::{Point3, Vector3};
use nohash_hasher::IntMap;
use rayon::prelude::*;

type FastMap<V> = IntMap<u64, V>;

#[derive(Clone, Debug)]
struct VoxelStat {
    sum: Vector3<f32>,
    count: usize,
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
        x = (x | (x << 32)) & 0x1f00_0000_00ff_ff;
        x = (x | (x << 16)) & 0x1f00_00ff_0000_ff;
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

    let (min_corner, max_corner) = points
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
        );

    let max_idx_x = ((max_corner.x - min_corner.x) * inv_voxel).floor() as u64;
    let max_idx_y = ((max_corner.y - min_corner.y) * inv_voxel).floor() as u64;
    let max_idx_z = ((max_corner.z - min_corner.z) * inv_voxel).floor() as u64;

    if max_idx_x > 0x1f_ffff || max_idx_y > 0x1f_ffff || max_idx_z > 0x1f_ffff {
        eprintln!(
            "Warning: point cloud extent exceeds Morton code 21-bit limit. Key collision may occur."
        );
    }

    let global_map: FastMap<VoxelStat> = points
        .par_iter()
        .fold(FastMap::<VoxelStat>::default, |mut local_map, p| {
            let ix = ((p.x - min_corner.x) * inv_voxel).floor().max(0.0) as u32;
            let iy = ((p.y - min_corner.y) * inv_voxel).floor().max(0.0) as u32;
            let iz = ((p.z - min_corner.z) * inv_voxel).floor().max(0.0) as u32;

            let key = morton3d(ix, iy, iz);

            local_map.entry(key).or_default().add_point(p);

            local_map
        })
        .reduce(FastMap::<VoxelStat>::default, |mut map_a, map_b| {
            for (key, stat_b) in map_b {
                map_a.entry(key).or_default().merge(&stat_b);
            }
            map_a
        });

    let mut result = Vec::with_capacity(global_map.len());

    for stat in global_map.values() {
        if stat.count > 0 {
            result.push(stat.centroid());
        }
    }

    result
}
