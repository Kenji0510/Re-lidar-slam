use std::collections::{HashSet, VecDeque};

use nalgebra::{Point3, Vector3};
// use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoxelKey {
    pub ix: i32,
    pub iy: i32,
    pub iz: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct LocalMapConfig {
    /// ハッシュグリッドのセルサイズ [m]。
    /// query_points_within_radius のハッシュルックアップ数 = (2*ceil(radius/index_voxel_size)+1)³ を決定する。
    /// データの精度（downsample_voxel_size）とは独立に設定できる。
    /// 大きいほどクエリが速く、小さいほどセルあたりの点数が減る。
    pub index_voxel_size: f32,
    pub max_points_per_voxel: usize,
    pub min_points_per_voxel: usize,

    pub min_observed_frames_per_voxel: usize,
    /// フレーム数ベースの追い出し上限。
    pub max_frames: usize,
    /// 距離ベースの追い出し上限 [m]。
    pub max_distance: f32,
}

#[derive(Debug, Clone)]
pub struct VoxelCell {
    // このvoxel内に入った代表点。
    // タプルの 2 要素目はフレーム ID。
    pub point: (Point3<f32>, u64),

    pub is_point: bool,

    // Mean of this voxel coordinates.
    pub mean: Point3<f32>,

    pub voxel_key: VoxelKey,
}

pub struct FrameEntry {
    pub frame_id: u64,
    pub origin: Point3<f32>,
    pub dirty_keys: HashSet<VoxelKey>,
}

pub type VoxelMap = HashMap<VoxelKey, VoxelCell>;

pub struct LOCALMap {
    pub voxel_map: VoxelMap,
    pub frame_index: VecDeque<FrameEntry>,
    pub config: LocalMapConfig,
    pub next_frame_id: u64,
}

impl LOCALMap {
    pub fn new(config: LocalMapConfig) -> Self {
        Self {
            voxel_map: VoxelMap::default(),
            frame_index: VecDeque::new(),
            config,
            next_frame_id: 0,
        }
    }
}

#[inline]
pub fn voxel_key(p: &Point3<f32>, voxel_size: f32) -> VoxelKey {
    VoxelKey {
        ix: (p.x / voxel_size).floor() as i32,
        iy: (p.y / voxel_size).floor() as i32,
        iz: (p.z / voxel_size).floor() as i32,
    }
}

impl VoxelCell {
    pub fn new() -> Self {
        Self {
            point: (Point3::new(0.0, 0.0, 0.0), 0),
            mean: Point3::new(0.0, 0.0, 0.0),
            voxel_key: VoxelKey {
                ix: 0,
                iy: 0,
                iz: 0,
            },
            is_point: true,
        }
    }

    pub fn from_key(key: &VoxelKey, voxel_size: f32, point: Point3<f32>, frame_id: u64) -> Self {
        let center = Point3::new(
            (key.ix as f32 + 0.5) * voxel_size,
            (key.iy as f32 + 0.5) * voxel_size,
            (key.iz as f32 + 0.5) * voxel_size,
        );
        Self {
            point: (point, frame_id),
            mean: center,
            voxel_key: *key,
            is_point: true,
        }
    }

    // pub fn recompute_mean(&mut self) {
    //     if self.points.is_empty() {
    //         self.mean = Point3::new(0.0, 0.0, 0.0);
    //         return;
    //     }

    //     let mut sum = Vector3::zeros();

    //     for p in &self.points {
    //         sum += p.coords;
    //     }

    //     let mean = sum / self.points.len() as f32;
    //     self.mean = Point3::from(mean);
    // }
}

pub fn build_voxel_map(
    points: &[Point3<f32>],
    voxel_size: f32,
    // max_points_per_voxel: usize,
) -> VoxelMap {
    let mut voxel_map = VoxelMap::new();

    for p in points {
        let key = voxel_key(p, voxel_size);

        let _cell = voxel_map
            .entry(key)
            .or_insert_with(|| VoxelCell::from_key(&key, voxel_size, *p, 0));

        // if cell.is_point {
        //     continue;
        // }
    }

    voxel_map
}
