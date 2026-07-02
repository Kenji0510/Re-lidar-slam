use std::collections::{HashSet, VecDeque};

use nalgebra::{Matrix3, Matrix4, Point3, Vector3};
use rayon::prelude::*;
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

    /// 共分散行列（compute_covariances() 呼び出し後に有効）。
    pub covariance: Matrix3<f32>,
    pub covariance_valid: bool,
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

    /// 変換済み点群をボクセルマップに追加する共通処理。
    /// 既存ボクセルはボクセル中心 (mean) に近い点を採用。
    fn insert_points(&mut self, source_points: &[Point3<f32>], global_pose: &Matrix4<f64>) {
        let pose_f32 = global_pose.cast::<f32>();
        let r_mat: Matrix3<f32> = pose_f32.fixed_view::<3, 3>(0, 0).into();
        let t_vec: Vector3<f32> = pose_f32.fixed_view::<3, 1>(0, 3).into();

        let voxel_size = self.config.index_voxel_size;
        let frame_id = self.next_frame_id;

        // 並列で全点をワールド座標変換してキーを計算
        let world_pts: Vec<(VoxelKey, Point3<f32>)> = source_points
            .par_iter()
            .map(|p| {
                let p_world = Point3::from(r_mat * p.coords + t_vec);
                let key = voxel_key(&p_world, voxel_size);
                (key, p_world)
            })
            .collect();

        // HashMap への挿入は順次（排他アクセスが必要）
        for (key, p_world) in world_pts {
            match self.voxel_map.entry(key) {
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(VoxelCell::from_key(&key, voxel_size, p_world, frame_id));
                }
                std::collections::hash_map::Entry::Occupied(mut e) => {
                    let cell = e.get_mut();
                    let existing_dist_sq = (cell.point.0.coords - cell.mean.coords).norm_squared();
                    let new_dist_sq = (p_world.coords - cell.mean.coords).norm_squared();
                    if new_dist_sq < existing_dist_sq {
                        cell.point = (p_world, frame_id);
                    }
                }
            }
        }

        self.next_frame_id += 1;
    }

    /// ICP で位置合わせ済みの source 点群をローカルマップに追加する。
    /// 追加後、自己位置から max_distance 以上のボクセルを破棄する。
    pub fn update_with_new_frame(
        &mut self,
        source_points: &[Point3<f32>],
        global_pose: &Matrix4<f64>,
    ) {
        self.insert_points(source_points, global_pose);

        // 自己位置から max_distance 以上のボクセルを破棄
        let pose_f32 = global_pose.cast::<f32>();
        let origin = Point3::from(Vector3::<f32>::from(pose_f32.fixed_view::<3, 1>(0, 3)));
        let max_dist_sq = self.config.max_distance * self.config.max_distance;
        self.voxel_map
            .retain(|_, cell| (cell.mean.coords - origin.coords).norm_squared() <= max_dist_sq);
    }

    /// ICP で位置合わせ済みの source 点群をワールドマップに追加する。
    /// ローカルマップと異なり、距離によるボクセル削除は行わない。
    pub fn update_world_map(&mut self, source_points: &[Point3<f32>], global_pose: &Matrix4<f64>) {
        self.insert_points(source_points, global_pose);
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
            covariance: Matrix3::identity(),
            covariance_valid: false,
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
            covariance: Matrix3::identity(),
            covariance_valid: false,
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
    neighbor_range: i32,
    is_target: bool,
) -> VoxelMap {
    let mut voxel_map = VoxelMap::new();

    for p in points {
        let key = voxel_key(p, voxel_size);
        voxel_map
            .entry(key)
            .or_insert_with(|| VoxelCell::from_key(&key, voxel_size, *p, 0));
    }

    if is_target {
        compute_covariances(&mut voxel_map, 5, neighbor_range);
    }

    voxel_map
}

/// ボクセルマップ内の各セルの共分散行列を計算する。
/// 対象セル + 周囲 `neighbor_range` ボクセルの代表点（各1点）を使用する。
/// 合計点数が `min_points` 未満のセルは `covariance_valid = false` のまま。
/// target 点群の VoxelMap に対してのみ呼び出す。
pub fn compute_covariances(voxel_map: &mut VoxelMap, min_points: usize, neighbor_range: i32) {
    let keys: Vec<VoxelKey> = voxel_map.keys().cloned().collect();

    // 並列読み取りパス: 各キーの mean・共分散を計算
    let updates: Vec<(VoxelKey, Option<(Point3<f32>, Matrix3<f32>)>)> = {
        let vm_ref: &VoxelMap = &*voxel_map;
        keys.par_iter()
            .map(|&key| {
                let mut all_points: Vec<Point3<f32>> = Vec::new();
                for dz in -neighbor_range..=neighbor_range {
                    for dy in -neighbor_range..=neighbor_range {
                        for dx in -neighbor_range..=neighbor_range {
                            let nkey = VoxelKey {
                                ix: key.ix + dx,
                                iy: key.iy + dy,
                                iz: key.iz + dz,
                            };
                            if let Some(nc) = vm_ref.get(&nkey) {
                                if nc.is_point {
                                    all_points.push(nc.point.0);
                                }
                            }
                        }
                    }
                }
                let n = all_points.len();
                if n < min_points {
                    return (key, None);
                }
                let mean_vec = all_points
                    .iter()
                    .fold(Vector3::zeros(), |acc, p| acc + p.coords)
                    / n as f32;
                let mut cov = Matrix3::zeros();
                for p in &all_points {
                    let d = p.coords - mean_vec;
                    cov += d * d.transpose();
                }
                cov /= (n - 1) as f32;
                (key, Some((Point3::from(mean_vec), cov)))
            })
            .collect()
    }; // vm_ref の借用ここで終了

    // 順次書き込みパス
    for (key, update) in updates {
        let cell = voxel_map.get_mut(&key).unwrap();
        if let Some((mean, cov)) = update {
            cell.mean = mean;
            cell.covariance = cov;
            cell.covariance_valid = true;
        } else {
            cell.covariance_valid = false;
        }
    }
}
