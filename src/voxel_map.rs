use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

use nalgebra::{Matrix3, Matrix4, Point3, Vector3};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::find_nearest_points::fit_plane;

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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SurfaceStatus {
    #[default]
    Unknown,
    Planar,
    NonPlanar,
}

#[derive(Debug, Clone, Copy)]
pub struct SurfacePlane {
    pub normal: Vector3<f32>,
    pub d: f32,
    pub rmse_m: f32,
    pub neighbor_count: usize,
    pub inlier_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceFilterConfig {
    /// 2 の場合、中心を含む 5x5x5 ボクセルを平面推定に使う。
    pub neighbor_radius_voxels: i32,
    pub min_neighbors: usize,
    pub min_ransac_inliers: usize,
    pub min_center_observed_frames: u64,
    pub ransac_iterations: usize,
    /// 少なくとも1回、3点すべてがインライアとなる仮説を引く目標確率。
    pub ransac_confidence: f64,
    /// 適応的打ち切りを許可する最小反復数。
    pub ransac_min_iterations: usize,
    /// RANSAC 仮説平面からこの距離以内の点を PCA 入力にする。
    pub ransac_inlier_distance_m: f32,
    pub min_inlier_ratio: f32,
    pub min_planarity: f32,
    pub max_surface_variation: f32,
    pub max_pca_rmse_m: f32,
    /// RANSACインライアからPCAで求めた平面と、中心ボクセル代表点との最大距離。
    pub max_center_distance_m: f32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SurfaceClassificationStats {
    pub evaluated: usize,
    pub planar: usize,
    pub non_planar: usize,
    pub unknown: usize,
    /// dirty voxel から中心候補を展開する際、評価済みとして除外したプローブ数。
    pub skipped_unchanged_probes: usize,
    pub immature_fast_path: usize,
}

/// 成熟した GlobalMap 平面を使って新規観測を更新するための設定。
///
/// `accept_distance_m` より近い点は既存面の観測として平面へ射影して統合する。
/// それより遠く `pending_distance_m` 以内にある点は、近接する二重壁や
/// 姿勢誤差の可能性があるため、即座に占有 voxel を作らず保留する。
#[derive(Debug, Clone, Copy)]
pub struct WorldMapUpdateFilterConfig {
    pub mature_plane_search_radius_voxels: i32,
    pub min_mature_observed_frames: u64,
    pub accept_distance_m: f32,
    pub pending_distance_m: f32,
    pub project_accepted_points: bool,
    pub pending_max_age_frames: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WorldMapUpdateStats {
    pub input_points: usize,
    /// 周囲に成熟平面がなく、未確定の新規構造として追加した点数。
    pub inserted_provisional: usize,
    /// 成熟平面と整合し、その平面へ射影して追加した点数。
    pub projected_to_mature_plane: usize,
    /// 成熟平面の近くにあるが整合しないため保留した点数。
    pub held_pending: usize,
    pub rejected_non_finite: usize,
    /// この更新後に保持されている pending voxel 数。
    pub pending_voxels: usize,
}

#[derive(Debug, Clone)]
struct PendingVoxelCell {
    mean: Point3<f32>,
    sample_count: u64,
    last_observed_frame_id: u64,
}

impl PendingVoxelCell {
    fn new(point: Point3<f32>, frame_id: u64) -> Self {
        Self {
            mean: point,
            sample_count: 1,
            last_observed_frame_id: frame_id,
        }
    }

    fn update(&mut self, point: Point3<f32>, frame_id: u64) {
        self.sample_count += 1;
        self.mean.coords += (point.coords - self.mean.coords) / self.sample_count as f32;

        self.last_observed_frame_id = frame_id;
    }
}

#[derive(Debug, Clone)]
pub struct VoxelCell {
    // このvoxel内に入った代表点。
    // タプルの 2 要素目はフレーム ID。
    pub point: (Point3<f32>, u64),

    pub is_point: bool,

    // Mean of this voxel coordinates.
    pub mean: Point3<f32>,

    // Welford法の共分散計算用の中間値
    pub m2: Matrix3<f32>,

    // 統計更新に使用された観測数
    pub sample_count: u64,

    // 異なるフレームから観測された回数
    pub observed_frames: u64,

    pub last_observed_frame_id: Option<u64>,

    pub voxel_key: VoxelKey,

    /// 共分散行列（compute_covariances() 呼び出し後に有効）。
    pub covariance: Matrix3<f32>,
    pub covariance_valid: bool,

    /// 観測データは保持したまま、平面マップへの出力可否だけを表す。
    pub surface_status: SurfaceStatus,
    pub surface_plane: Option<SurfacePlane>,

    /// このセルの平面判定に使用した SurfaceFilterConfig の世代。
    pub surface_evaluation_epoch: u64,
    /// このフレームまでの累積マップを使って平面判定済みであることを表す。
    pub surface_evaluated_through_frame_id: Option<u64>,
}

pub struct FrameEntry {
    pub frame_id: u64,
    pub origin: Point3<f32>,
    pub dirty_keys: FxHashSet<VoxelKey>,
}

#[derive(Debug, Clone, Copy)]
enum WorldPointUpdateDecision {
    InsertProvisional(Point3<f32>),
    InsertOnMaturePlane {
        original: Point3<f32>,
        insertion: Point3<f32>,
    },
    HoldPending(Point3<f32>),
    RejectNonFinite,
}

/// VoxelKey is generated internally from trusted point-cloud coordinates, so a
/// fast deterministic hasher is preferable to HashMap's HashDoS-resistant one.
pub type VoxelMap = FxHashMap<VoxelKey, VoxelCell>;

pub struct LOCALMap {
    pub voxel_map: VoxelMap,
    pub frame_index: VecDeque<FrameEntry>,
    pub config: LocalMapConfig,
    pub next_frame_id: u64,
    pending_voxel_map: FxHashMap<VoxelKey, PendingVoxelCell>,
    surface_filter_config: Option<SurfaceFilterConfig>,
    surface_evaluation_epoch: u64,
}

impl LOCALMap {
    pub fn new(config: LocalMapConfig) -> Self {
        Self {
            voxel_map: VoxelMap::default(),
            frame_index: VecDeque::new(),
            config,
            next_frame_id: 0,
            pending_voxel_map: FxHashMap::default(),
            surface_filter_config: None,
            surface_evaluation_epoch: 0,
        }
    }

    // 同一フレーム内の点をボクセルごとに平均し、
    // ボクセルの逐次平均・共分散を更新する。
    fn insert_points(
        &mut self,
        source_points: &[Point3<f32>],
        global_pose: &Matrix4<f64>,
    ) -> FrameEntry {
        // 並列で全点をワールド座標変換してキーを計算
        let pose_f32 = global_pose.cast::<f32>();
        let r_mat: Matrix3<f32> = pose_f32.fixed_view::<3, 3>(0, 0).into();
        let t_vec: Vector3<f32> = pose_f32.fixed_view::<3, 1>(0, 3).into();
        let world_points: Vec<Point3<f32>> = source_points
            .par_iter()
            .map(|p| Point3::from(r_mat * p.coords + t_vec))
            .collect();

        self.insert_world_points(&world_points, Point3::from(t_vec))
    }

    /// 既にワールド座標へ変換済みの点を挿入する。
    /// GlobalMap の平面ゲートで射影した点を再変換せず挿入するために分離している。
    fn insert_world_points(
        &mut self,
        world_points: &[Point3<f32>],
        origin: Point3<f32>,
    ) -> FrameEntry {
        let voxel_size = self.config.index_voxel_size;
        let frame_id = self.next_frame_id;

        let mut frame_voxels: FxHashMap<VoxelKey, (Vector3<f32>, usize)> = FxHashMap::default();

        for point in world_points {
            if !point.coords.iter().all(|value| value.is_finite()) {
                continue;
            }

            let key = voxel_key(point, voxel_size);
            frame_voxels
                .entry(key)
                .and_modify(|(sum, count)| {
                    *sum += point.coords;
                    *count += 1;
                })
                .or_insert((point.coords, 1));
        }

        // VoxelMap への挿入は順次（排他アクセスが必要）
        // for (key, p_world) in world_pts {
        //     match self.voxel_map.entry(key) {
        //         std::collections::hash_map::Entry::Vacant(e) => {
        //             e.insert(VoxelCell::from_key(&key, voxel_size, p_world, frame_id));
        //         }
        //         std::collections::hash_map::Entry::Occupied(mut e) => {
        //             let cell = e.get_mut();
        //             let existing_dist_sq = (cell.point.0.coords - cell.mean.coords).norm_squared();
        //             let new_dist_sq = (p_world.coords - cell.mean.coords).norm_squared();
        //             if new_dist_sq < existing_dist_sq {
        //                 cell.point = (p_world, frame_id);
        //             }
        //         }
        //     }
        // }

        let min_samples = self.config.min_points_per_voxel;
        let dirty_keys = frame_voxels.keys().copied().collect();

        for (key, (sum, count)) in frame_voxels {
            let frame_mean = Point3::from(sum / count as f32);

            match self.voxel_map.entry(key) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(VoxelCell::from_key(&key, voxel_size, frame_mean, frame_id));
                }

                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    let cell = entry.get_mut();
                    cell.update_statistics(frame_mean, frame_id, min_samples);
                    // 新しい観測が入ったセルは、遅延平面判定が再度完了するまで未確定。
                    cell.surface_status = SurfaceStatus::Unknown;
                    cell.surface_plane = None;
                }
            }
        }

        self.next_frame_id += 1;

        FrameEntry {
            frame_id,
            origin,
            dirty_keys,
        }
    }

    /// ICP で位置合わせ済みの source 点群をローカルマップに追加する。
    /// 追加後、自己位置から max_distance 以上のボクセルを破棄する。
    pub fn update_with_new_frame(
        &mut self,
        source_points: &[Point3<f32>],
        global_pose: &Matrix4<f64>,
    ) {
        let _ = self.insert_points(source_points, global_pose);

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
        let frame_entry = self.insert_points(source_points, global_pose);
        self.frame_index.push_back(frame_entry);
    }

    /// 成熟した GlobalMap 平面との整合性を確認してからワールドマップを更新する。
    ///
    /// - 成熟平面に近い点: 平面へ射影して統合する。
    /// - 成熟平面から少し離れた点: 二重壁候補として pending に保留する。
    /// - 近くに成熟平面がない点: 新規構造の provisional voxel として追加する。
    ///
    /// pending は GlobalMap/ICP/PCD 出力には使わない。現在の目的は壁の厚み抑制を
    /// 優先するため、近接した平行面を自動昇格させず、期限切れで破棄する。
    pub fn update_world_map_filtered(
        &mut self,
        source_points: &[Point3<f32>],
        global_pose: &Matrix4<f64>,
        filter_config: &WorldMapUpdateFilterConfig,
    ) -> WorldMapUpdateStats {
        assert!(filter_config.accept_distance_m.is_finite());
        assert!(filter_config.accept_distance_m > 0.0);
        assert!(filter_config.pending_distance_m.is_finite());
        assert!(filter_config.pending_distance_m >= filter_config.accept_distance_m);

        let pose_f32 = global_pose.cast::<f32>();
        let r_mat: Matrix3<f32> = pose_f32.fixed_view::<3, 3>(0, 0).into();
        let t_vec: Vector3<f32> = pose_f32.fixed_view::<3, 1>(0, 3).into();
        let origin = Point3::from(t_vec);
        let frame_id = self.next_frame_id;

        let map_ref: &LOCALMap = &*self;
        let decisions: Vec<WorldPointUpdateDecision> = source_points
            .par_iter()
            .map(|source_point| {
                if !source_point.coords.iter().all(|value| value.is_finite()) {
                    return WorldPointUpdateDecision::RejectNonFinite;
                }

                let world_point = Point3::from(r_mat * source_point.coords + t_vec);
                map_ref.classify_world_point_update(world_point, filter_config)
            })
            .collect();

        let mut stats = WorldMapUpdateStats {
            input_points: source_points.len(),
            ..WorldMapUpdateStats::default()
        };
        let mut accepted_world_points = Vec::with_capacity(source_points.len());
        let mut pending_frame_voxels: FxHashMap<VoxelKey, (Vector3<f32>, usize)> =
            FxHashMap::default();

        for decision in decisions {
            match decision {
                WorldPointUpdateDecision::InsertProvisional(point) => {
                    accepted_world_points.push(point);
                    self.pending_voxel_map
                        .remove(&voxel_key(&point, self.config.index_voxel_size));
                    stats.inserted_provisional += 1;
                }
                WorldPointUpdateDecision::InsertOnMaturePlane {
                    original,
                    insertion,
                } => {
                    accepted_world_points.push(insertion);
                    self.pending_voxel_map.remove(&voxel_key(
                        &original,
                        self.config.index_voxel_size,
                    ));
                    stats.projected_to_mature_plane += 1;
                }
                WorldPointUpdateDecision::HoldPending(point) => {
                    let key = voxel_key(&point, self.config.index_voxel_size);
                    pending_frame_voxels
                        .entry(key)
                        .and_modify(|(sum, count)| {
                            *sum += point.coords;
                            *count += 1;
                        })
                        .or_insert((point.coords, 1));
                    stats.held_pending += 1;
                }
                WorldPointUpdateDecision::RejectNonFinite => {
                    stats.rejected_non_finite += 1;
                }
            }
        }

        for (key, (sum, count)) in pending_frame_voxels {
            let frame_mean = Point3::from(sum / count as f32);
            match self.pending_voxel_map.entry(key) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(PendingVoxelCell::new(frame_mean, frame_id));
                }
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    entry.get_mut().update(frame_mean, frame_id);
                }
            }
        }

        let frame_entry = self.insert_world_points(&accepted_world_points, origin);
        self.frame_index.push_back(frame_entry);

        self.pending_voxel_map.retain(|_, pending| {
            frame_id.saturating_sub(pending.last_observed_frame_id)
                <= filter_config.pending_max_age_frames
        });
        stats.pending_voxels = self.pending_voxel_map.len();
        stats
    }

    pub fn pending_voxel_count(&self) -> usize {
        self.pending_voxel_map.len()
    }

    fn classify_world_point_update(
        &self,
        world_point: Point3<f32>,
        config: &WorldMapUpdateFilterConfig,
    ) -> WorldPointUpdateDecision {
        let Some(plane) = self.find_nearest_mature_plane(&world_point, config) else {
            return WorldPointUpdateDecision::InsertProvisional(world_point);
        };

        let signed_distance = plane.normal.dot(&world_point.coords) + plane.d;
        let absolute_distance = signed_distance.abs();

        if absolute_distance <= config.accept_distance_m {
            let insertion = if config.project_accepted_points {
                Point3::from(world_point.coords - plane.normal * signed_distance)
            } else {
                world_point
            };
            WorldPointUpdateDecision::InsertOnMaturePlane {
                original: world_point,
                insertion,
            }
        } else {
            WorldPointUpdateDecision::HoldPending(world_point)
        }
    }

    fn find_nearest_mature_plane(
        &self,
        world_point: &Point3<f32>,
        config: &WorldMapUpdateFilterConfig,
    ) -> Option<SurfacePlane> {
        let center_key = voxel_key(world_point, self.config.index_voxel_size);
        let radius = config.mature_plane_search_radius_voxels.max(0);
        let mut best: Option<(SurfacePlane, f32)> = None;

        for dz in -radius..=radius {
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let key = VoxelKey {
                        ix: center_key.ix + dx,
                        iy: center_key.iy + dy,
                        iz: center_key.iz + dz,
                    };
                    let Some(cell) = self.voxel_map.get(&key) else {
                        continue;
                    };
                    if cell.surface_status != SurfaceStatus::Planar
                        || cell.observed_frames < config.min_mature_observed_frames
                    {
                        continue;
                    }
                    let Some(plane) = cell.surface_plane else {
                        continue;
                    };

                    let distance = (plane.normal.dot(&world_point.coords) + plane.d).abs();
                    if !distance.is_finite() || distance > config.pending_distance_m {
                        continue;
                    }

                    let replace = best.map_or(true, |(current, current_distance)| {
                        distance < current_distance
                            || (distance == current_distance && plane.rmse_m < current.rmse_m)
                    });
                    if replace {
                        best = Some((plane, distance));
                    }
                }
            }
        }

        best.map(|(plane, _)| plane)
    }

    /// 現在フレームから `delay_frames` 以上古い更新領域を、現在までの累積点で再判定する。
    pub fn classify_delayed_surface_voxels(
        &mut self,
        delay_frames: u64,
        filter_config: &SurfaceFilterConfig,
    ) -> SurfaceClassificationStats {
        let profile_enabled = log::log_enabled!(log::Level::Info);
        let total_start = profile_enabled.then(Instant::now);
        let Some(current_frame_id) = self.next_frame_id.checked_sub(1) else {
            return SurfaceClassificationStats::default();
        };

        let queue_drain_start = profile_enabled.then(Instant::now);
        let mut dirty_versions = FxHashMap::default();
        while self
            .frame_index
            .front()
            .is_some_and(|entry| current_frame_id.saturating_sub(entry.frame_id) >= delay_frames)
        {
            if let Some(entry) = self.frame_index.pop_front() {
                for key in entry.dirty_keys {
                    dirty_versions
                        .entry(key)
                        .and_modify(|frame_id: &mut u64| {
                            *frame_id = (*frame_id).max(entry.frame_id);
                        })
                        .or_insert(entry.frame_id);
                }
            }
        }
        let queue_drain_wall = queue_drain_start
            .map(|start| start.elapsed())
            .unwrap_or_default();

        let config_epoch_start = profile_enabled.then(Instant::now);
        let evaluation_epoch = self.surface_evaluation_epoch(filter_config);
        let config_epoch_wall = config_epoch_start
            .map(|start| start.elapsed())
            .unwrap_or_default();
        let affected_classification_start = profile_enabled.then(Instant::now);
        let stats = self.classify_surface_voxels_affected_by(
            &dirty_versions,
            filter_config,
            current_frame_id,
            evaluation_epoch,
            profile_enabled,
        );
        let affected_classification_wall = affected_classification_start
            .map(|start| start.elapsed())
            .unwrap_or_default();

        if profile_enabled {
            let total_wall = total_start.map(|start| start.elapsed()).unwrap_or_default();
            let other_wall = total_wall
                .saturating_sub(queue_drain_wall)
                .saturating_sub(config_epoch_wall)
                .saturating_sub(affected_classification_wall);
            log::info!(
                "Delayed surface realtime frame={current_frame_id}: \
                 classify_delayed_surface_voxels wall_ms total={:.3}, queue_drain={:.3}, \
                 config_epoch={:.3}, classify_affected={:.3}, other={:.3} | \
                 dirty={}, evaluated={}, skipped_unchanged_probes={}",
                duration_ms(total_wall),
                duration_ms(queue_drain_wall),
                duration_ms(config_epoch_wall),
                duration_ms(affected_classification_wall),
                duration_ms(other_wall),
                dirty_versions.len(),
                stats.evaluated,
                stats.skipped_unchanged_probes,
            );
        }

        stats
    }

    /// 全ボクセルを現在の累積点で再判定する。終了時の最終確定用。
    pub fn classify_all_surface_voxels(
        &mut self,
        filter_config: &SurfaceFilterConfig,
    ) -> SurfaceClassificationStats {
        let evaluation_epoch = self.surface_evaluation_epoch(filter_config);
        let current_frame_id = self.next_frame_id.checked_sub(1);
        let center_keys: Vec<VoxelKey> = self.voxel_map.keys().copied().collect();
        self.classify_surface_voxels(
            &center_keys,
            filter_config,
            false,
            current_frame_id,
            evaluation_epoch,
        )
        .0
    }

    fn classify_surface_voxels_affected_by(
        &mut self,
        dirty_versions: &FxHashMap<VoxelKey, u64>,
        filter_config: &SurfaceFilterConfig,
        current_frame_id: u64,
        evaluation_epoch: u64,
        profile_enabled: bool,
    ) -> SurfaceClassificationStats {
        if dirty_versions.is_empty() {
            return SurfaceClassificationStats::default();
        }

        let total_start = profile_enabled.then(Instant::now);
        let candidate_generation_start = profile_enabled.then(Instant::now);
        let radius = filter_config.neighbor_radius_voxels.max(0);
        let candidate_accumulator = {
            let voxel_map = &self.voxel_map;

            // dirty voxel は、周囲 radius 内の各中心ボクセルの平面推定に影響する。
            // 評価済みの中心はここで除外し、中間 Map の挿入・マージ対象にしない。
            // 各中心には、未評価の dirty voxel の最新フレームを記録する。
            dirty_versions
                .par_iter()
                .fold(
                    AffectedCandidateAccumulator::default,
                    |mut accumulator, (dirty_key, dirty_frame_id)| {
                        for dz in -radius..=radius {
                            for dy in -radius..=radius {
                                for dx in -radius..=radius {
                                    let center_key = VoxelKey {
                                        ix: dirty_key.ix + dx,
                                        iy: dirty_key.iy + dy,
                                        iz: dirty_key.iz + dz,
                                    };
                                    let Some(cell) = voxel_map.get(&center_key) else {
                                        continue;
                                    };
                                    let config_changed =
                                        cell.surface_evaluation_epoch != evaluation_epoch;
                                    let has_unseen_update = cell
                                        .surface_evaluated_through_frame_id
                                        .is_none_or(|evaluated_frame_id| {
                                            evaluated_frame_id < *dirty_frame_id
                                        });
                                    if !config_changed && !has_unseen_update {
                                        accumulator.skipped_unchanged_probes =
                                            accumulator.skipped_unchanged_probes.saturating_add(1);
                                        continue;
                                    }

                                    accumulator
                                        .center_versions
                                        .entry(center_key)
                                        .and_modify(|frame_id: &mut u64| {
                                            *frame_id = (*frame_id).max(*dirty_frame_id);
                                        })
                                        .or_insert(*dirty_frame_id);
                                }
                            }
                        }

                        accumulator
                    },
                )
                .reduce(
                    AffectedCandidateAccumulator::default,
                    merge_candidate_accumulators,
                )
        };

        let candidate_generation_wall = candidate_generation_start
            .map(|start| start.elapsed())
            .unwrap_or_default();
        let maturity_partition_start = profile_enabled.then(Instant::now);
        let eligible_center_count = candidate_accumulator.center_versions.len();
        let skipped_unchanged_probes = candidate_accumulator.skipped_unchanged_probes;
        let mut center_keys = Vec::new();
        let mut immature_center_keys = Vec::new();
        for key in candidate_accumulator.center_versions.into_keys() {
            let Some(cell) = self.voxel_map.get(&key) else {
                continue;
            };
            if cell.observed_frames < filter_config.min_center_observed_frames {
                immature_center_keys.push(key);
            } else {
                center_keys.push(key);
            }
        }
        let maturity_partition_wall = maturity_partition_start
            .map(|start| start.elapsed())
            .unwrap_or_default();
        let (mut stats, timings) = self.classify_surface_voxels(
            &center_keys,
            filter_config,
            profile_enabled,
            Some(current_frame_id),
            evaluation_epoch,
        );

        // 中心セル自体が未成熟なら、近傍収集や RANSAC を行っても必ず Unknown になる。
        // Rayon の評価対象には入れず、結果と評価ウォーターマークだけを更新する。
        let immature_fast_path_start = profile_enabled.then(Instant::now);
        for key in immature_center_keys {
            let Some(cell) = self.voxel_map.get_mut(&key) else {
                continue;
            };
            cell.surface_status = SurfaceStatus::Unknown;
            cell.surface_plane = None;
            cell.surface_evaluation_epoch = evaluation_epoch;
            cell.surface_evaluated_through_frame_id = Some(current_frame_id);
            stats.evaluated += 1;
            stats.unknown += 1;
            stats.immature_fast_path += 1;
        }
        let immature_fast_path_wall = immature_fast_path_start
            .map(|start| start.elapsed())
            .unwrap_or_default();
        stats.skipped_unchanged_probes = skipped_unchanged_probes;

        if profile_enabled {
            let neighborhood_width = (radius as usize).saturating_mul(2).saturating_add(1);
            let candidate_probes = dirty_versions.len().saturating_mul(
                neighborhood_width
                    .saturating_mul(neighborhood_width)
                    .saturating_mul(neighborhood_width),
            );
            let total_wall = total_start.map(|start| start.elapsed()).unwrap_or_default();
            let wall_overhead = total_wall
                .saturating_sub(candidate_generation_wall)
                .saturating_sub(maturity_partition_wall)
                .saturating_sub(timings.total_wall)
                .saturating_sub(immature_fast_path_wall);
            let worker_time_sum = timings.evaluation_worker_sum;
            let effective_parallelism = ratio(
                worker_time_sum.as_secs_f64(),
                timings.parallel_evaluation_wall.as_secs_f64(),
            );
            let average_ransac_draws =
                ratio(timings.ransac_draws as f64, timings.ransac_calls as f64);
            let adaptive_stop_percent =
                percentage(timings.ransac_adaptive_stops, timings.ransac_calls);
            let point_test_total = timings
                .ransac_point_tests
                .saturating_add(timings.ransac_point_tests_skipped);
            let point_test_skip_percent =
                percentage(timings.ransac_point_tests_skipped, point_test_total);

            let classification_other_wall = timings
                .total_wall
                .saturating_sub(timings.parallel_evaluation_wall)
                .saturating_sub(timings.evaluation_reduce_wall)
                .saturating_sub(timings.result_apply_wall);

            log::info!(
                "Delayed surface realtime frame={current_frame_id}: \
                 classify_surface_voxels_affected_by wall_ms total={:.3}, \
                 candidate_generation={:.3}, maturity_partition={:.3}, \
                 classify_surface_voxels={:.3}, immature_fast_path={:.3}, other={:.3} | \
                 dirty={}, probes={}, eligible_centers={}, skipped_unchanged_probes={}, \
                 immature_fast_path_count={}, evaluated={}",
                duration_ms(total_wall),
                duration_ms(candidate_generation_wall),
                duration_ms(maturity_partition_wall),
                duration_ms(timings.total_wall),
                duration_ms(immature_fast_path_wall),
                duration_ms(wall_overhead),
                dirty_versions.len(),
                candidate_probes,
                eligible_center_count,
                skipped_unchanged_probes,
                stats.immature_fast_path,
                stats.evaluated,
            );
            log::info!(
                "Delayed surface realtime frame={current_frame_id}: classify_surface_voxels \
                 wall_ms total={:.3}, parallel_evaluation={:.3}, timing_reduce={:.3}, \
                 result_apply={:.3}, other={:.3} | evaluate_surface_voxel worker_estimate_ms \
                 total={:.3}, center_precheck={:.3}, neighborhood={:.3}, \
                 insufficient_neighbors={:.3}, ransac_plane_inliers={:.3}, \
                 post_ransac_gate={:.3}, fit_plane_pca={:.3}, \
                 rmse_and_quality_check={:.3}, other={:.3} | timing_sample={}/{}, \
                 scale={:.2}x, estimated_parallelism={:.2}x",
                duration_ms(timings.total_wall),
                duration_ms(timings.parallel_evaluation_wall),
                duration_ms(timings.evaluation_reduce_wall),
                duration_ms(timings.result_apply_wall),
                duration_ms(classification_other_wall),
                duration_ms(worker_time_sum),
                duration_ms(timings.center_precheck_worker_sum),
                duration_ms(timings.neighborhood_cpu_sum),
                duration_ms(timings.insufficient_neighbors_worker_sum),
                duration_ms(timings.ransac_cpu_sum),
                duration_ms(timings.post_ransac_gate_worker_sum),
                duration_ms(timings.pca_cpu_sum),
                duration_ms(timings.rmse_and_quality_check_worker_sum),
                duration_ms(timings.evaluation_other_cpu_sum),
                timings.timing_sample_count,
                timings.evaluation_count,
                timings.timing_sample_scale,
                effective_parallelism,
            );
            log::debug!(
                "Delayed surface RANSAC: calls={}, draws={} (avg={:.2}/{}), \
                 adaptive_stops={} ({:.1}%), iterations_saved={}, pruned={}, \
                 point_tests={}, point_tests_skipped={} ({:.1}%)",
                timings.ransac_calls,
                timings.ransac_draws,
                average_ransac_draws,
                filter_config.ransac_iterations,
                timings.ransac_adaptive_stops,
                adaptive_stop_percent,
                timings.ransac_iterations_saved,
                timings.ransac_pruned,
                timings.ransac_point_tests,
                timings.ransac_point_tests_skipped,
                point_test_skip_percent,
            );
        }

        stats
    }

    fn classify_surface_voxels(
        &mut self,
        center_keys: &[VoxelKey],
        filter_config: &SurfaceFilterConfig,
        profile_enabled: bool,
        current_frame_id: Option<u64>,
        evaluation_epoch: u64,
    ) -> (SurfaceClassificationStats, SurfaceClassificationTimings) {
        let total_start = profile_enabled.then(Instant::now);
        let radius = filter_config.neighbor_radius_voxels.max(0) as usize;
        let neighborhood_width = radius * 2 + 1;
        let max_neighbor_points = neighborhood_width
            .saturating_mul(neighborhood_width)
            .saturating_mul(neighborhood_width);
        let parallel_evaluation_start = profile_enabled.then(Instant::now);
        let evaluations: Vec<(VoxelKey, SurfaceEvaluation)> = {
            let voxel_map = &self.voxel_map;
            center_keys
                .par_iter()
                .map_init(
                    || SurfaceEvaluationScratch::with_capacity(max_neighbor_points),
                    |scratch, key| {
                        let detailed_profile_enabled =
                            profile_enabled && should_sample_surface_timing(*key);
                        evaluate_surface_voxel(
                            voxel_map,
                            *key,
                            filter_config,
                            scratch,
                            detailed_profile_enabled,
                        )
                        .map(|evaluation| (*key, evaluation))
                    },
                )
                .filter_map(|evaluation| evaluation)
                .collect()
        };
        let parallel_evaluation_wall = parallel_evaluation_start
            .map(|start| start.elapsed())
            .unwrap_or_default();

        let evaluation_count = evaluations.len();
        let evaluation_reduce_start = profile_enabled.then(Instant::now);
        let evaluation_cpu_sums = evaluations.iter().fold(
            SurfaceEvaluationTimings::default(),
            |mut total, (_, evaluation)| {
                total.add_assign(evaluation.timings);
                total
            },
        );
        let evaluation_reduce_wall = evaluation_reduce_start
            .map(|start| start.elapsed())
            .unwrap_or_default();
        let timing_sample_scale = ratio(
            evaluation_count as f64,
            evaluation_cpu_sums.timing_samples as f64,
        );

        let result_apply_start = profile_enabled.then(Instant::now);
        let mut stats = SurfaceClassificationStats::default();
        for (key, evaluation) in evaluations {
            let Some(cell) = self.voxel_map.get_mut(&key) else {
                continue;
            };

            cell.surface_status = evaluation.status;
            cell.surface_plane = evaluation.plane;
            cell.surface_evaluation_epoch = evaluation_epoch;
            cell.surface_evaluated_through_frame_id = current_frame_id;
            stats.evaluated += 1;
            match evaluation.status {
                SurfaceStatus::Unknown => stats.unknown += 1,
                SurfaceStatus::Planar => stats.planar += 1,
                SurfaceStatus::NonPlanar => stats.non_planar += 1,
            }
        }
        let result_apply_wall = result_apply_start
            .map(|start| start.elapsed())
            .unwrap_or_default();
        let total_wall = total_start.map(|start| start.elapsed()).unwrap_or_default();

        (
            stats,
            SurfaceClassificationTimings {
                total_wall,
                parallel_evaluation_wall,
                evaluation_reduce_wall,
                evaluation_worker_sum: scale_duration(
                    evaluation_cpu_sums.total,
                    timing_sample_scale,
                ),
                center_precheck_worker_sum: scale_duration(
                    evaluation_cpu_sums.center_precheck,
                    timing_sample_scale,
                ),
                neighborhood_cpu_sum: scale_duration(
                    evaluation_cpu_sums.neighborhood,
                    timing_sample_scale,
                ),
                insufficient_neighbors_worker_sum: scale_duration(
                    evaluation_cpu_sums.insufficient_neighbors,
                    timing_sample_scale,
                ),
                ransac_cpu_sum: scale_duration(evaluation_cpu_sums.ransac, timing_sample_scale),
                post_ransac_gate_worker_sum: scale_duration(
                    evaluation_cpu_sums.post_ransac_gate,
                    timing_sample_scale,
                ),
                pca_cpu_sum: scale_duration(evaluation_cpu_sums.pca, timing_sample_scale),
                rmse_and_quality_check_worker_sum: scale_duration(
                    evaluation_cpu_sums.rmse_and_quality_check,
                    timing_sample_scale,
                ),
                evaluation_other_cpu_sum: scale_duration(
                    evaluation_cpu_sums.other(),
                    timing_sample_scale,
                ),
                timing_sample_count: evaluation_cpu_sums.timing_samples,
                evaluation_count,
                timing_sample_scale,
                result_apply_wall,
                ransac_calls: evaluation_cpu_sums.ransac_calls,
                ransac_draws: evaluation_cpu_sums.ransac_draws,
                ransac_pruned: evaluation_cpu_sums.ransac_pruned,
                ransac_point_tests: evaluation_cpu_sums.ransac_point_tests,
                ransac_point_tests_skipped: evaluation_cpu_sums.ransac_point_tests_skipped,
                ransac_adaptive_stops: evaluation_cpu_sums.ransac_adaptive_stops,
                ransac_iterations_saved: evaluation_cpu_sums.ransac_iterations_saved,
            },
        )
    }

    fn surface_evaluation_epoch(&mut self, filter_config: &SurfaceFilterConfig) -> u64 {
        if self.surface_filter_config.as_ref() != Some(filter_config) {
            self.surface_filter_config = Some(*filter_config);
            self.surface_evaluation_epoch = self
                .surface_evaluation_epoch
                .checked_add(1)
                .expect("surface evaluation epoch overflow");
        }

        self.surface_evaluation_epoch
    }
}

#[derive(Default)]
struct AffectedCandidateAccumulator {
    center_versions: FxHashMap<VoxelKey, u64>,
    skipped_unchanged_probes: usize,
}

fn merge_candidate_accumulators(
    mut left: AffectedCandidateAccumulator,
    right: AffectedCandidateAccumulator,
) -> AffectedCandidateAccumulator {
    left.center_versions = merge_voxel_key_versions(left.center_versions, right.center_versions);
    left.skipped_unchanged_probes = left
        .skipped_unchanged_probes
        .saturating_add(right.skipped_unchanged_probes);
    left
}

fn merge_voxel_key_versions(
    mut left: FxHashMap<VoxelKey, u64>,
    mut right: FxHashMap<VoxelKey, u64>,
) -> FxHashMap<VoxelKey, u64> {
    // 小さい Map を大きい Map へ追加し、再ハッシュと挿入回数を抑える。
    if left.len() < right.len() {
        std::mem::swap(&mut left, &mut right);
    }
    for (key, frame_id) in right {
        left.entry(key)
            .and_modify(|existing| *existing = (*existing).max(frame_id))
            .or_insert(frame_id);
    }
    left
}

#[derive(Debug, Clone, Copy)]
struct SurfaceEvaluation {
    status: SurfaceStatus,
    plane: Option<SurfacePlane>,
    timings: SurfaceEvaluationTimings,
}

#[derive(Debug, Clone, Copy, Default)]
struct SurfaceEvaluationTimings {
    total: Duration,
    center_precheck: Duration,
    neighborhood: Duration,
    insufficient_neighbors: Duration,
    ransac: Duration,
    post_ransac_gate: Duration,
    pca: Duration,
    rmse_and_quality_check: Duration,
    timing_samples: usize,
    ransac_calls: usize,
    ransac_draws: usize,
    ransac_pruned: usize,
    ransac_point_tests: usize,
    ransac_point_tests_skipped: usize,
    ransac_adaptive_stops: usize,
    ransac_iterations_saved: usize,
}

impl SurfaceEvaluationTimings {
    fn add_assign(&mut self, other: Self) {
        self.total += other.total;
        self.center_precheck += other.center_precheck;
        self.neighborhood += other.neighborhood;
        self.insufficient_neighbors += other.insufficient_neighbors;
        self.ransac += other.ransac;
        self.post_ransac_gate += other.post_ransac_gate;
        self.pca += other.pca;
        self.rmse_and_quality_check += other.rmse_and_quality_check;
        self.timing_samples += other.timing_samples;
        self.ransac_calls += other.ransac_calls;
        self.ransac_draws += other.ransac_draws;
        self.ransac_pruned += other.ransac_pruned;
        self.ransac_point_tests += other.ransac_point_tests;
        self.ransac_point_tests_skipped += other.ransac_point_tests_skipped;
        self.ransac_adaptive_stops += other.ransac_adaptive_stops;
        self.ransac_iterations_saved += other.ransac_iterations_saved;
    }

    fn other(self) -> Duration {
        self.total
            .saturating_sub(self.center_precheck)
            .saturating_sub(self.neighborhood)
            .saturating_sub(self.insufficient_neighbors)
            .saturating_sub(self.ransac)
            .saturating_sub(self.post_ransac_gate)
            .saturating_sub(self.pca)
            .saturating_sub(self.rmse_and_quality_check)
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct SurfaceClassificationTimings {
    total_wall: Duration,
    parallel_evaluation_wall: Duration,
    evaluation_reduce_wall: Duration,
    evaluation_worker_sum: Duration,
    center_precheck_worker_sum: Duration,
    neighborhood_cpu_sum: Duration,
    insufficient_neighbors_worker_sum: Duration,
    ransac_cpu_sum: Duration,
    post_ransac_gate_worker_sum: Duration,
    pca_cpu_sum: Duration,
    rmse_and_quality_check_worker_sum: Duration,
    evaluation_other_cpu_sum: Duration,
    timing_sample_count: usize,
    evaluation_count: usize,
    timing_sample_scale: f64,
    result_apply_wall: Duration,
    ransac_calls: usize,
    ransac_draws: usize,
    ransac_pruned: usize,
    ransac_point_tests: usize,
    ransac_point_tests_skipped: usize,
    ransac_adaptive_stops: usize,
    ransac_iterations_saved: usize,
}

struct SurfaceEvaluationScratch {
    neighbor_points: Vec<Point3<f32>>,
    inlier_points: Vec<Point3<f32>>,
}

impl SurfaceEvaluationScratch {
    fn with_capacity(max_neighbor_points: usize) -> Self {
        Self {
            neighbor_points: Vec::with_capacity(max_neighbor_points),
            inlier_points: Vec::with_capacity(max_neighbor_points),
        }
    }
}

/// ボクセル内部の詳細時計測は 1/64 だけで行い、通常処理への観測負荷を抑える。
/// SplitMix64 のハッシュを使うため、連続した空間キーにも偏りにくい。
const SURFACE_TIMING_SAMPLE_RATE: usize = 64;

fn should_sample_surface_timing(key: VoxelKey) -> bool {
    // RANSAC の仮説列とは別の salt を使い、サンプリングと平面推定結果の相関を避ける。
    let mut sample_state = ransac_seed(key) ^ 0xd1b5_4a32_d192_ed03;
    next_random_index(&mut sample_state, SURFACE_TIMING_SAMPLE_RATE) == 0
}

fn evaluate_surface_voxel(
    voxel_map: &VoxelMap,
    center_key: VoxelKey,
    config: &SurfaceFilterConfig,
    scratch: &mut SurfaceEvaluationScratch,
    detailed_profile_enabled: bool,
) -> Option<SurfaceEvaluation> {
    let evaluation_start = detailed_profile_enabled.then(Instant::now);
    let mut timings = SurfaceEvaluationTimings::default();
    timings.timing_samples = usize::from(detailed_profile_enabled);

    let center_precheck_start = detailed_profile_enabled.then(Instant::now);
    scratch.neighbor_points.clear();
    scratch.inlier_points.clear();
    let center_cell = voxel_map.get(&center_key);
    let center_is_immature =
        center_cell.is_some_and(|cell| cell.observed_frames < config.min_center_observed_frames);
    timings.center_precheck = center_precheck_start
        .map(|start| start.elapsed())
        .unwrap_or_default();
    let center_cell = center_cell?;
    if center_is_immature {
        return Some(SurfaceEvaluation {
            status: SurfaceStatus::Unknown,
            plane: None,
            timings: finish_evaluation_timing(timings, evaluation_start),
        });
    }

    let neighborhood_start = detailed_profile_enabled.then(Instant::now);
    let radius = config.neighbor_radius_voxels.max(0);
    for dz in -radius..=radius {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let key = VoxelKey {
                    ix: center_key.ix + dx,
                    iy: center_key.iy + dy,
                    iz: center_key.iz + dz,
                };
                if let Some(cell) = voxel_map.get(&key) {
                    scratch.neighbor_points.push(cell.mean);
                }
            }
        }
    }
    timings.neighborhood = neighborhood_start
        .map(|start| start.elapsed())
        .unwrap_or_default();

    let insufficient_neighbors_start = detailed_profile_enabled.then(Instant::now);
    let has_insufficient_neighbors = scratch.neighbor_points.len() < config.min_neighbors;
    timings.insufficient_neighbors = insufficient_neighbors_start
        .map(|start| start.elapsed())
        .unwrap_or_default();
    if has_insufficient_neighbors {
        return Some(SurfaceEvaluation {
            status: SurfaceStatus::Unknown,
            plane: None,
            timings: finish_evaluation_timing(timings, evaluation_start),
        });
    }

    let ransac_start = detailed_profile_enabled.then(Instant::now);
    let ransac_outcome = ransac_plane_inliers(
        &scratch.neighbor_points,
        center_key,
        config.ransac_iterations,
        config.ransac_min_iterations,
        config.ransac_confidence,
        config.ransac_inlier_distance_m,
        &mut scratch.inlier_points,
    );
    timings.ransac = ransac_start
        .map(|start| start.elapsed())
        .unwrap_or_default();

    let post_ransac_gate_start = detailed_profile_enabled.then(Instant::now);
    timings.ransac_calls = 1;
    timings.ransac_draws = ransac_outcome.draws;
    timings.ransac_pruned = ransac_outcome.pruned_hypotheses;
    timings.ransac_point_tests = ransac_outcome.point_tests;
    timings.ransac_point_tests_skipped = ransac_outcome.point_tests_skipped;
    timings.ransac_adaptive_stops = usize::from(ransac_outcome.adaptive_stopped);
    timings.ransac_iterations_saved = ransac_outcome.iterations_saved;
    let inlier_ratio = scratch.inlier_points.len() as f32 / scratch.neighbor_points.len() as f32;
    let rejected_by_ransac_gate = !ransac_outcome.found
        || scratch.inlier_points.len() < config.min_ransac_inliers
        || inlier_ratio < config.min_inlier_ratio;
    timings.post_ransac_gate = post_ransac_gate_start
        .map(|start| start.elapsed())
        .unwrap_or_default();
    if rejected_by_ransac_gate {
        return Some(SurfaceEvaluation {
            status: SurfaceStatus::NonPlanar,
            plane: None,
            timings: finish_evaluation_timing(timings, evaluation_start),
        });
    }

    // RANSAC で外れ値を除去した後の PCA は、この1回だけ実施する。
    let pca_start = detailed_profile_enabled.then(Instant::now);
    let fitted_plane = fit_plane(&scratch.inlier_points);
    timings.pca = pca_start.map(|start| start.elapsed()).unwrap_or_default();

    let rmse_and_quality_check_start = detailed_profile_enabled.then(Instant::now);
    let Some(fitted_plane) = fitted_plane else {
        timings.rmse_and_quality_check = rmse_and_quality_check_start
            .map(|start| start.elapsed())
            .unwrap_or_default();
        return Some(SurfaceEvaluation {
            status: SurfaceStatus::NonPlanar,
            plane: None,
            timings: finish_evaluation_timing(timings, evaluation_start),
        });
    };

    let rmse_m = (scratch
        .inlier_points
        .iter()
        .map(|point| (fitted_plane.normal.dot(&point.coords) + fitted_plane.d).powi(2))
        .sum::<f32>()
        / scratch.inlier_points.len() as f32)
        .sqrt();
    let center_distance =
        (fitted_plane.normal.dot(&center_cell.mean.coords) + fitted_plane.d).abs();

    let is_planar = fitted_plane.planarity() >= config.min_planarity
        && fitted_plane.surface_variation() <= config.max_surface_variation
        && rmse_m <= config.max_pca_rmse_m
        && center_distance <= config.max_center_distance_m;

    let plane = is_planar.then_some(SurfacePlane {
        normal: fitted_plane.normal,
        d: fitted_plane.d,
        rmse_m,
        neighbor_count: scratch.neighbor_points.len(),
        inlier_count: scratch.inlier_points.len(),
    });
    timings.rmse_and_quality_check = rmse_and_quality_check_start
        .map(|start| start.elapsed())
        .unwrap_or_default();

    Some(SurfaceEvaluation {
        status: if is_planar {
            SurfaceStatus::Planar
        } else {
            SurfaceStatus::NonPlanar
        },
        plane,
        timings: finish_evaluation_timing(timings, evaluation_start),
    })
}

fn finish_evaluation_timing(
    mut timings: SurfaceEvaluationTimings,
    evaluation_start: Option<Instant>,
) -> SurfaceEvaluationTimings {
    timings.total = evaluation_start
        .map(|start| start.elapsed())
        .unwrap_or_default();
    timings
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn scale_duration(duration: Duration, scale: f64) -> Duration {
    if scale <= 0.0 || !scale.is_finite() {
        return Duration::default();
    }

    Duration::from_secs_f64(duration.as_secs_f64() * scale)
}

fn ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

fn percentage(part: usize, total: usize) -> f64 {
    ratio(part as f64, total as f64) * 100.0
}

#[derive(Debug, Clone, Copy, Default)]
struct RansacOutcome {
    found: bool,
    draws: usize,
    pruned_hypotheses: usize,
    point_tests: usize,
    point_tests_skipped: usize,
    adaptive_stopped: bool,
    iterations_saved: usize,
}

/// 3点仮説を決定論的にサンプリングし、最多インライアを `inliers` に格納する。
/// 同数の場合はインライアの二乗残差和が小さい仮説を優先する。
fn ransac_plane_inliers(
    points: &[Point3<f32>],
    center_key: VoxelKey,
    iterations: usize,
    min_iterations: usize,
    confidence: f64,
    inlier_distance_m: f32,
    inliers: &mut Vec<Point3<f32>>,
) -> RansacOutcome {
    inliers.clear();
    if points.len() < 3
        || iterations == 0
        || !inlier_distance_m.is_finite()
        || inlier_distance_m <= 0.0
    {
        return RansacOutcome::default();
    }

    let threshold_sq = inlier_distance_m * inlier_distance_m;
    let mut random_state = ransac_seed(center_key);
    let mut best_plane = None;
    let mut best_inlier_count = 0;
    let mut best_squared_error = f32::INFINITY;
    let mut outcome = RansacOutcome::default();
    let mut required_iterations = iterations;

    while outcome.draws < iterations && outcome.draws < required_iterations {
        outcome.draws += 1;
        let [i0, i1, i2] = sample_three_distinct_indices(&mut random_state, points.len());
        let p0 = points[i0];
        let v1 = points[i1].coords - p0.coords;
        let v2 = points[i2].coords - p0.coords;
        let normal = v1.cross(&v2);
        let normal_norm = normal.norm();
        if !normal_norm.is_finite() || normal_norm <= 1e-6 {
            continue;
        }

        let normal = normal / normal_norm;
        let d = -normal.dot(&p0.coords);
        let mut inlier_count = 0;
        let mut squared_error = 0.0;
        let mut hypothesis_pruned = false;

        for (point_index, point) in points.iter().enumerate() {
            let distance = normal.dot(&point.coords) + d;
            let distance_sq = distance * distance;
            if distance_sq <= threshold_sq {
                inlier_count += 1;
                squared_error += distance_sq;
            }

            let remaining_points = points.len() - point_index - 1;
            let maximum_possible_inliers = inlier_count + remaining_points;
            let cannot_beat_best = maximum_possible_inliers < best_inlier_count
                || (maximum_possible_inliers == best_inlier_count
                    && squared_error >= best_squared_error);
            if cannot_beat_best {
                outcome.pruned_hypotheses += 1;
                outcome.point_tests += point_index + 1;
                outcome.point_tests_skipped += remaining_points;
                hypothesis_pruned = true;
                break;
            }
        }

        if hypothesis_pruned {
            continue;
        }
        outcome.point_tests += points.len();

        // 全点がインライアなら、後続仮説で誤差が改善しても返す点集合は変わらない。
        // 後段ではこの全点から PCA 平面を再推定するため、ここで安全に打ち切れる。
        if inlier_count == points.len() {
            inliers.extend_from_slice(points);
            outcome.found = true;
            return outcome;
        }

        if inlier_count > best_inlier_count
            || (inlier_count == best_inlier_count && squared_error < best_squared_error)
        {
            best_plane = Some((normal, d));
            best_inlier_count = inlier_count;
            best_squared_error = squared_error;
            required_iterations = required_iterations.min(adaptive_ransac_iteration_limit(
                best_inlier_count,
                points.len(),
                confidence,
                min_iterations,
                iterations,
            ));
        }
    }

    if required_iterations < iterations && outcome.draws >= required_iterations {
        outcome.adaptive_stopped = true;
        outcome.iterations_saved = iterations - outcome.draws;
    }

    let Some((best_normal, best_d)) = best_plane else {
        return outcome;
    };

    inliers.extend(points.iter().copied().filter(|point| {
        let distance = best_normal.dot(&point.coords) + best_d;
        distance * distance <= threshold_sq
    }));
    outcome.found = !inliers.is_empty();
    outcome
}

fn adaptive_ransac_iteration_limit(
    best_inlier_count: usize,
    point_count: usize,
    confidence: f64,
    min_iterations: usize,
    max_iterations: usize,
) -> usize {
    if max_iterations == 0 {
        return 0;
    }

    let minimum = min_iterations.max(1).min(max_iterations);
    if best_inlier_count == 0
        || point_count == 0
        || !confidence.is_finite()
        || !(0.0..1.0).contains(&confidence)
    {
        return max_iterations;
    }

    let inlier_ratio = (best_inlier_count.min(point_count) as f64) / point_count as f64;
    let all_inlier_sample_probability = inlier_ratio.powi(3);
    if all_inlier_sample_probability >= 1.0 {
        return minimum;
    }

    // confidence <= 1 - (1 - w^3)^k を満たす最小の反復数 k を求める。
    let required = ((1.0 - confidence).ln() / (-all_inlier_sample_probability).ln_1p()).ceil();
    if !required.is_finite() || required <= 0.0 {
        return max_iterations;
    }

    (required as usize).clamp(minimum, max_iterations)
}

fn sample_three_distinct_indices(random_state: &mut u64, len: usize) -> [usize; 3] {
    let first = next_random_index(random_state, len);

    let mut second = next_random_index(random_state, len - 1);
    if second >= first {
        second += 1;
    }

    let mut third = next_random_index(random_state, len - 2);
    let lower_excluded = first.min(second);
    let upper_excluded = first.max(second);
    if third >= lower_excluded {
        third += 1;
    }
    if third >= upper_excluded {
        third += 1;
    }

    [first, second, third]
}

fn next_random_index(random_state: &mut u64, len: usize) -> usize {
    // SplitMix64: 外部乱数依存なしで、同じ VoxelKey から常に同じ仮説列を生成する。
    *random_state = random_state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *random_state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    (value % len as u64) as usize
}

fn ransac_seed(key: VoxelKey) -> u64 {
    let x = key.ix as u32 as u64;
    let y = key.iy as u32 as u64;
    let z = key.iz as u32 as u64;
    x.wrapping_mul(0x9e37_79b1_85eb_ca87)
        ^ y.wrapping_mul(0xc2b2_ae3d_27d4_eb4f)
        ^ z.wrapping_mul(0x1656_67b1_9e37_79f9)
        ^ 0xa076_1d64_78bd_642f
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
            point: (Point3::origin(), 0),
            is_point: true,
            mean: Point3::origin(),

            m2: Matrix3::zeros(),
            sample_count: 0,
            observed_frames: 0,
            last_observed_frame_id: None,

            voxel_key: VoxelKey {
                ix: 0,
                iy: 0,
                iz: 0,
            },

            covariance: Matrix3::zeros(),
            covariance_valid: false,

            surface_status: SurfaceStatus::Unknown,
            surface_plane: None,
            surface_evaluation_epoch: 0,
            surface_evaluated_through_frame_id: None,
        }
    }

    pub fn from_key(key: &VoxelKey, voxel_size: f32, point: Point3<f32>, frame_id: u64) -> Self {
        let _ = voxel_size;
        Self {
            point: (point, frame_id),
            is_point: true,

            mean: point,
            m2: Matrix3::zeros(),
            sample_count: 1,
            observed_frames: 1,
            last_observed_frame_id: Some(frame_id),

            voxel_key: *key,

            covariance: Matrix3::zeros(),
            covariance_valid: false,

            surface_status: SurfaceStatus::Unknown,
            surface_plane: None,
            surface_evaluation_epoch: 0,
            surface_evaluated_through_frame_id: None,
        }
    }

    pub fn update_statistics(
        &mut self,
        point: Point3<f32>,
        frame_id: u64,
        min_samples_for_covariance: usize,
    ) {
        if !point.coords.iter().all(|v| v.is_finite()) {
            return;
        }

        let new_count = self.sample_count + 1;

        let delta = point.coords - self.mean.coords;
        let new_mean = self.mean.coords + delta / new_count as f32;
        let delta2 = point.coords - new_mean;

        self.m2 += delta * delta2.transpose();

        self.sample_count = new_count;
        self.mean = Point3::from(new_mean);

        if self.last_observed_frame_id != Some(frame_id) {
            self.observed_frames += 1;
            self.last_observed_frame_id = Some(frame_id);
        }

        self.point = (self.mean, frame_id);

        if self.sample_count >= min_samples_for_covariance as u64 && self.sample_count >= 2 {
            let covariance = self.m2 / (self.sample_count - 1) as f32;

            self.covariance = (covariance + covariance.transpose()) * 0.5;

            self.covariance_valid = self.covariance.iter().all(|v| v.is_finite());
        } else {
            self.covariance_valid = false;
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
    let mut voxel_map = VoxelMap::default();

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

#[cfg(test)]
mod tests {
    use super::*;

    fn map_config() -> LocalMapConfig {
        LocalMapConfig {
            index_voxel_size: 0.05,
            max_points_per_voxel: 20,
            min_points_per_voxel: 2,
            min_observed_frames_per_voxel: 2,
            max_frames: 50,
            max_distance: 150.0,
        }
    }

    fn surface_filter_config() -> SurfaceFilterConfig {
        SurfaceFilterConfig {
            neighbor_radius_voxels: 2,
            min_neighbors: 10,
            min_ransac_inliers: 8,
            min_center_observed_frames: 2,
            ransac_iterations: 64,
            ransac_confidence: 0.999,
            ransac_min_iterations: 8,
            ransac_inlier_distance_m: 0.025,
            min_inlier_ratio: 0.50,
            min_planarity: 0.20,
            max_surface_variation: 0.04,
            max_pca_rmse_m: 0.025,
            max_center_distance_m: 0.025,
        }
    }

    fn world_update_filter_config() -> WorldMapUpdateFilterConfig {
        WorldMapUpdateFilterConfig {
            mature_plane_search_radius_voxels: 2,
            min_mature_observed_frames: 3,
            accept_distance_m: 0.02,
            pending_distance_m: 0.10,
            project_accepted_points: true,
            pending_max_age_frames: 2,
        }
    }

    fn insert_test_cell(map: &mut LOCALMap, key: VoxelKey, point: Point3<f32>) {
        let mut cell = VoxelCell::from_key(&key, map.config.index_voxel_size, point, 1);
        cell.sample_count = 2;
        cell.observed_frames = 2;
        map.voxel_map.insert(key, cell);
    }

    fn insert_xy_plane(map: &mut LOCALMap, center_is_outlier: bool) {
        for iy in -2..=2 {
            for ix in -2..=2 {
                let key = VoxelKey { ix, iy, iz: 0 };
                let is_center = ix == 0 && iy == 0;
                let is_other_outlier = ix == 2 && iy == 2;
                let z = if (center_is_outlier && is_center)
                    || (!center_is_outlier && is_other_outlier)
                {
                    0.15
                } else {
                    0.0
                };
                insert_test_cell(map, key, Point3::new(ix as f32 * 0.05, iy as f32 * 0.05, z));
            }
        }
    }

    fn insert_mature_yz_plane_cell(map: &mut LOCALMap) {
        let key = VoxelKey {
            ix: 0,
            iy: 0,
            iz: 0,
        };
        let mut cell = VoxelCell::from_key(
            &key,
            map.config.index_voxel_size,
            Point3::origin(),
            0,
        );
        cell.sample_count = 5;
        cell.observed_frames = 5;
        cell.surface_status = SurfaceStatus::Planar;
        cell.surface_plane = Some(SurfacePlane {
            normal: Vector3::new(1.0, 0.0, 0.0),
            d: 0.0,
            rmse_m: 0.005,
            neighbor_count: 20,
            inlier_count: 18,
        });
        map.voxel_map.insert(key, cell);
    }

    #[test]
    fn ransac_then_single_pca_keeps_center_on_dominant_plane() {
        let mut map = LOCALMap::new(map_config());
        insert_xy_plane(&mut map, false);

        map.classify_all_surface_voxels(&surface_filter_config());

        let center = map
            .voxel_map
            .get(&VoxelKey {
                ix: 0,
                iy: 0,
                iz: 0,
            })
            .unwrap();
        assert_eq!(center.surface_status, SurfaceStatus::Planar);
        assert_eq!(center.surface_plane.unwrap().inlier_count, 24);
    }

    #[test]
    fn ransac_then_single_pca_rejects_center_away_from_dominant_plane() {
        let mut map = LOCALMap::new(map_config());
        insert_xy_plane(&mut map, true);

        map.classify_all_surface_voxels(&surface_filter_config());

        let center = map
            .voxel_map
            .get(&VoxelKey {
                ix: 0,
                iy: 0,
                iz: 0,
            })
            .unwrap();
        assert_eq!(center.surface_status, SurfaceStatus::NonPlanar);
        assert!(center.surface_plane.is_none());
    }

    #[test]
    fn ransac_removes_parallel_noise_before_pca() {
        let mut points = Vec::new();
        for iy in -2..=2 {
            for ix in -2..=2 {
                points.push(Point3::new(ix as f32 * 0.05, iy as f32 * 0.05, 0.0));
            }
        }
        for ix in -2..=2 {
            points.push(Point3::new(ix as f32 * 0.05, 0.0, 0.06));
            points.push(Point3::new(ix as f32 * 0.05, 0.0, -0.06));
        }

        let center_key = VoxelKey {
            ix: 0,
            iy: 0,
            iz: 0,
        };
        let mut first = Vec::new();
        let mut second = Vec::new();
        assert!(ransac_plane_inliers(&points, center_key, 64, 8, 0.999, 0.025, &mut first,).found);
        assert!(ransac_plane_inliers(&points, center_key, 64, 8, 0.999, 0.025, &mut second,).found);

        assert_eq!(first, second);
        assert_eq!(first.len(), 25);
        assert!(first.iter().all(|point| point.z == 0.0));
    }

    #[test]
    fn surface_timing_sampling_is_deterministic_and_sparse() {
        let keys: Vec<VoxelKey> = (0..4096)
            .map(|ix| VoxelKey {
                ix,
                iy: ix.wrapping_mul(17),
                iz: ix.wrapping_mul(-31),
            })
            .collect();
        let first: Vec<bool> = keys
            .iter()
            .map(|key| should_sample_surface_timing(*key))
            .collect();
        let second: Vec<bool> = keys
            .iter()
            .map(|key| should_sample_surface_timing(*key))
            .collect();
        let sampled = first.iter().filter(|sampled| **sampled).count();

        assert_eq!(first, second);
        assert!((32..=96).contains(&sampled), "sampled={sampled}");
    }

    #[test]
    fn adaptive_ransac_limit_uses_best_inlier_ratio() {
        assert_eq!(adaptive_ransac_iteration_limit(50, 100, 0.999, 8, 48), 48);
        assert_eq!(adaptive_ransac_iteration_limit(60, 100, 0.999, 8, 48), 29);
        assert_eq!(adaptive_ransac_iteration_limit(70, 100, 0.999, 8, 48), 17);
        assert_eq!(adaptive_ransac_iteration_limit(80, 100, 0.999, 8, 48), 10);
        assert_eq!(adaptive_ransac_iteration_limit(90, 100, 0.999, 8, 48), 8);
    }

    #[test]
    fn ransac_prunes_losing_hypotheses() {
        let points = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
        ];
        let center_key = VoxelKey {
            ix: 7,
            iy: -3,
            iz: 11,
        };
        let mut inliers = Vec::new();

        let outcome = ransac_plane_inliers(&points, center_key, 64, 8, 0.999, 0.01, &mut inliers);

        assert!(outcome.found);
        assert!(outcome.pruned_hypotheses > 0);
        assert!(outcome.point_tests > 0);
        assert!(outcome.point_tests_skipped > 0);
        assert!(outcome.adaptive_stopped);
        assert!(outcome.iterations_saved > 0);
    }

    #[test]
    fn surface_filter_accepts_eight_planar_inliers_in_sparse_neighborhood() {
        let mut map = LOCALMap::new(map_config());
        for iy in -1..=1 {
            for ix in -1..=1 {
                if ix == 1 && iy == 1 {
                    continue;
                }
                let key = VoxelKey { ix, iy, iz: 0 };
                insert_test_cell(
                    &mut map,
                    key,
                    Point3::new(ix as f32 * 0.05, iy as f32 * 0.05, 0.0),
                );
            }
        }
        insert_test_cell(
            &mut map,
            VoxelKey {
                ix: 2,
                iy: 2,
                iz: 1,
            },
            Point3::new(0.10, 0.10, 0.08),
        );
        insert_test_cell(
            &mut map,
            VoxelKey {
                ix: -2,
                iy: -2,
                iz: -1,
            },
            Point3::new(-0.10, -0.10, -0.08),
        );

        map.classify_all_surface_voxels(&surface_filter_config());

        let center = map
            .voxel_map
            .get(&VoxelKey {
                ix: 0,
                iy: 0,
                iz: 0,
            })
            .unwrap();
        assert_eq!(center.surface_status, SurfaceStatus::Planar);
        assert!(center.surface_plane.unwrap().inlier_count >= 8);
    }

    #[test]
    fn delayed_classification_releases_frame_after_requested_age() {
        let mut map = LOCALMap::new(map_config());
        let pose = Matrix4::<f64>::identity();
        let points = [Point3::new(0.01, 0.01, 0.01)];
        let filter_config = surface_filter_config();

        map.update_world_map(&points, &pose);
        assert_eq!(
            map.classify_delayed_surface_voxels(2, &filter_config)
                .evaluated,
            0
        );
        assert_eq!(map.frame_index.len(), 1);

        map.update_world_map(&points, &pose);
        assert_eq!(
            map.classify_delayed_surface_voxels(2, &filter_config)
                .evaluated,
            0
        );
        assert_eq!(map.frame_index.len(), 2);

        map.update_world_map(&points, &pose);
        assert_eq!(
            map.classify_delayed_surface_voxels(2, &filter_config)
                .evaluated,
            1
        );
        assert_eq!(map.frame_index.len(), 2);
        assert_eq!(map.frame_index.front().unwrap().frame_id, 1);

        // フレーム 2 の評価は、その時点の累積マップ（フレーム 0..=2）を使用済み。
        // 後から解放されるフレーム 1, 2 の dirty entry では再評価しない。
        map.update_world_map(&points, &pose);
        let stats = map.classify_delayed_surface_voxels(2, &filter_config);
        assert_eq!(stats.evaluated, 0);
        assert_eq!(stats.skipped_unchanged_probes, 1);

        map.update_world_map(&points, &pose);
        let stats = map.classify_delayed_surface_voxels(2, &filter_config);
        assert_eq!(stats.evaluated, 0);
        assert_eq!(stats.skipped_unchanged_probes, 1);

        // フレーム 3 は前回評価より新しいため、delay 経過後に再評価する。
        map.update_world_map(&points, &pose);
        let stats = map.classify_delayed_surface_voxels(2, &filter_config);
        assert_eq!(stats.evaluated, 1);
        assert_eq!(stats.skipped_unchanged_probes, 0);
    }

    #[test]
    fn delayed_classification_re_evaluates_after_filter_config_change() {
        let mut map = LOCALMap::new(map_config());
        let pose = Matrix4::<f64>::identity();
        let points = [Point3::new(0.01, 0.01, 0.01)];
        let filter_config = surface_filter_config();

        for _ in 0..3 {
            map.update_world_map(&points, &pose);
        }
        assert_eq!(
            map.classify_delayed_surface_voxels(2, &filter_config)
                .evaluated,
            1
        );

        map.update_world_map(&points, &pose);
        let mut changed_config = filter_config;
        changed_config.min_neighbors += 1;
        let stats = map.classify_delayed_surface_voxels(2, &changed_config);
        assert_eq!(stats.evaluated, 1);
        assert_eq!(stats.skipped_unchanged_probes, 0);
    }

    #[test]
    fn delayed_classification_handles_immature_center_without_parallel_evaluation() {
        let mut map = LOCALMap::new(map_config());
        let pose = Matrix4::<f64>::identity();
        let filter_config = surface_filter_config();

        // 各フレームを互いの近傍半径外へ配置し、frame 0 の中心を observed_frames=1 の
        // 未成熟セルとして遅延判定へ渡す。
        map.update_world_map(&[Point3::new(0.01, 0.01, 0.01)], &pose);
        map.update_world_map(&[Point3::new(1.01, 0.01, 0.01)], &pose);
        map.update_world_map(&[Point3::new(2.01, 0.01, 0.01)], &pose);

        let stats = map.classify_delayed_surface_voxels(2, &filter_config);
        assert_eq!(stats.evaluated, 1);
        assert_eq!(stats.unknown, 1);
        assert_eq!(stats.immature_fast_path, 1);

        let cell = map
            .voxel_map
            .get(&VoxelKey {
                ix: 0,
                iy: 0,
                iz: 0,
            })
            .unwrap();
        assert_eq!(cell.surface_status, SurfaceStatus::Unknown);
        assert_eq!(cell.surface_evaluated_through_frame_id, Some(2));
    }

    #[test]
    fn filtered_world_update_projects_observation_onto_mature_plane() {
        let mut map = LOCALMap::new(map_config());
        insert_mature_yz_plane_cell(&mut map);

        let stats = map.update_world_map_filtered(
            &[Point3::new(0.015, 0.01, 0.01)],
            &Matrix4::<f64>::identity(),
            &world_update_filter_config(),
        );

        assert_eq!(stats.projected_to_mature_plane, 1);
        assert_eq!(stats.held_pending, 0);
        let updated = map
            .voxel_map
            .get(&VoxelKey {
                ix: 0,
                iy: 0,
                iz: 0,
            })
            .unwrap();
        assert!(updated.mean.x.abs() < 1e-6);
    }

    #[test]
    fn filtered_world_update_holds_seven_centimeter_parallel_layer() {
        let mut map = LOCALMap::new(map_config());
        insert_mature_yz_plane_cell(&mut map);
        let original_voxel_count = map.voxel_map.len();

        let stats = map.update_world_map_filtered(
            &[Point3::new(0.07, 0.01, 0.01)],
            &Matrix4::<f64>::identity(),
            &world_update_filter_config(),
        );

        assert_eq!(stats.held_pending, 1);
        assert_eq!(stats.projected_to_mature_plane, 0);
        assert_eq!(stats.pending_voxels, 1);
        assert_eq!(map.voxel_map.len(), original_voxel_count);
        assert!(!map.voxel_map.contains_key(&VoxelKey {
            ix: 1,
            iy: 0,
            iz: 0,
        }));
    }

    #[test]
    fn filtered_world_update_keeps_new_structure_without_nearby_mature_plane() {
        let mut map = LOCALMap::new(map_config());

        let stats = map.update_world_map_filtered(
            &[Point3::new(1.0, 0.0, 0.0)],
            &Matrix4::<f64>::identity(),
            &world_update_filter_config(),
        );

        assert_eq!(stats.inserted_provisional, 1);
        assert_eq!(stats.held_pending, 0);
        assert_eq!(map.voxel_map.len(), 1);
    }
}
