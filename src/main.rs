use anyhow::{Result, bail};
use nalgebra::{Matrix3, Matrix4, Point3, Quaternion, UnitQuaternion, Vector3};
use re_lidar_slam::{
    deskew_points::deskew_points,
    file_handler::{load_imu_data, load_pcd_files, load_pcd_xyzit, save_pcd_xyz},
    find_nearest_points::pickup_valid_source_points,
    icp::{apply_delta, build_point_to_plane_system, compute_rmse, solve_icp_delta},
    predict_pose_by_imu::{align_imu_timestamps, build_rotation_trajectory, predict_pose_by_imu},
    types::{CurrentFrameInfo, FrameLog, IMU, PointXYZ, SLAMMap},
    voxel_map::{
        LOCALMap, LocalMapConfig, SurfaceFilterConfig, SurfaceStatus, WorldMapUpdateFilterConfig,
        build_voxel_map,
    },
    voxelization::voxel_downsample_points,
};
use std::{
    ffi::OsString,
    time::{Duration, Instant},
};

const DATASET_DIR: &str = "data/input/08012026/08012026-airy96-mid70-07-church";
const SAVE_ROOT_DIR: &str = "data/output/debug/08192026";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LidarModel {
    Mid70,
    Airy96,
}

impl LidarModel {
    fn parse(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "mid70" | "mid-70" => Ok(Self::Mid70),
            "airy96" | "airy-96" | "airy" => Ok(Self::Airy96),
            _ => bail!("unsupported LiDAR model '{value}'; expected 'mid70' or 'airy96'"),
        }
    }

    fn input_subdir(self) -> &'static str {
        match self {
            Self::Mid70 => "mid-70",
            Self::Airy96 => "airy",
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Mid70 => "mid70",
            Self::Airy96 => "airy96",
        }
    }

    fn imu_to_lidar_rotation(self) -> UnitQuaternion<f64> {
        match self {
            Self::Mid70 => make_imu_to_mid70_rotation(),
            Self::Airy96 => make_imu_to_airy96_rotation(),
        }
    }
}

enum CommandLineAction {
    Run(LidarModel),
    Help,
}

fn parse_command_line<I>(args: I) -> Result<CommandLineAction>
where
    I: IntoIterator<Item = OsString>,
{
    let mut lidar_model = LidarModel::Mid70;
    let mut args = args.into_iter();

    while let Some(argument) = args.next() {
        let argument = argument
            .into_string()
            .map_err(|_| anyhow::anyhow!("command-line arguments must be valid UTF-8"))?;

        match argument.as_str() {
            "-h" | "--help" => return Ok(CommandLineAction::Help),
            "--lidar" => {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--lidar requires 'mid70' or 'airy96'"))?
                    .into_string()
                    .map_err(|_| anyhow::anyhow!("LiDAR model must be valid UTF-8"))?;
                lidar_model = LidarModel::parse(&value)?;
            }
            _ if argument.starts_with("--lidar=") => {
                lidar_model = LidarModel::parse(&argument["--lidar=".len()..])?;
            }
            _ => bail!("unknown argument '{argument}'; use --help for usage"),
        }
    }

    Ok(CommandLineAction::Run(lidar_model))
}

fn print_usage() {
    println!("Usage: re_lidar_slam [--lidar <mid70|airy96>]");
    println!("  --lidar  Select the point-cloud sensor (default: mid70)");
}

// Mid-70 sparse-cloud preset.
// The upper range matches the range used by the existing Mid-70 datasets.
const MIN_DIST: f32 = 0.5;
const MAX_DIST: f32 = 150.0;

// Airy-96内蔵IMU座標からAiry-96 LiDAR座標への外部回転。
// Quaternion (x, y, z, w): -0.705437, 0.708767, -0.00246579, 0.00097028
// Translation (x, y, z)  : 0.00425, 0.00418, -0.00446  [m]
const IMU_TO_AIRY96_QUAT_X: f64 = -0.705437;
const IMU_TO_AIRY96_QUAT_Y: f64 = 0.708767;
const IMU_TO_AIRY96_QUAT_Z: f64 = -0.00246579;
const IMU_TO_AIRY96_QUAT_W: f64 = 0.00097028;

// Mid-70原点をAiry-96座標で表した位置。回転変換の導出には回転成分のみを使う。
const MID70_ORIGIN_IN_AIRY96_X_M: f64 = 0.0;
const MID70_ORIGIN_IN_AIRY96_Y_M: f64 = 0.0;
const MID70_ORIGIN_IN_AIRY96_Z_M: f64 = -0.06;

// Mid-70 is sparser than Airy-96. Keep enough spatial support in each local-map
// cell for stable pose estimation; the stricter filters below are used to keep
// wall/ground boundary points out of the global map.
const DOWNSAMPLE_VOXEL_SIZE: f32 = 0.25; // m
const LOCAL_MAP_VOXEL_SIZE: f32 = 0.25; // m
const GLOBAL_MAP_VOXEL_SIZE: f32 = 0.05; // m

const NEIGHBOR_RANGE: i32 = 0; // Unused when build_voxel_map(..., is_target=false)

const LOCAL_KNN_K: usize = 5;
const GLOBAL_KNN_K: usize = 5;
// 1.0 m cells x 3 cells gives a 3.0 m search radius.
const SEARCH_RANGE: i32 = 3;
const MAX_DIST_FACTOR: f32 = 3.0;
// k近傍点が推定平面から離れてよい最大距離 [m]
const LOCAL_PLANE_POINT_DISTANCE_THRESHOLD_M: f32 = 0.25;
const GLOBAL_PLANE_POINT_DISTANCE_THRESHOLD_M: f32 = 0.20;
const LOCAL_SOURCE_PLANE_SCORE_THRESHOLD: f32 = 0.85;
const GLOBAL_SOURCE_PLANE_SCORE_THRESHOLD: f32 = 0.75;
// Source点と推定平面との最大距離 [m]
const GLOBAL_SOURCE_TO_PLANE_MAX_DISTANCE_M: f32 = 0.08;
// Wall/ground edges tend to form line-like or mixed neighborhoods. Reject them
// during global-map insertion, while retaining sparse Mid-70 observations.
const GLOBAL_MIN_PLANARITY: f32 = 0.10;

// Global map の累積点から局所平面を確定するための遅延・5x5x5 RANSAC/PCA 設定。
// 成熟平面を後続フレームの挿入ゲートに使うため、従来の30フレームより早く確定する。
const SURFACE_CLASSIFICATION_DELAY_FRAMES: u64 = 5;
const SURFACE_NEIGHBOR_RADIUS_VOXELS: i32 = 2;
const SURFACE_MIN_NEIGHBORS: usize = 10;
const SURFACE_MIN_RANSAC_INLIERS: usize = 8;
const SURFACE_MIN_CENTER_OBSERVED_FRAMES: u64 = 2;
const SURFACE_RANSAC_ITERATIONS: usize = 48;
const SURFACE_RANSAC_CONFIDENCE: f64 = 0.999;
const SURFACE_RANSAC_MIN_ITERATIONS: usize = 8;
const SURFACE_RANSAC_INLIER_DISTANCE_M: f32 = 0.025;
const SURFACE_MIN_INLIER_RATIO: f32 = 0.50;
const SURFACE_MIN_PLANARITY: f32 = 0.20;
const SURFACE_MAX_VARIATION: f32 = 0.04;
const SURFACE_MAX_PCA_RMSE_M: f32 = 0.025;
const SURFACE_MAX_CENTER_DISTANCE_M: f32 = 0.025;

// 成熟した GlobalMap 平面に対する新規観測の更新ゲート。
// 2 cm以内は同一面として平面へ射影し、2～10 cmは二重壁候補として保留する。
const WORLD_UPDATE_PLANE_SEARCH_RADIUS_VOXELS: i32 = 2;
const WORLD_UPDATE_MIN_MATURE_OBSERVED_FRAMES: u64 = 3;
const WORLD_UPDATE_ACCEPT_DISTANCE_M: f32 = 0.020;
const WORLD_UPDATE_PENDING_DISTANCE_M: f32 = 0.10;
const WORLD_UPDATE_PENDING_MAX_AGE_FRAMES: u64 = 30;

const ICP_ITERATIONS: usize = 8;
const ICP_RMSE_THRESHOLD: f32 = 0.10; // Absolute point-to-plane RMSE convergence threshold [m]
const ICP_RMSE_DIVERGE_THRESHOLD: f32 = 1.0; // Reject ICP and fall back to IMU prediction [m]
const MIN_IMU_SAMPLES_PER_POINT_CLOUD_FRAME: usize = 5;

const MAX_DIST_FOR_VOXEL_MAP: f32 = 150.0;

fn count_imu_samples_in_time_range(imu_data: &[IMU], start_time: f64, end_time: f64) -> usize {
    if !start_time.is_finite() || !end_time.is_finite() || start_time > end_time {
        return 0;
    }

    let start_idx = imu_data.partition_point(|sample| sample.timestamp < start_time);
    let end_idx = imu_data.partition_point(|sample| sample.timestamp <= end_time);

    end_idx.saturating_sub(start_idx)
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();

    let lidar_model = match parse_command_line(std::env::args_os().skip(1))? {
        CommandLineAction::Run(lidar_model) => lidar_model,
        CommandLineAction::Help => {
            print_usage();
            return Ok(());
        }
    };
    let load_dir = format!("{DATASET_DIR}/{}", lidar_model.input_subdir());
    let save_dir = format!("{SAVE_ROOT_DIR}/{}", lidar_model.name());
    let imu_to_lidar = lidar_model.imu_to_lidar_rotation();
    log::info!(
        "LiDAR model={}, input={}, output={}",
        lidar_model.name(),
        load_dir,
        save_dir,
    );

    // <--- Loading each data --->
    let pcd_dir = format!("{load_dir}/pcd");
    let pcd_files = load_pcd_files(&pcd_dir)?;

    log::debug!(
        "Found {} PCD files in directory: {}",
        pcd_files.len(),
        pcd_dir
    );

    let imu_dir = format!("{load_dir}/imu");
    let imu_file = format!("{}/imu_data.json", imu_dir);
    let imu_data = load_imu_data(&imu_file)?;
    let imu_data = align_imu_timestamps(&imu_data); // Align IMU timestamps to seconds
    // <--- Loading each data --->

    // <--- Initialize current frame info --->
    let mut current_frame_info = CurrentFrameInfo {
        current_global_pose: Matrix4::<f64>::identity(),
        current_velocity: Vector3::<f64>::zeros(),
    };
    // <--- Initialize current frame info --->

    // <--- Initialize SLAM map --->
    let local_map_config = LocalMapConfig {
        index_voxel_size: LOCAL_MAP_VOXEL_SIZE,
        max_points_per_voxel: 20,
        min_points_per_voxel: 5,
        min_observed_frames_per_voxel: 3,
        max_frames: 50,
        max_distance: MAX_DIST_FOR_VOXEL_MAP,
    };

    let global_map_config = LocalMapConfig {
        index_voxel_size: GLOBAL_MAP_VOXEL_SIZE,
        max_points_per_voxel: 20,
        min_points_per_voxel: 2,
        min_observed_frames_per_voxel: 2,
        max_frames: 50,
        max_distance: MAX_DIST_FOR_VOXEL_MAP,
    };
    let mut slam_map = SLAMMap {
        global_voxel_map: LOCALMap::new(global_map_config),
        local_voxel_map: LOCALMap::new(local_map_config),
    };
    let surface_filter_config = SurfaceFilterConfig {
        neighbor_radius_voxels: SURFACE_NEIGHBOR_RADIUS_VOXELS,
        min_neighbors: SURFACE_MIN_NEIGHBORS,
        min_ransac_inliers: SURFACE_MIN_RANSAC_INLIERS,
        min_center_observed_frames: SURFACE_MIN_CENTER_OBSERVED_FRAMES,
        ransac_iterations: SURFACE_RANSAC_ITERATIONS,
        ransac_confidence: SURFACE_RANSAC_CONFIDENCE,
        ransac_min_iterations: SURFACE_RANSAC_MIN_ITERATIONS,
        ransac_inlier_distance_m: SURFACE_RANSAC_INLIER_DISTANCE_M,
        min_inlier_ratio: SURFACE_MIN_INLIER_RATIO,
        min_planarity: SURFACE_MIN_PLANARITY,
        max_surface_variation: SURFACE_MAX_VARIATION,
        max_pca_rmse_m: SURFACE_MAX_PCA_RMSE_M,
        max_center_distance_m: SURFACE_MAX_CENTER_DISTANCE_M,
    };
    let world_map_update_filter_config = WorldMapUpdateFilterConfig {
        mature_plane_search_radius_voxels: WORLD_UPDATE_PLANE_SEARCH_RADIUS_VOXELS,
        min_mature_observed_frames: WORLD_UPDATE_MIN_MATURE_OBSERVED_FRAMES,
        accept_distance_m: WORLD_UPDATE_ACCEPT_DISTANCE_M,
        pending_distance_m: WORLD_UPDATE_PENDING_DISTANCE_M,
        project_accepted_points: true,
        pending_max_age_frames: WORLD_UPDATE_PENDING_MAX_AGE_FRAMES,
    };
    // <--- Initialize SLAM map --->

    let mut prev_frame_start_time: f64 = 0.0;
    let mut frame_logs: Vec<FrameLog> = Vec::new();

    //
    for (i, pcd_path) in pcd_files.iter().enumerate() {
        let frame_start = Instant::now();
        log::info!("Processing frame {}: {}", i, pcd_path.to_string_lossy());

        let load_pcd_start = Instant::now();
        let source_pcd = load_pcd_xyzit(&pcd_path.to_string_lossy())?;
        let load_pcd_time = load_pcd_start.elapsed();
        // Main per-frame processing excludes file I/O. The full elapsed time is
        // measured separately from frame_start and reported alongside it.
        let frame_processing_start = Instant::now();

        let timestamp_start = Instant::now();
        let current_frame_start_time = source_pcd
            .iter()
            .map(|p| p.timestamp)
            .fold(f64::INFINITY, f64::min);
        let current_frame_end_time = source_pcd
            .iter()
            .map(|p| p.timestamp)
            .fold(f64::NEG_INFINITY, f64::max);
        let timestamp_time = timestamp_start.elapsed();

        let imu_sample_count = count_imu_samples_in_time_range(
            &imu_data,
            current_frame_start_time,
            current_frame_end_time,
        );
        if imu_sample_count < MIN_IMU_SAMPLES_PER_POINT_CLOUD_FRAME {
            log::warn!(
                "Frame {i}: only {imu_sample_count} IMU samples in point-cloud interval \
                 [{current_frame_start_time:.6}, {current_frame_end_time:.6}] s \
                 (minimum {MIN_IMU_SAMPLES_PER_POINT_CLOUD_FRAME}); \
                 IMU/LiDAR timestamps may be misaligned"
            );
        }

        if i == 0 {
            prev_frame_start_time = current_frame_start_time;
        }

        // <--- Predict pose by IMU --->
        let predict_pose_start = Instant::now();
        let pose_prediction = predict_pose_by_imu(
            &imu_data,
            &imu_to_lidar,
            &current_frame_info.current_global_pose,
            &current_frame_info.current_velocity,
            prev_frame_start_time,
            current_frame_start_time,
        );
        let predict_pose_time = predict_pose_start.elapsed();
        // <--- Predict pose by IMU --->

        // <--- Build rotation trajectory --->
        let rotation_trajectory_start = Instant::now();
        let rotation_traj = build_rotation_trajectory(
            &imu_data,
            current_frame_start_time,
            current_frame_end_time,
            &imu_to_lidar,
        );
        let rotation_trajectory_time = rotation_trajectory_start.elapsed();
        // <--- Build rotation trajectory --->

        // --- Deskew source pcd ---
        let deskew_start = Instant::now();
        let deskewed_points = deskew_points(
            &source_pcd,
            &rotation_traj,
            &imu_to_lidar,
            current_frame_start_time,
            MIN_DIST,
            MAX_DIST,
        );
        let deskew_time = deskew_start.elapsed();
        // --- Deskew source pcd ---

        // --- Downsample deskewed points ---
        let voxel_start = Instant::now();
        let downsampled_source_points_for_local =
            voxel_downsample_points(&deskewed_points, DOWNSAMPLE_VOXEL_SIZE);
        let downsampled_source_points_for_global =
            voxel_downsample_points(&deskewed_points, GLOBAL_MAP_VOXEL_SIZE);
        let voxel_end = voxel_start.elapsed();
        log::debug!(
            "Frame {i}: Downsampled {} points → {} points in {:.2?}",
            deskewed_points.len(),
            downsampled_source_points_for_local.len(),
            voxel_end
        );
        // --- Downsample deskewed points ---

        // --- Build voxel map for source points ---
        let build_map_start = Instant::now();
        let source_voxel_map = build_voxel_map(
            &downsampled_source_points_for_local,
            DOWNSAMPLE_VOXEL_SIZE,
            NEIGHBOR_RANGE,
            false,
        );
        let source_voxel_map_for_global = build_voxel_map(
            &downsampled_source_points_for_global,
            GLOBAL_MAP_VOXEL_SIZE,
            NEIGHBOR_RANGE,
            false,
        );
        let build_map_end = build_map_start.elapsed();
        log::debug!("Frame {i}: Built voxel map in {:.2?}", build_map_end);
        // --- Build voxel map for source points ---

        // --- ICP (Point to Plane) ---
        // IMU 予測姿勢を初期値として (R, t) を取り出す
        let pred_pose = pose_prediction.0.cast::<f32>();
        let mut r_mat: Matrix3<f32> = pred_pose.fixed_view::<3, 3>(0, 0).into();
        let mut t_vec: Vector3<f32> = pred_pose.fixed_view::<3, 1>(0, 3).into();

        let mut prev_rmse = f32::INFINITY;
        let mut icp_ok = false; // ICP が有効な解を得られたか

        let loop_start = Instant::now();

        if slam_map.local_voxel_map.voxel_map.is_empty() {
            log::debug!("Frame {i}: local map empty, skipping ICP");
        } else {
            for _iter in 0..ICP_ITERATIONS {
                // 対応点をピックアップ
                // - source はローカル座標、target (local_voxel_map) はワールド座標
                // - 現在の (R,t) 推定値で source をワールド変換してから近傍探索
                let pickup_start = Instant::now();
                let correspondences = pickup_valid_source_points::<LOCAL_KNN_K>(
                    &source_voxel_map,
                    &slam_map.local_voxel_map.voxel_map,
                    slam_map.local_voxel_map.config.index_voxel_size,
                    SEARCH_RANGE,
                    MAX_DIST_FACTOR,
                    LOCAL_PLANE_POINT_DISTANCE_THRESHOLD_M,
                    LOCAL_SOURCE_PLANE_SCORE_THRESHOLD,
                    None,
                    None,
                    &r_mat,
                    &t_vec,
                );
                let pickup_end = pickup_start.elapsed();
                log::debug!(
                    "ICP iter {_iter}: Picked up {} correspondences in {:.2?}",
                    correspondences.len(),
                    pickup_end
                );

                // 線形システム構築
                let system_start = Instant::now();
                let system = build_point_to_plane_system(&correspondences, &r_mat, &t_vec);

                // 解く → pose 更新
                match solve_icp_delta(&system, 1e-6) {
                    Some(delta) => {
                        (r_mat, t_vec) = apply_delta(&r_mat, &t_vec, &delta);
                        icp_ok = true;
                    }
                    None => {
                        log::warn!("ICP iter {_iter}: solve failed (too few correspondences)");
                        break;
                    }
                }
                let system_end = system_start.elapsed();
                log::debug!(
                    "ICP iter {_iter}: Built point-to-plane system in {:.2?}",
                    system_end
                );

                // RMSE を計算して収束チェック
                let rmse = compute_rmse(&correspondences, &r_mat, &t_vec);
                log::debug!(
                    "ICP iter {_iter}: used={}, cost={:.6}, rmse={:.6}",
                    system.used_count,
                    system.cost,
                    rmse
                );

                // RMSE が発散した場合は ICP 結果を棄却して IMU 予測に戻す
                if rmse > ICP_RMSE_DIVERGE_THRESHOLD {
                    log::warn!(
                        "ICP iter {_iter}: RMSE diverged ({rmse:.4}), reverting to IMU prediction"
                    );
                    r_mat = pred_pose.fixed_view::<3, 3>(0, 0).into();
                    t_vec = pred_pose.fixed_view::<3, 1>(0, 3).into();
                    icp_ok = false;
                    break;
                }

                // if (prev_rmse - rmse).abs() < ICP_RMSE_THRESHOLD {
                if rmse < ICP_RMSE_THRESHOLD {
                    log::debug!(
                        "ICP converged at iter {_iter} (|Δrmse|={:.2e})",
                        (prev_rmse - rmse).abs()
                    );
                    break;
                }
                prev_rmse = rmse;
            }

            if !icp_ok {
                log::warn!("Frame {i}: ICP failed, using IMU prediction");
            }
        }
        let loop_end = loop_start.elapsed();
        log::debug!(
            "Frame {i}: ICP loop finished in {:.2?}, final RMSE={:.6}",
            loop_end,
            prev_rmse
        );
        // --- ICP (Point to Plane) ---

        // --- Update current frame info ---
        let pose_update_start = Instant::now();
        let prev_pos = current_frame_info
            .current_global_pose
            .fixed_view::<3, 1>(0, 3)
            .into_owned();
        let prev_r: Matrix3<f64> = current_frame_info
            .current_global_pose
            .fixed_view::<3, 3>(0, 0)
            .into_owned();

        let r64 = r_mat.cast::<f64>();
        let t64 = t_vec.cast::<f64>();
        let mut new_global_pose = Matrix4::<f64>::identity();
        new_global_pose.fixed_view_mut::<3, 3>(0, 0).copy_from(&r64);
        new_global_pose.fixed_view_mut::<3, 1>(0, 3).copy_from(&t64);

        let new_pos = new_global_pose.fixed_view::<3, 1>(0, 3).into_owned();
        let dt = (current_frame_start_time - prev_frame_start_time).max(1e-6);
        let raw_velocity = (new_pos - prev_pos) / dt;
        // 速度が異常に大きい場合（ICP 発散など）はクランプして安定化
        let new_velocity = raw_velocity.cap_magnitude(2.0);

        current_frame_info.current_global_pose = new_global_pose;
        current_frame_info.current_velocity = new_velocity;
        // --- Update current frame info ---

        // --- Record frame log ---
        let translation_m = (new_pos - prev_pos).norm();
        let delta_r = r64 * prev_r.transpose();
        let rotation_deg = ((delta_r.trace() - 1.0) / 2.0)
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        frame_logs.push(FrameLog {
            frame_index: i,
            timestamp: current_frame_start_time,
            icp_ok,
            rmse: if prev_rmse.is_finite() {
                Some(prev_rmse)
            } else {
                None
            },
            translation_m,
            rotation_deg,
            velocity_m_s: new_velocity.norm(),
            pose_x: new_pos.x,
            pose_y: new_pos.y,
            pose_z: new_pos.z,
        });
        // --- Record frame log ---
        let pose_update_time = pose_update_start.elapsed();

        // --- Filter valid source points, then update the WorldMap ---
        // ローカルマップが空（初回フレーム）の場合はフィルタなしで全点追加。
        // それ以外は pickup_valid_source_points で平面に乗っている点だけ抽出し、
        // ICP 収束後の最終姿勢 (r_mat, t_vec) でワールドマップに追加する。
        let global_filter_start = Instant::now();
        let global_source_points: Vec<Point3<f32>> =
            if slam_map.local_voxel_map.voxel_map.is_empty() {
                downsampled_source_points_for_global.clone()
            } else {
                let valid = pickup_valid_source_points::<GLOBAL_KNN_K>(
                    &source_voxel_map_for_global,
                    &slam_map.local_voxel_map.voxel_map,
                    slam_map.local_voxel_map.config.index_voxel_size,
                    SEARCH_RANGE,
                    MAX_DIST_FACTOR,
                    GLOBAL_PLANE_POINT_DISTANCE_THRESHOLD_M,
                    GLOBAL_SOURCE_PLANE_SCORE_THRESHOLD,
                    Some(GLOBAL_SOURCE_TO_PLANE_MAX_DISTANCE_M),
                    Some(GLOBAL_MIN_PLANARITY),
                    &r_mat,
                    &t_vec,
                );
                log::debug!(
                    "Frame {i}: {} / {} source points passed plane filter for global map",
                    valid.len(),
                    source_voxel_map_for_global.len(),
                );
                valid.into_iter().map(|c| c.src_point).collect()
            };
        let global_filter_time = global_filter_start.elapsed();

        let global_map_update_start = Instant::now();
        let global_update_stats = slam_map.global_voxel_map.update_world_map_filtered(
            &global_source_points,
            &current_frame_info.current_global_pose,
            &world_map_update_filter_config,
        );
        let global_map_update_time = global_map_update_start.elapsed();
        log::debug!(
            "Frame {i}: GlobalMap update input={}, provisional={}, projected={}, \
             pending={}, pending_voxels={}, non_finite={}",
            global_update_stats.input_points,
            global_update_stats.inserted_provisional,
            global_update_stats.projected_to_mature_plane,
            global_update_stats.held_pending,
            global_update_stats.pending_voxels,
            global_update_stats.rejected_non_finite,
        );

        let delayed_surface_start = Instant::now();
        let delayed_surface_stats = slam_map.global_voxel_map.classify_delayed_surface_voxels(
            SURFACE_CLASSIFICATION_DELAY_FRAMES,
            &surface_filter_config,
        );
        let delayed_surface_time = delayed_surface_start.elapsed();
        if delayed_surface_stats.evaluated > 0 {
            log::debug!(
                "Frame {i}: delayed surface classification evaluated={}, planar={}, \
                 non_planar={}, unknown={} in {:.2?}",
                delayed_surface_stats.evaluated,
                delayed_surface_stats.planar,
                delayed_surface_stats.non_planar,
                delayed_surface_stats.unknown,
                delayed_surface_time,
            );
        }
        // --- Filter valid source points, then update the WorldMap ---

        // --- Update the LocalMap with the new frame's points ---
        let local_map_update_start = Instant::now();
        slam_map.local_voxel_map.update_with_new_frame(
            &downsampled_source_points_for_local,
            &current_frame_info.current_global_pose,
        );
        let local_map_update_time = local_map_update_start.elapsed();
        // --- Update the LocalMap with the new frame's points ---

        prev_frame_start_time = current_frame_start_time;

        let frame_processing_time = frame_processing_start.elapsed();
        let frame_total_with_file_io = frame_start.elapsed();
        log::debug!(
            "Frame {i} timings [ms]: load_pcd={:.3} ms, timestamps={:.3} ms, \
             imu_predict={:.3} ms, rotation_trajectory={:.3} ms, deskew={:.3} ms, \
             downsample={:.3} ms, build_source_maps={:.3} ms, icp={:.3} ms, \
             pose_update={:.3} ms, global_filter={:.3} ms, global_map_update={:.3} ms, \
             delayed_surface={:.3} ms, local_map_update={:.3} ms, total={:.3} ms \
             (with_file_io={:.3} ms)",
            duration_ms(load_pcd_time),
            duration_ms(timestamp_time),
            duration_ms(predict_pose_time),
            duration_ms(rotation_trajectory_time),
            duration_ms(deskew_time),
            duration_ms(voxel_end),
            duration_ms(build_map_end),
            duration_ms(loop_end),
            duration_ms(pose_update_time),
            duration_ms(global_filter_time),
            duration_ms(global_map_update_time),
            duration_ms(delayed_surface_time),
            duration_ms(local_map_update_time),
            duration_ms(frame_processing_time),
            duration_ms(frame_total_with_file_io),
        );
    }

    // --- Save the global voxel maps before and after final plane classification ---
    let min_samples = slam_map.global_voxel_map.config.min_points_per_voxel as u64;

    let min_frames = slam_map
        .global_voxel_map
        .config
        .min_observed_frames_per_voxel as u64;

    // 平面処理前: 従来条件を満たす全観測セルを別ファイルへ保存する。
    let world_map_points_before_plane_filter: Vec<Point3<f32>> = slam_map
        .global_voxel_map
        .voxel_map
        .values()
        .filter(|cell| cell.sample_count >= min_samples && cell.observed_frames >= min_frames)
        .map(|cell| Point3::new(cell.mean.x, cell.mean.y, cell.mean.z))
        .collect();

    let downsampled_world_map_points_before_plane_filter =
        voxel_downsample_points(&world_map_points_before_plane_filter, GLOBAL_MAP_VOXEL_SIZE);
    let world_map_points_before_plane_filter_xyz: Vec<PointXYZ> =
        downsampled_world_map_points_before_plane_filter
            .iter()
            .map(|point| PointXYZ {
                x: point.x,
                y: point.y,
                z: point.z,
            })
            .collect();

    std::fs::create_dir_all(&save_dir)?;
    let world_map_before_plane_filter_path = format!(
        "{}/voxel-{}_world_map_before_plane_filter.pcd",
        save_dir, GLOBAL_MAP_VOXEL_SIZE
    );
    save_pcd_xyz(
        &world_map_points_before_plane_filter_xyz,
        &world_map_before_plane_filter_path,
    )?;
    log::info!(
        "Saved world map before plane filter: {} → {} points → {}",
        world_map_points_before_plane_filter.len(),
        world_map_points_before_plane_filter_xyz.len(),
        world_map_before_plane_filter_path,
    );

    // 終端処理: 遅延キューの状態に依存せず、現在の全累積点で全セルを再判定する。
    let final_surface_start = Instant::now();
    let final_surface_stats = slam_map
        .global_voxel_map
        .classify_all_surface_voxels(&surface_filter_config);
    log::info!(
        "Final surface classification: evaluated={}, planar={}, non_planar={}, unknown={} in {:.2?}",
        final_surface_stats.evaluated,
        final_surface_stats.planar,
        final_surface_stats.non_planar,
        final_surface_stats.unknown,
        final_surface_start.elapsed(),
    );

    let planar_world_map_points: Vec<Point3<f32>> = slam_map
        .global_voxel_map
        .voxel_map
        .values()
        .filter(|cell| {
            cell.sample_count >= min_samples
                && cell.observed_frames >= min_frames
                && cell.surface_status == SurfaceStatus::Planar
        })
        .map(|cell| Point3::new(cell.mean.x, cell.mean.y, cell.mean.z))
        .collect();

    let downsampled_planar_world_map_points =
        voxel_downsample_points(&planar_world_map_points, GLOBAL_MAP_VOXEL_SIZE);
    let planar_world_map_points_xyz: Vec<PointXYZ> = downsampled_planar_world_map_points
        .iter()
        .map(|point| PointXYZ {
            x: point.x,
            y: point.y,
            z: point.z,
        })
        .collect();

    // 既存ファイル名は平面処理後の最終マップとして維持する。
    let planar_world_map_path =
        format!("{}/voxel-{}_world_map.pcd", save_dir, GLOBAL_MAP_VOXEL_SIZE);
    save_pcd_xyz(&planar_world_map_points_xyz, &planar_world_map_path)?;
    log::info!(
        "Saved planar world map: {} → {} points → {}",
        planar_world_map_points.len(),
        planar_world_map_points_xyz.len(),
        planar_world_map_path,
    );
    // --- Save the global voxel maps before and after final plane classification ---

    // --- Save per-frame ICP logs to JSON ---
    let frame_logs_path = format!("{}/frame_logs.json", save_dir);
    let frame_logs_json = serde_json::to_string_pretty(&frame_logs)?;
    std::fs::write(&frame_logs_path, &frame_logs_json)?;
    log::info!(
        "Saved frame logs: {} frames → {}",
        frame_logs.len(),
        frame_logs_path
    );
    // --- Save per-frame ICP logs to JSON ---

    Ok(())
}

#[inline]
fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

/// Mid-70座標の点をAiry-96座標へ写す外部変換 `T_airy96_from_mid70`。
fn make_airy96_from_mid70_extrinsic() -> Matrix4<f64> {
    let rotation = Matrix3::<f64>::new(0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let translation = Vector3::<f64>::new(
        MID70_ORIGIN_IN_AIRY96_X_M,
        MID70_ORIGIN_IN_AIRY96_Y_M,
        MID70_ORIGIN_IN_AIRY96_Z_M,
    );

    let mut transform = Matrix4::<f64>::identity();
    transform.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation);
    transform
        .fixed_view_mut::<3, 1>(0, 3)
        .copy_from(&translation);
    transform
}

/// Airy-96内蔵IMU座標からMid-70座標への回転を返す。
///
/// `R_mid70_from_imu = R_mid70_from_airy96 * R_airy96_from_imu`
fn make_imu_to_mid70_rotation() -> UnitQuaternion<f64> {
    let imu_to_airy96 = make_imu_to_airy96_rotation();
    let airy96_from_mid70 = make_airy96_from_mid70_extrinsic();
    let airy96_from_mid70_rotation =
        UnitQuaternion::from_matrix(&airy96_from_mid70.fixed_view::<3, 3>(0, 0).into_owned());

    airy96_from_mid70_rotation.inverse() * imu_to_airy96
}

/// Airy-96内蔵IMU座標からAiry-96 LiDAR座標への回転を返す。
fn make_imu_to_airy96_rotation() -> UnitQuaternion<f64> {
    UnitQuaternion::new_normalize(Quaternion::new(
        IMU_TO_AIRY96_QUAT_W,
        IMU_TO_AIRY96_QUAT_X,
        IMU_TO_AIRY96_QUAT_Y,
        IMU_TO_AIRY96_QUAT_Z,
    ))
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use nalgebra::{Point3, Vector3};

    use super::{
        CommandLineAction, IMU, LidarModel, MID70_ORIGIN_IN_AIRY96_Z_M,
        count_imu_samples_in_time_range, make_airy96_from_mid70_extrinsic,
        make_imu_to_airy96_rotation, make_imu_to_mid70_rotation, parse_command_line,
    };

    fn imu_sample(timestamp: f64) -> IMU {
        IMU {
            timestamp,
            angular_velocity: [0.0; 3],
            linear_acceleration: [0.0; 3],
        }
    }

    fn parse_args(args: &[&str]) -> anyhow::Result<CommandLineAction> {
        parse_command_line(args.iter().map(OsString::from))
    }

    #[test]
    fn command_line_defaults_to_mid70() {
        let CommandLineAction::Run(model) = parse_args(&[]).unwrap() else {
            panic!("expected run action");
        };
        assert_eq!(model, LidarModel::Mid70);
    }

    #[test]
    fn command_line_selects_airy96() {
        let CommandLineAction::Run(model) = parse_args(&["--lidar", "airy96"]).unwrap() else {
            panic!("expected run action");
        };
        assert_eq!(model, LidarModel::Airy96);

        let CommandLineAction::Run(model) = parse_args(&["--lidar=airy"]).unwrap() else {
            panic!("expected run action");
        };
        assert_eq!(model, LidarModel::Airy96);
    }

    #[test]
    fn command_line_rejects_unknown_lidar() {
        let error = match parse_args(&["--lidar", "unknown"]) {
            Ok(_) => panic!("unknown LiDAR model must fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("expected 'mid70' or 'airy96'"));
    }

    #[test]
    fn counts_imu_samples_inside_inclusive_point_cloud_interval() {
        let imu_data = [
            imu_sample(0.9),
            imu_sample(1.0),
            imu_sample(1.5),
            imu_sample(2.0),
            imu_sample(2.1),
        ];

        assert_eq!(count_imu_samples_in_time_range(&imu_data, 1.0, 2.0), 3);
        assert_eq!(count_imu_samples_in_time_range(&imu_data, 3.0, 4.0), 0);
        assert_eq!(count_imu_samples_in_time_range(&imu_data, 2.0, 1.0), 0);
    }

    #[test]
    fn mid70_extrinsic_maps_axes_into_airy96_coordinates() {
        let airy96_from_mid70 = make_airy96_from_mid70_extrinsic();
        let mid70_x = Point3::new(1.0, 0.0, 0.0);
        let airy96_point = airy96_from_mid70.transform_point(&mid70_x);

        assert!((airy96_point.x - 0.0).abs() < 1e-12);
        assert!((airy96_point.y + 1.0).abs() < 1e-12);
        assert!((airy96_point.z - MID70_ORIGIN_IN_AIRY96_Z_M).abs() < 1e-12);
    }

    #[test]
    fn imu_to_mid70_rotation_composes_back_to_airy96_rotation() {
        let imu_vector = Vector3::new(0.3, -0.4, 0.5);
        let imu_to_mid70 = make_imu_to_mid70_rotation();
        let airy96_from_mid70 = make_airy96_from_mid70_extrinsic();
        let airy96_from_mid70_rotation = airy96_from_mid70.fixed_view::<3, 3>(0, 0).into_owned();

        let via_mid70 = airy96_from_mid70_rotation * (imu_to_mid70 * imu_vector);
        let imu_to_airy96 = make_imu_to_airy96_rotation();
        let direct = imu_to_airy96 * imu_vector;

        assert!((via_mid70 - direct).norm() < 1e-12);
    }
}
