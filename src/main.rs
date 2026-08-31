use anyhow::{Result, bail};
use nalgebra::{Matrix3, Matrix4, Point3, Quaternion, UnitQuaternion, Vector3};
use re_lidar_slam::{
    deskew_points::deskew_points,
    file_handler::{load_imu_data, load_pcd_files, load_pcd_xyzit, save_pcd_xyz},
    find_nearest_points::pickup_valid_source_points,
    icp::{
        Vector6f, apply_delta, build_robust_point_to_plane_system, compute_rmse,
        compute_robust_cost, solve_icp_delta_observable,
    },
    predict_pose_by_imu::{align_imu_timestamps, build_rotation_trajectory, predict_pose_by_imu},
    types::{CurrentFrameInfo, FrameLog, FrameTiming, IMU, PointXYZ, SLAMMap},
    voxel_map::{
        LOCALMap, LocalMapConfig, SurfaceFilterConfig, SurfaceStatus, WorldMapUpdateFilterConfig,
    },
    voxelization::{voxel_downsample_points, voxel_downsample_points_and_maps_dual},
};
use std::{
    ffi::OsString,
    time::{Duration, Instant},
};

// const DATASET_DIR: &str = "/mnt/nas/share/avia/08232026/01";
const DATASET_DIR: &str = "/mnt/nas/share/airy96/06212026/park05";
const SAVE_ROOT_DIR: &str = "data/output/debug/08292026";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LidarModel {
    Mid70,
    Airy96,
    Avia,
}

impl LidarModel {
    fn parse(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "mid70" | "mid-70" => Ok(Self::Mid70),
            "airy96" | "airy-96" | "airy" => Ok(Self::Airy96),
            "avia" | "livox-avia" => Ok(Self::Avia),
            _ => bail!("unsupported LiDAR model '{value}'; expected 'mid70', 'airy96', or 'avia'"),
        }
    }

    fn input_subdir(self) -> &'static str {
        match self {
            Self::Mid70 => "mid-70",
            Self::Airy96 => "",
            Self::Avia => "avia",
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Mid70 => "mid70",
            Self::Airy96 => "airy96",
            Self::Avia => "avia",
        }
    }

    fn imu_to_lidar_rotation(self) -> UnitQuaternion<f64> {
        match self {
            Self::Mid70 => make_imu_to_mid70_rotation(),
            Self::Airy96 => make_imu_to_airy96_rotation(),
            Self::Avia => make_imu_to_avia_rotation(),
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
                    .ok_or_else(|| {
                        anyhow::anyhow!("--lidar requires 'mid70', 'airy96', or 'avia'")
                    })?
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
    println!("Usage: re_lidar_slam [--lidar <mid70|airy96|avia>]");
    println!("  --lidar  Select the point-cloud sensor (default: mid70)");
}

// Mid-70 sparse-cloud preset.
// The upper range matches the range used by the existing Mid-70 datasets.
const MIN_DIST: f32 = 0.5;
const MAX_DIST: f32 = 200.0;

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
// このセンサー構成では、走行全体で 1 フレームあたり約 1800 個の GlobalMap
// voxel が増える。入力フレーム数から最終容量を先に確保し、実行途中の巨大な
// HashMap 再配置（数百 ms のスパイク）を避ける。
const GLOBAL_MAP_EXPECTED_NEW_VOXELS_PER_FRAME: usize = 2048;

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
const ICP_HUBER_DELTA_M: f32 = 0.08;
const ICP_MIN_CORRESPONDENCES: usize = 100;
const ICP_MIN_CORRESPONDENCE_RATIO: f32 = 0.05;
const ICP_MIN_OBSERVABLE_RANK: usize = 3;
const ICP_RELATIVE_EIGENVALUE_THRESHOLD: f32 = 0.02;
const ICP_DAMPING: f32 = 1e-6;
const ICP_MAX_FINAL_RMSE_M: f32 = 0.15;
// Limits apply to the ICP correction relative to the IMU prediction, not to
// the vehicle's total frame-to-frame motion.
const ICP_MAX_TRANSLATION_CORRECTION_M: f32 = 0.10;
const ICP_MAX_ROTATION_CORRECTION_DEG: f32 = 2.0;
const ICP_RMSE_CHANGE_THRESHOLD_M: f32 = 1e-4;
const ICP_TRANSLATION_DELTA_THRESHOLD_M: f32 = 0.001;
const ICP_ROTATION_DELTA_THRESHOLD_DEG: f32 = 0.01;
const LOCAL_SOURCE_TO_PLANE_MAX_DISTANCE_M: f32 = 0.15;
const LOCAL_MIN_PLANARITY: f32 = 0.10;
// If normal matching fails after a large IMU Z drift, retry once with a wider
// neighborhood and looser source-to-plane gates. Only the world-Z component of
// this coarse solve is applied; normal ICP must then validate and refine it.
const ICP_VERTICAL_RECOVERY_SEARCH_RANGE: i32 = 6;
const ICP_VERTICAL_RECOVERY_MAX_DIST_FACTOR: f32 = 6.0;
const ICP_VERTICAL_RECOVERY_SOURCE_PLANE_SCORE_THRESHOLD: f32 = 0.0;
const ICP_VERTICAL_RECOVERY_SOURCE_TO_PLANE_MAX_DISTANCE_M: f32 = 0.60;
const ICP_VERTICAL_RECOVERY_MAX_TRANSLATION_CORRECTION_M: f32 = 0.60;
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
    let expected_global_voxel_capacity = pcd_files
        .len()
        .saturating_mul(GLOBAL_MAP_EXPECTED_NEW_VOXELS_PER_FRAME);
    let mut slam_map = SLAMMap {
        global_voxel_map: LOCALMap::with_voxel_capacity(
            global_map_config,
            expected_global_voxel_capacity,
        ),
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
        let (local_source, global_source) = voxel_downsample_points_and_maps_dual(
            &deskewed_points,
            DOWNSAMPLE_VOXEL_SIZE,
            GLOBAL_MAP_VOXEL_SIZE,
        );
        let voxel_end = voxel_start.elapsed();
        let downsampled_source_points_for_local = local_source.points;
        let downsampled_source_points_for_global = global_source.points;
        log::debug!(
            "Frame {i}: Downsampled {} points → {} points in {:.2?}",
            deskewed_points.len(),
            downsampled_source_points_for_local.len(),
            voxel_end
        );
        // --- Downsample deskewed points ---

        // The source voxel maps were populated while emitting the centroids.
        let source_voxel_map = local_source.voxel_map;
        let source_voxel_map_for_global = global_source.voxel_map;
        let build_map_end = Duration::ZERO;

        // --- ICP (Point to Plane) ---
        // IMU 予測姿勢を初期値として (R, t) を取り出す
        let pred_pose = pose_prediction.0.cast::<f32>();
        let pred_r: Matrix3<f32> = pred_pose.fixed_view::<3, 3>(0, 0).into();
        let pred_t: Vector3<f32> = pred_pose.fixed_view::<3, 1>(0, 3).into();
        let mut r_mat = pred_r;
        let mut t_vec = pred_t;

        let local_map_was_empty = slam_map.local_voxel_map.voxel_map.is_empty();
        let source_point_count = source_voxel_map.len();
        let min_correspondences = ICP_MIN_CORRESPONDENCES
            .max((source_point_count as f32 * ICP_MIN_CORRESPONDENCE_RATIO).ceil() as usize);
        let mut previous_rmse: Option<f32> = None;
        let mut final_rmse: Option<f32> = None;
        let mut final_correspondence_count = 0usize;
        let mut final_correspondence_ratio = 0.0f32;
        let mut final_observable_rank = 0usize;
        let mut final_eigenvalue_ratio: Option<f32> = None;
        let mut icp_ok = false; // ICP が有効な解を得られたか
        let mut vertical_recovery_attempted = false;
        let mut use_vertical_recovery_correspondences = false;
        let mut vertical_recovery_pose = false;
        let mut icp_correspondence_search_time = Duration::ZERO;
        let mut icp_linear_system_time = Duration::ZERO;
        let mut icp_solver_time = Duration::ZERO;

        let loop_start = Instant::now();

        if local_map_was_empty {
            log::debug!("Frame {i}: local map empty, skipping ICP");
        } else {
            for _iter in 0..ICP_ITERATIONS {
                // 対応点をピックアップ
                // - source はローカル座標、target (local_voxel_map) はワールド座標
                // - 現在の (R,t) 推定値で source をワールド変換してから近傍探索
                let recovery_iteration = use_vertical_recovery_correspondences;
                let (
                    search_range,
                    max_dist_factor,
                    source_plane_score_threshold,
                    source_to_plane_max_distance_m,
                ) = if recovery_iteration {
                    (
                        ICP_VERTICAL_RECOVERY_SEARCH_RANGE,
                        ICP_VERTICAL_RECOVERY_MAX_DIST_FACTOR,
                        ICP_VERTICAL_RECOVERY_SOURCE_PLANE_SCORE_THRESHOLD,
                        ICP_VERTICAL_RECOVERY_SOURCE_TO_PLANE_MAX_DISTANCE_M,
                    )
                } else {
                    (
                        SEARCH_RANGE,
                        MAX_DIST_FACTOR,
                        LOCAL_SOURCE_PLANE_SCORE_THRESHOLD,
                        LOCAL_SOURCE_TO_PLANE_MAX_DISTANCE_M,
                    )
                };
                let pickup_start = Instant::now();
                let correspondences = pickup_valid_source_points::<LOCAL_KNN_K>(
                    &source_voxel_map,
                    &slam_map.local_voxel_map.voxel_map,
                    slam_map.local_voxel_map.config.index_voxel_size,
                    search_range,
                    max_dist_factor,
                    LOCAL_PLANE_POINT_DISTANCE_THRESHOLD_M,
                    source_plane_score_threshold,
                    Some(source_to_plane_max_distance_m),
                    Some(LOCAL_MIN_PLANARITY),
                    &r_mat,
                    &t_vec,
                );
                let correspondence_ratio = if source_point_count > 0 {
                    correspondences.len() as f32 / source_point_count as f32
                } else {
                    0.0
                };
                let pickup_end = pickup_start.elapsed();
                icp_correspondence_search_time += pickup_end;
                log::debug!(
                    "ICP iter {_iter}{}: Picked up {} correspondences ({:.1}%) in {:.2?}",
                    if recovery_iteration {
                        " [vertical recovery]"
                    } else {
                        ""
                    },
                    correspondences.len(),
                    correspondence_ratio * 100.0,
                    pickup_end
                );

                // A coarse Z estimate only needs a stable planar subset. The
                // recovered pose is still required to pass the normal 5% gate.
                let required_correspondences = if recovery_iteration {
                    ICP_MIN_CORRESPONDENCES
                } else {
                    min_correspondences
                };
                if correspondences.len() < required_correspondences {
                    log::warn!(
                        "ICP iter {_iter}: insufficient correspondences: {} < {}",
                        correspondences.len(),
                        required_correspondences,
                    );
                    if !vertical_recovery_attempted {
                        vertical_recovery_attempted = true;
                        use_vertical_recovery_correspondences = true;
                        log::warn!(
                            "ICP iter {_iter}: retrying with world-Z-only recovery matching"
                        );
                        continue;
                    }
                    break;
                }

                // 線形システム構築
                let system_start = Instant::now();
                let system = build_robust_point_to_plane_system(
                    &correspondences,
                    &r_mat,
                    &t_vec,
                    ICP_HUBER_DELTA_M,
                );
                let system_end = system_start.elapsed();
                icp_linear_system_time += system_end;

                let solve_start = Instant::now();
                let solve_result = solve_icp_delta_observable(
                    &system,
                    ICP_DAMPING,
                    ICP_RELATIVE_EIGENVALUE_THRESHOLD,
                );
                let solve_end = solve_start.elapsed();
                icp_solver_time += solve_end;
                let Some(solve_result) = solve_result else {
                    log::warn!("ICP iter {_iter}: observable solve failed");
                    break;
                };
                if solve_result.observable_rank < ICP_MIN_OBSERVABLE_RANK {
                    log::warn!(
                        "ICP iter {_iter}: degenerate geometry (observable rank {} < {})",
                        solve_result.observable_rank,
                        ICP_MIN_OBSERVABLE_RANK,
                    );
                    break;
                }

                let applied_delta = if recovery_iteration {
                    retain_world_z_translation(&solve_result.delta)
                } else {
                    solve_result.delta
                };

                let delta_translation_m = applied_delta.fixed_rows::<3>(3).norm();
                let delta_rotation_deg = applied_delta.fixed_rows::<3>(0).norm().to_degrees();
                let (candidate_r, candidate_t) = apply_delta(&r_mat, &t_vec, &applied_delta);
                let (total_translation_correction_m, total_rotation_correction_deg) =
                    pose_correction_magnitudes(&pred_r, &pred_t, &candidate_r, &candidate_t);
                let max_translation_correction_m = if recovery_iteration || vertical_recovery_pose {
                    ICP_VERTICAL_RECOVERY_MAX_TRANSLATION_CORRECTION_M
                } else {
                    ICP_MAX_TRANSLATION_CORRECTION_M
                };

                if total_translation_correction_m > max_translation_correction_m
                    || total_rotation_correction_deg > ICP_MAX_ROTATION_CORRECTION_DEG
                {
                    log::warn!(
                        "ICP iter {_iter}: correction exceeds prediction gate: \
                         translation={total_translation_correction_m:.4} m, \
                         rotation={total_rotation_correction_deg:.3} deg",
                    );
                    break;
                }

                let candidate_cost = compute_robust_cost(
                    &correspondences,
                    &candidate_r,
                    &candidate_t,
                    ICP_HUBER_DELTA_M,
                );
                if !candidate_cost.is_finite() || candidate_cost > system.cost * (1.0 + 1e-5) + 1e-8
                {
                    log::warn!(
                        "ICP iter {_iter}: robust cost did not improve ({:.6} -> {:.6})",
                        system.cost,
                        candidate_cost,
                    );
                    break;
                }

                r_mat = candidate_r;
                t_vec = candidate_t;
                icp_ok = true;
                if recovery_iteration {
                    vertical_recovery_pose = true;
                    use_vertical_recovery_correspondences = false;
                    previous_rmse = None;
                    log::warn!(
                        "ICP iter {_iter}: applied coarse world-Z recovery dz={:.4} m; \
                         returning to normal matching for validation",
                        applied_delta[5],
                    );
                }
                log::debug!(
                    "ICP iter {_iter}: point-to-plane system={:.2?}, solve={:.2?}",
                    system_end,
                    solve_end,
                );

                // RMSE を計算して収束チェック
                let rmse = compute_rmse(&correspondences, &r_mat, &t_vec);
                final_rmse = Some(rmse);
                final_correspondence_count = correspondences.len();
                final_correspondence_ratio = correspondence_ratio;
                final_observable_rank = solve_result.observable_rank;
                final_eigenvalue_ratio = Some(solve_result.min_observable_eigenvalue_ratio);
                log::debug!(
                    "ICP iter {_iter}: used={}, robust_cost={:.6}->{:.6}, rmse={:.6}, \
                     rank={}, min_eigen_ratio={:.3e}, delta_t={:.4} m, delta_r={:.3} deg",
                    system.used_count,
                    system.cost,
                    candidate_cost,
                    rmse,
                    solve_result.observable_rank,
                    solve_result.min_observable_eigenvalue_ratio,
                    delta_translation_m,
                    delta_rotation_deg,
                );

                let rmse_converged = previous_rmse
                    .is_some_and(|previous| (previous - rmse).abs() < ICP_RMSE_CHANGE_THRESHOLD_M);
                previous_rmse = Some(rmse);
                if recovery_iteration {
                    continue;
                }
                if rmse_converged
                    && delta_translation_m < ICP_TRANSLATION_DELTA_THRESHOLD_M
                    && delta_rotation_deg < ICP_ROTATION_DELTA_THRESHOLD_DEG
                {
                    log::debug!(
                        "ICP converged at iter {_iter}: delta_t={delta_translation_m:.3e} m, \
                         delta_r={delta_rotation_deg:.3e} deg",
                    );
                    break;
                }
            }

            // Rebuild correspondences at the accepted pose so stale matches cannot
            // make a bad final pose look valid.
            if icp_ok {
                let final_pickup_start = Instant::now();
                let correspondences = pickup_valid_source_points::<LOCAL_KNN_K>(
                    &source_voxel_map,
                    &slam_map.local_voxel_map.voxel_map,
                    slam_map.local_voxel_map.config.index_voxel_size,
                    SEARCH_RANGE,
                    MAX_DIST_FACTOR,
                    LOCAL_PLANE_POINT_DISTANCE_THRESHOLD_M,
                    LOCAL_SOURCE_PLANE_SCORE_THRESHOLD,
                    Some(LOCAL_SOURCE_TO_PLANE_MAX_DISTANCE_M),
                    Some(LOCAL_MIN_PLANARITY),
                    &r_mat,
                    &t_vec,
                );
                icp_correspondence_search_time += final_pickup_start.elapsed();
                final_correspondence_count = correspondences.len();
                final_correspondence_ratio = if source_point_count > 0 {
                    correspondences.len() as f32 / source_point_count as f32
                } else {
                    0.0
                };
                final_rmse = Some(compute_rmse(&correspondences, &r_mat, &t_vec));

                let final_system_start = Instant::now();
                let final_system = build_robust_point_to_plane_system(
                    &correspondences,
                    &r_mat,
                    &t_vec,
                    ICP_HUBER_DELTA_M,
                );
                icp_linear_system_time += final_system_start.elapsed();
                let final_solve_start = Instant::now();
                let final_solve = solve_icp_delta_observable(
                    &final_system,
                    ICP_DAMPING,
                    ICP_RELATIVE_EIGENVALUE_THRESHOLD,
                );
                icp_solver_time += final_solve_start.elapsed();
                final_observable_rank = 0;
                final_eigenvalue_ratio = None;
                if let Some(result) = &final_solve {
                    final_observable_rank = result.observable_rank;
                    final_eigenvalue_ratio = Some(result.min_observable_eigenvalue_ratio);
                }

                let final_quality_ok = final_correspondence_count >= min_correspondences
                    && final_observable_rank >= ICP_MIN_OBSERVABLE_RANK
                    && final_rmse
                        .is_some_and(|rmse| rmse.is_finite() && rmse <= ICP_MAX_FINAL_RMSE_M);
                if !final_quality_ok {
                    log::warn!(
                        "Frame {i}: final ICP quality rejected: correspondences={} (min {}), \
                         rank={}, rmse={:?}",
                        final_correspondence_count,
                        min_correspondences,
                        final_observable_rank,
                        final_rmse,
                    );
                    icp_ok = false;
                }
            }

            if !icp_ok {
                log::warn!("Frame {i}: ICP failed, using IMU prediction");
                r_mat = pred_r;
                t_vec = pred_t;
            }
        }
        let loop_end = loop_start.elapsed();
        log::debug!(
            "Frame {i}: ICP loop finished in {:.2?}, final RMSE={:?}",
            loop_end,
            final_rmse,
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
        // Rejected/degenerate ICP must not feed a position jump back into the
        // next IMU prediction. Keep the velocity predicted by the IMU instead.
        let new_velocity = if icp_ok {
            raw_velocity.cap_magnitude(2.0)
        } else {
            pose_prediction.1
        };

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
        let (icp_translation_correction_m, icp_rotation_correction_deg) = if icp_ok {
            pose_correction_magnitudes(&pred_r, &pred_t, &r_mat, &t_vec)
        } else {
            (0.0, 0.0)
        };
        let map_update_allowed = local_map_was_empty || icp_ok;
        // --- Record frame log ---
        let pose_update_time = pose_update_start.elapsed();

        // --- Filter valid source points, then update the WorldMap ---
        // ローカルマップが空（初回フレーム）の場合はフィルタなしで全点追加。
        // それ以外は pickup_valid_source_points で平面に乗っている点だけ抽出し、
        // ICP 収束後の最終姿勢 (r_mat, t_vec) でワールドマップに追加する。
        let global_filter_start = Instant::now();
        let global_source_points: Vec<Point3<f32>> = if !map_update_allowed {
            log::warn!("Frame {i}: map update skipped because ICP quality is insufficient");
            Vec::new()
        } else if local_map_was_empty {
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
        let global_update_stats = if map_update_allowed {
            slam_map.global_voxel_map.update_world_map_filtered(
                &global_source_points,
                &current_frame_info.current_global_pose,
                &world_map_update_filter_config,
            )
        } else {
            Default::default()
        };
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
        if map_update_allowed {
            slam_map.local_voxel_map.update_with_new_frame(
                &downsampled_source_points_for_local,
                &current_frame_info.current_global_pose,
            );
        }
        let local_map_update_time = local_map_update_start.elapsed();
        // --- Update the LocalMap with the new frame's points ---

        prev_frame_start_time = current_frame_start_time;

        let frame_processing_time = frame_processing_start.elapsed();
        let frame_total_with_file_io = frame_start.elapsed();
        let measured_icp_detail =
            icp_correspondence_search_time + icp_linear_system_time + icp_solver_time;
        let icp_other_time = loop_end.saturating_sub(measured_icp_detail);
        let timing = FrameTiming {
            load_pcd_ms: duration_ms(load_pcd_time),
            timestamps_ms: duration_ms(timestamp_time),
            imu_predict_ms: duration_ms(predict_pose_time),
            rotation_trajectory_ms: duration_ms(rotation_trajectory_time),
            deskew_ms: duration_ms(deskew_time),
            downsample_ms: duration_ms(voxel_end),
            build_source_maps_ms: duration_ms(build_map_end),
            icp_ms: duration_ms(loop_end),
            pose_update_ms: duration_ms(pose_update_time),
            global_filter_ms: duration_ms(global_filter_time),
            global_map_update_ms: duration_ms(global_map_update_time),
            delayed_surface_ms: duration_ms(delayed_surface_time),
            local_map_update_ms: duration_ms(local_map_update_time),
            processing_total_ms: duration_ms(frame_processing_time),
            total_with_file_io_ms: duration_ms(frame_total_with_file_io),
            icp_correspondence_search_ms: duration_ms(icp_correspondence_search_time),
            icp_linear_system_ms: duration_ms(icp_linear_system_time),
            icp_solver_ms: duration_ms(icp_solver_time),
            icp_other_ms: duration_ms(icp_other_time),
        };
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
        log::debug!(
            "Frame {i} ICP breakdown [ms]: correspondence_search={:.3}, \
             linear_system={:.3}, solver={:.3}, other={:.3}, total={:.3}",
            timing.icp_correspondence_search_ms,
            timing.icp_linear_system_ms,
            timing.icp_solver_ms,
            timing.icp_other_ms,
            timing.icp_ms,
        );
        frame_logs.push(FrameLog {
            frame_index: i,
            timestamp: current_frame_start_time,
            icp_ok,
            rmse: final_rmse.filter(|rmse| rmse.is_finite()),
            correspondence_count: final_correspondence_count,
            correspondence_ratio: final_correspondence_ratio,
            observable_rank: final_observable_rank,
            min_observable_eigenvalue_ratio: final_eigenvalue_ratio,
            icp_translation_correction_m,
            icp_rotation_correction_deg,
            map_updated: map_update_allowed,
            translation_m,
            rotation_deg,
            velocity_m_s: new_velocity.norm(),
            pose_x: new_pos.x,
            pose_y: new_pos.y,
            pose_z: new_pos.z,
            timing,
        });
    }

    log_timing_summary(&frame_logs);

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

#[derive(Debug)]
struct TimingStats {
    name: &'static str,
    mean_ms: f64,
    p95_ms: f64,
    max_ms: f64,
}

fn log_timing_summary(frame_logs: &[FrameLog]) {
    if frame_logs.is_empty() {
        return;
    }

    let stages: [(&str, fn(&FrameTiming) -> f64); 13] = [
        ("load_pcd", |t| t.load_pcd_ms),
        ("timestamps", |t| t.timestamps_ms),
        ("imu_predict", |t| t.imu_predict_ms),
        ("rotation_trajectory", |t| t.rotation_trajectory_ms),
        ("deskew", |t| t.deskew_ms),
        ("downsample", |t| t.downsample_ms),
        ("build_source_maps", |t| t.build_source_maps_ms),
        ("icp", |t| t.icp_ms),
        ("pose_update", |t| t.pose_update_ms),
        ("global_filter", |t| t.global_filter_ms),
        ("global_map_update", |t| t.global_map_update_ms),
        ("delayed_surface", |t| t.delayed_surface_ms),
        ("local_map_update", |t| t.local_map_update_ms),
    ];
    let mut summaries: Vec<TimingStats> = stages
        .into_iter()
        .map(|(name, get)| timing_stats(name, frame_logs, get))
        .collect();
    summaries.sort_by(|left, right| right.mean_ms.total_cmp(&left.mean_ms));

    let processing_total = timing_stats("processing_total", frame_logs, |t| t.processing_total_ms);
    let wall_total = timing_stats("total_with_file_io", frame_logs, |t| {
        t.total_with_file_io_ms
    });
    log::info!(
        "Timing summary for {} frames [ms]: processing mean={:.3}, p95={:.3}, max={:.3}; \
         with_file_io mean={:.3}, p95={:.3}, max={:.3}",
        frame_logs.len(),
        processing_total.mean_ms,
        processing_total.p95_ms,
        processing_total.max_ms,
        wall_total.mean_ms,
        wall_total.p95_ms,
        wall_total.max_ms,
    );
    for (rank, summary) in summaries.iter().enumerate() {
        let share = if wall_total.mean_ms > 0.0 {
            summary.mean_ms / wall_total.mean_ms * 100.0
        } else {
            0.0
        };
        log::info!(
            "Timing rank {:>2}: {:<20} mean={:>9.3} ms ({:>5.1}%), p95={:>9.3} ms, max={:>9.3} ms",
            rank + 1,
            summary.name,
            summary.mean_ms,
            share,
            summary.p95_ms,
            summary.max_ms,
        );
    }

    let icp_details: [(&str, fn(&FrameTiming) -> f64); 4] = [
        ("correspondence_search", |t| t.icp_correspondence_search_ms),
        ("linear_system", |t| t.icp_linear_system_ms),
        ("solver", |t| t.icp_solver_ms),
        ("other", |t| t.icp_other_ms),
    ];
    let mut icp_summaries: Vec<TimingStats> = icp_details
        .into_iter()
        .map(|(name, get)| timing_stats(name, frame_logs, get))
        .collect();
    icp_summaries.sort_by(|left, right| right.mean_ms.total_cmp(&left.mean_ms));
    for summary in icp_summaries {
        let share = if summaries
            .iter()
            .find(|summary| summary.name == "icp")
            .is_some_and(|summary| summary.mean_ms > 0.0)
        {
            let icp_mean = summaries
                .iter()
                .find(|summary| summary.name == "icp")
                .map_or(0.0, |summary| summary.mean_ms);
            summary.mean_ms / icp_mean * 100.0
        } else {
            0.0
        };
        log::info!(
            "ICP timing: {:<21} mean={:>9.3} ms ({:>5.1}% of ICP), p95={:>9.3} ms, max={:>9.3} ms",
            summary.name,
            summary.mean_ms,
            share,
            summary.p95_ms,
            summary.max_ms,
        );
    }
}

fn timing_stats(
    name: &'static str,
    frame_logs: &[FrameLog],
    get: fn(&FrameTiming) -> f64,
) -> TimingStats {
    let mut values: Vec<f64> = frame_logs
        .iter()
        .map(|frame| get(&frame.timing))
        .filter(|value| value.is_finite())
        .collect();
    values.sort_by(f64::total_cmp);
    let mean_ms = values.iter().sum::<f64>() / values.len().max(1) as f64;
    let p95_index = ((values.len() as f64 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(values.len().saturating_sub(1));

    TimingStats {
        name,
        mean_ms,
        p95_ms: values.get(p95_index).copied().unwrap_or_default(),
        max_ms: values.last().copied().unwrap_or_default(),
    }
}

fn pose_correction_magnitudes(
    reference_r: &Matrix3<f32>,
    reference_t: &Vector3<f32>,
    candidate_r: &Matrix3<f32>,
    candidate_t: &Vector3<f32>,
) -> (f32, f32) {
    let translation_m = (candidate_t - reference_t).norm();
    let delta_r = candidate_r * reference_r.transpose();
    let rotation_deg = ((delta_r.trace() - 1.0) * 0.5)
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees();
    (translation_m, rotation_deg)
}

fn retain_world_z_translation(delta: &Vector6f) -> Vector6f {
    let mut vertical_delta = Vector6f::zeros();
    vertical_delta[5] = delta[5];
    vertical_delta
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

/// Livox Avia内蔵IMU座標からLiDAR座標への回転を返す。
/// Aviaの内蔵IMUとLiDARの座標軸は同じ向きなので、回転は単位回転になる。
fn make_imu_to_avia_rotation() -> UnitQuaternion<f64> {
    UnitQuaternion::identity()
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use nalgebra::{Point3, Vector3};

    use super::{
        CommandLineAction, IMU, LidarModel, MID70_ORIGIN_IN_AIRY96_Z_M,
        count_imu_samples_in_time_range, make_airy96_from_mid70_extrinsic,
        make_imu_to_airy96_rotation, make_imu_to_avia_rotation, make_imu_to_mid70_rotation,
        parse_command_line, retain_world_z_translation,
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
    fn vertical_recovery_discards_rotation_and_xy_translation() {
        let delta = re_lidar_slam::icp::Vector6f::new(1.0, 2.0, 3.0, 4.0, 5.0, -0.45);

        assert_eq!(
            retain_world_z_translation(&delta),
            re_lidar_slam::icp::Vector6f::new(0.0, 0.0, 0.0, 0.0, 0.0, -0.45)
        );
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
    fn command_line_selects_avia() {
        let CommandLineAction::Run(model) = parse_args(&["--lidar", "avia"]).unwrap() else {
            panic!("expected run action");
        };
        assert_eq!(model, LidarModel::Avia);
        assert_eq!(model.input_subdir(), "avia");
        assert_eq!(model.name(), "avia");

        let CommandLineAction::Run(model) = parse_args(&["--lidar=livox-avia"]).unwrap() else {
            panic!("expected run action");
        };
        assert_eq!(model, LidarModel::Avia);
    }

    #[test]
    fn command_line_rejects_unknown_lidar() {
        let error = match parse_args(&["--lidar", "unknown"]) {
            Ok(_) => panic!("unknown LiDAR model must fail"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("expected 'mid70', 'airy96', or 'avia'")
        );
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

    #[test]
    fn imu_to_avia_rotation_preserves_axes() {
        let imu_vector = Vector3::new(0.3, -0.4, 0.5);
        assert!((make_imu_to_avia_rotation() * imu_vector - imu_vector).norm() < 1e-12);
    }
}
