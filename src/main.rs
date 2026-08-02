use std::{fmt, str::FromStr};

use anyhow::{Context, Result, bail, ensure};
use nalgebra::{Matrix3, Matrix4, Point3, Quaternion, UnitQuaternion, Vector3};
use re_lidar_slam::{
    deskew_points::{deskew_points, filter_points_by_distance},
    file_handler::{load_imu_data, load_pcd_files, load_pcd_xyzit, save_pcd_xyz},
    find_nearest_points::pickup_valid_source_points,
    icp::{apply_delta, build_point_to_plane_system, compute_rmse, solve_icp_delta},
    predict_pose_by_imu::{
        align_imu_timestamps, build_rotation_trajectory, predict_pose_by_constant_velocity,
        predict_pose_by_imu,
    },
    types::{CurrentFrameInfo, FrameLog, IMU, PointXYZ, SLAMMap},
    voxel_map::{LOCALMap, LocalMapConfig, build_voxel_map},
    voxelization::voxel_downsample_points,
};

/*
cargo run --release --bin re_lidar_slam -- \
  --sensor mid70 \
  --load-dir data/input/07262026/pcd/mid-70/test02 \
  --save-dir data/output/debug/07262026
 */

const LOAD_DIR_AIRY96: &str = "data/output/debug/checked_pcd/airy"; // /home/kenji/mnt/nfs/share/airy96/06212026/park05
const LOAD_DIR_AIRY96_IMU: &str = "data/input/08012026/08012026-airy96-mid70-07-church/airy/imu";
const LOAD_DIR_MID70: &str = "data/output/debug/checked_pcd/mid-70"; // /home/kenji/mnt/nfs/share/airy96/06212026/park05
const SAVE_DIR: &str = "data/output/debug/08012026";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SensorType {
    Airy96,
    Mid70,
}

impl FromStr for SensorType {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "airy96" | "airy-96" => Ok(Self::Airy96),
            "mid70" | "mid-70" => Ok(Self::Mid70),
            _ => bail!("Unsupported sensor '{value}'; expected airy96 or mid70"),
        }
    }
}

impl fmt::Display for SensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Airy96 => write!(f, "airy96"),
            Self::Mid70 => write!(f, "mid70"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AppConfig {
    sensor_type: SensorType,
    load_dir: String,
    save_dir: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            sensor_type: SensorType::Airy96,
            load_dir: LOAD_DIR_AIRY96.to_owned(),
            save_dir: SAVE_DIR.to_owned(),
        }
    }
}

impl AppConfig {
    fn from_args() -> Result<Self> {
        Self::parse(std::env::args().skip(1))
    }

    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self> {
        let mut config = Self::default();
        let mut args = args.into_iter();

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--sensor" => {
                    let value = args.next().context("--sensor requires a value")?;
                    config.sensor_type = value.parse()?;
                }
                "--load-dir" => {
                    config.load_dir = args.next().context("--load-dir requires a value")?;
                }
                "--save-dir" => {
                    config.save_dir = args.next().context("--save-dir requires a value")?;
                }
                _ => bail!("Unknown argument '{arg}'; use --help to see supported arguments"),
            }
        }

        Ok(config)
    }
}

struct ImuContext {
    samples: Vec<IMU>,
    imu_to_lidar: UnitQuaternion<f64>,
    imu_to_mid70: UnitQuaternion<f64>,
}

const MID70_ORIGIN_IN_AIRY_X_M: f64 = 0.0;
const MID70_ORIGIN_IN_AIRY_Y_M: f64 = 0.0;

// Airy原点がMID-70原点より上にあるため負値。
// 約30.3 mmは図面からの概算。実測値に置き換えるのが望ましい。
const MID70_ORIGIN_IN_AIRY_Z_M: f64 = -0.06;

// --- The parametrers for Airy 96 ---
const MIN_DIST_AIRY96: f32 = 0.5;
const MAX_DIST_AIRY96: f32 = 100.0;

// IMU coordination to LiDAR coordination (Robosense 96 beam)
// Quaternion (x, y, z, w): -0.705437, 0.708767, -0.00246579, 0.00097028
// Translation (x, y, z)  : 0.00425, 0.00418, -0.00446  [m]
const IMU_TO_LIDAR_QUAT_X: f64 = -0.705437;
const IMU_TO_LIDAR_QUAT_Y: f64 = 0.708767;
const IMU_TO_LIDAR_QUAT_Z: f64 = -0.00246579;
const IMU_TO_LIDAR_QUAT_W: f64 = 0.00097028;

const LOCAL_MAP_VOXEL_SIZE_AIRY96: f32 = 0.5; // m
const GLOBAL_MAP_VOXEL_SIZE_AIRY96: f32 = 0.25; // m

const NEIGHBOR_RANGE_AIRY96: i32 = 2; // Voxel search range for nearest neighbor search

const LOCAL_KNN_K_AIRY96: usize = 7;
const GLOBAL_KNN_K_AIRY96: usize = 5; // Number of nearest neighbors for plane fitting (Default: 5)
const SEARCH_RANGE_AIRY96: i32 = 2; // Voxel search range for nearest neighbor search
const MAX_DIST_FACTOR_AIRY96: f32 = 2.5; // Maximum distance factor for nearest neighbor search (Prev: 3.0)
// k近傍点が推定平面から離れてよい最大距離 [m]
const LOCAL_PLANE_POINT_DISTANCE_THRESHOLD_M_AIRY96: f32 = 0.1;
const GLOBAL_PLANE_POINT_DISTANCE_THRESHOLD_M_AIRY96: f32 = 0.1;
const LOCAL_SOURCE_PLANE_SCORE_THRESHOLD_AIRY96: f32 = 0.90;
const GLOBAL_SOURCE_PLANE_SCORE_THRESHOLD_AIRY96: f32 = 0.90;
// GlobalMapへ追加するSource点と既存平面との最大距離 [m]
const GLOBAL_SOURCE_TO_PLANE_MAX_DISTANCE_M_AIRY96: f32 = 0.015;
const GLOBAL_MIN_PLANARITY_AIRY96: f32 = 0.15;

const ICP_ITERATIONS_AIRY96: usize = 5; // Default: 5
const ICP_RMSE_THRESHOLD_AIRY96: f32 = 0.033; // 収束判定: RMSE の変化量がこれ以下なら停止 // voxel size 0.2m の場合、0.07m くらいが妥当
const ICP_RMSE_DIVERGE_THRESHOLD_AIRY96: f32 = 2.0; // 発散判定: RMSE がこれ以上なら結果棄却→予測姿勢にフォールバック

const MAX_DIST_FOR_VOXEL_MAP_AIRY96: f32 = 40.0;
// --- The parametrers for Airy 96 ---

// --- The parametrers for Mid-70 ---
const GLOBAL_MAP_VOXEL_SIZE_MID70: f32 = 0.15; // m
const GLOBAL_MAP_MIN_OBSERVED_FRAMES_MID70: usize = 2;

// Mid-70 は点分布が疎なため、平面検索用LocalMapを粗い1 mグリッドにまとめ、
// 最大3 mの範囲から5近傍を探索する。
const PLANE_FILTER_LOCAL_DOWNSAMPLE_SIZE_MID70: f32 = 0.5; // m
const PLANE_FILTER_LOCAL_MAP_VOXEL_SIZE_MID70: f32 = 1.0; // m
const PLANE_FILTER_SEARCH_RANGE_MID70: i32 = 3;
const PLANE_FILTER_K_MID70: usize = 5;
const PLANE_FILTER_MAX_DIST_FACTOR_MID70: f32 = 3.0;
const PLANE_FILTER_NEIGHBOR_DISTANCE_THRESHOLD_MID70: f32 = 0.25; // m
const PLANE_FILTER_SOURCE_SCORE_THRESHOLD_MID70: f32 = 0.85;
const PLANE_FILTER_SOURCE_TO_PLANE_MAX_DISTANCE_MID70: f32 = 0.10; // m
const PLANE_FILTER_MIN_PLANARITY_MID70: f32 = 0.05;

const MAX_DIST_FOR_VOXEL_MAP_MID70: f32 = 150.0;

const MIN_DIST_MID70: f32 = 0.5;
const MAX_DIST_MID70: f32 = 150.0;

// --- The parametrers for Mid-70 ---

fn main() -> Result<()> {
    // if std::env::args()
    //     .skip(1)
    //     .any(|arg| arg == "--help" || arg == "-h")
    // {
    //     print_usage();
    //     return Ok(());
    // }

    // let config = AppConfig::from_args()?;

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();
    // log::info!(
    //     "Sensor: {}, input: {}, output: {}",
    //     config.sensor_type,
    //     config.load_dir,
    //     config.save_dir
    // );

    // <--- Loading each data --->
    let pcd_dir_airy96 = format!("{}/pcd", LOAD_DIR_AIRY96);
    let pcd_files_airy96 = load_pcd_files(&pcd_dir_airy96)?;
    ensure!(
        !pcd_files_airy96.is_empty(),
        "No cloud_<number>.pcd files found in {pcd_dir_airy96}"
    );

    let pcd_dir_mid70 = format!("{}/pcd", LOAD_DIR_MID70);
    let pcd_files_mid70 = load_pcd_files(&pcd_dir_mid70)?;
    ensure!(
        !pcd_files_mid70.is_empty(),
        "No cloud_<number>.pcd files found in {pcd_dir_mid70}"
    );

    log::debug!(
        "Found {} PCD files in directory: {}",
        pcd_files_airy96.len(),
        pcd_dir_airy96
    );

    log::debug!(
        "Found {} PCD files in directory: {}",
        pcd_files_mid70.len(),
        pcd_dir_mid70
    );

    let airy_from_mid70 = make_airy_from_mid70_extrinsic();

    let imu_context = {
        let imu_file = format!("{}/imu_data.json", LOAD_DIR_AIRY96_IMU);
        let imu_data = load_imu_data(&imu_file)?;
        let imu_data = align_imu_timestamps(&imu_data); // Align IMU timestamps to seconds
        ensure!(!imu_data.is_empty(), "No IMU samples found in {imu_file}");
        log::info!("Loaded {} IMU samples from {}", imu_data.len(), imu_file);

        // <--- IMU coord to LiDAR coord transformation --->
        let imu_to_lidar = UnitQuaternion::new_normalize(Quaternion::new(
            IMU_TO_LIDAR_QUAT_W,
            IMU_TO_LIDAR_QUAT_X,
            IMU_TO_LIDAR_QUAT_Y,
            IMU_TO_LIDAR_QUAT_Z,
        ));
        let airy_from_mid70_rotation =
            UnitQuaternion::from_matrix(&airy_from_mid70.fixed_view::<3, 3>(0, 0).into_owned());
        let imu_to_mid70 = airy_from_mid70_rotation.inverse() * imu_to_lidar;
        // <--- IMU coord to LiDAR coord transformation --->

        Some(ImuContext {
            samples: imu_data,
            imu_to_lidar,
            imu_to_mid70,
        })
    };
    // <--- Loading each data --->

    // <--- Initialize current frame info --->
    let mut current_frame_info = CurrentFrameInfo {
        current_global_pose: Matrix4::<f64>::identity(),
        current_velocity: Vector3::<f64>::zeros(),
    };
    // <--- Initialize current frame info --->

    // <--- Initialize SLAM map --->
    let local_map_config = LocalMapConfig {
        index_voxel_size: LOCAL_MAP_VOXEL_SIZE_AIRY96,
        max_points_per_voxel: 20,
        min_points_per_voxel: 5,
        min_observed_frames_per_voxel: 3,
        max_frames: 50,
        max_distance: MAX_DIST_FOR_VOXEL_MAP_AIRY96,
    };

    let global_map_config = LocalMapConfig {
        index_voxel_size: GLOBAL_MAP_VOXEL_SIZE_AIRY96,
        max_points_per_voxel: 20,
        min_points_per_voxel: 2,
        min_observed_frames_per_voxel: 2,
        max_frames: 50,
        max_distance: MAX_DIST_FOR_VOXEL_MAP_AIRY96,
    };
    let mut slam_map = SLAMMap {
        global_voxel_map: LOCALMap::new(global_map_config),
        local_voxel_map: LOCALMap::new(local_map_config),
    };

    let global_map_config_mid70 = LocalMapConfig {
        index_voxel_size: GLOBAL_MAP_VOXEL_SIZE_MID70,
        max_points_per_voxel: 20,
        min_points_per_voxel: 2,
        min_observed_frames_per_voxel: GLOBAL_MAP_MIN_OBSERVED_FRAMES_MID70,
        max_frames: 50,
        max_distance: MAX_DIST_FOR_VOXEL_MAP_MID70,
    };
    let mut mid70_global_voxel_map = LOCALMap::new(global_map_config_mid70);

    let plane_filter_local_map_config_mid70 = LocalMapConfig {
        index_voxel_size: PLANE_FILTER_LOCAL_MAP_VOXEL_SIZE_MID70,
        max_points_per_voxel: 20,
        min_points_per_voxel: 2,
        min_observed_frames_per_voxel: 1,
        max_frames: 50,
        max_distance: MAX_DIST_FOR_VOXEL_MAP_MID70,
    };
    let mut mid70_plane_filter_local_map = LOCALMap::new(plane_filter_local_map_config_mid70);
    // <--- Initialize SLAM map --->

    let mut prev_frame_start_time: f64 = 0.0;
    let mut frame_logs: Vec<FrameLog> = Vec::new();

    ensure!(
        pcd_files_airy96.len() == pcd_files_mid70.len(),
        "PCD frame count mismatch: Airy96 has {} frames, Mid-70 has {} frames",
        pcd_files_airy96.len(),
        pcd_files_mid70.len()
    );

    //
    for (i, (pcd_path_airy96, pcd_path_mid70)) in pcd_files_airy96
        .iter()
        .zip(pcd_files_mid70.iter())
        .enumerate()
    {
        log::info!(
            "Processing frame {}: Airy96={}, Mid-70={}",
            i,
            pcd_path_airy96.to_string_lossy(),
            pcd_path_mid70.to_string_lossy()
        );

        let source_pcd_airy96 = load_pcd_xyzit(&pcd_path_airy96.to_string_lossy())?;

        let source_pcd_mid70 = load_pcd_xyzit(&pcd_path_mid70.to_string_lossy())?;

        let current_frame_start_time_airy96 = source_pcd_airy96
            .iter()
            .map(|p| p.timestamp)
            .fold(f64::INFINITY, f64::min);
        let current_frame_end_time_airy96 = source_pcd_airy96
            .iter()
            .map(|p| p.timestamp)
            .fold(f64::NEG_INFINITY, f64::max);
        let current_frame_start_time_mid70 = source_pcd_mid70
            .iter()
            .map(|p| p.timestamp)
            .fold(f64::INFINITY, f64::min);
        let current_frame_end_time_mid70 = source_pcd_mid70
            .iter()
            .map(|p| p.timestamp)
            .fold(f64::NEG_INFINITY, f64::max);

        if i == 0 {
            prev_frame_start_time = current_frame_start_time_airy96;
        }
        let frame_delta_time = current_frame_start_time_airy96 - prev_frame_start_time;

        // --- Predict pose for the ICP initial value and fallback ---
        let (pose_prediction, prediction_source) = match &imu_context {
            Some(imu) => (
                predict_pose_by_imu(
                    &imu.samples,
                    &imu.imu_to_lidar,
                    &current_frame_info.current_global_pose,
                    &current_frame_info.current_velocity,
                    prev_frame_start_time,
                    current_frame_start_time_airy96,
                )
                .0,
                "IMU",
            ),
            None => (
                predict_pose_by_constant_velocity(
                    &current_frame_info.current_global_pose,
                    &current_frame_info.current_velocity,
                    frame_delta_time,
                ),
                "previous-frame velocity",
            ),
        };
        // --- Predict pose for the ICP initial value and fallback ---

        let processed_points_airy96 = match &imu_context {
            Some(imu) => {
                // <--- Build rotation trajectory --->
                let rotation_traj = build_rotation_trajectory(
                    &imu.samples,
                    current_frame_start_time_airy96,
                    current_frame_end_time_airy96,
                    &imu.imu_to_lidar,
                );
                // <--- Build rotation trajectory --->

                // --- Deskew source pcd ---
                deskew_points(
                    &source_pcd_airy96,
                    &rotation_traj,
                    &imu.imu_to_lidar,
                    current_frame_start_time_airy96,
                    MIN_DIST_AIRY96,
                    MAX_DIST_AIRY96,
                )
                // --- Deskew source pcd ---
            }
            None => filter_points_by_distance(&source_pcd_airy96, MIN_DIST_AIRY96, MAX_DIST_AIRY96),
        };

        let processed_points_mid70 = match &imu_context {
            Some(imu) => {
                // <--- Build rotation trajectory --->
                let rotation_traj = build_rotation_trajectory(
                    &imu.samples,
                    current_frame_start_time_mid70,
                    current_frame_end_time_mid70,
                    &imu.imu_to_mid70,
                );
                // <--- Build rotation trajectory --->

                // --- Deskew source pcd ---
                deskew_points(
                    &source_pcd_mid70,
                    &rotation_traj,
                    &imu.imu_to_mid70,
                    current_frame_start_time_mid70,
                    MIN_DIST_MID70,
                    MAX_DIST_MID70,
                )
                // --- Deskew source pcd ---
            }
            None => filter_points_by_distance(&source_pcd_mid70, MIN_DIST_MID70, MAX_DIST_MID70),
        };

        // --- Downsample processed points ---
        let voxel_start = std::time::Instant::now();
        let downsampled_source_points_for_local_airy96 =
            voxel_downsample_points(&processed_points_airy96, LOCAL_MAP_VOXEL_SIZE_AIRY96);
        let downsampled_source_points_for_global_airy96 =
            voxel_downsample_points(&processed_points_airy96, GLOBAL_MAP_VOXEL_SIZE_AIRY96);
        // let voxel_end = voxel_start.elapsed();

        let downsampled_source_points_for_global_mid70 =
            voxel_downsample_points(&processed_points_mid70, GLOBAL_MAP_VOXEL_SIZE_MID70);
        let downsampled_source_points_for_plane_filter_mid70 = voxel_downsample_points(
            &processed_points_mid70,
            PLANE_FILTER_LOCAL_DOWNSAMPLE_SIZE_MID70,
        );
        let voxel_end = voxel_start.elapsed();

        log::debug!(
            "Airy96 Frame {i}: Downsampled {} points → {} points",
            processed_points_airy96.len(),
            downsampled_source_points_for_local_airy96.len()
        );
        log::debug!(
            "MID-70 Frame {i}: Downsampled {} points → {} points, Total Voxelization process {:.2?}",
            processed_points_mid70.len(),
            downsampled_source_points_for_global_mid70.len(),
            voxel_end
        );
        // --- Downsample processed points ---

        // --- Build voxel map for source points ---
        let build_map_start = std::time::Instant::now();
        let source_voxel_map_airy96 = build_voxel_map(
            &downsampled_source_points_for_local_airy96,
            LOCAL_MAP_VOXEL_SIZE_AIRY96,
            NEIGHBOR_RANGE_AIRY96,
            false,
        );
        let source_voxel_map_for_global_airy96 = build_voxel_map(
            &downsampled_source_points_for_global_airy96,
            GLOBAL_MAP_VOXEL_SIZE_AIRY96,
            NEIGHBOR_RANGE_AIRY96,
            false,
        );
        let source_voxel_map_for_global_mid70 = build_voxel_map(
            &downsampled_source_points_for_global_mid70,
            GLOBAL_MAP_VOXEL_SIZE_MID70,
            0,
            false,
        );

        let build_map_end = build_map_start.elapsed();
        log::debug!("Frame {i}: Built voxel map in {:.2?}", build_map_end);
        // --- Build voxel map for source points ---

        // --- ICP (Point to Plane) ---
        // センサーモードに応じた予測姿勢を初期値として (R, t) を取り出す
        let pred_pose = pose_prediction.cast::<f32>();
        let mut r_mat: Matrix3<f32> = pred_pose.fixed_view::<3, 3>(0, 0).into();
        let mut t_vec: Vector3<f32> = pred_pose.fixed_view::<3, 1>(0, 3).into();

        let mut prev_rmse = f32::INFINITY;
        let mut icp_ok = false; // ICP が有効な解を得られたか

        let loop_start = std::time::Instant::now();

        if slam_map.local_voxel_map.voxel_map.is_empty() {
            log::debug!("Frame {i}: local map empty, skipping ICP");
        } else {
            for _iter in 0..ICP_ITERATIONS_AIRY96 {
                // 対応点をピックアップ
                // - source はローカル座標、target (local_voxel_map) はワールド座標
                // - 現在の (R,t) 推定値で source をワールド変換してから近傍探索
                let pickup_start = std::time::Instant::now();
                let correspondences = pickup_valid_source_points(
                    &source_voxel_map_airy96,
                    &slam_map.local_voxel_map.voxel_map,
                    slam_map.local_voxel_map.config.index_voxel_size,
                    SEARCH_RANGE_AIRY96,
                    LOCAL_KNN_K_AIRY96,
                    MAX_DIST_FACTOR_AIRY96,
                    LOCAL_PLANE_POINT_DISTANCE_THRESHOLD_M_AIRY96,
                    LOCAL_SOURCE_PLANE_SCORE_THRESHOLD_AIRY96,
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
                let system_start = std::time::Instant::now();
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

                // RMSE が発散した場合は ICP 結果を棄却して予測姿勢に戻す
                if rmse > ICP_RMSE_DIVERGE_THRESHOLD_AIRY96 {
                    log::warn!(
                        "ICP iter {_iter}: RMSE diverged ({rmse:.4}), reverting to \
                         {prediction_source} prediction"
                    );
                    r_mat = pred_pose.fixed_view::<3, 3>(0, 0).into();
                    t_vec = pred_pose.fixed_view::<3, 1>(0, 3).into();
                    icp_ok = false;
                    break;
                }

                // if (prev_rmse - rmse).abs() < ICP_RMSE_THRESHOLD {
                if rmse < ICP_RMSE_THRESHOLD_AIRY96 {
                    log::debug!(
                        "ICP converged at iter {_iter} (|Δrmse|={:.2e})",
                        (prev_rmse - rmse).abs()
                    );
                    break;
                }
                prev_rmse = rmse;
            }

            if !icp_ok {
                log::warn!("Frame {i}: ICP failed, using {prediction_source} prediction");
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
        let dt = frame_delta_time.max(1e-6);
        let raw_velocity = (new_pos - prev_pos) / dt;
        // 速度が異常に大きい場合（ICP 発散など）はクランプして安定化
        let new_velocity = raw_velocity.cap_magnitude(2.0);

        current_frame_info.current_global_pose = new_global_pose;
        current_frame_info.current_velocity = new_velocity;
        // --- Update current frame info ---

        // MID-70点群をAiry SLAMと同じWorld座標へ変換する姿勢
        let mid70_global_pose = &current_frame_info.current_global_pose * &airy_from_mid70;
        let mid70_pose_f32 = mid70_global_pose.cast::<f32>();
        let r_mat_mid70: Matrix3<f32> = mid70_pose_f32.fixed_view::<3, 3>(0, 0).into();
        let t_vec_mid70: Vector3<f32> = mid70_pose_f32.fixed_view::<3, 1>(0, 3).into();

        // --- Record frame log ---
        let translation_m = (new_pos - prev_pos).norm();
        let delta_r = r64 * prev_r.transpose();
        let rotation_deg = ((delta_r.trace() - 1.0) / 2.0)
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        frame_logs.push(FrameLog {
            frame_index: i,
            timestamp: current_frame_start_time_airy96,
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

        // --- Filter valid source points, then update the WorldMap ---
        // ローカルマップが空（初回フレーム）の場合はフィルタなしで全点追加。
        // それ以外は pickup_valid_source_points で平面に乗っている点だけ抽出し、
        // ICP 収束後の最終姿勢 (r_mat, t_vec) でワールドマップに追加する。
        let global_source_points: Vec<Point3<f32>> =
            if slam_map.local_voxel_map.voxel_map.is_empty() {
                downsampled_source_points_for_global_airy96.clone()
            } else {
                let valid = pickup_valid_source_points(
                    &source_voxel_map_for_global_airy96,
                    &slam_map.local_voxel_map.voxel_map,
                    slam_map.local_voxel_map.config.index_voxel_size,
                    SEARCH_RANGE_AIRY96,
                    GLOBAL_KNN_K_AIRY96,
                    MAX_DIST_FACTOR_AIRY96,
                    GLOBAL_PLANE_POINT_DISTANCE_THRESHOLD_M_AIRY96,
                    GLOBAL_SOURCE_PLANE_SCORE_THRESHOLD_AIRY96,
                    Some(GLOBAL_SOURCE_TO_PLANE_MAX_DISTANCE_M_AIRY96),
                    Some(GLOBAL_MIN_PLANARITY_AIRY96),
                    &r_mat,
                    &t_vec,
                );
                log::debug!(
                    "Airy96 Frame {i}: {} / {} source points passed plane filter for global map",
                    valid.len(),
                    source_voxel_map_for_global_airy96.len(),
                );
                valid.into_iter().map(|c| c.src_point).collect()
            };
        slam_map.global_voxel_map.update_world_map(
            &global_source_points,
            &current_frame_info.current_global_pose,
        );
        // --- Filter valid source points, then update the WorldMap ---

        // --- Update the LocalMap with the new frame's points ---
        slam_map.local_voxel_map.update_with_new_frame(
            &downsampled_source_points_for_local_airy96,
            &current_frame_info.current_global_pose,
        );
        // --- Update the LocalMap with the new frame's points ---

        // Mid-70は疎な点分布を考慮した広域近傍探索で平面点を選別する。
        // 初回だけはLocalMapが空なので、地図のシードとして全点を追加する。
        let global_source_points_mid70: Vec<Point3<f32>> = if mid70_plane_filter_local_map
            .voxel_map
            .is_empty()
        {
            downsampled_source_points_for_global_mid70.clone()
        } else {
            let valid = pickup_valid_source_points(
                &source_voxel_map_for_global_mid70,
                &mid70_plane_filter_local_map.voxel_map,
                mid70_plane_filter_local_map.config.index_voxel_size,
                PLANE_FILTER_SEARCH_RANGE_MID70,
                PLANE_FILTER_K_MID70,
                PLANE_FILTER_MAX_DIST_FACTOR_MID70,
                PLANE_FILTER_NEIGHBOR_DISTANCE_THRESHOLD_MID70,
                PLANE_FILTER_SOURCE_SCORE_THRESHOLD_MID70,
                Some(PLANE_FILTER_SOURCE_TO_PLANE_MAX_DISTANCE_MID70),
                Some(PLANE_FILTER_MIN_PLANARITY_MID70),
                &r_mat_mid70,
                &t_vec_mid70,
            );
            log::debug!(
                "MID-70 Frame {i}: {} / {} source points passed sparse-cloud plane filter for global map",
                valid.len(),
                source_voxel_map_for_global_mid70.len(),
            );
            valid
                .into_iter()
                .map(|correspondence| correspondence.src_point)
                .collect()
        };
        mid70_global_voxel_map.update_world_map(&global_source_points_mid70, &mid70_global_pose);
        mid70_plane_filter_local_map.update_with_new_frame(
            &downsampled_source_points_for_plane_filter_mid70,
            &mid70_global_pose,
        );

        prev_frame_start_time = current_frame_start_time_airy96;
    }

    // --- Save the final global voxel map to a PCD file ---
    let min_samples = slam_map.global_voxel_map.config.min_points_per_voxel as u64;

    let min_samples_mid70 = mid70_global_voxel_map.config.min_points_per_voxel as u64;

    let min_frames = slam_map
        .global_voxel_map
        .config
        .min_observed_frames_per_voxel as u64;

    let min_frames_mid70 = mid70_global_voxel_map.config.min_observed_frames_per_voxel as u64;

    let world_map_points: Vec<Point3<f32>> = slam_map
        .global_voxel_map
        .voxel_map
        .values()
        .filter(|cell| cell.sample_count >= min_samples && cell.observed_frames >= min_frames)
        .map(|cell| Point3::new(cell.mean.x, cell.mean.y, cell.mean.z))
        .collect();

    let world_map_points_mid70: Vec<Point3<f32>> = mid70_global_voxel_map
        .voxel_map
        .values()
        .filter(|cell| {
            cell.sample_count >= min_samples_mid70 && cell.observed_frames >= min_frames_mid70
        })
        .map(|cell| Point3::new(cell.mean.x, cell.mean.y, cell.mean.z))
        .collect();

    let downsampled_world_map_points =
        voxel_downsample_points(&world_map_points, GLOBAL_MAP_VOXEL_SIZE_AIRY96);
    let downsampled_world_map_points_xyz: Vec<PointXYZ> = downsampled_world_map_points
        .iter()
        .map(|point| PointXYZ {
            x: point.x,
            y: point.y,
            z: point.z,
        })
        .collect();

    let downsampled_world_map_points_mid70 =
        voxel_downsample_points(&world_map_points_mid70, GLOBAL_MAP_VOXEL_SIZE_MID70);
    let downsampled_world_map_points_mid70_xyz: Vec<PointXYZ> = downsampled_world_map_points_mid70
        .iter()
        .map(|point| PointXYZ {
            x: point.x,
            y: point.y,
            z: point.z,
        })
        .collect();

    let world_map_path = format!(
        "{}/airy96_voxel-{}_world_map.pcd",
        SAVE_DIR, GLOBAL_MAP_VOXEL_SIZE_AIRY96
    );
    std::fs::create_dir_all(SAVE_DIR)?;
    save_pcd_xyz(&downsampled_world_map_points_xyz, &world_map_path)?;
    log::info!(
        "Saved downsampled world map: {} → {} points → {}",
        world_map_points.len(),
        downsampled_world_map_points_xyz.len(),
        world_map_path
    );

    let world_map_path_mid70 = format!(
        "{}/mid70_voxel-{}_world_map.pcd",
        SAVE_DIR, GLOBAL_MAP_VOXEL_SIZE_MID70
    );
    std::fs::create_dir_all(SAVE_DIR)?;
    save_pcd_xyz(
        &downsampled_world_map_points_mid70_xyz,
        &world_map_path_mid70,
    )?;
    log::info!(
        "Saved downsampled world map: {} → {} points → {}",
        world_map_points_mid70.len(),
        downsampled_world_map_points_mid70_xyz.len(),
        world_map_path_mid70
    );
    // --- Save the final global voxel map to a PCD file ---

    // --- Save per-frame ICP logs to JSON ---
    let frame_logs_path = format!("{}/frame_logs.json", SAVE_DIR);
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

fn print_usage() {
    println!(
        "Usage: re_lidar_slam [--sensor airy96|mid70] [--load-dir DIR] [--save-dir DIR]\n\
         Defaults: --sensor airy96 --load-dir {LOAD_DIR_AIRY96} --save-dir {SAVE_DIR}"
    );
}

#[cfg(test)]
mod tests {
    use nalgebra::{Matrix4, Point3};

    use super::{
        AppConfig, GLOBAL_MAP_MIN_OBSERVED_FRAMES_MID70, GLOBAL_MAP_VOXEL_SIZE_MID70,
        LOAD_DIR_AIRY96, MID70_ORIGIN_IN_AIRY_Z_M, PLANE_FILTER_K_MID70,
        PLANE_FILTER_LOCAL_MAP_VOXEL_SIZE_MID70, PLANE_FILTER_MAX_DIST_FACTOR_MID70,
        PLANE_FILTER_MIN_PLANARITY_MID70, PLANE_FILTER_SEARCH_RANGE_MID70,
        PLANE_FILTER_SOURCE_TO_PLANE_MAX_DISTANCE_MID70, SAVE_DIR, SensorType,
        make_airy_from_mid70_extrinsic,
    };

    #[test]
    fn app_config_preserves_airy96_defaults() {
        let config = AppConfig::parse(Vec::new()).unwrap();

        assert_eq!(config.sensor_type, SensorType::Airy96);
        assert_eq!(config.load_dir, LOAD_DIR_AIRY96);
        assert_eq!(config.save_dir, SAVE_DIR);
    }

    #[test]
    fn app_config_accepts_mid70_and_custom_directories() {
        let config = AppConfig::parse(
            [
                "--sensor",
                "mid-70",
                "--load-dir",
                "mid-input",
                "--save-dir",
                "mid-output",
            ]
            .map(str::to_owned),
        )
        .unwrap();

        assert_eq!(config.sensor_type, SensorType::Mid70);
        assert_eq!(config.load_dir, "mid-input");
        assert_eq!(config.save_dir, "mid-output");
    }

    #[test]
    fn mid70_extrinsic_maps_points_into_airy_coordinates() {
        let airy_from_mid70 = make_airy_from_mid70_extrinsic();
        let point_in_mid70 = Point3::new(1.0, 0.0, 0.0);
        let point_in_airy = airy_from_mid70.transform_point(&point_in_mid70);

        assert!((point_in_airy.x - 0.0).abs() < 1e-12);
        assert!((point_in_airy.y + 1.0).abs() < 1e-12);
        assert!((point_in_airy.z - MID70_ORIGIN_IN_AIRY_Z_M).abs() < 1e-12);

        let mut world_from_airy = Matrix4::<f64>::identity();
        world_from_airy[(0, 3)] = 10.0;
        world_from_airy[(1, 3)] = 20.0;
        world_from_airy[(2, 3)] = 30.0;
        let world_from_mid70 = world_from_airy * airy_from_mid70;
        let point_in_world = world_from_mid70.transform_point(&point_in_mid70);

        assert!((point_in_world.x - 10.0).abs() < 1e-12);
        assert!((point_in_world.y - 19.0).abs() < 1e-12);
        assert!((point_in_world.z - (30.0 + MID70_ORIGIN_IN_AIRY_Z_M)).abs() < 1e-12);
    }

    #[test]
    fn mid70_dense_map_preserves_selected_resolution_and_temporal_filter() {
        assert!((GLOBAL_MAP_VOXEL_SIZE_MID70 - 0.15).abs() < f32::EPSILON);
        assert_eq!(GLOBAL_MAP_MIN_OBSERVED_FRAMES_MID70, 2);
    }

    #[test]
    fn mid70_sparse_plane_filter_uses_wide_relaxed_neighborhood() {
        let search_extent_m =
            PLANE_FILTER_LOCAL_MAP_VOXEL_SIZE_MID70 * PLANE_FILTER_SEARCH_RANGE_MID70 as f32;
        let distance_cap_m =
            PLANE_FILTER_LOCAL_MAP_VOXEL_SIZE_MID70 * PLANE_FILTER_MAX_DIST_FACTOR_MID70;

        assert!((search_extent_m - 3.0).abs() < f32::EPSILON);
        assert!((distance_cap_m - 3.0).abs() < f32::EPSILON);
        assert_eq!(PLANE_FILTER_K_MID70, 5);
        assert!((PLANE_FILTER_SOURCE_TO_PLANE_MAX_DISTANCE_MID70 - 0.10).abs() < f32::EPSILON);
        assert!((PLANE_FILTER_MIN_PLANARITY_MID70 - 0.05).abs() < f32::EPSILON);
    }
}

fn make_airy_from_mid70_extrinsic() -> Matrix4<f64> {
    let rotation = Matrix3::<f64>::new(0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

    let translation = Vector3::<f64>::new(
        MID70_ORIGIN_IN_AIRY_X_M,
        MID70_ORIGIN_IN_AIRY_Y_M,
        MID70_ORIGIN_IN_AIRY_Z_M,
    );

    let mut transform = Matrix4::<f64>::identity();
    transform.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation);
    transform
        .fixed_view_mut::<3, 1>(0, 3)
        .copy_from(&translation);

    transform
}
