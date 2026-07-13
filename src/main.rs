use anyhow::Result;
use nalgebra::{Matrix3, Matrix4, Point3, Quaternion, UnitQuaternion, Vector3};
use re_lidar_slam::{
    deskew_points::deskew_points,
    file_handler::{load_imu_data, load_pcd_files, load_pcd_xyzit, save_pcd_xyz},
    find_nearest_points::pickup_valid_source_points,
    icp::{apply_delta, build_point_to_plane_system, compute_rmse, solve_icp_delta},
    predict_pose_by_imu::{align_imu_timestamps, build_rotation_trajectory, predict_pose_by_imu},
    types::{CurrentFrameInfo, FrameLog, PointXYZ, ProcessTimes, SLAMMap},
    voxel_map::{LOCALMap, LocalMapConfig, build_voxel_map},
    voxelization::voxel_downsample_points,
};

const LOAD_DIR: &str = "data/input/06212026/park05";
const SAVE_DIR: &str = "data/output/debug/07122026";

const MIN_DIST: f32 = 0.5;
const MAX_DIST: f32 = 40.0;

// IMU coordination to LiDAR coordination (Robosense 96 beam)
// Quaternion (x, y, z, w): -0.705437, 0.708767, -0.00246579, 0.00097028
// Translation (x, y, z)  : 0.00425, 0.00418, -0.00446  [m]
const IMU_TO_LIDAR_QUAT_X: f64 = -0.705437;
const IMU_TO_LIDAR_QUAT_Y: f64 = 0.708767;
const IMU_TO_LIDAR_QUAT_Z: f64 = -0.00246579;
const IMU_TO_LIDAR_QUAT_W: f64 = 0.00097028;

const DOWNSAMPLE_VOXEL_SIZE: f32 = 0.25; // m
const LOCAL_MAP_VOXEL_SIZE: f32 = 0.25; // m
const GLOBAL_MAP_VOXEL_SIZE: f32 = 0.1; // m

const NEIGHBOR_RANGE: i32 = 2; // Voxel search range for nearest neighbor search

const KNN_K: usize = 5; // Number of nearest neighbors for plane fitting
const SEARCH_RANGE: i32 = 2; // Voxel search range for nearest neighbor search
const MAX_DIST_FACTOR: f32 = 2.5; // Maximum distance factor for nearest neighbor search (Prev: 3.0)
const PLANE_FIT_THRESHOLD_FOR_LOCAL: f32 = 0.75; // Threshold for plane fitting (Fast-LIO2 基準 s > 0.9)
const PLANE_FIT_THRESHOLD_FOR_GLOBAL: f32 = 0.75; // Threshold for plane fitting (Fast-LIO2 基準 s > 0.9)
const MAX_POINTS_PER_VOXEL: usize = 8; // Max points collected per voxel (for covariance)

const ICP_ITERATIONS: usize = 5; // Default: 5
const ICP_RMSE_THRESHOLD: f32 = 0.077; // 収束判定: RMSE の変化量がこれ以下なら停止 // voxel size 0.2m の場合、0.07m くらいが妥当
const ICP_RMSE_DIVERGE_THRESHOLD: f32 = 2.0; // 発散判定: RMSE がこれ以上なら結果棄却→IMU予測にフォールバック

const MAX_DIST_FOR_VOXEL_MAP: f32 = 40.0;

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();

    let mut process_times = ProcessTimes {
        total: 0.0,
        find_nearest_points: 0.0,
        icp: 0.0,
        update_map: 0.0,
    };

    // <--- Loading each data --->
    let pcd_dir = format!("{}/pcd", LOAD_DIR);
    let pcd_files = load_pcd_files(&pcd_dir)?;

    log::debug!(
        "Found {} PCD files in directory: {}",
        pcd_files.len(),
        pcd_dir
    );

    let imu_dir = format!("{}/imu", LOAD_DIR);
    let imu_file = format!("{}/imu_data.json", imu_dir);
    let imu_data = load_imu_data(&imu_file)?;
    let imu_data = align_imu_timestamps(&imu_data); // Align IMU timestamps to seconds
    // <--- Loading each data --->

    // <--- IMU coord to LiDAR coord transformation --->
    let imu_to_lidar = UnitQuaternion::new_normalize(Quaternion::new(
        IMU_TO_LIDAR_QUAT_W,
        IMU_TO_LIDAR_QUAT_X,
        IMU_TO_LIDAR_QUAT_Y,
        IMU_TO_LIDAR_QUAT_Z,
    ));
    // <--- IMU coord to LiDAR coord transformation --->

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
        min_points_per_voxel: 3,
        min_observed_frames_per_voxel: 3,
        max_frames: 50,
        max_distance: MAX_DIST_FOR_VOXEL_MAP,
    };

    let global_map_config = LocalMapConfig {
        index_voxel_size: GLOBAL_MAP_VOXEL_SIZE,
        max_points_per_voxel: 20,
        min_points_per_voxel: 3,
        min_observed_frames_per_voxel: 3,
        max_frames: 50,
        max_distance: MAX_DIST_FOR_VOXEL_MAP,
    };
    let mut slam_map = SLAMMap {
        global_voxel_map: LOCALMap::new(global_map_config),
        local_voxel_map: LOCALMap::new(local_map_config),
    };
    // <--- Initialize SLAM map --->

    let mut prev_frame_start_time: f64 = 0.0;
    let mut frame_logs: Vec<FrameLog> = Vec::new();

    let start_time = std::time::Instant::now();

    //
    for (i, pcd_path) in pcd_files.iter().enumerate() {
        log::info!("Processing frame {}: {}", i, pcd_path.to_string_lossy());

        let source_pcd = load_pcd_xyzit(&pcd_path.to_string_lossy())?;

        let current_frame_start_time = source_pcd
            .iter()
            .map(|p| p.timestamp)
            .fold(f64::INFINITY, f64::min);
        let current_frame_end_time = source_pcd
            .iter()
            .map(|p| p.timestamp)
            .fold(f64::NEG_INFINITY, f64::max);

        if i == 0 {
            prev_frame_start_time = current_frame_start_time;
        }

        // <--- Predict pose by IMU --->
        let pose_prediction = predict_pose_by_imu(
            &imu_data,
            &imu_to_lidar,
            &current_frame_info.current_global_pose,
            &current_frame_info.current_velocity,
            prev_frame_start_time,
            current_frame_start_time,
        );
        // <--- Predict pose by IMU --->

        // <--- Build rotation trajectory --->
        let rotation_traj = build_rotation_trajectory(
            &imu_data,
            current_frame_start_time,
            current_frame_end_time,
            &imu_to_lidar,
        );
        // <--- Build rotation trajectory --->

        // --- Deskew source pcd ---
        let deskewed_points = deskew_points(
            &source_pcd,
            &rotation_traj,
            &imu_to_lidar,
            current_frame_start_time,
            MIN_DIST,
            MAX_DIST,
        );
        // --- Deskew source pcd ---

        // --- Downsample deskewed points ---
        let voxel_start = std::time::Instant::now();
        let downsampled_source_points_for_local =
            voxel_downsample_points(&deskewed_points, LOCAL_MAP_VOXEL_SIZE);
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
        let build_map_start = std::time::Instant::now();
        let source_voxel_map = build_voxel_map(
            &downsampled_source_points_for_local,
            LOCAL_MAP_VOXEL_SIZE,
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

        let loop_start = std::time::Instant::now();

        if slam_map.local_voxel_map.voxel_map.is_empty() {
            log::debug!("Frame {i}: local map empty, skipping ICP");
        } else {
            for _iter in 0..ICP_ITERATIONS {
                // 対応点をピックアップ
                // - source はローカル座標、target (local_voxel_map) はワールド座標
                // - 現在の (R,t) 推定値で source をワールド変換してから近傍探索
                let pickup_start = std::time::Instant::now();
                let correspondences = pickup_valid_source_points(
                    &source_voxel_map,
                    &slam_map.local_voxel_map.voxel_map,
                    slam_map.local_voxel_map.config.index_voxel_size,
                    SEARCH_RANGE,
                    KNN_K,
                    MAX_DIST_FACTOR,
                    PLANE_FIT_THRESHOLD_FOR_LOCAL,
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
            rmse: if prev_rmse.is_finite() { Some(prev_rmse) } else { None },
            translation_m,
            rotation_deg,
            velocity_m_s: new_velocity.norm(),
            pose_x: new_pos.x,
            pose_y: new_pos.y,
            pose_z: new_pos.z,
        });
        // --- Record frame log ---

        // --- Update the LocalMap with the new frame's points ---
        slam_map.local_voxel_map.update_with_new_frame(
            &downsampled_source_points_for_local,
            &current_frame_info.current_global_pose,
        );
        // --- Update the LocalMap with the new frame's points ---

        // --- Filter valid source points, then update the WorldMap ---
        // ローカルマップが空（初回フレーム）の場合はフィルタなしで全点追加。
        // それ以外は pickup_valid_source_points で平面に乗っている点だけ抽出し、
        // ICP 収束後の最終姿勢 (r_mat, t_vec) でワールドマップに追加する。
        let global_source_points: Vec<Point3<f32>> =
            if slam_map.local_voxel_map.voxel_map.is_empty() {
                downsampled_source_points_for_global.clone()
            } else {
                let valid = pickup_valid_source_points(
                    &source_voxel_map_for_global,
                    &slam_map.local_voxel_map.voxel_map,
                    slam_map.local_voxel_map.config.index_voxel_size,
                    SEARCH_RANGE,
                    KNN_K,
                    MAX_DIST_FACTOR,
                    PLANE_FIT_THRESHOLD_FOR_GLOBAL,
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
        slam_map.global_voxel_map.update_world_map(
            &global_source_points,
            &current_frame_info.current_global_pose,
        );
        // --- Filter valid source points, then update the WorldMap ---

        prev_frame_start_time = current_frame_start_time;
    }

    // --- Save the final global voxel map to a PCD file ---
    let world_map_points: Vec<PointXYZ> = slam_map
        .global_voxel_map
        .voxel_map
        .values()
        .filter(|cell| cell.is_point)
        .map(|cell| PointXYZ {
            x: cell.point.0.x,
            y: cell.point.0.y,
            z: cell.point.0.z,
        })
        .collect();

    let world_map_path = format!("{}/voxel-{}_world_map.pcd", SAVE_DIR, GLOBAL_MAP_VOXEL_SIZE);
    std::fs::create_dir_all(SAVE_DIR)?;
    save_pcd_xyz(&world_map_points, &world_map_path)?;
    log::info!(
        "Saved world map: {} points → {}",
        world_map_points.len(),
        world_map_path
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
