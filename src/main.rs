use anyhow::Result;
use nalgebra::{Matrix4, Point3, Quaternion, UnitQuaternion, Vector3};
use re_lidar_slam::{
    deskew_points::deskew_points,
    file_handler::{load_imu_data, load_pcd_files, load_pcd_xyzit},
    find_nearest_points::pickup_valid_source_points,
    predict_pose_by_imu::{align_imu_timestamps, build_rotation_trajectory, predict_pose_by_imu},
    types::{CurrentFrameInfo, SLAMMap},
    voxel_map::{LOCALMap, LocalMapConfig, build_voxel_map},
    voxelization::voxel_downsample_points,
};

const LOAD_DIR: &str = "/home/kenji/workspace/rust/gicp-slam-vulkan/data/input/06212026/park05";
const SAVE_DIR: &str = "data/output/06212026/debug";

const MIN_DIST: f32 = 0.1;
const MAX_DIST: f32 = 40.0;

// IMU coordination to LiDAR coordination (Robosense 96 beam)
// Quaternion (x, y, z, w): -0.705437, 0.708767, -0.00246579, 0.00097028
// Translation (x, y, z)  : 0.00425, 0.00418, -0.00446  [m]
const IMU_TO_LIDAR_QUAT_X: f64 = -0.705437;
const IMU_TO_LIDAR_QUAT_Y: f64 = 0.708767;
const IMU_TO_LIDAR_QUAT_Z: f64 = -0.00246579;
const IMU_TO_LIDAR_QUAT_W: f64 = 0.00097028;

const DOWNSAMPLE_VOXEL_SIZE: f32 = 0.2; // m

const KNN_K: usize = 5; // Number of nearest neighbors for plane fitting
const SEARCH_RANGE: i32 = 2; // Voxel search range for nearest neighbor search
const MAX_DIST_FACTOR: f32 = 3.0; // Maximum distance factor for nearest neighbor search
const PLANE_FIT_THRESHOLD: f32 = 0.1; // Threshold for plane fitting

const GICP_ITERATIONS: usize = 5; // Default: 5

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();

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
    let map_config = LocalMapConfig {
        index_voxel_size: 1.0,
        max_points_per_voxel: 20,
        min_points_per_voxel: 3,
        min_observed_frames_per_voxel: 3,
        max_frames: 50,
        max_distance: MAX_DIST,
    };
    let slam_map = SLAMMap {
        global_voxel_map: LOCALMap::new(map_config),
        local_voxel_map: LOCALMap::new(map_config),
    };
    // <--- Initialize SLAM map --->

    let mut prev_frame_start_time: f64 = 0.0;
    let mut target_points: Vec<Point3<f32>> = Vec::new();

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
        let downsampled_source_points =
            voxel_downsample_points(&deskewed_points, DOWNSAMPLE_VOXEL_SIZE);
        // --- Downsample deskewed points ---

        // --- Downsample target points ---
        let downsampled_target_points =
            voxel_downsample_points(&target_points, DOWNSAMPLE_VOXEL_SIZE);
        // --- Downsample target points ---

        // --- Build voxel map for source points ---
        let source_voxel_map = build_voxel_map(&downsampled_source_points, DOWNSAMPLE_VOXEL_SIZE);
        // --- Build voxel map for source points ---

        // --- Build voxel map for target points ---
        let target_voxel_map = build_voxel_map(&downsampled_target_points, DOWNSAMPLE_VOXEL_SIZE);
        // --- Build voxel map for target points ---

        // --- Pick up valid source points by checking if they are close enough to the target plane ---
        let valid_source_points = pickup_valid_source_points(
            &source_voxel_map,
            &target_voxel_map,
            DOWNSAMPLE_VOXEL_SIZE,
            SEARCH_RANGE,
            KNN_K,               // k
            MAX_DIST_FACTOR,     // max_dist_factor
            PLANE_FIT_THRESHOLD, // plane_fit_threshold
        );
        // --- Pick up valid source points by checking if they are close enough to the target plane ---
    }

    Ok(())
}
