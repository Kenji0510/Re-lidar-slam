use nalgebra::{Matrix4, Point3, Vector3};
use pcd_rs::{PcdDeserialize, PcdSerialize};
use serde::{Deserialize, Serialize};

use crate::voxel_map::LOCALMap;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoadIMU {
    pub timestamp: u64,
    pub angular_velocity: [f32; 3],
    pub linear_acceleration: [f32; 3],
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct IMU {
    pub timestamp: f64,
    pub angular_velocity: [f32; 3],
    pub linear_acceleration: [f32; 3],
}

#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct PointXYZ {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct PointXYZIT {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub intensity: f32,
    pub timestamp: f64,
}

#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct PointXYZCov {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub cov_xx: f32,
    pub cov_xy: f32,
    pub cov_xz: f32,
    pub cov_yy: f32,
    pub cov_yz: f32,
    pub cov_zz: f32,
}

#[derive(Debug, Clone)]
pub struct FrameData {
    pub points: Vec<Point3<f32>>,
    pub covariances: Vec<[[f32; 3]; 3]>,
}

#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct PointXYZNormal {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub normal_x: f32,
    pub normal_y: f32,
    pub normal_z: f32,
}

pub struct CurrentFrameInfo {
    pub current_global_pose: Matrix4<f64>,
    pub current_velocity: Vector3<f64>,
}

pub struct SLAMMap {
    pub global_voxel_map: LOCALMap,
    pub local_voxel_map: LOCALMap,
}

pub struct ProcessTimes {
    pub total: f64,
    pub find_nearest_points: f64,
    pub icp: f64,
    pub update_map: f64,
}

/// Wall-clock timings for one point-cloud frame, in milliseconds.
///
/// The top-level stages do not overlap and can therefore be compared directly.
/// The `icp_*` detail fields are a breakdown of `icp_ms` and must not be added
/// to the top-level stages a second time.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct FrameTiming {
    pub load_pcd_ms: f64,
    pub timestamps_ms: f64,
    pub imu_predict_ms: f64,
    pub rotation_trajectory_ms: f64,
    pub deskew_ms: f64,
    pub downsample_ms: f64,
    pub build_source_maps_ms: f64,
    pub icp_ms: f64,
    pub pose_update_ms: f64,
    pub global_filter_ms: f64,
    pub global_map_update_ms: f64,
    pub delayed_surface_ms: f64,
    pub local_map_update_ms: f64,
    /// Per-frame processing time excluding PCD file I/O.
    pub processing_total_ms: f64,
    /// Per-frame wall time including PCD file I/O.
    pub total_with_file_io_ms: f64,
    /// Time spent finding ICP correspondences, including final validation.
    pub icp_correspondence_search_ms: f64,
    /// Time spent building ICP point-to-plane linear systems.
    pub icp_linear_system_ms: f64,
    /// Time spent solving ICP linear systems.
    pub icp_solver_ms: f64,
    /// Remaining ICP work, calculated from `icp_ms`.
    pub icp_other_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameLog {
    pub frame_index: usize,
    pub timestamp: f64,
    pub icp_ok: bool,
    /// Final Point-to-Plane RMSE [m]. May be present for a candidate that was
    /// rejected by another quality gate; None when ICP produced no candidate.
    pub rmse: Option<f32>,
    #[serde(default)]
    pub correspondence_count: usize,
    #[serde(default)]
    pub correspondence_ratio: f32,
    #[serde(default)]
    pub observable_rank: usize,
    #[serde(default)]
    pub min_observable_eigenvalue_ratio: Option<f32>,
    #[serde(default)]
    pub icp_translation_correction_m: f32,
    #[serde(default)]
    pub icp_rotation_correction_deg: f32,
    #[serde(default)]
    pub map_updated: bool,
    /// Translation distance from previous frame [m].
    pub translation_m: f64,
    /// Rotation angle from previous frame [deg].
    pub rotation_deg: f64,
    /// Speed (velocity magnitude) [m/s].
    pub velocity_m_s: f64,
    pub pose_x: f64,
    pub pose_y: f64,
    pub pose_z: f64,
    #[serde(default)]
    pub timing: FrameTiming,
}

#[cfg(test)]
mod tests {
    use super::FrameLog;

    #[test]
    fn frame_log_reads_legacy_json_without_quality_fields() {
        let json = r#"{
            "frame_index": 1,
            "timestamp": 2.0,
            "icp_ok": true,
            "rmse": 0.03,
            "translation_m": 0.1,
            "rotation_deg": 0.2,
            "velocity_m_s": 1.0,
            "pose_x": 1.0,
            "pose_y": 2.0,
            "pose_z": 3.0
        }"#;

        let log: FrameLog = serde_json::from_str(json).unwrap();
        assert_eq!(log.correspondence_count, 0);
        assert_eq!(log.correspondence_ratio, 0.0);
        assert_eq!(log.observable_rank, 0);
        assert!(!log.map_updated);
        assert_eq!(log.timing.processing_total_ms, 0.0);
    }
}
