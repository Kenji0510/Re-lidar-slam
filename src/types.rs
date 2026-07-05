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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameLog {
    pub frame_index: usize,
    pub timestamp: f64,
    pub icp_ok: bool,
    /// Point-to-Plane RMSE [m]. None if ICP was not run (e.g. empty map on first frame).
    pub rmse: Option<f32>,
    /// Translation distance from previous frame [m].
    pub translation_m: f64,
    /// Rotation angle from previous frame [deg].
    pub rotation_deg: f64,
    /// Speed (velocity magnitude) [m/s].
    pub velocity_m_s: f64,
    pub pose_x: f64,
    pub pose_y: f64,
    pub pose_z: f64,
}
