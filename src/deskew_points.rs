use nalgebra::{Point3, UnitQuaternion, Vector3};
use rayon::prelude::*;

use crate::{
    convert_imu_data::DeltaRotation, predict_pose_by_imu::RotationTrajectory, types::PointXYZIT,
};

pub fn filter_points_by_distance(
    pcd: &[PointXYZIT],
    min_dist: f32,
    max_dist: f32,
) -> Vec<Point3<f32>> {
    pcd.par_iter()
        .filter(|p| is_valid_point(p, min_dist, max_dist))
        .map(|p| Point3::new(p.x, p.y, p.z))
        .collect()
}

pub fn deskew_points(
    pcd: &Vec<PointXYZIT>,
    trajectory: &RotationTrajectory,
    _imu_to_lidar: &UnitQuaternion<f64>,
    frame_min_time: f64,
    min_dist: f32,
    max_dist: f32,
) -> Vec<Point3<f32>> {
    let start_rotation = get_rotation_at_time(trajectory, frame_min_time);
    let start_rotation_inv = start_rotation.inverse();

    pcd.par_iter()
        .filter_map(|p| {
            if !is_valid_point(p, min_dist, max_dist) || !p.timestamp.is_finite() {
                return None;
            }
            let current_rotation = get_rotation_at_time(trajectory, p.timestamp);
            let relative_rotation = start_rotation_inv * current_rotation;
            let p_vec = Vector3::new(p.x as f64, p.y as f64, p.z as f64);
            // let deskewed = relative_rotation * (imu_to_lidar * p_vec);
            let deskewed = relative_rotation * p_vec;
            Some(Point3::new(
                deskewed.x as f32,
                deskewed.y as f32,
                deskewed.z as f32,
            ))
        })
        .collect()
}

fn is_valid_point(point: &PointXYZIT, min_dist: f32, max_dist: f32) -> bool {
    if !point.x.is_finite() || !point.y.is_finite() || !point.z.is_finite() {
        return false;
    }

    let dist_sq = point.x * point.x + point.y * point.y + point.z * point.z;
    dist_sq >= min_dist * min_dist && dist_sq <= max_dist * max_dist
}

// fn get_rotation_at_time(imu_data: &[DeltaRotation], timestamp: f64) -> UnitQuaternion<f64> {
//     let mut rotation = UnitQuaternion::<f64>::identity();

//     for delta in imu_data {
//         if delta.timestamp > timestamp {
//             break;
//         }
//         rotation = rotation * delta.delta_rotation; // body-frame ω → right-compose
//     }

//     rotation
// }

fn get_rotation_at_time(traj: &RotationTrajectory, t: f64) -> UnitQuaternion<f64> {
    if traj.is_empty() {
        return UnitQuaternion::identity();
    }
    if t <= traj.first().unwrap().0 {
        return traj.first().unwrap().1;
    }
    if t >= traj.last().unwrap().0 {
        return traj.last().unwrap().1;
    }

    // バイナリサーチ
    let idx = traj.partition_point(|(t0, _)| *t0 < t);
    let (t0, q0) = traj[idx - 1];
    let (t1, q1) = traj[idx];
    let denom = t1 - t0;
    if denom.abs() < 1e-9 {
        return q0;
    }
    let ratio = (t - t0) / denom;
    q0.slerp(&q1, ratio)
}

pub fn get_imu_range(
    imu_data: &Vec<DeltaRotation>,
    frame_time_range: (f64, f64), // (start_time, end_time) sec
) -> (usize, usize) {
    let start_idx = imu_data
        .iter()
        .position(|s| s.timestamp >= frame_time_range.0)
        .unwrap_or(0);

    let end_idx = imu_data
        .iter()
        .position(|s| s.timestamp >= frame_time_range.1)
        .unwrap_or(imu_data.len() - 1);

    (start_idx, end_idx)
}

#[cfg(test)]
mod tests {
    use super::filter_points_by_distance;
    use crate::types::PointXYZIT;

    fn point(x: f32, y: f32, z: f32) -> PointXYZIT {
        PointXYZIT {
            x,
            y,
            z,
            intensity: 0.0,
            timestamp: 0.0,
        }
    }

    #[test]
    fn distance_filter_removes_mid70_zero_and_non_finite_points() {
        let points = vec![
            point(0.0, 0.0, 0.0),
            point(0.5, 0.0, 0.0),
            point(2.0, 0.0, 0.0),
            point(40.0, 0.0, 0.0),
            point(41.0, 0.0, 0.0),
            point(f32::NAN, 0.0, 0.0),
        ];

        let filtered = filter_points_by_distance(&points, 0.5, 40.0);

        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered[0].x, 0.5);
        assert_eq!(filtered[1].x, 2.0);
        assert_eq!(filtered[2].x, 40.0);
    }
}
