use nalgebra::{Matrix4, Unit, UnitQuaternion, Vector3};

use crate::types::{IMU, LoadIMU};

pub fn align_imu_timestamps(imu_data: &Vec<LoadIMU>) -> Vec<IMU> {
    let mut revised_imu_data = Vec::<IMU>::with_capacity(imu_data.len());

    for imu in imu_data {
        revised_imu_data.push(IMU {
            timestamp: imu.timestamp as f64 / 1_000_000_000.0,
            angular_velocity: imu.angular_velocity,
            linear_acceleration: imu.linear_acceleration,
        });
    }

    revised_imu_data
}

#[derive(Debug, Clone)]
pub struct PosePrediction {
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    // pub rotation: Vector3<f64>,
    pub delta_transform: Matrix4<f64>,
    /// IMUから積分した回転量
    pub delta_rotation: UnitQuaternion<f64>,
}

const G: f64 = 9.80665;

/// `static_gravity_g`: センサ静止時のlinear_acceleration計測値（g単位）。
/// 初期ボディフレーム＝ワールドフレームとして、この方向をワールド重力として固定除去する。
pub fn predict_pose_by_imu(
    imu_data: &Vec<IMU>,
    imu_to_lidar: &UnitQuaternion<f64>,
    prev_pose: &Matrix4<f64>,
    prev_velocity: &Vector3<f64>,
    prev_timestamp: f64,
    current_timestamp: f64,
) -> (Matrix4<f64>, Vector3<f64>) {
    let tx = prev_pose[(0, 3)];
    let ty = prev_pose[(1, 3)];
    let tz = prev_pose[(2, 3)];
    let mut position = Vector3::new(tx, ty, tz);

    let mat3 = prev_pose.fixed_view::<3, 3>(0, 0).into_owned();
    let mut rotation = UnitQuaternion::from_matrix(&mat3);
    let mut velocity = *prev_velocity;

    // LiDARが水平な状態を前提とする
    let gravity = Vector3::new(0.0, 0.0, G);

    // 前回のフレームの終わりから今回のフレームの終わりまでのIMUデータを抽出
    let relevant_imu_data: Vec<&IMU> = imu_data
        .iter()
        .filter(|s| s.timestamp >= prev_timestamp && s.timestamp <= current_timestamp)
        .collect();

    let mut last_time = prev_timestamp;

    for sample in relevant_imu_data {
        let dt = sample.timestamp - last_time;
        if dt <= 1e-9 {
            continue;
        }

        // --- Update rotation ---
        let mut omega = Vector3::new(
            sample.angular_velocity[0] as f64,
            sample.angular_velocity[1] as f64,
            sample.angular_velocity[2] as f64,
        );
        omega = imu_to_lidar * omega;

        let angle = omega.norm() * dt;
        let axis = if angle < 1e-9 {
            Vector3::x_axis() // Default axis if angular velocity is very small
        } else {
            Unit::new_normalize(omega)
        };

        let delta_q = UnitQuaternion::from_axis_angle(&axis, angle);
        rotation = rotation * delta_q;
        rotation.renormalize();

        // --- Update velocity ---
        let acc = Vector3::new(
            sample.linear_acceleration[0] as f64,
            sample.linear_acceleration[1] as f64,
            sample.linear_acceleration[2] as f64,
        );

        // g単位→m/s²変換してワールドフレームへ回転し、重力を除去
        // static_gravity_g はワールドフレームの重力方向（初期ボディ=ワールド座標）
        // let acc_local = imu_to_lidar * acc;
        let acc_local = imu_to_lidar * acc * G;

        let acc_world = rotation * acc_local - gravity;

        // --- Update position and velocity ---
        velocity += acc_world * dt;
        position += velocity * dt + 0.5 * acc_world * dt * dt;

        // Update last_time for the next iteration
        last_time = sample.timestamp;
    }

    let rotation_matrix = rotation.to_rotation_matrix();
    let mut delta_transform = Matrix4::<f64>::identity();
    delta_transform
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&rotation_matrix.matrix());
    delta_transform
        .fixed_view_mut::<3, 1>(0, 3)
        .copy_from(&position);

    (delta_transform, velocity)
}

/// Predicts the next pose without IMU data.
///
/// Translation uses the previous world-frame velocity estimated from ICP.
/// Rotation stays at the latest ICP estimate instead of extrapolating an
/// unverified rotation during consecutive ICP fallbacks.
pub fn predict_pose_by_constant_velocity(
    prev_pose: &Matrix4<f64>,
    prev_velocity: &Vector3<f64>,
    delta_time: f64,
) -> Matrix4<f64> {
    let mut predicted_pose = *prev_pose;

    if !delta_time.is_finite() || delta_time <= 0.0 {
        return predicted_pose;
    }

    predicted_pose[(0, 3)] += prev_velocity.x * delta_time;
    predicted_pose[(1, 3)] += prev_velocity.y * delta_time;
    predicted_pose[(2, 3)] += prev_velocity.z * delta_time;

    predicted_pose
}

pub fn get_imu_range(
    imu_data: &Vec<IMU>,
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

pub type RotationTrajectory = Vec<(f64, UnitQuaternion<f64>)>;

pub fn build_rotation_trajectory(
    imu_data: &Vec<IMU>,
    start_time: f64,
    end_time: f64,
    imu_to_lidar: &UnitQuaternion<f64>,
) -> RotationTrajectory {
    let mut trajectory = Vec::new();
    let mut current_rotation = UnitQuaternion::<f64>::identity();

    let (start_idx, end_idx) = get_imu_range(imu_data, (start_time, end_time));
    let relevant_imu_data: Vec<&IMU> = imu_data[start_idx..end_idx].iter().collect();

    trajectory.push((imu_data[start_idx].timestamp, current_rotation));

    let mut last_time = if start_idx > 0 {
        imu_data[start_idx - 1].timestamp
    } else {
        start_time
    };

    for sample in relevant_imu_data {
        let dt = sample.timestamp - last_time;

        if dt <= 1e-9 {
            continue;
        }

        let omega = Vector3::new(
            sample.angular_velocity[0] as f64,
            sample.angular_velocity[1] as f64,
            sample.angular_velocity[2] as f64,
        );
        let omega = imu_to_lidar * omega;

        let angle_axis = omega * dt;
        let delta_q = UnitQuaternion::new(angle_axis);
        current_rotation = current_rotation * delta_q;
        current_rotation.renormalize();

        trajectory.push((sample.timestamp, current_rotation));
        last_time = sample.timestamp;
    }

    trajectory
}

#[cfg(test)]
mod tests {
    use nalgebra::{Matrix4, Rotation3, Vector3};

    use super::predict_pose_by_constant_velocity;

    #[test]
    fn constant_velocity_prediction_advances_translation_and_preserves_rotation() {
        let rotation = Rotation3::from_euler_angles(0.1, -0.2, 0.3);
        let mut pose = Matrix4::<f64>::identity();
        pose.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(rotation.matrix());
        pose[(0, 3)] = 1.0;
        pose[(1, 3)] = 2.0;
        pose[(2, 3)] = 3.0;

        let predicted =
            predict_pose_by_constant_velocity(&pose, &Vector3::new(2.0, -1.0, 0.5), 0.1);

        assert_eq!(
            predicted.fixed_view::<3, 3>(0, 0),
            pose.fixed_view::<3, 3>(0, 0)
        );
        assert!((predicted[(0, 3)] - 1.2).abs() < 1e-12);
        assert!((predicted[(1, 3)] - 1.9).abs() < 1e-12);
        assert!((predicted[(2, 3)] - 3.05).abs() < 1e-12);
    }

    #[test]
    fn constant_velocity_prediction_ignores_invalid_delta_time() {
        let mut pose = Matrix4::<f64>::identity();
        pose[(0, 3)] = 1.0;
        let velocity = Vector3::new(2.0, 0.0, 0.0);

        assert_eq!(
            predict_pose_by_constant_velocity(&pose, &velocity, -0.1),
            pose
        );
        assert_eq!(
            predict_pose_by_constant_velocity(&pose, &velocity, f64::NAN),
            pose
        );
    }
}
