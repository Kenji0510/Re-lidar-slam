use nalgebra::{Quaternion, Unit, UnitQuaternion, Vector3};

use crate::types::IMU;

#[derive(Debug, Clone)]
pub struct DeltaRotation {
    pub timestamp: f64,
    pub delta_rotation: Unit<Quaternion<f64>>,
}

pub fn convert_imu_data(imu_data: &Vec<IMU>) -> Vec<DeltaRotation> {
    // let mut current_rotation: Unit<nalgebra::Quaternion<f64>> = UnitQuaternion::<f64>::identity();
    let mut delta_rotations: Vec<DeltaRotation> = Vec::new();
    let mut last_time = imu_data[0].timestamp;

    for (i, sample) in imu_data.iter().enumerate() {
        if i == 0 {
            continue; // Skip the first sample since we don't have a previous timestamp to compare with
        }

        let dt = sample.timestamp - last_time;
        if dt <= 1e-9 {
            continue;
        }

        // --- Update rotation ---
        let omega = Vector3::new(
            sample.angular_velocity[0] as f64,
            sample.angular_velocity[1] as f64,
            sample.angular_velocity[2] as f64,
        );

        let angle = omega.norm() * dt;
        let axis = if angle < 1e-9 {
            Vector3::x_axis() // Default axis if angular velocity is very small
        } else {
            Unit::new_normalize(omega * dt)
        };

        // delta_rotation: body-frame rotation increment (right-compose for body-frame ω)
        let delta_rotation = UnitQuaternion::from_axis_angle(&axis, angle);
        delta_rotations.push(DeltaRotation {
            timestamp: sample.timestamp,
            delta_rotation,
        });

        // Update last_time for the next iteration
        last_time = sample.timestamp;
    }

    delta_rotations
}
