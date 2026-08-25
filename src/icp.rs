use nalgebra::{Matrix3, SMatrix, SVector, UnitQuaternion, Vector3};
use rayon::prelude::*;

use crate::find_nearest_points::PointCorrespondence;

pub type Matrix6f = SMatrix<f32, 6, 6>;
pub type Vector6f = SVector<f32, 6>;

// ---------------------------------------------------------------------------
// 線形システム
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct IcpLinearSystem {
    pub h: Matrix6f,
    pub b: Vector6f,
    /// Σ residual²
    pub cost: f32,
    /// 使用された対応点数
    pub used_count: usize,
}

#[derive(Debug, Clone)]
pub struct IcpSolveResult {
    pub delta: Vector6f,
    /// Number of observable directions retained from the normalized Hessian.
    pub observable_rank: usize,
    /// Smallest retained eigenvalue divided by the largest eigenvalue.
    pub min_observable_eigenvalue_ratio: f32,
}

impl Default for IcpLinearSystem {
    fn default() -> Self {
        Self {
            h: Matrix6f::zeros(),
            b: Vector6f::zeros(),
            cost: 0.0,
            used_count: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Point-to-Plane ICP
// ---------------------------------------------------------------------------

/// 現在の (R, t) で source 点を変換し、Point-to-Plane 線形システムを構築する。
///
/// 残差: e = n^T * (R*p_s + t - p_t)
/// ヤコビアン: J = [(R*p_s × n)^T,  n^T]  (1×6)
/// H += J^T * J,  b -= J^T * e
pub fn build_point_to_plane_system(
    correspondences: &[PointCorrespondence],
    r_mat: &Matrix3<f32>,
    t_vec: &Vector3<f32>,
) -> IcpLinearSystem {
    build_robust_point_to_plane_system(correspondences, r_mat, t_vec, f32::INFINITY)
}

/// Build a Point-to-Plane system with a Huber M-estimator.
///
/// `huber_delta_m` is the transition distance in metres. Passing infinity
/// preserves the unweighted least-squares behaviour.
pub fn build_robust_point_to_plane_system(
    correspondences: &[PointCorrespondence],
    r_mat: &Matrix3<f32>,
    t_vec: &Vector3<f32>,
    huber_delta_m: f32,
) -> IcpLinearSystem {
    type Accum = (Matrix6f, Vector6f, f32, usize);

    let (mut h, b, cost, used_count) = correspondences
        .par_iter()
        .fold(
            || (Matrix6f::zeros(), Vector6f::zeros(), 0.0f32, 0usize),
            |(mut h, mut b, mut cost, mut cnt), corr| {
                let rp = r_mat * corr.src_point.coords;
                let transformed = rp + t_vec;

                let residual = corr.plane_normal.dot(&transformed) + corr.plane_d;
                let (weight, loss) = huber_weight_and_loss(residual, huber_delta_m);

                let j_rot = rp.cross(&corr.plane_normal);
                let j_trans = corr.plane_normal;
                let j = [j_rot.x, j_rot.y, j_rot.z, j_trans.x, j_trans.y, j_trans.z];

                // H は対称なので上三角21要素だけを直接累積する。
                // 汎用の 6x1 * 1x6 外積と一時行列の生成を避ける。
                for row in 0..6 {
                    b[row] -= weight * j[row] * residual;
                    for col in row..6 {
                        h[(row, col)] += weight * j[row] * j[col];
                    }
                }
                cost += loss;
                cnt += 1;
                (h, b, cost, cnt)
            },
        )
        .reduce(
            || (Matrix6f::zeros(), Vector6f::zeros(), 0.0f32, 0usize),
            |(h1, b1, c1, n1): Accum, (h2, b2, c2, n2): Accum| (h1 + h2, b1 + b2, c1 + c2, n1 + n2),
        );

    for row in 1..6 {
        for col in 0..row {
            h[(row, col)] = h[(col, row)];
        }
    }

    IcpLinearSystem {
        h,
        b,
        cost,
        used_count,
    }
}

#[inline]
fn huber_weight_and_loss(residual: f32, delta: f32) -> (f32, f32) {
    let abs_residual = residual.abs();
    if !delta.is_finite() || abs_residual <= delta {
        (1.0, residual * residual)
    } else if delta > 0.0 && abs_residual.is_finite() {
        (
            delta / abs_residual.max(f32::EPSILON),
            2.0 * delta * abs_residual - delta * delta,
        )
    } else {
        (0.0, f32::INFINITY)
    }
}

#[inline]
fn huber_loss(residual: f32, delta: f32) -> f32 {
    huber_weight_and_loss(residual, delta).1
}

/// 線形システムを解いて pose 差分 δ = [δθ; δt] を返す。
/// 対応点が 6 未満なら None。
pub fn solve_icp_delta(system: &IcpLinearSystem, damping: f32) -> Option<Vector6f> {
    solve_icp_delta_observable(system, damping, 0.0).map(|result| result.delta)
}

/// Solve ICP in normalized Hessian coordinates and discard weak eigen-directions.
///
/// Rotation and translation have different units, so the rotation block is
/// scaled by the point cloud's characteristic lever arm. Eigenvectors whose
/// eigenvalue is less than `relative_eigenvalue_threshold * max_eigenvalue`
/// are treated as unobservable.
/// Because ICP starts from the motion prediction, a zero update in those
/// directions preserves the predicted pose instead of allowing tangent drift.
pub fn solve_icp_delta_observable(
    system: &IcpLinearSystem,
    damping: f32,
    relative_eigenvalue_threshold: f32,
) -> Option<IcpSolveResult> {
    if system.used_count < 6
        || !damping.is_finite()
        || damping < 0.0
        || !relative_eigenvalue_threshold.is_finite()
        || !(0.0..1.0).contains(&relative_eigenvalue_threshold)
        || system.h.iter().any(|value| !value.is_finite())
        || system.b.iter().any(|value| !value.is_finite())
    {
        return None;
    }

    let rotation_diagonal_sum = (0..3).map(|i| system.h[(i, i)].max(0.0)).sum::<f32>();
    let translation_diagonal_sum = (3..6).map(|i| system.h[(i, i)].max(0.0)).sum::<f32>();
    if !rotation_diagonal_sum.is_finite()
        || !translation_diagonal_sum.is_finite()
        || rotation_diagonal_sum + translation_diagonal_sum <= f32::EPSILON
    {
        return None;
    }

    // q = S*x with q_rot expressed as an equivalent displacement at the
    // cloud's characteristic lever arm. A single scale per unit block keeps
    // weak geometric directions weak; normalizing every diagonal separately
    // would incorrectly make an unobservable direction look fully constrained.
    let characteristic_length_m =
        if translation_diagonal_sum > f32::EPSILON && rotation_diagonal_sum > f32::EPSILON {
            (rotation_diagonal_sum / translation_diagonal_sum)
                .sqrt()
                .clamp(0.1, 100.0)
        } else {
            1.0
        };
    let scales = Vector6f::new(
        characteristic_length_m,
        characteristic_length_m,
        characteristic_length_m,
        1.0,
        1.0,
        1.0,
    );

    let mut normalized_h = Matrix6f::zeros();
    let mut normalized_b = Vector6f::zeros();
    for row in 0..6 {
        normalized_b[row] = system.b[row] / scales[row];
        for col in 0..6 {
            normalized_h[(row, col)] = system.h[(row, col)] / (scales[row] * scales[col]);
        }
    }
    normalized_h = (normalized_h + normalized_h.transpose()) * 0.5;

    let eigen = normalized_h.symmetric_eigen();
    let max_eigenvalue = eigen.eigenvalues.max().max(0.0);
    if !max_eigenvalue.is_finite() || max_eigenvalue <= f32::EPSILON {
        return None;
    }

    let cutoff = (max_eigenvalue * relative_eigenvalue_threshold).max(f32::EPSILON);
    let mut normalized_delta = Vector6f::zeros();
    let mut observable_rank = 0usize;
    let mut min_retained = max_eigenvalue;

    for i in 0..6 {
        let eigenvalue = eigen.eigenvalues[i];
        if !eigenvalue.is_finite() || eigenvalue < cutoff {
            continue;
        }

        let direction = eigen.eigenvectors.column(i);
        let projected_b = direction.dot(&normalized_b);
        normalized_delta += direction * (projected_b / (eigenvalue + damping));
        observable_rank += 1;
        min_retained = min_retained.min(eigenvalue);
    }

    if observable_rank == 0 {
        return None;
    }

    let mut delta = Vector6f::zeros();
    for i in 0..6 {
        delta[i] = normalized_delta[i] / scales[i];
    }
    if delta.iter().any(|value| !value.is_finite()) {
        return None;
    }

    Some(IcpSolveResult {
        delta,
        observable_rank,
        min_observable_eigenvalue_ratio: min_retained / max_eigenvalue,
    })
}

/// delta から (R, t) を更新する。
/// delta[0..3]: 回転（軸×角 スケール付き）, delta[3..6]: 並進
pub fn apply_delta(
    r_mat: &Matrix3<f32>,
    t_vec: &Vector3<f32>,
    delta: &Vector6f,
) -> (Matrix3<f32>, Vector3<f32>) {
    let d_rot = Vector3::new(delta[0], delta[1], delta[2]);
    let d_trans = Vector3::new(delta[3], delta[4], delta[5]);

    let r_delta = UnitQuaternion::from_scaled_axis(d_rot)
        .to_rotation_matrix()
        .into_inner();

    let new_r = r_delta * r_mat;
    let new_t = t_vec + d_trans;

    (new_r, new_t)
}

/// Point-to-Plane RMSE を計算する。
/// RMSE = sqrt( Σ e² / n ),  e = n^T * (R*p_s + t - p_t)
pub fn compute_rmse(
    correspondences: &[PointCorrespondence],
    r_mat: &Matrix3<f32>,
    t_vec: &Vector3<f32>,
) -> f32 {
    if correspondences.is_empty() {
        return f32::INFINITY;
    }

    let sum_sq: f32 = correspondences
        .iter()
        .map(|corr| {
            let transformed = r_mat * corr.src_point.coords + t_vec;
            let e = corr.plane_normal.dot(&transformed) + corr.plane_d;
            e * e
        })
        .sum();

    (sum_sq / correspondences.len() as f32).sqrt()
}

pub fn compute_robust_cost(
    correspondences: &[PointCorrespondence],
    r_mat: &Matrix3<f32>,
    t_vec: &Vector3<f32>,
    huber_delta_m: f32,
) -> f32 {
    correspondences
        .iter()
        .map(|corr| {
            let transformed = r_mat * corr.src_point.coords + t_vec;
            let residual = corr.plane_normal.dot(&transformed) + corr.plane_d;
            huber_loss(residual, huber_delta_m)
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_reference_robust_system(
        correspondences: &[PointCorrespondence],
        r_mat: &Matrix3<f32>,
        t_vec: &Vector3<f32>,
        huber_delta_m: f32,
    ) -> IcpLinearSystem {
        let mut system = IcpLinearSystem::default();
        for corr in correspondences {
            let rp = r_mat * corr.src_point.coords;
            let residual = corr.plane_normal.dot(&(rp + t_vec)) + corr.plane_d;
            let (weight, loss) = huber_weight_and_loss(residual, huber_delta_m);
            let j_rot = rp.cross(&corr.plane_normal);
            let j = Vector6f::new(
                j_rot.x,
                j_rot.y,
                j_rot.z,
                corr.plane_normal.x,
                corr.plane_normal.y,
                corr.plane_normal.z,
            );
            system.h += weight * (j * j.transpose());
            system.b -= weight * j * residual;
            system.cost += loss;
            system.used_count += 1;
        }
        system
    }

    fn assert_close(actual: f32, expected: f32) {
        let tolerance = 2e-5 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual}, expected={expected}, tolerance={tolerance}",
        );
    }

    fn diagonal_system(diagonal: [f32; 6], b: [f32; 6]) -> IcpLinearSystem {
        let mut h = Matrix6f::zeros();
        for (index, value) in diagonal.into_iter().enumerate() {
            h[(index, index)] = value;
        }
        IcpLinearSystem {
            h,
            b: Vector6f::from_row_slice(&b),
            cost: 0.0,
            used_count: 100,
        }
    }

    fn system_from_jacobians(jacobians: &[Vector6f], desired_delta: Vector6f) -> IcpLinearSystem {
        let h = jacobians
            .iter()
            .fold(Matrix6f::zeros(), |sum, j| sum + j * j.transpose());
        IcpLinearSystem {
            h,
            b: h * desired_delta,
            cost: 0.0,
            used_count: jacobians.len().max(6),
        }
    }

    #[test]
    fn observable_solver_keeps_only_wall_normal_translation() {
        let system = diagonal_system(
            [0.0, 0.0, 0.0, 100.0, 1e-4, 1e-4],
            [0.0, 0.0, 0.0, 10.0, 1.0, -1.0],
        );
        let result = solve_icp_delta_observable(&system, 1e-6, 0.01).unwrap();

        assert_eq!(result.observable_rank, 1);
        assert!((result.delta[3] - 0.1).abs() < 1e-5);
        assert_eq!(result.delta[4], 0.0);
        assert_eq!(result.delta[5], 0.0);
    }

    #[test]
    fn observable_solver_keeps_three_independent_translation_directions() {
        let system = diagonal_system(
            [0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
            [0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
        );
        let result = solve_icp_delta_observable(&system, 1e-6, 0.01).unwrap();

        assert_eq!(result.observable_rank, 3);
        assert!((result.delta[3] - 0.1).abs() < 1e-5);
        assert!((result.delta[4] - 0.2).abs() < 1e-5);
        assert!((result.delta[5] - 0.3).abs() < 1e-5);
    }

    #[test]
    fn floor_geometry_preserves_unobservable_xy_and_yaw_prediction() {
        let normal = Vector3::z();
        let jacobians: Vec<_> = (-2..=2)
            .flat_map(|x| {
                (-2..=2).map(move |y| {
                    let point = Vector3::new(x as f32, y as f32, 0.0);
                    let rotation = point.cross(&normal);
                    Vector6f::new(rotation.x, rotation.y, rotation.z, 0.0, 0.0, 1.0)
                })
            })
            .collect();
        let desired = Vector6f::new(0.01, -0.02, 0.5, 1.0, -1.0, 0.1);
        let result =
            solve_icp_delta_observable(&system_from_jacobians(&jacobians, desired), 1e-6, 0.01)
                .unwrap();

        assert_eq!(result.observable_rank, 3);
        assert!(result.delta[2].abs() < 1e-6);
        assert!(result.delta[3].abs() < 1e-6);
        assert!(result.delta[4].abs() < 1e-6);
        assert!((result.delta[5] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn wall_geometry_preserves_tangent_translation_prediction() {
        let normal = Vector3::x();
        let jacobians: Vec<_> = (-2..=2)
            .flat_map(|y| {
                (-2..=2).map(move |z| {
                    let point = Vector3::new(0.0, y as f32, z as f32);
                    let rotation = point.cross(&normal);
                    Vector6f::new(rotation.x, rotation.y, rotation.z, 1.0, 0.0, 0.0)
                })
            })
            .collect();
        let desired = Vector6f::new(0.5, 0.01, -0.02, 0.1, 1.0, -1.0);
        let result =
            solve_icp_delta_observable(&system_from_jacobians(&jacobians, desired), 1e-6, 0.01)
                .unwrap();

        assert_eq!(result.observable_rank, 3);
        assert!(result.delta[0].abs() < 1e-6);
        assert!((result.delta[3] - 0.1).abs() < 1e-5);
        assert!(result.delta[4].abs() < 1e-6);
        assert!(result.delta[5].abs() < 1e-6);
    }

    #[test]
    fn huber_weight_limits_large_outliers() {
        assert_eq!(huber_weight_and_loss(0.05, 0.1).0, 1.0);
        assert!((huber_weight_and_loss(1.0, 0.1).0 - 0.1).abs() < 1e-6);
        assert!(huber_loss(1.0, 0.1) < 0.5);
    }

    #[test]
    fn triangular_accumulation_matches_full_outer_product() {
        let correspondences: Vec<_> = (0..257)
            .map(|index| {
                let phase = index as f32 * 0.037;
                let raw_normal = Vector3::new(
                    0.3 + phase.sin().abs(),
                    0.4 + (phase * 1.7).cos().abs(),
                    0.5 + (phase * 0.7).sin().abs(),
                );
                PointCorrespondence {
                    src_point: nalgebra::Point3::new(
                        phase.sin() * 8.0,
                        phase.cos() * 5.0,
                        phase * 0.2 - 1.0,
                    ),
                    plane_normal: raw_normal.normalize(),
                    plane_d: (phase * 0.3).sin() * 0.2,
                }
            })
            .collect();
        let r_mat = UnitQuaternion::from_euler_angles(0.08, -0.04, 0.12)
            .to_rotation_matrix()
            .into_inner();
        let t_vec = Vector3::new(0.13, -0.07, 0.04);

        let actual = build_robust_point_to_plane_system(&correspondences, &r_mat, &t_vec, 0.08);
        let expected = build_reference_robust_system(&correspondences, &r_mat, &t_vec, 0.08);

        assert_eq!(actual.used_count, expected.used_count);
        assert_close(actual.cost, expected.cost);
        for (actual, expected) in actual.h.iter().zip(expected.h.iter()) {
            assert_close(*actual, *expected);
        }
        for (actual, expected) in actual.b.iter().zip(expected.b.iter()) {
            assert_close(*actual, *expected);
        }
    }
}
