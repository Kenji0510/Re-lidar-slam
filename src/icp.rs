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
/// H += J^T * J,  b += J^T * e
pub fn build_point_to_plane_system(
    correspondences: &[PointCorrespondence],
    r_mat: &Matrix3<f32>,
    t_vec: &Vector3<f32>,
) -> IcpLinearSystem {
    type Accum = (Matrix6f, Vector6f, f32, usize);

    let (h, b, cost, used_count) = correspondences
        .par_iter()
        .fold(
            || (Matrix6f::zeros(), Vector6f::zeros(), 0.0f32, 0usize),
            |(mut h, mut b, mut cost, mut cnt), corr| {
                let rp = r_mat * corr.src_point.coords;
                let transformed = rp + t_vec;

                let target_pt = if corr.target_cell.is_point {
                    corr.target_cell.point.0
                } else {
                    corr.target_cell.mean
                };

                let residual = corr.plane_normal.dot(&(transformed - target_pt.coords));

                let j_rot = rp.cross(&corr.plane_normal);
                let j_trans = corr.plane_normal;

                let mut j = Vector6f::zeros();
                j[0] = j_rot.x;
                j[1] = j_rot.y;
                j[2] = j_rot.z;
                j[3] = j_trans.x;
                j[4] = j_trans.y;
                j[5] = j_trans.z;

                h += j * j.transpose();
                b -= j * residual;
                cost += residual * residual;
                cnt += 1;
                (h, b, cost, cnt)
            },
        )
        .reduce(
            || (Matrix6f::zeros(), Vector6f::zeros(), 0.0f32, 0usize),
            |(h1, b1, c1, n1): Accum, (h2, b2, c2, n2): Accum| (h1 + h2, b1 + b2, c1 + c2, n1 + n2),
        );

    IcpLinearSystem { h, b, cost, used_count }
}

/// 線形システムを解いて pose 差分 δ = [δθ; δt] を返す。
/// 対応点が 6 未満なら None。
pub fn solve_icp_delta(system: &IcpLinearSystem, damping: f32) -> Option<Vector6f> {
    if system.used_count < 6 {
        return None;
    }

    let mut h = system.h;

    // Levenberg-Marquardt 安定化
    if damping > 0.0 {
        for i in 0..6 {
            h[(i, i)] += damping;
        }
    }

    // Cholesky → LU フォールバック
    if let Some(chol) = h.cholesky() {
        return Some(chol.solve(&system.b));
    }
    h.lu().solve(&system.b)
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
            let target_pt = if corr.target_cell.is_point {
                corr.target_cell.point.0
            } else {
                corr.target_cell.mean
            };
            let transformed = r_mat * corr.src_point.coords + t_vec;
            let e = corr.plane_normal.dot(&(transformed - target_pt.coords));
            e * e
        })
        .sum();

    (sum_sq / correspondences.len() as f32).sqrt()
}
