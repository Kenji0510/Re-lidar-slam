use nalgebra::{Matrix3, Point3, Vector3};
use rayon::prelude::*;

use crate::voxel_map::{VoxelCell, VoxelKey, VoxelMap, voxel_key};

#[derive(Debug, Clone)]
pub struct PlaneFit {
    pub normal: Vector3<f32>,
    pub d: f32,

    pub eigenvalues: Vector3<f32>,
}

impl PlaneFit {
    // 平面: 1に近い
    // 線・縮退形状: 0に近い
    pub fn planarity(&self) -> f32 {
        let lambda0 = self.eigenvalues.x;
        let lambda1 = self.eigenvalues.y;
        let lambda2 = self.eigenvalues.z;

        if !lambda2.is_finite() || lambda2 <= 1e-9 {
            return 0.0;
        }

        ((lambda1 - lambda0) / lambda2).clamp(0.0, 1.0)
    }

    pub fn surface_variation(&self) -> f32 {
        let sum = self.eigenvalues.sum();

        if !sum.is_finite() || sum <= 1e-9 {
            return 1.0;
        }

        (self.eigenvalues.x / sum).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// 平面フィッティング
// ---------------------------------------------------------------------------

/// k 点から最小二乗平面 `normal·p + d = 0` を PCA（共分散行列の固有値分解）で求める。
/// 返り値: (正規化済み法線, d)
/// 3 点未満の場合は None。
// pub fn fit_plane(points: &[Point3<f32>]) -> Option<(Vector3<f32>, f32)> {
//     if points.len() < 3 {
//         return None;
//     }

//     // 重心
//     let centroid = points
//         .iter()
//         .fold(Vector3::zeros(), |acc, p| acc + p.coords)
//         / points.len() as f32;

//     // 共分散行列
//     let mut cov = Matrix3::<f32>::zeros();
//     for p in points {
//         let d = p.coords - centroid;
//         cov += d * d.transpose();
//     }

//     // 固有値分解 — 最小固有値に対応する固有ベクトルが法線
//     let eigen = cov.symmetric_eigen();
//     let min_idx = eigen.eigenvalues.imin();
//     let normal: Vector3<f32> = eigen.eigenvectors.column(min_idx).into_owned();
//     let d = -normal.dot(&centroid);

//     Some((normal, d))
// }

pub fn fit_plane(points: &[Point3<f32>]) -> Option<PlaneFit> {
    if points.len() < 3 {
        return None;
    }

    if points
        .iter()
        .any(|p| p.coords.iter().any(|v| !v.is_finite()))
    {
        return None;
    }

    let point_count = points.len() as f32;

    // 重心
    let centroid = points
        .iter()
        .fold(Vector3::zeros(), |acc, p| acc + p.coords)
        / point_count;

    // 共分散行列
    let mut cov = Matrix3::<f32>::zeros();

    for p in points {
        let diff = p.coords - centroid;
        cov += diff * diff.transpose();
    }

    // 固有値を距離²[m²]として扱えるよう正規化
    cov /= point_count;

    let eigen = cov.symmetric_eigen();

    // symmetric_eigen()の固有値順序は前提にしない
    let min_idx = eigen.eigenvalues.imin();
    let normal: Vector3<f32> = eigen.eigenvectors.column(min_idx).into_owned();

    if normal.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let d = -normal.dot(&centroid);

    // 昇順 λ0 <= λ1 <= λ2 に並べる
    let mut values = [
        eigen.eigenvalues[0].max(0.0),
        eigen.eigenvalues[1].max(0.0),
        eigen.eigenvalues[2].max(0.0),
    ];

    values.sort_by(|a, b| a.total_cmp(b));

    let eigenvalues = Vector3::new(values[0], values[1], values[2]);

    Some(PlaneFit {
        normal,
        d,
        eigenvalues,
    })
}

/// k 点すべてが平面 `normal·p + d = 0` から `threshold` 以内にあるか検証する。
/// 法線は正規化済みであること。
pub fn check_points_on_plane(
    normal: &Vector3<f32>,
    d: f32,
    points: &[Point3<f32>],
    plane_point_distance_threshold_m: f32,
) -> bool {
    if points.len() < 3
        || !plane_point_distance_threshold_m.is_finite()
        || plane_point_distance_threshold_m <= 0.0
    {
        return false;
    }

    points.iter().all(|p| {
        let point_to_plane_distance = (normal.dot(&p.coords) + d).abs();

        point_to_plane_distance <= plane_point_distance_threshold_m
    })
}

// Source点と推定平面との絶対距離を判定する。
pub fn check_source_to_plane_absolute_distance(
    normal: &Vector3<f32>,
    d: f32,
    world_point: &Point3<f32>,
    max_distance_m: f32,
) -> bool {
    if !max_distance_m.is_finite() || max_distance_m <= 0.0 {
        return false;
    }

    let point_to_plane_distance = (normal.dot(&world_point.coords) + d).abs();

    point_to_plane_distance <= max_distance_m
}

// Fast-LIO2 スタイルの「source 点が平面に十分近いか」判定。
//
// ```text
// pd2 = normal·world_point + d        (ワールド座標での符号付き距離)
// s   = 1 - 0.9 * |pd2| / sqrt(sensor_dist)
// 有効: s > 0.9
// ```
// `sensor_dist`: センサ原点からの距離 (= src_point.coords.norm())
pub fn check_source_on_plane(
    normal: &Vector3<f32>,
    d: f32,
    world_point: &Point3<f32>,
    sensor_dist: f32,
    plane_fit_threshold: f32,
) -> bool {
    let pd2 = normal.dot(&world_point.coords) + d;
    let s = 1.0 - 0.9 * pd2.abs() / sensor_dist.sqrt().max(1e-6);
    s > plane_fit_threshold
}

// ---------------------------------------------------------------------------
// 対応点
// ---------------------------------------------------------------------------

/// Point-to-Plane ICP で使う1対応点。
/// - `src_point`: source の実点座標
/// - `target_cell`: k近傍中の**最近傍**セル（対応点）
/// - `plane_normal` / `plane_d`: k近傍 mean から求めた平面
pub struct PointCorrespondence<'a> {
    pub src_key: VoxelKey,
    pub src_point: Point3<f32>,
    pub target_cell: &'a VoxelCell,
    pub plane_normal: Vector3<f32>,
    pub plane_d: f32,
}

/// query_point に近い順に最大 `k` 個の target_map セルを返す。
/// 返り値: Vec<(&VoxelCell, f32)> — (セル参照, 距離の二乗)
fn find_k_nearest_target_voxels<'a>(
    query_point: &Point3<f32>,
    target_map: &'a VoxelMap,
    voxel_size: f32,
    search_range: i32,
    max_dist_sq: f32,
    k: usize,
) -> Vec<(&'a VoxelCell, f32)> {
    let base_key = voxel_key(query_point, voxel_size);

    // (dist_sq, cell) を収集してから距離順ソート
    let mut candidates: Vec<(&VoxelCell, f32)> = Vec::new();

    for dz in -search_range..=search_range {
        for dy in -search_range..=search_range {
            for dx in -search_range..=search_range {
                let key = VoxelKey {
                    ix: base_key.ix + dx,
                    iy: base_key.iy + dy,
                    iz: base_key.iz + dz,
                };

                let Some(target_cell) = target_map.get(&key) else {
                    continue;
                };

                let diff = query_point.coords - target_cell.mean.coords;
                let dist_sq = diff.dot(&diff);

                if dist_sq < max_dist_sq {
                    candidates.push((target_cell, dist_sq));
                }
            }
        }
    }

    candidates.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    candidates.truncate(k);
    candidates
}

/// source_map の各点について target_map から k 近傍セルを探し、
/// 以下の条件をすべて満たす場合のみ対応点として返す:
///   1. 近傍点がちょうど k 個見つかった
///   2. k 番目（最遠）の距離が `target_voxel_size * max_dist_factor` 以内
///   3. k 点が平面を形成できる (`plane_fit_threshold`)
///   4. source 点がその平面に十分近い (Fast-LIO2 基準 s > 0.9)
///
/// **座標系**: source 点はセンサローカル座標、target_map はワールド座標。
/// `r_mat` / `t_vec` は現在の ICP 推定姿勢で、対応点探索時に source を
/// ワールド座標に変換するために使う。
/// 返り値の `src_point` は**ローカル座標**（ICP ヤコビアン計算用）。
pub fn pickup_valid_source_points<'a>(
    source_map: &VoxelMap,
    target_map: &'a VoxelMap,
    target_voxel_size: f32,
    search_range: i32,
    k: usize,
    max_dist_factor: f32,
    plane_point_distance_threshold: f32,
    source_plane_score_threshold: f32,
    source_to_plane_max_distance_m: Option<f32>,
    min_planarity: Option<f32>,
    r_mat: &Matrix3<f32>,
    t_vec: &Vector3<f32>,
) -> Vec<PointCorrespondence<'a>> {
    let max_neighbor_dist_sq = (target_voxel_size * max_dist_factor).powi(2);

    source_map
        .par_iter()
        .filter_map(|(src_key, src_cell)| {
            let src_point = src_cell.point.0;

            // ローカル座標 → ワールド座標（現在の (R,t) 推定値を使用）
            let query_point = Point3::from(r_mat * src_point.coords + t_vec);

            let neighbors = find_k_nearest_target_voxels(
                &query_point,
                target_map,
                target_voxel_size,
                search_range,
                max_neighbor_dist_sq,
                k,
            );

            // 条件1: k 個揃っているか
            if neighbors.len() < k {
                return None;
            }

            // 条件2: k 番目（最遠）が遠すぎないか
            if neighbors.last().unwrap().1 > max_neighbor_dist_sq {
                return None;
            }

            // 条件3 & 4: 平面フィッティング（実点を優先、なければボクセル中心）
            let neighbor_points: Vec<Point3<f32>> = neighbors
                .iter()
                .map(|(c, _)| if c.is_point { c.point.0 } else { c.mean })
                .collect();

            let plane_fit = fit_plane(&neighbor_points)?;

            if let Some(min_planarity) = min_planarity {
                if !min_planarity.is_finite()
                    || min_planarity < 0.0
                    || min_planarity > 1.0
                    || plane_fit.planarity() < min_planarity
                {
                    return None;
                }
            }

            if !check_points_on_plane(
                &plane_fit.normal,
                plane_fit.d,
                &neighbor_points,
                plane_point_distance_threshold,
            ) {
                return None;
            }

            if let Some(max_distance_m) = source_to_plane_max_distance_m {
                if !check_source_to_plane_absolute_distance(
                    &plane_fit.normal,
                    plane_fit.d,
                    &query_point,
                    max_distance_m,
                ) {
                    return None;
                }
            }

            // センサ原点からの距離でスケールした閾値（ローカル座標の norm を使用）
            let sensor_dist = src_point.coords.norm();
            if !check_source_on_plane(
                &plane_fit.normal,
                plane_fit.d,
                &query_point,
                sensor_dist,
                source_plane_score_threshold,
            ) {
                return None;
            }

            // neighbors はソート済み → [0] が最近傍対応点
            Some(PointCorrespondence {
                src_key: *src_key,
                src_point,
                target_cell: neighbors[0].0,
                plane_normal: plane_fit.normal,
                plane_d: plane_fit.d,
            })
        })
        .collect()
}
