use std::{
    collections::HashMap,
    env,
    fs,
    path::Path,
    time::Instant,
};

use anyhow::{anyhow, bail, Context, Result};
use nalgebra::{Matrix3, Point3, SymmetricEigen, Vector3};
use pcd_rs::Reader;
use rayon::prelude::*;

use re_lidar_slam::{
    file_handler::save_pcd_xyz,
    types::PointXYZ,
};

/// 完成済みのグローバルマップPCDに対して、局所RANSAC平面フィルタを適用する。
///
/// 配置例:
///     src/bin/filter_world_map_ransac.rs
///
/// 実行例:
///     cargo run --release --bin filter_world_map_ransac -- \
///         data/output/debug/07122026/voxel-0.1_world_map.pcd \
///         data/output/debug/07122026/voxel-0.1_world_map_ransac.pcd
///
/// パラメータ指定例:
///     cargo run --release --bin filter_world_map_ransac -- \
///         input.pcd output.pcd \
///         --voxel-size 0.20 \
///         --neighbor-range 1 \
///         --fit-distance-threshold 0.060 \
        //  --keep-distance-threshold 0.015 \
///         --iterations 64 \
///         --min-neighbors 12 \
///         --min-inliers 8 \
///         --min-inlier-ratio 0.45 \
///         --max-planarity-ratio 0.15 \
///         --min-planar-spread 0.03 \
///         --max-planes 2 \
///         --removed-output removed_points.pcd
///
/// デフォルトでは、局所平面が成立しなかった領域の点は残す。
/// 非平面領域も削除して「平面インライアだけ」にしたい場合:
///     --remove-nonplanar
#[derive(Debug, Clone)]
struct FilterConfig {
    /// RANSAC処理単位となるボクセルサイズ [m]。
    ///
    /// 元のグローバルマップが0.1 m間隔でも、ここを0.2～0.3 mにすると
    /// 1セル内に複数点が入り、セル単位でRANSACできるため高速になる。
    voxel_size: f32,

    /// 対象セルの周囲何セルまでRANSAC点群に含めるか。
    ///
    /// voxel_size=0.20, neighbor_range=1 の場合、
    /// 3×3×3セル、およそ0.6 m幅の局所領域を使用する。
    neighbor_range: i32,

    /// RANSACで粗い平面を検出するときの距離しきい値 [m]。
    ///
    /// 太くなった壁でも平面モデル自体を発見できるよう、最終保持しきい値より
    /// 大きく設定する。例: 0.05～0.08 m。
    fit_distance_threshold: f32,

    /// 検出・PCA再フィット済み平面から、この距離以内の点だけを最終保持 [m]。
    ///
    /// 壁の最終的な半厚みに相当する。例: 0.01～0.02 m。
    keep_distance_threshold: f32,

    /// 1平面あたりのRANSAC最大反復回数。
    max_iterations: usize,

    /// RANSACを実行するために必要な局所点数。
    min_neighbor_points: usize,

    /// 有効平面として必要な最低インライア数。
    min_inliers: usize,

    /// 有効平面として必要な最低インライア率。
    ///
    /// 2平面が混ざる壁・床境界も考慮し、デフォルトは0.45。
    min_inlier_ratio: f32,

    /// PCA固有値による平面性判定:
    ///
    ///     lambda_min / lambda_mid <= max_planarity_ratio
    ///
    /// 小さいほど厳しい。
    max_planarity_ratio: f32,

    /// 平面内の第2方向の標準偏差 sqrt(lambda_mid) の最低値 [m]。
    ///
    /// 点が線状・一点集中している場合を平面として採用しないために使う。
    min_planar_spread: f32,

    /// 1局所領域から抽出する最大平面数。
    ///
    /// 2にすると、壁＋床のような複数面を残しやすい。
    max_planes_per_neighborhood: usize,

    /// 局所平面が一つも成立しなかった領域を残すか。
    ///
    /// true:
    ///     木、柱、曲面、角などの非平面構造を残す安全側の設定。
    ///
    /// false:
    ///     有効な平面インライアだけを保存する厳しい設定。
    keep_nonplanar_regions: bool,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            voxel_size: 0.20,
            neighbor_range: 1,
            fit_distance_threshold: 0.060,
            keep_distance_threshold: 0.015,
            max_iterations: 64,
            min_neighbor_points: 12,
            min_inliers: 8,
            min_inlier_ratio: 0.45,
            max_planarity_ratio: 0.15,
            min_planar_spread: 0.03,
            max_planes_per_neighborhood: 2,
            keep_nonplanar_regions: true,
        }
    }
}

#[derive(Debug)]
struct CliArgs {
    input_path: String,
    output_path: String,
    removed_output_path: Option<String>,
    config: FilterConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct VoxelKey {
    ix: i32,
    iy: i32,
    iz: i32,
}

type VoxelGrid = HashMap<VoxelKey, Vec<usize>>;

#[derive(Debug, Clone)]
struct PlaneModel {
    center: Point3<f32>,
    normal: Vector3<f32>,

    lambda_min: f32,
    lambda_mid: f32,
    lambda_max: f32,

    inlier_count: usize,
    inlier_ratio: f32,
}

impl PlaneModel {
    #[inline]
    fn distance(&self, point: &Point3<f32>) -> f32 {
        self.normal
            .dot(&(point.coords - self.center.coords))
            .abs()
    }
}

#[derive(Debug, Clone, Copy)]
enum PointDecision {
    /// 有効な局所平面のインライア。
    KeepPlaneInlier,

    /// 周囲の点数が不足しているため、判定せず残す。
    KeepSparse,

    /// 周囲は非平面領域だったため、安全側で残す。
    KeepNonPlanar,

    /// 有効平面があるが、どの平面にも属さないため削除。
    RemovePlaneOutlier,

    /// --remove-nonplanar 指定により、非平面領域なので削除。
    RemoveNonPlanar,
}

impl PointDecision {
    #[inline]
    fn keep(self) -> bool {
        matches!(
            self,
            PointDecision::KeepPlaneInlier
                | PointDecision::KeepSparse
                | PointDecision::KeepNonPlanar
        )
    }
}

#[derive(Debug)]
struct CellResult {
    decisions: Vec<(usize, PointDecision)>,
    extracted_plane_count: usize,
}

#[derive(Debug, Default)]
struct FilterStats {
    input_points: usize,
    occupied_voxels: usize,

    kept_plane_inliers: usize,
    kept_sparse: usize,
    kept_nonplanar: usize,

    removed_plane_outliers: usize,
    removed_nonplanar: usize,

    local_planes: usize,
}

impl FilterStats {
    fn kept_total(&self) -> usize {
        self.kept_plane_inliers + self.kept_sparse + self.kept_nonplanar
    }

    fn removed_total(&self) -> usize {
        self.removed_plane_outliers + self.removed_nonplanar
    }
}

/// 追加crateを増やさず、結果を再現可能にするための簡易乱数生成器。
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        };
        Self { state }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    #[inline]
    fn index(&mut self, upper: usize) -> usize {
        debug_assert!(upper > 0);
        (self.next_u64() % upper as u64) as usize
    }
}

fn main() -> Result<()> {
    let args = parse_args()?;
    validate_config(&args.config)?;

    println!("Input : {}", args.input_path);
    println!("Output: {}", args.output_path);
    if let Some(path) = &args.removed_output_path {
        println!("Removed points: {path}");
    }
    println!("Config: {:#?}", args.config);

    let total_start = Instant::now();

    let load_start = Instant::now();
    let loaded_points = load_pcd_xyz(&args.input_path)
        .with_context(|| format!("Failed to load input PCD: {}", args.input_path))?;

    let original_count = loaded_points.len();

    // NaN/Infを含む点はRANSAC計算を壊すため除外する。
    let points_xyz: Vec<PointXYZ> = loaded_points
        .into_iter()
        .filter(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite())
        .collect();

    let invalid_count = original_count - points_xyz.len();

    if points_xyz.is_empty() {
        bail!("Input PCD contains no finite XYZ points");
    }

    let points: Vec<Point3<f32>> = points_xyz
        .iter()
        .map(|p| Point3::new(p.x, p.y, p.z))
        .collect();

    println!(
        "Loaded {} finite points in {:.3?} (discarded non-finite: {})",
        points.len(),
        load_start.elapsed(),
        invalid_count
    );

    let grid_start = Instant::now();
    let grid = build_voxel_grid(&points, args.config.voxel_size);
    println!(
        "Built voxel grid: {} occupied cells in {:.3?}",
        grid.len(),
        grid_start.elapsed()
    );

    let filter_start = Instant::now();
    let (decisions, mut stats) = filter_points(&points, &grid, &args.config);
    stats.input_points = points.len();
    stats.occupied_voxels = grid.len();

    println!(
        "RANSAC filtering completed in {:.3?}",
        filter_start.elapsed()
    );

    let mut kept_points = Vec::with_capacity(stats.kept_total());
    let mut removed_points = Vec::with_capacity(stats.removed_total());

    for (point, decision) in points_xyz.into_iter().zip(decisions.into_iter()) {
        if decision.keep() {
            kept_points.push(point);
        } else {
            removed_points.push(point);
        }
    }

    ensure_parent_dir(&args.output_path)?;
    save_pcd_xyz(&kept_points, &args.output_path)
        .with_context(|| format!("Failed to save filtered PCD: {}", args.output_path))?;

    if let Some(path) = &args.removed_output_path {
        ensure_parent_dir(path)?;
        save_pcd_xyz(&removed_points, path)
            .with_context(|| format!("Failed to save removed-point PCD: {path}"))?;
    }

    println!();
    println!("========== Filter summary ==========");
    println!("Input finite points       : {}", stats.input_points);
    println!("Occupied filter voxels    : {}", stats.occupied_voxels);
    println!("Extracted local planes    : {}", stats.local_planes);
    println!("Kept plane inliers        : {}", stats.kept_plane_inliers);
    println!("Kept sparse points        : {}", stats.kept_sparse);
    println!("Kept non-planar points    : {}", stats.kept_nonplanar);
    println!("Removed plane outliers    : {}", stats.removed_plane_outliers);
    println!("Removed non-planar points : {}", stats.removed_nonplanar);
    println!("Kept total                : {}", stats.kept_total());
    println!("Removed total             : {}", stats.removed_total());
    println!(
        "Retention ratio           : {:.2}%",
        stats.kept_total() as f64 / stats.input_points as f64 * 100.0
    );
    println!("Total elapsed             : {:.3?}", total_start.elapsed());
    println!("Saved filtered map        : {}", args.output_path);

    Ok(())
}

/// アップロードされた load_pcd_xyzit() と同じ形式で、XYZ PCDをロードする。
fn load_pcd_xyz(file_path: &str) -> Result<Vec<PointXYZ>> {
    let reader = Reader::open(file_path)
        .map_err(|e| anyhow!("Failed to open PCD file '{}': {}", file_path, e))?;

    let points: Vec<PointXYZ> = reader
        .collect::<Result<Vec<PointXYZ>, _>>()
        .map_err(|e| anyhow!("Failed to read PCD data from '{}': {}", file_path, e))?;

    Ok(points)
}

fn ensure_parent_dir(file_path: &str) -> Result<()> {
    let path = Path::new(file_path);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create output directory: {}", parent.display())
            })?;
        }
    }
    Ok(())
}

fn build_voxel_grid(points: &[Point3<f32>], voxel_size: f32) -> VoxelGrid {
    let mut grid = VoxelGrid::with_capacity(points.len());

    for (index, point) in points.iter().enumerate() {
        grid.entry(voxel_key(point, voxel_size))
            .or_default()
            .push(index);
    }

    grid
}

#[inline]
fn voxel_key(point: &Point3<f32>, voxel_size: f32) -> VoxelKey {
    VoxelKey {
        ix: (point.x / voxel_size).floor() as i32,
        iy: (point.y / voxel_size).floor() as i32,
        iz: (point.z / voxel_size).floor() as i32,
    }
}

/// 占有ボクセル単位で局所平面を推定し、そのセル内の点だけを判定する。
///
/// 1点ごとに同じRANSACを繰り返さないため、完成済み大規模マップでも
/// 比較的実行しやすい構成。
fn filter_points(
    points: &[Point3<f32>],
    grid: &VoxelGrid,
    config: &FilterConfig,
) -> (Vec<PointDecision>, FilterStats) {
    let occupied_keys: Vec<VoxelKey> = grid.keys().copied().collect();

    let cell_results: Vec<CellResult> = occupied_keys
        .par_iter()
        .map(|key| classify_voxel(*key, points, grid, config))
        .collect();

    // 各点は必ず一つの対象セルにだけ所属するため、競合なく集約できる。
    let mut decisions = vec![PointDecision::KeepSparse; points.len()];
    let mut stats = FilterStats::default();

    for result in cell_results {
        stats.local_planes += result.extracted_plane_count;

        for (point_index, decision) in result.decisions {
            decisions[point_index] = decision;

            match decision {
                PointDecision::KeepPlaneInlier => stats.kept_plane_inliers += 1,
                PointDecision::KeepSparse => stats.kept_sparse += 1,
                PointDecision::KeepNonPlanar => stats.kept_nonplanar += 1,
                PointDecision::RemovePlaneOutlier => stats.removed_plane_outliers += 1,
                PointDecision::RemoveNonPlanar => stats.removed_nonplanar += 1,
            }
        }
    }

    (decisions, stats)
}

fn classify_voxel(
    key: VoxelKey,
    points: &[Point3<f32>],
    grid: &VoxelGrid,
    config: &FilterConfig,
) -> CellResult {
    let current_indices = match grid.get(&key) {
        Some(indices) => indices,
        None => {
            return CellResult {
                decisions: Vec::new(),
                extracted_plane_count: 0,
            };
        }
    };

    let neighborhood = collect_neighborhood_indices(grid, key, config.neighbor_range);

    if neighborhood.len() < config.min_neighbor_points {
        return CellResult {
            decisions: current_indices
                .iter()
                .copied()
                .map(|index| (index, PointDecision::KeepSparse))
                .collect(),
            extracted_plane_count: 0,
        };
    }

    let planes = extract_local_planes(points, &neighborhood, key, config);

    if planes.is_empty() {
        let decision = if config.keep_nonplanar_regions {
            PointDecision::KeepNonPlanar
        } else {
            PointDecision::RemoveNonPlanar
        };

        return CellResult {
            decisions: current_indices
                .iter()
                .copied()
                .map(|index| (index, decision))
                .collect(),
            extracted_plane_count: 0,
        };
    }

    let decisions = current_indices
        .iter()
        .copied()
        .map(|index| {
            let point = &points[index];

            let is_inlier = planes
                .iter()
                .any(|plane| {
                    plane.distance(point) <= config.keep_distance_threshold
                });

            let decision = if is_inlier {
                PointDecision::KeepPlaneInlier
            } else {
                PointDecision::RemovePlaneOutlier
            };

            (index, decision)
        })
        .collect();

    CellResult {
        decisions,
        extracted_plane_count: planes.len(),
    }
}

fn collect_neighborhood_indices(
    grid: &VoxelGrid,
    center: VoxelKey,
    range: i32,
) -> Vec<usize> {
    let side = (2 * range + 1).max(1) as usize;
    let mut result = Vec::with_capacity(side * side * side * 4);

    for dz in -range..=range {
        for dy in -range..=range {
            for dx in -range..=range {
                let key = VoxelKey {
                    ix: center.ix + dx,
                    iy: center.iy + dy,
                    iz: center.iz + dz,
                };

                if let Some(indices) = grid.get(&key) {
                    result.extend(indices.iter().copied());
                }
            }
        }
    }

    result
}

/// 局所領域から最大N枚の平面を順番に抽出する。
///
/// 1枚目のインライアを除外後、残った点に対して再度RANSACすることで、
/// 壁＋床などの複数平面を救済する。
fn extract_local_planes(
    points: &[Point3<f32>],
    neighborhood: &[usize],
    key: VoxelKey,
    config: &FilterConfig,
) -> Vec<PlaneModel> {
    let mut remaining = neighborhood.to_vec();
    let mut planes = Vec::with_capacity(config.max_planes_per_neighborhood);

    for plane_index in 0..config.max_planes_per_neighborhood {
        if remaining.len() < config.min_neighbor_points {
            break;
        }

        let seed = seed_from_key(key, plane_index as u64);

        let Some(plane) = fit_plane_ransac(points, &remaining, config, seed) else {
            break;
        };

        // RANSACで抽出した平面のインライアを次回候補から除外。
        remaining.retain(|&index| {
            plane.distance(&points[index]) > config.fit_distance_threshold
        });

        planes.push(plane);
    }

    planes
}

fn fit_plane_ransac(
    points: &[Point3<f32>],
    candidate_indices: &[usize],
    config: &FilterConfig,
    seed: u64,
) -> Option<PlaneModel> {
    if candidate_indices.len() < 3 {
        return None;
    }

    let mut rng = XorShift64::new(seed);

    let mut best_normal = Vector3::<f32>::zeros();
    let mut best_origin = Point3::<f32>::origin();
    let mut best_inlier_count = 0usize;
    let mut best_squared_error = f32::INFINITY;

    for _ in 0..config.max_iterations {
        let (i0, i1, i2) = sample_three_unique(&mut rng, candidate_indices.len());

        let p0 = points[candidate_indices[i0]];
        let p1 = points[candidate_indices[i1]];
        let p2 = points[candidate_indices[i2]];

        let v1 = p1 - p0;
        let v2 = p2 - p0;
        let cross = v1.cross(&v2);
        let norm_sq = cross.norm_squared();

        // ほぼ同一直線上の3点は平面を決められない。
        if norm_sq <= 1.0e-10 {
            continue;
        }

        let normal = cross / norm_sq.sqrt();

        let mut inlier_count = 0usize;
        let mut squared_error = 0.0f32;

        for &index in candidate_indices {
            let distance = normal
                .dot(&(points[index].coords - p0.coords))
                .abs();

            if distance <= config.fit_distance_threshold {
                inlier_count += 1;
                squared_error += distance * distance;
            }
        }

        let is_better = inlier_count > best_inlier_count
            || (inlier_count == best_inlier_count
                && squared_error < best_squared_error);

        if is_better {
            best_normal = normal;
            best_origin = p0;
            best_inlier_count = inlier_count;
            best_squared_error = squared_error;
        }
    }

    if best_inlier_count < config.min_inliers {
        return None;
    }

    let initial_ratio = best_inlier_count as f32 / candidate_indices.len() as f32;
    if initial_ratio < config.min_inlier_ratio {
        return None;
    }

    let first_inliers: Vec<usize> = candidate_indices
        .iter()
        .copied()
        .filter(|&index| {
            best_normal
                .dot(&(points[index].coords - best_origin.coords))
                .abs()
                <= config.fit_distance_threshold
        })
        .collect();

    // RANSACの3点モデルをそのまま使わず、全インライアからPCAで再フィット。
    let first_refined = fit_plane_pca(points, &first_inliers)?;

    let second_inliers: Vec<usize> = candidate_indices
        .iter()
        .copied()
        .filter(|&index| {
            first_refined.distance(&points[index]) <= config.fit_distance_threshold
        })
        .collect();

    if second_inliers.len() < config.min_inliers {
        return None;
    }

    let final_ratio = second_inliers.len() as f32 / candidate_indices.len() as f32;
    if final_ratio < config.min_inlier_ratio {
        return None;
    }

    // 再分類後のインライアだけでもう一度PCA。
    let mut final_plane = fit_plane_pca(points, &second_inliers)?;

    let denominator = final_plane.lambda_mid.max(1.0e-12);
    let planarity_ratio = final_plane.lambda_min / denominator;

    if !planarity_ratio.is_finite()
        || planarity_ratio > config.max_planarity_ratio
    {
        return None;
    }

    // lambda_mid が小さい場合、点群が線状または一点集中している。
    let planar_spread = final_plane.lambda_mid.max(0.0).sqrt();
    if planar_spread < config.min_planar_spread {
        return None;
    }

    final_plane.inlier_count = second_inliers.len();
    final_plane.inlier_ratio = final_ratio;

    Some(final_plane)
}

fn fit_plane_pca(
    points: &[Point3<f32>],
    indices: &[usize],
) -> Option<PlaneModel> {
    if indices.len() < 3 {
        return None;
    }

    let mut mean = Vector3::<f32>::zeros();
    for &index in indices {
        mean += points[index].coords;
    }
    mean /= indices.len() as f32;

    let mut covariance = Matrix3::<f32>::zeros();
    for &index in indices {
        let delta = points[index].coords - mean;
        covariance += delta * delta.transpose();
    }
    covariance /= indices.len() as f32;

    if !covariance.iter().all(|value| value.is_finite()) {
        return None;
    }

    let eigen = SymmetricEigen::new(covariance);

    let mut order = [0usize, 1usize, 2usize];
    order.sort_by(|&a, &b| {
        eigen.eigenvalues[a]
            .partial_cmp(&eigen.eigenvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let lambda_min = eigen.eigenvalues[order[0]].max(0.0);
    let lambda_mid = eigen.eigenvalues[order[1]].max(0.0);
    let lambda_max = eigen.eigenvalues[order[2]].max(0.0);

    let mut normal = eigen.eigenvectors.column(order[0]).into_owned();
    let normal_norm = normal.norm();

    if !normal_norm.is_finite() || normal_norm <= 1.0e-8 {
        return None;
    }

    normal /= normal_norm;

    Some(PlaneModel {
        center: Point3::from(mean),
        normal,
        lambda_min,
        lambda_mid,
        lambda_max,
        inlier_count: indices.len(),
        inlier_ratio: 1.0,
    })
}

fn sample_three_unique(
    rng: &mut XorShift64,
    length: usize,
) -> (usize, usize, usize) {
    debug_assert!(length >= 3);

    let i0 = rng.index(length);

    let mut i1 = rng.index(length);
    while i1 == i0 {
        i1 = rng.index(length);
    }

    let mut i2 = rng.index(length);
    while i2 == i0 || i2 == i1 {
        i2 = rng.index(length);
    }

    (i0, i1, i2)
}

fn seed_from_key(key: VoxelKey, plane_index: u64) -> u64 {
    // FNV-1a風の決定的ハッシュ。
    let mut hash = 0xcbf2_9ce4_8422_2325u64;

    for value in [
        key.ix as i64 as u64,
        key.iy as i64 as u64,
        key.iz as i64 as u64,
        plane_index,
    ] {
        hash ^= value;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }

    hash
}

fn parse_args() -> Result<CliArgs> {
    let args: Vec<String> = env::args().collect();

    if args.len() == 2 && matches!(args[1].as_str(), "-h" | "--help") {
        print_usage(&args[0]);
        std::process::exit(0);
    }

    if args.len() < 3 {
        print_usage(args.first().map(String::as_str).unwrap_or("filter_world_map_ransac"));
        bail!("Input and output PCD paths are required");
    }

    let input_path = args[1].clone();
    let output_path = args[2].clone();

    let mut removed_output_path = None;
    let mut config = FilterConfig::default();

    let mut index = 3usize;

    while index < args.len() {
        match args[index].as_str() {
            "--voxel-size" => {
                config.voxel_size = parse_next::<f32>(&args, &mut index, "--voxel-size")?;
            }
            "--neighbor-range" => {
                config.neighbor_range =
                    parse_next::<i32>(&args, &mut index, "--neighbor-range")?;
            }
            // 後方互換: 両方を同じ値に設定する。
            // 太い壁を薄くする用途では、下の2オプションを個別指定する方がよい。
            "--distance-threshold" => {
                let value =
                    parse_next::<f32>(&args, &mut index, "--distance-threshold")?;
                config.fit_distance_threshold = value;
                config.keep_distance_threshold = value;
            }
            "--fit-distance-threshold" => {
                config.fit_distance_threshold =
                    parse_next::<f32>(&args, &mut index, "--fit-distance-threshold")?;
            }
            "--keep-distance-threshold" => {
                config.keep_distance_threshold =
                    parse_next::<f32>(&args, &mut index, "--keep-distance-threshold")?;
            }
            "--iterations" => {
                config.max_iterations =
                    parse_next::<usize>(&args, &mut index, "--iterations")?;
            }
            "--min-neighbors" => {
                config.min_neighbor_points =
                    parse_next::<usize>(&args, &mut index, "--min-neighbors")?;
            }
            "--min-inliers" => {
                config.min_inliers =
                    parse_next::<usize>(&args, &mut index, "--min-inliers")?;
            }
            "--min-inlier-ratio" => {
                config.min_inlier_ratio =
                    parse_next::<f32>(&args, &mut index, "--min-inlier-ratio")?;
            }
            "--max-planarity-ratio" => {
                config.max_planarity_ratio =
                    parse_next::<f32>(&args, &mut index, "--max-planarity-ratio")?;
            }
            "--min-planar-spread" => {
                config.min_planar_spread =
                    parse_next::<f32>(&args, &mut index, "--min-planar-spread")?;
            }
            "--max-planes" => {
                config.max_planes_per_neighborhood =
                    parse_next::<usize>(&args, &mut index, "--max-planes")?;
            }
            "--removed-output" => {
                removed_output_path =
                    Some(parse_next::<String>(&args, &mut index, "--removed-output")?);
            }
            "--remove-nonplanar" => {
                config.keep_nonplanar_regions = false;
            }
            "--keep-nonplanar" => {
                config.keep_nonplanar_regions = true;
            }
            "-h" | "--help" => {
                print_usage(&args[0]);
                std::process::exit(0);
            }
            unknown => {
                bail!("Unknown argument: {unknown}");
            }
        }

        index += 1;
    }

    Ok(CliArgs {
        input_path,
        output_path,
        removed_output_path,
        config,
    })
}

fn parse_next<T>(
    args: &[String],
    index: &mut usize,
    option_name: &str,
) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    *index += 1;

    let value = args
        .get(*index)
        .ok_or_else(|| anyhow!("Missing value after {option_name}"))?;

    value
        .parse::<T>()
        .map_err(|error| anyhow!("Invalid value for {option_name}: '{value}': {error}"))
}

fn validate_config(config: &FilterConfig) -> Result<()> {
    if !config.voxel_size.is_finite() || config.voxel_size <= 0.0 {
        bail!("voxel_size must be finite and > 0");
    }

    if config.neighbor_range < 0 {
        bail!("neighbor_range must be >= 0");
    }

    if !config.fit_distance_threshold.is_finite()
        || config.fit_distance_threshold <= 0.0
    {
        bail!("fit_distance_threshold must be finite and > 0");
    }

    if !config.keep_distance_threshold.is_finite()
        || config.keep_distance_threshold <= 0.0
    {
        bail!("keep_distance_threshold must be finite and > 0");
    }

    if config.keep_distance_threshold > config.fit_distance_threshold {
        bail!(
            "keep_distance_threshold must be <= fit_distance_threshold"
        );
    }

    if config.max_iterations == 0 {
        bail!("max_iterations must be > 0");
    }

    if config.min_neighbor_points < 3 {
        bail!("min_neighbor_points must be >= 3");
    }

    if config.min_inliers < 3 {
        bail!("min_inliers must be >= 3");
    }

    if config.min_inliers > config.min_neighbor_points {
        bail!("min_inliers must be <= min_neighbor_points");
    }

    if !(0.0..=1.0).contains(&config.min_inlier_ratio) {
        bail!("min_inlier_ratio must be in [0, 1]");
    }

    if !config.max_planarity_ratio.is_finite()
        || config.max_planarity_ratio < 0.0
    {
        bail!("max_planarity_ratio must be finite and >= 0");
    }

    if !config.min_planar_spread.is_finite()
        || config.min_planar_spread < 0.0
    {
        bail!("min_planar_spread must be finite and >= 0");
    }

    if config.max_planes_per_neighborhood == 0 {
        bail!("max_planes_per_neighborhood must be > 0");
    }

    Ok(())
}

fn print_usage(program: &str) {
    eprintln!(
        r#"Usage:
  {program} <input.pcd> <output.pcd> [options]

Options:
  --voxel-size <m>              Filter-grid voxel size
                                Default: 0.20

  --neighbor-range <cells>      Neighbor-cell radius
                                Default: 1

  --fit-distance-threshold <m>  Broad threshold used to FIND a rough plane
                                Default: 0.060

  --keep-distance-threshold <m> Narrow threshold used to KEEP final points
                                Default: 0.015

  --distance-threshold <m>      Compatibility option: sets both thresholds
                                to the same value

  --iterations <count>          RANSAC iterations per plane
                                Default: 64

  --min-neighbors <count>       Minimum local points for RANSAC
                                Default: 12

  --min-inliers <count>         Minimum inliers for a valid plane
                                Default: 8

  --min-inlier-ratio <ratio>    Minimum inlier ratio
                                Default: 0.45

  --max-planarity-ratio <ratio> lambda_min / lambda_mid threshold
                                Default: 0.15

  --min-planar-spread <m>       Minimum sqrt(lambda_mid)
                                Default: 0.03

  --max-planes <count>          Maximum planes extracted per neighborhood
                                Default: 2

  --removed-output <path>       Save removed points to another PCD

  --remove-nonplanar            Remove regions where no valid plane exists
                                Default behavior is to keep them

  --keep-nonplanar              Explicitly keep non-planar regions

  -h, --help                    Show this help

Recommended first run:
  {program} input.pcd output.pcd \
    --removed-output removed.pcd

Stricter wall filtering:
  {program} input.pcd output_strict.pcd \
    --fit-distance-threshold 0.060 \
    --keep-distance-threshold 0.015 \
    --max-planarity-ratio 0.20 \
    --removed-output removed_strict.pcd
"#
    );
}
