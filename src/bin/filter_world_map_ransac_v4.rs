use std::{collections::HashMap, env, fs, path::Path, time::Instant};

use anyhow::{Context, Result, anyhow, bail};
use nalgebra::{Matrix3, Point3, SymmetricEigen, Vector3};
use pcd_rs::Reader;
use rayon::prelude::*;

use re_lidar_slam::{file_handler::save_pcd_xyz, types::PointXYZ};

/// 完成済みのグローバルマップPCDに対して、局所RANSAC平面フィルタを適用する。
///
/// 配置例:
///     src/bin/filter_world_map_ransac_v4.rs
///
/// 実行例:
///     cargo run --release --bin filter_world_map_ransac_v4 -- \
///         data/output/debug/07122026/voxel-0.1_world_map.pcd \
///         data/output/debug/07122026/voxel-0.1_world_map_ransac.pcd
///
/// パラメータ指定例:
///     cargo run --release --bin filter_world_map_ransac_v4 -- \
///         input.pcd output.pcd \
///         --voxel-size 0.20 \
///         --neighbor-range 1 \
///         --fit-distance-threshold 0.060 \
///         --keep-distance-threshold 0.015 \
///         --iterations 64 \
///         --min-neighbors 12 \
///         --min-inliers 8 \
///         --min-inlier-ratio 0.45 \
///         --max-planarity-ratio 0.15 \
///         --min-planar-spread 0.03 \
///         --max-planes 1 \
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

    /// 粗いRANSAC平面から法線方向の最頻層を選んだ後、
    /// PCA再フィットに使用する距離しきい値 [m]。
    ///
    /// fit_distance_threshold より小さく、keep_distance_threshold より
    /// 少し大きく設定する。例: 0.02～0.03 m。
    refit_distance_threshold: f32,

    /// 法線方向の残差分布から最頻層を探すヒストグラム幅 [m]。
    ///
    /// 太い壁を全体平均して中央へ寄せるのではなく、最も密な表面層を選ぶ。
    residual_bin_size: f32,

    /// 点を判定するとき、対象セル以外の周囲何セルの平面を参照するか。
    ///
    /// 1なら3×3×3セルの平面候補を利用し、セル境界の穴を減らす。
    plane_vote_range: i32,

    /// 自セルの平面とほぼ平行な近傍平面を利用するときに許容する
    /// 平面間距離 [m]。
    ///
    /// 離れた平行な二重壁層が、近傍セルの平面として点を救済するのを防ぐ。
    parallel_plane_tolerance: f32,

    /// 平行とみなす法線角度 [deg]。
    parallel_normal_angle_deg: f32,

    /// 近傍セルの平面群から同じ向きの表面クラスタを作る際の法線角度 [deg]。
    ///
    /// 小さいほど、異なる向きの壁・床を同じ表面として混ぜにくい。
    consensus_normal_angle_deg: f32,

    /// 近傍平面から代表平面を作るために最低限必要な平面数。
    ///
    /// 1なら孤立した平面も使用する。2以上にすると滑らかになるが、
    /// 疎な場所で参照平面が減る。
    min_consensus_planes: usize,

    /// 自セルに代表平面がない場合、周囲の代表平面を借用する最大セル範囲。
    ///
    /// plane_vote_range と同じ値から開始するのが安全。
    reference_plane_range: i32,

    /// 候補平面がセル内点を説明していると数える距離 [m]。
    ///
    /// 1セルにつき候補平面を1枚だけ選ぶためのスコアに使用する。
    reference_support_distance: f32,

    /// 厳密インライアが一つもないセルで、穴を防ぐために残す最低点数。
    ///
    /// 0なら救済なし。1を推奨。
    min_output_points_per_voxel: usize,

    /// 上記の救済点に許可する平面距離上限 [m]。
    ///
    /// 小さな凹みを1セル1点程度で残しつつ、厚いノイズ層全体は残さない。
    fallback_keep_distance: f32,

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
            refit_distance_threshold: 0.025,
            residual_bin_size: 0.010,
            plane_vote_range: 1,
            parallel_plane_tolerance: 0.025,
            parallel_normal_angle_deg: 15.0,
            consensus_normal_angle_deg: 12.0,
            min_consensus_planes: 1,
            reference_plane_range: 1,
            reference_support_distance: 0.060,
            min_output_points_per_voxel: 1,
            fallback_keep_distance: 0.050,
            max_iterations: 64,
            min_neighbor_points: 12,
            min_inliers: 8,
            min_inlier_ratio: 0.45,
            max_planarity_ratio: 0.15,
            min_planar_spread: 0.03,
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
struct PlaneCell {
    neighborhood_point_count: usize,
    primary_plane: Option<PlaneModel>,
}

type PlaneGrid = HashMap<VoxelKey, PlaneCell>;
type ConsensusPlaneGrid = HashMap<VoxelKey, PlaneModel>;

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
        self.normal.dot(&(point.coords - self.center.coords)).abs()
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

    /// 厳密インライアは無かったが、セル密度の穴を防ぐため
    /// 最も近い点を有限数だけ救済。
    KeepFallback,

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
                | PointDecision::KeepFallback
        )
    }
}

#[derive(Debug, Clone, Copy)]
enum PlaneReferenceSource {
    OwnConsensus,
    NeighborConsensus,
    None,
}

#[derive(Debug)]
struct CellResult {
    decisions: Vec<(usize, PointDecision)>,
    reference_source: PlaneReferenceSource,
}

#[derive(Debug, Default)]
struct FilterStats {
    input_points: usize,
    occupied_voxels: usize,

    kept_plane_inliers: usize,
    kept_sparse: usize,
    kept_nonplanar: usize,
    kept_fallback: usize,

    removed_plane_outliers: usize,
    removed_nonplanar: usize,

    local_planes: usize,
    consensus_planes: usize,

    own_reference_cells: usize,
    neighbor_reference_cells: usize,
    no_reference_cells: usize,
}

impl FilterStats {
    fn kept_total(&self) -> usize {
        self.kept_plane_inliers + self.kept_sparse + self.kept_nonplanar + self.kept_fallback
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
    println!("Consensus surface planes  : {}", stats.consensus_planes);
    println!("Own-reference cells       : {}", stats.own_reference_cells);
    println!(
        "Neighbor-reference cells  : {}",
        stats.neighbor_reference_cells
    );
    println!("No-reference cells        : {}", stats.no_reference_cells);
    println!("Kept plane inliers        : {}", stats.kept_plane_inliers);
    println!("Kept sparse points        : {}", stats.kept_sparse);
    println!("Kept non-planar points    : {}", stats.kept_nonplanar);
    println!("Kept fallback points      : {}", stats.kept_fallback);
    println!(
        "Removed plane outliers    : {}",
        stats.removed_plane_outliers
    );
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

    // Pass 1:
    // 各セルについてRANSAC/PCAによる主要平面を推定。
    let plane_entries: Vec<(VoxelKey, PlaneCell)> = occupied_keys
        .par_iter()
        .map(|&key| {
            let neighborhood = collect_neighborhood_indices(grid, key, config.neighbor_range);

            let primary_plane = if neighborhood.len() >= config.min_neighbor_points {
                fit_plane_ransac(points, &neighborhood, config, seed_from_key(key, 0))
            } else {
                None
            };

            (
                key,
                PlaneCell {
                    neighborhood_point_count: neighborhood.len(),
                    primary_plane,
                },
            )
        })
        .collect();

    let plane_grid: PlaneGrid = plane_entries.into_iter().collect();

    // Pass 2:
    // 近傍セルの平面を、法線方向と平面オフセットでクラスタリングし、
    // 各セルに対して一つの安定した代表平面を作る。
    //
    // v3のように「周囲のどれか一枚に近ければ保持」とすると、
    // 少しずつずれた複数の平面帯の和集合が残り、壁が厚くなる。
    let consensus_entries: Vec<(VoxelKey, PlaneModel)> = occupied_keys
        .par_iter()
        .filter_map(|&key| {
            build_consensus_plane(key, &plane_grid, config).map(|plane| (key, plane))
        })
        .collect();

    let consensus_grid: ConsensusPlaneGrid = consensus_entries.into_iter().collect();

    // Pass 3:
    // 各セルについて候補代表平面を列挙するが、点ごとのOR判定は行わない。
    // セル内点群を最も多く、かつ小さい残差で説明する一枚だけを選択する。
    let cell_results: Vec<CellResult> = occupied_keys
        .par_iter()
        .map(|&key| {
            classify_voxel_single_reference(key, points, grid, &plane_grid, &consensus_grid, config)
        })
        .collect();

    let mut decisions = vec![PointDecision::KeepSparse; points.len()];
    let mut stats = FilterStats::default();

    stats.local_planes = plane_grid
        .values()
        .filter(|cell| cell.primary_plane.is_some())
        .count();

    stats.consensus_planes = consensus_grid.len();

    for result in cell_results {
        match result.reference_source {
            PlaneReferenceSource::OwnConsensus => {
                stats.own_reference_cells += 1;
            }
            PlaneReferenceSource::NeighborConsensus => {
                stats.neighbor_reference_cells += 1;
            }
            PlaneReferenceSource::None => {
                stats.no_reference_cells += 1;
            }
        }

        for (point_index, decision) in result.decisions {
            decisions[point_index] = decision;

            match decision {
                PointDecision::KeepPlaneInlier => {
                    stats.kept_plane_inliers += 1;
                }
                PointDecision::KeepSparse => {
                    stats.kept_sparse += 1;
                }
                PointDecision::KeepNonPlanar => {
                    stats.kept_nonplanar += 1;
                }
                PointDecision::KeepFallback => {
                    stats.kept_fallback += 1;
                }
                PointDecision::RemovePlaneOutlier => {
                    stats.removed_plane_outliers += 1;
                }
                PointDecision::RemoveNonPlanar => {
                    stats.removed_nonplanar += 1;
                }
            }
        }
    }

    (decisions, stats)
}

/// 近傍の局所平面群から、対象セル用の一枚の代表平面を作る。
///
/// 手順:
/// 1. 法線が似ている平面クラスタを探す。
/// 2. そのクラスタ内で、法線方向オフセットが近い表面層を探す。
/// 3. 最も総支持重みが大きい層だけを加重平均する。
///
/// これにより、同じ壁の少しずれた推定平面を滑らかにしつつ、
/// 数cm離れた平行な二重層を同じ代表平面へ混ぜない。
fn build_consensus_plane(
    key: VoxelKey,
    plane_grid: &PlaneGrid,
    config: &FilterConfig,
) -> Option<PlaneModel> {
    let mut candidates: Vec<&PlaneModel> = Vec::new();

    for dz in -config.plane_vote_range..=config.plane_vote_range {
        for dy in -config.plane_vote_range..=config.plane_vote_range {
            for dx in -config.plane_vote_range..=config.plane_vote_range {
                let neighbor_key = VoxelKey {
                    ix: key.ix + dx,
                    iy: key.iy + dy,
                    iz: key.iz + dz,
                };

                if let Some(plane) = plane_grid
                    .get(&neighbor_key)
                    .and_then(|cell| cell.primary_plane.as_ref())
                {
                    candidates.push(plane);
                }
            }
        }
    }

    if candidates.is_empty() {
        return None;
    }

    let normal_cos_threshold = config.consensus_normal_angle_deg.to_radians().cos();

    // 最も支持重みの大きい法線クラスタのアンカーを選択。
    let mut best_anchor_index = 0usize;
    let mut best_normal_score = f32::NEG_INFINITY;

    for (candidate_index, &candidate) in candidates.iter().enumerate() {
        let mut score = 0.0f32;

        for &other in &candidates {
            if candidate.normal.dot(&other.normal).abs() >= normal_cos_threshold {
                score += plane_quality_weight(other);
            }
        }

        if score > best_normal_score {
            best_normal_score = score;
            best_anchor_index = candidate_index;
        }
    }

    let anchor_normal = candidates[best_anchor_index].normal;

    // アンカーと同じ向きの平面だけを集め、法線の符号をそろえる。
    let mut normal_cluster: Vec<(&PlaneModel, Vector3<f32>, f32)> = Vec::new();

    for plane in candidates {
        let dot = anchor_normal.dot(&plane.normal);

        if dot.abs() < normal_cos_threshold {
            continue;
        }

        let aligned_normal = if dot < 0.0 {
            -plane.normal
        } else {
            plane.normal
        };

        normal_cluster.push((plane, aligned_normal, plane_quality_weight(plane)));
    }

    if normal_cluster.len() < config.min_consensus_planes {
        return None;
    }

    let initial_normal_sum = normal_cluster
        .iter()
        .fold(Vector3::<f32>::zeros(), |sum, (_, normal, weight)| {
            sum + *normal * *weight
        });

    let initial_normal_norm = initial_normal_sum.norm();
    if !initial_normal_norm.is_finite() || initial_normal_norm <= 1.0e-8 {
        return None;
    }

    let initial_normal = initial_normal_sum / initial_normal_norm;

    // 共通法線に沿った平面オフセット n·x を計算。
    let mut offset_planes: Vec<(f32, f32, &PlaneModel, Vector3<f32>)> = normal_cluster
        .into_iter()
        .map(|(plane, aligned_normal, weight)| {
            (
                initial_normal.dot(&plane.center.coords),
                weight,
                plane,
                aligned_normal,
            )
        })
        .collect();

    offset_planes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // 幅 parallel_plane_tolerance 内で支持重みが最大となる表面層を選択。
    let mut left = 0usize;
    let mut running_weight = 0.0f32;
    let mut best_left = 0usize;
    let mut best_right = 0usize;
    let mut best_weight = f32::NEG_INFINITY;

    for right in 0..offset_planes.len() {
        running_weight += offset_planes[right].1;

        while offset_planes[right].0 - offset_planes[left].0 > config.parallel_plane_tolerance {
            running_weight -= offset_planes[left].1;
            left += 1;
        }

        let count = right - left + 1;
        let best_count = best_right.saturating_sub(best_left) + 1;

        if running_weight > best_weight || (running_weight == best_weight && count > best_count) {
            best_weight = running_weight;
            best_left = left;
            best_right = right;
        }
    }

    let selected = &offset_planes[best_left..=best_right];

    if selected.len() < config.min_consensus_planes {
        return None;
    }

    // 選択された同一表面層だけから最終代表平面を作成。
    let final_normal_sum = selected
        .iter()
        .fold(Vector3::<f32>::zeros(), |sum, (_, weight, _, normal)| {
            sum + *normal * *weight
        });

    let final_normal_norm = final_normal_sum.norm();
    if !final_normal_norm.is_finite() || final_normal_norm <= 1.0e-8 {
        return None;
    }

    let final_normal = final_normal_sum / final_normal_norm;

    let total_weight = selected
        .iter()
        .map(|(_, weight, _, _)| *weight)
        .sum::<f32>()
        .max(1.0e-12);

    let final_offset = selected
        .iter()
        .map(|(_, weight, plane, _)| final_normal.dot(&plane.center.coords) * *weight)
        .sum::<f32>()
        / total_weight;

    let cell_center = voxel_center_from_key(key, config.voxel_size);

    let center_shift = final_offset - final_normal.dot(&cell_center.coords);

    let final_center = Point3::from(cell_center.coords + final_normal * center_shift);

    let lambda_min = selected
        .iter()
        .map(|(_, weight, plane, _)| plane.lambda_min * *weight)
        .sum::<f32>()
        / total_weight;

    let lambda_mid = selected
        .iter()
        .map(|(_, weight, plane, _)| plane.lambda_mid * *weight)
        .sum::<f32>()
        / total_weight;

    let lambda_max = selected
        .iter()
        .map(|(_, weight, plane, _)| plane.lambda_max * *weight)
        .sum::<f32>()
        / total_weight;

    let inlier_count = selected
        .iter()
        .map(|(_, _, plane, _)| plane.inlier_count)
        .sum::<usize>();

    let inlier_ratio = selected
        .iter()
        .map(|(_, weight, plane, _)| plane.inlier_ratio * *weight)
        .sum::<f32>()
        / total_weight;

    Some(PlaneModel {
        center: final_center,
        normal: final_normal,
        lambda_min,
        lambda_mid,
        lambda_max,
        inlier_count,
        inlier_ratio,
    })
}

/// 各セルにつき候補平面を一枚だけ選び、その一枚に対して点を判定する。
///
/// v3の min(distance to ANY neighbor plane) は、ずれた平面帯の和集合を
/// 残してしまう。この関数では、セル内の点を最も多く説明し、残差が小さい
/// 一枚だけを選択する。
fn classify_voxel_single_reference(
    key: VoxelKey,
    points: &[Point3<f32>],
    grid: &VoxelGrid,
    raw_plane_grid: &PlaneGrid,
    consensus_grid: &ConsensusPlaneGrid,
    config: &FilterConfig,
) -> CellResult {
    let current_indices = match grid.get(&key) {
        Some(indices) => indices,
        None => {
            return CellResult {
                decisions: Vec::new(),
                reference_source: PlaneReferenceSource::None,
            };
        }
    };

    let (reference_plane, reference_source) =
        choose_single_reference_plane(key, current_indices, points, consensus_grid, config);

    let Some(reference_plane) = reference_plane else {
        let neighborhood_count = raw_plane_grid
            .get(&key)
            .map(|cell| cell.neighborhood_point_count)
            .unwrap_or(0);

        let decision = if neighborhood_count < config.min_neighbor_points {
            PointDecision::KeepSparse
        } else if config.keep_nonplanar_regions {
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
            reference_source: PlaneReferenceSource::None,
        };
    };

    let mut decisions: Vec<(usize, PointDecision)> = Vec::with_capacity(current_indices.len());

    let mut fallback_candidates: Vec<(f32, usize)> = Vec::new();
    let mut strict_keep_count = 0usize;

    for &point_index in current_indices {
        let distance = reference_plane.distance(&points[point_index]);

        if distance <= config.keep_distance_threshold {
            strict_keep_count += 1;
            decisions.push((point_index, PointDecision::KeepPlaneInlier));
        } else {
            let decision_position = decisions.len();

            decisions.push((point_index, PointDecision::RemovePlaneOutlier));

            if distance <= config.fallback_keep_distance {
                fallback_candidates.push((distance, decision_position));
            }
        }
    }

    // セルの密度穴を防ぐ有限救済。
    if strict_keep_count < config.min_output_points_per_voxel {
        fallback_candidates
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let rescue_count = config.min_output_points_per_voxel - strict_keep_count;

        for &(_, decision_position) in fallback_candidates.iter().take(rescue_count) {
            decisions[decision_position].1 = PointDecision::KeepFallback;
        }
    }

    CellResult {
        decisions,
        reference_source,
    }
}

/// 自セルおよび周囲セルの代表平面候補から、一枚だけを選択する。
///
/// スコア優先順位:
/// 1. セル内で reference_support_distance 以内に入る点数が多い
/// 2. その点の平均二乗残差が小さい
/// 3. 平面自体の支持点数が多い
///
/// 自セルの代表平面が同率なら優先する。
fn choose_single_reference_plane<'a>(
    key: VoxelKey,
    current_indices: &[usize],
    points: &[Point3<f32>],
    consensus_grid: &'a ConsensusPlaneGrid,
    config: &FilterConfig,
) -> (Option<&'a PlaneModel>, PlaneReferenceSource) {
    let own_plane = consensus_grid.get(&key);

    let mut best_plane: Option<&PlaneModel> = None;
    let mut best_source = PlaneReferenceSource::None;
    let mut best_support_count = 0usize;
    let mut best_mean_squared_error = f32::INFINITY;
    let mut best_plane_support = 0usize;
    let mut best_is_own = false;

    for dz in -config.reference_plane_range..=config.reference_plane_range {
        for dy in -config.reference_plane_range..=config.reference_plane_range {
            for dx in -config.reference_plane_range..=config.reference_plane_range {
                let candidate_key = VoxelKey {
                    ix: key.ix + dx,
                    iy: key.iy + dy,
                    iz: key.iz + dz,
                };

                let Some(candidate) = consensus_grid.get(&candidate_key) else {
                    continue;
                };

                let mut support_count = 0usize;
                let mut squared_error = 0.0f32;

                for &point_index in current_indices {
                    let distance = candidate.distance(&points[point_index]);

                    if distance <= config.reference_support_distance {
                        support_count += 1;
                        squared_error += distance * distance;
                    }
                }

                if support_count == 0 {
                    continue;
                }

                let mean_squared_error = squared_error / support_count as f32;

                let is_own = candidate_key == key;

                let better = support_count > best_support_count
                    || (support_count == best_support_count
                        && mean_squared_error < best_mean_squared_error)
                    || (support_count == best_support_count
                        && (mean_squared_error - best_mean_squared_error).abs() <= 1.0e-12
                        && candidate.inlier_count > best_plane_support)
                    || (support_count == best_support_count
                        && (mean_squared_error - best_mean_squared_error).abs() <= 1.0e-12
                        && candidate.inlier_count == best_plane_support
                        && is_own
                        && !best_is_own);

                if better {
                    best_plane = Some(candidate);
                    best_source = if is_own {
                        PlaneReferenceSource::OwnConsensus
                    } else {
                        PlaneReferenceSource::NeighborConsensus
                    };
                    best_support_count = support_count;
                    best_mean_squared_error = mean_squared_error;
                    best_plane_support = candidate.inlier_count;
                    best_is_own = is_own;
                }
            }
        }
    }

    // own_planeは上の探索範囲が0でも含まれるが、
    // 将来の設定変更に対する安全策として保持。
    if best_plane.is_none() {
        if let Some(plane) = own_plane {
            return (Some(plane), PlaneReferenceSource::OwnConsensus);
        }
    }

    (best_plane, best_source)
}

#[inline]
fn plane_quality_weight(plane: &PlaneModel) -> f32 {
    let planarity_ratio = plane.lambda_min / plane.lambda_mid.max(1.0e-12);

    let planarity_quality = (1.0 - planarity_ratio).clamp(0.05, 1.0);

    plane.inlier_count.max(1) as f32 * planarity_quality
}

#[inline]
fn voxel_center_from_key(key: VoxelKey, voxel_size: f32) -> Point3<f32> {
    Point3::new(
        (key.ix as f32 + 0.5) * voxel_size,
        (key.iy as f32 + 0.5) * voxel_size,
        (key.iz as f32 + 0.5) * voxel_size,
    )
}

fn collect_neighborhood_indices(grid: &VoxelGrid, center: VoxelKey, range: i32) -> Vec<usize> {
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

/// 局所領域から主要な1枚の平面をRANSACで抽出し、
/// 法線方向の最頻層を使ってPCA再フィットする。
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

        if norm_sq <= 1.0e-10 {
            continue;
        }

        let normal = cross / norm_sq.sqrt();

        let mut inlier_count = 0usize;
        let mut squared_error = 0.0f32;

        for &index in candidate_indices {
            let distance = normal.dot(&(points[index].coords - p0.coords)).abs();

            if distance <= config.fit_distance_threshold {
                inlier_count += 1;
                squared_error += distance * distance;
            }
        }

        let is_better = inlier_count > best_inlier_count
            || (inlier_count == best_inlier_count && squared_error < best_squared_error);

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

    let rough_ratio = best_inlier_count as f32 / candidate_indices.len() as f32;

    if rough_ratio < config.min_inlier_ratio {
        return None;
    }

    // 太い壁を±fit_distance_threshold全体で平均すると、
    // 小さな凹凸が消えたり、二重層の中央へ平面が寄ったりする。
    //
    // そこで粗いRANSAC法線に沿った符号付き残差をヒストグラム化し、
    // 最も点数の多い表面層を選んでPCA再フィットする。
    let mode_origin =
        find_dominant_residual_layer(points, candidate_indices, best_origin, best_normal, config)?;

    let first_refit_indices: Vec<usize> = candidate_indices
        .iter()
        .copied()
        .filter(|&index| {
            best_normal
                .dot(&(points[index].coords - mode_origin.coords))
                .abs()
                <= config.refit_distance_threshold
        })
        .collect();

    if first_refit_indices.len() < config.min_inliers {
        return None;
    }

    let first_refined = fit_plane_pca(points, &first_refit_indices)?;

    // PCA後の平面でもう一度狭い帯域に再分類。
    let final_indices: Vec<usize> = candidate_indices
        .iter()
        .copied()
        .filter(|&index| first_refined.distance(&points[index]) <= config.refit_distance_threshold)
        .collect();

    if final_indices.len() < config.min_inliers {
        return None;
    }

    let mut final_plane = fit_plane_pca(points, &final_indices)?;

    let denominator = final_plane.lambda_mid.max(1.0e-12);
    let planarity_ratio = final_plane.lambda_min / denominator;

    if !planarity_ratio.is_finite() || planarity_ratio > config.max_planarity_ratio {
        return None;
    }

    let planar_spread = final_plane.lambda_mid.max(0.0).sqrt();
    if planar_spread < config.min_planar_spread {
        return None;
    }

    final_plane.inlier_count = final_indices.len();
    final_plane.inlier_ratio = rough_ratio;

    Some(final_plane)
}

fn find_dominant_residual_layer(
    points: &[Point3<f32>],
    candidate_indices: &[usize],
    rough_origin: Point3<f32>,
    rough_normal: Vector3<f32>,
    config: &FilterConfig,
) -> Option<Point3<f32>> {
    let total_width = 2.0 * config.fit_distance_threshold;
    let bin_count = (total_width / config.residual_bin_size).ceil().max(1.0) as usize + 1;

    let mut counts = vec![0usize; bin_count];
    let mut sums = vec![0.0f32; bin_count];

    for &index in candidate_indices {
        let signed_distance = rough_normal.dot(&(points[index].coords - rough_origin.coords));

        if signed_distance.abs() > config.fit_distance_threshold {
            continue;
        }

        let normalized =
            (signed_distance + config.fit_distance_threshold) / config.residual_bin_size;

        let bin = (normalized.floor() as isize).clamp(0, bin_count as isize - 1) as usize;

        counts[bin] += 1;
        sums[bin] += signed_distance;
    }

    let best_bin = counts
        .iter()
        .enumerate()
        .max_by(|(index_a, count_a), (index_b, count_b)| {
            count_a.cmp(count_b).then_with(|| {
                // 同数なら粗いRANSAC平面に近い層を優先。
                let center_a = (*index_a as f32 + 0.5) * config.residual_bin_size
                    - config.fit_distance_threshold;

                let center_b = (*index_b as f32 + 0.5) * config.residual_bin_size
                    - config.fit_distance_threshold;

                center_b
                    .abs()
                    .partial_cmp(&center_a.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        })
        .map(|(index, _)| index)?;

    if counts[best_bin] == 0 {
        return None;
    }

    let mode_signed_distance = sums[best_bin] / counts[best_bin] as f32;

    Some(Point3::from(
        rough_origin.coords + rough_normal * mode_signed_distance,
    ))
}

fn fit_plane_pca(points: &[Point3<f32>], indices: &[usize]) -> Option<PlaneModel> {
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

fn sample_three_unique(rng: &mut XorShift64, length: usize) -> (usize, usize, usize) {
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
        print_usage(
            args.first()
                .map(String::as_str)
                .unwrap_or("filter_world_map_ransac"),
        );
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
                config.neighbor_range = parse_next::<i32>(&args, &mut index, "--neighbor-range")?;
            }
            // 後方互換: 両方を同じ値に設定する。
            // 太い壁を薄くする用途では、下の2オプションを個別指定する方がよい。
            "--distance-threshold" => {
                let value = parse_next::<f32>(&args, &mut index, "--distance-threshold")?;
                config.fit_distance_threshold = value;
                config.refit_distance_threshold = value;
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
            "--refit-distance-threshold" => {
                config.refit_distance_threshold =
                    parse_next::<f32>(&args, &mut index, "--refit-distance-threshold")?;
            }
            "--residual-bin-size" => {
                config.residual_bin_size =
                    parse_next::<f32>(&args, &mut index, "--residual-bin-size")?;
            }
            "--plane-vote-range" => {
                config.plane_vote_range =
                    parse_next::<i32>(&args, &mut index, "--plane-vote-range")?;
            }
            "--parallel-plane-tolerance" => {
                config.parallel_plane_tolerance =
                    parse_next::<f32>(&args, &mut index, "--parallel-plane-tolerance")?;
            }
            "--parallel-normal-angle" => {
                config.parallel_normal_angle_deg =
                    parse_next::<f32>(&args, &mut index, "--parallel-normal-angle")?;
            }
            "--consensus-normal-angle" => {
                config.consensus_normal_angle_deg =
                    parse_next::<f32>(&args, &mut index, "--consensus-normal-angle")?;
            }
            "--min-consensus-planes" => {
                config.min_consensus_planes =
                    parse_next::<usize>(&args, &mut index, "--min-consensus-planes")?;
            }
            "--reference-plane-range" => {
                config.reference_plane_range =
                    parse_next::<i32>(&args, &mut index, "--reference-plane-range")?;
            }
            "--reference-support-distance" => {
                config.reference_support_distance =
                    parse_next::<f32>(&args, &mut index, "--reference-support-distance")?;
            }
            "--min-output-points" => {
                config.min_output_points_per_voxel =
                    parse_next::<usize>(&args, &mut index, "--min-output-points")?;
            }
            "--fallback-keep-distance" => {
                config.fallback_keep_distance =
                    parse_next::<f32>(&args, &mut index, "--fallback-keep-distance")?;
            }
            "--iterations" => {
                config.max_iterations = parse_next::<usize>(&args, &mut index, "--iterations")?;
            }
            "--min-neighbors" => {
                config.min_neighbor_points =
                    parse_next::<usize>(&args, &mut index, "--min-neighbors")?;
            }
            "--min-inliers" => {
                config.min_inliers = parse_next::<usize>(&args, &mut index, "--min-inliers")?;
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
                let value = parse_next::<usize>(&args, &mut index, "--max-planes")?;
                if value != 1 {
                    bail!("v4 stores one dominant plane per filter cell; --max-planes must be 1");
                }
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

fn parse_next<T>(args: &[String], index: &mut usize, option_name: &str) -> Result<T>
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

    if !config.fit_distance_threshold.is_finite() || config.fit_distance_threshold <= 0.0 {
        bail!("fit_distance_threshold must be finite and > 0");
    }

    if !config.keep_distance_threshold.is_finite() || config.keep_distance_threshold <= 0.0 {
        bail!("keep_distance_threshold must be finite and > 0");
    }

    if config.keep_distance_threshold > config.fit_distance_threshold {
        bail!("keep_distance_threshold must be <= fit_distance_threshold");
    }

    if !config.refit_distance_threshold.is_finite() || config.refit_distance_threshold <= 0.0 {
        bail!("refit_distance_threshold must be finite and > 0");
    }

    if config.refit_distance_threshold > config.fit_distance_threshold {
        bail!("refit_distance_threshold must be <= fit_distance_threshold");
    }

    if config.keep_distance_threshold > config.refit_distance_threshold {
        bail!("keep_distance_threshold must be <= refit_distance_threshold");
    }

    if !config.residual_bin_size.is_finite() || config.residual_bin_size <= 0.0 {
        bail!("residual_bin_size must be finite and > 0");
    }

    if config.plane_vote_range < 0 {
        bail!("plane_vote_range must be >= 0");
    }

    if !config.parallel_plane_tolerance.is_finite() || config.parallel_plane_tolerance < 0.0 {
        bail!("parallel_plane_tolerance must be finite and >= 0");
    }

    if !config.parallel_normal_angle_deg.is_finite()
        || !(0.0..=90.0).contains(&config.parallel_normal_angle_deg)
    {
        bail!("parallel_normal_angle_deg must be in [0, 90]");
    }

    if !config.consensus_normal_angle_deg.is_finite()
        || !(0.0..=90.0).contains(&config.consensus_normal_angle_deg)
    {
        bail!("consensus_normal_angle_deg must be in [0, 90]");
    }

    if config.min_consensus_planes == 0 {
        bail!("min_consensus_planes must be > 0");
    }

    if config.reference_plane_range < 0 {
        bail!("reference_plane_range must be >= 0");
    }

    if !config.reference_support_distance.is_finite() || config.reference_support_distance <= 0.0 {
        bail!("reference_support_distance must be finite and > 0");
    }

    if !config.fallback_keep_distance.is_finite()
        || config.fallback_keep_distance < config.keep_distance_threshold
    {
        bail!("fallback_keep_distance must be finite and >= keep_distance_threshold");
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

    if !config.max_planarity_ratio.is_finite() || config.max_planarity_ratio < 0.0 {
        bail!("max_planarity_ratio must be finite and >= 0");
    }

    if !config.min_planar_spread.is_finite() || config.min_planar_spread < 0.0 {
        bail!("min_planar_spread must be finite and >= 0");
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

  --refit-distance-threshold <m>
                                Narrow band used for PCA re-fit after selecting
                                the dominant residual layer
                                Default: 0.025

  --residual-bin-size <m>       Histogram bin width used to select the dominant
                                wall layer along the plane normal
                                Default: 0.010

  --plane-vote-range <cells>    Neighbor plane-cell range used in pass 2
                                Default: 1

  --parallel-plane-tolerance <m>
                                Maximum separation for accepting a neighboring
                                nearly-parallel plane
                                Default: 0.035

  --parallel-normal-angle <deg> Legacy compatibility option
                                Default: 15

  --consensus-normal-angle <deg>
                                Maximum normal difference inside one surface
                                cluster
                                Default: 12

  --min-consensus-planes <n>    Minimum planes for a consensus surface
                                Default: 1

  --reference-plane-range <n>   Neighbor range searched when selecting exactly
                                one reference plane for a cell
                                Default: 1

  --reference-support-distance <m>
                                Distance used to score how many cell points a
                                candidate reference plane explains
                                Default: 0.060

  --min-output-points <count>   Minimum rescued points per occupied filter cell
                                Default: 1

  --fallback-keep-distance <m>  Maximum distance for rescued points
                                Default: 0.050

  --distance-threshold <m>      Compatibility option: sets fit and keep
                                thresholds to the same value
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

  --max-planes <count>          Compatibility option; v3 requires 1

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
