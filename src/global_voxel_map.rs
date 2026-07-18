use foldhash::{HashMap, HashMapExt};
use nalgebra::{Matrix3, Matrix4, Point3, Vector3};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoxelKey {
    pub ix: i32,
    pub iy: i32,
    pub iz: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaneState {
    Collecting,
    Stable,
    NoPlaneAtThisLevel,
}

#[derive(Debug, Clone, Copy)]
pub struct GlobalSample {
    pub position: Point3<f32>,
    pub frame_id: u64,
    pub weight: f32,
}

#[derive(Debug, Clone)]
pub struct PlaneEstimate {
    /// 平面インライアの重心。
    pub center: Point3<f32>,

    /// 単位法線。
    pub normal: Vector3<f32>,

    /// インライア点群の共分散。
    pub covariance: Matrix3<f32>,

    /// 昇順:
    /// eigenvalues.x = lambda_min
    /// eigenvalues.y = lambda_mid
    /// eigenvalues.z = lambda_max
    pub eigenvalues: Vector3<f32>,

    /// 法線方向の残差分散。
    pub residual_variance: f32,

    /// 現在の平面を支持した点数。
    pub support_count: u32,

    /// 現在の平面を支持した異なるフレーム数。
    pub observed_frame_count: u32,

    /// 逐次更新で使う有効重み。
    pub effective_weight: f32,

    pub first_frame_id: u64,
    pub last_frame_id: u64,
}

#[derive(Debug)]
pub struct GlobalPlaneNode {
    pub voxel_center: Point3<f32>,
    pub voxel_size: f32,
    pub depth: u8,

    // pub node_id: PlaneNodeId,
    pub plane_state: PlaneState,

    pub support_samples: Vec<GlobalSample>,     // Inliers
    pub outlier_samples: Vec<GlobalSample>,

    pub plane: Option<PlaneEstimate>,

    // pub observations: ObservationStats,
    pub children: Option<Box<[GlobalPlaneNode; 8]>>,
}

#[derive(Debug, Clone, Copy)]
pub struct OutputPoint {
    pub position: Point3<f32>,
    pub frame_id: u64,

    pub plane_residual: f32,
    // pub source_plane: Option<PlaneNodeId>,
}

#[derive(Debug, Clone)]
pub struct OutputVoxel {
    pub representative: OutputPoint,

    pub observations: u32,

    pub first_frame_id: u64,
    pub last_frame_id: u64,
}

#[derive(Debug, Clone)]
pub struct GlobalMapConfig {
    // ──────────────────────────────
    // 階層ボクセル
    // ──────────────────────────────
    /// PlaneMapのルートサイズ。
    ///
    /// 例: 0.4 m
    pub root_voxel_size: f32,

    /// 最大オクトリー深度。
    ///
    /// root=0.4 m、depth=2なら:
    /// depth 0: 0.4 m
    /// depth 1: 0.2 m
    /// depth 2: 0.1 m
    pub max_depth: u8,

    /// 最終点群用の固定ボクセルサイズ。
    ///
    /// 例: 0.1 m
    pub output_voxel_size: f32,

    // ──────────────────────────────
    // 点群保持数
    // ──────────────────────────────
    /// 各ノードが平面推定用に保持する最大点数。
    ///
    /// 例: 32～64点
    pub max_support_samples_per_node: usize,

    /// 現在の平面に入らなかった点を保持する上限。
    ///
    /// 子ノード作成や再RANSACに使用。
    pub max_outlier_samples_per_node: usize,

    /// 同じフレームから1ノードへ保存する最大点数。
    ///
    /// 例: 2～4点
    pub max_samples_per_frame_per_node: usize,

    // ──────────────────────────────
    // 平面確定条件
    // ──────────────────────────────
    pub min_points_for_ransac: usize,
    pub min_observed_frames: u32,

    pub ransac_distance_threshold: f32,
    pub stable_plane_distance_threshold: f32,

    pub min_ransac_inliers: usize,
    pub min_ransac_inlier_ratio: f32,

    /// lambda_min / lambda_mid の上限。
    pub max_planarity_ratio: f32,

    // ──────────────────────────────
    // 再構築条件
    // ──────────────────────────────
    /// アウトライアがこの数を超えたら、再RANSACや子ノード化を検討。
    pub rebuild_outlier_count: usize,

    /// 平面モデルを古い観測で固定しすぎないための有効重み上限。
    pub max_effective_weight: f32,
}

pub struct GlobalVoxelMap {
    pub map: HashMap<VoxelKey, GlobalPlaneNode>,

    pub output_voxel_map: HashMap<VoxelKey, OutputVoxel>,

    pub config: GlobalMapConfig,
}

#[inline]
pub fn voxel_key(p: &Point3<f32>, voxel_size: f32) -> VoxelKey {
    VoxelKey {
        ix: (p.x / voxel_size).floor() as i32,
        iy: (p.y / voxel_size).floor() as i32,
        iz: (p.z / voxel_size).floor() as i32,
    }
}

impl Default for GlobalMapConfig {
    fn default() -> Self {
        Self {
            root_voxel_size: 0.4,
            max_depth: 2,
            output_voxel_size: 0.1,

            max_support_samples_per_node: 24,
            max_outlier_samples_per_node: 24,
            max_samples_per_frame_per_node: 4,

            min_points_for_ransac: 16,
            min_observed_frames: 3,

            ransac_distance_threshold: 0.03,
            stable_plane_distance_threshold: 0.02,

            min_ransac_inliers: 10,
            min_ransac_inlier_ratio: 0.5,
            max_planarity_ratio: 0.15,

            rebuild_outlier_count: 16,
            max_effective_weight: 64.0,
        }
    }
}

impl GlobalVoxelMap {
    pub fn new(config: GlobalMapConfig) -> Self {
        Self {
            map: HashMap::new(),
            output_voxel_map: HashMap::new(),
            config,
        }
    }

    pub fn inserts_points(&mut self, points: &[Point3<f32>], global_pose: &Matrix4<f64>) {
        let pose_f32 = global_pose.cast::<f32>();
        let r_mat: Matrix3<f32> = pose_f32.fixed_view::<3, 3>(0, 0).into();
        let t_vec: Vector3<f32> = pose_f32.fixed_view::<3, 1>(0, 3).into();

        let voxel_size = self.config.root_voxel_size;

        let world_pts: Vec<(VoxelKey, Point3<f32>)> = points
            .par_iter()
            .map(|p| {
                let p_world = Point3::from(r_mat * p.coords + t_vec);
                let key = voxel_key(&p_world, voxel_size);
                (key, p_world)
            })
            .collect();
    }
}
