use anyhow::{Context, Result};
use plotters::prelude::*;
use re_lidar_slam::types::{FrameLog, FrameTiming};

const DEFAULT_JSON: &str = "data/output/debug/07112026/park06/frame_logs.json";

fn main() -> Result<()> {
    let json_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_JSON.to_string());

    let json_str = std::fs::read_to_string(&json_path)
        .with_context(|| format!("Cannot read: {}", json_path))?;
    let logs: Vec<FrameLog> =
        serde_json::from_str(&json_str).context("Failed to parse frame_logs.json")?;

    println!("Loaded {} frames from {}", logs.len(), json_path);

    // Output directory = same folder as the JSON
    let out_dir = std::path::Path::new(&json_path)
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .to_string_lossy()
        .to_string();

    std::fs::create_dir_all(&out_dir)?;

    plot_metrics(&logs, &out_dir)?;
    plot_trajectory_xy(&logs, &out_dir)?;
    if logs
        .iter()
        .any(|frame| frame.timing.total_with_file_io_ms > 0.0)
    {
        plot_processing_times(&logs, &out_dir)?;
        print_timing_ranking(&logs);
    } else {
        println!("Timing charts skipped: this is a legacy log without timing data");
    }

    println!("Saved charts to {}/", out_dir);
    Ok(())
}

type TimingAccessor = fn(&FrameTiming) -> f64;

const TIMING_STAGES: [(&str, TimingAccessor); 13] = [
    ("load PCD", |t| t.load_pcd_ms),
    ("timestamps", |t| t.timestamps_ms),
    ("IMU predict", |t| t.imu_predict_ms),
    ("rotation traj.", |t| t.rotation_trajectory_ms),
    ("deskew", |t| t.deskew_ms),
    ("downsample", |t| t.downsample_ms),
    ("build maps", |t| t.build_source_maps_ms),
    ("ICP", |t| t.icp_ms),
    ("pose update", |t| t.pose_update_ms),
    ("global filter", |t| t.global_filter_ms),
    ("global update", |t| t.global_map_update_ms),
    ("surface", |t| t.delayed_surface_ms),
    ("local update", |t| t.local_map_update_ms),
];

const ICP_TIMING_STAGES: [(&str, TimingAccessor); 4] = [
    ("correspondence search", |t| t.icp_correspondence_search_ms),
    ("linear system", |t| t.icp_linear_system_ms),
    ("solver", |t| t.icp_solver_ms),
    ("other", |t| t.icp_other_ms),
];

fn plot_processing_times(logs: &[FrameLog], out_dir: &str) -> Result<()> {
    let path = format!("{}/processing_times.png", out_dir);
    let root = BitMapBackend::new(&path, (1600, 1400)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((3, 1));

    let processing: Vec<(f32, f32)> = logs
        .iter()
        .map(|frame| {
            (
                frame.frame_index as f32,
                frame.timing.processing_total_ms as f32,
            )
        })
        .collect();
    let with_io: Vec<(f32, f32)> = logs
        .iter()
        .map(|frame| {
            (
                frame.frame_index as f32,
                frame.timing.total_with_file_io_ms as f32,
            )
        })
        .collect();
    let total_y_max = processing
        .iter()
        .chain(with_io.iter())
        .map(|(_, value)| *value)
        .fold(0.0f32, f32::max)
        .max(1.0);
    let x_start = logs.first().map_or(0.0, |frame| frame.frame_index as f32);
    let mut x_end = logs.last().map_or(1.0, |frame| frame.frame_index as f32);
    if x_end <= x_start {
        x_end = x_start + 1.0;
    }
    let mut totals_chart = ChartBuilder::on(&panels[0])
        .caption("Per-frame processing time", ("sans-serif", 22).into_font())
        .margin(15)
        .x_label_area_size(35)
        .y_label_area_size(70)
        .build_cartesian_2d(x_start..x_end, 0f32..total_y_max * 1.08)?;
    totals_chart
        .configure_mesh()
        .x_desc("Frame")
        .y_desc("Wall time [ms]")
        .draw()?;
    totals_chart
        .draw_series(LineSeries::new(processing, BLUE.stroke_width(2)))?
        .label("processing (without PCD I/O)")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 25, y)], BLUE.stroke_width(2)));
    totals_chart
        .draw_series(LineSeries::new(with_io, RED.stroke_width(1)))?
        .label("total (with PCD I/O)")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 25, y)], RED.stroke_width(1)));
    totals_chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()?;

    let stage_stats: Vec<(f64, f64)> = TIMING_STAGES
        .iter()
        .map(|(_, get)| mean_and_percentile(logs, *get, 0.95))
        .collect();
    let stage_y_max = stage_stats
        .iter()
        .flat_map(|(mean, p95)| [*mean, *p95])
        .fold(0.0f64, f64::max)
        .max(1.0) as f32;
    let stage_count = TIMING_STAGES.len();
    let mut breakdown_chart = ChartBuilder::on(&panels[1])
        .caption(
            "Processing-stage timing (mean and p95)",
            ("sans-serif", 22).into_font(),
        )
        .margin(15)
        .x_label_area_size(75)
        .y_label_area_size(70)
        .build_cartesian_2d(-0.5f32..stage_count as f32 - 0.5, 0f32..stage_y_max * 1.12)?;
    breakdown_chart
        .configure_mesh()
        .x_labels(stage_count)
        .x_label_formatter(&|value| {
            let index = value.round() as isize;
            if index >= 0 && (index as usize) < TIMING_STAGES.len() {
                TIMING_STAGES[index as usize].0.to_string()
            } else {
                String::new()
            }
        })
        .x_desc("Stage")
        .y_desc("Wall time [ms]")
        .draw()?;
    breakdown_chart
        .draw_series(stage_stats.iter().enumerate().map(|(index, (mean, _))| {
            Rectangle::new(
                [
                    (index as f32 - 0.34, 0.0),
                    (index as f32 - 0.02, *mean as f32),
                ],
                BLUE.filled(),
            )
        }))?
        .label("mean")
        .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 15, y + 5)], BLUE.filled()));
    breakdown_chart
        .draw_series(stage_stats.iter().enumerate().map(|(index, (_, p95))| {
            Rectangle::new(
                [
                    (index as f32 + 0.02, 0.0),
                    (index as f32 + 0.34, *p95 as f32),
                ],
                RED.mix(0.7).filled(),
            )
        }))?
        .label("p95")
        .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 15, y + 5)], RED.mix(0.7).filled()));
    breakdown_chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()?;

    let icp_stats: Vec<(f64, f64)> = ICP_TIMING_STAGES
        .iter()
        .map(|(_, get)| mean_and_percentile(logs, *get, 0.95))
        .collect();
    let icp_y_max = icp_stats
        .iter()
        .flat_map(|(mean, p95)| [*mean, *p95])
        .fold(0.0f64, f64::max)
        .max(1.0) as f32;
    let mut icp_chart = ChartBuilder::on(&panels[2])
        .caption("ICP timing breakdown", ("sans-serif", 22).into_font())
        .margin(15)
        .x_label_area_size(55)
        .y_label_area_size(70)
        .build_cartesian_2d(
            -0.5f32..ICP_TIMING_STAGES.len() as f32 - 0.5,
            0f32..icp_y_max * 1.12,
        )?;
    icp_chart
        .configure_mesh()
        .x_labels(ICP_TIMING_STAGES.len())
        .x_label_formatter(&|value| {
            let index = value.round() as isize;
            if index >= 0 && (index as usize) < ICP_TIMING_STAGES.len() {
                ICP_TIMING_STAGES[index as usize].0.to_string()
            } else {
                String::new()
            }
        })
        .x_desc("ICP stage")
        .y_desc("Wall time [ms]")
        .draw()?;
    icp_chart.draw_series(icp_stats.iter().enumerate().map(|(index, (mean, _))| {
        Rectangle::new(
            [
                (index as f32 - 0.34, 0.0),
                (index as f32 - 0.02, *mean as f32),
            ],
            BLUE.filled(),
        )
    }))?;
    icp_chart.draw_series(icp_stats.iter().enumerate().map(|(index, (_, p95))| {
        Rectangle::new(
            [
                (index as f32 + 0.02, 0.0),
                (index as f32 + 0.34, *p95 as f32),
            ],
            RED.mix(0.7).filled(),
        )
    }))?;

    root.present()?;
    println!("  processing_times.png");
    Ok(())
}

fn print_timing_ranking(logs: &[FrameLog]) {
    let mut ranking: Vec<(&str, f64, f64)> = TIMING_STAGES
        .iter()
        .map(|(name, get)| {
            let (mean, p95) = mean_and_percentile(logs, *get, 0.95);
            (*name, mean, p95)
        })
        .collect();
    ranking.sort_by(|left, right| right.1.total_cmp(&left.1));

    println!("Timing bottleneck ranking [ms]:");
    for (rank, (name, mean, p95)) in ranking.iter().enumerate() {
        println!(
            "  {:>2}. {:<16} mean={:>9.3}, p95={:>9.3}",
            rank + 1,
            name,
            mean,
            p95
        );
    }

    println!("ICP timing breakdown [ms]:");
    for (name, get) in ICP_TIMING_STAGES {
        let (mean, p95) = mean_and_percentile(logs, get, 0.95);
        println!("  {:<22} mean={:>9.3}, p95={:>9.3}", name, mean, p95);
    }
}

fn mean_and_percentile(logs: &[FrameLog], get: TimingAccessor, percentile: f64) -> (f64, f64) {
    let mut values: Vec<f64> = logs
        .iter()
        .map(|frame| get(&frame.timing))
        .filter(|value| value.is_finite())
        .collect();
    values.sort_by(f64::total_cmp);
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let index = ((values.len() as f64 * percentile).ceil() as usize)
        .saturating_sub(1)
        .min(values.len() - 1);
    (mean, values[index])
}

// ---------------------------------------------------------------------------
// 4-panel metrics chart: RMSE / Translation / Rotation / Velocity
// ---------------------------------------------------------------------------
fn plot_metrics(logs: &[FrameLog], out_dir: &str) -> Result<()> {
    let path = format!("{}/frame_metrics.png", out_dir);
    let root = BitMapBackend::new(&path, (1400, 1200)).into_drawing_area();
    root.fill(&WHITE)?;

    let panels = root.split_evenly((4, 1));

    draw_line_panel(
        &panels[0],
        "RMSE  [m]",
        &BLUE,
        logs.iter()
            .filter_map(|l| l.rmse.map(|r| (l.frame_index as f32, r)))
            .collect(),
        logs.len(),
    )?;

    draw_line_panel(
        &panels[1],
        "Translation  [m / frame]",
        &RED,
        logs.iter()
            .map(|l| (l.frame_index as f32, l.translation_m as f32))
            .collect(),
        logs.len(),
    )?;

    draw_line_panel(
        &panels[2],
        "Rotation  [deg / frame]",
        &GREEN,
        logs.iter()
            .map(|l| (l.frame_index as f32, l.rotation_deg as f32))
            .collect(),
        logs.len(),
    )?;

    draw_line_panel(
        &panels[3],
        "Velocity  [m/s]",
        &RGBColor(180, 0, 200),
        logs.iter()
            .map(|l| (l.frame_index as f32, l.velocity_m_s as f32))
            .collect(),
        logs.len(),
    )?;

    root.present()?;
    println!("  frame_metrics.png");
    Ok(())
}

fn draw_line_panel(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    title: &str,
    color: &RGBColor,
    data: Vec<(f32, f32)>,
    n_frames: usize,
) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }

    let y_max = data
        .iter()
        .map(|&(_, v)| v)
        .fold(f32::NEG_INFINITY, f32::max);
    let y_min = data.iter().map(|&(_, v)| v).fold(f32::INFINITY, f32::min);
    let y_range = (y_max - y_min).max(1e-6);
    let y_lo = (y_min - y_range * 0.05).min(0.0);
    let y_hi = y_max + y_range * 0.05;

    let mut chart = ChartBuilder::on(area)
        .caption(title, ("sans-serif", 18).into_font())
        .margin(10)
        .x_label_area_size(28)
        .y_label_area_size(60)
        .build_cartesian_2d(0f32..n_frames as f32, y_lo..y_hi)?;

    chart
        .configure_mesh()
        .x_labels(10)
        .y_labels(6)
        .x_label_formatter(&|v| format!("{:.0}", v))
        .y_label_formatter(&|v| format!("{:.3}", v))
        .draw()?;

    chart.draw_series(LineSeries::new(data, color.stroke_width(1)))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// XY trajectory chart
// ---------------------------------------------------------------------------
fn plot_trajectory_xy(logs: &[FrameLog], out_dir: &str) -> Result<()> {
    let path = format!("{}/trajectory_xy.png", out_dir);
    let root = BitMapBackend::new(&path, (900, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    let xs: Vec<f64> = logs.iter().map(|l| l.pose_x).collect();
    let ys: Vec<f64> = logs.iter().map(|l| l.pose_y).collect();

    let x_min = xs.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = ys.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let pad = ((x_max - x_min).max(y_max - y_min) * 0.05).max(1.0);

    let mut chart = ChartBuilder::on(&root)
        .caption("Trajectory  XY  [m]", ("sans-serif", 22).into_font())
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(
            (x_min - pad) as f32..(x_max + pad) as f32,
            (y_min - pad) as f32..(y_max + pad) as f32,
        )?;

    chart
        .configure_mesh()
        .x_desc("X [m]")
        .y_desc("Y [m]")
        .draw()?;

    // Trajectory line
    chart.draw_series(LineSeries::new(
        xs.iter()
            .zip(ys.iter())
            .map(|(&x, &y)| (x as f32, y as f32)),
        BLUE.stroke_width(1),
    ))?;

    // Start marker (green circle)
    if let (Some(&sx), Some(&sy)) = (xs.first(), ys.first()) {
        chart
            .draw_series(std::iter::once(Circle::new(
                (sx as f32, sy as f32),
                6,
                GREEN.filled(),
            )))?
            .label("Start")
            .legend(|(x, y)| Circle::new((x, y), 5, GREEN.filled()));
    }

    // End marker (red circle)
    if let (Some(&ex), Some(&ey)) = (xs.last(), ys.last()) {
        chart
            .draw_series(std::iter::once(Circle::new(
                (ex as f32, ey as f32),
                6,
                RED.filled(),
            )))?
            .label("End")
            .legend(|(x, y)| Circle::new((x, y), 5, RED.filled()));
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    println!("  trajectory_xy.png");
    Ok(())
}
