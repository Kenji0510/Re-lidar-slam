use anyhow::{Context, Result};
use plotters::prelude::*;
use re_lidar_slam::types::FrameLog;

const DEFAULT_JSON: &str = "data/output/debug/07112026/park06/frame_logs.json";

fn main() -> Result<()> {
    let json_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_JSON.to_string());

    let json_str = std::fs::read_to_string(&json_path)
        .with_context(|| format!("Cannot read: {}", json_path))?;
    let logs: Vec<FrameLog> = serde_json::from_str(&json_str)
        .context("Failed to parse frame_logs.json")?;

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

    println!("Saved charts to {}/", out_dir);
    Ok(())
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
    let y_min = data
        .iter()
        .map(|&(_, v)| v)
        .fold(f32::INFINITY, f32::min);
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
        xs.iter().zip(ys.iter()).map(|(&x, &y)| (x as f32, y as f32)),
        BLUE.stroke_width(1),
    ))?;

    // Start marker (green circle)
    if let (Some(&sx), Some(&sy)) = (xs.first(), ys.first()) {
        chart.draw_series(std::iter::once(Circle::new(
            (sx as f32, sy as f32),
            6,
            GREEN.filled(),
        )))?
        .label("Start")
        .legend(|(x, y)| Circle::new((x, y), 5, GREEN.filled()));
    }

    // End marker (red circle)
    if let (Some(&ex), Some(&ey)) = (xs.last(), ys.last()) {
        chart.draw_series(std::iter::once(Circle::new(
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
