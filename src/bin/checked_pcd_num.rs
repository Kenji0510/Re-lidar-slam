use std::{
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use re_lidar_slam::{
    file_handler::{load_pcd_files, load_pcd_xyzit},
    types::PointXYZIT,
};

const LOAD_DIR_AIRY96: &str = "data/input/08012026/08012026-airy96-mid70-08-church/airy/pcd";
const LOAD_DIR_MID70: &str = "data/input/08012026/08012026-airy96-mid70-08-church/mid-70/pcd";

const OUTPUT_ROOT: &str = "data/output/debug/checked_pcd";
const OUTPUT_DIR_AIRY96: &str = "data/output/debug/checked_pcd/airy/pcd";
const OUTPUT_DIR_MID70: &str = "data/output/debug/checked_pcd/mid-70/pcd";
const MATCHING_REPORT_PATH: &str = "data/output/debug/checked_pcd/matching_report.csv";

const MAX_TIME_DIFFERENCE_SECONDS: f64 = 0.050;

#[derive(Debug, Clone)]
struct FrameMetadata {
    source_path: PathBuf,
    source_number: u32,
    timestamp: f64,
}

#[derive(Debug, Clone, Copy)]
struct FramePair {
    airy_index: usize,
    mid70_index: usize,
    difference_seconds: f64,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let mut airy_frames = load_frame_metadata(LOAD_DIR_AIRY96)
        .with_context(|| format!("Failed to load Airy96 frames from {LOAD_DIR_AIRY96}"))?;
    let mut mid70_frames = load_frame_metadata(LOAD_DIR_MID70)
        .with_context(|| format!("Failed to load Mid-70 frames from {LOAD_DIR_MID70}"))?;

    airy_frames.sort_by(|a, b| a.timestamp.total_cmp(&b.timestamp));
    mid70_frames.sort_by(|a, b| a.timestamp.total_cmp(&b.timestamp));

    let pairs = match_frames(&airy_frames, &mid70_frames, MAX_TIME_DIFFERENCE_SECONDS);
    ensure!(
        !pairs.is_empty(),
        "No Airy96/Mid-70 frame pairs found within {:.1} ms",
        MAX_TIME_DIFFERENCE_SECONDS * 1_000.0
    );

    let output_root = Path::new(OUTPUT_ROOT);
    let output_airy = Path::new(OUTPUT_DIR_AIRY96);
    let output_mid70 = Path::new(OUTPUT_DIR_MID70);
    let report_path = Path::new(MATCHING_REPORT_PATH);

    fs::create_dir_all(output_airy)
        .with_context(|| format!("Failed to create {}", output_airy.display()))?;
    fs::create_dir_all(output_mid70)
        .with_context(|| format!("Failed to create {}", output_mid70.display()))?;
    ensure!(
        output_root.is_dir(),
        "{} is not a directory",
        output_root.display()
    );

    preflight_outputs(&pairs, &airy_frames, output_airy, output_mid70, report_path)?;

    for (position, pair) in pairs.iter().enumerate() {
        let airy = &airy_frames[pair.airy_index];
        let mid70 = &mid70_frames[pair.mid70_index];
        let output_name = output_filename(airy.source_number);

        copy_without_overwrite(&airy.source_path, &output_airy.join(&output_name))?;
        copy_without_overwrite(&mid70.source_path, &output_mid70.join(&output_name))?;

        if (position + 1) % 250 == 0 || position + 1 == pairs.len() {
            log::info!("Copied {}/{} frame pairs", position + 1, pairs.len());
        }
    }

    write_matching_report(report_path, &pairs, &airy_frames, &mid70_frames)?;

    let unmatched_airy = airy_frames.len() - pairs.len();
    let unmatched_mid70 = mid70_frames.len() - pairs.len();
    let maximum_difference_ms = pairs
        .iter()
        .map(|pair| pair.difference_seconds)
        .fold(0.0_f64, f64::max)
        * 1_000.0;

    log::info!(
        "Matching complete: {} pairs, {} unmatched Airy96 frames, {} unmatched Mid-70 frames, maximum difference {:.3} ms",
        pairs.len(),
        unmatched_airy,
        unmatched_mid70,
        maximum_difference_ms,
    );
    log::info!("Matching report: {}", report_path.display());

    Ok(())
}

fn load_frame_metadata(directory: &str) -> Result<Vec<FrameMetadata>> {
    let paths = load_pcd_files(directory)?;
    ensure!(
        !paths.is_empty(),
        "No cloud_<number>.pcd files found in {directory}"
    );

    paths
        .into_iter()
        .map(|path| {
            let source_number = parse_cloud_number(&path)?;
            let points = load_pcd_xyzit(&path.to_string_lossy())
                .with_context(|| format!("Failed to load {}", path.display()))?;
            let timestamp = representative_timestamp(&points)
                .with_context(|| format!("Invalid timestamps in {}", path.display()))?;

            Ok(FrameMetadata {
                source_path: path,
                source_number,
                timestamp,
            })
        })
        .collect()
}

fn parse_cloud_number(path: &Path) -> Result<u32> {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .with_context(|| format!("Invalid PCD filename: {}", path.display()))?;
    let number = stem
        .strip_prefix("cloud_")
        .with_context(|| format!("Expected cloud_<number>.pcd: {}", path.display()))?;

    number
        .parse::<u32>()
        .with_context(|| format!("Invalid cloud number in {}", path.display()))
}

fn representative_timestamp(points: &[PointXYZIT]) -> Result<f64> {
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;

    for timestamp in points.iter().map(|point| point.timestamp) {
        if timestamp.is_finite() {
            minimum = minimum.min(timestamp);
            maximum = maximum.max(timestamp);
        }
    }

    ensure!(
        minimum.is_finite() && maximum.is_finite(),
        "No finite timestamps"
    );
    Ok(minimum + (maximum - minimum) * 0.5)
}

/// Produces a maximum-cardinality, monotonic one-to-one matching for sorted timestamps.
fn match_frames(
    airy_frames: &[FrameMetadata],
    mid70_frames: &[FrameMetadata],
    tolerance_seconds: f64,
) -> Vec<FramePair> {
    let mut pairs = Vec::new();
    let mut airy_index = 0;
    let mut mid70_index = 0;

    while airy_index < airy_frames.len() && mid70_index < mid70_frames.len() {
        let difference = airy_frames[airy_index].timestamp - mid70_frames[mid70_index].timestamp;

        if difference.abs() <= tolerance_seconds {
            pairs.push(FramePair {
                airy_index,
                mid70_index,
                difference_seconds: difference.abs(),
            });
            airy_index += 1;
            mid70_index += 1;
        } else if difference < 0.0 {
            airy_index += 1;
        } else {
            mid70_index += 1;
        }
    }

    pairs
}

fn preflight_outputs(
    pairs: &[FramePair],
    airy_frames: &[FrameMetadata],
    output_airy: &Path,
    output_mid70: &Path,
    report_path: &Path,
) -> Result<()> {
    if report_path.exists() {
        bail!(
            "Refusing to overwrite existing report: {}",
            report_path.display()
        );
    }

    for pair in pairs {
        let output_name = output_filename(airy_frames[pair.airy_index].source_number);
        for path in [
            output_airy.join(&output_name),
            output_mid70.join(&output_name),
        ] {
            if path.exists() {
                bail!("Refusing to overwrite existing PCD: {}", path.display());
            }
        }
    }

    Ok(())
}

fn output_filename(airy_source_number: u32) -> String {
    format!("cloud_{airy_source_number}.pcd")
}

fn copy_without_overwrite(source: &Path, destination: &Path) -> Result<()> {
    let mut source_file = fs::File::open(source)
        .with_context(|| format!("Failed to open source PCD: {}", source.display()))?;
    let destination_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)
        .with_context(|| {
            format!(
                "Failed to create destination PCD without overwriting: {}",
                destination.display()
            )
        })?;
    let mut destination_file = BufWriter::new(destination_file);

    std::io::copy(&mut source_file, &mut destination_file).with_context(|| {
        format!(
            "Failed to copy {} to {}",
            source.display(),
            destination.display()
        )
    })?;
    destination_file
        .flush()
        .with_context(|| format!("Failed to flush {}", destination.display()))?;

    Ok(())
}

fn write_matching_report(
    report_path: &Path,
    pairs: &[FramePair],
    airy_frames: &[FrameMetadata],
    mid70_frames: &[FrameMetadata],
) -> Result<()> {
    let report_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(report_path)
        .with_context(|| format!("Failed to create report: {}", report_path.display()))?;
    let mut writer = BufWriter::new(report_file);

    writeln!(
        writer,
        "output_file,output_number,airy_source_file,airy_source_number,mid70_source_file,mid70_source_number,airy_timestamp,mid70_timestamp,time_difference_ms"
    )?;

    for pair in pairs {
        let airy = &airy_frames[pair.airy_index];
        let mid70 = &mid70_frames[pair.mid70_index];
        let output_name = output_filename(airy.source_number);
        writeln!(
            writer,
            "{},{},{},{},{},{},{:.9},{:.9},{:.6}",
            output_name,
            airy.source_number,
            airy.source_path.display(),
            airy.source_number,
            mid70.source_path.display(),
            mid70.source_number,
            airy.timestamp,
            mid70.timestamp,
            pair.difference_seconds * 1_000.0,
        )?;
    }

    writer
        .flush()
        .with_context(|| format!("Failed to flush report: {}", report_path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use re_lidar_slam::types::PointXYZIT;

    use super::{
        FrameMetadata, load_frame_metadata, match_frames, parse_cloud_number, preflight_outputs,
        representative_timestamp,
    };

    fn frame(number: u32, timestamp: f64) -> FrameMetadata {
        FrameMetadata {
            source_path: PathBuf::from(format!("cloud_{number}.pcd")),
            source_number: number,
            timestamp,
        }
    }

    fn point(timestamp: f64) -> PointXYZIT {
        PointXYZIT {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            intensity: 0.0,
            timestamp,
        }
    }

    #[test]
    fn representative_timestamp_uses_finite_range_midpoint() {
        let points = vec![point(f64::NAN), point(10.0), point(10.2), point(10.1)];
        assert!((representative_timestamp(&points).unwrap() - 10.1).abs() < 1e-12);
    }

    #[test]
    fn representative_timestamp_rejects_empty_or_non_finite_points() {
        assert!(representative_timestamp(&[]).is_err());
        assert!(representative_timestamp(&[point(f64::NAN)]).is_err());
    }

    #[test]
    fn matching_skips_leading_and_missing_frames_without_losing_alignment() {
        let airy = vec![
            frame(0, 0.0),
            frame(1, 0.1),
            frame(2, 0.2),
            frame(3, 0.3),
            frame(4, 0.4),
        ];
        let mid70 = vec![frame(10, 0.201), frame(12, 0.399)];

        let pairs = match_frames(&airy, &mid70, 0.05);
        let matched: Vec<(u32, u32)> = pairs
            .iter()
            .map(|pair| {
                (
                    airy[pair.airy_index].source_number,
                    mid70[pair.mid70_index].source_number,
                )
            })
            .collect();

        assert_eq!(matched, vec![(2, 10), (4, 12)]);
    }

    #[test]
    fn matching_rejects_out_of_tolerance_and_never_reuses_a_frame() {
        let airy = vec![frame(0, 1.0), frame(1, 1.01), frame(2, 2.0)];
        let mid70 = vec![frame(10, 1.005), frame(11, 2.051)];

        let pairs = match_frames(&airy, &mid70, 0.05);

        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].airy_index, 0);
        assert_eq!(pairs[0].mid70_index, 0);
    }

    #[test]
    fn parses_cloud_number_and_rejects_invalid_name() {
        assert_eq!(
            parse_cloud_number(PathBuf::from("cloud_23.pcd").as_path()).unwrap(),
            23
        );
        assert!(parse_cloud_number(PathBuf::from("frame_23.pcd").as_path()).is_err());
    }

    #[test]
    fn loading_rejects_empty_directory_and_unreadable_pcd() {
        let root = std::env::temp_dir().join(format!(
            "re_lidar_slam_checked_pcd_loading_{}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();

        assert!(load_frame_metadata(root.to_str().unwrap()).is_err());

        fs::write(root.join("cloud_0.pcd"), b"not a PCD file").unwrap();
        assert!(load_frame_metadata(root.to_str().unwrap()).is_err());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn preflight_rejects_existing_output() {
        let root = std::env::temp_dir().join(format!(
            "re_lidar_slam_checked_pcd_num_{}",
            std::process::id()
        ));
        let airy_output = root.join("airy");
        let mid70_output = root.join("mid70");
        fs::create_dir_all(&airy_output).unwrap();
        fs::create_dir_all(&mid70_output).unwrap();
        fs::write(airy_output.join("cloud_7.pcd"), b"existing").unwrap();

        let airy = vec![frame(7, 1.0)];
        let pairs = match_frames(&airy, &[frame(0, 1.0)], 0.05);
        let result = preflight_outputs(
            &pairs,
            &airy,
            &airy_output,
            &mid70_output,
            &root.join("matching_report.csv"),
        );

        assert!(result.is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
