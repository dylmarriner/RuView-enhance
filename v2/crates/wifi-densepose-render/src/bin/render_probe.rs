//! Real-data validation binary for ADR-352 Layer 1.
//!
//! Polls a real, live sensing-server a bounded number of times, renders each
//! frame deterministically, and writes both the raw sensor JSON and the
//! rendered PNG to disk — the first two of ADR-352's three required evidence
//! streams (raw sensor evidence, Layer-1-only visualization). The third
//! (Layer-1+2 composite) lands once Layer 2 exists.

use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;
use wifi_densepose_render::{render_frame, RenderConfig, SensingClient};

#[derive(Parser, Debug)]
struct Args {
    /// Base URL of the live sensing-server, e.g. http://localhost:3000
    #[arg(long, default_value = "http://localhost:3000")]
    sensing_url: String,

    /// Number of real frames to poll and render.
    #[arg(long, default_value_t = 10)]
    frames: u32,

    /// Delay between polls, milliseconds.
    #[arg(long, default_value_t = 200)]
    interval_ms: u64,

    /// Output directory for evidence artifacts.
    #[arg(long, default_value = "render-probe-out")]
    out_dir: PathBuf,
}

fn main() {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    fs::create_dir_all(&args.out_dir).expect("create output dir");
    let raw_dir = args.out_dir.join("raw");
    let frames_dir = args.out_dir.join("layer1");
    fs::create_dir_all(&raw_dir).expect("create raw dir");
    fs::create_dir_all(&frames_dir).expect("create layer1 dir");

    let client = SensingClient::new(&args.sensing_url);
    let cfg = RenderConfig::default();

    let mut ok = 0u32;
    let mut errs = 0u32;

    for i in 0..args.frames {
        match client.fetch_latest() {
            Ok(update) => {
                let raw_path = raw_dir.join(format!("frame_{i:04}_tick{}.json", update.tick));
                let raw_json = serde_json::to_string_pretty(&serde_json::json!({
                    "type": update.msg_type,
                    "timestamp": update.timestamp,
                    "source": update.source,
                    "tick": update.tick,
                    "estimated_persons": update.estimated_persons,
                    "classification": {
                        "confidence": update.classification.confidence,
                        "motion_level": update.classification.motion_level,
                        "presence": update.classification.presence,
                    },
                    "persons_count": update.persons.len(),
                    "nodes_count": update.nodes.len(),
                }))
                .unwrap();
                fs::write(&raw_path, raw_json).expect("write raw evidence");

                let pixmap = render_frame(&update, cfg);
                let png_path = frames_dir.join(format!("frame_{i:04}_tick{}.png", update.tick));
                pixmap.save_png(&png_path).expect("write layer1 png");

                println!(
                    "frame {i}: tick={} persons={} nodes={} motion={} -> {}",
                    update.tick,
                    update.persons.len(),
                    update.nodes.len(),
                    update.classification.motion_level,
                    png_path.display()
                );
                ok += 1;
            }
            Err(e) => {
                eprintln!("frame {i}: FETCH ERROR (no fabricated frame emitted): {e}");
                errs += 1;
            }
        }
        std::thread::sleep(Duration::from_millis(args.interval_ms));
    }

    println!("done: {ok} real frames rendered, {errs} fetch errors, output in {}", args.out_dir.display());
}
