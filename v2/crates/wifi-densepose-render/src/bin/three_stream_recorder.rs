//! ADR-352 three-stream evidence recorder.
//!
//! Produces the three real, synchronized artifacts ADR-352 requires from any
//! validation run:
//!   (a) raw sensor evidence      -> `raw.jsonl` (one real SensingUpdate summary per line)
//!   (b) Layer-1-only video       -> `layer1.mp4`
//!   (c) Layer-1+2 composite      -> `composite.mp4` (SSIM-gated, SYNTHETIC-burned-in)
//!
//! ## Real-time capture vs. fal.ai latency (honest scope note)
//!
//! Layer-1 capture polls the live sensing-server at a fixed real-time
//! interval and never blocks on the network. fal.ai calls, observed to take
//! anywhere from ~5s to 90s+ (cold starts), are made *after* the fast
//! capture pass completes, at a bounded number of real recorded frame
//! indices, not inline per-frame. This keeps Layer-1 capture genuinely
//! real-time-paced while keeping fal.ai calls real and bounded; the
//! recorded `tick`/`timestamp` in `raw.jsonl` is what proves synchronization
//! (which real Layer-1 frame each composite frame corresponds to), not wall
//! -clock adjacency of when each output file was written. A fully-inline
//! live pipeline (never buffering Layer-1 ahead of Layer-2) is real
//! follow-up work once Layer 2 has a lower-latency endpoint.
//!
//! Composite frames only get the burned-in SYNTHETIC label once at least
//! one real Layer-2 keyframe has actually been accepted for that frame's
//! position in the sequence (never before Layer 2 has contributed
//! anything) — matches ADR-352's rule that AI-enhanced output is always
//! visibly labeled.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use clap::Parser;
use wifi_densepose_render::{render_frame, BudgetGuard, CompositorState, FalClient, RenderConfig, SensingClient, SensingUpdate};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "http://localhost:3000")]
    sensing_url: String,

    /// Total real Layer-1 frames to capture.
    #[arg(long, default_value_t = 24)]
    frames: u32,

    #[arg(long, default_value_t = 200)]
    interval_ms: u64,

    /// Make one real fal.ai call every N captured frames (0 disables Layer 2
    /// entirely -- composite == Layer 1 for the whole run).
    #[arg(long, default_value_t = 8)]
    fal_every: u32,

    #[arg(long, default_value = "three-stream-out")]
    out_dir: PathBuf,

    #[arg(long, default_value_t = 15.0)]
    fal_budget_cap_usd: f64,

    #[arg(long, default_value_t = 0.01)]
    fal_cost_estimate_usd: f64,

    #[arg(long, default_value = "a real indoor room, photorealistic, natural lighting")]
    prompt: String,

    #[arg(long, default_value_t = 0.5)]
    strength: f32,

    #[arg(long, default_value_t = 4)]
    steps: u32,

    /// How strongly the held/warped Layer-2 keyframe shows through the
    /// composite (0 = pure Layer 1, 1 = pure warped Layer 2).
    #[arg(long, default_value_t = 0.7)]
    mix: f32,

    /// Output video frame rate (derived from --interval-ms if omitted).
    #[arg(long)]
    fps: Option<u32>,
}

fn main() {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    let fps = args.fps.unwrap_or_else(|| (1000 / args.interval_ms.max(1)).max(1) as u32);

    fs::create_dir_all(&args.out_dir).expect("create out dir");
    let layer1_dir = args.out_dir.join("layer1_frames");
    let composite_dir = args.out_dir.join("composite_frames");
    fs::create_dir_all(&layer1_dir).expect("create layer1_frames dir");
    fs::create_dir_all(&composite_dir).expect("create composite_frames dir");
    let raw_path = args.out_dir.join("raw.jsonl");

    // ---- Pass 1: real-time Layer-1 capture, never blocked on the network ----
    let client = SensingClient::new(&args.sensing_url);
    let cfg = RenderConfig::default();
    let mut raw_lines = Vec::new();
    let mut updates: Vec<SensingUpdate> = Vec::new();
    let mut layer1_pngs: Vec<Vec<u8>> = Vec::new();

    for i in 0..args.frames {
        match client.fetch_latest() {
            Ok(update) => {
                let pixmap = render_frame(&update, cfg);
                let png = pixmap.encode_png().expect("encode layer1 png");
                fs::write(layer1_dir.join(format!("f{i:05}.png")), &png).expect("write layer1 frame");

                raw_lines.push(
                    serde_json::json!({
                        "frame_index": i,
                        "tick": update.tick,
                        "timestamp": update.timestamp,
                        "source": update.source,
                        "estimated_persons": update.estimated_persons,
                        "classification": {
                            "motion_level": update.classification.motion_level,
                            "presence": update.classification.presence,
                            "confidence": update.classification.confidence,
                        },
                        "persons_count": update.persons.len(),
                        "nodes_count": update.nodes.len(),
                        "vital_signs": update.vital_signs.as_ref().map(|v| serde_json::json!({
                            "breathing_rate_bpm": v.breathing_rate_bpm,
                            "heart_rate_bpm": v.heart_rate_bpm,
                        })),
                    })
                    .to_string(),
                );
                layer1_pngs.push(png);
                updates.push(update);
                println!("captured real frame {i}/{}", args.frames);
            }
            Err(e) => {
                eprintln!("frame {i}: real sensing-server fetch failed, skipping (no fabricated frame): {e}");
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(args.interval_ms));
    }
    fs::write(&raw_path, raw_lines.join("\n")).expect("write raw.jsonl");
    println!("pass 1 done: {} real frames captured -> {}", updates.len(), raw_path.display());

    if updates.is_empty() {
        eprintln!("no real frames captured; nothing to encode.");
        std::process::exit(1);
    }

    // ---- Pass 2: bounded real fal.ai calls at fixed real frame indices ----
    let mut keyframe_bytes: Vec<Option<Vec<u8>>> = vec![None; updates.len()];
    if args.fal_every > 0 {
        match FalClient::from_env() {
            Ok(fal) => {
                let ledger_path = args.out_dir.join("fal-budget-ledger.json");
                let mut budget = BudgetGuard::load(&ledger_path, args.fal_budget_cap_usd).expect("load budget ledger");
                let mut idx = 0usize;
                while idx < updates.len() {
                    if let Err(e) = budget.authorize_and_record(args.fal_cost_estimate_usd) {
                        eprintln!("budget guard refused further fal.ai calls at frame {idx}: {e}");
                        break;
                    }
                    match fal.style_frame(&layer1_pngs[idx], &args.prompt, args.strength, args.steps) {
                        Ok(styled) => {
                            let ssim = wifi_densepose_render::ssim_compare(&layer1_pngs[idx], &styled.image_bytes)
                                .expect("ssim compare real bytes");
                            println!(
                                "frame {idx}: real fal call, inference_s={:.2} ssim={:.3} {}",
                                styled.inference_seconds,
                                ssim.score,
                                if ssim.accepted { "ACCEPTED" } else { "REJECTED (holding prior keyframe)" }
                            );
                            if ssim.accepted {
                                keyframe_bytes[idx] = Some(styled.image_bytes);
                            }
                        }
                        Err(e) => eprintln!("frame {idx}: real fal.ai call failed, no keyframe update: {e}"),
                    }
                    idx += args.fal_every as usize;
                }
                println!("pass 2 done: real fal spend ${:.4} of ${:.2} cap", budget.spent_usd(), args.fal_budget_cap_usd);
            }
            Err(e) => {
                eprintln!("no FAL_KEY: {e} -- composite stream will equal Layer 1 for this run (no fabricated Layer-2 content).");
            }
        }
    }

    // ---- Pass 3: compose, in real frame order, burning the SYNTHETIC label
    //      once real Layer-2 content has actually been accepted ----
    let mut compositor = CompositorState::new();
    for (i, l1_png) in layer1_pngs.iter().enumerate() {
        let l1_img = image::load_from_memory(l1_png).expect("decode real layer1 png").to_rgba8();
        if let Some(kf_bytes) = &keyframe_bytes[i] {
            let kf_img = image::load_from_memory(kf_bytes)
                .expect("decode real fal.ai jpeg")
                .resize_exact(l1_img.width(), l1_img.height(), image::imageops::FilterType::Triangle)
                .to_rgba8();
            compositor.accept_keyframe(kf_img, &l1_img);
        }
        let mut composite = compositor.composite(&l1_img, args.mix);
        if compositor.has_keyframe() {
            wifi_densepose_render::text::stamp(&mut composite, 8, l1_img.height() as i64 - 26, "SYNTHETIC", 2, (255, 210, 0), 1.0);
        }
        let out_path = composite_dir.join(format!("f{i:05}.png"));
        composite.save(&out_path).expect("write composite frame");
    }
    println!("pass 3 done: {} composite frames -> {}", layer1_pngs.len(), composite_dir.display());

    // ---- Pass 4: real ffmpeg encode of both PNG sequences to real mp4s ----
    for (name, dir) in [("layer1.mp4", &layer1_dir), ("composite.mp4", &composite_dir)] {
        let out = args.out_dir.join(name);
        let status = Command::new("ffmpeg")
            .args([
                "-y",
                "-framerate",
                &fps.to_string(),
                "-i",
            ])
            .arg(dir.join("f%05d.png"))
            .args(["-pix_fmt", "yuv420p", "-c:v", "libx264"])
            .arg(&out)
            .status();
        match status {
            Ok(s) if s.success() => println!("encoded {}", out.display()),
            Ok(s) => eprintln!("ffmpeg exited {s} encoding {}", out.display()),
            Err(e) => eprintln!("failed to run ffmpeg for {}: {e}", out.display()),
        }
    }

    println!(
        "done. three real evidence streams in {}: raw.jsonl, layer1.mp4, composite.mp4",
        args.out_dir.display()
    );
}
