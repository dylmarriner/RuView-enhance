//! Real end-to-end ADR-352 Layer-1 + Layer-2 + SSIM-gate validation on a
//! bounded number of real live frames. Requires `FAL_KEY` in the
//! environment (fetch it from GCP Secret Manager yourself; this binary never
//! embeds or logs it).
//!
//! Every fal.ai call is pre-authorized by [`BudgetGuard`] against a hard
//! cap, using an operator-supplied `--fal-cost-estimate-usd` per call —
//! see `src/layer2.rs` for why that number isn't hardcoded from a scraped
//! price page.

use std::fs;
use std::path::PathBuf;

use clap::Parser;
use wifi_densepose_render::{render_frame, BudgetGuard, FalClient, RenderConfig, SensingClient};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "http://localhost:3000")]
    sensing_url: String,

    /// Number of real Layer-1 frames to also restyle through Layer 2.
    #[arg(long, default_value_t = 3)]
    frames: u32,

    #[arg(long, default_value = "layer2-probe-out")]
    out_dir: PathBuf,

    /// Hard cap on cumulative real fal.ai spend for this task (ADR-352: $15).
    #[arg(long, default_value_t = 15.0)]
    fal_budget_cap_usd: f64,

    /// Operator-supplied per-call cost estimate — fal.ai does not publish a
    /// flat rate for this model; see ADR-352 for the reasoning behind this
    /// default. Deliberately conservative (padded well above the ~$0.003-
    /// 0.01 order-of-magnitude estimate from one real measured call).
    #[arg(long, default_value_t = 0.02)]
    fal_cost_estimate_usd: f64,

    #[arg(long, default_value = "a real indoor room, photorealistic, natural lighting")]
    prompt: String,

    #[arg(long, default_value_t = 0.55)]
    strength: f32,

    #[arg(long, default_value_t = 4)]
    steps: u32,
}

fn main() {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    fs::create_dir_all(&args.out_dir).expect("create output dir");
    let layer1_dir = args.out_dir.join("layer1");
    let layer2_dir = args.out_dir.join("layer2");
    fs::create_dir_all(&layer1_dir).expect("create layer1 dir");
    fs::create_dir_all(&layer2_dir).expect("create layer2 dir");

    let fal = match FalClient::from_env() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("cannot run Layer 2 without a real FAL_KEY: {e}");
            std::process::exit(1);
        }
    };

    let ledger_path = args.out_dir.join("fal-budget-ledger.json");
    let mut budget = BudgetGuard::load(&ledger_path, args.fal_budget_cap_usd)
        .expect("load/init budget ledger");

    let client = SensingClient::new(&args.sensing_url);
    let cfg = RenderConfig::default();

    let mut styled_ok = 0u32;
    let mut rejected = 0u32;
    let mut real_inference_seconds_total = 0.0_f64;

    for i in 0..args.frames {
        let update = match client.fetch_latest() {
            Ok(u) => u,
            Err(e) => {
                eprintln!("frame {i}: sensing-server fetch failed, skipping (no fabricated frame): {e}");
                continue;
            }
        };

        let pixmap = render_frame(&update, cfg);
        let layer1_png = pixmap.encode_png().expect("encode layer1 png");
        let l1_path = layer1_dir.join(format!("frame_{i:04}_tick{}.png", update.tick));
        fs::write(&l1_path, &layer1_png).expect("write layer1 evidence");

        if let Err(e) = budget.authorize_and_record(args.fal_cost_estimate_usd) {
            eprintln!("frame {i}: BUDGET GUARD REFUSED the fal.ai call: {e}");
            eprintln!("stopping — real spend must never exceed the ADR-352 cap.");
            break;
        }

        match fal.style_frame(&layer1_png, &args.prompt, args.strength, args.steps) {
            Ok(styled) => {
                real_inference_seconds_total += styled.inference_seconds;
                let ssim = wifi_densepose_render::ssim_compare(&layer1_png, &styled.image_bytes)
                    .expect("ssim compare real bytes");

                let l2_path = layer2_dir.join(format!(
                    "frame_{i:04}_tick{}_ssim{:.3}_{}.jpg",
                    update.tick,
                    ssim.score,
                    if ssim.accepted { "accepted" } else { "REJECTED" }
                ));
                fs::write(&l2_path, &styled.image_bytes).expect("write layer2 evidence");

                println!(
                    "frame {i}: tick={} real_inference_s={:.2} ssim={:.3} {} -> {}",
                    update.tick,
                    styled.inference_seconds,
                    ssim.score,
                    if ssim.accepted { "ACCEPTED" } else { "REJECTED (holding last-good composite)" },
                    l2_path.display()
                );
                if ssim.accepted {
                    styled_ok += 1;
                } else {
                    rejected += 1;
                }
            }
            Err(e) => {
                eprintln!("frame {i}: real fal.ai call failed: {e}");
            }
        }
    }

    println!(
        "done: {styled_ok} accepted, {rejected} SSIM-rejected, real fal spend so far ${:.4} of ${:.2} cap, real cumulative measured inference {:.2}s",
        budget.spent_usd(),
        args.fal_budget_cap_usd,
        real_inference_seconds_total
    );
}
