//! ADR-351: near-real-time generative room visualization from live RuView
//! sensing state via fal.ai `fast-sdxl`.
//!
//! Deliberate, explicitly-authorized exception to this project's default
//! local-data-only posture: real presence/motion classification derived
//! from a live household's WiFi sensing feed is sent to a third-party
//! hosted generative-image API. See docs/adr/ADR-351-*.md for the full
//! privacy/consent rationale. This binary does not transmit raw CSI or any
//! biometric time series — only the already-classified, coarse scene state
//! (`presence`, `motion_level`, `estimated_persons`) crosses the boundary.

use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use serde::Deserialize;

/// Selectable visual style preset. Each produces a genuinely distinct,
/// hand-crafted prompt template — not a shared base prompt with a style
/// suffix appended. See docs/adr/ADR-351-*.md for the full rationale.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum Style {
    /// High-end interior-design-magazine aesthetic. Literal room depiction.
    Architectural,
    /// Dramatic film-still aesthetic. Literal room depiction.
    Cinematic,
    /// Generative light/form art. Never depicts a literal human figure —
    /// a real, deliberate privacy property, not just an aesthetic choice.
    Abstract,
    /// Clean technical/dashboard line-art. Literal room depiction, abstracted
    /// into an occupancy-status diagram rather than a photoreal scene.
    Minimal,
    /// Matches RuView's own marketing hero graphic via image-to-image style
    /// reference: a glowing cyan/blue sci-fi data-overlay aesthetic with a
    /// visible human pose-skeleton figure, WiFi signal arcs, and floating
    /// vital-sign readouts. UNLIKE every other style here, this ONE
    /// deliberately depicts a literal human silhouette with body-tracking
    /// overlay, a materially different, weaker privacy posture than
    /// `Abstract`. The figure is fal.ai's own synthetic generation, not a
    /// reproduction of any real person's likeness, but callers who want to
    /// avoid any human-figure depiction should prefer `Abstract` instead.
    /// See docs/adr/ADR-351-*.md for the full privacy-framing note.
    Branded,
}

/// Coarse, shared scene-state abstraction. All four styles consume this
/// same struct; only the per-style prompt wording differs. This keeps the
/// sensing-to-meaning mapping honest and in one place, auditable
/// independent of any single style's prose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Activity {
    Vacant,
    OccupiedStill,
    OccupiedMoving,
}

#[derive(Debug, Clone, Copy)]
struct SceneState {
    activity: Activity,
    multiple_occupants: bool,
}

#[derive(Debug, Deserialize)]
struct SensingLatest {
    classification: Classification,
    estimated_persons: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct Classification {
    presence: bool,
    motion_level: String,
}

/// Deterministic, honest mapping from the real RuView classification fields
/// to the shared coarse `SceneState`. No fabrication: `Vacant` iff
/// `presence == false`; `OccupiedMoving` iff `presence` and `motion_level`
/// contains "moving"; otherwise `OccupiedStill`.
fn map_scene_state(latest: &SensingLatest) -> SceneState {
    let activity = if !latest.classification.presence {
        Activity::Vacant
    } else if latest.classification.motion_level.contains("moving") {
        Activity::OccupiedMoving
    } else {
        Activity::OccupiedStill
    };
    let multiple_occupants = latest.estimated_persons.unwrap_or(0) >= 2;
    SceneState {
        activity,
        multiple_occupants,
    }
}

const NEGATIVE_PROMPT: &str =
    "blurry, low quality, distorted anatomy, extra limbs, watermark, text, oversaturated, cartoon, deformed";

/// Real, hand-crafted prompt templates per style. Each style's language is
/// genuinely distinct in composition, lighting vocabulary, and color
/// language — not a shared base string with a style tag appended.
fn build_prompt(style: Style, scene: SceneState) -> String {
    match style {
        Style::Architectural => {
            let (room_desc, light_desc) = match (scene.activity, scene.multiple_occupants) {
                (Activity::Vacant, _) => (
                    "minimalist living room, perfectly styled and unoccupied",
                    "soft diffused daylight through sheer curtains, calm and still atmosphere",
                ),
                (Activity::OccupiedStill, false) => (
                    "elegant living room with a person seated calmly reading",
                    "warm afternoon light, quiet domestic stillness",
                ),
                (Activity::OccupiedStill, true) => (
                    "elegant living room with two people seated in quiet conversation",
                    "warm afternoon light, relaxed shared stillness",
                ),
                (Activity::OccupiedMoving, false) => (
                    "sophisticated living room with a person in gentle motion, mid-stride",
                    "dynamic natural light, sense of quiet daily life in motion",
                ),
                (Activity::OccupiedMoving, true) => (
                    "sophisticated living room with two people moving through the space together",
                    "dynamic natural light, lively shared domestic motion",
                ),
            };
            format!(
                "professional interior design magazine photograph of a {room_desc}, {light_desc}, \
                 architectural digest style, considered symmetrical composition, neutral \
                 sophisticated color palette of warm greige and soft white, natural light study, \
                 shot on medium format camera, ultra sharp focus, 8k editorial quality"
            )
        }
        Style::Cinematic => {
            let (room_desc, light_desc) = match (scene.activity, scene.multiple_occupants) {
                (Activity::Vacant, _) => (
                    "dimly lit empty living room at dusk",
                    "a single warm lamp glowing in the corner, long shadows, nobody home, melancholic stillness",
                ),
                (Activity::OccupiedStill, false) => (
                    "living room with a silhouetted figure sitting motionless in an armchair",
                    "low warm key light from a single source, heavy shadow, introspective quiet mood",
                ),
                (Activity::OccupiedStill, true) => (
                    "living room with two silhouetted figures seated in stillness, one lit, one in shadow",
                    "low warm key light, quiet tension in the composition",
                ),
                (Activity::OccupiedMoving, false) => (
                    "living room with a figure caught mid-movement, motion blur trailing",
                    "dramatic side lighting, high contrast, sense of restless energy",
                ),
                (Activity::OccupiedMoving, true) => (
                    "living room with two figures in motion, crossing paths",
                    "dramatic side lighting, high contrast, kinetic tension between the figures",
                ),
            };
            format!(
                "cinematic film still of a {room_desc}, {light_desc}, shot on anamorphic lens, \
                 shallow depth of field, dramatic chiaroscuro lighting, rich teal and amber color \
                 grade, 35mm film grain, A24 movie aesthetic, moody atmospheric interior"
            )
        }
        Style::Abstract => {
            // Deliberately never depicts a literal human figure — a real
            // privacy property of this preset, not only an aesthetic one.
            let (presence_desc, form_desc) = match (scene.activity, scene.multiple_occupants) {
                (Activity::Vacant, _) => (
                    "an empty stable space",
                    "a single calm dim ember of light floating motionless in the void, deep blue tones",
                ),
                (Activity::OccupiedStill, false) => (
                    "a stable occupied space",
                    "one warm glowing orb of soft amber light, gently pulsing, centered and steady",
                ),
                (Activity::OccupiedStill, true) => (
                    "a stable, multiply occupied space",
                    "two warm glowing orbs of soft amber light, gently pulsing near one another",
                ),
                (Activity::OccupiedMoving, false) => (
                    "an active occupied space",
                    "a warm glowing form trailing motion streaks of light, dynamic flowing energy, orange and gold gradients",
                ),
                (Activity::OccupiedMoving, true) => (
                    "an active, multiply occupied space",
                    "two glowing forms of light trailing motion streaks, interacting, warm complementary hues",
                ),
            };
            format!(
                "abstract generative art visualization of {presence_desc}, {form_desc}, soft \
                 volumetric light, glowing particle field, dark charcoal background, ambient \
                 occlusion, high-end data visualization aesthetic, generative light installation \
                 style, no human figures, pure light and form"
            )
        }
        Style::Minimal => {
            let (room_desc, status_desc) = match (scene.activity, scene.multiple_occupants) {
                (Activity::Vacant, _) => (
                    "an empty room floor plan",
                    "all zones shown inactive, cool grey tones, status: vacant",
                ),
                (Activity::OccupiedStill, false) => (
                    "a room floor plan with one occupancy marker",
                    "single stationary indicator glowing soft cyan, status: occupied, stationary",
                ),
                (Activity::OccupiedStill, true) => (
                    "a room floor plan with two occupancy markers",
                    "two stationary indicators glowing soft cyan, status: occupied, stationary",
                ),
                (Activity::OccupiedMoving, false) => (
                    "a room floor plan with one occupancy marker and a motion trail",
                    "indicator glowing cyan with a directional motion trail, status: occupied, active",
                ),
                (Activity::OccupiedMoving, true) => (
                    "a room floor plan with two occupancy markers and motion trails",
                    "two indicators glowing cyan with directional motion trails, status: occupied, active",
                ),
            };
            format!(
                "minimalist technical line-art rendering of {room_desc}, {status_desc}, \
                 monochrome architectural wireframe, clean vector linework, muted slate and white \
                 palette, dashboard UI aesthetic, isometric technical diagram style, precise and \
                 clinical"
            )
        }
        Style::Branded => {
            // Matches RuView's own hero graphic via image-to-image style
            // reference (see `reference_image_data_uri`). This prompt is
            // paired with that reference at call time, not used alone.
            let status_desc = match (scene.activity, scene.multiple_occupants) {
                (Activity::Vacant, _) => {
                    "an empty room, WiFi signal arcs radiating from a router, no glowing human \
                     pose-skeleton figure present, status overlay reading VACANT"
                }
                (Activity::OccupiedStill, false) => {
                    "a room with one glowing cyan pose-skeleton human figure standing still, \
                     joint markers steady, WiFi signal arcs radiating from a router, floating \
                     vital-sign readout panels, status overlay reading OCCUPIED - STATIONARY"
                }
                (Activity::OccupiedStill, true) => {
                    "a room with two glowing cyan pose-skeleton human figures standing still, \
                     joint markers steady on both, WiFi signal arcs radiating from a router, \
                     floating vital-sign readout panels, status overlay reading MULTIPLE OCCUPANTS - STATIONARY"
                }
                (Activity::OccupiedMoving, false) => {
                    "a room with one glowing cyan pose-skeleton human figure mid-stride in motion, \
                     joint markers with motion trails, WiFi signal arcs radiating from a router, \
                     floating vital-sign readout panels, status overlay reading OCCUPIED - ACTIVE"
                }
                (Activity::OccupiedMoving, true) => {
                    "a room with two glowing cyan pose-skeleton human figures in motion, joint \
                     markers with motion trails on both, WiFi signal arcs radiating from a router, \
                     floating vital-sign readout panels, status overlay reading MULTIPLE OCCUPANTS - ACTIVE"
                }
            };
            format!(
                "{status_desc}, dark navy-blue background, glowing cyan and white sci-fi data \
                 overlay aesthetic, futuristic WiFi sensing visualization, technical HUD readout \
                 style, high contrast glow, RuView WiFi-DensePose branding aesthetic"
            )
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "room-viz", about = "ADR-351 near-real-time RuView -> fal.ai room visualization")]
struct Cli {
    /// Visual style preset.
    #[arg(long, value_enum)]
    style: Style,
    /// Number of frames to capture.
    #[arg(long, default_value_t = 12)]
    frames: u32,
    /// Seconds between polls of the live sensing feed.
    #[arg(long, default_value_t = 2)]
    interval_secs: u64,
    /// Output directory for frames and the assembled video.
    #[arg(long)]
    out_dir: PathBuf,
    /// SSH host that can reach the live sensing server on its own loopback.
    #[arg(long, default_value = "ruv-mac-mini")]
    sensing_ssh_host: String,
    /// Fixed seed for visual consistency across a capture session. If unset,
    /// a session-specific seed is required via this flag (no `SystemTime`
    /// use inside this binary's hot path — pass one explicitly).
    #[arg(long)]
    seed: i64,
    /// Path to a local reference image for image-to-image style matching.
    /// Only consumed when `--style branded`; ignored for every other style.
    #[arg(long)]
    style_ref_image: Option<PathBuf>,
    /// Image-to-image strength (0..1, higher = more influenced by the
    /// prompt vs. the reference image). Only used with `--style branded`.
    #[arg(long, default_value_t = 0.55)]
    style_ref_strength: f32,
}

/// Poll the real live sensing endpoint via SSH loopback (the server is not
/// reachable directly over Tailscale from this host — confirmed this
/// session). This shells out per-poll rather than holding a tunnel open,
/// matching how this whole project session has interacted with that host.
fn poll_sensing_latest(ssh_host: &str) -> Result<SensingLatest> {
    let output = Command::new("ssh")
        .args([
            "-o",
            "ConnectTimeout=8",
            ssh_host,
            "curl -s -m 5 http://127.0.0.1:3000/api/v1/sensing/latest",
        ])
        .output()
        .context("spawning ssh to poll the live sensing endpoint")?;
    if !output.status.success() {
        bail!(
            "ssh poll failed (exit {:?}): {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let body = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&body).with_context(|| format!("parsing sensing/latest JSON: {body}"))
}

#[derive(Debug, Deserialize)]
struct FalImageResponse {
    images: Vec<FalImage>,
}

#[derive(Debug, Deserialize)]
struct FalImage {
    url: String,
}

async fn generate_frame(
    client: &reqwest::Client,
    fal_key: &str,
    prompt: &str,
    seed: i64,
    reference_image_data_uri: Option<&str>,
    reference_strength: f32,
) -> Result<Vec<u8>> {
    let (endpoint, request_body) = match reference_image_data_uri {
        // Style::Branded: image-to-image against RuView's own hero graphic.
        // A base64 data: URI is accepted directly in `image_url` -- no
        // separate fal.ai storage upload call needed, confirmed via a real
        // test call before this was written.
        Some(image_url) => (
            "https://fal.run/fal-ai/fast-sdxl/image-to-image",
            serde_json::json!({
                "image_url": image_url,
                "prompt": prompt,
                "negative_prompt": NEGATIVE_PROMPT,
                "strength": reference_strength,
                "num_inference_steps": 8,
                "seed": seed,
                "format": "jpeg",
            }),
        ),
        None => (
            "https://fal.run/fal-ai/fast-sdxl",
            serde_json::json!({
                "prompt": prompt,
                "negative_prompt": NEGATIVE_PROMPT,
                "num_inference_steps": 6,
                "image_size": "landscape_16_9",
                "seed": seed,
                "format": "jpeg",
            }),
        ),
    };
    let response = client
        .post(endpoint)
        .header("Authorization", format!("Key {fal_key}"))
        .json(&request_body)
        .send()
        .await
        .context("calling fal.ai fast-sdxl")?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        bail!("fal.ai returned {status}: {body}");
    }
    let parsed: FalImageResponse = response.json().await.context("parsing fal.ai response")?;
    let image_url = parsed
        .images
        .first()
        .context("fal.ai response had no images")?
        .url
        .clone();
    let image_bytes = client
        .get(&image_url)
        .send()
        .await
        .context("downloading generated frame")?
        .bytes()
        .await
        .context("reading generated frame bytes")?;
    Ok(image_bytes.to_vec())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    let fal_key = std::env::var("FAL_KEY")
        .context("FAL_KEY env var must be set (fetch from GCP Secret Manager at the point of use, never commit it)")?;

    std::fs::create_dir_all(&cli.out_dir).context("creating output directory")?;
    let client = reqwest::Client::new();

    let reference_image_data_uri: Option<String> = if matches!(cli.style, Style::Branded) {
        let ref_path = cli
            .style_ref_image
            .as_ref()
            .context("--style-ref-image is required when --style branded")?;
        let bytes = std::fs::read(ref_path)
            .with_context(|| format!("reading reference image {}", ref_path.display()))?;
        use base64::Engine as _;
        let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
        Some(format!("data:image/jpeg;base64,{encoded}"))
    } else {
        None
    };

    let mut previous_activity: Option<Activity> = None;
    for frame_index in 0..cli.frames {
        let latest = poll_sensing_latest(&cli.sensing_ssh_host)?;
        let scene = map_scene_state(&latest);
        if Some(scene.activity) != previous_activity {
            tracing::info!(frame = frame_index, activity = ?scene.activity, "scene state changed");
        }
        previous_activity = Some(scene.activity);

        let prompt = build_prompt(cli.style, scene);
        tracing::info!(frame = frame_index, %prompt, "generating frame");

        let frame_bytes = generate_frame(
            &client,
            &fal_key,
            &prompt,
            cli.seed,
            reference_image_data_uri.as_deref(),
            cli.style_ref_strength,
        )
        .await?;
        let frame_path = cli.out_dir.join(format!("frame_{frame_index:04}.jpg"));
        std::fs::write(&frame_path, &frame_bytes)
            .with_context(|| format!("writing {}", frame_path.display()))?;
        tracing::info!(frame = frame_index, path = %frame_path.display(), bytes = frame_bytes.len(), "frame saved");

        if frame_index + 1 < cli.frames {
            tokio::time::sleep(Duration::from_secs(cli.interval_secs)).await;
        }
    }

    let video_path = cli.out_dir.join("room-viz.mp4");
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-framerate",
            "1",
            "-i",
        ])
        .arg(cli.out_dir.join("frame_%04d.jpg"))
        .args([
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "scale=1024:-2",
        ])
        .arg(&video_path)
        .status()
        .context("spawning ffmpeg")?;
    if !status.success() {
        bail!("ffmpeg exited with {status}");
    }
    println!("wrote {}", video_path.display());
    Ok(())
}
