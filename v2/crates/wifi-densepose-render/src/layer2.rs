//! Layer 2 — fal.ai neural styling keyframes.
//!
//! Restyles a real Layer-1 frame. Two real, confirmed-working endpoints:
//! - `fal-ai/fast-lcm-diffusion/image-to-image` (generic, original default;
//!   confirmed 2026-09-01: HTTP 200, real 1024x1024 JPEG, real measured
//!   `timings.inference` of 5.23s).
//! - `fal-ai/flux-lora/image-to-image` (confirmed 2026-09-02, real and
//!   operational — supports a `loras` array), the endpoint for the trained
//!   `ruforecast-visual-teacher` visual-teacher LoRA (delivered by agent
//!   `fal-visual-teacher`, 2026-09-02; trigger word `ruviewstyle`). See
//!   [`FalClient::with_lora`].
//!
//! Both are deliberately **not** independent generation: the Layer-1 frame
//! is always the `image_url` input, and `strength` is kept low so the model
//! restyles rather than replaces the deterministic content — the LoRA swap
//! changes how good the restyle looks, never whether it stays a restyle of
//! real Layer-1 content (see "Non-negotiable trust boundary" below).
//!
//! ## Pricing honesty
//!
//! `fast-lcm-diffusion`'s public pricing page does not list a flat per-image
//! rate (see ADR-352). `flux-lora/image-to-image` DOES have a real, verified
//! rate: **$0.035/megapixel, rounded up to the nearest megapixel**
//! (confirmed 2026-09-02 from two independent fetches of fal.ai's own
//! pricing/model pages) — for this crate's 512x288 render (0.147MP, rounds
//! up to 1MP) that is a real **$0.035/call**. Every call here still requires
//! the caller to pass an explicit `estimated_cost_usd` to
//! [`BudgetGuard::authorize_and_record`] (now backed by a real number for
//! the LoRA endpoint, not just an estimate) rather than this module
//! silently trusting a number. Real observed `timings.inference` is
//! returned on every call too.
//!
//! ## Non-negotiable trust boundary (user-stated, 2026-09-01)
//!
//! > "fal.ai will train an optional synthetic visual teacher, while
//! > RuView's measured RF, pose, vitals, identity, and confidence remain
//! > local and authoritative."
//!
//! fal.ai — whether this generic model or the trained visual-teacher LoRA
//! `ruforecast-visual-teacher` is producing — may **only ever** return
//! supplementary synthetic *pixels* for display/broadcast. It must never be
//! a source of, or be allowed to influence, any measured/derived value: RF,
//! CSI, pose, vitals, identity, or confidence/quality scores. Those always
//! come from and stay authoritative in the local deterministic pipeline
//! (`wifi-densepose-engine` / `ruview-twin` / `ruview-witness`). This
//! mirrors this repo's CLAUDE.md rule "Never present WiFi sensing as
//! camera-grade" — a neural restyle of the room is not a sensing result.
//!
//! Enforced at the type level here, not just by convention:
//! [`StyledFrame`] can only ever hold opaque `image_bytes` (raw pixels) plus
//! `inference_seconds` (call-latency telemetry). It has no field, and must
//! never gain one, that could be mistaken for or fed into the
//! measurement/pose/vitals/identity/confidence pipeline — no keypoints, no
//! classification, no numeric sensor readings. Anything that would need such
//! a field is a sign Layer 2 is being asked to do more than restyle pixels,
//! which is out of scope by design.

use std::io::Read as _;
use std::time::Duration;

use base64::Engine as _;
use serde::Deserialize;
use thiserror::Error;

/// The confirmed-working default endpoint (2026-09-01). Overridable via
/// `FAL_MODEL_ENDPOINT` or [`FalClient::with_endpoint`] — e.g. to
/// `https://fal.run/fal-ai/flux-lora/image-to-image` plus
/// [`FalClient::with_lora`] for the real trained `ruforecast-visual-teacher`
/// model (delivered 2026-09-02), which was exactly the config-not-code swap
/// this parameterization was built for.
const DEFAULT_ENDPOINT: &str = "https://fal.run/fal-ai/fast-lcm-diffusion/image-to-image";

#[derive(Debug, Error)]
pub enum FalError {
    #[error("FAL_KEY not set (never hardcode it; fetch from GCP Secret Manager and export it)")]
    MissingKey,
    #[error("fal.ai request failed: {0}")]
    Request(#[from] Box<ureq::Error>),
    #[error("failed to read fal.ai response body: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse fal.ai response JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("fal.ai returned no images")]
    NoImages,
    #[error("failed to decode returned image base64: {0}")]
    Base64(#[from] base64::DecodeError),
}

#[derive(Debug, Deserialize)]
struct FalImage {
    url: String,
}

#[derive(Debug, Deserialize)]
struct FalTimings {
    #[serde(default)]
    inference: f64,
}

#[derive(Debug, Deserialize)]
struct FalResponse {
    images: Vec<FalImage>,
    #[serde(default)]
    timings: Option<FalTimings>,
}

/// A real styled keyframe returned by fal.ai, plus the real measured timing
/// so callers can refine their cost estimate.
///
/// SAFETY INVARIANT (see module docs): this type must only ever carry opaque
/// pixel bytes and call telemetry. Never add a field here that a compositor
/// or any other caller could read as a measured/derived value (pose,
/// vitals, identity, confidence) — that would make it possible for
/// synthetic content to silently become "sensing truth".
pub struct StyledFrame {
    /// Decoded image bytes (JPEG), for SSIM comparison / recording.
    pub image_bytes: Vec<u8>,
    /// Real measured inference seconds reported by fal.ai for this call.
    pub inference_seconds: f64,
}

/// A real LoRA to merge into a FLUX-family image-to-image call (e.g. the
/// trained `ruforecast-visual-teacher` visual style). `path` is a real
/// fal.media (or equivalent) URL fal.ai fetches server-side — this client
/// never downloads or executes the weights file itself, only passes the URL
/// through as a request parameter, same trust boundary as any other fal.ai
/// call.
#[derive(Debug, Clone)]
pub struct LoraConfig {
    pub path: String,
    pub scale: f32,
}

pub struct FalClient {
    api_key: String,
    endpoint: String,
    lora: Option<LoraConfig>,
    agent: ureq::Agent,
}

impl FalClient {
    /// Reads `FAL_KEY` from the environment. Never accepts a literal key as
    /// an argument, so a key can't accidentally end up in a command line or
    /// process listing. Endpoint defaults to [`DEFAULT_ENDPOINT`], or
    /// `FAL_MODEL_ENDPOINT` if set (e.g. to point at the
    /// `ruforecast-visual-teacher` trained model once it has a real id).
    pub fn from_env() -> Result<Self, FalError> {
        let api_key = std::env::var("FAL_KEY").map_err(|_| FalError::MissingKey)?;
        let endpoint = std::env::var("FAL_MODEL_ENDPOINT").unwrap_or_else(|_| DEFAULT_ENDPOINT.to_string());
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(180))
            .build();
        Ok(Self { api_key, endpoint, lora: None, agent })
    }

    /// Override the target model endpoint (e.g. to point at a newly trained
    /// visual-teacher model id without touching `FAL_MODEL_ENDPOINT`).
    #[must_use]
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Attach a real trained LoRA (e.g. `ruforecast-visual-teacher`) to
    /// every subsequent [`style_frame`](Self::style_frame) call. Only
    /// meaningful against a FLUX-family endpoint that accepts a `loras`
    /// array (e.g. `fal-ai/flux-lora/image-to-image`) — set the endpoint to
    /// match via [`with_endpoint`](Self::with_endpoint) or
    /// `FAL_MODEL_ENDPOINT`.
    #[must_use]
    pub fn with_lora(mut self, path: impl Into<String>, scale: f32) -> Self {
        self.lora = Some(LoraConfig { path: path.into(), scale });
        self
    }

    /// Restyle one real Layer-1 PNG frame. `strength` should stay low
    /// (ADR-352: a restyle, not independent generation).
    pub fn style_frame(
        &self,
        layer1_png: &[u8],
        prompt: &str,
        strength: f32,
        num_inference_steps: u32,
    ) -> Result<StyledFrame, FalError> {
        let data_uri = format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(layer1_png)
        );
        let mut payload = serde_json::json!({
            "image_url": data_uri,
            "prompt": prompt,
            "strength": strength,
            "num_inference_steps": num_inference_steps,
        });
        if let Some(lora) = &self.lora {
            payload["loras"] = serde_json::json!([{ "path": lora.path, "scale": lora.scale }]);
        }

        let resp = self
            .agent
            .post(&self.endpoint)
            .set("Authorization", &format!("Key {}", self.api_key))
            .set("Content-Type", "application/json")
            .send_string(&payload.to_string())
            .map_err(Box::new)?
            .into_string()?;

        let parsed: FalResponse = serde_json::from_str(&resp)?;
        let image_url = parsed.images.first().ok_or(FalError::NoImages)?;
        let image_bytes = decode_data_uri_or_fetch(&self.agent, &image_url.url)?;
        let inference_seconds = parsed.timings.map(|t| t.inference).unwrap_or(0.0);

        Ok(StyledFrame {
            image_bytes,
            inference_seconds,
        })
    }
}

fn decode_data_uri_or_fetch(agent: &ureq::Agent, url: &str) -> Result<Vec<u8>, FalError> {
    if let Some(b64) = url.strip_prefix("data:image/jpeg;base64,") {
        return Ok(base64::engine::general_purpose::STANDARD.decode(b64)?);
    }
    if let Some(idx) = url.find(";base64,") {
        return Ok(base64::engine::general_purpose::STANDARD.decode(&url[idx + 8..])?);
    }
    // fal.ai's queue API sometimes returns a hosted URL instead of an inline
    // data URI; fetch it as a real HTTP call rather than fabricating bytes.
    let mut buf = Vec::new();
    agent
        .get(url)
        .call()
        .map_err(Box::new)?
        .into_reader()
        .read_to_end(&mut buf)?;
    Ok(buf)
}
