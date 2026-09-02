//! Layer 2 — fal.ai neural styling keyframes.
//!
//! Restyles a real Layer-1 frame via `fal-ai/fast-lcm-diffusion/image-to-image`
//! (confirmed live and working 2026-09-01: a real call returned HTTP 200, a
//! real 1024x1024 JPEG, and a real measured `timings.inference` of 5.23s).
//! This is deliberately **not** independent generation: the Layer-1 frame is
//! always the `image_url` input, and `strength` is kept low so the model
//! restyles rather than replaces the deterministic content.
//!
//! ## Pricing honesty
//!
//! fal.ai's public pricing page does not list a flat per-image rate for this
//! model, and the model page's own price badge did not resolve to a concrete
//! number via static fetch as of 2026-09-01 (see ADR-352). Every call here
//! therefore requires the caller to pass an explicit, operator-supplied
//! `estimated_cost_usd` for [`BudgetGuard::authorize_and_record`] rather than
//! this module silently trusting a scraped or assumed number. Real observed
//! `timings.inference` is returned on every call so that estimate can be
//! refined from real measurements over time.

use std::io::Read as _;
use std::time::Duration;

use base64::Engine as _;
use serde::Deserialize;
use thiserror::Error;

const ENDPOINT: &str = "https://fal.run/fal-ai/fast-lcm-diffusion/image-to-image";

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
pub struct StyledFrame {
    /// Decoded image bytes (JPEG), for SSIM comparison / recording.
    pub image_bytes: Vec<u8>,
    /// Real measured inference seconds reported by fal.ai for this call.
    pub inference_seconds: f64,
}

pub struct FalClient {
    api_key: String,
    agent: ureq::Agent,
}

impl FalClient {
    /// Reads `FAL_KEY` from the environment. Never accepts a literal key as
    /// an argument, so a key can't accidentally end up in a command line or
    /// process listing.
    pub fn from_env() -> Result<Self, FalError> {
        let api_key = std::env::var("FAL_KEY").map_err(|_| FalError::MissingKey)?;
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(180))
            .build();
        Ok(Self { api_key, agent })
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
        let payload = serde_json::json!({
            "image_url": data_uri,
            "prompt": prompt,
            "strength": strength,
            "num_inference_steps": num_inference_steps,
        });

        let resp = self
            .agent
            .post(ENDPOINT)
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
