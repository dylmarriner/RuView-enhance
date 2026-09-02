//! Minimal HTTP client for the live `GET /api/v1/sensing/latest` contract.
//!
//! Field shapes here were captured from a real running sensing-server
//! instance (2026-09-01, reached via SSH loopback through `ruv-mac-mini`),
//! not copied from `wifi-densepose-sensing-server`'s `SensingUpdate` struct —
//! the live deployed server and this worktree's checked-out source have
//! drifted (the live server additionally reports `persons` with named
//! 17-keypoint skeletons and `room_inference`, which the checked-out
//! `SensingUpdate` struct does not declare). This client deserializes only
//! the fields Layer 1 renders; unknown fields (`features`, `node_features`,
//! `signal_field`, per-node `amplitude`/`sync`) are ignored, not dropped
//! silently from a stricter contract — `serde` simply never asked for them.

use serde::Deserialize;
use std::time::Duration;
use thiserror::Error;

/// A single named skeletal keypoint, in the sensing-server's own image-plane
/// coordinates (not yet room/world coordinates).
#[derive(Debug, Clone, Deserialize)]
pub struct Keypoint {
    pub name: String,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BBox {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Person {
    pub id: u64,
    pub confidence: f64,
    pub bbox: BBox,
    pub keypoints: Vec<Keypoint>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NodeInference {
    pub classification: String,
    pub confidence: f64,
    pub age_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NodeInfo {
    pub node_id: u8,
    pub position: [f64; 3],
    pub rssi_dbm: f64,
    #[serde(default)]
    pub node_inference: Option<NodeInference>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ClassificationInfo {
    pub confidence: f64,
    pub motion_level: String,
    pub presence: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VitalSigns {
    #[serde(default)]
    pub breathing_rate_bpm: Option<f64>,
    #[serde(default)]
    pub heart_rate_bpm: Option<f64>,
    pub breathing_confidence: f64,
    pub heartbeat_confidence: f64,
    pub signal_quality: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RoomInference {
    pub classification: String,
    pub confidence: f64,
    pub contributing_nodes: u32,
}

/// The subset of the live `GET /api/v1/sensing/latest` response Layer 1
/// consumes. Real data, real field names, captured from a running instance —
/// not fabricated.
#[derive(Debug, Clone, Deserialize)]
pub struct SensingUpdate {
    #[serde(rename = "type")]
    pub msg_type: String,
    pub timestamp: f64,
    pub source: String,
    pub tick: u64,
    pub estimated_persons: u32,
    pub signal_quality_score: f64,
    pub classification: ClassificationInfo,
    #[serde(default)]
    pub vital_signs: Option<VitalSigns>,
    #[serde(default)]
    pub persons: Vec<Person>,
    #[serde(default)]
    pub nodes: Vec<NodeInfo>,
    #[serde(default)]
    pub room_inference: Option<RoomInference>,
}

#[derive(Debug, Error)]
pub enum SensingClientError {
    #[error("HTTP request to {url} failed: {source}")]
    Request {
        url: String,
        #[source]
        source: Box<ureq::Error>,
    },
    #[error("failed to decode sensing-server response body: {0}")]
    Decode(#[from] std::io::Error),
    #[error("failed to parse sensing-server JSON: {0}")]
    Json(#[from] serde_json::Error),
}

/// Polls one real live frame at a time from a sensing-server's
/// `GET /api/v1/sensing/latest`. Deliberately does not retry internally — the
/// caller (the render loop) decides whether a transient error means "hold
/// last good frame" or "surface the outage", matching ADR-352's rule that a
/// stale/unreachable sensor never gets silently papered over with a
/// fabricated frame.
pub struct SensingClient {
    url: String,
    agent: ureq::Agent,
}

impl SensingClient {
    #[must_use]
    pub fn new(base_url: impl Into<String>) -> Self {
        let base_url = base_url.into();
        let url = format!("{}/api/v1/sensing/latest", base_url.trim_end_matches('/'));
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(5))
            .build();
        Self { url, agent }
    }

    /// Fetch exactly one real live frame. Never fabricates a frame on error.
    pub fn fetch_latest(&self) -> Result<SensingUpdate, SensingClientError> {
        let body = self
            .agent
            .get(&self.url)
            .call()
            .map_err(|e| SensingClientError::Request {
                url: self.url.clone(),
                source: Box::new(e),
            })?
            .into_string()?;
        Ok(serde_json::from_str(&body)?)
    }
}
