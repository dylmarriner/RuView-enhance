//! `wifi-densepose-render` — ADR-352 real-time dual-renderer.
//!
//! Layer 1 (this crate, first milestone) is a **deterministic** local
//! renderer: it consumes real, live frames from the sensing-server's
//! `GET /api/v1/sensing/latest` JSON contract and rasterizes them with no
//! model inference and no randomness. It is the "sensor truth" layer that
//! ADR-352's Layer-2 fal.ai neural styling is only ever allowed to restyle,
//! never replace.
//!
//! Deliberately **not** sourced here: `ruview-twin` output. `ruview-twin` is
//! an explicit SYNTHETIC/L0 propagation *model* (ADR-315) — using it as if it
//! were live sensor state would misrepresent a simulation as measurement.

#![forbid(unsafe_code)]

pub mod budget;
pub mod layer1;
pub mod sensing_client;

pub use budget::{BudgetGuard, BudgetError};
pub use layer1::{render_frame, RenderConfig};
pub use sensing_client::{ClassificationInfo, NodeInfo, SensingClient, SensingUpdate, VitalSigns};
