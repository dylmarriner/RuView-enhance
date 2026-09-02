//! ADR-352 fal.ai spend guard: a hard $15 cap on real Layer-2 API spend for
//! this task, tracked independently of any other agent's budget (e.g. the
//! unrelated `wifi-densepose-room-viz` H3 work has its own $10-20 cap).
//!
//! fal.ai's public pricing page does not list a flat per-image rate for
//! `fal-ai/fast-lcm-diffusion/image-to-image` (billed by GPU-second,
//! architecture-dependent — confirmed unclear from
//! `https://fal.ai/pricing` and the model's own page on 2026-09-01). This
//! guard does not trust a scraped number: callers pass the real
//! `estimated_cost_usd` for the call they are about to make (derived from a
//! conservative measured/estimated per-second rate x real observed
//! `timings.inference` seconds from a prior real call), and the guard is the
//! single place that decides whether cumulative real spend may proceed.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BudgetError {
    #[error("call would bring cumulative spend to ${projected:.4}, over the ${cap:.2} cap (already spent ${spent:.4})")]
    WouldExceedCap {
        projected: f64,
        cap: f64,
        spent: f64,
    },
    #[error("failed to read/write budget ledger at {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse budget ledger: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct Ledger {
    spent_usd: f64,
    calls: u64,
}

/// Persistent, file-backed spend guard. Every real fal.ai call must be
/// pre-authorized here before it is made, and recorded here immediately
/// after — so a crash mid-run cannot silently lose track of real spend.
pub struct BudgetGuard {
    cap_usd: f64,
    ledger_path: PathBuf,
    ledger: Ledger,
}

impl BudgetGuard {
    /// Load (or initialize) the ledger at `ledger_path` with a hard cap of
    /// `cap_usd`.
    pub fn load(ledger_path: impl AsRef<Path>, cap_usd: f64) -> Result<Self, BudgetError> {
        let ledger_path = ledger_path.as_ref().to_path_buf();
        let ledger = if ledger_path.exists() {
            let raw = fs::read_to_string(&ledger_path).map_err(|e| BudgetError::Io {
                path: ledger_path.display().to_string(),
                source: e,
            })?;
            serde_json::from_str(&raw)?
        } else {
            Ledger::default()
        };
        Ok(Self {
            cap_usd,
            ledger_path,
            ledger,
        })
    }

    #[must_use]
    pub fn spent_usd(&self) -> f64 {
        self.ledger.spent_usd
    }

    #[must_use]
    pub fn remaining_usd(&self) -> f64 {
        (self.cap_usd - self.ledger.spent_usd).max(0.0)
    }

    /// Authorize and record one real call costing `estimated_cost_usd`.
    /// Refuses (without mutating state) if it would exceed the cap.
    pub fn authorize_and_record(&mut self, estimated_cost_usd: f64) -> Result<(), BudgetError> {
        let projected = self.ledger.spent_usd + estimated_cost_usd;
        if projected > self.cap_usd {
            return Err(BudgetError::WouldExceedCap {
                projected,
                cap: self.cap_usd,
                spent: self.ledger.spent_usd,
            });
        }
        self.ledger.spent_usd = projected;
        self.ledger.calls += 1;
        self.save()
    }

    fn save(&self) -> Result<(), BudgetError> {
        let raw = serde_json::to_string_pretty(&self.ledger)?;
        fs::write(&self.ledger_path, raw).map_err(|e| BudgetError::Io {
            path: self.ledger_path.display().to_string(),
            source: e,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_call_that_would_exceed_cap() {
        let dir = std::env::temp_dir().join(format!("render-budget-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ledger_path = dir.join("ledger.json");
        let mut guard = BudgetGuard::load(&ledger_path, 1.0).unwrap();
        guard.authorize_and_record(0.6).unwrap();
        assert!(guard.authorize_and_record(0.6).is_err());
        assert!((guard.spent_usd() - 0.6).abs() < 1e-9);
        std::fs::remove_dir_all(&dir).ok();
    }
}
