//! Anti-Goodhart evaluation harness for the WiFlow-STD WiFi-CSI pose benchmark.
//!
//! Milestone 1 of the action-conditioned dynamics objective (see `../STATE.md`).
//! Guards land before any modelling capability, deliberately: this repo has already
//! retracted one accuracy claim that a guard of this kind would have caught.
//!
//! ## What "validated" means here
//!
//! The harness is held to fixtures produced by upstream's own code path
//! (`scripts/make_fixtures.py`), and is checked two independent ways:
//!   a. score the dumped `(pred, gt)` and match `results/eval_retrained.json`
//!   b. rebuild ground truth from `all_keypoints.npy` with its own reader and
//!      zero-cleaning, and match the dumped ground truth bit-for-bit
//!
//! ## Reproducibility floor (MEASURED, not assumed)
//!
//! `scripts/diag_reproducibility_floor.py` measured that toggling TF32 alone moves
//! PCK@20 by 1.36e-5 and MPJPE by 5.7e-6 on this checkpoint and split, with the
//! recorded 2026-06-10 value falling between the TF32-on and TF32-off results.
//! So the forward pass is not bit-reproducible, and **no improvement smaller than
//! ~1.4e-5 PCK@20 is a real improvement**.

pub mod guards;
pub mod metrics;
pub mod npy;

/// Tolerance for comparing a metric against a recorded reference, set to the
/// measured reproducibility floor rounded up one significant figure.
pub const FLOOR_PCK: f64 = 2e-5;
/// See [`FLOOR_PCK`].
pub const FLOOR_MPJPE: f64 = 1e-5;
