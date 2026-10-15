//! Adapters from RuView CSI containers to [`Snapshots`].
//!
//! `wifi-densepose-signal` does not depend on `wifi-densepose-hardware`, so the
//! adapters take plain slices. For a hardware `CsiFrame`, pass
//! `frame.subcarriers.iter().map(|s| (s.i, s.q))` collected into a `Vec` plus
//! `frame.metadata.n_antennas`: the ESP32 parser stores I/Q **antenna-major**
//! (all subcarriers of antenna 0, then antenna 1, ...).
//!
//! Only phase-synchronous receive chains give meaningful AoA. A frame from a
//! single-antenna board has `n_antennas == 1` and is rejected by the estimators.

use super::{AoaError, Res, Snapshots};
use crate::csi_processor::CsiData;
use ndarray::Array3;
use num_complex::Complex64;

/// Build snapshots from raw I/Q frames (one slice per packet, antenna-major).
///
/// Every frame must have `n_antennas * n_subcarriers` pairs; the subcarrier
/// count is taken from the first frame and must agree across frames.
pub fn snapshots_from_iq_frames(n_antennas: usize, frames: &[&[(i16, i16)]]) -> Res<Snapshots> {
    if n_antennas == 0 {
        return Err(AoaError::InvalidParameter("n_antennas must be >= 1".into()));
    }
    let first = frames
        .first()
        .ok_or_else(|| AoaError::InsufficientData("no frames".into()))?;
    if first.is_empty() || first.len() % n_antennas != 0 {
        return Err(AoaError::ShapeMismatch(format!(
            "{} I/Q pairs not divisible by {n_antennas} antennas",
            first.len()
        )));
    }
    let k = first.len() / n_antennas;
    let mut data = Array3::<Complex64>::zeros((n_antennas, k, frames.len()));
    for (p, f) in frames.iter().enumerate() {
        if f.len() != n_antennas * k {
            return Err(AoaError::ShapeMismatch(format!(
                "frame {p} has {} pairs, expected {}",
                f.len(),
                n_antennas * k
            )));
        }
        for a in 0..n_antennas {
            for s in 0..k {
                let (i, q) = f[a * k + s];
                data[[a, s, p]] = Complex64::new(i as f64, q as f64);
            }
        }
    }
    Snapshots::new(data)
}

/// Build snapshots from processed [`CsiData`] frames (amplitude/phase,
/// `antennas x subcarriers`). All frames must share one shape.
pub fn snapshots_from_csi_data(frames: &[CsiData]) -> Res<Snapshots> {
    let first = frames
        .first()
        .ok_or_else(|| AoaError::InsufficientData("no frames".into()))?;
    let (m, k) = first.amplitude.dim();
    let mut data = Array3::<Complex64>::zeros((m, k, frames.len()));
    for (p, f) in frames.iter().enumerate() {
        if f.amplitude.dim() != (m, k) || f.phase.dim() != (m, k) {
            return Err(AoaError::ShapeMismatch(format!(
                "frame {p} is {:?}/{:?}, expected ({m}, {k})",
                f.amplitude.dim(),
                f.phase.dim()
            )));
        }
        for a in 0..m {
            for s in 0..k {
                data[[a, s, p]] = Complex64::from_polar(f.amplitude[[a, s]], f.phase[[a, s]]);
            }
        }
    }
    Snapshots::new(data)
}
