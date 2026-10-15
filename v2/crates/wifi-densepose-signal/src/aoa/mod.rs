//! Phase-coherent angle-of-arrival (AoA) estimation for multi-antenna CSI.
//!
//! This module turns phase-synchronous multi-antenna CSI (antennas x
//! subcarriers x packets) into bearings using classical array processing:
//!
//! * [`geometry`] - arbitrary antenna positions, ULA/URA constructors, steering
//!   vectors, per-subcarrier wavelength.
//! * [`calibration`] - per-antenna (optionally per-subcarrier) complex offset
//!   estimation from a reference capture, and per-packet common-phase removal.
//! * [`covariance`] - sample covariance, forward-backward averaging, spatial
//!   smoothing.
//! * [`estimators`] - Bartlett, MVDR/Capon, MUSIC (1-D azimuth or 2-D
//!   azimuth/elevation), MDL/AIC source counting, peak picking with sub-grid
//!   refinement, and a one-call [`estimators::estimate_aoa`] pipeline.
//! * [`spotfi`] - joint AoA + time-of-flight MUSIC using subcarriers as a
//!   virtual array (SpotFi-style).
//! * [`fusion`] - least-squares ray intersection of bearings from several arrays.
//! * [`adapter`] - conversion from RuView CSI frames to [`Snapshots`].
//!
//! # Hardware caveat
//!
//! AoA needs **phase-coherent** receive chains (shared reference clock/LO), e.g.
//! an ESPARGOS-style ESP32 array, esp-crab style two-chip ESP32-C5 sync, or
//! multi-antenna NICs. A single ESP32-S3 board has one RF chain and cannot
//! produce AoA. Everything here is validated on **SYNTHETIC** plane-wave data
//! only; there is no hardware evidence. See `docs/aoa.md`.
//!
//! # Conventions
//!
//! Azimuth is measured counter-clockwise from the array +x axis, elevation from
//! the xy plane toward +z. A plane wave arriving *from* unit direction `u`
//! produces CSI phase `exp(+j k p.u)` at an antenna at position `p`
//! (`k = 2 pi / lambda`). A linear array only observes `cos(azimuth)`, so its
//! bearings are ambiguous about the array axis (scan `[0, pi]`).
//!
//! The algorithms are written from the standard array-processing literature
//! (Schmidt MUSIC, Capon, Wax-Kailath MDL, Shan-Wax-Kailath smoothing,
//! forward-backward averaging, SpotFi). No third-party code is copied.

pub mod adapter;
pub mod calibration;
pub mod covariance;
pub mod estimators;
pub mod fusion;
pub mod geometry;
pub mod spotfi;

#[cfg(test)]
mod tests;

use ndarray::Array3;
use num_complex::Complex64;

pub use adapter::{snapshots_from_csi_data, snapshots_from_iq_frames};
pub use calibration::{remove_common_phase, Calibration, Reference};
pub use covariance::{
    forward_backward, sample_covariance, spatial_smoothing, Covariance, Smoothing,
};
pub use estimators::{
    bartlett, estimate_aoa, estimate_num_sources, find_peaks, music, mvdr, AoaConfig, AoaResult,
    Method, Peak, ScanGrid, SourceCountCriterion, Spectrum,
};
pub use fusion::{locate_2d, ArrayPose, Bearing, PositionFix};
pub use geometry::ArrayGeometry;
pub use spotfi::{spotfi_music, JointPeak, JointSpectrum, SpotFiConfig};

/// Speed of light in vacuum (m/s).
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

/// Errors from AoA processing. Every public entry point returns these instead
/// of panicking on bad input.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum AoaError {
    /// Array/matrix dimensions do not agree.
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),
    /// Input contains NaN or infinity.
    #[error("non-finite value in {0}")]
    NonFinite(&'static str),
    /// Not enough snapshots / antennas for the requested operation.
    #[error("insufficient data: {0}")]
    InsufficientData(String),
    /// A parameter is out of range or inconsistent.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
    /// Geometry does not support the requested operation (e.g. smoothing on a
    /// non-grid array, forward-backward on an asymmetric array).
    #[error("unsupported geometry: {0}")]
    UnsupportedGeometry(String),
    /// Numerical failure (eigendecomposition, singular system).
    #[error("numerical failure: {0}")]
    Numerical(String),
}

/// Result alias for this module.
pub type Res<T> = Result<T, AoaError>;

/// Multi-antenna CSI snapshots: `data[[antenna, subcarrier, packet]]`.
///
/// Each `(subcarrier, packet)` column is one narrowband snapshot of the
/// `n_antennas`-vector. Subcarrier frequencies are optional metadata.
#[derive(Debug, Clone)]
pub struct Snapshots {
    data: Array3<Complex64>,
    subcarrier_freqs_hz: Option<Vec<f64>>,
}

impl Snapshots {
    /// Build from a validated array (finite, non-empty in every dimension).
    pub fn new(data: Array3<Complex64>) -> Res<Self> {
        let (m, k, p) = data.dim();
        if m == 0 || k == 0 || p == 0 {
            return Err(AoaError::InsufficientData(format!(
                "empty snapshot tensor ({m} antennas x {k} subcarriers x {p} packets)"
            )));
        }
        if data.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
            return Err(AoaError::NonFinite("snapshots"));
        }
        Ok(Self {
            data,
            subcarrier_freqs_hz: None,
        })
    }

    /// Attach absolute subcarrier frequencies (Hz); length must equal `n_subcarriers`.
    pub fn with_subcarrier_freqs(mut self, freqs_hz: Vec<f64>) -> Res<Self> {
        if freqs_hz.len() != self.n_subcarriers() {
            return Err(AoaError::ShapeMismatch(format!(
                "{} subcarrier frequencies for {} subcarriers",
                freqs_hz.len(),
                self.n_subcarriers()
            )));
        }
        if freqs_hz.iter().any(|f| !f.is_finite()) {
            return Err(AoaError::NonFinite("subcarrier frequencies"));
        }
        self.subcarrier_freqs_hz = Some(freqs_hz);
        Ok(self)
    }

    /// Number of antennas.
    pub fn n_antennas(&self) -> usize {
        self.data.dim().0
    }
    /// Number of subcarriers.
    pub fn n_subcarriers(&self) -> usize {
        self.data.dim().1
    }
    /// Number of packets.
    pub fn n_packets(&self) -> usize {
        self.data.dim().2
    }
    /// Total narrowband snapshots (`subcarriers * packets`).
    pub fn n_snapshots(&self) -> usize {
        self.n_subcarriers() * self.n_packets()
    }
    /// Raw tensor.
    pub fn data(&self) -> &Array3<Complex64> {
        &self.data
    }
    /// Subcarrier frequencies, if known.
    pub fn subcarrier_freqs_hz(&self) -> Option<&[f64]> {
        self.subcarrier_freqs_hz.as_deref()
    }
    pub(crate) fn from_parts(data: Array3<Complex64>, freqs: Option<Vec<f64>>) -> Self {
        Self {
            data,
            subcarrier_freqs_hz: freqs,
        }
    }
}
