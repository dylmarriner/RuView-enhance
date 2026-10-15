//! SpotFi-style joint AoA + time-of-flight MUSIC.
//!
//! Subcarriers act as a second virtual array: a path with delay `tau` adds the
//! phase `exp(-j 2 pi df l tau)` on subcarrier `l`. Sliding a window of `Ls`
//! subcarriers over the band gives `(K - Ls + 1)` coherent-decorrelating
//! snapshots (subcarrier smoothing) of an `M * Ls` vector whose steering
//! vector is `a(theta) (x) b(tau)`. A 2-D MUSIC scan then separates paths that
//! share an angle but differ in delay (or vice versa).
//!
//! Absolute ToF is **not** observable: the packet-detection/sampling-time
//! offset adds an unknown common delay, so only delay *differences* between
//! paths are meaningful (SpotFi removes a per-packet linear phase ramp first;
//! see [`super::calibration::remove_common_phase`], but that also removes the
//! mean delay, so run this estimator on data with the ramp left in only if the
//! chains are delay-synchronised).

use super::covariance::{covariance_of_columns, hermitian_eigen};
use super::estimators::{axis_at, pick_peaks};
use super::geometry::ArrayGeometry;
use super::{AoaError, Res, Snapshots};
use ndarray::Array2;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Configuration for [`spotfi_music`].
#[derive(Debug, Clone)]
pub struct SpotFiConfig {
    /// Subcarrier window length `Ls` (2..=K-1). Roughly K/2 is customary.
    pub window: usize,
    /// Number of paths to resolve.
    pub n_paths: usize,
    /// Subcarrier spacing (Hz); `None` derives it from the snapshot frequencies.
    pub subcarrier_spacing_hz: Option<f64>,
    /// Azimuth scan `(start, stop, n_points)` in radians.
    pub azimuth: (f64, f64, usize),
    /// Delay scan `(min_s, max_s, n_points)`.
    pub tof: (f64, f64, usize),
}

/// Joint AoA/ToF pseudo-spectrum, `values[[tof, azimuth]]`.
#[derive(Debug, Clone)]
pub struct JointSpectrum {
    /// Azimuth axis (rad).
    pub azimuth: Vec<f64>,
    /// Delay axis (s).
    pub tof_s: Vec<f64>,
    /// MUSIC pseudo-spectrum.
    pub values: Array2<f64>,
}

/// A joint peak.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointPeak {
    /// Azimuth (rad).
    pub azimuth: f64,
    /// Relative delay (s).
    pub tof_s: f64,
    /// Spectrum value.
    pub value: f64,
}

/// Joint AoA/ToF MUSIC; returns the spectrum and the `n_paths` strongest peaks.
pub fn spotfi_music(
    snap: &Snapshots,
    geom: &ArrayGeometry,
    cfg: &SpotFiConfig,
) -> Res<(JointSpectrum, Vec<JointPeak>)> {
    let (m, k, p) = snap.data().dim();
    if geom.n_antennas() != m {
        return Err(AoaError::ShapeMismatch(format!(
            "snapshots have {m} antennas, geometry has {}",
            geom.n_antennas()
        )));
    }
    if cfg.window < 2 || cfg.window >= k {
        return Err(AoaError::InvalidParameter(format!(
            "window must be in 2..{k}, got {}",
            cfg.window
        )));
    }
    let ls = cfg.window;
    let dim = m * ls;
    if cfg.n_paths == 0 || cfg.n_paths >= dim {
        return Err(AoaError::InvalidParameter(format!(
            "n_paths must be in 1..{dim}"
        )));
    }
    let df = match cfg.subcarrier_spacing_hz {
        Some(d) => d,
        None => {
            let f = snap.subcarrier_freqs_hz().ok_or_else(|| {
                AoaError::InvalidParameter(
                    "subcarrier spacing unknown: set it or attach subcarrier frequencies".into(),
                )
            })?;
            if f.len() < 2 {
                return Err(AoaError::InsufficientData("need >= 2 subcarriers".into()));
            }
            (f[f.len() - 1] - f[0]) / (f.len() - 1) as f64
        }
    };
    if !(df.is_finite() && df != 0.0) {
        return Err(AoaError::InvalidParameter("subcarrier spacing must be non-zero".into()));
    }
    let (a0, a1, na) = cfg.azimuth;
    let (t0, t1, nt) = cfg.tof;
    if na < 3 || nt < 3 || a1 <= a0 || t1 <= t0 || ![a0, a1, t0, t1].iter().all(|v| v.is_finite()) {
        return Err(AoaError::InvalidParameter("invalid azimuth/tof scan".into()));
    }
    let n_cols = (k - ls + 1) * p;
    if n_cols < 2 * cfg.n_paths {
        return Err(AoaError::InsufficientData(format!(
            "{n_cols} smoothed snapshots for {} paths; need at least {}",
            cfg.n_paths,
            2 * cfg.n_paths
        )));
    }
    let mut x = Array2::<Complex64>::zeros((dim, n_cols));
    let mut c = 0;
    for pk in 0..p {
        for k0 in 0..=(k - ls) {
            for a in 0..m {
                for l in 0..ls {
                    x[[a * ls + l, c]] = snap.data()[[a, k0 + l, pk]];
                }
            }
            c += 1;
        }
    }
    let eig = hermitian_eigen(&covariance_of_columns(&x)?)?;

    let azimuth: Vec<f64> = (0..na).map(|i| a0 + (a1 - a0) * i as f64 / (na - 1) as f64).collect();
    let tof_s: Vec<f64> = (0..nt).map(|i| t0 + (t1 - t0) * i as f64 / (nt - 1) as f64).collect();
    let steer: Vec<Vec<Complex64>> = azimuth.iter().map(|&az| geom.steering(az, 0.0)).collect();
    let mut values = Array2::<f64>::zeros((nt, na));
    let mut v = vec![Complex64::new(0.0, 0.0); dim];
    for (it, &tau) in tof_s.iter().enumerate() {
        let delay: Vec<Complex64> = (0..ls)
            .map(|l| Complex64::from_polar(1.0, -2.0 * PI * df * l as f64 * tau))
            .collect();
        for (ia, a) in steer.iter().enumerate() {
            for ai in 0..m {
                for l in 0..ls {
                    v[ai * ls + l] = a[ai] * delay[l];
                }
            }
            let mut den = 0.0;
            for col in cfg.n_paths..dim {
                let mut pr = Complex64::new(0.0, 0.0);
                for (i, vi) in v.iter().enumerate() {
                    pr += eig.vectors[[i, col]].conj() * vi;
                }
                den += pr.norm_sqr();
            }
            values[[it, ia]] = dim as f64 / den.max(1e-14 * dim as f64);
        }
    }
    let peaks = pick_peaks(&values, false, cfg.n_paths, 0.0)
        .into_iter()
        .map(|rp| JointPeak {
            azimuth: axis_at(&azimuth, rp.col, false),
            tof_s: axis_at(&tof_s, rp.row, false),
            value: rp.value,
        })
        .collect();
    Ok((
        JointSpectrum {
            azimuth,
            tof_s,
            values,
        },
        peaks,
    ))
}
