//! Per-antenna phase calibration and per-packet common-phase removal.
//!
//! Phase-synchronous arrays still carry fixed per-chain complex offsets
//! (cable length, PCB trace, LNA gain). They are estimated once from a
//! *reference capture* and divided out of every later snapshot:
//!
//! * [`Reference::Common`] - one reference signal is distributed to all
//!   antennas (the ESPARGOS-style calibration). Every antenna then sees the same
//!   waveform, so the dominant eigenvector of the covariance *is* the offset
//!   vector.
//! * [`Reference::Bearing`] - a source at a known bearing; the dominant
//!   eigenvector divided by the ideal steering vector is the offset vector.
//!
//! Offsets are referenced to antenna 0 (phase 0) and amplitude-normalised to
//! unit mean magnitude, since only relative values matter for AoA.

use super::covariance::{covariance_of_columns, hermitian_eigen};
use super::geometry::ArrayGeometry;
use super::{AoaError, Res, Snapshots};
use ndarray::{Array2, Array3};
use num_complex::Complex64;

/// How the reference capture was produced.
#[derive(Debug, Clone, Copy)]
pub enum Reference<'a> {
    /// A shared reference signal feeding every antenna identically.
    Common,
    /// A far-field source at a known bearing (radians) relative to `geometry`.
    Bearing {
        /// Array geometry.
        geometry: &'a ArrayGeometry,
        /// Azimuth (rad).
        azimuth: f64,
        /// Elevation (rad).
        elevation: f64,
    },
}

/// Per-antenna complex offsets `[antenna, subcarrier]`; a single subcarrier
/// column is broadcast to all subcarriers.
#[derive(Debug, Clone)]
pub struct Calibration {
    offsets: Array2<Complex64>,
}

impl Calibration {
    /// No-op calibration for `n_antennas`.
    pub fn identity(n_antennas: usize) -> Self {
        Self {
            offsets: Array2::from_elem((n_antennas, 1), Complex64::new(1.0, 0.0)),
        }
    }

    /// From explicit per-antenna offsets (applied as division).
    pub fn from_offsets(offsets: Vec<Complex64>) -> Res<Self> {
        if offsets.is_empty() {
            return Err(AoaError::InsufficientData("no offsets".into()));
        }
        validate_offsets(offsets.iter())?;
        let n = offsets.len();
        Ok(Self {
            offsets: Array2::from_shape_vec((n, 1), offsets)
                .map_err(|e| AoaError::ShapeMismatch(e.to_string()))?,
        })
    }

    /// Per-antenna offsets (first subcarrier column).
    pub fn antenna_offsets(&self) -> Vec<Complex64> {
        self.offsets.column(0).to_vec()
    }

    /// Full `[antenna, subcarrier]` offset table.
    pub fn offsets(&self) -> &Array2<Complex64> {
        &self.offsets
    }

    /// Estimate one offset per antenna from a reference capture (all
    /// subcarriers and packets pooled), or one per antenna *and* subcarrier when
    /// `per_subcarrier` is set (needs >= `n_antennas` packets, models
    /// frequency-dependent cable delay).
    pub fn estimate(snap: &Snapshots, reference: Reference<'_>, per_subcarrier: bool) -> Res<Self> {
        let m = snap.n_antennas();
        if let Reference::Bearing { geometry, .. } = reference {
            if geometry.n_antennas() != m {
                return Err(AoaError::ShapeMismatch(format!(
                    "geometry has {} antennas, capture has {m}",
                    geometry.n_antennas()
                )));
            }
        }
        let k_total = snap.n_subcarriers();
        let groups: Vec<Vec<usize>> = if per_subcarrier {
            (0..k_total).map(|k| vec![k]).collect()
        } else {
            vec![(0..k_total).collect()]
        };
        let mut offsets = Array2::<Complex64>::zeros((m, groups.len()));
        for (g, ks) in groups.iter().enumerate() {
            let n = ks.len() * snap.n_packets();
            if n < m {
                return Err(AoaError::InsufficientData(format!(
                    "calibration group has {n} snapshots for {m} antennas; need at least {m}"
                )));
            }
            let mut cols = Array2::<Complex64>::zeros((m, n));
            let mut c = 0;
            for &k in ks {
                for p in 0..snap.n_packets() {
                    for a in 0..m {
                        cols[[a, c]] = snap.data()[[a, k, p]];
                    }
                    c += 1;
                }
            }
            let eig = hermitian_eigen(&covariance_of_columns(&cols)?)?;
            let freq = snap.subcarrier_freqs_hz().and_then(|f| {
                if per_subcarrier {
                    Some(f[ks[0]])
                } else {
                    None
                }
            });
            let ideal: Vec<Complex64> = match reference {
                Reference::Common => vec![Complex64::new(1.0, 0.0); m],
                Reference::Bearing {
                    geometry,
                    azimuth,
                    elevation,
                } => match freq {
                    Some(f) => geometry.steering_at(azimuth, elevation, f),
                    None => geometry.steering(azimuth, elevation),
                },
            };
            let v = eig.vectors.column(0);
            let mut c_vec: Vec<Complex64> = (0..m).map(|a| v[a] / ideal[a]).collect();
            let r0 = c_vec[0];
            if r0.norm() < 1e-12 {
                return Err(AoaError::Numerical(
                    "reference antenna has ~zero response; cannot calibrate".into(),
                ));
            }
            let rot = r0.conj() / r0.norm();
            c_vec.iter_mut().for_each(|c| *c *= rot);
            let mean_mag = c_vec.iter().map(|c| c.norm()).sum::<f64>() / m as f64;
            if mean_mag < 1e-12 {
                return Err(AoaError::Numerical("degenerate reference capture".into()));
            }
            for a in 0..m {
                offsets[[a, g]] = c_vec[a] / mean_mag;
            }
        }
        validate_offsets(offsets.iter())?;
        Ok(Self { offsets })
    }

    /// Divide the offsets out of a capture.
    pub fn apply(&self, snap: &Snapshots) -> Res<Snapshots> {
        let (m, k, p) = snap.data().dim();
        if self.offsets.nrows() != m {
            return Err(AoaError::ShapeMismatch(format!(
                "calibration has {} antennas, capture has {m}",
                self.offsets.nrows()
            )));
        }
        let kc = self.offsets.ncols();
        if kc != 1 && kc != k {
            return Err(AoaError::ShapeMismatch(format!(
                "calibration has {kc} subcarrier columns, capture has {k}"
            )));
        }
        let mut out = Array3::<Complex64>::zeros((m, k, p));
        for a in 0..m {
            for s in 0..k {
                let c = self.offsets[[a, if kc == 1 { 0 } else { s }]];
                for pk in 0..p {
                    out[[a, s, pk]] = snap.data()[[a, s, pk]] / c;
                }
            }
        }
        Ok(Snapshots::from_parts(
            out,
            snap.subcarrier_freqs_hz().map(|f| f.to_vec()),
        ))
    }
}

fn validate_offsets<'a>(it: impl Iterator<Item = &'a Complex64>) -> Res<()> {
    for c in it {
        if !c.re.is_finite() || !c.im.is_finite() {
            return Err(AoaError::NonFinite("calibration offsets"));
        }
        if c.norm() < 1e-9 {
            return Err(AoaError::InvalidParameter("calibration offset is zero".into()));
        }
    }
    Ok(())
}

/// Remove the per-packet phase ramp across subcarriers (CFO/SFO/STO residue)
/// and the per-packet constant phase, **identically for all antennas**.
///
/// The ramp is estimated from the antenna-summed phase difference of adjacent
/// subcarriers, so it is one common slope per packet - relative phase between
/// antennas (the AoA information) is untouched. The slope also contains the
/// mean propagation delay common to all antennas; only delay *differences*
/// between antennas/paths remain afterwards. `subcarrier_index` gives the
/// (possibly non-contiguous) subcarrier numbers; `None` means `0..K`.
pub fn remove_common_phase(snap: &Snapshots, subcarrier_index: Option<&[f64]>) -> Res<Snapshots> {
    let (m, k, p) = snap.data().dim();
    let idx: Vec<f64> = match subcarrier_index {
        Some(i) if i.len() == k => i.to_vec(),
        Some(i) => {
            return Err(AoaError::ShapeMismatch(format!(
                "{} subcarrier indices for {k} subcarriers",
                i.len()
            )))
        }
        None => (0..k).map(|x| x as f64).collect(),
    };
    if idx.iter().any(|v| !v.is_finite()) {
        return Err(AoaError::NonFinite("subcarrier indices"));
    }
    let mut out = snap.data().clone();
    let min_step = idx
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|d| *d > 0.0)
        .fold(f64::INFINITY, f64::min);
    let mean_idx = idx.iter().sum::<f64>() / k as f64;
    for pk in 0..p {
        let mut slope = 0.0;
        if k >= 2 && min_step.is_finite() {
            let mut z = Complex64::new(0.0, 0.0);
            for s in 0..k - 1 {
                if ((idx[s + 1] - idx[s]) - min_step).abs() < 1e-9 {
                    for a in 0..m {
                        z += snap.data()[[a, s + 1, pk]] * snap.data()[[a, s, pk]].conj();
                    }
                }
            }
            if z.norm() > 0.0 {
                slope = z.arg() / min_step;
            }
        }
        let mut sum = Complex64::new(0.0, 0.0);
        for s in 0..k {
            let rot = Complex64::from_polar(1.0, -slope * (idx[s] - mean_idx));
            for a in 0..m {
                out[[a, s, pk]] *= rot;
                sum += out[[a, s, pk]];
            }
        }
        if sum.norm() > 0.0 {
            let rot = sum.conj() / sum.norm();
            for s in 0..k {
                for a in 0..m {
                    out[[a, s, pk]] *= rot;
                }
            }
        }
    }
    Ok(Snapshots::from_parts(
        out,
        snap.subcarrier_freqs_hz().map(|f| f.to_vec()),
    ))
}
