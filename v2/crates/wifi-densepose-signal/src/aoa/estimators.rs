//! Spatial-spectrum estimators (Bartlett, MVDR/Capon, MUSIC), source counting
//! (MDL/AIC), peak picking, and the one-call [`estimate_aoa`] pipeline.

use super::covariance::{
    forward_backward, sample_covariance, spatial_smoothing, Covariance, Smoothing,
};
use super::geometry::ArrayGeometry;
use super::{AoaError, Res, Snapshots};
use ndarray::Array2;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Azimuth/elevation scan grid. Spectrum values are indexed `[elevation, azimuth]`.
#[derive(Debug, Clone)]
pub struct ScanGrid {
    /// Azimuth samples (rad).
    pub azimuth: Vec<f64>,
    /// Elevation samples (rad); a single `0.0` for 1-D azimuth scans.
    pub elevation: Vec<f64>,
    /// Whether the azimuth axis covers the full circle (peaks wrap around).
    pub wrap_azimuth: bool,
}

impl ScanGrid {
    /// 1-D azimuth scan over `[start, stop]` (radians, inclusive) at elevation 0.
    pub fn azimuth_range(start: f64, stop: f64, n: usize) -> Res<Self> {
        if n < 3 || !(start.is_finite() && stop.is_finite() && stop > start) {
            return Err(AoaError::InvalidParameter(
                "azimuth scan needs >= 3 points and stop > start".into(),
            ));
        }
        let step = (stop - start) / (n - 1) as f64;
        Ok(Self {
            azimuth: (0..n).map(|i| start + step * i as f64).collect(),
            elevation: vec![0.0],
            wrap_azimuth: false,
        })
    }

    /// Full-circle azimuth (`[-pi, pi)`, wrapping) and an elevation range
    /// `[el_start, el_stop]` (`n_el` points; `n_el == 1` uses `el_start`).
    pub fn azimuth_elevation(
        n_az: usize,
        el_start: f64,
        el_stop: f64,
        n_el: usize,
    ) -> Res<Self> {
        if n_az < 3 || n_el == 0 || !(el_start.is_finite() && el_stop.is_finite()) {
            return Err(AoaError::InvalidParameter("invalid scan grid".into()));
        }
        if n_el > 1 && el_stop <= el_start {
            return Err(AoaError::InvalidParameter("el_stop must exceed el_start".into()));
        }
        let azimuth = (0..n_az)
            .map(|i| -PI + 2.0 * PI * i as f64 / n_az as f64)
            .collect();
        let elevation = if n_el == 1 {
            vec![el_start]
        } else {
            (0..n_el)
                .map(|i| el_start + (el_stop - el_start) * i as f64 / (n_el - 1) as f64)
                .collect()
        };
        Ok(Self {
            azimuth,
            elevation,
            wrap_azimuth: true,
        })
    }

    /// Sensible default for a geometry at `step_deg` resolution: collinear arrays
    /// along x scan azimuth `[0, pi]` (front half-plane); planar arrays scan full
    /// azimuth and elevation `[0, 90 deg]`; 3-D arrays scan elevation `[-90, 90]`.
    pub fn default_for(geom: &ArrayGeometry, step_deg: f64) -> Res<Self> {
        if !(step_deg.is_finite() && step_deg > 0.0 && step_deg <= 45.0) {
            return Err(AoaError::InvalidParameter("step must be in (0, 45] deg".into()));
        }
        let step = step_deg.to_radians();
        let p0 = geom.positions()[0];
        let along_x = geom
            .positions()
            .iter()
            .all(|p| (p[1] - p0[1]).abs() < 1e-12 && (p[2] - p0[2]).abs() < 1e-12);
        if geom.is_linear() && along_x {
            return Self::azimuth_range(0.0, PI, (PI / step).round() as usize + 1);
        }
        let n_az = (2.0 * PI / step).round() as usize;
        if geom.is_linear() {
            return Self::azimuth_elevation(n_az, 0.0, 0.0, 1);
        }
        if geom.is_planar_xy() {
            let n_el = (0.5 * PI / step).round() as usize + 1;
            return Self::azimuth_elevation(n_az, 0.0, 0.5 * PI, n_el);
        }
        let n_el = (PI / step).round() as usize + 1;
        Self::azimuth_elevation(n_az, -0.5 * PI, 0.5 * PI, n_el)
    }
}

/// A computed spatial spectrum on a [`ScanGrid`]; `values[[el, az]]`.
#[derive(Debug, Clone)]
pub struct Spectrum {
    /// The scan grid.
    pub grid: ScanGrid,
    /// Pseudo-power values (>= 0).
    pub values: Array2<f64>,
}

/// A spectral peak with sub-grid refinement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Peak {
    /// Azimuth (rad), refined between grid points.
    pub azimuth: f64,
    /// Elevation (rad), refined when the grid has >= 3 elevation points.
    pub elevation: f64,
    /// Spectrum value at the nearest grid point.
    pub value: f64,
}

fn check(cov: &Covariance, geom: &ArrayGeometry) -> Res<()> {
    if cov.matrix.nrows() != cov.matrix.ncols() || cov.dim() != geom.n_antennas() {
        return Err(AoaError::ShapeMismatch(format!(
            "covariance is {}x{} but geometry has {} antennas",
            cov.matrix.nrows(),
            cov.matrix.ncols(),
            geom.n_antennas()
        )));
    }
    if cov.matrix.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
        return Err(AoaError::NonFinite("covariance"));
    }
    Ok(())
}

fn scan(
    geom: &ArrayGeometry,
    grid: &ScanGrid,
    f: impl Fn(&[Complex64]) -> f64,
) -> Spectrum {
    let mut values = Array2::<f64>::zeros((grid.elevation.len(), grid.azimuth.len()));
    for (ie, &el) in grid.elevation.iter().enumerate() {
        for (ia, &az) in grid.azimuth.iter().enumerate() {
            values[[ie, ia]] = f(&geom.steering(az, el));
        }
    }
    Spectrum {
        grid: grid.clone(),
        values,
    }
}

fn quad_form(r: &Array2<Complex64>, a: &[Complex64]) -> f64 {
    let m = a.len();
    let mut s = Complex64::new(0.0, 0.0);
    for i in 0..m {
        let mut row = Complex64::new(0.0, 0.0);
        for j in 0..m {
            row += r[[i, j]] * a[j];
        }
        s += a[i].conj() * row;
    }
    s.re
}

/// Bartlett (conventional beamformer) spectrum `a^H R a / (a^H a)`.
pub fn bartlett(cov: &Covariance, geom: &ArrayGeometry, grid: &ScanGrid) -> Res<Spectrum> {
    check(cov, geom)?;
    let m = geom.n_antennas() as f64;
    Ok(scan(geom, grid, |a| (quad_form(&cov.matrix, a) / m).max(0.0)))
}

/// MVDR/Capon spectrum `1 / (a^H (R + d I)^-1 a)` with diagonal loading
/// `d = loading * trace(R) / M` (relative; 1e-3 is a good default).
pub fn mvdr(
    cov: &Covariance,
    geom: &ArrayGeometry,
    grid: &ScanGrid,
    loading: f64,
) -> Res<Spectrum> {
    check(cov, geom)?;
    if !(loading.is_finite() && loading >= 0.0) {
        return Err(AoaError::InvalidParameter("loading must be >= 0".into()));
    }
    let eig = cov.eigen()?;
    let m = geom.n_antennas();
    let trace: f64 = eig.values.iter().sum();
    let delta = (loading * trace / m as f64).max(1e-12 * trace.abs()).max(f64::MIN_POSITIVE);
    let inv: Vec<f64> = eig.values.iter().map(|l| 1.0 / (l.max(0.0) + delta)).collect();
    Ok(scan(geom, grid, |a| {
        let mut s = 0.0;
        for (c, w) in inv.iter().enumerate() {
            let mut p = Complex64::new(0.0, 0.0);
            for (i, ai) in a.iter().enumerate() {
                p += eig.vectors[[i, c]].conj() * ai;
            }
            s += w * p.norm_sqr();
        }
        m as f64 / s
    }))
}

/// MUSIC pseudo-spectrum `1 / (a^H E_n E_n^H a)` for `n_sources` signals
/// (`1 <= n_sources < M`). 2-D when the grid has several elevations.
pub fn music(
    cov: &Covariance,
    geom: &ArrayGeometry,
    grid: &ScanGrid,
    n_sources: usize,
) -> Res<Spectrum> {
    check(cov, geom)?;
    let m = geom.n_antennas();
    if n_sources == 0 || n_sources >= m {
        return Err(AoaError::InvalidParameter(format!(
            "n_sources must be in 1..{m}, got {n_sources}"
        )));
    }
    let eig = cov.eigen()?;
    Ok(scan(geom, grid, |a| {
        let mut den = 0.0;
        for c in n_sources..m {
            let mut p = Complex64::new(0.0, 0.0);
            for (i, ai) in a.iter().enumerate() {
                p += eig.vectors[[i, c]].conj() * ai;
            }
            den += p.norm_sqr();
        }
        let norm = a.len() as f64;
        norm / den.max(1e-14 * norm)
    }))
}

/// Source-count information criterion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceCountCriterion {
    /// Minimum description length (consistent; the safer default).
    Mdl,
    /// Akaike information criterion (tends to over-estimate).
    Aic,
}

/// Wax-Kailath source-count estimate from covariance eigenvalues (any order).
///
/// `n_snapshots` should be the number of *independent* snapshots; adjacent
/// subcarriers are correlated, so pass a conservative effective count if known.
/// Returns a value in `0..M`.
pub fn estimate_num_sources(
    eigenvalues: &[f64],
    n_snapshots: usize,
    criterion: SourceCountCriterion,
) -> Res<usize> {
    let m = eigenvalues.len();
    if m < 2 {
        return Err(AoaError::InsufficientData("need >= 2 eigenvalues".into()));
    }
    if n_snapshots < 2 {
        return Err(AoaError::InsufficientData("need >= 2 snapshots".into()));
    }
    if eigenvalues.iter().any(|v| !v.is_finite()) {
        return Err(AoaError::NonFinite("eigenvalues"));
    }
    let mut l: Vec<f64> = eigenvalues.to_vec();
    l.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let floor = l[0].abs().max(f64::MIN_POSITIVE) * 1e-12;
    l.iter_mut().for_each(|v| *v = v.max(floor));
    let n = n_snapshots as f64;
    let mut best = (f64::INFINITY, 0usize);
    for k in 0..m {
        let tail = &l[k..];
        let cnt = (m - k) as f64;
        let log_g = tail.iter().map(|v| v.ln()).sum::<f64>() / cnt;
        let log_a = (tail.iter().sum::<f64>() / cnt).ln();
        let fit = -n * cnt * (log_g - log_a);
        let free = k as f64 * (2.0 * m as f64 - k as f64);
        let crit = match criterion {
            SourceCountCriterion::Mdl => fit + 0.5 * free * n.ln(),
            SourceCountCriterion::Aic => 2.0 * fit + 2.0 * free,
        };
        if crit < best.0 {
            best = (crit, k);
        }
    }
    Ok(best.1)
}

/// A raw peak in index space (fractional after refinement).
#[derive(Debug, Clone, Copy)]
pub(crate) struct RawPeak {
    pub row: f64,
    pub col: f64,
    pub value: f64,
}

/// Local maxima of `values` with parabolic refinement on `ln(values)`.
pub(crate) fn pick_peaks(
    values: &Array2<f64>,
    wrap_cols: bool,
    max_peaks: usize,
    min_rel: f64,
) -> Vec<RawPeak> {
    let (nr, nc) = values.dim();
    let global = values.iter().cloned().fold(0.0_f64, f64::max);
    let mut out = Vec::new();
    if global <= 0.0 {
        return out;
    }
    let at = |r: isize, c: isize| -> Option<f64> {
        if r < 0 || r >= nr as isize {
            return None;
        }
        let c = if wrap_cols {
            c.rem_euclid(nc as isize)
        } else if c < 0 || c >= nc as isize {
            return None;
        } else {
            c
        };
        Some(values[[r as usize, c as usize]])
    };
    let ln = |v: f64| v.max(1e-300).ln();
    for r in 0..nr as isize {
        for c in 0..nc as isize {
            let v = values[[r as usize, c as usize]];
            if v < min_rel * global {
                continue;
            }
            let mut is_peak = true;
            let mut strict = false;
            for dr in -1..=1 {
                for dc in -1..=1 {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    if let Some(n) = at(r + dr, c + dc) {
                        if n > v {
                            is_peak = false;
                        }
                        if n < v {
                            strict = true;
                        }
                    }
                }
            }
            if !is_peak || !strict {
                continue;
            }
            let refine = |m1: Option<f64>, p1: Option<f64>| -> f64 {
                match (m1, p1) {
                    (Some(a), Some(b)) => {
                        let (a, y, b) = (ln(a), ln(v), ln(b));
                        let den = a - 2.0 * y + b;
                        if den.abs() < 1e-12 {
                            0.0
                        } else {
                            (0.5 * (a - b) / den).clamp(-0.5, 0.5)
                        }
                    }
                    _ => 0.0,
                }
            };
            let dcol = refine(at(r, c - 1), at(r, c + 1));
            let drow = refine(at(r - 1, c), at(r + 1, c));
            out.push(RawPeak {
                row: r as f64 + drow,
                col: c as f64 + dcol,
                value: v,
            });
        }
    }
    out.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap_or(std::cmp::Ordering::Equal));
    out.truncate(max_peaks);
    out
}

/// Interpolate an axis at a fractional index (wrapping adds the period).
pub(crate) fn axis_at(axis: &[f64], idx: f64, wrap: bool) -> f64 {
    let n = axis.len();
    let lo = idx.floor();
    let frac = idx - lo;
    let get = |i: isize| -> f64 {
        if wrap {
            let w = i.rem_euclid(n as isize) as usize;
            // Unwrap across the seam so interpolation is continuous.
            let k = i.div_euclid(n as isize) as f64;
            axis[w] + k * 2.0 * PI
        } else {
            axis[i.clamp(0, n as isize - 1) as usize]
        }
    };
    let a = get(lo as isize);
    let b = get(lo as isize + 1);
    let v = a + (b - a) * frac;
    if wrap && v >= PI {
        v - 2.0 * PI
    } else if wrap && v < -PI {
        v + 2.0 * PI
    } else {
        v
    }
}

/// Up to `max_peaks` strongest local maxima of a spectrum, refined to sub-grid
/// accuracy by parabolic interpolation of the log spectrum. Peaks below
/// `min_rel` times the global maximum are dropped.
pub fn find_peaks(spec: &Spectrum, max_peaks: usize, min_rel: f64) -> Vec<Peak> {
    pick_peaks(&spec.values, spec.grid.wrap_azimuth, max_peaks, min_rel)
        .into_iter()
        .map(|p| Peak {
            azimuth: axis_at(&spec.grid.azimuth, p.col, spec.grid.wrap_azimuth),
            elevation: axis_at(&spec.grid.elevation, p.row, false),
            value: p.value,
        })
        .collect()
}

/// Spectrum method for [`estimate_aoa`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Method {
    /// Conventional beamformer.
    Bartlett,
    /// Capon with relative diagonal loading.
    Mvdr {
        /// Relative diagonal loading (e.g. `1e-3`).
        loading: f64,
    },
    /// MUSIC.
    Music,
}

/// Configuration for [`estimate_aoa`].
#[derive(Debug, Clone)]
pub struct AoaConfig {
    /// Spectrum estimator.
    pub method: Method,
    /// Apply forward-backward averaging (needs a centro-symmetric array).
    pub forward_backward: bool,
    /// Spatial smoothing sub-array (ULA/URA only); apply for coherent multipath.
    pub smoothing: Option<Smoothing>,
    /// Fixed number of sources, or `None` to estimate it.
    pub n_sources: Option<usize>,
    /// Criterion used when `n_sources` is `None`.
    pub criterion: SourceCountCriterion,
    /// Upper bound on reported peaks when counting automatically.
    pub max_sources: usize,
    /// Scan resolution in degrees (ignored if `grid` is set).
    pub grid_step_deg: f64,
    /// Explicit scan grid.
    pub grid: Option<ScanGrid>,
}

impl Default for AoaConfig {
    fn default() -> Self {
        Self {
            method: Method::Music,
            forward_backward: false,
            smoothing: None,
            n_sources: None,
            criterion: SourceCountCriterion::Mdl,
            max_sources: 4,
            grid_step_deg: 1.0,
            grid: None,
        }
    }
}

/// Output of [`estimate_aoa`].
#[derive(Debug, Clone)]
pub struct AoaResult {
    /// Sources assumed/estimated.
    pub n_sources: usize,
    /// Covariance eigenvalues (descending) after smoothing/FB.
    pub eigenvalues: Vec<f64>,
    /// Strongest `n_sources` peaks (strongest first). Empty if no source found.
    pub peaks: Vec<Peak>,
    /// The spectrum.
    pub spectrum: Spectrum,
}

/// Full pipeline: covariance -> (smoothing) -> (forward-backward) -> source
/// count -> spectrum -> refined peaks.
pub fn estimate_aoa(snap: &Snapshots, geom: &ArrayGeometry, cfg: &AoaConfig) -> Res<AoaResult> {
    if snap.n_antennas() != geom.n_antennas() {
        return Err(AoaError::ShapeMismatch(format!(
            "snapshots have {} antennas, geometry has {}",
            snap.n_antennas(),
            geom.n_antennas()
        )));
    }
    let mut cov = sample_covariance(snap)?;
    let mut g = geom.clone();
    if let Some(s) = cfg.smoothing {
        let (c, sg) = spatial_smoothing(&cov, &g, s)?;
        cov = c;
        g = sg;
    }
    if cfg.forward_backward {
        cov = forward_backward(&cov, &g)?;
    }
    let eig = cov.eigen()?;
    let m = g.n_antennas();
    let d = match cfg.n_sources {
        Some(d) => {
            if d == 0 || d >= m {
                return Err(AoaError::InvalidParameter(format!(
                    "n_sources must be in 1..{m}, got {d}"
                )));
            }
            d
        }
        None => estimate_num_sources(&eig.values, cov.n_snapshots, cfg.criterion)?
            .min(cfg.max_sources),
    };
    let grid = match &cfg.grid {
        Some(g) => g.clone(),
        None => ScanGrid::default_for(&g, cfg.grid_step_deg)?,
    };
    let spectrum = match cfg.method {
        Method::Bartlett => bartlett(&cov, &g, &grid)?,
        Method::Mvdr { loading } => mvdr(&cov, &g, &grid, loading)?,
        Method::Music => music(&cov, &g, &grid, d.max(1))?,
    };
    let peaks = if d == 0 {
        Vec::new()
    } else {
        find_peaks(&spectrum, d, 0.0)
    };
    Ok(AoaResult {
        n_sources: d,
        eigenvalues: eig.values,
        peaks,
        spectrum,
    })
}
