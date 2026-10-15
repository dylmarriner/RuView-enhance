//! Spatial covariance, forward-backward averaging and spatial smoothing.

use super::geometry::ArrayGeometry;
use super::{AoaError, Res, Snapshots};
use nalgebra::{DMatrix, SymmetricEigen};
use ndarray::{Array2, Axis};
use num_complex::Complex64;

/// Hermitian spatial covariance matrix plus the number of snapshots behind it.
#[derive(Debug, Clone)]
pub struct Covariance {
    /// `M x M` Hermitian matrix.
    pub matrix: Array2<Complex64>,
    /// Independent-snapshot count used for the estimate (for MDL/AIC).
    pub n_snapshots: usize,
}

/// Spatial-smoothing sub-array size (columns x rows of the grid).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Smoothing {
    /// Sub-array columns (x).
    pub sub_x: usize,
    /// Sub-array rows (y).
    pub sub_y: usize,
}

/// Eigendecomposition of a Hermitian matrix, eigenvalues descending.
#[derive(Debug, Clone)]
pub(crate) struct Eigen {
    pub values: Vec<f64>,
    /// Columns are the eigenvectors matching `values`.
    pub vectors: Array2<Complex64>,
}

impl Covariance {
    /// Matrix dimension (number of antennas).
    pub fn dim(&self) -> usize {
        self.matrix.nrows()
    }
    pub(crate) fn eigen(&self) -> Res<Eigen> {
        hermitian_eigen(&self.matrix)
    }
}

/// Sample covariance `R = (1/N) sum x x^H` over all `(subcarrier, packet)` snapshots.
///
/// Requires at least as many snapshots as antennas so the estimate is not
/// trivially rank-deficient.
pub fn sample_covariance(snap: &Snapshots) -> Res<Covariance> {
    let m = snap.n_antennas();
    let n = snap.n_snapshots();
    if n < m {
        return Err(AoaError::InsufficientData(format!(
            "{n} snapshots for {m} antennas; need at least {m}"
        )));
    }
    let mut cols = Array2::<Complex64>::zeros((m, n));
    let mut c = 0;
    for p in 0..snap.n_packets() {
        for k in 0..snap.n_subcarriers() {
            for a in 0..m {
                cols[[a, c]] = snap.data()[[a, k, p]];
            }
            c += 1;
        }
    }
    Ok(Covariance {
        matrix: covariance_of_columns(&cols)?,
        n_snapshots: n,
    })
}

/// `(1/N) X X^H` for an `M x N` matrix of snapshot columns.
pub(crate) fn covariance_of_columns(x: &Array2<Complex64>) -> Res<Array2<Complex64>> {
    let (m, n) = x.dim();
    if n == 0 {
        return Err(AoaError::InsufficientData("no snapshots".into()));
    }
    let mut r = Array2::<Complex64>::zeros((m, m));
    for col in x.axis_iter(Axis(1)) {
        for i in 0..m {
            for j in i..m {
                r[[i, j]] += col[i] * col[j].conj();
            }
        }
    }
    let inv = 1.0 / n as f64;
    for i in 0..m {
        for j in i..m {
            let v = r[[i, j]] * inv;
            r[[i, j]] = if i == j { Complex64::new(v.re, 0.0) } else { v };
            r[[j, i]] = r[[i, j]].conj();
        }
    }
    Ok(r)
}

/// Forward-backward averaging `(R + J R* J) / 2`; decorrelates up to two coherent
/// sources on centro-symmetric arrays and doubles the effective snapshots.
pub fn forward_backward(cov: &Covariance, geom: &ArrayGeometry) -> Res<Covariance> {
    check_dim(cov, geom)?;
    let perm = geom.reflection_permutation()?;
    let m = cov.dim();
    let mut out = Array2::<Complex64>::zeros((m, m));
    for i in 0..m {
        for j in 0..m {
            out[[i, j]] = 0.5 * (cov.matrix[[i, j]] + cov.matrix[[perm[i], perm[j]]].conj());
        }
    }
    Ok(Covariance {
        matrix: out,
        n_snapshots: cov.n_snapshots,
    })
}

/// Spatial smoothing over shifted `sx x sy` sub-arrays of a ULA/URA grid.
///
/// Returns the smoothed covariance (dimension `sx*sy`) together with the matching
/// sub-array geometry to hand to the estimators. Restores rank for coherent
/// sources at the price of aperture.
pub fn spatial_smoothing(
    cov: &Covariance,
    geom: &ArrayGeometry,
    smoothing: Smoothing,
) -> Res<(Covariance, ArrayGeometry)> {
    check_dim(cov, geom)?;
    let (nx, ny) = geom.grid().ok_or_else(|| {
        AoaError::UnsupportedGeometry("spatial smoothing needs a ULA/URA grid geometry".into())
    })?;
    let sub = geom.subarray(smoothing.sub_x, smoothing.sub_y)?;
    let (sx, sy) = (smoothing.sub_x, smoothing.sub_y);
    let shifts_x = nx - sx + 1;
    let shifts_y = ny - sy + 1;
    let idx = |ox: usize, oy: usize| -> Vec<usize> {
        let mut v = Vec::with_capacity(sx * sy);
        for iy in 0..sy {
            for ix in 0..sx {
                v.push((oy + iy) * nx + ox + ix);
            }
        }
        v
    };
    let ms = sx * sy;
    let mut out = Array2::<Complex64>::zeros((ms, ms));
    for oy in 0..shifts_y {
        for ox in 0..shifts_x {
            let id = idx(ox, oy);
            for a in 0..ms {
                for b in 0..ms {
                    out[[a, b]] += cov.matrix[[id[a], id[b]]];
                }
            }
        }
    }
    out.mapv_inplace(|v| v / (shifts_x * shifts_y) as f64);
    Ok((
        Covariance {
            matrix: out,
            n_snapshots: cov.n_snapshots,
        },
        sub,
    ))
}

fn check_dim(cov: &Covariance, geom: &ArrayGeometry) -> Res<()> {
    if cov.matrix.nrows() != cov.matrix.ncols() || cov.dim() != geom.n_antennas() {
        return Err(AoaError::ShapeMismatch(format!(
            "covariance is {}x{} but geometry has {} antennas",
            cov.matrix.nrows(),
            cov.matrix.ncols(),
            geom.n_antennas()
        )));
    }
    Ok(())
}

/// Hermitian eigendecomposition via nalgebra (`SymmetricEigen` on complex input).
pub(crate) fn hermitian_eigen(r: &Array2<Complex64>) -> Res<Eigen> {
    let n = r.nrows();
    if n == 0 || r.ncols() != n {
        return Err(AoaError::ShapeMismatch("matrix must be square and non-empty".into()));
    }
    if r.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
        return Err(AoaError::NonFinite("covariance"));
    }
    // Symmetrise to remove asymmetry from round-off.
    let m = DMatrix::from_fn(n, n, |i, j| 0.5 * (r[[i, j]] + r[[j, i]].conj()));
    let eig = SymmetricEigen::try_new(m, f64::EPSILON * 16.0, 1000)
        .ok_or_else(|| AoaError::Numerical("Hermitian eigendecomposition did not converge".into()))?;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        eig.eigenvalues[b]
            .partial_cmp(&eig.eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let values = order.iter().map(|&i| eig.eigenvalues[i]).collect();
    let mut vectors = Array2::<Complex64>::zeros((n, n));
    for (c, &i) in order.iter().enumerate() {
        for row in 0..n {
            vectors[[row, c]] = eig.eigenvectors[(row, i)];
        }
    }
    Ok(Eigen { values, vectors })
}
