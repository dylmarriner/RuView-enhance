//! Antenna array geometry and steering vectors.

use super::{AoaError, Res, SPEED_OF_LIGHT};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Receive-antenna array geometry.
///
/// Positions are in metres in the array frame (x right, y forward, z up).
/// Arrays built by [`ArrayGeometry::ula`] / [`ArrayGeometry::ura`] remember
/// their grid shape, which enables spatial smoothing.
#[derive(Debug, Clone, PartialEq)]
pub struct ArrayGeometry {
    positions: Vec<[f64; 3]>,
    /// `(nx, ny)`: antenna index = `iy * nx + ix`.
    grid: Option<(usize, usize)>,
    center_freq_hz: f64,
}

impl ArrayGeometry {
    /// Arbitrary geometry. Needs >= 2 finite positions and a positive centre frequency.
    pub fn new(positions: Vec<[f64; 3]>, center_freq_hz: f64) -> Res<Self> {
        if positions.len() < 2 {
            return Err(AoaError::InsufficientData(
                "an array needs at least 2 antennas".into(),
            ));
        }
        if positions.iter().flatten().any(|v| !v.is_finite()) {
            return Err(AoaError::NonFinite("antenna positions"));
        }
        if !(center_freq_hz.is_finite() && center_freq_hz > 0.0) {
            return Err(AoaError::InvalidParameter(format!(
                "centre frequency must be > 0 Hz, got {center_freq_hz}"
            )));
        }
        Ok(Self {
            positions,
            grid: None,
            center_freq_hz,
        })
    }

    /// Uniform linear array along +x with `n` elements and `spacing_m` spacing.
    pub fn ula(n: usize, spacing_m: f64, center_freq_hz: f64) -> Res<Self> {
        Self::ura(n, 1, spacing_m, spacing_m, center_freq_hz)
    }

    /// ULA with lambda/2 spacing at `center_freq_hz`.
    pub fn ula_half_wave(n: usize, center_freq_hz: f64) -> Res<Self> {
        Self::ula(n, 0.5 * wavelength(center_freq_hz), center_freq_hz)
    }

    /// Uniform rectangular array in the xy plane: `nx` columns (spacing `dx`)
    /// by `ny` rows (spacing `dy`). Antenna index = `iy * nx + ix`.
    pub fn ura(nx: usize, ny: usize, dx_m: f64, dy_m: f64, center_freq_hz: f64) -> Res<Self> {
        if nx == 0 || ny == 0 {
            return Err(AoaError::InvalidParameter("array dimensions must be >= 1".into()));
        }
        if !(dx_m.is_finite() && dx_m > 0.0 && dy_m.is_finite() && dy_m > 0.0) {
            return Err(AoaError::InvalidParameter("spacings must be > 0".into()));
        }
        let mut pos = Vec::with_capacity(nx * ny);
        for iy in 0..ny {
            for ix in 0..nx {
                pos.push([ix as f64 * dx_m, iy as f64 * dy_m, 0.0]);
            }
        }
        let mut g = Self::new(pos, center_freq_hz)?;
        g.grid = Some((nx, ny));
        Ok(g)
    }

    /// URA with lambda/2 spacing (e.g. ESPARGOS-like 2x4 is `ura_half_wave(4, 2, f)`).
    pub fn ura_half_wave(nx: usize, ny: usize, center_freq_hz: f64) -> Res<Self> {
        let d = 0.5 * wavelength(center_freq_hz);
        Self::ura(nx, ny, d, d, center_freq_hz)
    }

    /// Number of antennas.
    pub fn n_antennas(&self) -> usize {
        self.positions.len()
    }
    /// Antenna positions (m).
    pub fn positions(&self) -> &[[f64; 3]] {
        &self.positions
    }
    /// Grid shape `(nx, ny)` for ULA/URA arrays.
    pub fn grid(&self) -> Option<(usize, usize)> {
        self.grid
    }
    /// Centre frequency (Hz).
    pub fn center_freq_hz(&self) -> f64 {
        self.center_freq_hz
    }
    /// Wavelength at the centre frequency (m).
    pub fn wavelength(&self) -> f64 {
        wavelength(self.center_freq_hz)
    }
    /// Wavelength at an arbitrary (e.g. per-subcarrier) frequency (m).
    pub fn wavelength_at(&self, freq_hz: f64) -> f64 {
        wavelength(freq_hz)
    }

    /// Unit vector pointing *toward* a source at azimuth/elevation (rad).
    pub fn direction(azimuth: f64, elevation: f64) -> [f64; 3] {
        let (sa, ca) = azimuth.sin_cos();
        let (se, ce) = elevation.sin_cos();
        [ce * ca, ce * sa, se]
    }

    /// Steering vector at the centre frequency.
    pub fn steering(&self, azimuth: f64, elevation: f64) -> Vec<Complex64> {
        self.steering_at(azimuth, elevation, self.center_freq_hz)
    }

    /// Steering vector at `freq_hz` (per-subcarrier wavelength).
    pub fn steering_at(&self, azimuth: f64, elevation: f64, freq_hz: f64) -> Vec<Complex64> {
        let u = Self::direction(azimuth, elevation);
        let k = 2.0 * PI / wavelength(freq_hz);
        self.positions
            .iter()
            .map(|p| Complex64::from_polar(1.0, k * (p[0] * u[0] + p[1] * u[1] + p[2] * u[2])))
            .collect()
    }

    /// True when all antennas are collinear (bearings ambiguous about the axis).
    pub fn is_linear(&self) -> bool {
        let p0 = self.positions[0];
        let d0 = self
            .positions
            .iter()
            .map(|p| sub(*p, p0))
            .find(|d| norm(*d) > 1e-12);
        let Some(d0) = d0 else { return true };
        self.positions.iter().all(|p| {
            let c = cross(sub(*p, p0), d0);
            norm(c) <= 1e-9 * norm(d0).max(1.0)
        })
    }

    /// True when all antennas lie in the z = const plane.
    pub fn is_planar_xy(&self) -> bool {
        let z0 = self.positions[0][2];
        self.positions.iter().all(|p| (p[2] - z0).abs() < 1e-12)
    }

    /// Sub-array made of the first `sx` columns and `sy` rows of the grid.
    pub fn subarray(&self, sx: usize, sy: usize) -> Res<Self> {
        let (nx, ny) = self.grid.ok_or_else(|| {
            AoaError::UnsupportedGeometry("sub-arrays need a ULA/URA grid geometry".into())
        })?;
        if sx == 0 || sy == 0 || sx > nx || sy > ny {
            return Err(AoaError::InvalidParameter(format!(
                "sub-array {sx}x{sy} does not fit a {nx}x{ny} grid"
            )));
        }
        let mut pos = Vec::with_capacity(sx * sy);
        for iy in 0..sy {
            for ix in 0..sx {
                pos.push(self.positions[iy * nx + ix]);
            }
        }
        if pos.len() < 2 {
            return Err(AoaError::InvalidParameter("sub-array needs >= 2 antennas".into()));
        }
        Ok(Self {
            positions: pos,
            grid: Some((sx, sy)),
            center_freq_hz: self.center_freq_hz,
        })
    }

    /// Permutation `j` with `p[j[i]] = 2c - p[i]` (point reflection about the
    /// centroid `c`), needed for forward-backward averaging. Errors if the
    /// array is not centro-symmetric.
    pub fn reflection_permutation(&self) -> Res<Vec<usize>> {
        let n = self.n_antennas();
        let mut c = [0.0; 3];
        for p in &self.positions {
            for a in 0..3 {
                c[a] += p[a] / n as f64;
            }
        }
        let scale = self
            .positions
            .iter()
            .map(|p| norm(sub(*p, c)))
            .fold(0.0, f64::max)
            .max(1e-12);
        let mut perm = Vec::with_capacity(n);
        for p in &self.positions {
            let target = [2.0 * c[0] - p[0], 2.0 * c[1] - p[1], 2.0 * c[2] - p[2]];
            let j = self
                .positions
                .iter()
                .position(|q| norm(sub(*q, target)) < 1e-9 * scale)
                .ok_or_else(|| {
                    AoaError::UnsupportedGeometry(
                        "array is not centro-symmetric; forward-backward averaging unavailable"
                            .into(),
                    )
                })?;
            perm.push(j);
        }
        Ok(perm)
    }
}

/// Wavelength (m) for a frequency (Hz).
pub fn wavelength(freq_hz: f64) -> f64 {
    SPEED_OF_LIGHT / freq_hz
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}
