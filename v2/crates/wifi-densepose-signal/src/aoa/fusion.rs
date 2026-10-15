//! Combine bearings from several arrays at known poses into a 2-D position.
//!
//! Each bearing defines a ray `a + r (cos t, sin t)`. The position minimising
//! the weighted sum of squared perpendicular distances to all rays is found in
//! closed form (2x2 normal equations). Because a bearing error `sigma_t` maps
//! to a cross-range error `r * sigma_t`, weights are re-evaluated with the
//! current range estimate (iteratively re-weighted least squares).

use super::{AoaError, Res};

/// Pose of an array in the world frame (2-D).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArrayPose {
    /// World x (m).
    pub x: f64,
    /// World y (m).
    pub y: f64,
    /// Rotation (rad, CCW) of the array +x axis relative to the world +x axis.
    pub yaw: f64,
}

/// A bearing observation in the world frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bearing {
    /// Array position `(x, y)` in metres.
    pub origin: [f64; 2],
    /// World-frame azimuth of the source as seen from the array (rad).
    pub azimuth: f64,
    /// 1-sigma bearing uncertainty (rad), `> 0`.
    pub sigma_rad: f64,
}

impl Bearing {
    /// Build from an array-frame azimuth (as returned by the estimators) and the array pose.
    pub fn from_local(pose: ArrayPose, local_azimuth: f64, sigma_rad: f64) -> Self {
        Self {
            origin: [pose.x, pose.y],
            azimuth: pose.yaw + local_azimuth,
            sigma_rad,
        }
    }
}

/// Result of [`locate_2d`].
#[derive(Debug, Clone, PartialEq)]
pub struct PositionFix {
    /// Estimated `(x, y)` in metres.
    pub position: [f64; 2],
    /// Position covariance (m^2) implied by the bearing sigmas.
    pub covariance: [[f64; 2]; 2],
    /// RMS perpendicular distance from the fix to the rays (m); large values
    /// mean inconsistent bearings (multipath, bad calibration).
    pub residual_rms_m: f64,
    /// Signed range along each ray to the fix (negative: the fix is behind the array).
    pub ranges_m: Vec<f64>,
    /// Number of rays whose range is negative.
    pub n_behind: usize,
}

impl PositionFix {
    /// 1-sigma radial uncertainty `sqrt(trace(cov))` (m).
    pub fn uncertainty_m(&self) -> f64 {
        (self.covariance[0][0] + self.covariance[1][1]).max(0.0).sqrt()
    }
}

/// Least-squares ray intersection. Needs >= 2 non-parallel bearings.
pub fn locate_2d(bearings: &[Bearing]) -> Res<PositionFix> {
    if bearings.len() < 2 {
        return Err(AoaError::InsufficientData(
            "need at least 2 bearings to intersect".into(),
        ));
    }
    for b in bearings {
        if !(b.origin[0].is_finite() && b.origin[1].is_finite() && b.azimuth.is_finite()) {
            return Err(AoaError::NonFinite("bearing"));
        }
        if !(b.sigma_rad.is_finite() && b.sigma_rad > 0.0) {
            return Err(AoaError::InvalidParameter("bearing sigma must be > 0".into()));
        }
    }
    let mut ranges: Vec<f64> = vec![1.0; bearings.len()];
    let mut sol = ([0.0, 0.0], [[0.0; 2]; 2]);
    for _ in 0..4 {
        let (mut a11, mut a12, mut a22, mut b1, mut b2) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for (b, r) in bearings.iter().zip(&ranges) {
            let (s, c) = b.azimuth.sin_cos();
            let (nx, ny) = (-s, c);
            let sd = b.sigma_rad * r.abs().max(0.05);
            let w = 1.0 / (sd * sd);
            let k = nx * b.origin[0] + ny * b.origin[1];
            a11 += w * nx * nx;
            a12 += w * nx * ny;
            a22 += w * ny * ny;
            b1 += w * nx * k;
            b2 += w * ny * k;
        }
        let det = a11 * a22 - a12 * a12;
        let tr = a11 + a22;
        if !(det.is_finite() && det > 1e-9 * tr * tr) {
            return Err(AoaError::Numerical(
                "bearings are (nearly) parallel; position is unobservable".into(),
            ));
        }
        let inv = [[a22 / det, -a12 / det], [-a12 / det, a11 / det]];
        let p = [
            inv[0][0] * b1 + inv[0][1] * b2,
            inv[1][0] * b1 + inv[1][1] * b2,
        ];
        sol = (p, inv);
        for (b, r) in bearings.iter().zip(ranges.iter_mut()) {
            let (s, c) = b.azimuth.sin_cos();
            *r = (p[0] - b.origin[0]) * c + (p[1] - b.origin[1]) * s;
        }
    }
    let (p, cov) = sol;
    let mut ss = 0.0;
    for b in bearings {
        let (s, c) = b.azimuth.sin_cos();
        let perp = -(p[0] - b.origin[0]) * s + (p[1] - b.origin[1]) * c;
        ss += perp * perp;
    }
    Ok(PositionFix {
        position: p,
        covariance: cov,
        residual_rms_m: (ss / bearings.len() as f64).sqrt(),
        n_behind: ranges.iter().filter(|r| **r < 0.0).count(),
        ranges_m: ranges,
    })
}
