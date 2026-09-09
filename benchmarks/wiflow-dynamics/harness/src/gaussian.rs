//! Joint-Gaussian action-conditioned dynamics model.
//!
//! Models `p(z_t, z_{t+1}, y_{t+1})` as one multivariate Gaussian, where
//! `z` is the 540-dim band representation of a CSI window and `y` is the 30-dim pose.
//!
//! ## What the "action" is (and is not)
//!
//! An action is a **probe set** `S`: the subset of bands whose values at `t+1` are
//! revealed. Everything is closed-form, so `p(y_{t+1} | z_t, z_{t+1}[S])` -- the
//! charter's `p(observation | intervention)` -- is exact rather than approximated.
//!
//! This is **active sensing on offline data**: the action selects which
//! already-recorded coordinates are revealed. It is NOT a causal intervention on the
//! world, and no result from it licenses a causal claim.
//!
//! ## The homoscedasticity caveat (pre-registered, load-bearing)
//!
//! For a Gaussian, the conditional covariance does not depend on the observed
//! *values*, so `EIG(S)` is a function of `Sigma` alone. The EIG-optimal probe set is
//! therefore **the same at every timestep** -- this is information-theoretic set
//! selection, NOT adaptive sensing. Adaptive probing needs an input-dependent
//! `Sigma(z_t)`, which this rung deliberately does not have.

use nalgebra::{DMatrix, DVector};

use crate::bands::{group_dims, BAND_DIM, NUM_PROBE_GROUPS};

pub const POSE_DIM: usize = 30;
/// `[z_{t+1} (540) | y_{t+1} (30)]` -- the block left after conditioning on `z_t`.
pub const REST_DIM: usize = BAND_DIM + POSE_DIM;
pub const JOINT_DIM: usize = BAND_DIM + REST_DIM;

/// Flat indices in the *rest* block covered by probe group `g`.
pub fn rest_group_dims(g: usize) -> std::ops::Range<usize> {
    group_dims(g)
}

/// Streaming mean/covariance accumulator over the 1110-dim joint vector.
pub struct JointStats {
    n: usize,
    sum: DVector<f64>,
    outer: DMatrix<f64>,
}

impl Default for JointStats {
    fn default() -> Self {
        Self::new()
    }
}

impl JointStats {
    pub fn new() -> Self {
        Self { n: 0, sum: DVector::zeros(JOINT_DIM), outer: DMatrix::zeros(JOINT_DIM, JOINT_DIM) }
    }

    pub fn push(&mut self, v: &[f64]) {
        debug_assert_eq!(v.len(), JOINT_DIM);
        self.n += 1;
        for i in 0..JOINT_DIM {
            self.sum[i] += v[i];
            let vi = v[i];
            for j in i..JOINT_DIM {
                self.outer[(i, j)] += vi * v[j];
            }
        }
    }

    pub fn count(&self) -> usize {
        self.n
    }

    /// Finalise into mean and (symmetric, unbiased) covariance.
    pub fn finish(&self) -> (DVector<f64>, DMatrix<f64>) {
        assert!(self.n > JOINT_DIM, "need more samples ({}) than dims ({JOINT_DIM})", self.n);
        let n = self.n as f64;
        let mean = &self.sum / n;
        let mut cov = DMatrix::zeros(JOINT_DIM, JOINT_DIM);
        for i in 0..JOINT_DIM {
            for j in i..JOINT_DIM {
                let c = (self.outer[(i, j)] - n * mean[i] * mean[j]) / (n - 1.0);
                cov[(i, j)] = c;
                cov[(j, i)] = c;
            }
        }
        (mean, cov)
    }
}

/// Cholesky-based log-determinant. Returns `None` if not positive definite.
pub fn logdet(m: &DMatrix<f64>) -> Option<f64> {
    let chol = m.clone().cholesky()?;
    Some(2.0 * chol.l().diagonal().iter().map(|d| d.ln()).sum::<f64>())
}

/// Differential entropy of a Gaussian with covariance `m`, in nats.
pub fn entropy(m: &DMatrix<f64>) -> Option<f64> {
    let d = m.nrows() as f64;
    logdet(m).map(|ld| 0.5 * (d * (2.0 * std::f64::consts::PI * std::f64::consts::E).ln() + ld))
}

/// The fitted model, already conditioned on `z_t`.
///
/// `Sigma_rest` is the covariance of `[z_{t+1} | y_{t+1}]` given `z_t`. It is constant
/// across samples (that is the homoscedasticity caveat); only the mean moves.
pub struct DynamicsModel {
    pub mean: DVector<f64>,
    /// Regression matrix mapping a centred `z_t` to the conditional mean of `rest`.
    pub a: DMatrix<f64>,
    pub sigma_rest: DMatrix<f64>,
    pub shrinkage: f64,
}

impl DynamicsModel {
    /// Fit from accumulated joint statistics, with ridge shrinkage `lambda` applied
    /// to the diagonal (scaled by mean variance so it is unit-free).
    pub fn fit(stats: &JointStats, lambda: f64) -> Result<Self, String> {
        let (mean, mut cov) = stats.finish();
        let scale = cov.diagonal().mean();
        for i in 0..JOINT_DIM {
            cov[(i, i)] += lambda * scale;
        }

        let s_zz = cov.view((0, 0), (BAND_DIM, BAND_DIM)).into_owned();
        let s_rz = cov.view((BAND_DIM, 0), (REST_DIM, BAND_DIM)).into_owned();
        let s_rr = cov.view((BAND_DIM, BAND_DIM), (REST_DIM, REST_DIM)).into_owned();

        let chol_zz = s_zz
            .clone()
            .cholesky()
            .ok_or("Sigma_{z_t,z_t} is not positive definite -- increase shrinkage")?;
        // A = S_rz * S_zz^{-1}
        let a = chol_zz.solve(&s_rz.transpose()).transpose();
        let mut sigma_rest = &s_rr - &a * s_rz.transpose();
        // Re-symmetrise: the subtraction accumulates asymmetry at the 1e-16 level,
        // which Cholesky is entitled to reject.
        sigma_rest = (&sigma_rest + &sigma_rest.transpose()) * 0.5;

        Ok(Self { mean, a, sigma_rest, shrinkage: lambda })
    }

    /// Conditional mean of `[z_{t+1} | y_{t+1}]` given this sample's `z_t`.
    pub fn rest_mean(&self, z_t: &[f64]) -> DVector<f64> {
        let centred = DVector::from_iterator(BAND_DIM, z_t.iter().zip(self.mean.iter()).map(|(v, m)| v - m));
        let mu_rest = self.mean.rows(BAND_DIM, REST_DIM).into_owned();
        mu_rest + &self.a * centred
    }

    /// Flat `rest`-block indices revealed by probe set `S`.
    pub fn probe_dims(bands: &[usize]) -> Vec<usize> {
        let mut d: Vec<usize> = bands.iter().flat_map(|&b| rest_group_dims(b)).collect();
        d.sort_unstable();
        d
    }

    fn pose_dims() -> Vec<usize> {
        (BAND_DIM..REST_DIM).collect()
    }

    fn submatrix(&self, rows: &[usize], cols: &[usize]) -> DMatrix<f64> {
        DMatrix::from_fn(rows.len(), cols.len(), |i, j| self.sigma_rest[(rows[i], cols[j])])
    }

    /// Posterior covariance of `y_{t+1}` after revealing bands `S`.
    ///
    /// Independent of the observed values -- see the homoscedasticity caveat.
    pub fn pose_posterior_cov(&self, bands: &[usize]) -> Result<DMatrix<f64>, String> {
        let y = Self::pose_dims();
        let s_yy = self.submatrix(&y, &y);
        if bands.is_empty() {
            return Ok(s_yy);
        }
        let p = Self::probe_dims(bands);
        let s_yp = self.submatrix(&y, &p);
        let s_pp = self.submatrix(&p, &p);
        let chol = s_pp
            .clone()
            .cholesky()
            .ok_or("probe block is not positive definite -- increase shrinkage")?;
        let mut post = &s_yy - &s_yp * chol.solve(&s_yp.transpose());
        post = (&post + &post.transpose()) * 0.5;
        Ok(post)
    }

    /// Posterior mean of `y_{t+1}` given `z_t` and the revealed band values.
    pub fn pose_posterior_mean(
        &self,
        z_t: &[f64],
        bands: &[usize],
        revealed: &[f64],
    ) -> Result<DVector<f64>, String> {
        let mu_rest = self.rest_mean(z_t);
        let y = Self::pose_dims();
        let mu_y = DVector::from_iterator(POSE_DIM, y.iter().map(|&i| mu_rest[i]));
        if bands.is_empty() {
            return Ok(mu_y);
        }
        let p = Self::probe_dims(bands);
        assert_eq!(revealed.len(), p.len(), "revealed values must match probe dims");
        let s_yp = self.submatrix(&y, &p);
        let s_pp = self.submatrix(&p, &p);
        let chol = s_pp.clone().cholesky().ok_or("probe block not positive definite")?;
        let resid = DVector::from_iterator(p.len(), p.iter().zip(revealed).map(|(&i, v)| v - mu_rest[i]));
        Ok(mu_y + &s_yp * chol.solve(&resid))
    }

    /// Expected information gain about `y_{t+1}` from revealing bands `S`, in nats.
    ///
    /// `EIG(S) = H(y | z_t) - H(y | z_t, z_S)`. For a Gaussian this reduces to a
    /// log-determinant ratio and does not depend on the observed values.
    pub fn eig(&self, bands: &[usize]) -> Result<f64, String> {
        let prior = self.pose_posterior_cov(&[])?;
        let post = self.pose_posterior_cov(bands)?;
        let hp = entropy(&prior).ok_or("prior covariance not PD")?;
        let hq = entropy(&post).ok_or("posterior covariance not PD")?;
        Ok(hp - hq)
    }

    /// Greedy EIG-maximising probe set of size `k`.
    pub fn greedy_probe_set(&self, k: usize) -> Result<Vec<usize>, String> {
        let mut chosen: Vec<usize> = Vec::new();
        while chosen.len() < k {
            let mut best = (f64::NEG_INFINITY, usize::MAX);
            for b in 0..NUM_PROBE_GROUPS {
                if chosen.contains(&b) {
                    continue;
                }
                let mut cand = chosen.clone();
                cand.push(b);
                if let Ok(g) = self.eig(&cand) {
                    if g > best.0 {
                        best = (g, b);
                    }
                }
            }
            if best.1 == usize::MAX {
                return Err(format!("no admissible band to add at size {}", chosen.len()));
            }
            chosen.push(best.1);
        }
        Ok(chosen)
    }
}

/// Multivariate Gaussian negative log-likelihood of `x` under `N(mu, cov)`, in nats.
/// This is the pre-registered proper scoring rule.
pub fn gaussian_nll(x: &DVector<f64>, mu: &DVector<f64>, cov: &DMatrix<f64>) -> Option<f64> {
    let d = x.len() as f64;
    let chol = cov.clone().cholesky()?;
    let r = x - mu;
    let solved = chol.solve(&r);
    let quad = r.dot(&solved);
    let ld = 2.0 * chol.l().diagonal().iter().map(|v| v.ln()).sum::<f64>();
    Some(0.5 * (d * (2.0 * std::f64::consts::PI).ln() + ld + quad))
}
