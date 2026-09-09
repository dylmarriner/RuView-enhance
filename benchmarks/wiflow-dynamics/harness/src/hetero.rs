//! Rung 3: neural heteroscedastic action-conditioned dynamics head (Burn, CPU).
//!
//! See the rung-3 pre-registration in `../STATE.md`. The single question:
//! **does an input-dependent `Sigma(z_t, mask)` let EIG close any of the
//! target-peeking gap that rung 2 left open?**
//!
//! ## Why the variance head is blind to revealed values
//!
//! To score a *candidate* probe set we must know how much observing it would shrink
//! pose uncertainty -- before observing it. If `sigma` depended on the revealed
//! values, that would require sampling them (or peeking). By making the variance head
//! a function of `(z_t, mask)` only, EIG stays closed-form AND peek-free:
//!
//! ```text
//! EIG(S) = 0.5 * sum_i [ log sigma_i^2(z_t, M) - log sigma_i^2(z_t, M u S) ]
//! ```
//!
//! `sigma` still depends on `z_t`, so the EIG-optimal set now **varies per sample** --
//! precisely the capability rung 2 lacked.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::bands::{BAND_DIM, BANDS_PER_GROUP, NUM_PROBE_GROUPS};
use crate::gaussian::POSE_DIM;

/// Lower bound on predicted variance, in log space. Without a floor the NLL can be
/// driven to -inf by collapsing sigma on easy samples -- a Goodhart failure mode for
/// this loss, and one the coverage check would catch only after the fact.
pub const MIN_LOG_VAR: f64 = -14.0;
pub const MAX_LOG_VAR: f64 = 4.0;

#[derive(Module, Debug)]
pub struct HeteroModel<B: Backend> {
    mean1: Linear<B>,
    mean2: Linear<B>,
    mean3: Linear<B>,
    var1: Linear<B>,
    var2: Linear<B>,
    var3: Linear<B>,
    act: Relu,
}

impl<B: Backend> HeteroModel<B> {
    pub fn new(device: &B::Device, hidden_mean: usize, hidden_var: usize) -> Self {
        // mean head sees z_t, the masked next observation, and the mask
        let mean_in = BAND_DIM * 3;
        // variance head sees ONLY z_t and the mask (see module docs)
        let var_in = BAND_DIM * 2;
        Self {
            mean1: LinearConfig::new(mean_in, hidden_mean).init(device),
            mean2: LinearConfig::new(hidden_mean, hidden_mean / 2).init(device),
            mean3: LinearConfig::new(hidden_mean / 2, POSE_DIM).init(device),
            var1: LinearConfig::new(var_in, hidden_var).init(device),
            var2: LinearConfig::new(hidden_var, hidden_var / 2).init(device),
            var3: LinearConfig::new(hidden_var / 2, POSE_DIM).init(device),
            act: Relu::new(),
        }
    }

    /// `z_t`: [batch, BAND_DIM]; `masked_next`: [batch, BAND_DIM] (revealed values,
    /// zeros elsewhere); `mask`: [batch, BAND_DIM] of 1.0/0.0.
    /// Returns `(mu, log_var)`, each [batch, POSE_DIM].
    pub fn forward(
        &self,
        z_t: Tensor<B, 2>,
        masked_next: Tensor<B, 2>,
        mask: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let mean_in = Tensor::cat(vec![z_t.clone(), masked_next, mask.clone()], 1);
        let h = self.act.forward(self.mean1.forward(mean_in));
        let h = self.act.forward(self.mean2.forward(h));
        let mu = self.mean3.forward(h);

        let var_in = Tensor::cat(vec![z_t, mask], 1);
        let g = self.act.forward(self.var1.forward(var_in));
        let g = self.act.forward(self.var2.forward(g));
        let log_var = self.var3.forward(g).clamp(MIN_LOG_VAR, MAX_LOG_VAR);

        (mu, log_var)
    }

    /// Variance head alone -- all that EIG needs.
    pub fn log_var(&self, z_t: Tensor<B, 2>, mask: Tensor<B, 2>) -> Tensor<B, 2> {
        let var_in = Tensor::cat(vec![z_t, mask], 1);
        let g = self.act.forward(self.var1.forward(var_in));
        let g = self.act.forward(self.var2.forward(g));
        self.var3.forward(g).clamp(MIN_LOG_VAR, MAX_LOG_VAR)
    }
}

/// Mean Gaussian NLL for a diagonal predictive, in nats per sample.
///
/// `0.5 * sum_i [ (y_i - mu_i)^2 / sigma_i^2 + log sigma_i^2 + log(2*pi) ]`
pub fn hetero_nll<B: Backend>(
    mu: Tensor<B, 2>,
    log_var: Tensor<B, 2>,
    y: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let two_pi = (2.0 * std::f64::consts::PI).ln();
    let inv_var = log_var.clone().neg().exp();
    let sq = (y - mu).powf_scalar(2.0);
    let per_dim = sq * inv_var + log_var + two_pi;
    per_dim.sum_dim(1).squeeze::<1>() * 0.5
}

/// Expand a per-probe-group mask to the per-band feature mask the model consumes.
pub fn expand_mask(groups: &[usize]) -> Vec<f32> {
    let mut m = vec![0.0f32; BAND_DIM];
    for &g in groups {
        debug_assert!(g < NUM_PROBE_GROUPS);
        for d in (g * BANDS_PER_GROUP)..((g + 1) * BANDS_PER_GROUP) {
            m[d] = 1.0;
        }
    }
    m
}
