//! Rung 4: value-aware heteroscedastic head with an observation head, for
//! Monte-Carlo expected information gain.
//!
//! ## What changed from rung 3, and why
//!
//! The rung-3 target-peeking bound proved the information is present (+2.9 pts PCK@20
//! at K=9, t=+22.3) and that rung-3's closed-form EIG cannot reach it (−0.4 pts).
//! Diagnosis: rung 3's variance head is **blind to revealed values** by construction,
//! so it can only express "how much would set S teach me on average", never "how much
//! did *these* values teach me". The bound shows probe value is value-dependent.
//!
//! Rung 4 therefore:
//!   - lets the **variance head see revealed values**, and
//!   - adds an **observation head** `p(z_{t+1} | z_t, revealed)` so candidate values
//!     can be *sampled from the model* when scoring a probe.
//!
//! The price is that EIG no longer has a closed form -- which is the point, since a
//! closed form is precisely what could not represent value-dependence.
//!
//! **Still peek-free:** candidate values come from the observation head, never from
//! the future. Only the target-peeking *bound* reads the future, and it is labelled
//! as an unachievable ceiling.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::bands::BAND_DIM;
use crate::gaussian::POSE_DIM;
use crate::hetero::{MAX_LOG_VAR, MIN_LOG_VAR};

#[derive(Module, Debug)]
pub struct HeteroV2<B: Backend> {
    trunk1: Linear<B>,
    trunk2: Linear<B>,
    pose_mu: Linear<B>,
    pose_lv: Linear<B>,
    obs_mu: Linear<B>,
    obs_lv: Linear<B>,
    act: Relu,
}

impl<B: Backend> HeteroV2<B> {
    pub fn new(device: &B::Device, hidden: usize) -> Self {
        let input = BAND_DIM * 3; // z_t, mask*z_{t+1}, mask
        Self {
            trunk1: LinearConfig::new(input, hidden).init(device),
            trunk2: LinearConfig::new(hidden, hidden / 2).init(device),
            pose_mu: LinearConfig::new(hidden / 2, POSE_DIM).init(device),
            // Variance head now shares the trunk, so it SEES revealed values.
            pose_lv: LinearConfig::new(hidden / 2, POSE_DIM).init(device),
            obs_mu: LinearConfig::new(hidden / 2, BAND_DIM).init(device),
            obs_lv: LinearConfig::new(hidden / 2, BAND_DIM).init(device),
            act: Relu::new(),
        }
    }

    fn trunk(&self, z_t: Tensor<B, 2>, masked_next: Tensor<B, 2>, mask: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = Tensor::cat(vec![z_t, masked_next, mask], 1);
        let h = self.act.forward(self.trunk1.forward(x));
        self.act.forward(self.trunk2.forward(h))
    }

    /// `(pose_mu, pose_log_var, obs_mu, obs_log_var)`.
    pub fn forward(
        &self,
        z_t: Tensor<B, 2>,
        masked_next: Tensor<B, 2>,
        mask: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let h = self.trunk(z_t, masked_next, mask);
        (
            self.pose_mu.forward(h.clone()),
            self.pose_lv.forward(h.clone()).clamp(MIN_LOG_VAR, MAX_LOG_VAR),
            self.obs_mu.forward(h.clone()),
            self.obs_lv.forward(h).clamp(MIN_LOG_VAR, MAX_LOG_VAR),
        )
    }

    /// Pose head only.
    pub fn pose(
        &self,
        z_t: Tensor<B, 2>,
        masked_next: Tensor<B, 2>,
        mask: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let h = self.trunk(z_t, masked_next, mask);
        (
            self.pose_mu.forward(h.clone()),
            self.pose_lv.forward(h).clamp(MIN_LOG_VAR, MAX_LOG_VAR),
        )
    }

    /// Observation head only -- supplies the sampling distribution for MC EIG.
    pub fn observation(
        &self,
        z_t: Tensor<B, 2>,
        masked_next: Tensor<B, 2>,
        mask: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let h = self.trunk(z_t, masked_next, mask);
        (
            self.obs_mu.forward(h.clone()),
            self.obs_lv.forward(h).clamp(MIN_LOG_VAR, MAX_LOG_VAR),
        )
    }
}

/// Diagonal-Gaussian NLL summed over dims, averaged over the batch, in nats.
pub fn diag_nll<B: Backend>(mu: Tensor<B, 2>, log_var: Tensor<B, 2>, y: Tensor<B, 2>) -> Tensor<B, 1> {
    let two_pi = (2.0 * std::f64::consts::PI).ln();
    let inv_var = log_var.clone().neg().exp();
    let per_dim = (y - mu).powf_scalar(2.0) * inv_var + log_var + two_pi;
    per_dim.sum_dim(1).squeeze::<1>() * 0.5
}

/// Observation NLL restricted to the UNREVEALED dims -- predicting values already
/// revealed is free and would flatter the model.
pub fn masked_obs_nll<B: Backend>(
    mu: Tensor<B, 2>,
    log_var: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let two_pi = (2.0 * std::f64::consts::PI).ln();
    let keep = mask.neg().add_scalar(1.0); // 1 where UNrevealed
    let inv_var = log_var.clone().neg().exp();
    let per_dim = ((target - mu).powf_scalar(2.0) * inv_var + log_var + two_pi) * keep.clone();
    let n = keep.sum_dim(1).squeeze::<1>().clamp(1.0, BAND_DIM as f64);
    per_dim.sum_dim(1).squeeze::<1>() * 0.5 / n
}
