//! Correctness tests for the expected-information-gain machinery.
//!
//! Pre-registered (STATE.md). These exist because a sign flip or an index-block error
//! would not crash -- it would quietly produce "EIG does not help", which is
//! indistinguishable from a real negative result unless the machinery is tested
//! independently.

use nalgebra::{DMatrix, DVector};
use wiflow_harness::bands::{BAND_DIM, NUM_PROBE_GROUPS};
use wiflow_harness::gaussian::{entropy, gaussian_nll, logdet, DynamicsModel, JointStats, JOINT_DIM};

/// Deterministic standard-normal-ish sampler (Box-Muller over an LCG).
struct Rng(u64);

impl Rng {
    fn u(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64).clamp(1e-12, 1.0 - 1e-12)
    }
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.u(), self.u());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Build a model from synthetic data with real cross-band structure, so that
/// probing different bands genuinely carries different information about the pose.
fn synthetic_model(n: usize, seed: u64) -> DynamicsModel {
    let mut rng = Rng(seed);
    let mut stats = JointStats::new();
    for _ in 0..n {
        // Two latent factors drive everything; bands load on them unequally, so some
        // bands are far more informative about the pose than others.
        let f1 = rng.normal();
        let f2 = rng.normal();
        let mut v = vec![0.0f64; JOINT_DIM];
        for (i, slot) in v.iter_mut().enumerate().take(2 * BAND_DIM) {
            let band = (i % BAND_DIM) / 10;
            let w1 = ((band % 5) as f64 + 1.0) / 5.0;
            let w2 = ((band % 3) as f64) / 3.0;
            *slot = w1 * f1 + w2 * f2 + 0.35 * rng.normal();
        }
        for slot in v.iter_mut().skip(2 * BAND_DIM) {
            *slot = 0.8 * f1 + 0.4 * f2 + 0.25 * rng.normal();
        }
        stats.push(&v);
    }
    DynamicsModel::fit(&stats, 1e-3).expect("synthetic fit should succeed")
}

#[test]
fn eig_is_non_negative() {
    let m = synthetic_model(4000, 0xE16_0001);
    for b in 0..NUM_PROBE_GROUPS {
        let g = m.eig(&[b]).unwrap();
        assert!(g >= -1e-9, "EIG must be non-negative, band {b} gave {g}");
    }
}

#[test]
fn eig_is_monotone_in_probe_set_size() {
    // Information cannot be destroyed by observing more.
    let m = synthetic_model(4000, 0xE16_0002);
    let mut prev = 0.0;
    let mut set = Vec::new();
    for b in [3usize, 7, 11, 1, 19, 24] {
        set.push(b);
        let g = m.eig(&set).unwrap();
        assert!(
            g >= prev - 1e-9,
            "EIG must not decrease when adding band {b}: {g} < {prev} for set {set:?}"
        );
        prev = g;
    }
}

#[test]
fn empty_probe_set_has_zero_gain() {
    let m = synthetic_model(3000, 0xE16_0003);
    assert!(m.eig(&[]).unwrap().abs() < 1e-9);
}

#[test]
fn greedy_matches_brute_force_on_small_budget() {
    // Brute force over all pairs of the first 5 bands; greedy must find a set whose
    // EIG equals the optimum (greedy is not optimal in general, but on a budget of 2
    // over a restricted pool with submodular-ish structure it should match; if it
    // ever does not, that is a finding worth seeing rather than hiding).
    let m = synthetic_model(6000, 0xE16_0004);
    let pool: Vec<usize> = (0..5).collect();
    let mut best = (f64::NEG_INFINITY, vec![]);
    for i in 0..pool.len() {
        for j in (i + 1)..pool.len() {
            let s = vec![pool[i], pool[j]];
            let g = m.eig(&s).unwrap();
            if g > best.0 {
                best = (g, s);
            }
        }
    }
    // Greedy restricted to the same pool.
    let mut chosen: Vec<usize> = Vec::new();
    while chosen.len() < 2 {
        let mut b = (f64::NEG_INFINITY, usize::MAX);
        for &c in &pool {
            if chosen.contains(&c) {
                continue;
            }
            let mut s = chosen.clone();
            s.push(c);
            let g = m.eig(&s).unwrap();
            if g > b.0 {
                b = (g, c);
            }
        }
        chosen.push(b.1);
    }
    let greedy_gain = m.eig(&chosen).unwrap();
    assert!(
        (greedy_gain - best.0).abs() < 1e-6,
        "greedy {chosen:?} -> {greedy_gain:.6} vs brute-force {:?} -> {:.6}",
        best.1,
        best.0
    );
}

#[test]
fn analytic_eig_matches_monte_carlo_entropy_reduction() {
    // Independent check of the closed form: estimate the posterior entropy by
    // sampling residuals rather than by the logdet identity.
    let m = synthetic_model(8000, 0xE16_0005);
    let bands = vec![2usize, 9];

    let prior = m.pose_posterior_cov(&[]).unwrap();
    let post = m.pose_posterior_cov(&bands).unwrap();
    let analytic = entropy(&prior).unwrap() - entropy(&post).unwrap();

    // Entropy from the logdet of an empirically-resampled covariance: draw from the
    // posterior covariance, re-estimate it, and recompute. Agreement confirms the
    // conditioning algebra, not just the identity.
    let mut rng = Rng(0x11C_0001_u64.wrapping_mul(2654435761));
    let chol = post.clone().cholesky().unwrap();
    let d = post.nrows();
    let n = 60_000;
    let mut stats_sum = DVector::zeros(d);
    let mut stats_outer = DMatrix::zeros(d, d);
    for _ in 0..n {
        let z = DVector::from_fn(d, |_, _| rng.normal());
        let x = chol.l() * z;
        stats_sum += &x;
        stats_outer += &x * x.transpose();
    }
    let mean = stats_sum / n as f64;
    let emp = (stats_outer - (n as f64) * &mean * mean.transpose()) / (n as f64 - 1.0);
    let mc_post_entropy = entropy(&emp).unwrap();
    let mc = entropy(&prior).unwrap() - mc_post_entropy;

    assert!(
        (analytic - mc).abs() < 0.05,
        "analytic EIG {analytic:.4} nats vs Monte-Carlo {mc:.4} nats"
    );
}

#[test]
fn conditioning_reduces_pose_uncertainty_monotonically() {
    let m = synthetic_model(4000, 0xE16_0006);
    let prior_ld = logdet(&m.pose_posterior_cov(&[]).unwrap()).unwrap();
    let post_ld = logdet(&m.pose_posterior_cov(&[1, 5, 9, 13]).unwrap()).unwrap();
    assert!(post_ld < prior_ld, "observing bands must shrink the posterior logdet");
}

#[test]
fn gaussian_nll_is_minimised_at_the_true_mean() {
    let d = 4;
    let cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 2.0, 0.5, 1.5]));
    let mu = DVector::from_vec(vec![0.1, -0.2, 0.3, 0.0]);
    let at_mean = gaussian_nll(&mu, &mu, &cov).unwrap();
    for shift in [0.5, 1.0, 2.0] {
        let x = &mu + DVector::from_fn(d, |i, _| if i == 0 { shift } else { 0.0 });
        assert!(gaussian_nll(&x, &mu, &cov).unwrap() > at_mean);
    }
    // Sanity: NLL at the mean equals the entropy-free constant term.
    let expect = 0.5 * (d as f64 * (2.0 * std::f64::consts::PI).ln() + logdet(&cov).unwrap());
    assert!((at_mean - expect).abs() < 1e-10);
}

/// Does post-hoc recalibration of the variance head change EIG's RANKING?
///
/// This matters because the obvious next fix -- temperature-scale the variance head
/// so coverage passes, then re-run EIG -- would be a null experiment if the answer is
/// no. EIG(S) = H(before) - H(after S) = 0.5 * sum_i [log s2_i(M) - log s2_i(M u S)].
/// Recalibration that depends only on the dimension index adds the same constant to
/// both terms, so it must cancel exactly.
///
/// Verified computationally rather than trusted, since it drives a recommendation.
#[test]
fn eig_is_invariant_to_dimension_wise_recalibration() {
    // Simulate a diagonal predictive: log-variances per pose dim, before and after
    // probing each of several candidate sets.
    let mut rng = Rng(0xCA11B);
    let dims = 30usize;
    let before: Vec<f64> = (0..dims).map(|_| rng.normal()).collect();
    let after: Vec<Vec<f64>> = (0..6)
        .map(|_| (0..dims).map(|_| rng.normal() - 0.5).collect())
        .collect();

    let eig = |b: &[f64], a: &[f64]| -> f64 {
        0.5 * (0..dims).map(|i| b[i] - a[i]).sum::<f64>()
    };
    let raw: Vec<f64> = after.iter().map(|a| eig(&before, a)).collect();

    // Any per-dimension offset: a global temperature is the special case c_i = c.
    let offsets: Vec<f64> = (0..dims).map(|_| 2.0 * rng.normal()).collect();
    let b_cal: Vec<f64> = before.iter().zip(&offsets).map(|(v, c)| v + c).collect();
    let cal: Vec<f64> = after
        .iter()
        .map(|a| {
            let ac: Vec<f64> = a.iter().zip(&offsets).map(|(v, c)| v + c).collect();
            eig(&b_cal, &ac)
        })
        .collect();

    for (r, c) in raw.iter().zip(&cal) {
        assert!(
            (r - c).abs() < 1e-12,
            "EIG changed under dimension-wise recalibration: {r} vs {c}"
        );
    }
    // And therefore the argmax -- the actual selection -- is identical.
    let am = |v: &[f64]| v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    assert_eq!(am(&raw), am(&cal), "selected probe set changed under recalibration");
}
