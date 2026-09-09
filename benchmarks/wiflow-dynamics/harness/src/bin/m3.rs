//! M3 + M4: action-conditioned dynamics model and information-gain probe selection.
//!
//! Follows the pre-registration in `../STATE.md`. Every baseline, metric and guard
//! below was fixed before the model was fitted.

use nalgebra::{DMatrix, DVector};
use std::time::Instant;

use wiflow_harness::bands::{BAND_DIM, NUM_BANDS, NUM_PROBE_GROUPS, CHANNELS_PER_BAND, BANDS_PER_GROUP};
use wiflow_harness::gaussian::{
    entropy, gaussian_nll, DynamicsModel, JointStats, JOINT_DIM, POSE_DIM, REST_DIM,
};
use wiflow_harness::guards::{check_constant_pose, CorruptionMasks};
use wiflow_harness::metrics::score;
use wiflow_harness::npy::Npy;

const RESULTS: &str =
    "/home/ruvultra/projects/ruview-worktrees/ruforecast-rust/benchmarks/wiflow-std/results";

/// Central-interval z-scores for the pre-registered coverage levels.
const COVERAGE_LEVELS: [(f64, f64); 4] =
    [(0.50, 0.6744897501960817), (0.80, 1.2815515655446004), (0.90, 1.6448536269514722), (0.95, 1.959963984540054)];

/// Pre-registered compute bounds (see STATE.md addendum).
const FIT_SUBSAMPLE: usize = 60_000; // >= 54x the 1110-dim joint
const M4_SUBSAMPLE: usize = 2_000; // paired across all policies
const RANDOM_SEEDS: usize = 10;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() >> 11) as usize % n
    }
}

fn subsample(items: &[usize], k: usize, seed: u64) -> Vec<usize> {
    if items.len() <= k {
        return items.to_vec();
    }
    let mut rng = Rng(seed);
    let mut pool = items.to_vec();
    for i in 0..k {
        let j = i + rng.below(pool.len() - i);
        pool.swap(i, j);
    }
    pool.truncate(k);
    pool
}

/// Valid `(t, t+1)` pairs inside one split: same file, both windows uncorrupted.
fn build_pairs(in_split: &[bool], masks: &CorruptionMasks, w2f: &[i64]) -> Vec<usize> {
    (0..in_split.len() - 1)
        .filter(|&t| {
            in_split[t]
                && w2f[t] == w2f[t + 1]
                && !masks.is_corrupt(t)
                && !masks.is_corrupt(t + 1)
        })
        .collect()
}

fn joint_vector(z: &[f32], gt: &[f32], t: usize) -> Vec<f64> {
    let mut v = Vec::with_capacity(JOINT_DIM);
    v.extend((0..BAND_DIM).map(|i| z[t * BAND_DIM + i] as f64));
    v.extend((0..BAND_DIM).map(|i| z[(t + 1) * BAND_DIM + i] as f64));
    v.extend((0..POSE_DIM).map(|i| gt[(t + 1) * POSE_DIM + i] as f64));
    v
}

fn main() {
    let fx = std::env::args()
        .nth(1)
        .unwrap_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/../fixtures").to_string());
    let load = |f: &str| Npy::open(format!("{fx}/{f}")).unwrap_or_else(|e| panic!("{e}"));

    let t0 = Instant::now();
    let z_npy = load("band_features.npy");
    let z = z_npy.f32_slice().unwrap();
    let gt_npy = load("all_gt_clean.npy");
    let gt = gt_npy.f32_slice().unwrap();
    let w2f = load("window_to_file.npy").to_i64().unwrap();
    let n_windows = z_npy.shape[0];

    let masks = CorruptionMasks::load(
        &format!("{RESULTS}/nan_windows_mask.npy"),
        &format!("{RESULTS}/big_windows_mask.npy"),
    )
    .unwrap();

    let mut splits = Vec::new();
    for name in ["train_idx.npy", "val_idx.npy", "test_idx.npy"] {
        let idx = load(name).to_i64().unwrap();
        let mut m = vec![false; n_windows];
        for &i in &idx {
            m[i as usize] = true;
        }
        splits.push(m);
    }
    let train_pairs = build_pairs(&splits[0], &masks, &w2f);
    let val_pairs = build_pairs(&splits[1], &masks, &w2f);
    let test_pairs = build_pairs(&splits[2], &masks, &w2f);
    eprintln!(
        "pairs -- train {} val {} test {} ({:.1?})",
        train_pairs.len(),
        val_pairs.len(),
        test_pairs.len(),
        t0.elapsed()
    );

    // ---- rung 1: persistence -------------------------------------------------
    // Is one-step band dynamics trivial? Measure before interpreting anything else.
    let mut ss_res = 0.0f64;
    let mut ss_tot = 0.0f64;
    let mut mean_acc = vec![0.0f64; BAND_DIM];
    for &t in &test_pairs {
        for i in 0..BAND_DIM {
            mean_acc[i] += z[(t + 1) * BAND_DIM + i] as f64;
        }
    }
    for m in mean_acc.iter_mut() {
        *m /= test_pairs.len() as f64;
    }
    for &t in &test_pairs {
        for i in 0..BAND_DIM {
            let truth = z[(t + 1) * BAND_DIM + i] as f64;
            let d = truth - z[t * BAND_DIM + i] as f64;
            ss_res += d * d;
            let dm = truth - mean_acc[i];
            ss_tot += dm * dm;
        }
    }
    let persistence_r2 = 1.0 - ss_res / ss_tot;
    let persistence_rmse = (ss_res / (test_pairs.len() * BAND_DIM) as f64).sqrt();

    // ---- rung 2: joint Gaussian ---------------------------------------------
    let fit_pairs = subsample(&train_pairs, FIT_SUBSAMPLE, 42);
    let mut stats = JointStats::new();
    for &t in &fit_pairs {
        stats.push(&joint_vector(z, gt, t));
    }
    eprintln!("accumulated {} fit samples ({:.1?})", stats.count(), t0.elapsed());

    // Choose shrinkage on VAL, never on test.
    let val_eval = subsample(&val_pairs, 4_000, 7);
    let mut best = (f64::INFINITY, 0.0f64);
    let mut lambda_curve = Vec::new();
    for &lam in &[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1] {
        let Ok(m) = DynamicsModel::fit(&stats, lam) else { continue };
        let Some(nll) = mean_pose_nll(&m, z, gt, &val_eval, &[]) else { continue };
        lambda_curve.push(serde_json::json!({ "lambda": lam, "val_pose_nll": nll }));
        if nll < best.0 {
            best = (nll, lam);
        }
    }
    let lambda = best.1;
    let model = DynamicsModel::fit(&stats, lambda).expect("fit at selected lambda");
    eprintln!("selected lambda {lambda:.0e} (val pose NLL {:.4}) ({:.1?})", best.0, t0.elapsed());

    // ---- M3 evaluation on test ----------------------------------------------
    let test_eval = subsample(&test_pairs, 20_000, 1234);
    let m3_none = evaluate(&model, z, gt, &test_eval, &[]);

    // Degeneracy guard on the dynamics head: the predicted mean must vary across t.
    let mut mu_series: Vec<f32> = Vec::with_capacity(test_eval.len() * POSE_DIM);
    for &t in &test_eval {
        let zt: Vec<f64> = (0..BAND_DIM).map(|i| z[t * BAND_DIM + i] as f64).collect();
        let mu = model.pose_posterior_mean(&zt, &[], &[]).unwrap();
        mu_series.extend(mu.iter().map(|&v| v as f32));
    }
    let mu_verdict = check_constant_pose(&mu_series, test_eval.len());
    mu_verdict.assert_ok("joint-Gaussian dynamics head");

    // Mean-pose bar restricted to exactly these evaluation pairs, so the comparison
    // is like-for-like rather than against the M2 whole-split number.
    let bar = mean_pose_bar(gt, &test_eval);

    // ---- M4: probe policies --------------------------------------------------
    let m4_pairs = subsample(&test_pairs, M4_SUBSAMPLE, 999);
    let mut m4 = Vec::new();
    for &k in &[1usize, 3, 6, 9] {
        let greedy = model.greedy_probe_set(k).expect("greedy probe set");
        let eig_greedy = model.eig(&greedy).unwrap();

        // Fixed policy: evenly spaced bands, chosen a priori (not on test).
        let fixed: Vec<usize> = (0..k).map(|i| i * NUM_PROBE_GROUPS / k).collect();
        let eig_fixed = model.eig(&fixed).unwrap();

        let g = evaluate(&model, z, gt, &m4_pairs, &greedy);
        let f = evaluate(&model, z, gt, &m4_pairs, &fixed);

        // Random policy: mean +/- CI over seeds.
        let mut rand_pck = Vec::new();
        let mut rand_nll = Vec::new();
        let mut rand_eig = Vec::new();
        for s in 0..RANDOM_SEEDS {
            let mut rng = Rng(rand_seed(s));
            let mut pool: Vec<usize> = (0..NUM_PROBE_GROUPS).collect();
            for i in 0..k {
                let j = i + rng.below(pool.len() - i);
                pool.swap(i, j);
            }
            let set: Vec<usize> = pool[..k].to_vec();
            rand_eig.push(model.eig(&set).unwrap());
            let r = evaluate(&model, z, gt, &m4_pairs, &set);
            rand_pck.push(r.pck20);
            rand_nll.push(r.nll);
        }

        // Oracle bracket: per-sample greedy on the TRUE residual. Upper bound.
        let oracle = oracle_evaluate(&model, z, gt, &m4_pairs, k);

        m4.push(serde_json::json!({
            "budget_k": k,
            "eig_greedy_set": greedy,
            "eig_nats": { "greedy": eig_greedy, "fixed": eig_fixed,
                          "random_mean": mean(&rand_eig), "random_sd": sd(&rand_eig) },
            "pck@20": { "eig_greedy": g.pck20, "fixed": f.pck20,
                        "random_mean": mean(&rand_pck), "random_sd": sd(&rand_pck),
                        "oracle": oracle.0 },
            "pose_nll": { "eig_greedy": g.nll, "fixed": f.nll,
                          "random_mean": mean(&rand_nll), "random_sd": sd(&rand_nll) },
            "mpjpe": { "eig_greedy": g.mpjpe, "fixed": f.mpjpe, "oracle": oracle.1 },
            "coverage": g.coverage,
        }));
    }

    let report = serde_json::json!({
        "milestone": "M3 + M4 (rung 2: joint Gaussian). MEASURED by harness/src/bin/m3.rs, this run.",
        "framing": "Active sensing / masking on OFFLINE data. The action selects which \
                    already-recorded coordinates are revealed. NOT a causal intervention.",
        "representation": {
            "bands": NUM_BANDS, "channels_per_band": CHANNELS_PER_BAND,
            "probe_groups": NUM_PROBE_GROUPS, "bands_per_group": BANDS_PER_GROUP,
            "time": "averaged over all 20 frames (MEASURED to be near-free; see bands.rs)",
            "band_dim": BAND_DIM, "joint_dim": JOINT_DIM,
            "lag1_channel_autocorr_justifying_contiguous_bands": 0.739
        },
        "pairs": { "train": train_pairs.len(), "val": val_pairs.len(), "test": test_pairs.len(),
                   "fit_subsample": fit_pairs.len(), "m3_eval": test_eval.len(),
                   "m4_eval": m4_pairs.len() },
        "rung1_persistence": { "band_r2": persistence_r2, "band_rmse": persistence_rmse },
        "shrinkage": { "selected_lambda": lambda, "chosen_on": "val", "curve": lambda_curve },
        "rung2_m3_no_probe": {
            "pose_nll": m3_none.nll, "pck@20": m3_none.pck20, "mpjpe": m3_none.mpjpe,
            "coverage": m3_none.coverage
        },
        "honesty_bar_on_same_pairs": { "pck@20": bar.0, "mpjpe": bar.1 },
        "beats_bar": m3_none.pck20 > bar.0,
        "degeneracy_guard": mu_verdict,
        "m4_policies": m4,
        "caveats": [
            "Sigma is homoscedastic, so EIG(S) depends only on the covariance, not on \
             observed values: the EIG-optimal probe set is IDENTICAL at every timestep. \
             This is information-theoretic set selection, NOT adaptive sensing.",
            "Pose accuracy from 27 band means is NOT comparable to the 96.09% full-CSI \
             number -- different input abstraction. Compare only to the mean-pose bar.",
            "Single fit, single split; no cross-validation over splits."
        ]
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    eprintln!("done ({:.1?})", t0.elapsed());
}


fn rand_seed(s: usize) -> u64 {
    0x9E3779B97F4A7C15u64.wrapping_mul(s as u64 + 1)
}

struct Eval {
    nll: f64,
    pck20: f64,
    mpjpe: f64,
    coverage: serde_json::Value,
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn sd(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() as f64 - 1.0).max(1.0)).sqrt()
}

fn mean_pose_nll(
    model: &DynamicsModel,
    z: &[f32],
    gt: &[f32],
    pairs: &[usize],
    probe: &[usize],
) -> Option<f64> {
    let cov = model.pose_posterior_cov(probe).ok()?;
    let mut total = 0.0;
    for &t in pairs {
        let zt: Vec<f64> = (0..BAND_DIM).map(|i| z[t * BAND_DIM + i] as f64).collect();
        let revealed: Vec<f64> = DynamicsModel::probe_dims(probe)
            .iter()
            .map(|&d| z[(t + 1) * BAND_DIM + d] as f64)
            .collect();
        let mu = model.pose_posterior_mean(&zt, probe, &revealed).ok()?;
        let y = DVector::from_iterator(POSE_DIM, (0..POSE_DIM).map(|i| gt[(t + 1) * POSE_DIM + i] as f64));
        total += gaussian_nll(&y, &mu, &cov)?;
    }
    Some(total / pairs.len() as f64)
}

fn evaluate(model: &DynamicsModel, z: &[f32], gt: &[f32], pairs: &[usize], probe: &[usize]) -> Eval {
    let cov = model.pose_posterior_cov(probe).expect("posterior covariance");
    let sd_marginal: Vec<f64> = (0..POSE_DIM).map(|i| cov[(i, i)].max(0.0).sqrt()).collect();

    let mut preds = Vec::with_capacity(pairs.len() * POSE_DIM);
    let mut truths = Vec::with_capacity(pairs.len() * POSE_DIM);
    let mut nll = 0.0;
    let mut inside = [0usize; COVERAGE_LEVELS.len()];

    for &t in pairs {
        let zt: Vec<f64> = (0..BAND_DIM).map(|i| z[t * BAND_DIM + i] as f64).collect();
        let revealed: Vec<f64> = DynamicsModel::probe_dims(probe)
            .iter()
            .map(|&d| z[(t + 1) * BAND_DIM + d] as f64)
            .collect();
        let mu = model.pose_posterior_mean(&zt, probe, &revealed).expect("posterior mean");
        let y = DVector::from_iterator(POSE_DIM, (0..POSE_DIM).map(|i| gt[(t + 1) * POSE_DIM + i] as f64));
        nll += gaussian_nll(&y, &mu, &cov).expect("nll");
        for (li, (_, zscore)) in COVERAGE_LEVELS.iter().enumerate() {
            inside[li] += (0..POSE_DIM)
                .filter(|&i| (y[i] - mu[i]).abs() <= zscore * sd_marginal[i])
                .count();
        }
        preds.extend(mu.iter().map(|&v| v as f32));
        truths.extend(y.iter().map(|&v| v as f32));
    }

    let s = score(&preds, &truths, pairs.len());
    let denom = (pairs.len() * POSE_DIM) as f64;
    Eval {
        nll: nll / pairs.len() as f64,
        pck20: s.pck_at(0.2),
        mpjpe: s.mpjpe,
        coverage: serde_json::json!(COVERAGE_LEVELS
            .iter()
            .enumerate()
            .map(|(i, (lvl, _))| serde_json::json!({
                "nominal": lvl,
                "empirical": inside[i] as f64 / denom,
                "error_pp": (inside[i] as f64 / denom - lvl) * 100.0
            }))
            .collect::<Vec<_>>()),
    }
}

/// Oracle bracket: choose the probe set per sample using the TRUE next observation.
/// Upper-bounds any admissible policy; if EIG ever exceeds it, the machinery is wrong.
fn oracle_evaluate(
    model: &DynamicsModel,
    z: &[f32],
    gt: &[f32],
    pairs: &[usize],
    k: usize,
) -> (f64, f64) {
    let mut preds = Vec::with_capacity(pairs.len() * POSE_DIM);
    let mut truths = Vec::with_capacity(pairs.len() * POSE_DIM);
    for &t in pairs {
        let zt: Vec<f64> = (0..BAND_DIM).map(|i| z[t * BAND_DIM + i] as f64).collect();
        let y: Vec<f64> = (0..POSE_DIM).map(|i| gt[(t + 1) * POSE_DIM + i] as f64).collect();
        let mut chosen: Vec<usize> = Vec::new();
        while chosen.len() < k {
            let mut best = (f64::INFINITY, usize::MAX);
            for b in 0..NUM_PROBE_GROUPS {
                if chosen.contains(&b) {
                    continue;
                }
                let mut cand = chosen.clone();
                cand.push(b);
                let revealed: Vec<f64> = DynamicsModel::probe_dims(&cand)
                    .iter()
                    .map(|&d| z[(t + 1) * BAND_DIM + d] as f64)
                    .collect();
                if let Ok(mu) = model.pose_posterior_mean(&zt, &cand, &revealed) {
                    let err: f64 = (0..POSE_DIM).map(|i| (mu[i] - y[i]).powi(2)).sum();
                    if err < best.0 {
                        best = (err, b);
                    }
                }
            }
            chosen.push(best.1);
        }
        let revealed: Vec<f64> = DynamicsModel::probe_dims(&chosen)
            .iter()
            .map(|&d| z[(t + 1) * BAND_DIM + d] as f64)
            .collect();
        let mu = model.pose_posterior_mean(&zt, &chosen, &revealed).unwrap();
        preds.extend(mu.iter().map(|&v| v as f32));
        truths.extend(y.iter().map(|&v| v as f32));
    }
    let s = score(&preds, &truths, pairs.len());
    (s.pck_at(0.2), s.mpjpe)
}

/// Mean-pose honesty bar computed on exactly the given pairs.
fn mean_pose_bar(gt: &[f32], pairs: &[usize]) -> (f64, f64) {
    let n = pairs.len();
    let mut mu = vec![0.0f64; POSE_DIM];
    for &t in pairs {
        for i in 0..POSE_DIM {
            mu[i] += gt[(t + 1) * POSE_DIM + i] as f64;
        }
    }
    let bar: Vec<f32> = mu.iter().map(|v| (v / n as f64) as f32).collect();
    let preds: Vec<f32> = bar.repeat(n);
    let truths: Vec<f32> = pairs
        .iter()
        .flat_map(|&t| (0..POSE_DIM).map(move |i| gt[(t + 1) * POSE_DIM + i]))
        .collect();
    let s = score(&preds, &truths, n);
    (s.pck_at(0.2), s.mpjpe)
}

#[allow(dead_code)]
fn unused(_: &DMatrix<f64>, _: fn(&DMatrix<f64>) -> Option<f64>) {
    let _ = entropy;
    let _ = REST_DIM;
}
