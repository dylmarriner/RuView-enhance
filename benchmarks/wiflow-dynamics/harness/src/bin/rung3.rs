//! Rung 3 driver: train the heteroscedastic head, then run it through the UNCHANGED
//! rung-2 evaluation (same 20,000 test pairs, same bar, same coverage protocol, same
//! policies and paired tests).
//!
//! Pass condition, pre-registered in `../STATE.md` before this file existed:
//!   1. beats rung-2 joint-Gaussian held-out pose NLL, AND
//!   2. coverage within +/-3 pp at ALL four levels, AND
//!   3. EIG-greedy beats random PCK@20 by more than 2 SE, paired, at some budget.
//! Miss any one -> "rung 3 did not help".

use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::backend::Autodiff;
use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Tensor, TensorData};
use std::time::Instant;

use wiflow_harness::bands::{BAND_DIM, NUM_PROBE_GROUPS};
use wiflow_harness::gaussian::POSE_DIM;
use wiflow_harness::guards::{check_constant_pose, CorruptionMasks};
use wiflow_harness::hetero::{expand_mask, hetero_nll, HeteroModel};
use wiflow_harness::metrics::score;
use wiflow_harness::npy::Npy;

type B = Autodiff<NdArray<f32>>;
type BI = NdArray<f32>;

const RESULTS: &str =
    "/home/ruvultra/projects/ruview-worktrees/ruforecast-rust/benchmarks/wiflow-std/results";
const COVERAGE_LEVELS: [(f64, f64); 4] = [
    (0.50, 0.6744897501960817),
    (0.80, 1.2815515655446004),
    (0.90, 1.6448536269514722),
    (0.95, 1.959963984540054),
];
/// CITED from results/m3_m4.json -- the rung-2 numbers rung 3 must beat.
const RUNG2_NLL: f64 = -113.29;
const RUNG2_BEST_PCK: f64 = 0.8208;

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

fn build_pairs(in_split: &[bool], masks: &CorruptionMasks, w2f: &[i64]) -> Vec<usize> {
    (0..in_split.len() - 1)
        .filter(|&t| in_split[t] && w2f[t] == w2f[t + 1] && !masks.is_corrupt(t) && !masks.is_corrupt(t + 1))
        .collect()
}

fn random_groups(rng: &mut Rng, k: usize) -> Vec<usize> {
    let mut pool: Vec<usize> = (0..NUM_PROBE_GROUPS).collect();
    for i in 0..k {
        let j = i + rng.below(pool.len() - i);
        pool.swap(i, j);
    }
    pool[..k].to_vec()
}

/// CONTROL for the target-peeking bound: best-of-C RANDOM sets, also chosen by peeking.
///
/// Why this is required before any "the information is real" claim. The oracle takes an
/// argmax over ~C candidates per sample, scored against the SAME target used to choose
/// them. Per-sample PCK@20 is a mean of 15 indicators, so it is noisy, and the expected
/// maximum of C noisy scores exceeds the true best whether or not any information is
/// exploited. If best-of-C RANDOM reaches the oracle's number, the oracle's margin is
/// selection noise, not information. `c` is matched to the oracle's greedy candidate count.
fn best_of_c_random_peeking(
    m: &HeteroModel<BI>,
    z: &[f32],
    gt: &[f32],
    pairs: &[usize],
    k: usize,
    c: usize,
    seed: u64,
    d: &NdArrayDevice,
) -> (f64, Vec<f64>) {
    let n = pairs.len();
    let y: Vec<f32> = pairs.iter().flat_map(|&t| (0..POSE_DIM).map(move |i| gt[(t + 1) * POSE_DIM + i])).collect();
    let mut best_err = vec![f64::INFINITY; n];
    let mut best_pred = vec![0.0f32; n * POSE_DIM];
    let mut rng = Rng(seed);

    for _cand in 0..c {
        let gs: Vec<Vec<usize>> = (0..n).map(|_| random_groups(&mut rng, k)).collect();
        let (zt, mn, mk, _yy) = make_batch::<BI>(z, gt, pairs, &gs, d);
        let (pmu, _) = m.forward(zt, mn, mk);
        let mud: Vec<f32> = pmu.into_data().to_vec().unwrap();
        for i in 0..n {
            let e: f64 = (0..POSE_DIM)
                .map(|q| { let j = i * POSE_DIM + q; ((mud[j] - y[j]) as f64).powi(2) })
                .sum();
            if e < best_err[i] {
                best_err[i] = e;
                best_pred[i * POSE_DIM..(i + 1) * POSE_DIM]
                    .copy_from_slice(&mud[i * POSE_DIM..(i + 1) * POSE_DIM]);
            }
        }
    }
    let sc = score(&best_pred, &y, n);
    let per_sample: Vec<f64> = (0..n)
        .map(|i| score(&best_pred[i * POSE_DIM..(i + 1) * POSE_DIM], &y[i * POSE_DIM..(i + 1) * POSE_DIM], 1).pck_at(0.2))
        .collect();
    (sc.pck_at(0.2), per_sample)
}

/// File-level block bootstrap on a paired per-sample difference.
///
/// Adjacent windows within a recording file are enormously correlated (pose persistence
/// between adjacent windows is 0.9970), so a per-sample i.i.d. SE understates uncertainty
/// badly. Resampling whole FILES respects that clustering.
/// Returns (observed delta, 2.5th pct, 97.5th pct).
fn block_bootstrap(
    a: &[f64],
    b: &[f64],
    pairs: &[usize],
    w2f: &[i64],
    reps: usize,
    seed: u64,
) -> (f64, f64, f64) {
    let diffs: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let mut by_file: std::collections::HashMap<i64, Vec<usize>> = std::collections::HashMap::new();
    for (i, &t) in pairs.iter().enumerate() {
        by_file.entry(w2f[t]).or_default().push(i);
    }
    let files: Vec<&Vec<usize>> = by_file.values().collect();
    let nf = files.len();
    let mut rng = Rng(seed);
    let mut means = Vec::with_capacity(reps);
    for _ in 0..reps {
        let (mut sum, mut cnt) = (0.0f64, 0usize);
        for _ in 0..nf {
            for &i in files[rng.below(nf)] {
                sum += diffs[i];
                cnt += 1;
            }
        }
        means.push(sum / cnt.max(1) as f64);
    }
    means.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let obs = diffs.iter().sum::<f64>() / diffs.len() as f64;
    let lo = means[(reps as f64 * 0.025) as usize];
    let hi = means[((reps as f64 * 0.975) as usize).min(reps - 1)];
    (obs, lo, hi)
}

/// Per-sample greedy probe selection using the TRUE y_{t+1} -- a target-peeking upper
/// bound on every policy, achievable by none.
fn oracle_probe_sets(
    model: &HeteroModel<BI>,
    z: &[f32],
    gt: &[f32],
    pairs: &[usize],
    k: usize,
    device: &NdArrayDevice,
) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new(); pairs.len()];
    for _step in 0..k {
        let mut best: Vec<(f64, usize)> = vec![(f64::INFINITY, usize::MAX); pairs.len()];
        for cand in 0..NUM_PROBE_GROUPS {
            let gs: Vec<Vec<usize>> = (0..pairs.len())
                .map(|i| {
                    let mut set = out[i].clone();
                    if !set.contains(&cand) {
                        set.push(cand);
                    }
                    set
                })
                .collect();
            let (zt, mn, mk, yy) = make_batch::<BI>(z, gt, pairs, &gs, device);
            let (mu, _) = model.forward(zt, mn, mk);
            let mud: Vec<f32> = mu.into_data().to_vec().unwrap();
            let yd: Vec<f32> = yy.into_data().to_vec().unwrap();
            for i in 0..pairs.len() {
                if out[i].contains(&cand) {
                    continue;
                }
                let err: f64 = (0..POSE_DIM)
                    .map(|d| {
                        let j = i * POSE_DIM + d;
                        ((mud[j] - yd[j]) as f64).powi(2)
                    })
                    .sum();
                if err < best[i].0 {
                    best[i] = (err, cand);
                }
            }
        }
        for i in 0..pairs.len() {
            if best[i].1 != usize::MAX {
                out[i].push(best[i].1);
            }
        }
    }
    out
}

/// Assemble one training/eval batch.
fn make_batch<Bk: burn::tensor::backend::Backend>(
    z: &[f32],
    gt: &[f32],
    pairs: &[usize],
    groups_per: &[Vec<usize>],
    device: &Bk::Device,
) -> (Tensor<Bk, 2>, Tensor<Bk, 2>, Tensor<Bk, 2>, Tensor<Bk, 2>) {
    let n = pairs.len();
    let mut zt = Vec::with_capacity(n * BAND_DIM);
    let mut mn = Vec::with_capacity(n * BAND_DIM);
    let mut mk = Vec::with_capacity(n * BAND_DIM);
    let mut yy = Vec::with_capacity(n * POSE_DIM);
    for (i, &t) in pairs.iter().enumerate() {
        let m = expand_mask(&groups_per[i]);
        for d in 0..BAND_DIM {
            zt.push(z[t * BAND_DIM + d]);
            mn.push(m[d] * z[(t + 1) * BAND_DIM + d]);
            mk.push(m[d]);
        }
        for d in 0..POSE_DIM {
            yy.push(gt[(t + 1) * POSE_DIM + d]);
        }
    }
    let t2 = |v: Vec<f32>, c: usize| {
        Tensor::<Bk, 2>::from_data(TensorData::new(v, [n, c]).convert::<f32>(), device)
    };
    (t2(zt, BAND_DIM), t2(mn, BAND_DIM), t2(mk, BAND_DIM), t2(yy, POSE_DIM))
}

fn main() {
    let fx = concat!(env!("CARGO_MANIFEST_DIR"), "/../fixtures");
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
    let train_pairs = subsample(&build_pairs(&splits[0], &masks, &w2f), 60_000, 42);
    let val_pairs = subsample(&build_pairs(&splits[1], &masks, &w2f), 4_000, 7);
    // IDENTICAL evaluation set to rung 2.
    let test_pairs = subsample(&build_pairs(&splits[2], &masks, &w2f), 20_000, 999);

    let device = NdArrayDevice::default();
    // Seed Burn's global RNG: LinearConfig::init draws from it, so without this the
    // run is not reproducible even though masks and shuffles use a seeded LCG.
    let init_seed: u64 = std::env::var("RUNG3_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(2026);
    <B as burn::tensor::backend::Backend>::seed(&device, init_seed);
    let mut model: HeteroModel<B> = HeteroModel::new(&device, 512, 256);
    let mut opt = AdamConfig::new().init();
    let mut rng = Rng(2026);

    let epochs: usize = std::env::var("RUNG3_EPOCHS").ok().and_then(|v| v.parse().ok()).unwrap_or(400);
    let batch = 512usize;
    let lr = 1e-3;
    let mut history = Vec::new();
    let mut best_val = f64::INFINITY;
    let mut best_epoch = 0usize;
    let mut best_model: Option<HeteroModel<B>> = None;

    for ep in 0..epochs {
        // fresh shuffle each epoch
        let order = subsample(&train_pairs, train_pairs.len(), 100 + ep as u64);
        let mut running = 0.0f64;
        let mut nb = 0usize;
        for chunk in order.chunks(batch) {
            // Random mask per sample per epoch: the model must learn p(. | z_t, S)
            // for arbitrary S, not one fixed probing pattern.
            let groups: Vec<Vec<usize>> = chunk
                .iter()
                .map(|_| {
                    let k = rng.below(NUM_PROBE_GROUPS / 2 + 1);
                    random_groups(&mut rng, k)
                })
                .collect();
            let (zt, mn, mk, yy) = make_batch::<B>(z, gt, chunk, &groups, &device);
            let (mu, lv) = model.forward(zt, mn, mk);
            let loss = hetero_nll(mu, lv, yy).mean();
            running += loss.clone().into_scalar() as f64;
            nb += 1;
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = opt.step(lr, model, grads);
        }
        let val = eval_nll(&model, z, gt, &val_pairs, &[], &device);
        history.push(serde_json::json!({ "epoch": ep, "train_nll": running / nb as f64, "val_nll": val }));
        if ep % 10 == 0 || ep + 1 == epochs {
            eprintln!("epoch {ep}: train {:.3} val {:.3} ({:.0?})", running / nb as f64, val, t0.elapsed());
        }
        // Early stopping on val NLL, patience 40 epochs. The first run stopped at a
        // fixed 12 epochs while the loss was still descending steeply -- undertrained,
        // not converged.
        if val + 1e-6 < best_val {
            best_val = val;
            best_epoch = ep;
            best_model = Some(model.clone());
        } else if ep - best_epoch >= 40 {
            eprintln!("early stop at epoch {ep} (best {best_epoch}, val {best_val:.3})");
            break;
        }
    }
    let model = best_model.unwrap_or(model);

    let inf = model.valid();

    // ---- evaluation, reusing the rung-2 protocol unchanged -------------------
    let no_probe = evaluate(&inf, z, gt, &test_pairs, &vec![vec![]; test_pairs.len()], &device);
    let bar = mean_pose_bar(gt, &test_pairs);

    let mut policies = Vec::new();
    for &k in &[1usize, 3, 6, 9] {
        // EIG-greedy, now genuinely PER-SAMPLE because sigma depends on z_t.
        let eig_sets = greedy_eig_per_sample(&inf, z, &test_pairs, k, &device);
        let g = evaluate(&inf, z, gt, &test_pairs, &eig_sets, &device);

        let mut r = Rng(0xBEEF + k as u64);
        let rnd_sets: Vec<Vec<usize>> = test_pairs.iter().map(|_| random_groups(&mut r, k)).collect();
        let rd = evaluate(&inf, z, gt, &test_pairs, &rnd_sets, &device);

        let fixed: Vec<usize> = (0..k).map(|i| i * NUM_PROBE_GROUPS / k).collect();
        let fx_e = evaluate(&inf, z, gt, &test_pairs, &vec![fixed.clone(); test_pairs.len()], &device);

        let (d, se, t) = paired(&g.per_sample, &rd.per_sample);
        let (dgf, segf, tgf) = paired(&g.per_sample, &fx_e.per_sample);
        // The comparison the charter actually turns on: does probing beat NOT probing?
        let (dn_g, sen_g, tn_g) = paired(&g.per_sample, &no_probe.per_sample);
        // Cluster-robust intervals. Adjacent windows within a recording file are
        // correlated at 0.9970 (pose persistence), so per-sample i.i.d. SEs understate
        // uncertainty. These resample whole FILES. Applied to the NEGATIVE findings with
        // the same rigour as to the positive one -- an over-claimed negative is the same
        // error with the sign flipped.
        let (bd_gn, blo_gn, bhi_gn) =
            block_bootstrap(&g.per_sample, &no_probe.per_sample, &test_pairs, &w2f, 2000, 21);
        let (bd_gr, blo_gr, bhi_gr) =
            block_bootstrap(&g.per_sample, &rd.per_sample, &test_pairs, &w2f, 2000, 22);
        let (bd_rn, blo_rn, bhi_rn) =
            block_bootstrap(&rd.per_sample, &no_probe.per_sample, &test_pairs, &w2f, 2000, 23);
        let (dn_r, sen_r, tn_r) = paired(&rd.per_sample, &no_probe.per_sample);
        let (dn_f, sen_f, tn_f) = paired(&fx_e.per_sample, &no_probe.per_sample);
        // How often does the per-sample EIG set actually differ from the modal one?
        let mut counts = std::collections::HashMap::new();
        for s in &eig_sets {
            *counts.entry(format!("{s:?}")).or_insert(0usize) += 1;
        }
        let modal = counts.values().copied().max().unwrap_or(0);
        policies.push(serde_json::json!({
            "budget_k": k,
            "pck@20": { "eig_greedy": g.pck20, "random": rd.pck20, "fixed": fx_e.pck20 },
            "pose_nll": { "eig_greedy": g.nll, "random": rd.nll, "fixed": fx_e.nll },
            "paired_greedy_minus_random": { "delta": d, "se": se, "t": t, "beats_by_2se": t > 2.0,
                "note": "per-sample i.i.d., NOT cluster-corrected -- see block_bootstrap_file_level" },
            "block_bootstrap_file_level": {
                "note": "2000 reps resampling whole recording files; a CI excluding 0 is significant",
                "greedy_minus_no_probe": { "delta": bd_gn, "ci95": [blo_gn, bhi_gn], "significant": bhi_gn < 0.0 || blo_gn > 0.0 },
                "greedy_minus_random":   { "delta": bd_gr, "ci95": [blo_gr, bhi_gr], "significant": bhi_gr < 0.0 || blo_gr > 0.0 },
                "random_minus_no_probe": { "delta": bd_rn, "ci95": [blo_rn, bhi_rn], "significant": bhi_rn < 0.0 || blo_rn > 0.0 }
            },
            "paired_greedy_minus_fixed": { "delta": dgf, "se": segf, "t": tgf },
            "vs_no_probe": {
                "note": "positive t means probing BEATS not probing at all",
                "greedy_minus_noprobe": { "delta": dn_g, "se": sen_g, "t": tn_g },
                "random_minus_noprobe": { "delta": dn_r, "se": sen_r, "t": tn_r },
                "fixed_minus_noprobe":  { "delta": dn_f, "se": sen_f, "t": tn_f }
            },
            "adaptivity": {
                "distinct_probe_sets": counts.len(),
                "modal_set_fraction": modal as f64 / eig_sets.len() as f64,
                "note": "rung 2 had exactly 1 distinct set by construction (homoscedastic)"
            }
        }));
    }

    // ---- discriminating experiment: target-peeking bound AT RUNG 3 ----------
    // Open question this resolves: is the marginal information ABSENT, or merely
    // unreachable by the EIG criterion / this architecture? An oracle that selects
    // probes using the TRUE y_{t+1} upper-bounds every policy. If even IT cannot beat
    // not probing, the information is not there for this observation definition.
    // Subsampled to 5,000 pairs for cost; no-probe is rescored on the SAME pairs.
    let oracle_pairs = subsample(&test_pairs, 5_000, 999);
    let oracle_noprobe = evaluate(&inf, z, gt, &oracle_pairs, &vec![vec![]; oracle_pairs.len()], &device);
    let mut oracle_rows = Vec::new();
    for &k in &[3usize, 9] {
        let sets = oracle_probe_sets(&inf, z, gt, &oracle_pairs, k, &device);
        let o = evaluate(&inf, z, gt, &oracle_pairs, &sets, &device);
        let (d, se, t) = paired(&o.per_sample, &oracle_noprobe.per_sample);
        oracle_rows.push(serde_json::json!({
            "budget_k": k,
            "oracle_pck@20": o.pck20,
            "no_probe_pck@20": oracle_noprobe.pck20,
            "oracle_minus_noprobe": { "delta": d, "se": se, "t": t, "beats_by_2se": t > 2.0 }
        }));
    }
    // Control + cluster-robust intervals for the bound.
    let mut control_rows = Vec::new();
    for &k in &[3usize, 9] {
        // Greedy at budget k evaluates sum_{j=0}^{k-1} (27-j) candidates per sample.
        let c: usize = (0..k).map(|j| NUM_PROBE_GROUPS - j).sum();
        let sets = oracle_probe_sets(&inf, z, gt, &oracle_pairs, k, &device);
        let orc = evaluate(&inf, z, gt, &oracle_pairs, &sets, &device);
        let (boc_pck, boc_per_sample) =
            best_of_c_random_peeking(&inf, z, gt, &oracle_pairs, k, c, 0xC0FFEE + k as u64, &device);
        let (d_on, lo_on, hi_on) =
            block_bootstrap(&orc.per_sample, &oracle_noprobe.per_sample, &oracle_pairs, &w2f, 2000, 11);
        let (d_ob, lo_ob, hi_ob) =
            block_bootstrap(&orc.per_sample, &boc_per_sample, &oracle_pairs, &w2f, 2000, 12);
        let (d_bn, lo_bn, hi_bn) =
            block_bootstrap(&boc_per_sample, &oracle_noprobe.per_sample, &oracle_pairs, &w2f, 2000, 13);
        control_rows.push(serde_json::json!({
            "budget_k": k,
            "candidates_C": c,
            "oracle_pck@20": orc.pck20,
            "best_of_C_random_peeking_pck@20": boc_pck,
            "no_probe_pck@20": oracle_noprobe.pck20,
            "oracle_minus_noprobe":   { "delta": d_on, "ci95_file_block": [lo_on, hi_on] },
            "oracle_minus_bestOfC":   { "delta": d_ob, "ci95_file_block": [lo_ob, hi_ob],
                                        "significant": lo_ob > 0.0 },
            "bestOfC_minus_noprobe":  { "delta": d_bn, "ci95_file_block": [lo_bn, hi_bn] }
        }));
    }
    let information_survives_control = control_rows
        .iter()
        .any(|r| r["oracle_minus_bestOfC"]["significant"].as_bool().unwrap_or(false));

    let oracle_beats_noprobe = oracle_rows
        .iter()
        .any(|r| r["oracle_minus_noprobe"]["beats_by_2se"].as_bool().unwrap_or(false));

    let cov_ok = no_probe
        .coverage
        .as_array()
        .unwrap()
        .iter()
        .all(|c| c["error_pp"].as_f64().unwrap().abs() <= 3.0);
    let nll_ok = no_probe.nll < RUNG2_NLL;
    let eig_ok = policies
        .iter()
        .any(|p| p["paired_greedy_minus_random"]["beats_by_2se"].as_bool().unwrap_or(false));

    let any_probe_beats_noprobe = policies.iter().any(|p| {
        p["vs_no_probe"]["greedy_minus_noprobe"]["t"].as_f64().unwrap_or(0.0) > 2.0
    });
    let verdict = if nll_ok && cov_ok && eig_ok {
        "PASS -- rung 3 helped on all three pre-registered criteria"
    } else {
        "RUNG 3 DID NOT HELP -- at least one pre-registered criterion was missed"
    };

    let report = serde_json::json!({
        "milestone": "Rung 3 -- neural heteroscedastic head. MEASURED by harness/src/bin/rung3.rs, this run.",
        "backend": "burn-ndarray (CPU), by design -- see STATE.md rung-3 pre-registration",
        "init_seed": init_seed,
        "training": { "epochs": epochs, "batch": batch, "lr": lr, "train_pairs": train_pairs.len(),
                      "history": history },
        "no_probe": { "pose_nll": no_probe.nll, "pck@20": no_probe.pck20,
                      "mpjpe": no_probe.mpjpe, "coverage": no_probe.coverage },
        "honesty_bar_same_pairs": { "pck@20": bar.0, "mpjpe": bar.1 },
        "beats_bar": no_probe.pck20 > bar.0,
        "degeneracy_guard": check_constant_pose(&no_probe.preds, test_pairs.len()),
        "policies": policies,
        "preregistered_criteria": {
            "1_beats_rung2_nll": { "rung2": RUNG2_NLL, "rung3": no_probe.nll, "pass": nll_ok },
            "2_coverage_within_3pp_all_levels": cov_ok,
            "3_eig_beats_random_by_2se": eig_ok,
            "rung2_best_pck_for_reference": RUNG2_BEST_PCK
        },
        "selection_inflation_control": {
            "question": "is the target-peeking bound real information, or the expected \
                         maximum of C noisy peeked scores?",
            "method": "best-of-C RANDOM sets, also chosen by peeking, C matched to the \
                       oracle's greedy candidate count; file-level block bootstrap (2000 \
                       reps, whole recording files resampled) for all intervals",
            "rows": control_rows,
            "information_survives_control": information_survives_control,
            "verdict": if information_survives_control {
                "Oracle beats best-of-C-random at matched search size: the margin is \
                 exploitable information, and the honest bound is oracle MINUS best-of-C."
            } else {
                "Oracle does NOT beat best-of-C-random at matched search size: the \
                 apparent gain is SELECTION NOISE, not information. Any claim that \
                 'the information is real' is WITHDRAWN."
            }
        },
        "target_peeking_bound_at_rung3": {
            "pairs": oracle_pairs.len(),
            "rows": oracle_rows,
            "oracle_beats_no_probe_by_2se": oracle_beats_noprobe,
            "interpretation": if oracle_beats_noprobe {
                "Information EXISTS but the EIG criterion / architecture cannot reach it."
            } else {
                "Even a target-peeking oracle cannot beat not probing: the marginal \
                 information in time-averaged z_{t+1} bands is ABSENT, not merely \
                 unreachable. The primitive needs a different observation definition."
            }
        },
        "does_probing_help_at_all": {
            "any_budget_where_eig_greedy_beats_no_probe_by_2se": any_probe_beats_noprobe,
            "no_probe_pck@20": no_probe.pck20,
            "note": "If false, EIG is selecting the LEAST HARMFUL probes rather than \
                     helpful ones -- revealing time-averaged z_{t+1} bands adds no \
                     measurable information about y_{t+1} beyond z_t."
        },
        "verdict": verdict
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    eprintln!("{verdict} ({:.0?})", t0.elapsed());
}

struct Ev {
    nll: f64,
    pck20: f64,
    mpjpe: f64,
    coverage: serde_json::Value,
    per_sample: Vec<f64>,
    preds: Vec<f32>,
}

fn paired(a: &[f64], b: &[f64]) -> (f64, f64, f64) {
    let n = a.len();
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let m = d.iter().sum::<f64>() / n as f64;
    let var = d.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    let se = (var / n as f64).sqrt();
    (m, se, if se > 0.0 { m / se } else { 0.0 })
}

fn eval_nll(
    model: &HeteroModel<B>,
    z: &[f32],
    gt: &[f32],
    pairs: &[usize],
    groups: &[usize],
    device: &NdArrayDevice,
) -> f64 {
    let g = vec![groups.to_vec(); pairs.len()];
    let (zt, mn, mk, yy) = make_batch::<B>(z, gt, pairs, &g, device);
    let (mu, lv) = model.forward(zt, mn, mk);
    hetero_nll(mu, lv, yy).mean().into_scalar() as f64
}

fn evaluate(
    model: &HeteroModel<BI>,
    z: &[f32],
    gt: &[f32],
    pairs: &[usize],
    groups: &[Vec<usize>],
    device: &NdArrayDevice,
) -> Ev {
    let mut preds = Vec::with_capacity(pairs.len() * POSE_DIM);
    let mut truths = Vec::with_capacity(pairs.len() * POSE_DIM);
    let mut nll = 0.0f64;
    let mut inside = [0usize; COVERAGE_LEVELS.len()];

    for (chunk_i, chunk) in pairs.chunks(2048).enumerate() {
        let gs: Vec<Vec<usize>> = (0..chunk.len())
            .map(|i| groups[chunk_i * 2048 + i].clone())
            .collect();
        let (zt, mn, mk, yy) = make_batch::<BI>(z, gt, chunk, &gs, device);
        let (mu, lv) = model.forward(zt, mn, mk);
        nll += hetero_nll(mu.clone(), lv.clone(), yy.clone()).sum().into_scalar() as f64;
        let mud: Vec<f32> = mu.into_data().to_vec().unwrap();
        let lvd: Vec<f32> = lv.into_data().to_vec().unwrap();
        let yd: Vec<f32> = yy.into_data().to_vec().unwrap();
        for i in 0..chunk.len() {
            for d in 0..POSE_DIM {
                let j = i * POSE_DIM + d;
                let sd = (lvd[j] as f64 * 0.5).exp();
                for (li, (_, zs)) in COVERAGE_LEVELS.iter().enumerate() {
                    if ((yd[j] - mud[j]) as f64).abs() <= zs * sd {
                        inside[li] += 1;
                    }
                }
            }
        }
        preds.extend_from_slice(&mud);
        truths.extend_from_slice(&yd);
    }

    let s = score(&preds, &truths, pairs.len());
    let per_sample: Vec<f64> = (0..pairs.len())
        .map(|i| {
            let b = i * POSE_DIM;
            score(&preds[b..b + POSE_DIM], &truths[b..b + POSE_DIM], 1).pck_at(0.2)
        })
        .collect();
    let denom = (pairs.len() * POSE_DIM) as f64;
    Ev {
        nll: nll / pairs.len() as f64,
        pck20: s.pck_at(0.2),
        mpjpe: s.mpjpe,
        per_sample,
        preds,
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

/// Per-sample greedy EIG. Closed-form and peek-free: the variance head does not see
/// revealed values, so extending the mask gives the post-probe entropy directly.
fn greedy_eig_per_sample(
    model: &HeteroModel<BI>,
    z: &[f32],
    pairs: &[usize],
    k: usize,
    device: &NdArrayDevice,
) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new(); pairs.len()];
    for _step in 0..k {
        // Score every candidate group for every sample in one batched pass per group.
        let mut best: Vec<(f64, usize)> = vec![(f64::NEG_INFINITY, usize::MAX); pairs.len()];
        for cand in 0..NUM_PROBE_GROUPS {
            let mut zt = Vec::with_capacity(pairs.len() * BAND_DIM);
            let mut mk = Vec::with_capacity(pairs.len() * BAND_DIM);
            for (i, &t) in pairs.iter().enumerate() {
                let mut set = out[i].clone();
                if !set.contains(&cand) {
                    set.push(cand);
                }
                let m = expand_mask(&set);
                for d in 0..BAND_DIM {
                    zt.push(z[t * BAND_DIM + d]);
                    mk.push(m[d]);
                }
            }
            let n = pairs.len();
            let zt = Tensor::<BI, 2>::from_data(TensorData::new(zt, [n, BAND_DIM]), device);
            let mk = Tensor::<BI, 2>::from_data(TensorData::new(mk, [n, BAND_DIM]), device);
            let lv: Vec<f32> = model.log_var(zt, mk).into_data().to_vec().unwrap();
            for i in 0..n {
                if out[i].contains(&cand) {
                    continue;
                }
                // Lower total log-variance == more information gained.
                let s: f64 = -(0..POSE_DIM).map(|d| lv[i * POSE_DIM + d] as f64).sum::<f64>();
                if s > best[i].0 {
                    best[i] = (s, cand);
                }
            }
        }
        for i in 0..pairs.len() {
            if best[i].1 != usize::MAX {
                out[i].push(best[i].1);
            }
        }
    }
    out
}

fn mean_pose_bar(gt: &[f32], pairs: &[usize]) -> (f64, f64) {
    let n = pairs.len();
    let mut mu = vec![0.0f64; POSE_DIM];
    for &t in pairs {
        for i in 0..POSE_DIM {
            mu[i] += gt[(t + 1) * POSE_DIM + i] as f64;
        }
    }
    let bar: Vec<f32> = mu.iter().map(|v| (v / n as f64) as f32).collect();
    let truths: Vec<f32> = pairs
        .iter()
        .flat_map(|&t| (0..POSE_DIM).map(move |i| gt[(t + 1) * POSE_DIM + i]))
        .collect();
    let s = score(&bar.repeat(n), &truths, n);
    (s.pck_at(0.2), s.mpjpe)
}
