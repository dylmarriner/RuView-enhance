//! Rung 4 driver: value-aware variance head + observation head, Monte-Carlo EIG.
//!
//! Pass condition, pre-registered in `../STATE.md` before this file existed:
//!   1. MC-EIG-greedy beats NO-PROBE by >2 SE at some budget, AND
//!   2. beats RANDOM at that budget by >2 SE, with a consistent sign across >=3 seeds.
//! Anything less -> "rung 4 did not close the gap", with the fraction of the oracle
//! gap it did close reported as a measurement.

use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::backend::Autodiff;
use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Tensor, TensorData};
use std::time::Instant;

use wiflow_harness::bands::{group_dims, BAND_DIM, NUM_PROBE_GROUPS};
use wiflow_harness::gaussian::POSE_DIM;
use wiflow_harness::guards::{check_constant_pose, CorruptionMasks};
use wiflow_harness::hetero::expand_mask;
use wiflow_harness::hetero2::{diag_nll, masked_obs_nll, HeteroV2};
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
const EVAL_PAIRS: usize = 5_000;
const MC_SAMPLES: usize = 4;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() >> 11) as usize % n
    }
    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) as f64 / (1u64 << 53) as f64).clamp(1e-12, 1.0 - 1e-12)
    }
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.unit(), self.unit());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
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

fn paired(a: &[f64], b: &[f64]) -> (f64, f64, f64) {
    let n = a.len();
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let m = d.iter().sum::<f64>() / n as f64;
    let var = d.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    let se = (var / n as f64).sqrt();
    (m, se, if se > 0.0 { m / se } else { 0.0 })
}

/// Build raw input planes. `override_vals` optionally replaces the revealed values
/// (used to inject Monte-Carlo samples instead of true future values).
fn planes(
    z: &[f32],
    pairs: &[usize],
    groups: &[Vec<usize>],
    override_vals: Option<&[f32]>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = pairs.len();
    let mut zt = Vec::with_capacity(n * BAND_DIM);
    let mut mn = Vec::with_capacity(n * BAND_DIM);
    let mut mk = Vec::with_capacity(n * BAND_DIM);
    for (i, &t) in pairs.iter().enumerate() {
        let m = expand_mask(&groups[i]);
        for d in 0..BAND_DIM {
            zt.push(z[t * BAND_DIM + d]);
            let v = match override_vals {
                Some(o) => o[i * BAND_DIM + d],
                None => z[(t + 1) * BAND_DIM + d],
            };
            mn.push(m[d] * v);
            mk.push(m[d]);
        }
    }
    (zt, mn, mk)
}

fn to_t<Bk: burn::tensor::backend::Backend>(v: Vec<f32>, n: usize, c: usize, d: &Bk::Device) -> Tensor<Bk, 2> {
    Tensor::<Bk, 2>::from_data(TensorData::new(v, [n, c]), d)
}

fn main() {
    let fx = concat!(env!("CARGO_MANIFEST_DIR"), "/../fixtures");
    let load = |f: &str| Npy::open(format!("{fx}/{f}")).unwrap_or_else(|e| panic!("{e}"));
    let t0 = Instant::now();
    let seed: u64 = std::env::var("RUNG4_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(2026);

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
    let eval_pairs = subsample(&build_pairs(&splits[2], &masks, &w2f), EVAL_PAIRS, 999);

    let device = NdArrayDevice::default();
    <B as burn::tensor::backend::Backend>::seed(&device, seed);
    let mut model: HeteroV2<B> = HeteroV2::new(&device, 512);
    let mut opt = AdamConfig::new().init();
    let mut rng = Rng(seed ^ 0xA5A5);

    let (epochs, batch, lr) = (400usize, 512usize, 1e-3);
    let mut best_val = f64::INFINITY;
    let mut best_epoch = 0usize;
    let mut best: Option<HeteroV2<B>> = None;
    let mut history = Vec::new();

    for ep in 0..epochs {
        let order = subsample(&train_pairs, train_pairs.len(), 100 + ep as u64);
        let (mut run, mut nb) = (0.0f64, 0usize);
        for chunk in order.chunks(batch) {
            let groups: Vec<Vec<usize>> = chunk
                .iter()
                .map(|_| {
                    let k = rng.below(NUM_PROBE_GROUPS / 2 + 1);
                    random_groups(&mut rng, k)
                })
                .collect();
            let n = chunk.len();
            let (zt, mn, mk) = planes(z, chunk, &groups, None);
            let ztt = to_t::<B>(zt, n, BAND_DIM, &device);
            let mnt = to_t::<B>(mn, n, BAND_DIM, &device);
            let mkt = to_t::<B>(mk.clone(), n, BAND_DIM, &device);
            let yv: Vec<f32> = chunk
                .iter()
                .flat_map(|&t| (0..POSE_DIM).map(move |d| gt[(t + 1) * POSE_DIM + d]))
                .collect();
            let zn: Vec<f32> = chunk
                .iter()
                .flat_map(|&t| (0..BAND_DIM).map(move |d| z[(t + 1) * BAND_DIM + d]))
                .collect();
            let yt = to_t::<B>(yv, n, POSE_DIM, &device);
            let znt = to_t::<B>(zn, n, BAND_DIM, &device);

            let (pmu, plv, omu, olv) = model.forward(ztt, mnt, mkt.clone());
            // Joint objective: pose NLL + observation NLL on the UNREVEALED dims.
            let loss = diag_nll(pmu, plv, yt).mean() + masked_obs_nll(omu, olv, znt, mkt).mean();
            run += loss.clone().into_scalar() as f64;
            nb += 1;
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = opt.step(lr, model, grads);
        }
        let val = val_pose_nll(&model, z, gt, &val_pairs, &device);
        history.push(serde_json::json!({ "epoch": ep, "train": run / nb as f64, "val_pose_nll": val }));
        if ep % 20 == 0 {
            eprintln!("epoch {ep}: train {:.3} val {:.3} ({:.0?})", run / nb as f64, val, t0.elapsed());
        }
        if val + 1e-6 < best_val {
            best_val = val;
            best_epoch = ep;
            best = Some(model.clone());
        } else if ep - best_epoch >= 40 {
            eprintln!("early stop at {ep} (best {best_epoch}, val {best_val:.3})");
            break;
        }
    }
    let inf = best.unwrap_or(model).valid();

    // ---- policies -----------------------------------------------------------
    let no_probe = eval(&inf, z, gt, &eval_pairs, &vec![vec![]; eval_pairs.len()], &device);
    let mut rows = Vec::new();
    let mut any_beats_noprobe = false;
    let mut any_beats_random = false;

    for &k in &[3usize, 9] {
        let mc = mc_eig_greedy(&inf, z, &eval_pairs, k, MC_SAMPLES, seed ^ 0x5EED, &device);
        let g = eval(&inf, z, gt, &eval_pairs, &mc, &device);

        let mut r = Rng(seed ^ 0xBEEF ^ k as u64);
        let rnd: Vec<Vec<usize>> = eval_pairs.iter().map(|_| random_groups(&mut r, k)).collect();
        let rd = eval(&inf, z, gt, &eval_pairs, &rnd, &device);

        let oracle_sets = oracle_probe_sets(&inf, z, gt, &eval_pairs, k, &device);
        let orc = eval(&inf, z, gt, &eval_pairs, &oracle_sets, &device);

        let (dn, sen, tn) = paired(&g.per_sample, &no_probe.per_sample);
        let (dr, ser, tr) = paired(&g.per_sample, &rd.per_sample);
        let (do_, seo, to_) = paired(&orc.per_sample, &no_probe.per_sample);
        if tn > 2.0 {
            any_beats_noprobe = true;
        }
        if tr > 2.0 {
            any_beats_random = true;
        }
        // How much of the oracle's headroom did MC-EIG capture?
        let closed = if do_.abs() > 1e-12 { dn / do_ } else { 0.0 };

        // Overconfidence diagnostic. Hypothesis: EIG picks sets the model *predicts*
        // will shrink entropy; under a miscalibrated head that selects for
        // overconfidence, not information. Test: if EIG-chosen sets have LOWER
        // predicted variance but HIGHER actual error than random sets, confirmed.
        let oc_eig = confidence_vs_error(&inf, z, gt, &eval_pairs, &mc, &device);
        let oc_rnd = confidence_vs_error(&inf, z, gt, &eval_pairs, &rnd, &device);

        let mut counts = std::collections::HashMap::new();
        for s in &mc {
            *counts.entry(format!("{s:?}")).or_insert(0usize) += 1;
        }
        rows.push(serde_json::json!({
            "budget_k": k,
            "pck@20": { "mc_eig": g.pck20, "random": rd.pck20,
                        "no_probe": no_probe.pck20, "target_peeking_bound": orc.pck20 },
            "mc_eig_minus_no_probe": { "delta": dn, "se": sen, "t": tn },
            "mc_eig_minus_random":   { "delta": dr, "se": ser, "t": tr },
            "bound_minus_no_probe":  { "delta": do_, "se": seo, "t": to_ },
            "fraction_of_oracle_gap_closed": closed,
            "distinct_probe_sets": counts.len(),
            "overconfidence_diagnostic": {
                "hypothesis": "EIG selects sets where the model is overconfident, not \
                               where it is informative. Confirmed if mc_eig has LOWER \
                               mean predicted variance but HIGHER mean actual error.",
                "mc_eig":  { "mean_predicted_var": oc_eig.0, "mean_actual_sq_err": oc_eig.1 },
                "random":  { "mean_predicted_var": oc_rnd.0, "mean_actual_sq_err": oc_rnd.1 },
                "confirmed": oc_eig.0 < oc_rnd.0 && oc_eig.1 > oc_rnd.1
            }
        }));
    }

    let verdict = if any_beats_noprobe && any_beats_random {
        "PASS (single seed) -- needs >=3-seed sign consistency before any claim"
    } else {
        "RUNG 4 DID NOT CLOSE THE GAP"
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "milestone": "Rung 4 -- value-aware variance + observation head, Monte-Carlo EIG",
            "measured_by": "harness/src/bin/rung4.rs, this run",
            "seed": seed,
            "mc_samples": MC_SAMPLES,
            "eval_pairs": eval_pairs.len(),
            "peek_free": "candidate values are SAMPLED from the observation head; only the \
                          target-peeking bound reads the future, and it is unachievable",
            "training": { "best_epoch": best_epoch, "best_val_pose_nll": best_val, "history": history },
            "no_probe": { "pose_nll": no_probe.nll, "pck@20": no_probe.pck20,
                          "mpjpe": no_probe.mpjpe, "coverage": no_probe.coverage },
            "degeneracy_guard": check_constant_pose(&no_probe.preds, eval_pairs.len()),
            "policies": rows,
            "preregistered": {
                "1_beats_no_probe_by_2se": any_beats_noprobe,
                "2_beats_random_by_2se": any_beats_random,
                "3_sign_consistent_across_3_seeds": "run RUNG4_SEED=... and compare"
            },
            "verdict": verdict
        }))
        .unwrap()
    );
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

fn val_pose_nll(m: &HeteroV2<B>, z: &[f32], gt: &[f32], pairs: &[usize], d: &NdArrayDevice) -> f64 {
    let n = pairs.len();
    let g = vec![vec![]; n];
    let (zt, mn, mk) = planes(z, pairs, &g, None);
    let y: Vec<f32> = pairs.iter().flat_map(|&t| (0..POSE_DIM).map(move |i| gt[(t + 1) * POSE_DIM + i])).collect();
    let (pmu, plv) = m.pose(to_t::<B>(zt, n, BAND_DIM, d), to_t::<B>(mn, n, BAND_DIM, d), to_t::<B>(mk, n, BAND_DIM, d));
    diag_nll(pmu, plv, to_t::<B>(y, n, POSE_DIM, d)).mean().into_scalar() as f64
}

fn eval(m: &HeteroV2<BI>, z: &[f32], gt: &[f32], pairs: &[usize], groups: &[Vec<usize>], d: &NdArrayDevice) -> Ev {
    let n = pairs.len();
    let (zt, mn, mk) = planes(z, pairs, groups, None);
    let y: Vec<f32> = pairs.iter().flat_map(|&t| (0..POSE_DIM).map(move |i| gt[(t + 1) * POSE_DIM + i])).collect();
    let (pmu, plv) = m.pose(to_t::<BI>(zt, n, BAND_DIM, d), to_t::<BI>(mn, n, BAND_DIM, d), to_t::<BI>(mk, n, BAND_DIM, d));
    let nll = diag_nll(pmu.clone(), plv.clone(), to_t::<BI>(y.clone(), n, POSE_DIM, d)).mean().into_scalar() as f64;
    let mud: Vec<f32> = pmu.into_data().to_vec().unwrap();
    let lvd: Vec<f32> = plv.into_data().to_vec().unwrap();

    let mut inside = [0usize; COVERAGE_LEVELS.len()];
    for j in 0..n * POSE_DIM {
        let sd = (lvd[j] as f64 * 0.5).exp();
        for (li, (_, zs)) in COVERAGE_LEVELS.iter().enumerate() {
            if ((y[j] - mud[j]) as f64).abs() <= zs * sd {
                inside[li] += 1;
            }
        }
    }
    let s = score(&mud, &y, n);
    let per_sample: Vec<f64> = (0..n)
        .map(|i| score(&mud[i * POSE_DIM..(i + 1) * POSE_DIM], &y[i * POSE_DIM..(i + 1) * POSE_DIM], 1).pck_at(0.2))
        .collect();
    let denom = (n * POSE_DIM) as f64;
    Ev {
        nll,
        pck20: s.pck_at(0.2),
        mpjpe: s.mpjpe,
        per_sample,
        preds: mud,
        coverage: serde_json::json!(COVERAGE_LEVELS
            .iter()
            .enumerate()
            .map(|(i, (lvl, _))| serde_json::json!({
                "nominal": lvl, "empirical": inside[i] as f64 / denom,
                "error_pp": (inside[i] as f64 / denom - lvl) * 100.0 }))
            .collect::<Vec<_>>()),
    }
}

/// Monte-Carlo EIG greedy selection. Peek-free: candidate values for the extended
/// mask are SAMPLED from the observation head, never read from `z[t+1]`.
fn mc_eig_greedy(
    m: &HeteroV2<BI>,
    z: &[f32],
    pairs: &[usize],
    k: usize,
    samples: usize,
    seed: u64,
    d: &NdArrayDevice,
) -> Vec<Vec<usize>> {
    let n = pairs.len();
    let mut rng = Rng(seed);
    let mut out = vec![Vec::new(); n];

    for _step in 0..k {
        // Current entropy and the sampling distribution, under the current mask.
        let (zt0, mn0, mk0) = planes(z, pairs, &out, None);
        let (_, plv0) = m.pose(
            to_t::<BI>(zt0.clone(), n, BAND_DIM, d),
            to_t::<BI>(mn0.clone(), n, BAND_DIM, d),
            to_t::<BI>(mk0.clone(), n, BAND_DIM, d),
        );
        let h_before: Vec<f64> = {
            let lv: Vec<f32> = plv0.into_data().to_vec().unwrap();
            (0..n).map(|i| 0.5 * (0..POSE_DIM).map(|q| lv[i * POSE_DIM + q] as f64).sum::<f64>()).collect()
        };
        let (omu, olv) = m.observation(
            to_t::<BI>(zt0.clone(), n, BAND_DIM, d),
            to_t::<BI>(mn0.clone(), n, BAND_DIM, d),
            to_t::<BI>(mk0, n, BAND_DIM, d),
        );
        let omud: Vec<f32> = omu.into_data().to_vec().unwrap();
        let olvd: Vec<f32> = olv.into_data().to_vec().unwrap();

        let mut best: Vec<(f64, usize)> = vec![(f64::NEG_INFINITY, usize::MAX); n];
        for cand in 0..NUM_PROBE_GROUPS {
            let gs: Vec<Vec<usize>> = (0..n)
                .map(|i| {
                    let mut s = out[i].clone();
                    if !s.contains(&cand) {
                        s.push(cand);
                    }
                    s
                })
                .collect();
            let mut h_after = vec![0.0f64; n];
            for _m in 0..samples {
                // Draw candidate values for the newly revealed dims from the model.
                let mut vals = vec![0.0f32; n * BAND_DIM];
                for i in 0..n {
                    for q in 0..BAND_DIM {
                        vals[i * BAND_DIM + q] = z[(pairs[i] + 1) * BAND_DIM + q];
                    }
                    for q in group_dims(cand) {
                        let j = i * BAND_DIM + q;
                        let sd = (olvd[j] as f64 * 0.5).exp();
                        vals[j] = (omud[j] as f64 + sd * rng.normal()) as f32;
                    }
                }
                let (zt, mn, mk) = planes(z, pairs, &gs, Some(&vals));
                let (_, plv) = m.pose(
                    to_t::<BI>(zt, n, BAND_DIM, d),
                    to_t::<BI>(mn, n, BAND_DIM, d),
                    to_t::<BI>(mk, n, BAND_DIM, d),
                );
                let lv: Vec<f32> = plv.into_data().to_vec().unwrap();
                for i in 0..n {
                    h_after[i] += 0.5 * (0..POSE_DIM).map(|q| lv[i * POSE_DIM + q] as f64).sum::<f64>();
                }
            }
            for i in 0..n {
                if out[i].contains(&cand) {
                    continue;
                }
                let gain = h_before[i] - h_after[i] / samples as f64;
                if gain > best[i].0 {
                    best[i] = (gain, cand);
                }
            }
        }
        for i in 0..n {
            if best[i].1 != usize::MAX {
                out[i].push(best[i].1);
            }
        }
    }
    out
}

/// Mean predicted pose variance and mean actual squared error under a given policy.
fn confidence_vs_error(
    m: &HeteroV2<BI>,
    z: &[f32],
    gt: &[f32],
    pairs: &[usize],
    groups: &[Vec<usize>],
    d: &NdArrayDevice,
) -> (f64, f64) {
    let n = pairs.len();
    let (zt, mn, mk) = planes(z, pairs, groups, None);
    let y: Vec<f32> = pairs.iter().flat_map(|&t| (0..POSE_DIM).map(move |i| gt[(t + 1) * POSE_DIM + i])).collect();
    let (pmu, plv) = m.pose(
        to_t::<BI>(zt, n, BAND_DIM, d),
        to_t::<BI>(mn, n, BAND_DIM, d),
        to_t::<BI>(mk, n, BAND_DIM, d),
    );
    let mud: Vec<f32> = pmu.into_data().to_vec().unwrap();
    let lvd: Vec<f32> = plv.into_data().to_vec().unwrap();
    let mut var = 0.0f64;
    let mut err = 0.0f64;
    for j in 0..n * POSE_DIM {
        var += (lvd[j] as f64).exp();
        err += ((mud[j] - y[j]) as f64).powi(2);
    }
    let dn = (n * POSE_DIM) as f64;
    (var / dn, err / dn)
}

/// Target-peeking upper bound, recomputed for this model on the same pairs.
fn oracle_probe_sets(
    m: &HeteroV2<BI>,
    z: &[f32],
    gt: &[f32],
    pairs: &[usize],
    k: usize,
    d: &NdArrayDevice,
) -> Vec<Vec<usize>> {
    let n = pairs.len();
    let y: Vec<f32> = pairs.iter().flat_map(|&t| (0..POSE_DIM).map(move |i| gt[(t + 1) * POSE_DIM + i])).collect();
    let mut out = vec![Vec::new(); n];
    for _step in 0..k {
        let mut best: Vec<(f64, usize)> = vec![(f64::INFINITY, usize::MAX); n];
        for cand in 0..NUM_PROBE_GROUPS {
            let gs: Vec<Vec<usize>> = (0..n)
                .map(|i| {
                    let mut s = out[i].clone();
                    if !s.contains(&cand) {
                        s.push(cand);
                    }
                    s
                })
                .collect();
            let (zt, mn, mk) = planes(z, pairs, &gs, None);
            let (pmu, _) = m.pose(
                to_t::<BI>(zt, n, BAND_DIM, d),
                to_t::<BI>(mn, n, BAND_DIM, d),
                to_t::<BI>(mk, n, BAND_DIM, d),
            );
            let mud: Vec<f32> = pmu.into_data().to_vec().unwrap();
            for i in 0..n {
                if out[i].contains(&cand) {
                    continue;
                }
                let e: f64 = (0..POSE_DIM)
                    .map(|q| {
                        let j = i * POSE_DIM + q;
                        ((mud[j] - y[j]) as f64).powi(2)
                    })
                    .sum();
                if e < best[i].0 {
                    best[i] = (e, cand);
                }
            }
        }
        for i in 0..n {
            if best[i].1 != usize::MAX {
                out[i].push(best[i].1);
            }
        }
    }
    out
}
