//! Milestone 2: validate the Rust harness against fixtures from upstream's own code.
//!
//! Exits non-zero on any failure. Prints a JSON report to stdout.
//!
//! Two independent validations, per the milestone plan:
//!   a. score the dumped (pred, gt) and match `results/eval_retrained.json`
//!   b. rebuild ground truth from `all_keypoints.npy` with this crate's own reader
//!      and zero-cleaning, and match the dumped ground truth bit-for-bit
//!
//! It also answers the open question the charter left unresolved: what does a
//! mean-pose predictor score on the WiFlow test split? That number is the honesty
//! bar for everything built afterwards.

use std::collections::BTreeMap;
use wiflow_harness::guards::{
    assert_finite, broadcast_pose, check_constant_pose, clean_frame_zeros, mean_pose,
    CorruptionMasks,
};
use wiflow_harness::metrics::{score, NUM_KEYPOINTS};
use wiflow_harness::npy::Npy;
use wiflow_harness::{FLOOR_MPJPE, FLOOR_PCK};

const RESULTS: &str =
    "/home/ruvultra/projects/ruview-worktrees/ruforecast-rust/benchmarks/wiflow-std/results";
const DATASET: &str = "/home/ruvultra/wiflow-std-bench/preprocessed_csi_data";

// CITED from results/eval_retrained.json (the honest baseline this must reproduce).
const RECORDED_FULL_PCK20: f64 = 0.9608815324571398;
const RECORDED_FULL_MPJPE: f64 = 0.009834060806367133;
const RECORDED_CLEAN_PCK20: f64 = 0.9661454100405608;
const RECORDED_CLEAN_MPJPE: f64 = 0.009432755044379373;

fn fail(msg: String) -> ! {
    eprintln!("VALIDATION FAILED: {msg}");
    std::process::exit(1);
}

fn check(name: &str, got: f64, want: f64, tol: f64, out: &mut Vec<(String, serde_json::Value)>) {
    let delta = (got - want).abs();
    let pass = delta < tol;
    out.push((
        name.to_string(),
        serde_json::json!({ "got": got, "recorded": want, "delta": delta, "tolerance": tol, "pass": pass }),
    ));
    if !pass {
        fail(format!(
            "{name}: got {got:.10}, recorded {want:.10}, delta {delta:.3e} exceeds tolerance {tol:.0e}"
        ));
    }
}

fn main() {
    let fixtures = std::env::args().nth(1).unwrap_or_else(|| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../fixtures").to_string()
    });
    let p = |f: &str| format!("{fixtures}/{f}");

    let open = |path: String| Npy::open(&path).unwrap_or_else(|e| fail(e));

    // ---- load fixtures -------------------------------------------------------
    let pred_npy = open(p("test_pred.npy"));
    let gt_npy = open(p("test_gt.npy"));
    let pred = pred_npy.f32_slice().unwrap_or_else(|e| fail(e));
    let gt = gt_npy.f32_slice().unwrap_or_else(|e| fail(e));
    let n = pred_npy.shape[0];

    if pred_npy.shape != vec![54000, NUM_KEYPOINTS, 2] {
        fail(format!("unexpected pred shape {:?}", pred_npy.shape));
    }
    if gt_npy.shape != pred_npy.shape {
        fail(format!("gt shape {:?} != pred shape {:?}", gt_npy.shape, pred_npy.shape));
    }
    assert_finite(pred, "predictions");
    assert_finite(gt, "ground truth");

    let mut checks: Vec<(String, serde_json::Value)> = Vec::new();

    // ---- validation (a): reproduce the honest baseline ----------------------
    let full = score(pred, gt, n);
    check("full.pck@20", full.pck_at(0.2), RECORDED_FULL_PCK20, FLOOR_PCK, &mut checks);
    check("full.mpjpe", full.mpjpe, RECORDED_FULL_MPJPE, FLOOR_MPJPE, &mut checks);

    let clean_sel = open(p("test_clean_sel.npy")).to_bool().unwrap_or_else(|e| fail(e));
    let mut cp = Vec::new();
    let mut cg = Vec::new();
    for (i, &keep) in clean_sel.iter().enumerate() {
        if keep {
            let b = i * NUM_KEYPOINTS * 2;
            cp.extend_from_slice(&pred[b..b + NUM_KEYPOINTS * 2]);
            cg.extend_from_slice(&gt[b..b + NUM_KEYPOINTS * 2]);
        }
    }
    let n_clean = cp.len() / (NUM_KEYPOINTS * 2);
    if n_clean != 52560 {
        fail(format!("clean subset is {n_clean}, expected 52560"));
    }
    let clean = score(&cp, &cg, n_clean);
    check("clean.pck@20", clean.pck_at(0.2), RECORDED_CLEAN_PCK20, FLOOR_PCK, &mut checks);
    check("clean.mpjpe", clean.mpjpe, RECORDED_CLEAN_MPJPE, FLOOR_MPJPE, &mut checks);

    // ---- validation (b): independent GT reconstruction -----------------------
    // This crate's own .npy reader + its own zero-cleaning must rebuild the exact
    // ground truth the Python path produced. Nothing is shared but the data.
    let ak_npy = open(format!("{DATASET}/all_keypoints.npy"));
    let ak = ak_npy.f32_slice().unwrap_or_else(|e| fail(e));
    let gfi = open(p("test_global_frame_idx.npy")).to_i64().unwrap_or_else(|e| fail(e));
    if gfi.len() != n {
        fail(format!("frame index length {} != {n}", gfi.len()));
    }
    let d = NUM_KEYPOINTS * 2;
    let mut rebuilt: Vec<f32> = Vec::with_capacity(n * d);
    for &g in &gfi {
        let b = g as usize * d;
        rebuilt.extend_from_slice(&ak[b..b + d]);
    }
    let pre_clean_mismatches = rebuilt.iter().zip(gt).filter(|(a, b)| a != b).count();
    clean_frame_zeros(&mut rebuilt, n);
    let mismatches = rebuilt.iter().zip(gt).filter(|(a, b)| a != b).count();
    checks.push((
        "gt_reconstruction".to_string(),
        serde_json::json!({
            "mismatches_after_clean": mismatches,
            "mismatches_before_clean": pre_clean_mismatches,
            "pass": mismatches == 0,
            "note": "independent Rust .npy read + zero-clean must equal the Python-dumped GT bit-for-bit"
        }),
    ));
    if mismatches != 0 {
        fail(format!("GT reconstruction differs in {mismatches} of {} values", n * d));
    }
    if pre_clean_mismatches == 0 {
        fail("zero-cleaning changed nothing -- the code path is untested by this fixture".into());
    }

    // ---- corruption masks ----------------------------------------------------
    let masks = CorruptionMasks::load(
        &format!("{RESULTS}/nan_windows_mask.npy"),
        &format!("{RESULTS}/big_windows_mask.npy"),
    )
    .unwrap_or_else(|e| fail(e));
    let test_idx = open(p("test_idx.npy")).to_i64().unwrap_or_else(|e| fail(e));
    let derived_clean = masks.clean_selector(&test_idx);
    let derived_n = derived_clean.iter().filter(|&&b| b).count();
    let selector_agrees = derived_clean == clean_sel;
    checks.push((
        "corruption_masks".to_string(),
        serde_json::json!({
            "nan_windows": masks.nan.iter().filter(|&&b| b).count(),
            "big_windows": masks.big.iter().filter(|&&b| b).count(),
            "union": masks.corrupt_count(),
            "clean_test_windows": derived_n,
            "selector_matches_fixture": selector_agrees,
            "pass": derived_n == 52560 && selector_agrees
        }),
    ));
    if derived_n != 52560 || !selector_agrees {
        fail(format!("mask-derived clean selector disagrees (n={derived_n})"));
    }

    // ---- guards on the real model -------------------------------------------
    let verdict = check_constant_pose(pred, n);
    verdict.assert_ok("retrained WiFlow-STD checkpoint");

    // ---- the honesty bar (open question 3, previously UNKNOWN) --------------
    // Two variants. The train-fitted one is what a deployable constant predictor
    // would actually score. The test-fitted one is an ORACLE upper bound -- the best
    // any constant predictor could do on this split -- and is the stricter bar.
    let train_mean = open(p("train_mean_pose.npy")).to_f32().unwrap_or_else(|e| fail(e));
    let bar_train = score(&broadcast_pose(&train_mean, n), gt, n);
    let bar_oracle = score(&broadcast_pose(&mean_pose(gt, n), n), gt, n);

    let model_pck20 = full.pck_at(0.2);
    let beats_bar = model_pck20 > bar_oracle.pck_at(0.2);

    let report = serde_json::json!({
        "milestone": "M2 -- reproduce the honest baseline through the Rust harness",
        "measured_by": "harness/src/bin/validate.rs (this run)",
        "reproducibility_floor": {
            "pck@20": FLOOR_PCK,
            "mpjpe": FLOOR_MPJPE,
            "basis": "MEASURED by scripts/diag_reproducibility_floor.py: TF32 on/off alone \
                      moves PCK@20 by 1.36e-5; the recorded value lies between the two runs"
        },
        "checks": checks.into_iter().collect::<BTreeMap<_, _>>(),
        "scores": { "test_full": full, "test_clean": clean },
        "degeneracy_guard": verdict,
        "honesty_bar_mean_pose": {
            "train_fitted": { "pck@20": bar_train.pck_at(0.2), "mpjpe": bar_train.mpjpe },
            "test_fitted_oracle": { "pck@20": bar_oracle.pck_at(0.2), "mpjpe": bar_oracle.mpjpe },
            "model_pck@20": model_pck20,
            "model_beats_oracle_bar": beats_bar,
            "note": "On the ESP32 set (RESULTS.md measurement b) the mean-pose bar was 95.9% \
                     and nothing beat it. This is the equivalent number for WiFlow-STD."
        },
        "verdict": if beats_bar { "PASS" } else { "PASS-WITH-WARNING: model does not beat the constant-pose bar" }
    });

    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}
