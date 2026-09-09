//! Regression test for the retraction this project must not repeat.
//!
//! History (`RESULTS.md`, "Measurement (b)"): a model was reported at 92.9% PCK@20.
//! The trainer's `pck()` used an ABSOLUTE 0.2-image-unit threshold rather than the
//! torso-normalized one, and the model emitted a constant pose (pred std 0.0000)
//! over 69 near-static frames. A mean predictor scores 100% under that protocol.
//! The torso-normalized value on the same holdout was 19.1%.
//!
//! These tests construct that exact situation and assert the harness rejects what
//! the old protocol accepted. This is the charter's "prove it can't" requirement:
//! the guard is a failing test, not a sentence in a report.

use wiflow_harness::guards::{check_constant_pose, mean_pose, broadcast_pose, prediction_std};
use wiflow_harness::metrics::{absolute_pck, score, NUM_KEYPOINTS};

/// Deterministic jitter. A dependency-free LCG keeps the fixture reproducible
/// without pinning an RNG crate's stream behaviour.
struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // Map the top 24 bits to (-0.5, 0.5).
        ((self.0 >> 40) as f32 / (1u32 << 24) as f32) - 0.5
    }
}

/// A near-static subject: one canonical pose plus small per-frame jitter.
/// Torso (NECK idx 2 to PELVIS idx 12) is ~0.20 image units by construction.
fn near_static_scene(n: usize, jitter: f32) -> (Vec<f32>, Vec<f32>) {
    let mut base = vec![0.0f32; NUM_KEYPOINTS * 2];
    for k in 0..NUM_KEYPOINTS {
        base[k * 2] = 0.5;
        base[k * 2 + 1] = 0.30 + 0.03 * k as f32;
    }
    base[2 * 2 + 1] = 0.40; // NECK
    base[12 * 2 + 1] = 0.60; // PELVIS  -> torso = 0.20

    let mut rng = Lcg(0x5EED_1234_ABCD_0001);
    let mut gt = Vec::with_capacity(n * NUM_KEYPOINTS * 2);
    for _ in 0..n {
        for c in 0..NUM_KEYPOINTS * 2 {
            gt.push(base[c] + jitter * rng.next_f32());
        }
    }
    // The retracted model's behaviour: the same pose for every frame.
    let constant_pred = broadcast_pose(&base, n);
    (gt, constant_pred)
}

#[test]
fn constant_pose_detector_trips_on_the_retracted_failure_mode() {
    let (_gt, pred) = near_static_scene(69, 0.10);
    let verdict = check_constant_pose(&pred, 69);

    assert!(
        verdict.degenerate,
        "constant prediction must be flagged degenerate, got pred_std {:.3e}",
        verdict.pred_std
    );
    // The retracted model measured exactly 0.0000; a truly constant one is 0.
    assert_eq!(verdict.pred_std, 0.0, "a broadcast pose has zero spread by construction");
}

#[test]
fn honest_model_spread_is_not_flagged() {
    // measurement (b)'s honest fine-tune measured pred std 0.0113 -- two orders of
    // magnitude above the 1e-4 threshold. Confirm the guard has real headroom and
    // does not reject legitimate models.
    let n = 500;
    let mut rng = Lcg(0xABCD_0000_1111_2222);
    let mut pred = Vec::with_capacity(n * NUM_KEYPOINTS * 2);
    for _ in 0..n {
        for k in 0..NUM_KEYPOINTS * 2 {
            pred.push(0.5 + 0.04 * rng.next_f32() + 0.0001 * k as f32);
        }
    }
    let verdict = check_constant_pose(&pred, n);
    assert!(!verdict.degenerate, "an honest varying model must not be flagged");
    assert!(
        verdict.pred_std > 1e-3,
        "sanity: constructed spread should be well above threshold, got {:.3e}",
        verdict.pred_std
    );
}

#[test]
fn broken_absolute_protocol_passes_where_torso_normalized_rejects() {
    // The heart of the retraction. Same predictions, same ground truth, two
    // protocols -- one of which is a lie.
    let n = 69;
    let (gt, pred) = near_static_scene(n, 0.10);

    let broken = absolute_pck(&pred, &gt, n, 0.2);
    let honest = score(&pred, &gt, n).pck_at(0.2);

    assert!(
        broken > 0.99,
        "the retracted absolute-0.2 protocol should accept this constant predictor \
         (that is why it was wrong); got {broken:.4}"
    );
    assert!(
        honest < 0.60,
        "torso-normalized PCK@20 must NOT accept a constant predictor on this scene; \
         got {honest:.4}"
    );
    assert!(
        broken - honest > 0.35,
        "the two protocols must disagree materially: absolute {broken:.4} vs torso {honest:.4}"
    );

    // And the degeneracy guard catches it regardless of which metric was reported.
    check_constant_pose(&pred, n).assert_ok_is_err();
}

/// Helper: assert that `assert_ok` would panic, without aborting the test.
trait AssertOkIsErr {
    fn assert_ok_is_err(&self);
}

impl AssertOkIsErr for wiflow_harness::guards::ConstantPoseVerdict {
    fn assert_ok_is_err(&self) {
        let v = self.clone();
        let r = std::panic::catch_unwind(move || v.assert_ok("retraction fixture"));
        assert!(r.is_err(), "assert_ok should have panicked on a degenerate prediction");
    }
}

#[test]
fn mean_predictor_is_the_honesty_bar_on_near_static_data() {
    // Why the bar exists: on a near-static subject a constant mean pose scores very
    // high under the CORRECT protocol too. measurement (b) measured 95.9% torso-PCK@20
    // for exactly this reason. A model that cannot beat this has demonstrated nothing.
    let n = 400;
    let (gt, _) = near_static_scene(n, 0.02); // very little motion
    let bar = broadcast_pose(&mean_pose(&gt, n), n);
    let bar_pck = score(&bar, &gt, n).pck_at(0.2);

    assert!(
        bar_pck > 0.90,
        "on near-static data the mean-pose bar should be high -- that is the point; got {bar_pck:.4}"
    );
    // It is a constant predictor, so the degeneracy guard must also flag it.
    assert!(check_constant_pose(&bar, n).degenerate);
    assert_eq!(prediction_std(&bar, n), 0.0);
}

#[test]
fn torso_clamp_prevents_divide_by_zero_inflation() {
    // A frame whose NECK and PELVIS coincide has torso 0. Without upstream's
    // clamp(min=0.01) the normalized distance is +inf and PCK silently drops to 0
    // (or NaN). Confirm the clamp is in force.
    let n = 2;
    let mut gt = vec![0.5f32; n * NUM_KEYPOINTS * 2];
    // Frame 0: degenerate torso (neck == pelvis, both already 0.5/0.5).
    // Frame 1: normal torso.
    gt[NUM_KEYPOINTS * 2 + 2 * 2 + 1] = 0.40;
    gt[NUM_KEYPOINTS * 2 + 12 * 2 + 1] = 0.60;

    let pred = gt.clone(); // perfect prediction
    let s = score(&pred, &gt, n);
    assert_eq!(s.pck_at(0.2), 1.0, "a perfect prediction must score 1.0 even with a degenerate torso");
    assert!(s.mpjpe.is_finite() && s.mpjpe == 0.0);
}
