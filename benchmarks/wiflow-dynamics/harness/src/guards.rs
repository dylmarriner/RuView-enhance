//! Anti-Goodhart guards.
//!
//! Charter requirement: "Any evaluation path that can score high with a constant
//! output is a defect -- prove it can't." These are the proofs. They are wired as
//! assertions and tests, not as numbers in a report, because a guard that is merely
//! *reported* is a guard nobody runs.
//!
//! Three independent things can make a high score meaningless:
//!   1. the model emits a (near-)constant pose            -> [`ConstantPoseVerdict`]
//!   2. the task is so easy a mean predictor wins         -> [`mean_pose`] + score it
//!   3. corrupted windows silently enter the split        -> [`CorruptionMasks`]

use crate::metrics::NUM_KEYPOINTS;

/// Prediction spread across frames.
///
/// Definition is deliberately the one used by measurement (b) in `RESULTS.md`, so
/// numbers are comparable across this repo's history: per (keypoint, coordinate)
/// standard deviation across frames, then averaged over the 30 series.
///
/// Reference values from that measurement:
///   - retracted constant-pose model: 0.0000
///   - honest fine-tuned model:       0.0113
pub fn prediction_std(pred: &[f32], n: usize) -> f64 {
    assert!(n > 1, "prediction spread is undefined for n <= 1");
    let d = NUM_KEYPOINTS * 2;
    let mut total = 0.0f64;
    for c in 0..d {
        let mean = (0..n).map(|i| pred[i * d + c] as f64).sum::<f64>() / n as f64;
        let var = (0..n)
            .map(|i| {
                let x = pred[i * d + c] as f64 - mean;
                x * x
            })
            .sum::<f64>()
            / n as f64;
        total += var.sqrt();
    }
    total / d as f64
}

/// Below this, a prediction is treated as constant. The retracted model measured
/// exactly 0.0000; an honest one measured 0.0113. Two orders of magnitude of
/// headroom sit between the threshold and the honest value.
pub const DEGENERACY_THRESHOLD: f64 = 1e-4;

#[derive(Debug, Clone, serde::Serialize)]
pub struct ConstantPoseVerdict {
    pub pred_std: f64,
    pub threshold: f64,
    pub degenerate: bool,
}

/// Trips when a model's output barely varies across frames -- the failure mode that
/// produced this repo's retracted 92.9% claim.
pub fn check_constant_pose(pred: &[f32], n: usize) -> ConstantPoseVerdict {
    let pred_std = prediction_std(pred, n);
    ConstantPoseVerdict {
        pred_std,
        threshold: DEGENERACY_THRESHOLD,
        degenerate: pred_std < DEGENERACY_THRESHOLD,
    }
}

impl ConstantPoseVerdict {
    /// Hard-fail an evaluation whose predictions are degenerate.
    pub fn assert_ok(&self, context: &str) {
        assert!(
            !self.degenerate,
            "{context}: DEGENERATE PREDICTION -- pred_std {:.6e} < {:.0e}. \
             The model is emitting a (near-)constant pose; any PCK reported from it \
             is meaningless. This is the exact failure that retracted the 92.9% claim.",
            self.pred_std, self.threshold
        );
    }
}

/// The honesty bar: the constant pose that minimises error over a reference split.
///
/// Fitted on TRAIN data and scored on TEST. Anything that cannot beat this has not
/// demonstrated pose estimation -- measurement (b) in `RESULTS.md` is precisely the
/// case where nothing did (mean-pose 95.9% vs best fine-tune 65.0% torso-PCK@20).
pub fn mean_pose(gt: &[f32], n: usize) -> Vec<f32> {
    assert!(n > 0, "cannot fit a mean pose to an empty split");
    let d = NUM_KEYPOINTS * 2;
    (0..d)
        .map(|c| ((0..n).map(|i| gt[i * d + c] as f64).sum::<f64>() / n as f64) as f32)
        .collect()
}

/// Broadcast a single pose to `n` frames, so it can be scored by the normal path.
pub fn broadcast_pose(pose: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(pose.len(), NUM_KEYPOINTS * 2);
    pose.repeat(n)
}

/// Corruption masks over the full 360,000-window index space.
///
/// `RESULTS.md`: dataset files 487-499 carry NaN plus amplitudes to 3.4e38 in data
/// that is otherwise [0,1]-normalized. 9,070 windows are non-finite; 9,072 exceed
/// |1.5|; the union is 9,072.
pub struct CorruptionMasks {
    pub nan: Vec<bool>,
    pub big: Vec<bool>,
}

impl CorruptionMasks {
    pub fn load(nan_path: &str, big_path: &str) -> Result<Self, String> {
        let nan = crate::npy::Npy::open(nan_path)?.to_bool()?;
        let big = crate::npy::Npy::open(big_path)?.to_bool()?;
        if nan.len() != big.len() {
            return Err(format!("mask length mismatch: {} vs {}", nan.len(), big.len()));
        }
        Ok(Self { nan, big })
    }

    pub fn is_corrupt(&self, window_idx: usize) -> bool {
        self.nan[window_idx] || self.big[window_idx]
    }

    pub fn corrupt_count(&self) -> usize {
        (0..self.nan.len()).filter(|&i| self.is_corrupt(i)).count()
    }

    /// Boolean selector over a split: `true` where the window is clean.
    pub fn clean_selector(&self, split_indices: &[i64]) -> Vec<bool> {
        split_indices.iter().map(|&i| !self.is_corrupt(i as usize)).collect()
    }
}

/// Assert an array carries no non-finite values before it is scored.
///
/// Upstream has no NaN handling at all; a single NaN silently poisons a mean.
pub fn assert_finite(x: &[f32], context: &str) {
    if let Some((i, v)) = x.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        panic!("{context}: non-finite value {v} at flat index {i}");
    }
}

/// Upstream `_clean_single_frame_zeros`: any keypoint that is exactly (0,0) is
/// replaced by the mean of that frame's non-zero keypoints. Frames with no non-zero
/// keypoint are left untouched.
///
/// Reimplemented here so the Rust harness can rebuild ground truth from
/// `all_keypoints.npy` independently of the Python fixture path.
pub fn clean_frame_zeros(kp: &mut [f32], n: usize) {
    let d = NUM_KEYPOINTS * 2;
    for i in 0..n {
        let b = i * d;
        let nonzero: Vec<usize> = (0..NUM_KEYPOINTS)
            .filter(|&k| kp[b + k * 2] != 0.0 || kp[b + k * 2 + 1] != 0.0)
            .collect();
        if nonzero.is_empty() || nonzero.len() == NUM_KEYPOINTS {
            continue;
        }
        // f32 accumulation, matching numpy's float32 `.mean(axis=0)` on this array.
        let mx = nonzero.iter().map(|&k| kp[b + k * 2]).sum::<f32>() / nonzero.len() as f32;
        let my = nonzero.iter().map(|&k| kp[b + k * 2 + 1]).sum::<f32>() / nonzero.len() as f32;
        for k in 0..NUM_KEYPOINTS {
            if kp[b + k * 2] == 0.0 && kp[b + k * 2 + 1] == 0.0 {
                kp[b + k * 2] = mx;
                kp[b + k * 2 + 1] = my;
            }
        }
    }
}
