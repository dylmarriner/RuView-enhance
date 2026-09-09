//! Pose metrics, matching upstream WiFlow-STD `utils/metrics.py` exactly.
//!
//! The exactness matters: an earlier "92.9% PCK@20" claim in this repo was retracted
//! because the trainer used an ABSOLUTE 0.2-image-unit threshold instead of the
//! torso-normalized one, under which a constant-pose predictor scores 100%.
//! [`absolute_pck`] below reproduces that broken protocol deliberately, as a foil
//! the tests use to prove this harness rejects what the old one accepted.

pub const NUM_KEYPOINTS: usize = 15;
/// Upstream `NECK_IDX` on the 15-keypoint convention.
pub const NECK_IDX: usize = 2;
/// Upstream `PELVIS_IDX` on the 15-keypoint convention.
pub const PELVIS_IDX: usize = 12;
/// Upstream `torch.clamp(normalize_distances, min=0.01)`.
pub const TORSO_CLAMP: f64 = 0.01;

pub const THRESHOLDS: [f64; 5] = [0.1, 0.2, 0.3, 0.4, 0.5];

#[derive(Debug, Clone, serde::Serialize)]
pub struct Scores {
    pub samples: usize,
    pub mpjpe: f64,
    pub pck: Vec<(String, f64)>,
}

impl Scores {
    pub fn pck_at(&self, t: f64) -> f64 {
        let key = format!("pck@{}", (t * 100.0).round() as i64);
        self.pck
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| *v)
            .unwrap_or_else(|| panic!("threshold {t} not computed"))
    }
}

/// Per-frame torso length, upstream definition: `‖target[NECK] − target[PELVIS]‖`,
/// clamped below at 0.01 so a degenerate frame cannot divide by ~0.
pub fn torso_norms(gt: &[f32], n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let b = i * NUM_KEYPOINTS * 2;
            let (nx, ny) = (gt[b + NECK_IDX * 2] as f64, gt[b + NECK_IDX * 2 + 1] as f64);
            let (px, py) = (gt[b + PELVIS_IDX * 2] as f64, gt[b + PELVIS_IDX * 2 + 1] as f64);
            ((nx - px).powi(2) + (ny - py).powi(2)).sqrt().max(TORSO_CLAMP)
        })
        .collect()
}

/// Torso-normalized PCK and MPJPE.
///
/// Upstream takes a single flat mean over (frames x keypoints) -- NOT a mean of
/// per-frame means. Computed in f64: the f32 path carries ~1e-9 of accumulation
/// error, which is irrelevant next to the forward pass's ~1.4e-5 floor but costs
/// nothing to avoid here.
pub fn score(pred: &[f32], gt: &[f32], n: usize) -> Scores {
    assert_eq!(pred.len(), n * NUM_KEYPOINTS * 2, "pred length mismatch");
    assert_eq!(gt.len(), n * NUM_KEYPOINTS * 2, "gt length mismatch");

    let torso = torso_norms(gt, n);
    let mut sum_dist = 0.0f64;
    let mut hits = [0usize; THRESHOLDS.len()];

    for i in 0..n {
        let b = i * NUM_KEYPOINTS * 2;
        for k in 0..NUM_KEYPOINTS {
            let dx = pred[b + k * 2] as f64 - gt[b + k * 2] as f64;
            let dy = pred[b + k * 2 + 1] as f64 - gt[b + k * 2 + 1] as f64;
            let d = (dx * dx + dy * dy).sqrt();
            sum_dist += d;
            let nd = d / torso[i];
            for (j, &t) in THRESHOLDS.iter().enumerate() {
                if nd <= t {
                    hits[j] += 1;
                }
            }
        }
    }

    let total = (n * NUM_KEYPOINTS) as f64;
    Scores {
        samples: n,
        mpjpe: sum_dist / total,
        pck: THRESHOLDS
            .iter()
            .zip(hits)
            .map(|(&t, h)| (format!("pck@{}", (t * 100.0).round() as i64), h as f64 / total))
            .collect(),
    }
}

/// The BROKEN protocol behind the retracted 92.9% claim: an absolute pixel/image-unit
/// threshold with no torso normalization.
///
/// Present only so tests can demonstrate the difference. Never report a number from
/// this function as an accuracy result.
pub fn absolute_pck(pred: &[f32], gt: &[f32], n: usize, threshold: f64) -> f64 {
    let mut hits = 0usize;
    for i in 0..n {
        let b = i * NUM_KEYPOINTS * 2;
        for k in 0..NUM_KEYPOINTS {
            let dx = pred[b + k * 2] as f64 - gt[b + k * 2] as f64;
            let dy = pred[b + k * 2 + 1] as f64 - gt[b + k * 2 + 1] as f64;
            if (dx * dx + dy * dy).sqrt() <= threshold {
                hits += 1;
            }
        }
    }
    hits as f64 / (n * NUM_KEYPOINTS) as f64
}
