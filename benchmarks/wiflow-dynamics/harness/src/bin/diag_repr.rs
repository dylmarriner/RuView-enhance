//! Diagnostic: is the bottleneck the REPRESENTATION or the MODEL CLASS?
//!
//! M3 rung 2 predicts `y_{t+1}` from `z_t` and lands exactly at the mean-pose bar.
//! Two very different explanations:
//!   (i)  the 27-band-mean representation destroys the pose signal, or
//!   (ii) a linear-Gaussian model cannot extract it.
//!
//! This separates them by asking the EASIEST possible version of the question:
//! predict the pose of the SAME window from its own band features. That removes the
//! forecasting gap entirely. If same-timestep pose is still at the bar, the
//! representation is lossy and no amount of model capacity on `z` will help.
//!
//! Also sweeps band granularity, because "27 bands of 20 channels" was a
//! pre-registered choice, not a measured optimum.

use nalgebra::DVector;
use wiflow_harness::bands::{NUM_CHANNELS, NUM_FRAMES};
use wiflow_harness::gaussian::{gaussian_nll, POSE_DIM};
use wiflow_harness::guards::CorruptionMasks;
use wiflow_harness::metrics::score;
use wiflow_harness::npy::Npy;

use nalgebra::{DMatrix, DVector as DV};

const RESULTS: &str =
    "/home/ruvultra/projects/ruview-worktrees/ruforecast-rust/benchmarks/wiflow-std/results";
const DATASET: &str = "/home/ruvultra/wiflow-std-bench/preprocessed_csi_data";

struct Rng(u64);
impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as usize % n
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

/// Reduce a raw window to `nb` contiguous bands, keeping `frames_kept` time samples
/// (stride-subsampled) per band.
fn reduce(csi: &[f32], nb: usize, frames_kept: usize, out: &mut [f64]) {
    let cpb = NUM_CHANNELS / nb;
    // frames_kept == 0 is a sentinel for "mean over all 20 frames".
    if frames_kept == 0 {
        for b in 0..nb {
            let mut acc = 0.0f64;
            for c in 0..cpb {
                for f in 0..NUM_FRAMES {
                    acc += csi[(b * cpb + c) * NUM_FRAMES + f] as f64;
                }
            }
            out[b] = acc / (cpb * NUM_FRAMES) as f64;
        }
        return;
    }
    let fstride = NUM_FRAMES / frames_kept;
    for b in 0..nb {
        for fi in 0..frames_kept {
            let f = fi * fstride;
            let mut acc = 0.0f64;
            for c in 0..cpb {
                acc += csi[(b * cpb + c) * NUM_FRAMES + f] as f64;
            }
            out[b * frames_kept + fi] = acc / cpb as f64;
        }
    }
}

/// Fit `p(y | v)` as a joint Gaussian and report held-out pose metrics.
fn run(label: &str, nb: usize, frames_kept: usize, csi: &[f32], gt: &[f32], train: &[usize], test: &[usize]) {
    let vd = if frames_kept == 0 { nb } else { nb * frames_kept };
    let jd = vd + POSE_DIM;
    let per_window = NUM_CHANNELS * NUM_FRAMES;

    // Accumulate joint stats over [v, y].
    let mut n = 0usize;
    let mut sum: DV<f64> = DV::zeros(jd);
    let mut outer: DMatrix<f64> = DMatrix::zeros(jd, jd);
    let mut v = vec![0.0f64; jd];
    for &t in train {
        reduce(&csi[t * per_window..(t + 1) * per_window], nb, frames_kept, &mut v[..vd]);
        for i in 0..POSE_DIM {
            v[vd + i] = gt[t * POSE_DIM + i] as f64;
        }
        n += 1;
        for i in 0..jd {
            sum[i] += v[i];
            for j in i..jd {
                outer[(i, j)] += v[i] * v[j];
            }
        }
    }
    let mean = &sum / n as f64;
    let mut cov: DMatrix<f64> = DMatrix::zeros(jd, jd);
    for i in 0..jd {
        for j in i..jd {
            let c = (outer[(i, j)] - n as f64 * mean[i] * mean[j]) / (n as f64 - 1.0);
            cov[(i, j)] = c;
            cov[(j, i)] = c;
        }
    }
    let scale = cov.diagonal().mean();
    for i in 0..jd {
        cov[(i, i)] += 1e-5 * scale;
    }

    let s_vv = cov.view((0, 0), (vd, vd)).into_owned();
    let s_yv = cov.view((vd, 0), (POSE_DIM, vd)).into_owned();
    let s_yy = cov.view((vd, vd), (POSE_DIM, POSE_DIM)).into_owned();
    let Some(chol) = s_vv.clone().cholesky() else {
        println!("{label}: covariance not PD, skipped");
        return;
    };
    let a = chol.solve(&s_yv.transpose()).transpose();
    let mut post: DMatrix<f64> = &s_yy - &a * s_yv.transpose();
    post = (&post + &post.transpose()) * 0.5;

    let mut preds = Vec::with_capacity(test.len() * POSE_DIM);
    let mut truths = Vec::with_capacity(test.len() * POSE_DIM);
    let mut nll = 0.0;
    for &t in test {
        reduce(&csi[t * per_window..(t + 1) * per_window], nb, frames_kept, &mut v[..vd]);
        let centred = DV::from_fn(vd, |i, _| v[i] - mean[i]);
        let mu_y = DV::from_fn(POSE_DIM, |i, _| mean[vd + i]) + &a * centred;
        let y = DVector::from_iterator(POSE_DIM, (0..POSE_DIM).map(|i| gt[t * POSE_DIM + i] as f64));
        nll += gaussian_nll(&y, &mu_y, &post).unwrap_or(f64::NAN);
        preds.extend(mu_y.iter().map(|&x| x as f32));
        truths.extend(y.iter().map(|&x| x as f32));
    }
    let s = score(&preds, &truths, test.len());

    // Bar on exactly these test windows.
    let mut mu = vec![0.0f64; POSE_DIM];
    for &t in test {
        for i in 0..POSE_DIM {
            mu[i] += gt[t * POSE_DIM + i] as f64;
        }
    }
    let barv: Vec<f32> = mu.iter().map(|x| (x / test.len() as f64) as f32).collect();
    let bs = score(&barv.repeat(test.len()), &truths, test.len());

    println!(
        "{label:<34} dim {vd:>5}  PCK@20 {:.4}  (bar {:.4}, delta {:+.4})  MPJPE {:.5}  NLL {:.2}",
        s.pck_at(0.2),
        bs.pck_at(0.2),
        s.pck_at(0.2) - bs.pck_at(0.2),
        s.mpjpe,
        nll / test.len() as f64
    );
}

fn main() {
    let fx = concat!(env!("CARGO_MANIFEST_DIR"), "/../fixtures");
    let load = |f: &str| Npy::open(format!("{fx}/{f}")).unwrap_or_else(|e| panic!("{e}"));
    let csi_npy = Npy::open(format!("{DATASET}/csi_windows.npy")).unwrap();
    let csi = csi_npy.f32_slice().unwrap();
    let gt_npy = load("all_gt_clean.npy");
    let gt = gt_npy.f32_slice().unwrap();
    let masks = CorruptionMasks::load(
        &format!("{RESULTS}/nan_windows_mask.npy"),
        &format!("{RESULTS}/big_windows_mask.npy"),
    )
    .unwrap();

    let pick = |f: &str, k: usize, seed: u64| {
        let idx: Vec<usize> = load(f)
            .to_i64()
            .unwrap()
            .into_iter()
            .map(|v| v as usize)
            .filter(|&i| !masks.is_corrupt(i))
            .collect();
        subsample(&idx, k, seed)
    };
    let train = pick("train_idx.npy", 40_000, 42);
    let test = pick("test_idx.npy", 10_000, 1234);

    println!("SAME-TIMESTEP pose from band features (no forecasting gap).");
    println!("train {} / test {} windows\n", train.len(), test.len());
    for (label, nb, fk) in [
        ("27 bands x 20 frames (M3 choice)", 27usize, 20usize),
        ("27 bands x 4 frames", 27, 4),
        ("54 bands x 10 frames", 54, 10),
        ("108 bands x 5 frames", 108, 5),
        ("135 bands x 4 frames", 135, 4),
        ("270 bands x 2 frames", 270, 2),
        ("540 channels x 1 frame", 540, 1),
        ("540 channels, MEAN over frames", 540, 0),
        ("270 bands, MEAN over frames", 270, 0),
        ("135 bands, MEAN over frames", 135, 0),
        ("540 ch mean + 540 ch frame0 is not tested here", 27, 0),
    ] {
        run(label, nb, fk, csi, gt, &train, &test);
    }
}
