//! Criterion benchmarks for the ADR-151 calibration hot paths.
//!
//! Backs the ADR-151 claim that the per-room specialists "run in microseconds
//! on a Pi CPU" by measuring the two per-window hot paths the live runtime
//! executes every feature window:
//!
//! 1. **`MixtureOfSpecialists::infer`** — one full mixture inference over a
//!    fully-trained [`SpecialistBank`]: anomaly veto → presence gate →
//!    restlessness → posture/breathing/heartbeat → fused [`RoomState`].
//! 2. **`Features::from_series`** — feature extraction (mean/variance/motion
//!    plus the breathing/heart-band autocorrelation periodicity) over a
//!    realistic synthetic CSI-derived scalar window (~150 samples @ 15 Hz),
//!    the O(n·lags) inner loop that dominates extraction cost.
//!
//! All inputs are deterministic (a fixed LCG + analytic sines) so timings are
//! reproducible across runs and across the host ↔ aarch64 cross-build.
//!
//! Run (compile-only check):
//!   cargo bench -p wifi-densepose-calibration --no-run
//!
//! Run to completion (generates HTML in target/criterion/):
//!   cargo bench -p wifi-densepose-calibration --bench specialist_bench

use std::f32::consts::PI;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use wifi_densepose_calibration::anchor::AnchorLabel;
use wifi_densepose_calibration::extract::{AnchorFeature, Features};
use wifi_densepose_calibration::{MixtureOfSpecialists, SpecialistBank};

const BASELINE_ID: &str = "bench-baseline-1";
const FS: f32 = 15.0; // CSI frame rate (Hz) — matches ADR-135/151 fixtures.

// ---------------------------------------------------------------------------
// Deterministic fixtures (no RNG that breaks reproducibility).
// ---------------------------------------------------------------------------

/// A labelled anchor with hand-set features — mirrors what enrollment fits.
fn af(label: AnchorLabel, variance: f32, motion: f32) -> AnchorFeature {
    AnchorFeature {
        room_id: "bench-room".into(),
        label,
        features: Features {
            mean: 1.0,
            variance,
            motion,
            breathing_score: 0.0,
            breathing_hz: 0.0,
            heart_score: 0.0,
            heart_hz: 0.0,
        },
    }
}

/// A fully-trained bank: every specialist (presence, posture, breathing,
/// heartbeat, restlessness, anomaly) is fit so `infer` exercises the whole
/// fusion path.
fn full_bank() -> SpecialistBank {
    let anchors = vec![
        af(AnchorLabel::Empty, 1.0, 0.1),
        af(AnchorLabel::StandStill, 10.0, 0.2),
        af(AnchorLabel::Sit, 6.0, 0.2),
        af(AnchorLabel::LieDown, 3.0, 0.2),
        af(AnchorLabel::SmallMove, 4.0, 1.2),
        af(AnchorLabel::SleepPosture, 3.0, 0.1),
    ];
    SpecialistBank::train("bench-room", BASELINE_ID, &anchors, 1_000).unwrap()
}

/// A live feature window for an occupied, breathing room — drives the
/// "present, not vetoed, vitals reported" branch (the most expensive path).
fn live_present() -> Features {
    Features {
        mean: 1.0,
        variance: 10.0,
        motion: 0.4,
        breathing_score: 0.85,
        breathing_hz: 0.30, // 18 BPM
        heart_score: 0.55,
        heart_hz: 1.20, // 72 BPM
    }
}

/// A synthetic CSI-derived scalar series: breathing tone + faint heartbeat tone
/// + a small deterministic LCG perturbation, of the realistic per-window length
/// produced by a `subcarriers`-wide × `samples`-deep window collapsed to a
/// per-frame scalar (mean amplitude). `from_series` consumes the scalar series,
/// so `subcarriers` sets only the deterministic per-frame jitter.
fn synthetic_window(subcarriers: usize, samples: usize) -> Vec<f32> {
    let mut s: u32 = 0x1234_5678;
    (0..samples)
        .map(|i| {
            let t = i as f32 / FS;
            let breathing = 1.0 * (2.0 * PI * 0.30 * t).sin();
            let heart = 0.15 * (2.0 * PI * 1.20 * t).sin();
            // Deterministic LCG jitter scaled by the (notional) subcarrier count.
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let u = (s >> 8) as f32 / (1u32 << 24) as f32; // [0,1)
            let jitter = (u - 0.5) * 0.05 * (subcarriers as f32 / 64.0);
            1.0 + breathing + heart + jitter
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Bench 1: one full mixture inference over a trained bank.
// ---------------------------------------------------------------------------

fn bench_mixture_infer(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixture_infer");
    group.throughput(Throughput::Elements(1)); // one window → one RoomState

    let mix = MixtureOfSpecialists::new(full_bank());
    let window = live_present();

    group.bench_function("present_window", |b| {
        b.iter(|| {
            let state = mix.infer(black_box(&window), black_box(BASELINE_ID));
            black_box(state);
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Bench 2: feature extraction (autocorrelation periodicity) over a window.
// ---------------------------------------------------------------------------

fn bench_feature_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_extraction");

    // Realistic per-window depths at fs=15 Hz: ~150 samples ≈ 10 s window
    // (the breathing-band integration time). 64-subcarrier capture is the
    // HT20 commodity-CSI tier.
    const SUBCARRIERS: usize = 64;
    for &samples in &[150usize, 300, 450] {
        let window = synthetic_window(SUBCARRIERS, samples);
        group.throughput(Throughput::Elements(samples as u64));
        group.bench_with_input(
            BenchmarkId::new("from_series_64sc", samples),
            &window,
            |b, w| {
                b.iter(|| {
                    let f = Features::from_series(black_box(w), black_box(FS));
                    black_box(f);
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_mixture_infer, bench_feature_extraction);
criterion_main!(benches);
