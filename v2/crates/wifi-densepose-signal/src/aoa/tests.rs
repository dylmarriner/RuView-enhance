//! Tests on SYNTHETIC plane-wave data. Nothing here is hardware evidence.
//!
//! The generator below computes plane-wave phases from first principles
//! (independently of `ArrayGeometry::steering`) and uses a private
//! deterministic RNG so results are reproducible.

use super::adapter::{snapshots_from_csi_data, snapshots_from_iq_frames};
use super::calibration::{remove_common_phase, Calibration, Reference};
use super::covariance::{forward_backward, sample_covariance, spatial_smoothing, Smoothing};
use super::estimators::*;
use super::fusion::*;
use super::geometry::ArrayGeometry;
use super::spotfi::*;
use super::{AoaError, Snapshots};
use ndarray::{Array2, Array3};
use num_complex::Complex64;
use std::f64::consts::PI;

const FC: f64 = 2.437e9; // 2.4 GHz channel 6
const C0: f64 = 299_792_458.0;

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
    fn cgauss(&mut self) -> Complex64 {
        Complex64::new(self.gauss(), self.gauss()) * std::f64::consts::FRAC_1_SQRT_2
    }
}

/// Plane-wave phase at position `p` for a source at (az, el), from first principles.
fn plane_wave(p: [f64; 3], az: f64, el: f64) -> Complex64 {
    let u = [el.cos() * az.cos(), el.cos() * az.sin(), el.sin()];
    let k = 2.0 * PI * FC / C0;
    Complex64::from_polar(1.0, k * (p[0] * u[0] + p[1] * u[1] + p[2] * u[2]))
}

struct Src {
    az: f64,
    el: f64,
    /// If `Some((i, g))`, this source is the coherent copy `g * source[i]`.
    coherent_with: Option<(usize, Complex64)>,
}
fn src(az_deg: f64) -> Src {
    Src {
        az: az_deg.to_radians(),
        el: 0.0,
        coherent_with: None,
    }
}

/// Synthesise `(M, K, P)` snapshots with unit-power sources and noise at `snr_db`.
fn synth(
    pos: &[[f64; 3]],
    sources: &[Src],
    k: usize,
    p: usize,
    snr_db: f64,
    offsets: Option<&[Complex64]>,
    rng: &mut Rng,
) -> Snapshots {
    let m = pos.len();
    let sigma = 10f64.powf(-snr_db / 20.0);
    let mut data = Array3::<Complex64>::zeros((m, k, p));
    for pk in 0..p {
        for s in 0..k {
            let mut sym: Vec<Complex64> = Vec::new();
            for src in sources {
                sym.push(match src.coherent_with {
                    Some((i, g)) => g * sym[i],
                    None => rng.cgauss(),
                });
            }
            for a in 0..m {
                let mut v = Complex64::new(0.0, 0.0);
                for (i, src) in sources.iter().enumerate() {
                    v += plane_wave(pos[a], src.az, src.el) * sym[i];
                }
                if let Some(o) = offsets {
                    v *= o[a];
                }
                data[[a, s, pk]] = v + rng.cgauss() * sigma;
            }
        }
    }
    Snapshots::new(data).unwrap()
}

fn ang_err_deg(a: f64, b: f64) -> f64 {
    let mut d = (a - b).to_degrees() % 360.0;
    if d > 180.0 {
        d -= 360.0;
    }
    if d < -180.0 {
        d += 360.0;
    }
    d.abs()
}

fn sorted_az_deg(peaks: &[Peak]) -> Vec<f64> {
    let mut v: Vec<f64> = peaks.iter().map(|p| p.azimuth.to_degrees()).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

fn random_offsets(m: usize, rng: &mut Rng) -> Vec<Complex64> {
    (0..m)
        .map(|i| {
            if i == 0 {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::from_polar(0.8 + 0.4 * rng.uniform(), 2.0 * PI * rng.uniform() - PI)
            }
        })
        .collect()
}

// ---------------------------------------------------------------- geometry

#[test]
fn geometry_ula_ura_and_wavelength() {
    let g = ArrayGeometry::ula_half_wave(4, FC).unwrap();
    let lam = C0 / FC;
    assert!((g.wavelength() - lam).abs() < 1e-12);
    assert!((g.positions()[1][0] - 0.5 * lam).abs() < 1e-12);
    assert!(g.is_linear());
    let u = ArrayGeometry::ura_half_wave(4, 2, FC).unwrap();
    assert_eq!(u.n_antennas(), 8);
    assert_eq!(u.grid(), Some((4, 2)));
    assert!(!u.is_linear() && u.is_planar_xy());
    // Per-subcarrier wavelength differs slightly across a 20 MHz channel.
    assert!(g.wavelength_at(FC + 10e6) < g.wavelength_at(FC - 10e6));
    // Broadside source: all steering phases equal.
    let a = g.steering(PI / 2.0, 0.0);
    assert!(a.iter().all(|z| (z - a[0]).norm() < 1e-9));
    // Steering matches the independent first-principles generator.
    let a = g.steering(0.7, 0.2);
    for (i, p) in g.positions().iter().enumerate() {
        assert!((a[i] - plane_wave(*p, 0.7, 0.2)).norm() < 1e-9);
    }
}

#[test]
fn geometry_rejects_bad_input() {
    assert!(ArrayGeometry::ula(1, 0.06, FC).is_err());
    assert!(ArrayGeometry::ula(4, -1.0, FC).is_err());
    assert!(ArrayGeometry::ula(4, 0.06, 0.0).is_err());
    assert!(ArrayGeometry::new(vec![[0.0, 0.0, 0.0], [f64::NAN, 0.0, 0.0]], FC).is_err());
    let g = ArrayGeometry::new(vec![[0.0; 3], [0.1, 0.0, 0.0], [0.0, 0.13, 0.0]], FC).unwrap();
    assert!(g.reflection_permutation().is_err());
    assert!(g.subarray(2, 1).is_err());
}

// ---------------------------------------------------------------- estimators

/// SYNTHETIC: 8-element lambda/2 ULA, 1 source at 70 deg, SNR 10 dB,
/// 16 subcarriers x 8 packets = 128 snapshots.
#[test]
fn single_source_music_bartlett_mvdr() {
    let mut rng = Rng(1);
    let g = ArrayGeometry::ula_half_wave(8, FC).unwrap();
    let snap = synth(g.positions(), &[src(70.0)], 16, 8, 10.0, None, &mut rng);
    let cov = sample_covariance(&snap).unwrap();
    let grid = ScanGrid::default_for(&g, 0.5).unwrap();
    let truth = 70f64.to_radians();

    let mu = find_peaks(&music(&cov, &g, &grid, 1).unwrap(), 1, 0.0);
    let ba = find_peaks(&bartlett(&cov, &g, &grid).unwrap(), 1, 0.0);
    let mv = find_peaks(&mvdr(&cov, &g, &grid, 1e-3).unwrap(), 1, 0.0);
    let (e_mu, e_ba, e_mv) = (
        ang_err_deg(mu[0].azimuth, truth),
        ang_err_deg(ba[0].azimuth, truth),
        ang_err_deg(mv[0].azimuth, truth),
    );
    println!("single source errors deg: music {e_mu:.3} bartlett {e_ba:.3} mvdr {e_mv:.3}");
    assert!(e_mu < 0.5, "music {e_mu}");
    assert!(e_ba < 1.0, "bartlett {e_ba}");
    assert!(e_mv < 0.75, "mvdr {e_mv}");
}

/// SYNTHETIC: sub-grid refinement beats a coarse 5 deg grid.
#[test]
fn subgrid_refinement_beats_grid_step() {
    let mut rng = Rng(2);
    let g = ArrayGeometry::ula_half_wave(8, FC).unwrap();
    let snap = synth(g.positions(), &[src(63.7)], 16, 8, 20.0, None, &mut rng);
    let cov = sample_covariance(&snap).unwrap();
    let grid = ScanGrid::default_for(&g, 5.0).unwrap();
    let peaks = find_peaks(&music(&cov, &g, &grid, 1).unwrap(), 1, 0.0);
    let e = ang_err_deg(peaks[0].azimuth, 63.7f64.to_radians());
    println!("coarse-grid refined error {e:.3} deg (grid 5 deg)");
    assert!(e < 1.5, "refined error {e}");
}

/// SYNTHETIC: two uncorrelated sources 55/85 deg, SNR 15 dB, 128 snapshots.
#[test]
fn two_sources_with_mdl_count() {
    let mut rng = Rng(3);
    let g = ArrayGeometry::ula_half_wave(8, FC).unwrap();
    let snap = synth(g.positions(), &[src(55.0), src(85.0)], 16, 8, 15.0, None, &mut rng);
    let res = estimate_aoa(&snap, &g, &AoaConfig::default()).unwrap();
    assert_eq!(res.n_sources, 2, "MDL count");
    let az = sorted_az_deg(&res.peaks);
    println!("two-source MUSIC: {az:?}");
    assert!((az[0] - 55.0).abs() < 1.0 && (az[1] - 85.0).abs() < 1.0, "{az:?}");
}

#[test]
fn mdl_counts_noise_only_as_zero_and_aic_not_below() {
    let mut rng = Rng(4);
    let g = ArrayGeometry::ula_half_wave(6, FC).unwrap();
    let snap = synth(g.positions(), &[], 16, 8, 0.0, None, &mut rng);
    let cov = sample_covariance(&snap).unwrap();
    let eig = cov.eigen().unwrap();
    assert_eq!(
        estimate_num_sources(&eig.values, cov.n_snapshots, SourceCountCriterion::Mdl).unwrap(),
        0
    );
    for d in 1..=3usize {
        let srcs: Vec<Src> = (0..d).map(|i| src(40.0 + 35.0 * i as f64)).collect();
        let snap = synth(g.positions(), &srcs, 16, 8, 15.0, None, &mut rng);
        let cov = sample_covariance(&snap).unwrap();
        let eig = cov.eigen().unwrap();
        let n = estimate_num_sources(&eig.values, cov.n_snapshots, SourceCountCriterion::Mdl).unwrap();
        assert_eq!(n, d, "MDL for {d} sources");
    }
}

/// SYNTHETIC: fully coherent two-path (50 / 100 deg, second = 0.8 e^{j1} first),
/// 8-element ULA, SNR 20 dB, 128 snapshots. Plain MUSIC cannot resolve them;
/// spatial smoothing (5-element subarrays) + forward-backward averaging does.
#[test]
fn coherent_multipath_needs_smoothing() {
    let mut rng = Rng(5);
    let g = ArrayGeometry::ula_half_wave(8, FC).unwrap();
    let srcs = [
        src(50.0),
        Src {
            az: 100f64.to_radians(),
            el: 0.0,
            coherent_with: Some((0, Complex64::from_polar(0.8, 1.0))),
        },
    ];
    let snap = synth(g.positions(), &srcs, 16, 8, 20.0, None, &mut rng);
    let grid = ScanGrid::default_for(&g, 0.5).unwrap();

    // Plain covariance is rank-1 for coherent paths: second eigenvalue is noise-level.
    let cov = sample_covariance(&snap).unwrap();
    let ev = cov.eigen().unwrap().values;
    assert!(ev[1] / ev[0] < 0.02, "plain covariance should be ~rank 1: {ev:?}");
    let plain = find_peaks(&music(&cov, &g, &grid, 2).unwrap(), 2, 0.0);
    let plain_worst = sorted_az_deg(&plain)
        .iter()
        .zip([50.0, 100.0])
        .map(|(a, t)| (a - t).abs())
        .fold(0.0, f64::max);

    let cfg = AoaConfig {
        smoothing: Some(Smoothing { sub_x: 5, sub_y: 1 }),
        forward_backward: true,
        n_sources: Some(2),
        grid_step_deg: 0.5,
        ..Default::default()
    };
    let res = estimate_aoa(&snap, &g, &cfg).unwrap();
    let az = sorted_az_deg(&res.peaks);
    println!("coherent: plain MUSIC worst err {plain_worst:.2} deg; FB+SS peaks {az:?}");
    assert_eq!(az.len(), 2);
    assert!((az[0] - 50.0).abs() < 2.0 && (az[1] - 100.0).abs() < 2.0, "{az:?}");
    assert!(plain_worst > 2.0 * 2.0, "plain MUSIC unexpectedly fine: {plain_worst}");
}

/// SYNTHETIC: 2-D azimuth/elevation on a 4x4 URA and an ESPARGOS-like 2x4,
/// one source at az 40 / el 30, SNR 15 dB, 128 snapshots.
#[test]
fn planar_array_azimuth_elevation() {
    for (nx, ny, tol_az, tol_el) in [(4usize, 4usize, 1.0, 1.5), (4, 2, 1.5, 3.0)] {
        let mut rng = Rng(6 + nx as u64 + ny as u64);
        let g = ArrayGeometry::ura_half_wave(nx, ny, FC).unwrap();
        let s = Src {
            az: 40f64.to_radians(),
            el: 30f64.to_radians(),
            coherent_with: None,
        };
        let snap = synth(g.positions(), &[s], 16, 8, 15.0, None, &mut rng);
        let cfg = AoaConfig {
            n_sources: Some(1),
            grid_step_deg: 1.0,
            ..Default::default()
        };
        let res = estimate_aoa(&snap, &g, &cfg).unwrap();
        let p = res.peaks[0];
        let (ea, ee) = (
            ang_err_deg(p.azimuth, 40f64.to_radians()),
            ang_err_deg(p.elevation, 30f64.to_radians()),
        );
        println!("{nx}x{ny} URA: az err {ea:.2} deg, el err {ee:.2} deg");
        assert!(ea < tol_az && ee < tol_el, "{nx}x{ny}: az {ea} el {ee}");
    }
}

/// SYNTHETIC: SNR sweep, 1 source, 8-element ULA, 128 snapshots; RMS error over 20 trials.
#[test]
fn snr_sweep_rms_error() {
    let g = ArrayGeometry::ula_half_wave(8, FC).unwrap();
    let grid = ScanGrid::default_for(&g, 1.0).unwrap();
    for (snr, bound) in [(0.0, 3.0), (10.0, 1.0)] {
        let mut sq = 0.0;
        for t in 0..20u64 {
            let mut rng = Rng(100 + t);
            let truth = 30.0 + 6.0 * t as f64;
            let snap = synth(g.positions(), &[src(truth)], 16, 8, snr, None, &mut rng);
            let cov = sample_covariance(&snap).unwrap();
            let pk = find_peaks(&music(&cov, &g, &grid, 1).unwrap(), 1, 0.0);
            let e = ang_err_deg(pk[0].azimuth, truth.to_radians());
            sq += e * e;
        }
        let rms = (sq / 20.0).sqrt();
        println!("SNR {snr} dB: RMS error {rms:.3} deg over 20 trials");
        assert!(rms < bound, "SNR {snr}: rms {rms}");
    }
}

// ---------------------------------------------------------------- calibration

/// SYNTHETIC: random offsets (phase uniform, gain 0.8-1.2) recovered from a
/// shared-reference capture (SNR 25 dB, 64 snapshots); then AoA works after
/// applying them and does not before.
#[test]
fn calibration_common_reference_recovers_offsets() {
    let mut rng = Rng(7);
    let m = 8;
    let g = ArrayGeometry::ula_half_wave(m, FC).unwrap();
    let offs = random_offsets(m, &mut rng);

    // Reference capture: the same waveform on every antenna (zero phase).
    let sigma = 10f64.powf(-25.0 / 20.0);
    let mut data = Array3::<Complex64>::zeros((m, 8, 8));
    for p in 0..8 {
        for k in 0..8 {
            let s = rng.cgauss();
            for a in 0..m {
                data[[a, k, p]] = offs[a] * s + rng.cgauss() * sigma;
            }
        }
    }
    let reference = Snapshots::new(data).unwrap();
    let cal = Calibration::estimate(&reference, Reference::Common, false).unwrap();
    let est = cal.antenna_offsets();
    let mean_mag = offs.iter().map(|o| o.norm()).sum::<f64>() / m as f64;
    let mut worst_phase: f64 = 0.0;
    let mut worst_gain: f64 = 0.0;
    for a in 0..m {
        let truth = offs[a] / offs[0] * (offs[0].norm() / mean_mag); // phase rel. antenna 0, unit-mean gain
        worst_phase = worst_phase.max((est[a] * truth.conj()).arg().abs());
        worst_gain = worst_gain.max((est[a].norm() - offs[a].norm() / mean_mag).abs());
    }
    println!("calibration: worst phase err {worst_phase:.4} rad, gain err {worst_gain:.4}");
    assert!(worst_phase < 0.03 && worst_gain < 0.05);

    let snap = synth(g.positions(), &[src(65.0)], 16, 8, 15.0, Some(&offs), &mut rng);
    let cfg = AoaConfig {
        n_sources: Some(1),
        ..Default::default()
    };
    let raw = estimate_aoa(&snap, &g, &cfg).unwrap();
    let fixed = estimate_aoa(&cal.apply(&snap).unwrap(), &g, &cfg).unwrap();
    let (e_raw, e_fix) = (
        ang_err_deg(raw.peaks[0].azimuth, 65f64.to_radians()),
        ang_err_deg(fixed.peaks[0].azimuth, 65f64.to_radians()),
    );
    println!("uncalibrated err {e_raw:.2} deg, calibrated err {e_fix:.2} deg");
    assert!(e_fix < 1.0 && e_raw > 3.0 * e_fix);
}

/// SYNTHETIC: calibration from a known-bearing source (30 deg), then a source at 110 deg.
#[test]
fn calibration_known_bearing() {
    let mut rng = Rng(8);
    let m = 6;
    let g = ArrayGeometry::ula_half_wave(m, FC).unwrap();
    let offs = random_offsets(m, &mut rng);
    let refcap = synth(g.positions(), &[src(30.0)], 8, 8, 25.0, Some(&offs), &mut rng);
    let cal = Calibration::estimate(
        &refcap,
        Reference::Bearing {
            geometry: &g,
            azimuth: 30f64.to_radians(),
            elevation: 0.0,
        },
        false,
    )
    .unwrap();
    let snap = synth(g.positions(), &[src(110.0)], 16, 8, 15.0, Some(&offs), &mut rng);
    let cfg = AoaConfig {
        n_sources: Some(1),
        ..Default::default()
    };
    let r = estimate_aoa(&cal.apply(&snap).unwrap(), &g, &cfg).unwrap();
    let e = ang_err_deg(r.peaks[0].azimuth, 110f64.to_radians());
    println!("known-bearing calibration: AoA err {e:.2} deg");
    assert!(e < 1.5);
}

/// SYNTHETIC: per-subcarrier calibration handles a cable delay (phase ramp over subcarriers).
#[test]
fn calibration_per_subcarrier_ramp() {
    let mut rng = Rng(9);
    let (m, k) = (4, 16);
    let mut data = Array3::<Complex64>::zeros((m, k, 16));
    let delay = |a: usize, s: usize| Complex64::from_polar(1.0, 0.3 * a as f64 + 0.15 * (a * s) as f64);
    for p in 0..16 {
        for s in 0..k {
            let sym = rng.cgauss();
            for a in 0..m {
                data[[a, s, p]] = delay(a, s) * sym;
            }
        }
    }
    let snap = Snapshots::new(data).unwrap();
    let cal = Calibration::estimate(&snap, Reference::Common, true).unwrap();
    assert_eq!(cal.offsets().dim(), (m, k));
    let fixed = cal.apply(&snap).unwrap();
    // After calibration every antenna sees the same phase on each subcarrier.
    for s in 0..k {
        let r = fixed.data()[[0, s, 3]];
        for a in 1..m {
            assert!((fixed.data()[[a, s, 3]] / r - Complex64::new(1.0, 0.0)).norm() < 1e-6);
        }
    }
    // A 1-column calibration cannot fix this; mismatched subcarrier count is an error.
    let bad = Snapshots::new(Array3::from_elem((m, k + 1, 2), Complex64::new(1.0, 0.0))).unwrap();
    assert!(matches!(cal.apply(&bad), Err(AoaError::ShapeMismatch(_))));
}

/// SYNTHETIC, noiseless: random per-packet common phase + slope removed identically across antennas.
#[test]
fn common_phase_removal_is_antenna_consistent() {
    let mut rng = Rng(10);
    let (m, k, p) = (4, 32, 6);
    let g = ArrayGeometry::ula_half_wave(m, FC).unwrap();
    let mut data = Array3::<Complex64>::zeros((m, k, p));
    for pk in 0..p {
        let phi = 2.0 * PI * rng.uniform();
        let slope = 0.4 * (rng.uniform() - 0.5);
        for s in 0..k {
            for a in 0..m {
                let ang = phi + slope * s as f64;
                data[[a, s, pk]] = plane_wave(g.positions()[a], 1.0, 0.0) * Complex64::from_polar(1.0, ang);
            }
        }
    }
    let snap = Snapshots::new(data).unwrap();
    let out = remove_common_phase(&snap, None).unwrap();
    for pk in 1..p {
        for s in 0..k {
            for a in 0..m {
                let d = (out.data()[[a, s, pk]] - out.data()[[a, s, 0]]).norm();
                assert!(d < 1e-9, "packet {pk} sub {s} ant {a}: {d}");
            }
        }
    }
    // Inter-antenna phase (the AoA information) is preserved.
    let rel = out.data()[[1, 5, 0]] / out.data()[[0, 5, 0]];
    let want = plane_wave(g.positions()[1], 1.0, 0.0) / plane_wave(g.positions()[0], 1.0, 0.0);
    assert!((rel - want).norm() < 1e-9);
    assert!(remove_common_phase(&snap, Some(&[0.0, 1.0])).is_err());
}

// ---------------------------------------------------------------- covariance

#[test]
fn forward_backward_is_hermitian_persymmetric_and_errors_on_asymmetric() {
    let mut rng = Rng(11);
    let g = ArrayGeometry::ula_half_wave(6, FC).unwrap();
    let snap = synth(g.positions(), &[src(60.0)], 8, 8, 10.0, None, &mut rng);
    let cov = sample_covariance(&snap).unwrap();
    let fb = forward_backward(&cov, &g).unwrap();
    for i in 0..6 {
        for j in 0..6 {
            assert!((fb.matrix[[i, j]] - fb.matrix[[j, i]].conj()).norm() < 1e-12);
            assert!((fb.matrix[[i, j]] - fb.matrix[[5 - i, 5 - j]].conj()).norm() < 1e-12);
        }
    }
    let asym = ArrayGeometry::new(vec![[0.0; 3], [0.05, 0.0, 0.0], [0.2, 0.1, 0.0]], FC).unwrap();
    let snap3 = synth(asym.positions(), &[src(60.0)], 8, 8, 10.0, None, &mut rng);
    let cov3 = sample_covariance(&snap3).unwrap();
    assert!(matches!(
        forward_backward(&cov3, &asym),
        Err(AoaError::UnsupportedGeometry(_))
    ));
    assert!(matches!(
        spatial_smoothing(&cov3, &asym, Smoothing { sub_x: 2, sub_y: 1 }),
        Err(AoaError::UnsupportedGeometry(_))
    ));
}

// ---------------------------------------------------------------- fusion

/// SYNTHETIC: two 8-element ULAs at (0,0) and (6,0) each estimate a bearing
/// to a source at (2,3) (SNR 15 dB, 128 snapshots); fused position error is bounded.
#[test]
fn fusion_two_arrays_position() {
    let g = ArrayGeometry::ula_half_wave(8, FC).unwrap();
    let truth = [2.0, 3.0];
    let poses = [
        ArrayPose { x: 0.0, y: 0.0, yaw: 0.0 },
        ArrayPose { x: 6.0, y: 0.0, yaw: 0.0 },
    ];
    let mut bearings = Vec::new();
    for (i, pose) in poses.iter().enumerate() {
        let mut rng = Rng(200 + i as u64);
        let local = (truth[1] - pose.y).atan2(truth[0] - pose.x) - pose.yaw;
        let s = Src { az: local, el: 0.0, coherent_with: None };
        let snap = synth(g.positions(), &[s], 16, 8, 15.0, None, &mut rng);
        let cfg = AoaConfig { n_sources: Some(1), grid_step_deg: 0.5, ..Default::default() };
        let r = estimate_aoa(&snap, &g, &cfg).unwrap();
        bearings.push(Bearing::from_local(*pose, r.peaks[0].azimuth, 1f64.to_radians()));
    }
    let fix = locate_2d(&bearings).unwrap();
    let err = ((fix.position[0] - truth[0]).powi(2) + (fix.position[1] - truth[1]).powi(2)).sqrt();
    println!(
        "fusion: pos ({:.3}, {:.3}) err {err:.3} m, residual {:.4} m, sigma {:.3} m",
        fix.position[0], fix.position[1], fix.residual_rms_m, fix.uncertainty_m()
    );
    assert!(err < 0.1, "position error {err}");
    assert_eq!(fix.n_behind, 0);
    assert!(fix.uncertainty_m() > 0.0 && fix.residual_rms_m < 0.1);
}

#[test]
fn fusion_exact_rays_and_rotated_pose_and_errors() {
    let truth = [1.5, -2.0];
    let poses = [
        ArrayPose { x: 0.0, y: 0.0, yaw: 0.0 },
        ArrayPose { x: 5.0, y: 1.0, yaw: 0.8 },
        ArrayPose { x: -3.0, y: -4.0, yaw: -1.2 },
    ];
    let bs: Vec<Bearing> = poses
        .iter()
        .map(|p| {
            let world = (truth[1] - p.y).atan2(truth[0] - p.x);
            Bearing::from_local(*p, world - p.yaw, 0.02)
        })
        .collect();
    let fix = locate_2d(&bs).unwrap();
    assert!((fix.position[0] - truth[0]).abs() < 1e-9 && (fix.position[1] - truth[1]).abs() < 1e-9);
    assert!(fix.residual_rms_m < 1e-9);

    let parallel = [
        Bearing { origin: [0.0, 0.0], azimuth: 0.5, sigma_rad: 0.02 },
        Bearing { origin: [0.0, 1.0], azimuth: 0.5, sigma_rad: 0.02 },
    ];
    assert!(locate_2d(&parallel).is_err());
    assert!(locate_2d(&bs[..1]).is_err());
    let nan = [bs[0], Bearing { azimuth: f64::NAN, ..bs[1] }];
    assert!(matches!(locate_2d(&nan), Err(AoaError::NonFinite(_))));
    let zero_sigma = [bs[0], Bearing { sigma_rad: 0.0, ..bs[1] }];
    assert!(locate_2d(&zero_sigma).is_err());
}

// ---------------------------------------------------------------- SpotFi

/// SYNTHETIC: 3-antenna ULA, 30 subcarriers at 312.5 kHz, two paths
/// (60 deg / 20 ns, 120 deg / 180 ns), SNR 25 dB, 20 packets with
/// independent complex path gains.
#[test]
fn spotfi_joint_aoa_tof() {
    let mut rng = Rng(12);
    let m = 3;
    let g = ArrayGeometry::ula_half_wave(m, FC).unwrap();
    let (k, p) = (30, 20);
    let df = 312.5e3;
    let paths = [(60f64.to_radians(), 20e-9), (120f64.to_radians(), 180e-9)];
    let sigma = 10f64.powf(-25.0 / 20.0);
    let mut data = Array3::<Complex64>::zeros((m, k, p));
    for pk in 0..p {
        let gains = [rng.cgauss(), rng.cgauss()];
        for s in 0..k {
            for a in 0..m {
                let mut v = Complex64::new(0.0, 0.0);
                for (i, (az, tau)) in paths.iter().enumerate() {
                    v += gains[i]
                        * plane_wave(g.positions()[a], *az, 0.0)
                        * Complex64::from_polar(1.0, -2.0 * PI * df * s as f64 * tau);
                }
                data[[a, s, pk]] = v + rng.cgauss() * sigma;
            }
        }
    }
    let snap = Snapshots::new(data).unwrap();
    let cfg = SpotFiConfig {
        window: 15,
        n_paths: 2,
        subcarrier_spacing_hz: Some(df),
        azimuth: (0.0, PI, 181),
        tof: (0.0, 300e-9, 151),
    };
    let (_, peaks) = spotfi_music(&snap, &g, &cfg).unwrap();
    assert_eq!(peaks.len(), 2);
    let mut found = peaks.clone();
    found.sort_by(|a, b| a.azimuth.partial_cmp(&b.azimuth).unwrap());
    println!("spotfi peaks: {:?}", found.iter().map(|p| (p.azimuth.to_degrees(), p.tof_s * 1e9)).collect::<Vec<_>>());
    for (f, (az, tau)) in found.iter().zip(paths) {
        assert!(ang_err_deg(f.azimuth, az) < 3.0, "az {} vs {}", f.azimuth.to_degrees(), az.to_degrees());
        assert!((f.tof_s - tau).abs() < 15e-9, "tof {} vs {}", f.tof_s, tau);
    }
    let bad = SpotFiConfig { window: 1, ..cfg.clone() };
    assert!(spotfi_music(&snap, &g, &bad).is_err());
}

// ---------------------------------------------------------------- validation & adapters

#[test]
fn input_validation_returns_errors() {
    // NaN
    let mut d = Array3::from_elem((4, 4, 4), Complex64::new(1.0, 0.0));
    d[[1, 1, 1]] = Complex64::new(f64::NAN, 0.0);
    assert!(matches!(Snapshots::new(d), Err(AoaError::NonFinite(_))));
    // empty
    assert!(Snapshots::new(Array3::<Complex64>::zeros((0, 4, 4))).is_err());
    // too few snapshots
    let g = ArrayGeometry::ula_half_wave(8, FC).unwrap();
    let few = Snapshots::new(Array3::from_elem((8, 2, 2), Complex64::new(1.0, 0.1))).unwrap();
    assert!(matches!(sample_covariance(&few), Err(AoaError::InsufficientData(_))));
    assert!(estimate_aoa(&few, &g, &AoaConfig::default()).is_err());
    // antenna mismatch
    let mut rng = Rng(13);
    let g4 = ArrayGeometry::ula_half_wave(4, FC).unwrap();
    let snap = synth(g4.positions(), &[src(60.0)], 8, 4, 10.0, None, &mut rng);
    assert!(matches!(estimate_aoa(&snap, &g, &AoaConfig::default()), Err(AoaError::ShapeMismatch(_))));
    // bad source counts and grids
    let cov = sample_covariance(&snap).unwrap();
    let grid = ScanGrid::default_for(&g4, 1.0).unwrap();
    assert!(music(&cov, &g4, &grid, 0).is_err());
    assert!(music(&cov, &g4, &grid, 4).is_err());
    assert!(music(&cov, &g, &grid, 1).is_err());
    assert!(ScanGrid::azimuth_range(1.0, 0.5, 10).is_err());
    assert!(estimate_num_sources(&[1.0, f64::NAN], 10, SourceCountCriterion::Mdl).is_err());
    assert!(estimate_num_sources(&[1.0, 0.5], 1, SourceCountCriterion::Mdl).is_err());
    assert!(snap.clone().with_subcarrier_freqs(vec![1.0]).is_err());
    // Single-antenna (e.g. ESP32-S3) data cannot form an array.
    assert!(ArrayGeometry::new(vec![[0.0; 3]], FC).is_err());
}

#[test]
fn adapters_build_snapshots() {
    // Antenna-major I/Q: antenna 0 = 1+0j, antenna 1 = 0+2j for 3 subcarriers.
    let f: Vec<(i16, i16)> = vec![(1, 0), (1, 0), (1, 0), (0, 2), (0, 2), (0, 2)];
    let s = snapshots_from_iq_frames(2, &[&f, &f]).unwrap();
    assert_eq!((s.n_antennas(), s.n_subcarriers(), s.n_packets()), (2, 3, 2));
    assert_eq!(s.data()[[1, 2, 1]], Complex64::new(0.0, 2.0));
    assert!(snapshots_from_iq_frames(4, &[&f]).is_err());
    assert!(snapshots_from_iq_frames(2, &[&f, &f[..4]]).is_err());
    assert!(snapshots_from_iq_frames(2, &[]).is_err());

    use crate::csi_processor::CsiData;
    let amp = Array2::from_elem((2, 3), 2.0);
    let ph = Array2::from_elem((2, 3), PI / 2.0);
    let frame = CsiData::builder()
        .amplitude(amp)
        .phase(ph)
        .frequency(FC)
        .bandwidth(20e6)
        .snr(20.0)
        .build()
        .unwrap();
    let s = snapshots_from_csi_data(&[frame.clone(), frame]).unwrap();
    assert!((s.data()[[0, 0, 0]] - Complex64::new(0.0, 2.0)).norm() < 1e-12);
}
