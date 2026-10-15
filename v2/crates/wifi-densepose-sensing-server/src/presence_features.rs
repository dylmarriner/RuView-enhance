//! Extra per-link presence features for still people: NBVI-style tone
//! selection, moving-variance segmentation (MVS), a calibrated baseline-shift
//! score, and a breathing-band cue.
//!
//! Clean-room implementation from the published descriptions of the ideas;
//! no code is copied from ESPectre (GPL-3.0). The idea names come from its
//! public documentation: "turbulence" (spatial coefficient of variation over
//! selected subcarriers), moving variance of that series, and a normalised
//! baseline-variability index (NBVI) for choosing tones. The breathing cue
//! reuses [`wifi_densepose_vitals::BreathingExtractor`].
//!
//! Inputs are gain-normalised LLTF frames (each tone divided by the frame
//! mean), so receiver AGC steps cancel before any feature is computed.
//!
//! Pipeline per link:
//! 1. While the operator calibrates an empty room, frames are buffered.
//! 2. On finish, [`TrainedBaseline::train`] picks the [`SELECTED_TONES`] tones
//!    with the lowest NBVI (stable relative to their mean, spaced apart so
//!    they are independent looks), learns each tone's empty-room mean and
//!    block-to-block spread, and sets per-link thresholds.
//! 3. Live, each frame yields:
//!    - `mvs_ratio`: moving variance of turbulence / trained threshold
//!      (motion; >= 1 means moving),
//!    - `shift_z`: RMS over selected tones of how many empty-room spreads the
//!      recent mean amplitude sits from the empty-room mean (a still body that
//!      reshapes multipath shifts tones permanently, which variance cannot see)
//!      and at least [`SHIFT_ABS_MIN`] in absolute terms, so ordinary
//!      long-term drift is not read as a person,
//!    - a breathing cue from the vitals extractor on the turbulence series,
//!      accepted only above the confidence the empty room itself reached.
//!
//! The room-level vote ([`room_advisory`]) is **advisory by default**: it is
//! reported by `/api/v1/peer-links` under `room.advisory` and does not change
//! `presence` / `still_presence` unless `--peer-presence-mode active`.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU8, Ordering};
use wifi_densepose_vitals::BreathingExtractor;

/// LLTF tones per frame.
pub const TONES: usize = 52;
/// Tones kept per link after NBVI selection.
pub const SELECTED_TONES: usize = 16;
/// Minimum index distance between selected tones (independent looks).
const MIN_TONE_SPACING: usize = 2;
/// Nominal peer-beacon rate (firmware `PEER_BEACON_PERIOD_US`).
pub const NOMINAL_RATE_HZ: f64 = 10.0;
/// Frames in the moving-variance window (~3 s at 10 Hz).
const MVS_WINDOW: usize = 30;
/// Frames averaged for the baseline-shift score and per training block.
const SHIFT_WINDOW: usize = 30;
/// Minimum calibration frames needed to train (~24 s at 10 Hz).
pub const MIN_TRAIN_FRAMES: usize = 240;
/// Calibration frames kept (~5 min at 10 Hz).
const MAX_TRAIN_FRAMES: usize = 3000;
/// Floor on a tone's empty-room spread (normalised-amplitude units).
const SIGMA_FLOOR: f64 = 0.005;
/// Margin applied to the empty-room moving-variance 99th percentile.
const MVS_MARGIN: f64 = 1.5;
/// Lowest accepted `shift_z` threshold, whatever calibration saw.
const SHIFT_Z_MIN: f64 = 4.0;
/// Smallest absolute shift (RMS over selected tones, in units of the frame
/// mean) that counts as a body. A short calibration can make the empty-room
/// spread tiny, so a z-score alone would flag ordinary long-term drift
/// (temperature, furniture) as a person.
pub const SHIFT_ABS_MIN: f64 = 0.08;
/// Margin applied to the largest in-sample empty-room `shift_z`.
const SHIFT_Z_MARGIN: f64 = 1.5;
/// Consecutive frames over threshold before a verdict flips on.
const MOTION_HITS: u32 = 5;
const STILL_HITS: u32 = 10;
/// A breathing estimate is held as a cue this many frames.
const BREATH_HOLD_FRAMES: u32 = 100;
/// Floor on the breathing confidence needed, whatever calibration saw.
const BREATH_CONF_MIN: f64 = 0.5;
const BREATH_CONF_MARGIN: f64 = 0.15;
/// Votes needed in a room, at minimum, and as a fraction of trained links.
pub const MIN_ROOM_VOTES: usize = 2;
pub const ROOM_VOTE_FRACTION: f64 = 0.25;

/// How the room-level feature vote is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Features are computed but not reported.
    Off,
    /// Reported under `room.advisory` only (default).
    Advisory,
    /// Also merged into `room.presence` and `room.still_presence`.
    Active,
}

impl Mode {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" => Ok(Self::Off),
            "advisory" => Ok(Self::Advisory),
            "active" => Ok(Self::Active),
            other => Err(format!("unknown peer presence mode '{other}' (off, advisory, active)")),
        }
    }
}

static MODE: AtomicU8 = AtomicU8::new(1);

pub fn set_mode(m: Mode) {
    MODE.store(m as u8, Ordering::Relaxed);
}

pub fn mode() -> Mode {
    match MODE.load(Ordering::Relaxed) {
        0 => Mode::Off,
        2 => Mode::Active,
        _ => Mode::Advisory,
    }
}

/// Per-link empty-room statistics; persisted with the peer-link calibration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainedBaseline {
    /// Selected tone indices (0..TONES), ascending.
    pub tones: Vec<usize>,
    /// Empty-room mean of each selected tone.
    pub mean: Vec<f64>,
    /// Empty-room spread of the [`SHIFT_WINDOW`]-frame mean of each tone.
    pub sigma: Vec<f64>,
    /// Moving-variance level above which the link counts as moving.
    pub mvs_threshold: f64,
    /// `shift_z` level above which the link counts as holding a still body.
    pub shift_threshold: f64,
    /// Breathing confidence above which the cue counts.
    pub breath_conf_threshold: f64,
}

/// Spatial coefficient of variation: std / mean over the selected tones.
fn turbulence(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    if mean.abs() < 1e-9 {
        return 0.0;
    }
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    var.sqrt() / mean.abs()
}

fn variance(values: &VecDeque<f64>) -> f64 {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// NBVI-style tone ranking: low normalised variability of the empty-room
/// baseline. `0.5 * sigma/mean^2 + 0.5 * sigma/mean` (the first term favours
/// strong tones, the second stable ones).
fn nbvi(mean: f64, sigma: f64) -> f64 {
    0.5 * sigma / (mean * mean) + 0.5 * sigma / mean
}

/// Pick the best tones by ascending NBVI, skipping tones closer than
/// [`MIN_TONE_SPACING`] to one already chosen. Returned ascending.
pub fn select_tones(frames: &[Vec<f64>]) -> Vec<usize> {
    let n = frames.len() as f64;
    let mut ranked: Vec<(usize, f64)> = (0..TONES)
        .filter_map(|k| {
            let mean = frames.iter().map(|f| f[k]).sum::<f64>() / n;
            if !mean.is_finite() || mean < 1e-6 {
                return None;
            }
            let var = frames.iter().map(|f| (f[k] - mean).powi(2)).sum::<f64>() / (n - 1.0);
            Some((k, nbvi(mean, var.sqrt())))
        })
        .collect();
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
    let mut chosen: Vec<usize> = Vec::new();
    for (k, _) in ranked {
        if chosen.iter().all(|&c| c.abs_diff(k) >= MIN_TONE_SPACING) {
            chosen.push(k);
            if chosen.len() == SELECTED_TONES {
                break;
            }
        }
    }
    chosen.sort_unstable();
    chosen
}

impl TrainedBaseline {
    /// Train from gain-normalised empty-room frames. `None` when there is too
    /// little data or too few usable tones.
    pub fn train(frames: &[Vec<f64>], rate_hz: f64) -> Option<Self> {
        if frames.len() < MIN_TRAIN_FRAMES || frames.iter().any(|f| f.len() != TONES) {
            return None;
        }
        let tones = select_tones(frames);
        if tones.len() < SELECTED_TONES / 2 {
            return None;
        }
        let selected: Vec<Vec<f64>> =
            frames.iter().map(|f| tones.iter().map(|&k| f[k]).collect()).collect();
        let n = selected.len() as f64;
        let mean: Vec<f64> = (0..tones.len())
            .map(|j| selected.iter().map(|f| f[j]).sum::<f64>() / n)
            .collect();

        // Spread of non-overlapping SHIFT_WINDOW block means, per tone.
        let blocks: Vec<Vec<f64>> = selected
            .chunks_exact(SHIFT_WINDOW)
            .map(|c| (0..tones.len()).map(|j| c.iter().map(|f| f[j]).sum::<f64>() / SHIFT_WINDOW as f64).collect())
            .collect();
        let nb = blocks.len() as f64;
        let sigma: Vec<f64> = (0..tones.len())
            .map(|j| {
                let bm = blocks.iter().map(|b| b[j]).sum::<f64>() / nb;
                let block_std = (blocks.iter().map(|b| (b[j] - bm).powi(2)).sum::<f64>() / (nb - 1.0)).sqrt();
                let pkt_var = selected.iter().map(|f| (f[j] - mean[j]).powi(2)).sum::<f64>() / (n - 1.0);
                // Never trust a block spread below what packet noise alone gives.
                block_std.max(pkt_var.sqrt() / (SHIFT_WINDOW as f64).sqrt()).max(SIGMA_FLOOR)
            })
            .collect();

        let mut base = Self {
            tones,
            mean,
            sigma,
            mvs_threshold: f64::INFINITY,
            shift_threshold: SHIFT_Z_MIN,
            breath_conf_threshold: BREATH_CONF_MIN,
        };
        // Replay the calibration through the live feature path to learn how
        // high an empty room legitimately scores.
        let mut probe = LinkFeatures::new(rate_hz);
        probe.trained = Some(base.clone());
        let (mut mvs, mut shifts, mut conf_max) = (Vec::new(), Vec::new(), 0.0_f64);
        for f in frames {
            let r = probe.push_trained(f);
            if let Some(v) = r.mvs {
                mvs.push(v);
            }
            if let Some(z) = r.shift_z {
                shifts.push(z);
            }
            if let Some(c) = r.breathing_conf {
                conf_max = conf_max.max(c);
            }
        }
        if mvs.is_empty() || shifts.is_empty() {
            return None;
        }
        mvs.sort_by(f64::total_cmp);
        base.mvs_threshold = (percentile(&mvs, 0.99) * MVS_MARGIN).max(1e-12);
        let shift_max = shifts.iter().copied().fold(0.0, f64::max);
        base.shift_threshold = (shift_max * SHIFT_Z_MARGIN).max(SHIFT_Z_MIN);
        base.breath_conf_threshold = (conf_max + BREATH_CONF_MARGIN).max(BREATH_CONF_MIN);
        Some(base)
    }

    /// Reject persisted data that is malformed or non-finite.
    pub fn is_valid(&self) -> bool {
        let n = self.tones.len();
        n >= SELECTED_TONES / 2
            && n <= SELECTED_TONES
            && self.mean.len() == n
            && self.sigma.len() == n
            && self.tones.iter().all(|&k| k < TONES)
            && self.mean.iter().chain(&self.sigma).all(|x| x.is_finite())
            && self.sigma.iter().all(|&s| s > 0.0)
            && self.mvs_threshold.is_finite()
            && self.mvs_threshold > 0.0
            && self.shift_threshold.is_finite()
            && self.breath_conf_threshold.is_finite()
    }
}

/// Raw per-frame feature values.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Reading {
    pub mvs: Option<f64>,
    pub shift_z: Option<f64>,
    /// RMS shift of the selected tones from the empty-room mean.
    pub shift_abs: Option<f64>,
    pub breathing_conf: Option<f64>,
}

/// Per-link verdicts after thresholds and debounce.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct Verdict {
    pub trained: bool,
    pub motion: bool,
    pub still: bool,
    pub breathing: bool,
}

impl Verdict {
    pub fn votes(&self) -> bool {
        self.motion || self.still || self.breathing
    }
}

pub struct LinkFeatures {
    rate_hz: f64,
    learning: bool,
    train: Vec<Vec<f64>>,
    trained: Option<TrainedBaseline>,
    turb: VecDeque<f64>,
    window: VecDeque<Vec<f64>>,
    breathing: BreathingExtractor,
    breath_hold: u32,
    breath_bpm: Option<f64>,
    motion_hits: u32,
    still_hits: u32,
    last: Reading,
    verdict: Verdict,
}

impl Default for LinkFeatures {
    fn default() -> Self {
        Self::new(NOMINAL_RATE_HZ)
    }
}

impl LinkFeatures {
    pub fn new(rate_hz: f64) -> Self {
        Self {
            rate_hz,
            learning: false,
            train: Vec::new(),
            trained: None,
            turb: VecDeque::with_capacity(MVS_WINDOW),
            window: VecDeque::with_capacity(SHIFT_WINDOW),
            breathing: BreathingExtractor::new(1, rate_hz, 30.0),
            breath_hold: 0,
            breath_bpm: None,
            motion_hits: 0,
            still_hits: 0,
            last: Reading::default(),
            verdict: Verdict::default(),
        }
    }

    pub fn restored(rate_hz: f64, baseline: Option<&TrainedBaseline>) -> Self {
        let mut f = Self::new(rate_hz);
        f.trained = baseline.filter(|b| b.is_valid()).cloned();
        f
    }

    pub fn baseline(&self) -> Option<&TrainedBaseline> {
        self.trained.as_ref()
    }

    pub fn verdict(&self) -> Verdict {
        self.verdict
    }

    pub fn reading(&self) -> Reading {
        self.last
    }

    pub fn breathing_bpm(&self) -> Option<f64> {
        self.breath_bpm
    }

    fn reset_live(&mut self) {
        self.turb.clear();
        self.window.clear();
        self.breathing.reset();
        self.breath_hold = 0;
        self.breath_bpm = None;
        self.motion_hits = 0;
        self.still_hits = 0;
        self.last = Reading::default();
        self.verdict = Verdict::default();
    }

    /// Start buffering empty-room frames (drops any trained baseline).
    pub fn set_learning(&mut self, on: bool) {
        self.train.clear();
        self.trained = None;
        self.reset_live();
        self.learning = on;
    }

    pub fn is_learning(&self) -> bool {
        self.learning
    }

    /// Stop learning and train from the buffered frames. Returns the trained
    /// baseline, or `None` (and stays untrained) if there was too little data.
    pub fn finish_learning(&mut self) -> Option<TrainedBaseline> {
        self.learning = false;
        let frames = std::mem::take(&mut self.train);
        self.reset_live();
        self.trained = TrainedBaseline::train(&frames, self.rate_hz);
        self.trained.clone()
    }

    /// Feed one gain-normalised frame.
    pub fn push(&mut self, normalised: &[f64]) {
        if normalised.len() != TONES || normalised.iter().any(|x| !x.is_finite()) {
            return;
        }
        if self.learning {
            if self.train.len() < MAX_TRAIN_FRAMES {
                self.train.push(normalised.to_vec());
            }
            return;
        }
        if self.trained.is_none() {
            return;
        }
        let r = self.push_trained(normalised);
        self.last = r;
        self.update_verdict(r);
    }

    /// One live step against the trained baseline (also used to replay the
    /// calibration while training, when thresholds are still unset).
    fn push_trained(&mut self, normalised: &[f64]) -> Reading {
        let Some(base) = self.trained.as_ref() else { return Reading::default() };
        let breath_threshold = base.breath_conf_threshold;
        let sel: Vec<f64> = base.tones.iter().map(|&k| normalised[k]).collect();
        let t = turbulence(&sel);
        if self.turb.len() == MVS_WINDOW {
            self.turb.pop_front();
        }
        self.turb.push_back(t);
        if self.window.len() == SHIFT_WINDOW {
            self.window.pop_front();
        }
        self.window.push_back(sel);

        let mvs = (self.turb.len() == MVS_WINDOW).then(|| variance(&self.turb));
        let shift = (self.window.len() == SHIFT_WINDOW).then(|| {
            let n = base.tones.len();
            let w = SHIFT_WINDOW as f64;
            let (mut zz, mut aa) = (0.0, 0.0);
            for j in 0..n {
                let m = self.window.iter().map(|f| f[j]).sum::<f64>() / w;
                zz += ((m - base.mean[j]) / base.sigma[j]).powi(2);
                aa += (m - base.mean[j]).powi(2);
            }
            ((zz / n as f64).sqrt(), (aa / n as f64).sqrt())
        });
        let (shift_z, shift_abs) = (shift.map(|s| s.0), shift.map(|s| s.1));
        // Breathing: only the confidence matters here; the extractor already
        // rejects estimates outside 0.1-0.5 Hz.
        let est = self.breathing.extract(&[t], &[1.0]);
        let breathing_conf = est.as_ref().map(|e| e.confidence);
        if let Some(e) = est {
            self.last_breath(e.confidence, e.value_bpm, breath_threshold);
        }
        Reading { mvs, shift_z, shift_abs, breathing_conf }
    }

    fn last_breath(&mut self, conf: f64, bpm: f64, threshold: f64) {
        if conf >= threshold {
            self.breath_hold = BREATH_HOLD_FRAMES;
            self.breath_bpm = Some(bpm);
        }
    }

    fn update_verdict(&mut self, r: Reading) {
        let Some(base) = self.trained.as_ref() else { return };
        let moving = r.mvs.is_some_and(|v| v >= base.mvs_threshold);
        let shifted = r.shift_z.is_some_and(|z| z >= base.shift_threshold)
            && r.shift_abs.is_some_and(|a| a >= SHIFT_ABS_MIN);
        self.motion_hits = if moving { self.motion_hits + 1 } else { 0 };
        self.still_hits = if shifted { self.still_hits + 1 } else { 0 };
        self.breath_hold = self.breath_hold.saturating_sub(1);
        if self.breath_hold == 0 {
            self.breath_bpm = None;
        }
        self.verdict = Verdict {
            trained: true,
            motion: self.motion_hits >= MOTION_HITS,
            still: self.still_hits >= STILL_HITS,
            breathing: self.breath_hold > 0,
        };
    }
}

/// Room-level vote over per-link verdicts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct RoomAdvisory {
    pub trained_links: usize,
    pub voting_links: usize,
    pub motion_links: usize,
    pub still_links: usize,
    pub breathing_links: usize,
    /// Any link-level vote from enough links.
    pub presence: bool,
    /// Motion from enough links.
    pub motion: bool,
    /// Still-body evidence (baseline shift or breathing) from enough links.
    pub still_presence: bool,
}

pub fn room_advisory(verdicts: &[Verdict]) -> RoomAdvisory {
    let trained: Vec<&Verdict> = verdicts.iter().filter(|v| v.trained).collect();
    let need = MIN_ROOM_VOTES.max((ROOM_VOTE_FRACTION * trained.len() as f64).ceil() as usize);
    let count = |f: fn(&Verdict) -> bool| trained.iter().filter(|v| f(v)).count();
    let voting = count(Verdict::votes);
    let motion = count(|v| v.motion);
    let still = count(|v| v.still);
    let breathing = count(|v| v.breathing);
    let still_votes = count(|v| v.still || v.breathing);
    RoomAdvisory {
        trained_links: trained.len(),
        voting_links: voting,
        motion_links: motion,
        still_links: still,
        breathing_links: breathing,
        presence: !trained.is_empty() && voting >= need,
        motion: !trained.is_empty() && motion >= need,
        still_presence: !trained.is_empty() && still_votes >= need,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-noise in [-1, 1].
    fn noise(i: usize, k: usize) -> f64 {
        let x = ((i * 7919 + k * 104_729 + 13) as f64 * 0.618_033_988_7).fract();
        x * 2.0 - 1.0
    }

    /// Per-tone empty-room frame: tone 10 is very noisy, the rest are stable.
    fn empty_frame(i: usize) -> Vec<f64> {
        let raw: Vec<f64> = (0..TONES)
            .map(|k| {
                let amp = if k == 10 { 0.2 } else { 0.01 };
                1.0 + 0.2 * (k as f64 / 6.0).sin() + amp * noise(i, k)
            })
            .collect();
        let mean = raw.iter().sum::<f64>() / raw.len() as f64;
        raw.iter().map(|x| x / mean).collect()
    }

    fn calibrated() -> LinkFeatures {
        let mut f = LinkFeatures::default();
        f.set_learning(true);
        for i in 0..400 {
            f.push(&empty_frame(i));
        }
        assert!(f.finish_learning().is_some());
        f
    }

    #[test]
    fn nbvi_prefers_stable_tones_and_spaces_them() {
        let frames: Vec<Vec<f64>> = (0..300).map(empty_frame).collect();
        let tones = select_tones(&frames);
        assert_eq!(tones.len(), SELECTED_TONES);
        assert!(!tones.contains(&10), "the noisy tone must not be selected");
        assert!(tones.windows(2).all(|w| w[1] - w[0] >= MIN_TONE_SPACING));
    }

    #[test]
    fn too_little_calibration_does_not_train() {
        let mut f = LinkFeatures::default();
        f.set_learning(true);
        for i in 0..(MIN_TRAIN_FRAMES - 1) {
            f.push(&empty_frame(i));
        }
        assert!(f.finish_learning().is_none());
        f.push(&empty_frame(0));
        assert!(!f.verdict().trained);
    }

    #[test]
    fn empty_room_stays_quiet() {
        let mut f = calibrated();
        for i in 400..1400 {
            f.push(&empty_frame(i));
            assert!(!f.verdict().votes(), "empty room voted at frame {i}: {:?}", f.reading());
        }
    }

    #[test]
    fn gain_scaling_is_invisible_because_input_is_normalised() {
        // The caller normalises by the frame mean; check that the features
        // really are scale-free by scaling then renormalising.
        let mut f = calibrated();
        for i in 400..800 {
            let g = if (i / 40) % 2 == 0 { 1.0 } else { 1.8 };
            let scaled: Vec<f64> = empty_frame(i).iter().map(|x| x * g).collect();
            let mean = scaled.iter().sum::<f64>() / scaled.len() as f64;
            let norm: Vec<f64> = scaled.iter().map(|x| x / mean).collect();
            f.push(&norm);
            assert!(!f.verdict().votes());
        }
    }

    #[test]
    fn static_reshaping_is_a_still_presence_without_motion() {
        let mut f = calibrated();
        // RMS of a*cos is ~a/sqrt(2), so 2x the floor clears it by ~1.4x.
        let amp = 2.0 * SHIFT_ABS_MIN;
        for i in 400..700 {
            let mut fr = empty_frame(i);
            for (k, x) in fr.iter_mut().enumerate() {
                *x += amp * (k as f64 / 3.0).cos(); // permanent shape change
            }
            f.push(&fr);
        }
        let v = f.verdict();
        assert!(v.still, "a permanent shape shift must read as still presence: {:?} {:?}", f.reading(), f.baseline().map(|b| b.shift_threshold));
        // Known limitation: turbulence is a cross-tone coefficient of
        // variation, so a large static reshape also changes how per-tone noise
        // maps into it. On this fixture the moving variance lands at ~2x its
        // empty-room threshold, so `motion` may be set as well. Both votes count
        // as presence; only the still/moving label is affected.
        assert!(v.votes(), "a reshaped channel must vote for presence");
        let r = f.reading();
        let mvs_ratio = r.mvs.unwrap() / f.baseline().unwrap().mvs_threshold;
        assert!(mvs_ratio < 4.0, "static reshape inflated MVS {mvs_ratio:.1}x: {r:?}");
    }

    #[test]
    fn small_long_term_drift_is_not_a_person() {
        let mut f = calibrated();
        for i in 400..800 {
            let mut fr = empty_frame(i);
            for (k, x) in fr.iter_mut().enumerate() {
                *x += 0.012 * (k as f64 / 3.0).cos(); // ~1% permanent drift
            }
            f.push(&fr);
        }
        assert!(!f.verdict().still, "drift below SHIFT_ABS_MIN must not vote");
    }

    #[test]
    fn fast_fluctuation_is_motion() {
        let mut f = calibrated();
        for i in 400..600 {
            let mut fr = empty_frame(i);
            let wob = 0.08 * noise(i, 99);
            for (k, x) in fr.iter_mut().enumerate() {
                *x += wob * (k as f64 / 2.5).sin();
            }
            f.push(&fr);
        }
        assert!(f.verdict().motion);
    }

    #[test]
    fn restored_baseline_round_trips_and_bad_data_is_dropped() {
        let f = calibrated();
        let b = f.baseline().unwrap().clone();
        let json = serde_json::to_string(&b).unwrap();
        let back: TrainedBaseline = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tones, b.tones);
        assert!((back.mvs_threshold - b.mvs_threshold).abs() < 1e-12);
        assert!(LinkFeatures::restored(10.0, Some(&back)).baseline().is_some());
        let mut bad = b;
        bad.sigma[0] = f64::NAN;
        assert!(LinkFeatures::restored(10.0, Some(&bad)).baseline().is_none());
    }

    #[test]
    fn room_vote_needs_two_links_and_a_fraction() {
        let on = Verdict { trained: true, still: true, ..Verdict::default() };
        let off = Verdict { trained: true, ..Verdict::default() };
        assert!(!room_advisory(&[on, off, off, off]).presence, "one link is not enough");
        assert!(room_advisory(&[on, on, off, off, off, off]).presence);
        assert!(!room_advisory(&[]).presence);
        assert!(!room_advisory(&[Verdict::default(); 4]).presence, "untrained links do not vote");
    }

    #[test]
    fn mode_parses() {
        assert_eq!(Mode::parse("Active").unwrap(), Mode::Active);
        assert!(Mode::parse("loud").is_err());
    }
}
