//! Adaptive activity detector for one sensing channel (a node or a peer link).
//!
//! Port of the "HMS" state machine in Espressif's `esp_wifi_sensing` component
//! (`esp_wifi_sensing.c`, `hms_preprocess_channel` / `hms_process_channel_state`),
//! SPDX-FileCopyrightText: 2025-2026 Espressif Systems (Shanghai) CO LTD,
//! SPDX-License-Identifier: Apache-2.0. Modified: rewritten in Rust, fixed-point
//! scaling dropped, the separate "gold" long-term baseline folded into the
//! runtime baseline, wall-clock windows replaced by sample counts.
//!
//! Per sample:
//! 1. Startup: average the first [`INIT_SAMPLES`] samples into a baseline. If
//!    the signal then sits far below the baseline for [`RESET_AFTER`] samples,
//!    start over. A busy room at boot cannot freeze a bad baseline.
//! 2. Asymmetric smoothing (falls twice as fast as it rises).
//! 3. Noise envelope of |smooth - baseline|: fast attack, slow release.
//! 4. Baseline follows drift with a clamped step, 4x slower during activity.
//! 5. Enter/exit thresholds with hysteresis plus a noise allowance; entering
//!    needs [`DEBOUNCE_ENTER`] consecutive hits, leaving needs
//!    [`DEBOUNCE_EXIT`], and ACTIVE is held at least [`HOLD_SAMPLES`].

/// Samples averaged into the startup baseline (esp: 2 s window, >= 50 packets).
pub const INIT_SAMPLES: u32 = 50;
/// Consecutive out-of-band samples after startup that force re-initialisation.
pub const RESET_AFTER: u32 = 50;
const SMOOTH_COEF: f64 = 0.20; // esp: clamp(0.06..0.38) from sensitivity 0.5
const BASELINE_COEF: f64 = 0.020; // esp: HMS_BASELINE_COEF_FIXED
const NOISE_BAND_P: f64 = 0.30; // esp: HMS_NOISE_BAND_DEFAULT
const NOISE_BAND_N: f64 = NOISE_BAND_P * (0.70 / 0.30); // esp: HMS_NOISE_NEGATIVE_RATIO
const NOISE_ATTACK: f64 = 0.18; // esp: HMS_NOISE_ADAPT_DEFAULT
const NOISE_RELEASE: f64 = NOISE_ATTACK * (0.03 / 0.18); // esp: HMS_NOISE_RELEASE_RATIO
const BASELINE_CLAMP_GAIN: f64 = NOISE_ATTACK * (0.35 / 0.18); // esp: HMS_BASELINE_CLAMP_RATIO
const THRESHOLD_P: f64 = 0.55; // esp: ratio in 0.20..0.90 at sensitivity 0.5
const HYSTERESIS_INACTIVE: f64 = 0.28; // esp: HMS_HYSTERESIS_DEFAULT
const HYSTERESIS_ACTIVE: f64 = HYSTERESIS_INACTIVE * (0.08 / 0.28); // esp: HMS_HYSTERESIS_ACTIVE_RATIO
const NOISE_GAIN_ENTER: f64 = NOISE_ATTACK;
const NOISE_GAIN_EXIT: f64 = NOISE_ATTACK * (0.08 / 0.18); // esp: HMS_NOISE_GAIN_EXIT_RATIO
pub const DEBOUNCE_ENTER: u32 = 2; // esp: HMS_DEBOUNCE_COUNT_DEFAULT
pub const DEBOUNCE_EXIT: u32 = DEBOUNCE_ENTER * 2; // esp: HMS_DEBOUNCE_INACTIVE_RATIO
/// Minimum ACTIVE duration in samples (esp `active_filter_ms`; ~1 s at 10 Hz).
pub const HOLD_SAMPLES: u32 = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Init,
    Settling,
    Stable,
}

#[derive(Debug, Clone)]
pub struct PresenceFsm {
    stage: Stage,
    init_sum: f64,
    init_count: u32,
    smooth: f64,
    baseline: f64,
    noise: f64,
    reset_counter: u32,
    active: bool,
    debounce: u32,
    last_diff_positive: bool,
    hold_left: u32,
}

impl Default for PresenceFsm {
    fn default() -> Self {
        Self {
            stage: Stage::Init,
            init_sum: 0.0,
            init_count: 0,
            smooth: 0.0,
            baseline: 0.0,
            noise: 0.0,
            reset_counter: 0,
            active: false,
            debounce: 0,
            last_diff_positive: false,
            hold_left: 0,
        }
    }
}

impl PresenceFsm {
    /// Start already calibrated from a persisted empty-room baseline.
    pub fn with_baseline(baseline: f64) -> Self {
        if !baseline.is_finite() || baseline <= 0.0 {
            return Self::default();
        }
        Self { stage: Stage::Stable, smooth: baseline, baseline, ..Self::default() }
    }

    pub fn stage(&self) -> Stage {
        self.stage
    }
    pub fn is_active(&self) -> bool {
        self.active
    }
    pub fn baseline(&self) -> Option<f64> {
        (self.stage != Stage::Init).then_some(self.baseline)
    }
    /// (smooth - baseline) relative to the baseline; `None` until stable.
    pub fn excess(&self) -> Option<f64> {
        (self.stage == Stage::Stable && self.baseline > 1e-9)
            .then(|| (self.smooth - self.baseline) / self.baseline)
    }

    fn within_band(&self, diff: f64) -> bool {
        diff <= NOISE_BAND_P * self.baseline && diff >= -NOISE_BAND_N * self.baseline
    }

    /// Feed one sample of the activity metric (non-negative, larger = busier).
    pub fn update(&mut self, raw: f64) {
        if !raw.is_finite() {
            return;
        }
        if self.stage == Stage::Init {
            self.init_sum += raw;
            self.init_count += 1;
            if self.init_count >= INIT_SAMPLES {
                let avg = self.init_sum / f64::from(self.init_count);
                *self = Self { stage: Stage::Settling, smooth: avg, baseline: avg, ..Self::default() };
            }
            return;
        }

        let coef = if raw < self.smooth { (SMOOTH_COEF * 2.0).min(1.0) } else { SMOOTH_COEF };
        self.smooth += (raw - self.smooth) * coef;
        let diff = self.smooth - self.baseline;

        if self.stage == Stage::Settling {
            if self.within_band(diff) {
                self.reset_counter = 0;
                self.stage = Stage::Stable;
            } else {
                self.reset_counter += 1;
                if self.reset_counter >= RESET_AFTER {
                    *self = Self::default();
                }
            }
            return;
        }

        // Collapsed-baseline recovery (esp: hms_recover_invalid_runtime_baseline).
        // Activity only raises the metric, so a signal persistently far *below*
        // the baseline means the baseline itself is wrong: start over.
        if diff < -NOISE_BAND_N * self.baseline {
            self.reset_counter += 1;
            if self.reset_counter >= RESET_AFTER {
                *self = Self::default();
                return;
            }
        } else {
            self.reset_counter = 0;
        }

        // Noise envelope, then a clamped, activity-aware baseline step.
        let residual = diff.abs();
        let k = if residual > self.noise { NOISE_ATTACK } else { NOISE_RELEASE };
        self.noise += (residual - self.noise) * k;
        let background = self.within_band(diff);
        let noise_limit = self.noise * BASELINE_CLAMP_GAIN;
        let pos = (NOISE_BAND_P * self.baseline).max(noise_limit);
        let neg = (NOISE_BAND_N * self.baseline).max(noise_limit);
        let step_coef = if background { BASELINE_COEF } else { BASELINE_COEF * 0.25 };
        self.baseline = (self.baseline + diff.clamp(-neg, pos) * step_coef).max(0.0);

        self.step_state(self.smooth - self.baseline);
    }

    fn step_state(&mut self, diff: f64) {
        if (diff > 0.0) != self.last_diff_positive {
            self.debounce = 0; // direction flip: hits must not accumulate across signs
        }
        self.last_diff_positive = diff > 0.0;
        let enter = THRESHOLD_P * self.baseline * (1.0 + HYSTERESIS_ACTIVE) + self.noise * NOISE_GAIN_ENTER;
        let exit = THRESHOLD_P * self.baseline * (1.0 - HYSTERESIS_INACTIVE) + self.noise * NOISE_GAIN_EXIT;
        if !self.active {
            if diff > enter {
                self.debounce += 1;
                if self.debounce >= DEBOUNCE_ENTER {
                    self.active = true;
                    self.debounce = 0;
                    self.hold_left = HOLD_SAMPLES;
                }
            } else {
                self.debounce = 0;
            }
        } else {
            self.hold_left = self.hold_left.saturating_sub(1);
            if diff < exit && self.hold_left == 0 {
                self.debounce += 1;
                if self.debounce >= DEBOUNCE_EXIT {
                    self.active = false;
                    self.debounce = 0;
                }
            } else {
                self.debounce = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quiet(i: u32) -> f64 {
        0.10 + f64::from(i % 3) * 0.005
    }

    fn run(fsm: &mut PresenceFsm, n: u32, f: impl Fn(u32) -> f64) {
        for i in 0..n {
            fsm.update(f(i));
        }
    }

    #[test]
    fn learns_baseline_then_detects_and_releases_activity() {
        let mut fsm = PresenceFsm::default();
        run(&mut fsm, INIT_SAMPLES + 20, quiet);
        assert_eq!(fsm.stage(), Stage::Stable);
        assert!(!fsm.is_active());
        run(&mut fsm, 10, |_| 0.30); // body in the path: 3x quiet
        assert!(fsm.is_active(), "sustained excursion must enter ACTIVE");
        run(&mut fsm, 2, quiet);
        assert!(fsm.is_active(), "brief drop must not release (hold + debounce)");
        run(&mut fsm, 40, quiet);
        assert!(!fsm.is_active(), "sustained quiet must release");
    }

    #[test]
    fn single_spike_does_not_trigger() {
        let mut fsm = PresenceFsm::default();
        run(&mut fsm, INIT_SAMPLES + 20, quiet);
        fsm.update(0.40);
        run(&mut fsm, 5, quiet);
        assert!(!fsm.is_active());
    }

    #[test]
    fn bad_startup_baseline_reinitialises() {
        let mut fsm = PresenceFsm::default();
        run(&mut fsm, INIT_SAMPLES, |_| 1.0); // busy room at boot
        assert_ne!(fsm.stage(), Stage::Init);
        run(&mut fsm, RESET_AFTER + 10, quiet); // room empties, far below baseline
        assert_eq!(fsm.stage(), Stage::Init, "collapsed baseline must restart");
        run(&mut fsm, INIT_SAMPLES + 20, quiet);
        assert_eq!(fsm.stage(), Stage::Stable);
        assert!((fsm.baseline().unwrap() - 0.105).abs() < 0.02);
    }

    #[test]
    fn ignores_non_finite_input() {
        let mut fsm = PresenceFsm::default();
        fsm.update(f64::NAN);
        fsm.update(f64::INFINITY);
        assert_eq!(fsm.stage(), Stage::Init);
    }
}
