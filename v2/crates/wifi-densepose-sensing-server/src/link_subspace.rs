//! Channel-shape features for one sensing link: "jitter" (motion) and
//! "wander" (presence, including a person sitting still).
//!
//! Clean-room implementation of the jitter/wander idea described in
//! Espressif's esp-radar documentation and public API. Its core
//! (`esp_radar_motion_dec`) ships only as a compiled library, so nothing here
//! is derived from that code. Method:
//!
//! - Take [`COLUMNS`] evenly spaced LLTF amplitudes per frame (every 4th of 52).
//! - Over a window of [`WINDOW`] frames, form the column covariance matrix
//!   and its dominant eigenvector `v` (power iteration). `v` is the link's
//!   dominant multipath "shape".
//! - jitter = 1 - |<v_now, v_prev>|: how fast the shape is changing.
//! - wander = 1 - max_j |<v_now, t_j>| over shapes `t_j` learned while the
//!   room was empty. A body anywhere in the path, even still, moves `v` away
//!   from every empty-room template.

use std::collections::VecDeque;

pub const COLUMNS: usize = 13;
pub const WINDOW: usize = 20;
/// Empty-room shapes kept per link (esp-radar keeps up to 10).
pub const MAX_TEMPLATES: usize = 10;
/// A candidate template this close to an existing one is not stored.
const TEMPLATE_DEDUP_SIMILARITY: f64 = 0.995;
const POWER_ITERATIONS: usize = 30;

#[derive(Debug, Clone, Default)]
pub struct LinkSubspace {
    frames: VecDeque<[f64; COLUMNS]>,
    current: Option<[f64; COLUMNS]>,
    previous: Option<[f64; COLUMNS]>,
    templates: Vec<[f64; COLUMNS]>,
    learning: bool,
}

/// Reduce a 52-tone LLTF amplitude vector to [`COLUMNS`] columns.
pub fn columns(amplitudes: &[f64]) -> Option<[f64; COLUMNS]> {
    if amplitudes.len() < COLUMNS * 4 {
        return None;
    }
    let mut out = [0.0; COLUMNS];
    for (i, v) in out.iter_mut().enumerate() {
        *v = amplitudes[i * 4];
    }
    Some(out)
}

fn dot(a: &[f64; COLUMNS], b: &[f64; COLUMNS]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Dominant eigenvector (unit length) of the column covariance of `frames`.
pub fn dominant_vector(frames: &VecDeque<[f64; COLUMNS]>) -> Option<[f64; COLUMNS]> {
    let n = frames.len();
    if n < 2 {
        return None;
    }
    let mut mean = [0.0; COLUMNS];
    for f in frames {
        for (m, x) in mean.iter_mut().zip(f) {
            *m += x / n as f64;
        }
    }
    let mut cov = [[0.0; COLUMNS]; COLUMNS];
    for f in frames {
        for i in 0..COLUMNS {
            let di = f[i] - mean[i];
            for j in 0..COLUMNS {
                cov[i][j] += di * (f[j] - mean[j]) / (n - 1) as f64;
            }
        }
    }
    let mut v = [1.0 / (COLUMNS as f64).sqrt(); COLUMNS];
    for _ in 0..POWER_ITERATIONS {
        let mut w = [0.0; COLUMNS];
        for i in 0..COLUMNS {
            w[i] = (0..COLUMNS).map(|j| cov[i][j] * v[j]).sum();
        }
        let norm = dot(&w, &w).sqrt();
        if norm < 1e-12 {
            return None; // flat channel: no defined shape
        }
        for (vi, wi) in v.iter_mut().zip(w) {
            *vi = wi / norm;
        }
    }
    Some(v)
}

impl LinkSubspace {
    pub fn push(&mut self, amplitudes: &[f64]) {
        let Some(c) = columns(amplitudes) else { return };
        if self.frames.len() == WINDOW {
            self.frames.pop_front();
        }
        self.frames.push_back(c);
        if self.frames.len() < WINDOW {
            return;
        }
        if let Some(v) = dominant_vector(&self.frames) {
            self.previous = self.current.replace(v);
            if self.learning {
                self.learn(v);
            }
        }
    }

    fn learn(&mut self, v: [f64; COLUMNS]) {
        let similar = self.templates.iter().any(|t| dot(t, &v).abs() >= TEMPLATE_DEDUP_SIMILARITY);
        if !similar && self.templates.len() < MAX_TEMPLATES {
            self.templates.push(v);
        }
    }

    /// Start (clearing old shapes) or stop learning empty-room templates.
    pub fn set_learning(&mut self, on: bool) {
        if on {
            self.templates.clear();
        }
        self.learning = on;
    }

    pub fn templates(&self) -> Vec<Vec<f64>> {
        self.templates.iter().map(|t| t.to_vec()).collect()
    }

    /// Restore persisted templates; malformed entries are dropped.
    pub fn set_templates(&mut self, saved: &[Vec<f64>]) {
        self.templates = saved
            .iter()
            .filter_map(|t| <[f64; COLUMNS]>::try_from(t.as_slice()).ok())
            .filter(|t| t.iter().all(|x| x.is_finite()))
            .take(MAX_TEMPLATES)
            .collect();
    }

    /// Natural empty-room shape spread: 1 - the lowest similarity between
    /// any two templates. A link whose empty shape already varies a lot needs
    /// more wander before it means anything. `None` with < 2 templates.
    pub fn wander_floor(&self) -> Option<f64> {
        let mut min_sim: Option<f64> = None;
        for (i, a) in self.templates.iter().enumerate() {
            for b in &self.templates[i + 1..] {
                let s = dot(a, b).abs();
                min_sim = Some(min_sim.map_or(s, |m| m.min(s)));
            }
        }
        min_sim.map(|s| 1.0 - s)
    }

    pub fn template_count(&self) -> usize {
        self.templates.len()
    }

    pub fn jitter(&self) -> Option<f64> {
        Some(1.0 - dot(self.current.as_ref()?, self.previous.as_ref()?).abs())
    }

    /// `None` until empty-room templates exist.
    pub fn wander(&self) -> Option<f64> {
        let v = self.current.as_ref()?;
        let best = self.templates.iter().map(|t| dot(t, v).abs()).fold(None, |m: Option<f64>, s| {
            Some(m.map_or(s, |m| m.max(s)))
        })?;
        Some(1.0 - best)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 52 tones whose shape follows `shape(k)` scaled by a per-frame gain.
    fn frame(gain: f64, shape: impl Fn(usize) -> f64) -> Vec<f64> {
        (0..52).map(|k| 10.0 + gain * shape(k)).collect()
    }

    fn feed(s: &mut LinkSubspace, n: usize, shape: impl Fn(usize) -> f64 + Copy) {
        for i in 0..n {
            s.push(&frame(((i as f64) * 0.7).sin(), shape));
        }
    }

    #[test]
    fn same_shape_has_low_wander_new_shape_has_high() {
        let empty = |k: usize| (k as f64 / 8.0).sin();
        let occupied = |k: usize| (k as f64 / 3.0).cos();
        let mut s = LinkSubspace::default();
        s.set_learning(true);
        feed(&mut s, WINDOW + 10, empty);
        s.set_learning(false);
        assert!(s.template_count() >= 1);
        assert!(s.wander().unwrap() < 0.05, "empty room must match its template");
        feed(&mut s, WINDOW, occupied);
        assert!(s.wander().unwrap() > 0.3, "a changed multipath shape must wander");
    }

    #[test]
    fn no_templates_means_no_wander_and_short_frames_ignored() {
        let mut s = LinkSubspace::default();
        feed(&mut s, WINDOW + 2, |k| (k as f64).sin());
        assert!(s.wander().is_none());
        assert!(s.jitter().is_some());
        s.push(&[1.0; 10]);
        assert!(columns(&[1.0; 10]).is_none());
    }

    #[test]
    fn wander_floor_reflects_template_spread() {
        let mut s = LinkSubspace::default();
        assert!(s.wander_floor().is_none());
        let mut a = [0.0; COLUMNS];
        a[0] = 1.0;
        let mut b = [0.0; COLUMNS];
        b[0] = 0.8;
        b[1] = 0.6;
        s.templates = vec![a, b];
        assert!((s.wander_floor().unwrap() - 0.2).abs() < 1e-9);
    }

    #[test]
    fn flat_channel_has_no_shape() {
        let frames: VecDeque<[f64; COLUMNS]> = (0..WINDOW).map(|_| [5.0; COLUMNS]).collect();
        assert!(dominant_vector(&frames).is_none());
    }
}
