//! Webcam-supervised activity recognition with an honest held-out evaluation.
//!
//! Adapted (re-implemented in Rust, not copied) from the approach in
//! wifi-sensing-kit by Andris Gauracs — https://github.com/andrisgauracs/wifi-sensing-kit
//! (MIT licence; itself built on RuView firmware). What is borrowed is the
//! method, described in that project's `docs/how-it-works.md`:
//!
//! - A *link* is one transmitter -> receiver pair (router -> board, or
//!   board -> board). Only links that stream >= 5 frames/s are used.
//! - Every 0.5 s, for every link, over the last 1.5 s of frames: the
//!   gain-normalised mean amplitude shape over the 52 LLTF data tones, the
//!   per-tone spread (std) and the per-tone mean frame-to-frame change.
//! - A camera teacher (MediaPipe Pose, running in the *browser*) turns a
//!   skeleton into a movement label. Only the label string reaches this
//!   server; video never leaves the browser. Uncertain moments are not
//!   recorded.
//! - k-NN (k = 7) on standardised features, prediction smoothed over 3 s.
//! - Honest scoring: the test session is recorded separately, after
//!   training. Test windows are never drawn from training recordings.
//!
//! Everything here is pure computation plus small, bounded JSONL/JSON files
//! under `<data_dir>/activity/`. HTTP lives in `activity_api.rs`.

use std::borrow::Borrow;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};

// ── Constants ────────────────────────────────────────────────────────────────

/// LLTF data tones per frame. Frames with any other tone count are ignored so
/// feature columns always mean the same subcarriers.
pub const N_TONES: usize = 52;
/// Features per link: mean shape + per-tone std + per-tone mean |frame delta|.
pub const FEATURES_PER_LINK: usize = N_TONES * 3;
/// Sliding window the features are computed over.
pub const WINDOW_SECS: f64 = 1.5;
/// A new feature vector (and sample) is produced at most this often.
pub const HOP_SECS: f64 = 0.5;
/// Minimum frames inside a window for the link to produce a feature.
pub const MIN_FRAMES: usize = 4;
/// A link must stream at least this many frames/s to be selected for a session.
pub const MIN_LINK_RATE_HZ: f64 = 5.0;
/// Rate threshold for a link that is merely "still alive" in a later session.
const ALIVE_LINK_RATE_HZ: f64 = 0.5;
const RATE_WINDOW_SECS: f64 = 3.0;
/// A briefly stalled link reuses its last window for this long.
const MAX_STALE_SECS: f64 = 3.0;
/// Live predictions are smoothed over this many seconds.
pub const SMOOTH_SECS: f64 = 3.0;
pub const DEFAULT_K: usize = 7;
pub const MAX_K: usize = 51;

/// Hard bounds so a misbehaving client or network cannot grow memory/disk.
pub const MAX_TRACKED_LINKS: usize = 32;
pub const MAX_SESSION_LINKS: usize = 8;
pub const MAX_BUFFERED_FRAMES: usize = 512;
pub const MAX_SAMPLES_PER_SESSION: usize = 7200;
pub const MAX_TOTAL_SAMPLES: usize = 60_000;
pub const MAX_SESSIONS: usize = 64;
pub const MAX_CLASSES: usize = 32;
pub const MAX_LABEL_LEN: usize = 32;
const MAX_LABEL_HISTORY: usize = 200;
const MAX_SESSION_FILE_BYTES: u64 = 128 * 1024 * 1024;
/// Evaluation / persisted-model row caps (even stride subsampling).
pub const MAX_EVAL_TRAIN: usize = 2000;
pub const MAX_EVAL_TEST: usize = 1500;
pub const MAX_MODEL_ROWS: usize = 3000;
/// Label window: at least this many camera labels, this fraction agreeing.
const LABEL_MIN_COUNT: usize = 3;
const LABEL_AGREE: f64 = 0.6;
/// Minimum training windows (and classes) needed to fit a model.
pub const MIN_TRAIN_SAMPLES: usize = 10;
const MODEL_VERSION: u32 = 1;
const STD_EPS: f32 = 1e-6;

// ── Errors ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActivityError {
    /// Bad client input (HTTP 400).
    Invalid(String),
    /// Request is valid but the current state forbids it (HTTP 409).
    Conflict(String),
    /// Not found (HTTP 404).
    NotFound(String),
    /// Filesystem failure; detail is logged, never sent to the client.
    Io(String),
}

impl std::fmt::Display for ActivityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(m) | Self::Conflict(m) | Self::NotFound(m) | Self::Io(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for ActivityError {}

// ── Labels ───────────────────────────────────────────────────────────────────

/// A label is 1..=32 chars of `[a-z0-9_-]`. `unsure` is reserved for "camera
/// is not certain" and is never recorded.
pub fn validate_label(label: &str) -> Result<&str, ActivityError> {
    if label.is_empty() || label.len() > MAX_LABEL_LEN {
        return Err(ActivityError::Invalid(format!("label must be 1..={MAX_LABEL_LEN} characters")));
    }
    if !label.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-') {
        return Err(ActivityError::Invalid("label may only contain a-z, 0-9, '_' and '-'".into()));
    }
    Ok(label)
}

// ── Feature extraction ───────────────────────────────────────────────────────

/// Divide a frame by its own mean level, so receiver gain steps (which scale
/// every tone equally) cancel. `None` for a non-finite or ~zero-level frame.
pub fn normalise_frame(amps: &[f64]) -> Option<Vec<f64>> {
    if amps.len() != N_TONES || amps.iter().any(|a| !a.is_finite()) {
        return None;
    }
    let mean = amps.iter().sum::<f64>() / N_TONES as f64;
    if mean <= 1e-9 {
        return None;
    }
    Some(amps.iter().map(|a| a / mean).collect())
}

/// Feature vector for one link over one window of raw frames:
/// `[mean shape (52) | per-tone std (52) | per-tone mean |delta| (52)]`.
/// `None` when fewer than [`MIN_FRAMES`] valid frames are present.
pub fn window_features(frames: &[&[f64]]) -> Option<Vec<f32>> {
    let norm: Vec<Vec<f64>> = frames.iter().filter_map(|f| normalise_frame(f)).collect();
    let n = norm.len();
    if n < MIN_FRAMES {
        return None;
    }
    let mut out = Vec::with_capacity(FEATURES_PER_LINK);
    let mut means = [0.0f64; N_TONES];
    for k in 0..N_TONES {
        means[k] = norm.iter().map(|f| f[k]).sum::<f64>() / n as f64;
    }
    out.extend(means.iter().map(|&m| m as f32));
    for k in 0..N_TONES {
        let var = norm.iter().map(|f| (f[k] - means[k]).powi(2)).sum::<f64>() / n as f64;
        out.push(var.sqrt() as f32);
    }
    for k in 0..N_TONES {
        let d = norm.windows(2).map(|w| (w[1][k] - w[0][k]).abs()).sum::<f64>() / (n - 1) as f64;
        out.push(d as f32);
    }
    Some(out)
}

/// Rolling per-link frame buffer producing windowed features.
#[derive(Debug, Default)]
struct LinkBuffer {
    frames: VecDeque<(f64, Vec<f64>)>,
    arrivals: VecDeque<f64>,
    last_feature: Option<(f64, Vec<f32>)>,
}

impl LinkBuffer {
    fn push(&mut self, t: f64, amps: &[f64]) {
        self.frames.push_back((t, amps.to_vec()));
        while self.frames.len() > MAX_BUFFERED_FRAMES
            || self.frames.front().is_some_and(|f| t - f.0 > WINDOW_SECS)
        {
            self.frames.pop_front();
        }
        self.arrivals.push_back(t);
        while self.arrivals.front().is_some_and(|a| t - a > RATE_WINDOW_SECS) {
            self.arrivals.pop_front();
        }
    }

    /// Frames/s over the last [`RATE_WINDOW_SECS`].
    fn rate(&self, now: f64) -> f64 {
        self.arrivals.iter().filter(|&&a| now - a <= RATE_WINDOW_SECS).count() as f64 / RATE_WINDOW_SECS
    }

    /// Feature for the window ending at `now`, falling back to the last good
    /// one for up to [`MAX_STALE_SECS`].
    fn feature(&mut self, now: f64) -> Option<Vec<f32>> {
        let window: Vec<&[f64]> =
            self.frames.iter().filter(|(t, _)| now - t <= WINDOW_SECS).map(|(_, a)| a.as_slice()).collect();
        if let Some(f) = window_features(&window) {
            self.last_feature = Some((now, f.clone()));
            return Some(f);
        }
        self.last_feature.as_ref().filter(|(t, _)| now - t < MAX_STALE_SECS).map(|(_, f)| f.clone())
    }
}

// ── k-NN + standardisation ───────────────────────────────────────────────────

/// Fit per-column mean/std on `rows` (population std, +eps like the kit).
pub fn fit_standardiser(rows: &[&[f32]]) -> (Vec<f32>, Vec<f32>) {
    let dim = rows.first().map_or(0, |r| r.len());
    let n = rows.len().max(1) as f32;
    let mut mean = vec![0.0f32; dim];
    for r in rows {
        for (m, x) in mean.iter_mut().zip(r.iter()) {
            *m += x / n;
        }
    }
    let mut var = vec![0.0f32; dim];
    for r in rows {
        for ((v, x), m) in var.iter_mut().zip(r.iter()).zip(&mean) {
            *v += (x - m).powi(2) / n;
        }
    }
    (mean, var.into_iter().map(|v| v.sqrt() + STD_EPS).collect())
}

pub fn standardise(x: &[f32], mean: &[f32], std: &[f32]) -> Vec<f32> {
    x.iter().zip(mean).zip(std).map(|((x, m), s)| (x - m) / s).collect()
}

fn dist2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Vote fractions (over `n_labels`) among the `k` nearest rows by Euclidean
/// distance. Ties between labels are resolved by [`argmax_first`].
pub fn knn_votes(rows: &[Vec<f32>], y: &[usize], n_labels: usize, z: &[f32], k: usize) -> Vec<f32> {
    let mut votes = vec![0.0f32; n_labels];
    if rows.is_empty() {
        return votes;
    }
    let mut d: Vec<(f32, usize)> = rows.iter().enumerate().map(|(i, r)| (dist2(r, z), i)).collect();
    let k = k.clamp(1, d.len());
    d.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    for &(_, i) in &d[..k] {
        votes[y[i]] += 1.0 / k as f32;
    }
    votes
}

/// Index of the largest value; the lowest index wins ties (deterministic,
/// labels are kept alphabetically sorted).
pub fn argmax_first(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// Evenly strided subsample of at most `cap` indices out of `n`.
fn stride_indices(n: usize, cap: usize) -> Vec<usize> {
    if n <= cap {
        return (0..n).collect();
    }
    (0..cap).map(|i| i * n / cap).collect()
}

// ── Sessions ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionKind {
    Train,
    Test,
}

impl SessionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Test => "test",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sample {
    /// Seconds since the server's activity epoch (ordering only).
    pub t: f64,
    pub label: String,
    /// `link_ids.len() * FEATURES_PER_LINK` features, links in `link_ids` order.
    pub x: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SessionHeader {
    #[serde(rename = "type")]
    kind_tag: String,
    id: String,
    kind: SessionKind,
    link_ids: Vec<String>,
    features_per_link: usize,
    started_unix: u64,
}

#[derive(Debug, Clone)]
pub struct Session {
    pub id: String,
    pub kind: SessionKind,
    pub link_ids: Vec<String>,
    pub started_unix: u64,
    pub samples: Vec<Sample>,
}

impl Session {
    pub fn summary(&self) -> serde_json::Value {
        serde_json::json!({
            "id": self.id,
            "kind": self.kind.as_str(),
            "link_ids": self.link_ids,
            "started_unix": self.started_unix,
            "windows": self.samples.len(),
            "labels": label_counts(&self.samples),
        })
    }
}

fn label_counts(samples: &[Sample]) -> BTreeMap<String, usize> {
    let mut m = BTreeMap::new();
    for s in samples {
        *m.entry(s.label.clone()).or_insert(0) += 1;
    }
    m
}

fn round4(x: f32) -> f32 {
    (x * 1e4).round() / 1e4
}

struct ActiveSession {
    id: String,
    kind: SessionKind,
    link_ids: Vec<String>,
    started_unix: u64,
    writer: BufWriter<File>,
    samples: Vec<Sample>,
    skipped: u32,
    no_label: u32,
    no_wifi: u32,
    truncated: bool,
}

/// Counters returned when a session stops (and while it records).
#[derive(Debug, Clone, Serialize)]
pub struct SessionProgress {
    pub id: String,
    pub kind: SessionKind,
    pub link_ids: Vec<String>,
    pub windows: usize,
    pub skipped: u32,
    pub no_label: u32,
    pub no_wifi: u32,
    pub truncated: bool,
}

impl ActiveSession {
    fn progress(&self) -> SessionProgress {
        SessionProgress {
            id: self.id.clone(),
            kind: self.kind,
            link_ids: self.link_ids.clone(),
            windows: self.samples.len(),
            skipped: self.skipped,
            no_label: self.no_label,
            no_wifi: self.no_wifi,
            truncated: self.truncated,
        }
    }
}

// ── Model ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub version: u32,
    pub k: usize,
    pub link_ids: Vec<String>,
    pub features_per_link: usize,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    /// Alphabetically sorted class names.
    pub labels: Vec<String>,
    /// Standardised training rows.
    pub rows: Vec<Vec<f32>>,
    pub y: Vec<usize>,
    pub trained_unix: u64,
    pub train_sessions: Vec<String>,
    /// Training windows before any row-cap subsampling.
    pub n_train_windows: usize,
}

impl Model {
    fn valid(&self) -> bool {
        let dim = self.link_ids.len() * self.features_per_link;
        self.version == MODEL_VERSION
            && dim > 0
            && self.mean.len() == dim
            && self.std.len() == dim
            && !self.labels.is_empty()
            && self.rows.len() == self.y.len()
            && self.rows.iter().all(|r| r.len() == dim)
            && self.y.iter().all(|&i| i < self.labels.len())
    }

    /// Vote fractions per class for one raw (un-standardised) feature vector.
    pub fn predict_votes(&self, x: &[f32]) -> Vec<f32> {
        let z = standardise(x, &self.mean, &self.std);
        knn_votes(&self.rows, &self.y, self.labels.len(), &z, self.k)
    }
}

fn check_k(k: usize) -> Result<usize, ActivityError> {
    if (1..=MAX_K).contains(&k) {
        Ok(k)
    } else {
        Err(ActivityError::Invalid(format!("k must be in 1..={MAX_K}")))
    }
}

/// Split sessions into the reference-compatible set. The reference link list
/// is that of the first train session (by id) so train and test always share
/// identical feature columns.
fn compatible<S: Borrow<Session>>(
    sessions: &[S],
) -> Result<(Vec<String>, Vec<&Session>, Vec<String>), ActivityError> {
    let mut sorted: Vec<&Session> = sessions.iter().map(Borrow::borrow).collect();
    sorted.sort_by(|a, b| a.id.cmp(&b.id));
    let reference = sorted
        .iter()
        .find(|s| s.kind == SessionKind::Train)
        .map(|s| s.link_ids.clone())
        .ok_or_else(|| ActivityError::Conflict("no training session recorded".into()))?;
    let (ok, bad): (Vec<&Session>, Vec<&Session>) = sorted.into_iter().partition(|s| s.link_ids == reference);
    Ok((reference, ok, bad.into_iter().map(|s| s.id.clone()).collect()))
}

/// Fit a model on TRAIN sessions only. Test sessions are never read.
pub fn train_model<S: Borrow<Session>>(sessions: &[S], k: usize, trained_unix: u64) -> Result<Model, ActivityError> {
    let k = check_k(k)?;
    let (link_ids, usable, _) = compatible(sessions)?;
    let train: Vec<&Session> = usable.into_iter().filter(|s| s.kind == SessionKind::Train).collect();
    let samples: Vec<&Sample> = train.iter().flat_map(|s| s.samples.iter()).collect();
    let labels: Vec<String> = samples.iter().map(|s| s.label.clone()).collect::<BTreeSet<_>>().into_iter().collect();
    if samples.len() < MIN_TRAIN_SAMPLES || labels.len() < 2 {
        return Err(ActivityError::Conflict(format!(
            "need at least {MIN_TRAIN_SAMPLES} training windows across 2+ labels (have {} windows, {} labels)",
            samples.len(),
            labels.len()
        )));
    }
    let raw: Vec<&[f32]> = samples.iter().map(|s| s.x.as_slice()).collect();
    let (mean, std) = fit_standardiser(&raw);
    let keep = stride_indices(samples.len(), MAX_MODEL_ROWS);
    let rows = keep.iter().map(|&i| standardise(raw[i], &mean, &std).into_iter().map(round4).collect()).collect();
    let y = keep.iter().map(|&i| labels.binary_search(&samples[i].label).unwrap_or(0)).collect();
    Ok(Model {
        version: MODEL_VERSION,
        k,
        link_ids,
        features_per_link: FEATURES_PER_LINK,
        mean,
        std,
        labels,
        rows,
        y,
        trained_unix,
        train_sessions: train.iter().map(|s| s.id.clone()).collect(),
        n_train_windows: samples.len(),
    })
}

// ── Evaluation ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct ClassStat {
    pub support: u32,
    pub correct: u32,
    pub recall: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct LinkImportance {
    pub link: String,
    /// Test accuracy with this link's columns removed (model refit without it).
    pub accuracy_without: f64,
    /// `accuracy - accuracy_without`: how much the link contributes.
    pub accuracy_drop: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Evaluation {
    pub k: usize,
    pub link_ids: Vec<String>,
    pub train_sessions: Vec<String>,
    pub test_sessions: Vec<String>,
    /// Sessions skipped because their link set differs from the reference.
    pub excluded_sessions: Vec<String>,
    pub n_train: usize,
    pub n_test: usize,
    pub subsampled: bool,
    pub labels: Vec<String>,
    pub accuracy: f64,
    /// 1 / number of classes.
    pub chance_baseline: f64,
    /// Accuracy of always predicting the most common TRAINING label.
    pub majority_baseline: f64,
    pub majority_class: String,
    pub per_class: BTreeMap<String, ClassStat>,
    /// `confusion[truth][predicted]` counts, test windows only.
    pub confusion: BTreeMap<String, BTreeMap<String, u32>>,
    pub link_importance: Vec<LinkImportance>,
    pub evidence_tag: &'static str,
    pub note: &'static str,
}

/// Predict every test row from standardised train rows, restricted to `cols`.
fn predict_all(
    train: &[Vec<f32>],
    train_y: &[usize],
    test: &[Vec<f32>],
    n_labels: usize,
    k: usize,
    cols: &[usize],
) -> Vec<usize> {
    let project = |r: &Vec<f32>| -> Vec<f32> { cols.iter().map(|&c| r[c]).collect() };
    let tr: Vec<Vec<f32>> = train.iter().map(project).collect();
    test.iter().map(|r| argmax_first(&knn_votes(&tr, train_y, n_labels, &project(r), k))).collect()
}

fn accuracy(pred: &[usize], truth: &[usize]) -> f64 {
    if truth.is_empty() {
        return 0.0;
    }
    pred.iter().zip(truth).filter(|(p, t)| p == t).count() as f64 / truth.len() as f64
}

/// Score ONLY windows from `test` sessions with a model fitted on `train`
/// sessions. Never mixes the two: a window's session kind decides its role.
pub fn evaluate<S: Borrow<Session>>(sessions: &[S], k: usize) -> Result<Evaluation, ActivityError> {
    let k = check_k(k)?;
    let (link_ids, usable, excluded) = compatible(sessions)?;
    let train_s: Vec<&Session> = usable.iter().copied().filter(|s| s.kind == SessionKind::Train).collect();
    let test_s: Vec<&Session> = usable.iter().copied().filter(|s| s.kind == SessionKind::Test).collect();
    let tr_samples: Vec<&Sample> = train_s.iter().flat_map(|s| s.samples.iter()).collect();
    let te_samples: Vec<&Sample> = test_s.iter().flat_map(|s| s.samples.iter()).collect();
    if te_samples.is_empty() {
        return Err(ActivityError::Conflict("no test session windows: record a separate test session first".into()));
    }
    let labels: Vec<String> =
        tr_samples.iter().chain(&te_samples).map(|s| s.label.clone()).collect::<BTreeSet<_>>().into_iter().collect();
    let train_labels: BTreeSet<&str> = tr_samples.iter().map(|s| s.label.as_str()).collect();
    if tr_samples.len() < MIN_TRAIN_SAMPLES || train_labels.len() < 2 {
        return Err(ActivityError::Conflict(format!(
            "need at least {MIN_TRAIN_SAMPLES} training windows across 2+ labels"
        )));
    }
    let tr_keep = stride_indices(tr_samples.len(), MAX_EVAL_TRAIN);
    let te_keep = stride_indices(te_samples.len(), MAX_EVAL_TEST);
    let subsampled = tr_keep.len() < tr_samples.len() || te_keep.len() < te_samples.len();
    let raw_tr: Vec<&[f32]> = tr_keep.iter().map(|&i| tr_samples[i].x.as_slice()).collect();
    let (mean, std) = fit_standardiser(&raw_tr); // fitted on TRAIN rows only
    let z_tr: Vec<Vec<f32>> = raw_tr.iter().map(|r| standardise(r, &mean, &std)).collect();
    let z_te: Vec<Vec<f32>> = te_keep.iter().map(|&i| standardise(&te_samples[i].x, &mean, &std)).collect();
    let idx = |s: &Sample| labels.binary_search(&s.label).unwrap_or(0);
    let y_tr: Vec<usize> = tr_keep.iter().map(|&i| idx(tr_samples[i])).collect();
    let y_te: Vec<usize> = te_keep.iter().map(|&i| idx(te_samples[i])).collect();

    let dim = link_ids.len() * FEATURES_PER_LINK;
    let all_cols: Vec<usize> = (0..dim).collect();
    let pred = predict_all(&z_tr, &y_tr, &z_te, labels.len(), k, &all_cols);
    let acc = accuracy(&pred, &y_te);

    let mut confusion: BTreeMap<String, BTreeMap<String, u32>> = BTreeMap::new();
    let mut per_class: BTreeMap<String, ClassStat> = BTreeMap::new();
    for (&p, &t) in pred.iter().zip(&y_te) {
        *confusion.entry(labels[t].clone()).or_default().entry(labels[p].clone()).or_insert(0) += 1;
        let c = per_class.entry(labels[t].clone()).or_insert(ClassStat { support: 0, correct: 0, recall: 0.0 });
        c.support += 1;
        c.correct += u32::from(p == t);
    }
    for c in per_class.values_mut() {
        c.recall = f64::from(c.correct) / f64::from(c.support.max(1));
    }

    let mut counts = vec![0usize; labels.len()];
    for &t in &y_tr {
        counts[t] += 1;
    }
    let maj = counts.iter().enumerate().fold(0, |b, (i, &c)| if c > counts[b] { i } else { b });
    let majority_baseline = y_te.iter().filter(|&&t| t == maj).count() as f64 / y_te.len() as f64;

    // Leave-one-link-out ablation: drop a link's columns, rescore.
    let mut link_importance = Vec::new();
    if link_ids.len() >= 2 {
        for (li, link) in link_ids.iter().enumerate() {
            let lo = li * FEATURES_PER_LINK;
            let cols: Vec<usize> = all_cols.iter().copied().filter(|c| *c < lo || *c >= lo + FEATURES_PER_LINK).collect();
            let a = accuracy(&predict_all(&z_tr, &y_tr, &z_te, labels.len(), k, &cols), &y_te);
            link_importance.push(LinkImportance { link: link.clone(), accuracy_without: a, accuracy_drop: acc - a });
        }
    }

    Ok(Evaluation {
        k,
        link_ids,
        train_sessions: train_s.iter().map(|s| s.id.clone()).collect(),
        test_sessions: test_s.iter().map(|s| s.id.clone()).collect(),
        excluded_sessions: excluded,
        n_train: tr_samples.len(),
        n_test: te_samples.len(),
        subsampled,
        chance_baseline: 1.0 / labels.len() as f64,
        majority_baseline,
        majority_class: labels[maj].clone(),
        labels,
        accuracy: acc,
        per_class,
        confusion,
        link_importance,
        evidence_tag: "MEASURED",
        note: "Scored on this server's own separately recorded test session(s); a single room and \
               operator. Not camera-grade and not a cross-environment claim. Reproduce with \
               scripts/eval_activity.py on the saved sessions.",
    })
}

// ── Engine ───────────────────────────────────────────────────────────────────

/// Smoothed live WiFi-only prediction.
#[derive(Debug, Clone, Serialize)]
pub struct LivePrediction {
    pub status: &'static str,
    pub best: Option<String>,
    pub confidence: f64,
    pub probs: BTreeMap<String, f64>,
    pub window_secs: f64,
}

pub struct ActivityEngine {
    dir: PathBuf,
    epoch: Instant,
    links: BTreeMap<String, LinkBuffer>,
    labels: VecDeque<(f64, String)>,
    last_tick: f64,
    sessions: Vec<Arc<Session>>,
    active: Option<ActiveSession>,
    model: Option<Model>,
    recent: VecDeque<Vec<f32>>,
    seq: u32,
}

impl Default for ActivityEngine {
    fn default() -> Self {
        Self::new(PathBuf::from("data"))
    }
}

impl ActivityEngine {
    /// Empty engine writing under `<data_dir>/activity/`. Does not touch disk.
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            dir: data_dir.join("activity"),
            epoch: Instant::now(),
            links: BTreeMap::new(),
            labels: VecDeque::new(),
            last_tick: f64::NEG_INFINITY,
            sessions: Vec::new(),
            active: None,
            model: None,
            recent: VecDeque::new(),
            seq: 0,
        }
    }

    /// Engine with persisted sessions and model loaded from `data_dir`.
    pub fn load(data_dir: &Path) -> Self {
        let mut e = Self::new(data_dir.to_path_buf());
        e.sessions = load_sessions(&e.sessions_dir());
        e.model = std::fs::read_to_string(e.model_path())
            .ok()
            .and_then(|t| serde_json::from_str::<Model>(&t).ok())
            .filter(Model::valid);
        tracing::info!(
            "activity: loaded {} sessions, model {}",
            e.sessions.len(),
            if e.model.is_some() { "present" } else { "absent" }
        );
        e
    }

    fn sessions_dir(&self) -> PathBuf {
        self.dir.join("sessions")
    }

    fn model_path(&self) -> PathBuf {
        self.dir.join("model.json")
    }

    // ── ingest ──────────────────────────────────────────────────────────────

    /// Feed one frame of a link. `link` is e.g. `"3>1"` (board 3 -> board 1)
    /// or `"R>1"` (router -> board 1).
    pub fn observe(&mut self, link: &str, amps: &[f64], now: Instant) {
        let t = now.saturating_duration_since(self.epoch).as_secs_f64();
        self.observe_at(link, amps, t);
    }

    pub fn observe_at(&mut self, link: &str, amps: &[f64], t: f64) {
        if amps.len() != N_TONES || !t.is_finite() {
            return;
        }
        if !self.links.contains_key(link) && self.links.len() >= MAX_TRACKED_LINKS {
            return;
        }
        self.links.entry(link.to_string()).or_default().push(t, amps);
        if t - self.last_tick >= HOP_SECS {
            self.last_tick = t;
            self.tick(t);
        }
    }

    /// Record a camera label (already produced in the browser).
    pub fn add_label(&mut self, label: &str, now: Instant) -> Result<(), ActivityError> {
        let t = now.saturating_duration_since(self.epoch).as_secs_f64();
        self.add_label_at(label, t)
    }

    pub fn add_label_at(&mut self, label: &str, t: f64) -> Result<(), ActivityError> {
        validate_label(label)?;
        if self.labels.len() >= MAX_LABEL_HISTORY {
            self.labels.pop_front();
        }
        self.labels.push_back((t, label.to_string()));
        Ok(())
    }

    /// Majority camera label over the last window, if confident enough.
    fn window_label(&self, now: f64) -> Option<String> {
        let recent: Vec<&str> =
            self.labels.iter().filter(|(t, _)| now - t <= WINDOW_SECS && *t <= now).map(|(_, l)| l.as_str()).collect();
        if recent.len() < LABEL_MIN_COUNT {
            return None;
        }
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for l in &recent {
            *counts.entry(l).or_insert(0) += 1;
        }
        let (best, n) = counts.into_iter().max_by_key(|(_, n)| *n)?;
        (n as f64 / recent.len() as f64 >= LABEL_AGREE && best != "unsure").then(|| best.to_string())
    }

    fn joint_feature(&mut self, ids: &[String], now: f64) -> Option<Vec<f32>> {
        let mut out = Vec::with_capacity(ids.len() * FEATURES_PER_LINK);
        for id in ids {
            out.extend(self.links.get_mut(id)?.feature(now)?);
        }
        Some(out)
    }

    fn tick(&mut self, now: f64) {
        if let Some(ids) = self.active.as_ref().map(|a| a.link_ids.clone()) {
            let label = self.window_label(now);
            let feat = self.joint_feature(&ids, now);
            if let Some(a) = self.active.as_mut() {
                match (label, feat) {
                    (Some(label), Some(x)) => {
                        let x: Vec<f32> = x.into_iter().map(round4).collect();
                        let s = Sample { t: now, label, x };
                        if let Ok(line) = serde_json::to_string(&s) {
                            if writeln!(a.writer, "{line}").is_err() {
                                tracing::warn!("activity: failed to append sample");
                            }
                        }
                        a.samples.push(s);
                        if a.samples.len() % 10 == 0 {
                            let _ = a.writer.flush();
                        }
                    }
                    (l, f) => {
                        a.skipped += 1;
                        a.no_label += u32::from(l.is_none());
                        a.no_wifi += u32::from(f.is_none());
                    }
                }
                if a.samples.len() >= MAX_SAMPLES_PER_SESSION {
                    a.truncated = true;
                }
            }
            if self.active.as_ref().is_some_and(|a| a.truncated) {
                let _ = self.stop_session();
            }
        }
        if let Some(ids) = self.model.as_ref().map(|m| m.link_ids.clone()) {
            if let Some(x) = self.joint_feature(&ids, now) {
                if let Some(m) = self.model.as_ref() {
                    let votes = m.predict_votes(&x);
                    self.recent.push_back(votes);
                    while self.recent.len() > (SMOOTH_SECS / HOP_SECS).round() as usize {
                        self.recent.pop_front();
                    }
                }
            }
        }
    }

    // ── sessions ────────────────────────────────────────────────────────────

    fn total_samples(&self) -> usize {
        self.sessions.iter().map(|s| s.samples.len()).sum()
    }

    /// Links currently streaming at least `min_rate` frames/s, fastest first.
    fn live_links(&self, now: f64, min_rate: f64) -> Vec<(String, f64)> {
        let mut v: Vec<(String, f64)> = self
            .links
            .iter()
            .map(|(k, b)| (k.clone(), b.rate(now)))
            .filter(|(_, r)| *r >= min_rate)
            .collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        v
    }

    pub fn start_session(&mut self, kind: SessionKind, now: Instant, unix: u64) -> Result<SessionProgress, ActivityError> {
        let t = now.saturating_duration_since(self.epoch).as_secs_f64();
        self.start_session_at(kind, t, unix)
    }

    pub fn start_session_at(&mut self, kind: SessionKind, t: f64, unix: u64) -> Result<SessionProgress, ActivityError> {
        if self.active.is_some() {
            return Err(ActivityError::Conflict("a session is already recording; stop it first".into()));
        }
        if self.sessions.len() >= MAX_SESSIONS {
            return Err(ActivityError::Conflict(format!("session limit ({MAX_SESSIONS}) reached; delete old sessions")));
        }
        if self.total_samples() >= MAX_TOTAL_SAMPLES {
            return Err(ActivityError::Conflict("stored window limit reached; delete old sessions".into()));
        }
        let reference = self
            .sessions
            .iter()
            .min_by(|a, b| a.id.cmp(&b.id))
            .map(|s| s.link_ids.clone());
        let link_ids = match reference {
            Some(ids) => {
                let alive: BTreeSet<String> = self.live_links(t, ALIVE_LINK_RATE_HZ).into_iter().map(|l| l.0).collect();
                if !ids.iter().all(|i| alive.contains(i)) {
                    return Err(ActivityError::Conflict(format!(
                        "links changed (existing sessions use {ids:?}); delete sessions to start over"
                    )));
                }
                ids
            }
            None => {
                let mut ids: Vec<String> = self
                    .live_links(t, MIN_LINK_RATE_HZ)
                    .into_iter()
                    .take(MAX_SESSION_LINKS)
                    .map(|l| l.0)
                    .collect();
                ids.sort();
                ids
            }
        };
        if link_ids.is_empty() {
            return Err(ActivityError::Conflict(format!("no link is streaming at {MIN_LINK_RATE_HZ} frames/s or more")));
        }
        std::fs::create_dir_all(self.sessions_dir()).map_err(|e| ActivityError::Io(e.to_string()))?;
        self.seq = self.seq.wrapping_add(1);
        let mut id = format!("{}-{unix}-{}", kind.as_str(), self.seq);
        while self.sessions.iter().any(|s| s.id == id) {
            self.seq = self.seq.wrapping_add(1);
            id = format!("{}-{unix}-{}", kind.as_str(), self.seq);
        }
        let path = self.sessions_dir().join(format!("{id}.jsonl"));
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .map_err(|e| ActivityError::Io(e.to_string()))?;
        let mut writer = BufWriter::new(file);
        let header = SessionHeader {
            kind_tag: "header".into(),
            id: id.clone(),
            kind,
            link_ids: link_ids.clone(),
            features_per_link: FEATURES_PER_LINK,
            started_unix: unix,
        };
        let line = serde_json::to_string(&header).map_err(|e| ActivityError::Io(e.to_string()))?;
        writeln!(writer, "{line}").and_then(|()| writer.flush()).map_err(|e| ActivityError::Io(e.to_string()))?;
        let a = ActiveSession {
            id,
            kind,
            link_ids,
            started_unix: unix,
            writer,
            samples: Vec::new(),
            skipped: 0,
            no_label: 0,
            no_wifi: 0,
            truncated: false,
        };
        let p = a.progress();
        self.active = Some(a);
        Ok(p)
    }

    pub fn stop_session(&mut self) -> Result<SessionProgress, ActivityError> {
        let mut a = self.active.take().ok_or_else(|| ActivityError::Conflict("no session is recording".into()))?;
        a.writer.flush().map_err(|e| ActivityError::Io(e.to_string()))?;
        let p = a.progress();
        self.sessions.push(Arc::new(Session {
            id: a.id,
            kind: a.kind,
            link_ids: a.link_ids,
            started_unix: a.started_unix,
            samples: a.samples,
        }));
        Ok(p)
    }

    /// Delete a stored session (file + memory). `id` must already be path-safe.
    pub fn delete_session(&mut self, id: &str) -> Result<(), ActivityError> {
        if self.active.as_ref().is_some_and(|a| a.id == id) {
            return Err(ActivityError::Conflict("session is recording; stop it first".into()));
        }
        let pos = self
            .sessions
            .iter()
            .position(|s| s.id == id)
            .ok_or_else(|| ActivityError::NotFound("no such session".into()))?;
        std::fs::remove_file(self.sessions_dir().join(format!("{id}.jsonl")))
            .map_err(|e| ActivityError::Io(e.to_string()))?;
        self.sessions.remove(pos);
        Ok(())
    }

    pub fn sessions(&self) -> &[Arc<Session>] {
        &self.sessions
    }

    pub fn recording(&self) -> Option<SessionProgress> {
        self.active.as_ref().map(ActiveSession::progress)
    }

    // ── model ───────────────────────────────────────────────────────────────

    pub fn model(&self) -> Option<&Model> {
        self.model.as_ref()
    }

    /// Install (and persist, atomically) a freshly trained model.
    pub fn set_model(&mut self, model: Model) -> Result<(), ActivityError> {
        std::fs::create_dir_all(&self.dir).map_err(|e| ActivityError::Io(e.to_string()))?;
        let path = self.model_path();
        let tmp = path.with_extension("json.tmp");
        let bytes = serde_json::to_vec(&model).map_err(|e| ActivityError::Io(e.to_string()))?;
        std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, &path)).map_err(|e| ActivityError::Io(e.to_string()))?;
        self.model = Some(model);
        self.recent.clear();
        Ok(())
    }

    /// Smoothed WiFi-only prediction over the last [`SMOOTH_SECS`].
    pub fn live(&self) -> LivePrediction {
        let empty = |status| LivePrediction {
            status,
            best: None,
            confidence: 0.0,
            probs: BTreeMap::new(),
            window_secs: SMOOTH_SECS,
        };
        let Some(m) = self.model.as_ref() else { return empty("no_model") };
        if self.recent.is_empty() {
            return empty("waiting_for_links");
        }
        let n = self.recent.len() as f32;
        let mut avg = vec![0.0f32; m.labels.len()];
        for v in &self.recent {
            for (a, x) in avg.iter_mut().zip(v) {
                *a += x / n;
            }
        }
        let b = argmax_first(&avg);
        LivePrediction {
            status: if avg[b] >= 0.5 { "ok" } else { "unsure" },
            best: Some(m.labels[b].clone()),
            confidence: f64::from(avg[b]),
            probs: m.labels.iter().cloned().zip(avg.iter().map(|&p| f64::from(p))).collect(),
            window_secs: SMOOTH_SECS,
        }
    }

    pub fn live_link_summary(&self, now: Instant) -> Vec<serde_json::Value> {
        let t = now.saturating_duration_since(self.epoch).as_secs_f64();
        self.live_links(t, 0.0)
            .into_iter()
            .map(|(id, rate)| serde_json::json!({ "id": id, "rate_hz": rate, "usable": rate >= MIN_LINK_RATE_HZ }))
            .collect()
    }
}

// ── Disk loading ─────────────────────────────────────────────────────────────

/// Parse one session file. Malformed/oversized files and bad lines are skipped.
pub fn parse_session(reader: impl BufRead) -> Option<Session> {
    let mut lines = reader.lines();
    let header: SessionHeader = serde_json::from_str(&lines.next()?.ok()?).ok()?;
    if header.kind_tag != "header"
        || header.features_per_link != FEATURES_PER_LINK
        || header.link_ids.is_empty()
        || header.link_ids.len() > MAX_SESSION_LINKS
        || crate::path_safety::safe_id(&header.id).is_err()
    {
        return None;
    }
    let dim = header.link_ids.len() * FEATURES_PER_LINK;
    let mut samples = Vec::new();
    for line in lines {
        let Ok(line) = line else { break };
        let Ok(s) = serde_json::from_str::<Sample>(&line) else { continue };
        if s.x.len() == dim && s.x.iter().all(|v| v.is_finite()) && validate_label(&s.label).is_ok() {
            samples.push(s);
            if samples.len() >= MAX_SAMPLES_PER_SESSION {
                break;
            }
        }
    }
    Some(Session { id: header.id, kind: header.kind, link_ids: header.link_ids, started_unix: header.started_unix, samples })
}

fn load_sessions(dir: &Path) -> Vec<Arc<Session>> {
    let Ok(rd) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut paths: Vec<PathBuf> = rd
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
        .collect();
    paths.sort();
    let mut out: Vec<Arc<Session>> = Vec::new();
    let mut total = 0usize;
    for p in paths.into_iter().take(MAX_SESSIONS) {
        if std::fs::metadata(&p).map_or(true, |m| m.len() > MAX_SESSION_FILE_BYTES) {
            continue;
        }
        let Some(s) = File::open(&p).ok().and_then(|f| parse_session(BufReader::new(f))) else { continue };
        // File name and header id must agree, so deletes can never target another file.
        if p.file_stem().and_then(|x| x.to_str()) != Some(s.id.as_str()) {
            continue;
        }
        total += s.samples.len();
        if total > MAX_TOTAL_SAMPLES {
            break;
        }
        out.push(Arc::new(s));
    }
    out
}

// ── Tests ────────────────────────────────────────────────────────────────────
//
// All numbers below come from SYNTHETIC frames generated in this file. They
// verify the maths and the train/test discipline, NOT real-world accuracy.

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random in [0,1).
    fn rnd(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64) / (1u64 << 31) as f64
    }

    /// Synthetic 52-tone frame: a class-specific shape plus noise.
    fn frame(class: usize, seed: &mut u64, noise: f64) -> Vec<f64> {
        (0..N_TONES)
            .map(|k| {
                let shape = 10.0 + 4.0 * ((k as f64 * 0.3) * (class as f64 + 1.0)).sin();
                shape * (1.0 + noise * (rnd(seed) - 0.5))
            })
            .collect()
    }

    fn sample_for(class: usize, link_ids: &[String], seed: &mut u64, noise: f64) -> Vec<f32> {
        let mut x = Vec::new();
        for _ in link_ids {
            let frames: Vec<Vec<f64>> = (0..8).map(|_| frame(class, seed, noise)).collect();
            let refs: Vec<&[f64]> = frames.iter().map(Vec::as_slice).collect();
            x.extend(window_features(&refs).unwrap());
        }
        x
    }

    fn session(id: &str, kind: SessionKind, links: &[&str], classes: &[&str], per: usize, seed: &mut u64, noise: f64) -> Session {
        let link_ids: Vec<String> = links.iter().map(|s| s.to_string()).collect();
        let mut samples = Vec::new();
        for (ci, c) in classes.iter().enumerate() {
            for i in 0..per {
                samples.push(Sample { t: i as f64, label: c.to_string(), x: sample_for(ci, &link_ids, seed, noise) });
            }
        }
        Session { id: id.into(), kind, link_ids, started_unix: 0, samples }
    }

    #[test]
    fn features_have_expected_dimension_and_need_min_frames() {
        let mut s = 1;
        let frames: Vec<Vec<f64>> = (0..MIN_FRAMES).map(|_| frame(0, &mut s, 0.1)).collect();
        let refs: Vec<&[f64]> = frames.iter().map(Vec::as_slice).collect();
        assert_eq!(window_features(&refs).unwrap().len(), FEATURES_PER_LINK);
        assert!(window_features(&refs[..MIN_FRAMES - 1]).is_none());
    }

    #[test]
    fn features_are_gain_invariant() {
        let mut s = 7;
        let frames: Vec<Vec<f64>> = (0..10).map(|_| frame(2, &mut s, 0.2)).collect();
        let refs: Vec<&[f64]> = frames.iter().map(Vec::as_slice).collect();
        let base = window_features(&refs).unwrap();
        // Scale each frame by its own (different) AGC gain.
        let scaled: Vec<Vec<f64>> =
            frames.iter().enumerate().map(|(i, f)| f.iter().map(|a| a * (0.5 + i as f64 * 0.7)).collect()).collect();
        let srefs: Vec<&[f64]> = scaled.iter().map(Vec::as_slice).collect();
        let got = window_features(&srefs).unwrap();
        for (a, b) in base.iter().zip(&got) {
            assert!((a - b).abs() < 1e-4, "feature changed under gain: {a} vs {b}");
        }
    }

    #[test]
    fn std_and_delta_track_motion() {
        let still: Vec<Vec<f64>> = (0..8).map(|_| vec![5.0; N_TONES]).collect();
        let moving: Vec<Vec<f64>> = (0..8)
            .map(|i| (0..N_TONES).map(|k| 5.0 + if (i + k) % 2 == 0 { 2.0 } else { -2.0 }).collect())
            .collect();
        let f = |v: &Vec<Vec<f64>>| window_features(&v.iter().map(Vec::as_slice).collect::<Vec<_>>()).unwrap();
        let (a, b) = (f(&still), f(&moving));
        assert!(a[N_TONES..].iter().all(|&x| x.abs() < 1e-6));
        assert!(b[N_TONES..2 * N_TONES].iter().all(|&x| x > 0.1));
        assert!(b[2 * N_TONES..].iter().all(|&x| x > 0.1));
    }

    #[test]
    fn bad_frames_are_rejected() {
        assert!(normalise_frame(&[1.0; 51]).is_none());
        assert!(normalise_frame(&[0.0; N_TONES]).is_none());
        let mut nan = vec![1.0; N_TONES];
        nan[3] = f64::NAN;
        assert!(normalise_frame(&nan).is_none());
    }

    #[test]
    fn knn_picks_majority_of_k_nearest() {
        let rows = vec![vec![0.0], vec![0.1], vec![0.2], vec![5.0], vec![5.1]];
        let y = vec![0, 0, 0, 1, 1];
        assert_eq!(argmax_first(&knn_votes(&rows, &y, 2, &[0.05], 3)), 0);
        assert_eq!(argmax_first(&knn_votes(&rows, &y, 2, &[5.0], 1)), 1);
        // k=5 over everything: 3 vs 2 votes regardless of the query.
        let v = knn_votes(&rows, &y, 2, &[5.0], 5);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.4).abs() < 1e-6);
        // k larger than the data is clamped.
        assert_eq!(knn_votes(&rows, &y, 2, &[0.0], 99).iter().sum::<f32>().round(), 1.0);
    }

    #[test]
    fn knn_tie_is_deterministic() {
        let rows = vec![vec![0.0], vec![1.0]];
        let v = knn_votes(&rows, &[0, 1], 2, &[0.5], 2);
        assert_eq!(argmax_first(&v), 0);
    }

    #[test]
    fn standardiser_uses_train_statistics() {
        let a = [1.0f32, 10.0];
        let b = [3.0f32, 30.0];
        let (m, s) = fit_standardiser(&[&a, &b]);
        assert_eq!(m, vec![2.0, 20.0]);
        let z = standardise(&a, &m, &s);
        assert!((z[0] + 1.0).abs() < 1e-3 && (z[1] + 1.0).abs() < 1e-3);
    }

    #[test]
    fn label_validation() {
        assert!(validate_label("walking").is_ok());
        assert!(validate_label("sit-down_2").is_ok());
        assert!(validate_label("").is_err());
        assert!(validate_label("Walking").is_err());
        assert!(validate_label("a b").is_err());
        assert!(validate_label("../x").is_err());
        assert!(validate_label(&"a".repeat(MAX_LABEL_LEN + 1)).is_err());
    }

    #[test]
    fn evaluation_separable_synthetic_and_baselines() {
        let mut s = 42;
        let classes = ["sitting", "walking", "waving", "nobody"];
        let train = session("train-1", SessionKind::Train, &["R>1", "2>1"], &classes, 30, &mut s, 0.1);
        let test = session("test-1", SessionKind::Test, &["R>1", "2>1"], &classes, 10, &mut s, 0.1);
        let ev = evaluate(&[train, test], 7).unwrap();
        assert_eq!(ev.n_train, 120);
        assert_eq!(ev.n_test, 40);
        assert_eq!(ev.train_sessions, vec!["train-1"]);
        assert_eq!(ev.test_sessions, vec!["test-1"]);
        // SYNTHETIC: classes differ strongly, so this must be near perfect.
        assert!(ev.accuracy > 0.95, "synthetic accuracy {}", ev.accuracy);
        assert!((ev.chance_baseline - 0.25).abs() < 1e-12);
        // Balanced test set: majority baseline = 1/4 too.
        assert!((ev.majority_baseline - 0.25).abs() < 1e-12);
        let total: u32 = ev.confusion.values().flat_map(|r| r.values()).sum();
        assert_eq!(total as usize, ev.n_test);
        assert_eq!(ev.per_class.len(), 4);
        assert_eq!(ev.link_importance.len(), 2);
    }

    #[test]
    fn majority_baseline_uses_training_majority() {
        let mut s = 5;
        // Train: 30 "a" + 10 "b"; test: 5 "a" + 15 "b".
        let mut train = session("train-1", SessionKind::Train, &["R>1"], &["a", "b"], 10, &mut s, 0.1);
        train.samples.extend(session("x", SessionKind::Train, &["R>1"], &["a"], 20, &mut s, 0.1).samples);
        let mut test = session("test-1", SessionKind::Test, &["R>1"], &["a"], 5, &mut s, 0.1);
        test.samples.extend(session("y", SessionKind::Test, &["R>1"], &["b"], 15, &mut s, 0.1).samples);
        let ev = evaluate(&[train, test], 3).unwrap();
        assert_eq!(ev.majority_class, "a");
        assert!((ev.majority_baseline - 5.0 / 20.0).abs() < 1e-12);
    }

    #[test]
    fn evaluation_never_scores_training_windows() {
        let mut s = 9;
        // Train session carries all the signal; the test session holds random,
        // mislabelled windows. If evaluation leaked train windows into the
        // score, accuracy would be high; it must reflect only the test data.
        let train = session("train-1", SessionKind::Train, &["R>1"], &["a", "b"], 30, &mut s, 0.05);
        let mut test = session("test-1", SessionKind::Test, &["R>1"], &["a", "b"], 10, &mut s, 0.05);
        for smp in &mut test.samples {
            smp.label = if smp.label == "a" { "b".into() } else { "a".into() }; // deliberately wrong
        }
        let ev = evaluate(&[train, test], 7).unwrap();
        assert_eq!(ev.n_test, 20);
        assert!(ev.accuracy < 0.1, "test labels were inverted, accuracy must collapse: {}", ev.accuracy);
    }

    #[test]
    fn evaluation_requires_a_separate_test_session() {
        let mut s = 3;
        let train = session("train-1", SessionKind::Train, &["R>1"], &["a", "b"], 20, &mut s, 0.1);
        assert!(matches!(evaluate(&[train], 7), Err(ActivityError::Conflict(_))));
    }

    #[test]
    fn train_model_ignores_test_sessions() {
        let mut s = 11;
        let train = session("train-1", SessionKind::Train, &["R>1"], &["a", "b"], 10, &mut s, 0.1);
        let test = session("test-1", SessionKind::Test, &["R>1"], &["a", "b", "c"], 10, &mut s, 0.1);
        let m = train_model(&[train, test], 7, 0).unwrap();
        assert_eq!(m.labels, vec!["a", "b"], "label 'c' exists only in the test session");
        assert_eq!(m.train_sessions, vec!["train-1"]);
        assert_eq!(m.n_train_windows, 20);
        assert!(m.valid());
    }

    #[test]
    fn incompatible_link_sets_are_excluded_not_mixed() {
        let mut s = 13;
        let train = session("train-1", SessionKind::Train, &["R>1"], &["a", "b"], 20, &mut s, 0.1);
        let ok_test = session("test-1", SessionKind::Test, &["R>1"], &["a", "b"], 5, &mut s, 0.1);
        let bad_test = session("test-2", SessionKind::Test, &["R>2"], &["a", "b"], 5, &mut s, 0.1);
        let ev = evaluate(&[train, ok_test, bad_test], 7).unwrap();
        assert_eq!(ev.excluded_sessions, vec!["test-2"]);
        assert_eq!(ev.n_test, 10);
    }

    #[test]
    fn ablation_flags_the_informative_link() {
        let mut s = 21;
        // Link "A" carries the class shape; link "B" is class-independent noise.
        let classes = ["a", "b"];
        let build = |id: &str, kind, per, s: &mut u64| {
            let mut samples = Vec::new();
            for (ci, c) in classes.iter().enumerate() {
                for i in 0..per {
                    let mut x = sample_for(ci, &["A".to_string()], s, 0.05);
                    x.extend(sample_for(0, &["B".to_string()], s, 0.6)); // class-independent
                    samples.push(Sample { t: i as f64, label: c.to_string(), x });
                }
            }
            Session { id: id.into(), kind, link_ids: vec!["A".into(), "B".into()], started_unix: 0, samples }
        };
        let tr = build("train-1", SessionKind::Train, 40, &mut s);
        let te = build("test-1", SessionKind::Test, 20, &mut s);
        let ev = evaluate(&[tr, te], 7).unwrap();
        let drop_of = |l: &str| ev.link_importance.iter().find(|i| i.link == l).unwrap().accuracy_drop;
        assert!(drop_of("A") > 0.3, "removing the informative link must hurt: {}", drop_of("A"));
        assert!(drop_of("B") < 0.1, "removing the noise link must barely matter: {}", drop_of("B"));
    }

    #[test]
    fn engine_records_session_from_frames_and_labels() {
        let dir = tempfile::tempdir().unwrap();
        let mut e = ActivityEngine::new(dir.path().to_path_buf());
        let mut seed = 99;
        // No link yet -> cannot start.
        assert!(matches!(e.start_session_at(SessionKind::Train, 0.0, 1), Err(ActivityError::Conflict(_))));
        // 10 frames/s on one link for 3 s.
        for i in 0..30 {
            let t = i as f64 * 0.1;
            e.observe_at("R>1", &frame(0, &mut seed, 0.1), t);
        }
        let p = e.start_session_at(SessionKind::Train, 3.0, 1700000000).unwrap();
        assert_eq!(p.link_ids, vec!["R>1"]);
        assert!(matches!(e.start_session_at(SessionKind::Test, 3.0, 1), Err(ActivityError::Conflict(_))));
        for i in 30..80 {
            let t = i as f64 * 0.1;
            let _ = e.add_label_at("walking", t);
            e.observe_at("R>1", &frame(0, &mut seed, 0.1), t);
        }
        let done = e.stop_session().unwrap();
        assert!(done.windows >= 5, "got {} windows", done.windows);
        // Round trip through disk.
        let loaded = ActivityEngine::load(dir.path());
        assert_eq!(loaded.sessions().len(), 1);
        assert_eq!(loaded.sessions()[0].samples.len(), done.windows);
        assert_eq!(loaded.sessions()[0].kind, SessionKind::Train);
    }

    #[test]
    fn engine_skips_windows_without_confident_label() {
        let dir = tempfile::tempdir().unwrap();
        let mut e = ActivityEngine::new(dir.path().to_path_buf());
        let mut seed = 5;
        for i in 0..30 {
            e.observe_at("R>1", &frame(0, &mut seed, 0.1), i as f64 * 0.1);
        }
        e.start_session_at(SessionKind::Train, 3.0, 1).unwrap();
        for i in 30..60 {
            let t = i as f64 * 0.1;
            // Camera flips between two labels: no 60% majority.
            let _ = e.add_label_at(if i % 2 == 0 { "a" } else { "b" }, t);
            e.observe_at("R>1", &frame(0, &mut seed, 0.1), t);
        }
        let done = e.stop_session().unwrap();
        assert_eq!(done.windows, 0);
        assert!(done.no_label > 0);
    }

    #[test]
    fn model_round_trips_and_live_prediction_smooths() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = 77;
        let train = session("train-1", SessionKind::Train, &["R>1"], &["a", "b"], 20, &mut s, 0.05);
        let model = train_model(&[train], 7, 1).unwrap();
        let mut e = ActivityEngine::new(dir.path().to_path_buf());
        assert_eq!(e.live().status, "no_model");
        e.set_model(model).unwrap();
        let reloaded = ActivityEngine::load(dir.path());
        assert!(reloaded.model().is_some());
        // Stream class-0 ("a") frames; prediction must settle on "a".
        for i in 0..80 {
            e.observe_at("R>1", &frame(0, &mut s, 0.05), i as f64 * 0.1);
        }
        let live = e.live();
        assert_eq!(live.best.as_deref(), Some("a"));
        assert!(live.confidence >= 0.5);
    }

    #[test]
    fn stride_subsampling_is_bounded_and_even() {
        assert_eq!(stride_indices(5, 10), vec![0, 1, 2, 3, 4]);
        let v = stride_indices(1000, 10);
        assert_eq!(v.len(), 10);
        assert_eq!(v[0], 0);
        assert!(v.windows(2).all(|w| w[1] > w[0]));
    }

    #[test]
    fn parse_session_skips_bad_lines_and_wrong_dims() {
        let hdr = r#"{"type":"header","id":"train-1","kind":"train","link_ids":["R>1"],"features_per_link":156,"started_unix":1}"#;
        let good = serde_json::to_string(&Sample { t: 0.0, label: "a".into(), x: vec![0.0; FEATURES_PER_LINK] }).unwrap();
        let short = serde_json::to_string(&Sample { t: 0.0, label: "a".into(), x: vec![0.0; 3] }).unwrap();
        let text = format!("{hdr}\n{good}\nnot json\n{short}\n{good}\n");
        let s = parse_session(std::io::Cursor::new(text)).unwrap();
        assert_eq!(s.samples.len(), 2);
        let evil = hdr.replace("train-1", "../evil");
        assert!(parse_session(std::io::Cursor::new(evil)).is_none());
    }
}
