//! SYNTHETIC evaluation of peer-link presence: existing detectors versus the
//! NBVI/MVS/baseline-shift/breathing vote, on simulated link streams.
//!
//! Nothing here is measured on hardware. The simulator is a multipath channel
//! (static paths with slow drift, thermal noise, int8 I/Q quantisation,
//! receiver AGC steps and per-packet gain jitter, weak transmitters) plus a
//! body path whose strength falls off with the excess path length of the
//! person's position (a first-Fresnel-zone style falloff). Still-breathing is
//! a +-5 mm chest displacement at 0.25 Hz plus slow sway; moving is a fast,
//! large phase random walk.
//!
//! Reproduce:
//!   PRESENCE_EVAL_TRIALS=30 cargo test -p wifi-densepose-sensing-server \
//!       presence_eval::synthetic_report -- --ignored --nocapture
//! (add `--release` for speed; defaults to 30 trials).

use super::*;
use std::sync::atomic::Ordering;

/// xorshift64* PRNG with Box-Muller normals; deterministic per seed.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.uniform().max(1e-12), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Ornstein-Uhlenbeck process stepped at the link frame rate (10 Hz).
#[derive(Clone, Copy)]
struct Ou {
    x: f64,
    sigma: f64,
    tau_s: f64,
}

impl Ou {
    fn step(&mut self, rng: &mut Rng) -> f64 {
        let dt = 0.1;
        let a = (-dt / self.tau_s).exp();
        self.x = a * self.x + self.sigma * (1.0 - a * a).sqrt() * rng.normal();
        self.x
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Scene {
    Empty,
    StillBreathing,
    Moving,
}

const NODES: [[f64; 2]; 4] = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
/// Nodes whose transmissions are weak (2 of 4 boards, as on the real mesh).
const WEAK_TX: [usize; 2] = [2, 3];
/// Per-packet receiver gain jitter (fraction), link SNRs (dB). Tuned so the
/// *existing* detectors reproduce the measured empty-room behaviour.
const AGC_JITTER: f64 = 0.04;
const WEAK_SNR_DB: f64 = 26.0;
const STRONG_SNR_DB: f64 = 32.0;
/// Body-path gain relative to the direct path when standing on the link.
const BODY_SCALE: f64 = 0.35;
/// Long-term environment change applied before the "hours later" scene:
/// per-path gain offset (fraction) and phase offset (rad), standard deviations.
const AGED_DRIFT: f64 = 0.03;
const AGED_PHASE_RAD: f64 = 0.10;

/// `PRESENCE_EVAL_BODY` overrides [`BODY_SCALE`] for sensitivity sweeps.
fn body_scale() -> f64 {
    std::env::var("PRESENCE_EVAL_BODY").ok().and_then(|v| v.parse().ok()).unwrap_or(BODY_SCALE)
}

struct LinkSim {
    tx: u8,
    rx: u8,
    weak: bool,
    path_gain: Vec<f64>,
    path_delay_ns: Vec<f64>,
    path_phase: Vec<f64>,
    drift: Vec<Ou>,
    body_strength: f64,
    body_delay_ns: f64,
    body_phase: f64,
    sway: Ou,
    walk: Ou,
    agc_db: f64,
    next_agc_step: usize,
    t: usize,
    breath_phase: f64,
}

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

impl LinkSim {
    fn new(tx: usize, rx: usize, rng: &mut Rng) -> Self {
        let n_paths = 5;
        let mut gains = vec![1.0];
        gains.extend((1..n_paths).map(|_| rng.range(0.25, 0.7)));
        Self {
            tx: tx as u8 + 1,
            rx: rx as u8 + 1,
            weak: WEAK_TX.contains(&tx),
            path_gain: gains,
            path_delay_ns: (0..n_paths).map(|i| if i == 0 { 0.0 } else { rng.range(20.0, 160.0) }).collect(),
            path_phase: (0..n_paths).map(|_| rng.range(0.0, 6.28)).collect(),
            drift: (0..n_paths).map(|_| Ou { x: 0.0, sigma: 0.015, tau_s: 60.0 }).collect(),
            body_strength: 0.0,
            body_delay_ns: rng.range(30.0, 140.0),
            body_phase: rng.range(0.0, 6.28),
            sway: Ou { x: 0.0, sigma: 0.12, tau_s: 5.0 },
            walk: Ou { x: 0.0, sigma: 2.5, tau_s: 1.5 },
            agc_db: 0.0,
            next_agc_step: rng.range(30.0, 150.0) as usize,
            t: 0,
            breath_phase: rng.range(0.0, 6.28),
        }
    }

    /// Place the person: body strength falls off with excess path length.
    fn place_person(&mut self, pos: [f64; 2]) {
        let tx = NODES[self.tx as usize - 1];
        let rx = NODES[self.rx as usize - 1];
        let excess = dist(pos, tx) + dist(pos, rx) - dist(tx, rx);
        self.body_strength = body_scale() * (-excess / 0.6).exp();
    }

    /// Hours later: paths have drifted (temperature, furniture, doors).
    fn age(&mut self, rng: &mut Rng) {
        for d in self.drift.iter_mut() {
            d.x = AGED_DRIFT * rng.normal();
        }
        for p in self.path_phase.iter_mut() {
            *p += AGED_PHASE_RAD * rng.normal();
        }
    }

    fn frame(&mut self, scene: Scene, rng: &mut Rng) -> Vec<f64> {
        self.t += 1;
        let time_s = self.t as f64 * 0.1;
        // AGC: occasional steps of a few dB, bounded, plus per-packet jitter.
        if self.t >= self.next_agc_step {
            let step = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0][(rng.uniform() * 6.0) as usize % 6];
            self.agc_db = (self.agc_db + step).clamp(-6.0, 6.0);
            self.next_agc_step = self.t + rng.range(30.0, 150.0) as usize;
        }
        let gain = 10f64.powf(self.agc_db / 20.0) * (1.0 + AGC_JITTER * rng.normal());

        let drifts: Vec<f64> = self.drift.iter_mut().map(|d| d.step(rng)).collect();
        let sway = self.sway.step(rng);
        let walk = self.walk.step(rng);
        let body_psi = match scene {
            Scene::Empty => 0.0,
            Scene::StillBreathing => {
                0.5 * (2.0 * std::f64::consts::PI * 0.25 * time_s + self.breath_phase).sin() + sway
            }
            Scene::Moving => walk,
        };
        let body_gain = match scene {
            Scene::Empty => 0.0,
            Scene::StillBreathing => self.body_strength,
            Scene::Moving => self.body_strength * (1.0 + 0.3 * (walk / 2.5).clamp(-1.0, 1.0)),
        };

        let (level, snr_db) = if self.weak { (9.0, WEAK_SNR_DB) } else { (30.0, STRONG_SNR_DB) };
        let noise = 10f64.powf(-snr_db / 20.0);
        let mut out = Vec::with_capacity(52);
        for k in (-26i32..=-1).chain(1..=26) {
            let f_hz = f64::from(k) * 312_500.0;
            let (mut re, mut im) = (0.0, 0.0);
            for p in 0..self.path_gain.len() {
                let g = self.path_gain[p] * (1.0 + drifts[p]);
                let ph = self.path_phase[p] - 2.0 * std::f64::consts::PI * f_hz * self.path_delay_ns[p] * 1e-9;
                re += g * ph.cos();
                im += g * ph.sin();
            }
            if body_gain > 0.0 {
                let ph = self.body_phase + body_psi
                    - 2.0 * std::f64::consts::PI * f_hz * self.body_delay_ns * 1e-9;
                re += body_gain * ph.cos();
                im += body_gain * ph.sin();
            }
            let mag = (re * re + im * im).sqrt();
            re += noise * mag * rng.normal();
            im += noise * mag * rng.normal();
            let scale = level * gain;
            let (qi, qq) = ((re * scale).round().clamp(-127.0, 127.0), (im * scale).round().clamp(-127.0, 127.0));
            out.push((qi * qi + qq * qq).sqrt());
        }
        out
    }
}

fn frame_with(amps: Vec<f64>) -> Esp32Frame {
    let mut f = parse_esp32_frame(&{
        let mut b = vec![0u8; 20 + 128];
        b[0..4].copy_from_slice(&0xC511_0001u32.to_le_bytes());
        b[5] = 1;
        b[6..8].copy_from_slice(&64u16.to_le_bytes());
        b
    })
    .unwrap();
    f.amplitudes = amps;
    f
}

/// Per-signal tallies for one scene over all trials.
#[derive(Default, Clone)]
struct Tally {
    trials: usize,
    /// Trials with at least one alarm run of >= EPISODE_FRAMES frames.
    trials_with_episode: usize,
    /// Trials where the signal was on for >= half of the scored frames.
    trials_mostly_on: usize,
    on_frames: usize,
    frames: usize,
}

const EPISODE_FRAMES: usize = 50; // 5 s at 10 Hz
const CAL_FRAMES: usize = 600;
const SETTLE_FRAMES: usize = 100;
const SCORE_FRAMES: usize = 600;

const SIGNALS: [&str; 8] = [
    "before: motion presence",
    "before: still_presence (wander)",
    "before: either",
    "after:  MVS motion vote",
    "after:  still vote (shift+breath)",
    "after:  advisory presence",
    "after:  advisory still only",
    "combined: before OR advisory",
];

fn run_trial(seed: u64, with_breathing_signal: bool) -> Vec<(Scene, [Vec<bool>; 8])> {
    let mut rng = Rng::new(seed);
    let mut sims: Vec<LinkSim> = Vec::new();
    for tx in 0..4 {
        for rx in 0..4 {
            if tx != rx {
                sims.push(LinkSim::new(tx, rx, &mut rng));
            }
        }
    }
    let now = std::time::Instant::now();
    let mut links: HashMap<(u8, u8), PeerLinkState> = HashMap::new();
    for s in &sims {
        links.insert((s.tx, s.rx), PeerLinkState::new(now));
    }
    // Empty-room calibration, as the operator would run it.
    for l in links.values_mut() {
        l.begin_calibration();
    }
    for _ in 0..CAL_FRAMES {
        for s in sims.iter_mut() {
            let amps = s.frame(Scene::Empty, &mut rng);
            links.get_mut(&(s.tx, s.rx)).unwrap().observe(&frame_with(amps), now);
        }
    }
    let keys: Vec<(u8, u8)> = links.keys().copied().collect();
    for k in keys {
        links.get_mut(&k).unwrap().finish_calibration(k.0, k.1);
    }
    // Person position for the occupied scenes.
    let pos = [rng.range(0.5, 3.5), rng.range(0.5, 3.5)];
    for s in sims.iter_mut() {
        s.place_person(pos);
        if !with_breathing_signal {
            s.breath_phase = 0.0;
        }
    }

    let plan = [
        (Scene::Empty, false),
        (Scene::StillBreathing, false),
        (Scene::Empty, false),
        (Scene::Moving, false),
        (Scene::Empty, true),
    ];
    let mut result = Vec::new();
    for (i, (scene, aged)) in plan.into_iter().enumerate() {
        if aged {
            for s in sims.iter_mut() {
                s.age(&mut rng);
            }
        }
        let settle = if i == 0 { 0 } else { SETTLE_FRAMES };
        let mut series: [Vec<bool>; 8] = Default::default();
        for step in 0..(settle + SCORE_FRAMES) {
            for s in sims.iter_mut() {
                let amps = s.frame(scene, &mut rng);
                links.get_mut(&(s.tx, s.rx)).unwrap().observe(&frame_with(amps), now);
            }
            if step < settle {
                continue;
            }
            let r = peer_room_summary(&links, now);
            let adv = &r["advisory"];
            let motion = r["presence"].as_bool().unwrap();
            let still = r["still_presence"].as_bool().unwrap();
            let a_presence = adv["presence"].as_bool().unwrap();
            let flags = [
                motion,
                still,
                motion || still,
                adv["motion"].as_bool().unwrap(),
                adv["still_presence"].as_bool().unwrap(),
                a_presence,
                adv["still_presence"].as_bool().unwrap(),
                motion || still || a_presence,
            ];
            for (v, f) in series.iter_mut().zip(flags) {
                v.push(f);
            }
        }
        result.push((scene, series));
    }
    result
}

fn longest_run(v: &[bool]) -> usize {
    let (mut best, mut cur) = (0, 0);
    for &b in v {
        cur = if b { cur + 1 } else { 0 };
        best = best.max(cur);
    }
    best
}

fn fmt(t: &Tally) -> String {
    format!(
        "episodes {:>2}/{:<2}  mostly-on {:>2}/{:<2}  duty {:>5.1}%",
        t.trials_with_episode,
        t.trials,
        t.trials_mostly_on,
        t.trials,
        100.0 * t.on_frames as f64 / t.frames.max(1) as f64
    )
}

#[test]
#[ignore = "SYNTHETIC evaluation report; run with --ignored --nocapture"]
fn synthetic_report() {
    let trials: usize =
        std::env::var("PRESENCE_EVAL_TRIALS").ok().and_then(|v| v.parse().ok()).unwrap_or(30);
    let mode = std::env::var("PRESENCE_EVAL_MODE").unwrap_or_else(|_| "advisory".into());
    presence_features::set_mode(presence_features::Mode::parse(&mode).unwrap());

    // [scene index in plan][signal]
    let mut tallies: Vec<Vec<Tally>> = vec![vec![Tally::default(); 8]; 5];
    let workers = std::thread::available_parallelism().map_or(2, |n| n.get()).min(8);
    let next = std::sync::atomic::AtomicU64::new(0);
    let results = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let seed = next.fetch_add(1, Ordering::Relaxed);
                if seed >= trials as u64 {
                    break;
                }
                let r = run_trial(1000 + seed, true);
                results.lock().unwrap().push(r);
            });
        }
    });
    for trial in results.into_inner().unwrap() {
        for (i, (_, series)) in trial.into_iter().enumerate() {
            for (j, v) in series.iter().enumerate() {
                let t = &mut tallies[i][j];
                t.trials += 1;
                t.frames += v.len();
                let on = v.iter().filter(|&&b| b).count();
                t.on_frames += on;
                t.trials_with_episode += usize::from(longest_run(v) >= EPISODE_FRAMES);
                t.trials_mostly_on += usize::from(on * 2 >= v.len());
            }
        }
    }
    let names = ["empty (right after calibration)", "still, breathing person", "empty again (after person left)", "moving person", "empty, hours later (environment drifted, no recalibration)"];
    println!("\nSYNTHETIC peer-link presence evaluation: {trials} trials, 12 links (6 weak), 60 s calibration,");
    println!("{SCORE_FRAMES} scored frames (60 s) per scene. Room-level output. NOT hardware data.");
    println!("body scale (direct-path ratio when standing on a link): {}", body_scale());
    for (i, name) in names.iter().enumerate() {
        println!("\n== {name} ==");
        for (j, sig) in SIGNALS.iter().enumerate() {
            println!("  {sig:<36} {}", fmt(&tallies[i][j]));
        }
    }
}

/// Frames per phase of the occupancy-latch scenario (10 Hz).
const LATCH_WALK_IN: usize = 100; // 10 s
const LATCH_SIT: usize = 6000; // 10 min
const LATCH_GONE: usize = 6000; // 10 min

/// One latch trial: calibrate, walk in, sit still 10 min, leave for 10 min,
/// then 10 min of an empty room after environment drift. Returns, per hold:
/// (fraction of the sitting phase reported present without the latch, with
/// it, seconds until release after leaving, fraction of the drifted empty
/// phase held occupied).
fn run_latch_trial(seed: u64, holds_s: &[u64]) -> Vec<(f64, f64, Option<f64>, f64)> {
    let mut rng = Rng::new(seed);
    let mut sims: Vec<LinkSim> = Vec::new();
    for tx in 0..4 {
        for rx in 0..4 {
            if tx != rx {
                sims.push(LinkSim::new(tx, rx, &mut rng));
            }
        }
    }
    // A running clock at the 10 Hz frame rate, so link freshness and
    // delivery rate behave as on the real mesh.
    let start = std::time::Instant::now();
    let mut frame = 0u64;
    let mut tick = || {
        frame += 1;
        start + std::time::Duration::from_millis(100 * frame)
    };
    let mut links: HashMap<(u8, u8), PeerLinkState> = HashMap::new();
    for s in &sims {
        links.insert((s.tx, s.rx), PeerLinkState::new(start));
    }
    for l in links.values_mut() {
        l.begin_calibration();
    }
    for _ in 0..CAL_FRAMES {
        let t = tick();
        for s in sims.iter_mut() {
            let amps = s.frame(Scene::Empty, &mut rng);
            links.get_mut(&(s.tx, s.rx)).unwrap().observe(&frame_with(amps), t);
        }
    }
    let keys: Vec<(u8, u8)> = links.keys().copied().collect();
    for k in keys {
        links.get_mut(&k).unwrap().finish_calibration(k.0, k.1);
    }
    let pos = [rng.range(0.5, 3.5), rng.range(0.5, 3.5)];
    for s in sims.iter_mut() {
        s.place_person(pos);
    }

    let mut latches: Vec<occupancy_latch::OccupancyLatch> = holds_s
        .iter()
        .map(|&h| occupancy_latch::OccupancyLatch::new(std::time::Duration::from_secs(h)))
        .collect();
    let mut sit_raw = 0usize;
    let mut sit_held = vec![0usize; holds_s.len()];
    let mut release: Vec<Option<usize>> = vec![None; holds_s.len()];
    let mut aged_held = vec![0usize; holds_s.len()];
    let plan = [
        (Scene::Moving, LATCH_WALK_IN, false),
        (Scene::StillBreathing, LATCH_SIT, false),
        (Scene::Empty, LATCH_GONE, false),
        (Scene::Empty, LATCH_GONE, true),
    ];
    for (phase, (scene, n, aged)) in plan.into_iter().enumerate() {
        if aged {
            for s in sims.iter_mut() {
                s.age(&mut rng);
            }
        }
        for i in 0..n {
            let t = tick();
            for s in sims.iter_mut() {
                let amps = s.frame(scene, &mut rng);
                links.get_mut(&(s.tx, s.rx)).unwrap().observe(&frame_with(amps), t);
            }
            let ev = peer_room_evidence(&links, t);
            if phase == 1 && (ev.motion || ev.still_room) {
                sit_raw += 1;
            }
            for (j, l) in latches.iter_mut().enumerate() {
                let held = l.observe(ev, t);
                match phase {
                    1 => sit_held[j] += usize::from(held),
                    2 if !held && release[j].is_none() => release[j] = Some(i),
                    2 if held => release[j] = None,
                    3 => aged_held[j] += usize::from(held),
                    _ => {}
                }
            }
        }
    }
    (0..holds_s.len())
        .map(|j| {
            (
                sit_raw as f64 / LATCH_SIT as f64,
                sit_held[j] as f64 / LATCH_SIT as f64,
                release[j].map(|f| f as f64 / 10.0),
                aged_held[j] as f64 / LATCH_GONE as f64,
            )
        })
        .collect()
}

#[test]
#[ignore = "SYNTHETIC occupancy-latch report; run with --ignored --nocapture"]
fn synthetic_latch_report() {
    let trials: usize =
        std::env::var("PRESENCE_EVAL_TRIALS").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    let mode = std::env::var("PRESENCE_EVAL_MODE").unwrap_or_else(|_| "advisory".into());
    presence_features::set_mode(presence_features::Mode::parse(&mode).unwrap());
    let holds = [60u64, 180, 300];
    let workers = std::thread::available_parallelism().map_or(2, |n| n.get()).min(8);
    let next = std::sync::atomic::AtomicU64::new(0);
    let results = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let seed = next.fetch_add(1, Ordering::Relaxed);
                if seed >= trials as u64 {
                    break;
                }
                let r = run_latch_trial(2000 + seed, &holds);
                results.lock().unwrap().push(r);
            });
        }
    });
    let results = results.into_inner().unwrap();
    println!("\nSYNTHETIC occupancy latch: {trials} trials, mode {mode}. Walk in 10 s, sit still 10 min,");
    println!("leave for 10 min, then 10 min empty after environment drift. NOT hardware data.");
    let raw: f64 = results.iter().map(|r| r[0].0).sum::<f64>() / results.len() as f64;
    println!("  sitting, reported present without latch (room motion or still vote): {:5.1}%", 100.0 * raw);
    for (j, h) in holds.iter().enumerate() {
        let n = results.len() as f64;
        let held: f64 = results.iter().map(|r| r[j].1).sum::<f64>() / n;
        let rel: Vec<f64> = results.iter().filter_map(|r| r[j].2).collect();
        let never = results.iter().filter(|r| r[j].2.is_none()).count();
        let mean_rel = rel.iter().sum::<f64>() / rel.len().max(1) as f64;
        let aged: f64 = results.iter().map(|r| r[j].3).sum::<f64>() / n;
        let aged_any = results.iter().filter(|r| r[j].3 > 0.0).count();
        println!(
            "  hold {h:>3} s: sitting held {:5.1}% | release after leaving mean {:6.1} s (never released in {never}/{}) | drifted empty held {:5.1}% (any in {aged_any}/{})",
            100.0 * held,
            mean_rel,
            results.len(),
            100.0 * aged,
            results.len()
        );
    }
}

#[test]
fn smoke_one_trial_runs_and_empty_room_is_quiet_for_the_new_vote() {
    presence_features::set_mode(presence_features::Mode::Advisory);
    let result = run_trial(7, true);
    assert_eq!(result.len(), 5);
    let (scene, series) = &result[0];
    assert_eq!(*scene, Scene::Empty);
    assert_eq!(series[5].len(), SCORE_FRAMES);
    // Fresh calibration, nobody home: the advisory vote must not hold an alarm.
    assert!(longest_run(&series[5]) < EPISODE_FRAMES, "empty room raised an advisory alarm");
    // A moving person must be seen by the motion vote in at least some frames.
    let (_, moving) = &result[3];
    assert!(moving[3].iter().any(|&b| b), "MVS never fired for a moving person");
}

#[test]
#[ignore = "debug dump of per-link features in the empty scene"]
fn debug_links() {
    let mut rng = Rng::new(1000);
    let mut sims: Vec<LinkSim> = Vec::new();
    for tx in 0..4 {
        for rx in 0..4 {
            if tx != rx {
                sims.push(LinkSim::new(tx, rx, &mut rng));
            }
        }
    }
    let now = std::time::Instant::now();
    let mut links: HashMap<(u8, u8), PeerLinkState> = HashMap::new();
    for s in &sims {
        links.insert((s.tx, s.rx), PeerLinkState::new(now));
    }
    for l in links.values_mut() {
        l.begin_calibration();
    }
    for _ in 0..CAL_FRAMES {
        for s in sims.iter_mut() {
            let amps = s.frame(Scene::Empty, &mut rng);
            links.get_mut(&(s.tx, s.rx)).unwrap().observe(&frame_with(amps), now);
        }
    }
    let keys: Vec<(u8, u8)> = links.keys().copied().collect();
    for k in keys {
        links.get_mut(&k).unwrap().finish_calibration(k.0, k.1);
    }
    let mut wmax = vec![0.0f64; sims.len()];
    for _ in 0..600 {
        for (i, s) in sims.iter_mut().enumerate() {
            let amps = s.frame(Scene::Empty, &mut rng);
            let l = links.get_mut(&(s.tx, s.rx)).unwrap();
            l.observe(&frame_with(amps), now);
            wmax[i] = wmax[i].max(l.subspace.wander().unwrap_or(0.0));
        }
    }
    for (i, s) in sims.iter().enumerate() {
        let l = &links[&(s.tx, s.rx)];
        println!(
            "{}->{} weak={} templates={} wander_now={:.3} wander_max={:.3} floor={:?} fsm_active={} stage={:?} feat={}",
            s.tx, s.rx, s.weak, l.subspace.template_count(), l.subspace.wander().unwrap_or(-1.0), wmax[i],
            l.subspace.wander_floor(), l.fsm.is_active(), l.fsm.stage(), l.features_json()
        );
    }
}
