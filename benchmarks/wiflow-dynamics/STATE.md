# Action-Conditioned Dynamics Model — Objective State

**Purpose of this file:** a later session resumes mid-flight from here. Read it first.

Last updated: 2026-09-08 (session start)
Worktree: `/home/ruvultra/projects/ruview-worktrees/wiflow-dynamics`
Branch: `feat/wiflow-action-conditioned-dynamics` (off RuView `d6407ae0`)

## Objective (chartered — do not silently widen)

Implement an **action-conditioned dynamics model** for WiFi-CSI → pose/occupancy with
trustworthy evaluation on the existing WiFlow-STD benchmark. The gap being filled:
the stack currently has only constant-velocity extrapolation and deterministic
kinematics; neither gives `p(observation | intervention)`, which
information-gain-driven experiment selection requires.

**Explicit NON-goal:** beating SOTA. No number from this work may be reported as
SOTA-beating. This repo has already been burned by exactly that (see Retraction below).

## Framing that must not drift

This is **active sensing / masking on offline data**. The "action" selects which
*already-recorded* coordinates of an observation are revealed to the model. It is
**not** a causal intervention on the world. The dataset is observational; nothing here
licenses a causal claim. State it this way in every write-up.

## Ground truth (VERIFIED by reading files, 2026-09-08)

| Fact | Value | How verified |
|---|---|---|
| Benchmark dir | `ruview-worktrees/ruforecast-rust/benchmarks/wiflow-std/` | ls |
| Staged assets | `~/wiflow-std-bench/` | ls |
| Dataset | `~/wiflow-std-bench/preprocessed_csi_data/` → kagglehub cache | ls -la (symlink) |
| `csi_windows.npy` | 15,552,000,128 B = 360,000 × 540 × 20 fp32 | ls -la, arithmetic |
| `all_keypoints.npy` | 43,200,128 B = 360,000 × 15 × 2 fp32 | ls -la, arithmetic |
| Honest baseline PCK@20 | **0.9608815** full-test (54,000) / **0.9661454** clean (52,560) | `results/eval_retrained.json` |
| Honest baseline MPJPE | 0.0098341 full / 0.0094328 clean | same |
| Local GPU | RTX 5080, 16,303 MiB | nvidia-smi |
| Disk free | 197 GB | df -h |

### Upstream protocol (read from source, not assumed)

- **Split** (`dataset.py:create_preprocessed_train_val_test_loaders`):
  `random.seed(42)`; `random.shuffle(file_indices)` over 500 files; floor(0.7·500)=350
  train / floor(0.15·500)=75 val / remaining 75 test. Each file = 720 windows →
  test = 54,000. **File-level split, not window-level.**
- **PCK** (`utils/metrics.py:calculate_pck`): torso norm =
  `‖target[:,2] − target[:,12]‖` (NECK_IDX=2, PELVIS_IDX=12 on the 15-kp convention),
  `clamp(min=0.01)`; `dist = ‖pred−target‖₂` per keypoint;
  `normalized = dist / norm`; PCK@t = `mean over (batch × keypoints)` of
  `normalized <= t`. Note: a **single flat mean**, not mean-of-per-frame-means.
- **MPJPE**: `mean over (batch × keypoints)` of per-keypoint L2. Normalized image
  units, **not metres**.
- **GT in NPY mode**: `all_keypoints.npy` is already filtered to 15 kp and
  already scaled — `keypoint_scale=1000.0` is applied **only in the CSV fallback
  path**, never in NPY mode. Do not divide again.
- **`enable_temporal_clean=True`** in NPY mode applies `_clean_single_frame_zeros`:
  any keypoint that is exactly (0,0) is replaced by the mean of that frame's
  non-zero keypoints. Per-frame, order-independent → replicable in Rust.
- Corruption: dataset files 487–499 (9,072 windows, 2.52%) carry NaN + amplitudes
  to 3.4e38. Masks committed at `results/nan_windows_mask.npy`,
  `results/big_windows_mask.npy`.

### The retraction this work must not repeat

An earlier "92.9% PCK@20" was **retracted**: the trainer's `pck()` used an absolute
0.2-image-unit threshold (NOT torso-normalized) and the model emitted a **constant
pose** (pred std 0.0000). A mean predictor scores 100% under that broken protocol.
True torso-normalized value: 19.1%. Any evaluation path that can score high with a
constant output is a defect.

## Charter discrepancy found (report to lead — NOT charter-invalidating)

The charter states measurement (b) is `BLOCKED-ON-DATA` with only 1,077 paired
windows. **`RESULTS.md` says otherwise:** a 2026-06-10 22:10–22:40 session collected
**2,046 paired windows** (`~/wiflow-std-bench/paired-20260610.jsonl`, 34 MB, present
on disk) and measurement (b) was **completed and measured** (`results/measurement_b.json`).
Its verdict: *no run beats the mean-pose baseline* (mean-pose 95.9% torso-PCK@20 vs
best fine-tune 65.0%) — fine-tuning transfers as optimization aid only, not feature
transfer.

Operational effect: **none**. The charter's real constraint — never touch/upload real
CSI person data — is unchanged and still binding. This work stays entirely on the
public Kaggle WiFlow dataset. Recording the discrepancy so the ledger is accurate.

## Hard constraints (binding)

- **Privacy (ADR-299):** real CSI at `RuView/data/recordings/*.csi.jsonl` and
  `RuView/v2/data/recordings/` is person data. Never upload to a rented host, never
  git-add, never allowlist to silence the guard. Only the public Kaggle dataset may
  leave this box. `paired-20260610.jsonl` is real capture → same rule.
- **Worktree:** never write into `/home/ruvultra/projects/ruflo` (operator cwd,
  unrelated repo). One writer only, this worktree.
- **Spend:** vast.ai capped at 1 instance / $25 / 6 h with mandatory trap-teardown.
  **DECISION: not renting.** The local RTX 5080 covers this workload; renting adds
  teardown and credential risk for zero upside. Revisit only if a measured local
  run proves infeasible.
- **Rust, not Python,** for new model/training components. Existing Python benchmark
  scripts may be READ and RUN. One narrow exception is taken below, and justified.

## Justified deviation: one Python fixture-generation run

Reproducing the honest baseline requires consuming a PyTorch `.pth` checkpoint and
the upstream `Dataset`/split code, including `file_mappings.pkl` (a Python pickle,
not readable from Rust without reimplementing pickle). Porting the 2.23M-param
Conv1d/Conv2d/axial-attention network to Burn purely to re-derive a number that is
already measured would consume the guards budget for no epistemic gain — and the
ported net is **not needed** for the dynamics model.

Instead: **one Python run that generates fixtures, not models.** It dumps split index
arrays, the reference predictions, and the GT exactly as `calculate_pck` consumes it,
with sha256s. This is fixture generation, not new model/training code — no Rust model
or training logic is written in Python. The Rust harness is then validated two
independent ways against those fixtures. Recording this as a deliberate, bounded
deviation.

## Milestones (guards before capability — non-negotiable order)

- [x] **M1 — Anti-Goodhart harness.** DONE 2026-09-08. `harness/` (Rust, no Burn yet).
      5/5 guard tests pass (`cargo test --release`), including the retraction
      reproduction.
- [x] **M2 — Reproduce the honest baseline.** DONE 2026-09-08. `validate` exits 0;
      raw report `results/m2_validation.json`.
- [x] **M3 — Action-conditioned dynamics model.** DONE (rung 2, joint Gaussian),
      2026-09-08. Beats the mean-pose bar by **+6.3 pts**. Calibration **marginally
      FAILS** the pre-registered criterion at one of four levels. `results/m3_m4.json`.
- [x] **M4 — Information-gain evaluation.** DONE 2026-09-08. EIG-greedy wins on
      information at every budget but **does NOT** convert that into pose accuracy —
      an honest negative, with an oracle bracket quantifying what adaptivity is worth.
- [ ] **M5 — Lineage/evidence recorded for every claim.**

## Open questions — ALL RESOLVED 2026-09-08 (MEASURED)

1. **Windows temporally ordered within a file? YES.** `config.npz`: `window_size=20`,
   `stride=20` → non-overlapping. `window_to_frame` within file 0 is `0,1,…,719`,
   strictly increasing; 720 windows/file × 500 files = 360,000. So window *i* and
   *i+1* are temporally adjacent **provided both lie in the same file** — next-observation
   dynamics is well-defined, with a file-boundary guard.
2. **540-channel layout: flat, unfactorized.** Upstream itself names it
   `NUM_SUBCARRIERS = 540` (`config.py:9`) and feeds it as a flat axis. Empirically
   (200 windows, file 0): data is [0,1]-normalized, and there is **no** antenna /
   amplitude-phase block structure — first and second halves have near-identical
   mean/std, and channel-mean autocorrelation shows no period-30 or period-180 peak.
   What it *does* show is strong local smoothness: **lag-1 autocorrelation 0.739**.
   → The defensible action unit is a **contiguous subcarrier band**, justified by that
   measured local correlation. Do NOT invent an antenna×subcarrier factorization.
3. **Mean-predictor bar on WiFlow test: 75.37% torso-PCK@20** (train-fitted;
   test-fitted oracle 75.46%, MPJPE 0.0261). MEASURED by `validate`.
   **The retrained model's 96.09% beats the bar by 20.7 points.**
   Contrast with the ESP32 set, where the bar was 95.9% and *nothing* beat it:
   WiFlow-STD contains real CSI→pose signal; the ESP32 set did not demonstrate any.
   This is the honesty bar for every subsequent claim.

## M3/M4 PRE-REGISTRATION (written 2026-09-08 ~22:55, BEFORE any model was fitted)

Pre-registered because this repo's one retracted claim came from a protocol chosen
after seeing results. Deviations from this section must be recorded as deviations.

### Representation (chosen once, not swept)

- **Bands:** 540 channels → **B = 27 contiguous bands of 20 channels**. Justified by
  the measured lag-1 channel autocorrelation (0.739): neighbours are correlated, so a
  contiguous band is a coherent probe unit. Not an antenna factorization — none exists
  in this data (verified).
- **Band feature `z`:** per-band mean over its 20 channels, **kept per frame** →
  `z ∈ R^{27×20} = R^540`. Time is deliberately NOT collapsed: averaging over the
  20 frames would destroy the motion signal the pose head needs.
- **Action:** reveal one band = reveal its 20 time-dims together. Budget = K bands.
- **Pair (t, t+1):** consecutive window indices in the same file (359,500 exist).
- **Split:** the same seed-42 file-level split as M2, so pose numbers stay comparable
  to the 75.4% bar.

### Model ladder (each rung must earn the next)

1. **Persistence** — predict `z_{t+1} = z_t`. Measured first. If one-step band
   dynamics is near-trivial (R² ≈ 0.99), say so plainly; the real content is then the
   pose conditional, not the dynamics.
2. **Joint Gaussian** over `(z_t, z_{t+1}, y_{t+1})` (1110-dim), shrinkage `λI` with
   λ chosen on **val**. Conditioning on any revealed subset is closed-form, so NLL,
   coverage and EIG are analytic and hand-checkable. **This alone satisfies M3's
   letter**: it is `p(next observation | which sensors are read)` with calibrated
   uncertainty.
3. **Neural heteroscedastic head** (Burn) — attempted only after 1–2 pass. Must beat
   the joint Gaussian's held-out NLL *and* stay calibrated, or it is reported as
   not helping.

### Metrics (fixed now)

- **Proper scoring rule:** multivariate Gaussian NLL on held-out (per-dim CRPS if cheap).
- **Calibration:** empirical coverage of central 50/80/90/95% predictive intervals vs
  nominal. **Pass = every level within ±3 percentage points.** Report the full curve,
  never a single scalar.
- **Point error:** RMSE on `z`, torso-PCK@20 + MPJPE on pose (harness, unchanged).

### Guards extended for M3 (failing checks, not reported numbers)

- **Predicted-mean degeneracy:** std of `μ_{t+1}` across t must exceed 1e-4 — same
  detector, applied to the dynamics head. A model that predicts one constant next
  observation must fail.
- **Must beat persistence** on NLL, and **must beat the joint Gaussian** for the
  neural rung to be reported as an improvement.
- **Must beat the 75.4% mean-pose bar** for any pose claim.
- Improvements smaller than the measured floor (~1.4e-5 PCK@20) are **not** reported
  as improvements.

### M4 protocol (fixed now)

- Policies compared at **identical budget K**, paired on the same test pairs, seeded:
  **EIG-greedy** vs **random-K** (mean ± CI over ≥10 seeds) vs **fixed-K**
  (chosen on **train**, never on test).
- **Bracketing check — the machinery is wrong if this fails:** an **oracle** policy
  that peeks at the true `z_{t+1}` to choose S must upper-bound, and random must
  lower-bound. EIG must sit strictly between them.
- **EIG unit tests:** EIG ≥ 0; monotone non-decreasing in |S|; greedy matches
  brute-force on a 5-band synthetic; analytic EIG matches a Monte-Carlo estimate.
- **EIG target is pose** `y_{t+1}` (the charter's actual question — "which probe most
  reduces uncertainty about pose"). EIG about unobserved bands is secondary.

### RUNG 3 PRE-REGISTRATION (written 2026-09-08 ~23:05, BEFORE the model was written)

**The question rung 3 asks, and nothing else:** *does an input-dependent `Σ(z_t, mask)`
let EIG close any of the target-peeking gap?* Rung 2 established that adaptivity with
no information buys nothing, and that the whole gap is "knowing which bands matter at
this timestep". A heteroscedastic head is the minimal thing that could supply that.

**Architecture (fixed now).** Two heads over the same inputs:
- **variance head:** `(z_t, mask) → log σ²` for the 30 pose dims. It deliberately does
  **not** see the revealed values. That makes EIG computable in closed form without
  sampling *and* without peeking — the only reason the design is shaped this way.
  Crucially `σ` now depends on `z_t`, so the EIG-optimal probe set **varies per
  sample**. That is the one thing rung 2 could not do.
- **mean head:** `(z_t, mask ⊙ z_{t+1}, mask) → μ`.
- Diagonal Gaussian predictive; loss = Gaussian NLL; masks drawn at random per sample
  per epoch so the model learns `p(· | z_t, S)` for arbitrary `S`.
- Backend: **`burn-ndarray` (CPU), chosen deliberately.** The model is tiny and 60k
  samples train in minutes; this removes the Blackwell/sm_120 CUDA build risk. Rung 3
  is a capability question, not a throughput one.

**Pass condition — ALL THREE, or rung 3 is reported as "did not help":**
1. beats the rung-2 joint Gaussian on held-out pose NLL; **and**
2. coverage within ±3 pp at **all four** levels (the criterion rung 2 marginally
   missed at 50%); **and**
3. EIG-greedy beats random on PCK@20 by **more than 2 SE**, paired, at some budget K.

**Evaluation is reused unchanged** — same 20,000 test pairs, same mean-pose bar, same
coverage protocol, same policies and paired tests. Nothing about the harness moves.

**Hard time-box.** If rung 3 does not produce a measured result tonight, it is
recorded as "not attempted / incomplete", never as a partial claim. Rung 3 is
optional; the charter is already satisfied by rungs 1–2.

### Honesty constraints agreed in advance

- **A homoscedastic Σ makes "EIG-driven" a FIXED policy.** For a Gaussian,
  EIG(S) = ½ logdet-ratio of covariance blocks — it depends only on Σ, never on the
  observed values. So with rung 2, the EIG-optimal probe set is the *same at every
  timestep*, and "EIG vs fixed" compares information-theoretic set selection against
  naive set selection. That is a legitimate result but it is **NOT adaptive sensing**,
  and must be reported in those words. Adaptive EIG requires input-dependent
  `Σ(z_t, mask)` — which is precisely what rung 3 would buy, and the only thing that
  would license the word "adaptive".
- **Pose accuracy from band features is NOT comparable to the 96% full-CSI number** —
  different input abstraction (540 → 27 band means). It is reported against the
  **75.4% mean-pose bar only**, with that non-comparability stated.
- Nothing here is a causal intervention (see "Framing that must not drift").

## Reporting rules

Every number states **MEASURED** (by this work, this run, command shown) or **CITED**
(from existing results). Never blurred. A blocked milestone honestly reported is a
success; a green number from a broken protocol is a failure. No claims about "AGI",
"SOTA", or generalization beyond what the measured protocol supports.

## Escalate to operator only for

(a) exceeding the $25 / 6 h / 1-instance caps; (b) anything requiring real-CSI person
data; (c) a finding that invalidates the charter itself.

## Progress log

- **2026-09-08 ~22:15** — Session start. Ground truth verified (table above).
  Worktree + branch created. Charter discrepancy on measurement (b) found and
  recorded. Decision: no vast.ai rental. Next: M1.

- **2026-09-08 ~22:45 — M1 + M2 COMPLETE.** All three open questions resolved.

  **Reproducibility floor discovered and MEASURED (new, load-bearing).** The first
  fixture run failed a 1e-9 assertion: PCK@20 came out 0.9608877 vs the recorded
  0.9608815, a 6.1e-6 gap. First hypothesis — float32 summation order — was **wrong**:
  replicating upstream's exact per-batch-of-256 accumulation reproduced the same
  0.9608877, and MPJPE (threshold-free) differed by 1.5e-4 relative while three
  accumulation orders of the *same* predictions agreed to 5e-10. So the predictions
  themselves differed. `scripts/diag_reproducibility_floor.py` then tested TF32
  directly: toggling it alone moves PCK@20 by **1.36e-5** and MPJPE by **5.7e-6**, and
  the recorded 2026-06-10 value falls *between* the TF32-on (0.9608877) and TF32-off
  (0.9608741) results. Conclusion: the forward pass is not bit-reproducible across
  cuDNN/TF32 configurations. **No PCK@20 improvement smaller than ~1.4e-5 is real.**
  Tolerances throughout are set to this measured floor, not to a guessed epsilon.

  **M1 — `harness/` (Rust, 4 modules, no Burn yet — guards should be fast to run).**
  `metrics.rs` implements upstream's PCK exactly (torso = ‖t[2]−t[12]‖, clamp 0.01,
  single flat mean over frames×keypoints) *and* `absolute_pck`, the broken protocol,
  deliberately retained as a foil. `guards.rs` has the constant-pose detector
  (measurement-(b)'s pred-std definition, threshold 1e-4), the mean-pose bar, the
  corruption masks, `assert_finite`, and a Rust reimplementation of upstream's
  zero-cleaning. `npy.rs` is a mmap .npy reader (the CSI array is 15.5 GB).
  `cargo test --release`: **5/5 pass**, including
  `broken_absolute_protocol_passes_where_torso_normalized_rejects` — the charter's
  "prove it can't": on a synthetic near-static scene with a constant predictor, the
  retracted absolute-0.2 protocol scores >0.99 while torso-normalized scores <0.60
  and the degeneracy guard trips.

  **M2 — `validate` exits 0; all 6 checks pass** (`results/m2_validation.json`):
  | Check | Got | Recorded | Delta | Tol |
  |---|---|---|---|---|
  | full.pck@20 | 0.9608876543 | 0.9608815325 | 6.1e-6 | 2e-5 |
  | full.mpjpe | 0.0098355175 | 0.0098340608 | 1.5e-6 | 1e-5 |
  | clean.pck@20 | 0.9661491629 | 0.9661454100 | 3.8e-6 | 2e-5 |
  | clean.mpjpe | 0.0094340722 | 0.0094327550 | 1.3e-6 | 1e-5 |

  The Rust f64 result is **bit-identical** to the independent Python f64 computation
  (0.9608876543209877 both), and both sit inside the measured floor of the recorded
  value. Independent GT reconstruction: Rust's own .npy reader + its own zero-cleaning
  rebuilt the Python-dumped GT with **0 mismatches**, after differing in **8,098**
  values pre-clean — so the cleaning path is genuinely exercised, not a no-op.
  Masks re-derived independently: 9,070 NaN / 9,072 big / 9,072 union / 52,560 clean.
  Degeneracy guard on the real checkpoint: pred_std **0.0259** (retracted model 0.0000;
  ESP32 honest model 0.0113) — healthy, not flagged.

  **Next: M3.** Design constraint carried forward from the M3/M4 dependency: computing
  expected information gain for a candidate probe *without peeking* requires
  p(unobserved CSI | observed, mask), so the model needs an **observation head**
  alongside the pose head, and the loader must serve full windows plus masks.

- **2026-09-08 ~23:15 — M3 + M4 COMPLETE (rung 2, joint Gaussian).**

  ### DEVIATION FROM PRE-REGISTRATION (recorded, with the measurement that forced it)

  The pre-registered representation — *"27 bands of 20 channels, keeping all 20
  frames … time is deliberately NOT collapsed: averaging the 20 frames would destroy
  the motion signal the pose head needs"* — was **measured to be wrong, and was the
  worst of seven variants tested.** `scripts`-free diagnostic `harness/src/bin/diag_repr.rs`
  asked the easiest possible version of the question (same-timestep pose, no
  forecasting gap) at fixed feature dimension 540:

  | representation | same-timestep PCK@20 | vs bar (0.7523) |
  |---|---|---|
  | **27 bands x 20 frames (pre-registered)** | **0.7495** | **−0.0029** |
  | 54 x 10 | 0.7668 | +0.0145 |
  | 108 x 5 | 0.7782 | +0.0259 |
  | 135 x 4 | 0.7942 | +0.0419 |
  | 270 x 2 | 0.8127 | +0.0603 |
  | 540 x 1 | 0.8150 | +0.0627 |
  | **270 bands, mean over time** | **0.8132 at HALF the dim** | +0.0609 |

  Pose information lives in **fine channel structure**; averaging 20 adjacent channels
  destroys it, while averaging over the 20 frames costs almost nothing. The
  pre-registered choice was the only variant that failed to beat the bar. This is
  exactly why the representation was pre-registered *and then measured* rather than
  assumed.

  **Revised representation:** `z` = **270 bands of 2 channels, averaged over all 20
  frames** (best NLL of any variant, half the dimension). Separately, the error of
  conflating *feature* granularity with *action* granularity was corrected: features
  stay fine (270 bands) while an action reveals a contiguous **group of 10 bands =
  20 channels** — a physically meaningful subcarrier group, 27 probe groups total.

  ### M3 results (MEASURED, `results/m3_m4.json`)

  - Persistence rung: band R² **0.871** — one-step dynamics is *not* trivial, so the
    dynamics term is doing real work.
  - Shrinkage λ = 1e-5, selected on **val**, never test.
  - Conditioning on `z_t` only: **PCK@20 0.8195 vs bar 0.7561 → +6.33 pts, BEATS the
    bar.** MPJPE 0.02073, pose NLL −113.29.
  - Degeneracy guard: passes (predicted means vary across t).
  - **Calibration FAILS the pre-registered criterion**, marginally and at one level.
    Coverage error vs nominal: **50% → +3.48 pp (FAIL, tolerance ±3)**, 80% → +0.35,
    90% → −1.63, 95% → −2.53 (all PASS). Reported as a fail; the tolerance is not
    being widened after the fact. The predictive is slightly over-dispersed at the
    centre — the residuals are more peaked than Gaussian.

  ### M4 results — the honest negative

  | K | EIG greedy | EIG fixed | EIG random | PCK greedy | PCK fixed | PCK random | **PCK oracle** |
  |---|---|---|---|---|---|---|---|
  | 1 | 0.017 | 0.014 | 0.013 | 0.8217 | 0.8214 | 0.8213 ± 0.0005 | **0.8323** |
  | 3 | 0.045 | 0.036 | 0.038 | 0.8214 | 0.8224 | 0.8217 ± 0.0008 | **0.8454** |
  | 6 | 0.082 | 0.070 | 0.072 | 0.8211 | 0.8210 | 0.8219 ± 0.0005 | **0.8547** |
  | 9 | 0.115 | 0.108 | 0.104 | 0.8230 | 0.8219 | 0.8221 ± 0.0007 | **0.8596** |

  - **EIG-greedy does maximise what it optimises:** it beats fixed and random on EIG
    nats at every budget, monotonically in K, and the bracketing check holds (oracle
    strictly above every policy at every K). The machinery is unit-tested
    independently (7/7 in `tests/eig.rs`, including an analytic-vs-Monte-Carlo check).
  - **It does NOT convert into pose accuracy.** Greedy/fixed/random differ by ~0.001
    PCK@20 — inside the random policy's own seed spread (±0.0005–0.0008). At this rung,
    **EIG-driven probing is not measurably better than probing at random.**
  - **The oracle bracket says why, and how much is on the table.** A policy that peeks
    at the true `z_{t+1}` reaches 0.8596 at K=9 vs 0.8230 for EIG-greedy — **+3.7 pts
    of headroom** that a *per-sample adaptive* policy could in principle capture.
  - Root cause is the pre-registered caveat, now confirmed empirically: Σ is
    homoscedastic, so the EIG-optimal probe set is **identical at every timestep**.
    It cannot exploit that different bands are informative at different moments —
    which is precisely what the oracle does. This is information-theoretic set
    selection, **not adaptive sensing**.

  ### Power check + mechanism isolation (added after the first M4 run)

  The first M4 ran on 2,000 pairs, where SE ≈ 0.0016 and the policy gaps were ~0.001 —
  so "EIG ≈ random" could have been a **power** problem rather than a real null.
  Rerun at **20,000 pairs** (SE ≈ 0.0002), with paired per-sample tests:

  | K | greedy | fixed | random | random-per-sample | target-peeking bound | greedy−random t |
  |---|---|---|---|---|---|---|
  | 1 | 0.8208 | 0.8207 | 0.8205 | 0.8206 | **0.8311** | +1.55 |
  | 3 | 0.8204 | 0.8208 | 0.8206 | 0.8205 | **0.8427** | −0.65 |
  | 6 | 0.8199 | 0.8200 | 0.8208 | 0.8207 | **0.8523** | **−3.15** |
  | 9 | 0.8203 | 0.8208 | 0.8208 | 0.8208 | **0.8568** | **−2.09** |

  - The null is now **underpowered-proof**. With 10× the samples, EIG-greedy is still
    not better than random — and at K=6 and K=9 it is *slightly and significantly
    **worse*** (t = −3.15, −2.09), though the magnitude is tiny (≤0.0009 PCK@20).
  - **Mechanism, stated plainly:** EIG maximises the log-determinant of the 30-dim
    pose posterior — total joint entropy reduction. PCK@20 is a *thresholded,
    per-keypoint* criterion. These are different utilities, so maximising the first
    can very slightly hurt the second. EIG is not the wrong *machinery*; it is the
    wrong *objective for this metric*.
  - **Adaptivity-with-no-information is worth nothing**, as the homoscedastic account
    predicts: drawing a *fresh random probe set per sample* matches a fixed set
    (t = −0.91, −1.49, +2.44, +0.18 — no consistent sign). So the whole of the
    upper-bound gap is "knowing which bands matter *at this timestep*".

  ### Correction: the oracle is a TARGET-PEEKING bound, not an achievable policy

  It selects bands by residual against the **true `y_{t+1}`** — i.e. using the target
  itself, not merely the future observation. So it upper-bounds *every* policy,
  including a perfect observation-adaptive one; it is **not** a reachable target.
  Renamed `target_peeking_upper_bound` in the JSON. The +3.6 pt gap at K=9 is
  therefore a **ceiling on the value of adaptivity**, not a promise.

  ### Non-comparability note (do not turn this into a claim)

  M3 forecasts `y_{t+1}` from `z_t` at 0.8195, while `diag_repr` *senses* `y_t` from
  `z_t` at 0.8132. It is tempting to say "forecasting beats sensing" — **do not**.
  The two runs use different evaluation subsets (pairs vs windows), different train
  sizes (60k vs 40k), and different λ (selected vs fixed 1e-5); their mean-pose bars
  differ too (0.7561 vs 0.7523). They are not like-for-like and no comparison between
  them is licensed.

  ### What this licenses, and what it does not

  - Licensed: "a closed-form action-conditioned dynamics model over WiFi-CSI band
    features predicts next-window pose 6.3 points above the constant-pose bar, with
    approximately-calibrated uncertainty (3 of 4 coverage levels within ±3 pp)."
  - **NOT licensed:** any claim that EIG-driven probing improves pose estimation. It
    did not, and that is the measured result.
  - **NOT comparable** to the 96.09% full-CSI number — different input abstraction
    (540 raw channels x 20 frames → 270 band means). The only valid comparison is to
    the 75.61% mean-pose bar on the same pairs.
  - The +3.7 pt oracle gap is the well-posed target for rung 3 (neural heteroscedastic
    Σ). Rung 3 is **not yet attempted**; nothing is claimed for it.
