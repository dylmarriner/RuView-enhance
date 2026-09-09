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
- [ ] **M3 — Action-conditioned dynamics model.** Next-observation prediction
      conditioned on an explicit action (probe/subcarrier/antenna-group selection),
      emitting prediction + **calibrated** uncertainty. Report a proper scoring rule
      and a calibration metric, not just point error.
- [ ] **M4 — Information-gain evaluation.** EIG-driven probe selection vs (a) random
      and (b) fixed probing, on held-out data. Paired, seeded, same budget.
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
