# Action-Conditioned Dynamics Model — Objective State

**Purpose of this file:** a later session resumes mid-flight from here. Read it first.

## TL;DR for a resuming session (all five milestones complete)

All numbers below are **MEASURED by this work**; the 0.9609 baseline is **CITED**.

1. **Guards land first and hold.** 12/12 Rust tests, including a reproduction of this
   repo's retracted 92.9% claim: on a constant predictor the old absolute-0.2 protocol
   scores >0.99 where the torso-normalized one scores <0.60 and the degeneracy guard
   trips. `validate` reproduces the honest baseline (0.9608877 vs 0.9608815 recorded).
2. **A reproducibility floor was measured, not assumed:** TF32 on/off alone moves
   PCK@20 by 1.36e-5. **No improvement below ~1.4e-5 is real.**
3. **The honesty bar on WiFlow-STD, previously unknown, is 75.4% torso-PCK@20.**
4. **Measurement refuted my own pre-registered representation.** "27 bands x 20 frames,
   never collapse time" was the *worst* of seven variants and the only one failing the
   bar. Pose signal lives in fine channel structure; time resolution barely matters.
5. **Point accuracy: rung 3 reaches PCK@20 ≈0.942 (bar 0.7557, +18.7 pts)**, stable
   across three seeds — but this comes **entirely from the nonlinear mean head on
   `z_t`**, not from the probing.
6. **PROBING DOES NOT HELP — and under CLUSTER-ROBUST inference the honest statement
   is narrower than I first wrote.** Adjacent windows within a recording file are
   correlated at 0.9970, so per-sample i.i.d. SEs badly understate uncertainty. A
   file-level block bootstrap (2,000 reps, whole recordings resampled) gives:

   | K | greedy − no-probe | CI95 (file-block) | sig? | (i.i.d. t was) |
   |---|---|---|---|---|
   | 1 | −0.00029 | [−0.00068, +0.00010] | no | −1.42 |
   | 3 | −0.00152 | [−0.00305, +0.00018] | **no** | −5.04 |
   | 6 | −0.00339 | [−0.00602, −0.00076] | **yes** | −8.54 |
   | 9 | −0.00447 | [−0.00690, −0.00184] | **yes** | −10.16 |

   | K | greedy − random | CI95 (file-block) | sig? | (i.i.d. t was) |
   |---|---|---|---|---|
   | 1 | +0.00026 | [−0.00028, +0.00088] | no | +0.94 |
   | 3 | −0.00021 | [−0.00123, +0.00090] | no | −0.56 |
   | 6 | −0.00168 | [−0.00398, +0.00018] | **no** | −3.70 |
   | 9 | −0.00083 | [−0.00203, +0.00028] | no | −1.67 |

   **Corrected claims:** (a) EIG-greedy is **indistinguishable from random at every
   budget** — the "significantly worse" reading came from uncorrected i.i.d. SEs and
   does not survive; (b) probing is significantly worse than **not probing** only at
   **K=6 and K=9**, not at low budgets. An over-claimed negative is the same error as
   an over-claimed positive with the sign flipped, and this correction was applied to
   my own negative result at the same standard I applied to the positive one.
   **Cluster-robust and seed-robust survivor — the claim to lead with:** greedy −
   no-probe is negative in **all 16 seed × budget cells** (4 seeds × K=1,3,6,9) and
   grows monotonically more negative with budget. A sign test over 16 cells is
   p ≈ 1.5e-5 **with no within-file independence assumption whatsoever**, so it is
   immune to the clustering that weakened the t-statistics above.
   The +18.7 pts is the nonlinear mean head; the action-conditioning contributes nothing.
7. **Rung 3 FAILS ALL THREE of its pre-registered criteria.** An apparent
   "EIG beats random by 3–5 SE" result was produced by a single **unseeded**
   initialisation and **did not replicate** across three seeds — retracted within this
   session. **No calibrated-uncertainty claim may be made from rung 3.** Measured, not
   speculated: diagonalising rung 2's covariance gives NLL −81.92 vs rung 3's ≈−98, so
   heteroscedasticity does pay on marginals; rung 2 won on joint NLL via its full
   30×30 covariance. Next: low-rank + diagonal head, heavier-tailed predictive.
8. **The information exists but is FIVE TIMES SMALLER than I first claimed.** I ran a
   target-peeking bound at rung 3 and reported "+2.9 pts — the information is real".
   **That was ~80% selection artifact.** The oracle takes an argmax over C candidates
   scored against the *same* target used to choose them; the expected maximum of C noisy
   per-sample scores exceeds the true best whether or not any information is exploited.
   The control — **best-of-C RANDOM sets, also chosen by peeking, C matched to the
   oracle's greedy candidate count** — separates them (n=5,000 pairs, file-level block
   bootstrap):

   | K | C | oracle | best-of-C random+peek | no-probe | oracle−bestOfC (CI95) |
   |---|---|---|---|---|---|
   | 3 | 78 | 0.9620 | **0.9585** | 0.9416 | **+0.0034** [+0.0024, +0.0046] |
   | 9 | 207 | 0.9705 | **0.9649** | 0.9416 | **+0.0057** [+0.0047, +0.0068] |

   Best-of-C-random alone buys **+0.0170 (K=3)** and **+0.0233 (K=9)** over no-probe
   with **zero information exploited** — that is pure selection-on-target inflation, and
   it is **80–83% of the naive oracle gain**.

   **Corrected conclusion:** the information is **real and significant** (the margin
   survives a file-level block bootstrap comfortably) but it is worth **≈ +0.57 points
   at K=9, not +2.9**. The honest ceiling for *any* probe policy is oracle-minus-
   best-of-C ≈ +0.6 pts. EIG-greedy loses ~0.4 pts, so the gap to the honest ceiling is
   **≈ 1.0 point, not 3.3**. Persistence R² 0.871 pointed the right way after all — the
   marginal information is present but small.

   **Standing rule this establishes:** any peeking oracle in this artifact must be
   reported as *oracle minus best-of-C-random at matched search size*, never as *oracle
   minus no-probe*. The latter is an upper bound contaminated by its own search.

9. **Why EIG cannot reach even that ~0.6 pt ceiling — a design choice I made.** My
   variance head is deliberately blind to revealed *values* (only `z_t` and the mask),
   which is what kept EIG closed-form and peek-free. But this bound shows the value of
   a probe here is **value-dependent**, not merely mask-dependent: the oracle exploits
   *which values actually came back*, and a mask-only variance head structurally cannot
   represent that. **The fix is the observation head the original design constraint
   called for and I traded away for tractability:** sample candidate `z_{t+1}[S]` from
   `p(z_{t+1}[S] | z_t, revealed)` and average the resulting pose entropies — proper
   Monte-Carlo EIG. That is the single highest-value next experiment.

10. **RUNG 4 (value-aware variance + observation head + Monte-Carlo EIG): DID NOT
    CLOSE THE GAP — it went backwards.** MC-EIG is *worse than random*
    (t = −4.13 at K=9) and worse than no-probe (t = −9.07), closing **−40%** of the
    oracle gap. **Corrected (see item 8):** the bound's naive "+2.9 pts" is ~80%
    selection-on-target inflation; the information that actually exists is **≈ +0.57 pts**
    (oracle minus best-of-C-random at matched search size). So rung 4 failed to reach a
    ceiling much lower than I first reported — which makes its failure less dramatic and
    the whole probing question smaller-stakes than stated at the time.

11. **CORRECTED mechanism (my first reading of my own diagnostic was backwards).**
    At K=9, MC-EIG picks sets with lower predicted variance (2.91e-4 vs random's
    3.45e-4) but higher actual squared error (2.78e-4 vs 2.52e-4). Error/variance
    ratio — **1.0 if calibrated** — is **0.96 for EIG, 0.73 for random**.
    I first read this as "EIG seeks overconfidence". **That is wrong.** 0.96 is the
    *well-calibrated* end; 0.73 means the variance head assigns **falsely high**
    variance to the sets random happens to pick — sets that are actually accurate.
    So EIG is not hunting unearned confidence; it is **avoiding sets the head is
    over-cautious about, and those are precisely the accurate ones**. The failure is
    variance-head miscalibration in the *under*-confident direction corrupting EIG's
    ranking.
    **The unifying claim survives the correction** — miscalibration in *either*
    direction breaks the ranking EIG depends on, so **calibration is a precondition
    for information-gain probing, not a nice-to-have**, and every rung here failed its
    calibration criterion. But "overconfidence-seeking" is retracted.
    **Caveat on strength of evidence:** this two-number comparison comes from absolute
    predicted variance on the *selected* sets, whereas EIG selects on *entropy
    reduction from a baseline*. It is **suggestive, not a direct test**. The direct
    test is item 12.
    Actionable next step is therefore *not* a better EIG estimator but a **calibrated**
    predictive — low-rank + diagonal covariance, or heavier tails.
    **NOT post-hoc temperature scaling: item 13 proves that is a null by construction**
    (EIG's ranking is exactly invariant to it).

12. **THE DIRECT MEASUREMENT, which supersedes both readings in item 11.** Across all
    27 single-probe candidates, per sample, correlating EIG's **predicted** entropy
    reduction against the **realised** squared-error reduction:
    **mean per-sample Pearson r = +0.039**; fraction of samples with r > 0 = **0.543**
    (chance = 0.5).
    **EIG's predicted gain is essentially uncorrelated with realised benefit on this
    problem.** That, and not any story about over- or under-confidence, is why EIG
    performs like random selection — and why its one systematic tendency (preferring
    low predicted variance) is, if anything, mildly harmful.
    - **MEASURED:** the criterion is uninformative here (r ≈ 0.04).
    - **INFERRED, not measured:** that variance-head miscalibration is *why* it is
      uninformative. That remains the leading explanation — every rung failed its
      calibration criterion — but the causal link is an inference. The clean test is
      the pre-registered next experiment: recalibrate the variance head, then re-run EIG
      and re-measure this same correlation. **Note the recalibration must be
      mask-dependent — item 13 proves a temperature or any per-dimension rescaling
      leaves EIG's ranking bit-identical.** If r rises, the link is established; if not,
      the criterion is
      inadequate for a different reason.

13. **The obvious next fix is a NULL BY CONSTRUCTION — proved, not guessed.**
    I was about to recommend "temperature-scale the variance head until coverage
    passes, then re-run EIG". **That experiment cannot produce any change.**
    `EIG(S) = 0.5 Σ_i [log σ²_i(M) − log σ²_i(M ∪ S)]` — any recalibration whose
    offset depends only on the dimension index (a global temperature is the special
    case) adds the same constant to both terms and cancels exactly. Verified
    computationally in `tests/eig.rs::eig_is_invariant_to_dimension_wise_recalibration`
    (8/8 pass), including that the arg-max — the actual probe selection — is identical.
    **Consequence, and it redirects the whole line of work:** EIG only ever sees
    *differences between masks*, so the variance head's failure here is in its
    **mask-dependence**, not its overall scale. Marginal-calibration fixes
    (temperature scaling, per-dimension recalibration, isotonic on the marginals)
    are all provably useless for this. The fix has to make `σ²(z_t, mask)` correctly
    *responsive to which bands are revealed* — a modelling problem, not a
    post-processing one. Candidates: train the variance head with an explicit
    mask-contrastive objective, or drop the diagonal assumption so the covariance can
    express which revealed dims actually constrain which pose dims.
    (This supersedes the "post-hoc temperature scaling" next step in item 11–12.)

14. **Robustness caveat:** rung 4 is **single-seed (2026)**. Its headline verdict is
    unlikely to flip (t = −9.07 against no-probe), but the diagnostic ratios in item 11
    and the correlation in item 12 have not been replicated across seeds. Rung 3
    taught exactly this lesson. Treat items 11–12 as one-seed measurements.

**Nothing here is SOTA-beating.** But the incomparability claim I made earlier was
wrong in both clauses, and is CORRECTED here:

- **"A different task (forecasting)" — FALSE, retracted.** The model receives CSI at
  `t` **and at `t+1`** and predicts pose at `t+1`. `CSI(t+1)` is **contemporaneous with
  the target**. So this is concurrent CSI→pose sensing with one extra window of
  temporal context — **the same task** the cited baseline solves, with strictly *more*
  input. The `t+1` label is cosmetic when the corresponding observation is in the input
  set. To earn the "forecasting" label the inputs would have to be restricted to
  `CSI(≤t)`; they are not.
- **"A different input abstraction" — true but far weaker than stated.** It was fair
  when rung 2 sat 14 points back. Rung 3 reaches **0.9425 on the same 270-band
  features**, leaving only **~1.8 points** to the cited 0.9609. So the band reduction
  costs ~1.8 points, not 14.

**The applicable comparison is therefore the cited 0.9609 baseline, and against it this
work is ~1.8 points short — not 6.33 points ahead.** The honest ladder:

| | PCK@20 | note |
|---|---|---|
| mean pose (split-fitted bar) | 0.7561 | weakest rung |
| per-file mean pose | 0.7380 | leakage probe (lead, measured) |
| rung 2 (joint Gaussian) | 0.8208 | |
| **rung 3 (neural heteroscedastic)** | **0.9425** | same features as rung 2 |
| **cited WiFlow CSI→pose** | **0.9609** | **same task, LESS input — the applicable bar** |
| persistence (consumes GT pose) | 0.9970 | oracle, unreachable from CSI |

**Persistence is NOT a fair bar** — it consumes ground-truth `pose(t)`, which this model
never receives at any timestep. Different information sets. Its narrow lesson is only:
never build a model whose job is `pose(t) → pose(t+1)`; that mapping is solved by copying.

**The 14-point deficit decomposes ~87% model capacity / ~13% features** — rung 3 closed
12.2 of it with features held fixed. **Caveat on that attribution:** rungs 2 and 3 differ
in *two* ways at once (closed-form-linear vs neural, AND homoscedastic vs
heteroscedastic), so "capacity" is not cleanly isolated. See the homoscedastic control
spec below.

**Bar instability (important for anyone quoting these numbers).** The mean-pose bar is
**split-specific**: across sampled files it ranges 0.007–0.806 with IQR ≈ **0.056**, and
0.7561 sits near its p75. That spread is three orders of magnitude larger than the
1.4e-5 reproducibility floor and larger than several effects measured here. All
comparisons in this work use `honesty_bar_on_same_pairs` — the bar recomputed on the
model's own split — which is what makes them valid. **No bar figure from this work may
be compared against a bar computed on a different split.**

**Evaluation-set sizes, stated explicitly** (previously ambiguous and it cost an
outside reviewer real time): the test split holds **52,487** valid pairs;
`n = 20,000` is a **seeded subsample** of it, and the M4/oracle sets are further
subsamples. Corrupt files 487–499 are excluded via the committed masks, and PCK
aggregation is a **single flat mean over frames × keypoints** (not per-frame-then-mean).

## HOW TO RESUME (added after a resume trial FAILED — see "Resume trial" below)

Everything below is regenerable; nothing here needs a rerun to *read* the results.

**Fixtures are gitignored and a fresh checkout has none.** Regenerate in this order —
step 1 is Python (the one justified deviation, see below), steps 2+ are Rust:

```bash
cd benchmarks/wiflow-dynamics
~/wiflow-std-bench/venv/bin/python scripts/make_fixtures.py      # split idx, preds, GT, sha256 manifest
~/wiflow-std-bench/venv/bin/python scripts/dump_m3_fixtures.py   # train/val idx, all_gt_clean, pair list
cd harness
cargo build --release
./target/release/prepare        # 15.5 GB CSI -> fixtures/band_features.npy (270 bands), ~1 s cached
```

**Then, to verify the artifact is intact:**

```bash
cd benchmarks/wiflow-dynamics/harness
cargo test --release                      # expect 13 passed (8 eig + 5 retraction)
./target/release/validate                 # M2; must print verdict PASS, exit 0
```

**To re-run an experiment** (all write to `../results/`, all seeded):

```bash
./target/release/m3                       # rungs 1-2: joint Gaussian, M3 + M4      -> m3_m4.json
./target/release/diag_repr                # representation sweep                    -> repr_diagnostic.txt
./target/release/rung3                    # rung 3 + oracle + controls (RUNG3_SEED) -> rung3.json
RUNG3_SEED=7 ./target/release/rung3       # a replication seed                      -> redirect yourself
./target/release/rung4                    # rung 4: MC-EIG (RUNG4_SEED)             -> rung4.json
```

### File map — which result belongs to which milestone

| file | milestone | what it holds |
|---|---|---|
| `fixtures/manifest.json` | M1/M2 | sha256s + the metrics the fixtures were validated against |
| `fixtures/reproducibility_floor.json` | M1 | the TF32 measurement establishing the 1.4e-5 floor |
| `results/m2_validation.json` | **M2** | baseline reproduction, 6 checks |
| `results/repr_diagnostic.txt` | M3 (pre-work) | the 7-variant representation sweep that refuted my pre-registration |
| `results/m3_m4.json` | **M3 + M4**, rung 2 | joint Gaussian, EIG policies, diagonalised-NLL annotation |
| `results/rung3.json` | rung 3 (seed 2026) | heteroscedastic head, oracle, selection control, block bootstrap |
| `results/rung3_seed7.json` | rung 3 | replication seed 7 — part of the set that retracted the EIG win |
| `results/rung3_seed101.json` | rung 3 | replication seed 101 |
| `results/rung3_seed555.json` | rung 3 | replication seed 555 |
| `results/rung3_undertrained_12ep.json` | rung 3 | the deliberately-kept 12-epoch undertrained run |
| `results/rung4.json` | rung 4 | value-aware Σ, Monte-Carlo EIG, overconfidence + r=0.039 diagnostics |

### The next action, unambiguously

**All five chartered milestones are complete; rungs 3-4 are beyond charter and both
failed their pre-registered criteria.** The single highest-value next experiment is the
**homoscedastic same-backbone control** specified under "Untested controls" below —
it is the one test that isolates whether per-sample σ carries information at all, and
it needs no new modelling, only a frozen mean head and a global Σ̂.
Do **not** start another rung before it.

### Resume trial (RUN, and it FAILED — this section is the fix)

The claim "a later session resumes mid-flight from here" was untested, so I tested it
mechanically against the committed artifacts. **Three real gaps, all now fixed above:**
1. **No runnable command existed anywhere in STATE.md** — a reader could not reproduce
   M2 at all. Fixed by the command blocks above.
2. **`fixtures/*.npy` are gitignored**, so a fresh checkout has none and `validate`
   would fail immediately; the regeneration order was undocumented. Fixed.
3. **Four result files were unmapped** to any milestone (`repr_diagnostic.txt`,
   `rung3_seed{7,101,555}.json`, `rung4.json`). Fixed by the file map.

Last updated: 2026-09-09 (resume trial + selection/clustering controls)
Originally started: 2026-09-08
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
- [x] **M4 — Information-gain evaluation.** DONE 2026-09-08, across two model classes.
      **Rung 2 (homoscedastic): honest negative** — EIG wins on information at every
      budget but does not convert into pose accuracy (underpowered-proof at n=20,000).
      **Rung 3 (heteroscedastic): EIG-greedy beats random by 3–5 SE at every budget.**
      The rung-2 negative was a property of the model class, not of EIG probing.
- [x] **M5 — Lineage/evidence recorded for every claim.** STATE.md + `results/*.json` +
      `fixtures/manifest.json` (sha256s). Every number states MEASURED or CITED.

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

### RUNG 4 PRE-REGISTRATION (written 2026-09-09 ~00:00, BEFORE the model was written)

> **SUPERSEDING NOTE (added later, pre-registration text left intact below).** The
> "+2.9 pts" motivating figure was subsequently shown to be ~80% selection-on-target
> inflation; the real headroom is ≈ +0.57 pts (item 8). The pre-registration is not
> rewritten — it recorded what was believed when rung 4 was designed — but a reader
> should carry the corrected number forward.

**Why this exists:** the rung-3 target-peeking bound proved the information is present
(+2.9 pts at K=9, t=+22.3) and that closed-form EIG cannot reach it (−0.4 pts). The
diagnosis is specific: my rung-3 variance head is **blind to revealed values** by
design, so it can only express *"how much would set S teach me on average"*, never
*"how much did these particular values teach me"*. The bound shows probe value is
**value-dependent**. Rung 4 removes exactly that limitation.

**Architecture change (the whole point):**
- **Variance head now SEES revealed values** — `(z_t, mask ⊙ z_{t+1}, mask) → log σ²_y`.
- **New observation head** — `(z_t, mask ⊙ z_{t+1}, mask) → (μ_z, log σ²_z)` over the
  270 next-window band dims. This is the piece the original M3/M4 design constraint
  called for and that rung 3 traded away for closed-form tractability.
- Trained jointly: pose NLL + observation NLL, random masks per sample.

**EIG is now Monte-Carlo, and must be** (a closed form no longer exists, which is the
price of value-dependence):
```
H_before = 0.5 * sum_i log sigma^2_y(z_t, M, revealed)
for m in 1..M:  sample z_S^(m) ~ p(z_{t+1}[S] | z_t, M, revealed)   # observation head
                H_m = 0.5 * sum_i log sigma^2_y(z_t, M u S, revealed u z_S^(m))
EIG(S) = H_before - mean_m(H_m)
```
Still **peek-free**: candidate values are *sampled from the model*, never read from the
future. M = 4 samples, K in {3, 9}, 5,000 eval pairs (cost-bounded; same pairs for
every policy and for the bound).

**Pass condition, fixed now — rung 4 helps only if BOTH:**
1. MC-EIG-greedy **beats no-probe** by >2 SE at some budget (rung 3 failed this: it
   *hurt* by 5–15 SE); **and**
2. MC-EIG-greedy beats **random** at the same budget by >2 SE, replicated across
   **≥3 seeds with a consistent sign** (the bar rung 3's retracted claim failed).

Anything less is reported as **"rung 4 did not close the gap"**, with the fraction of
the oracle gap it did close stated as a measurement. The target-peeking bound is
recomputed on the same pairs as the reference ceiling.

**Not a criterion change:** rungs 1–3 and their verdicts are untouched. Rung 4 is a new
experiment with its own pre-registration, not a re-scoring of an earlier one.

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

## Rung 3 — run log and one methodology change (recorded as it happened)

**Run 1 (12 epochs, `results/rung3_undertrained_12ep.json`): self-reported
"RUNG 3 DID NOT HELP".** Criterion 1 failed (val NLL −81.24 vs rung-2 −113.29) and
criterion 2 failed (coverage); criterion 3 passed. **The run was undertrained, not
converged** — 12 epochs took 16 s and the loss was still descending steeply
(−68 → −81 over the final epochs). Two things in it are worth keeping regardless:
- PCK@20 **0.8357** vs bar 0.7557 (**+8.0 pts**) — already above rung 2's 0.8195,
  from an undertrained model.
- **71 distinct probe sets** across the test pairs, modal set only 42% —
  the probe set genuinely **varies per sample**. Rung 2 had exactly **1** distinct set
  by construction. The capability rung 3 was built to add demonstrably exists.

**Methodology change, stated plainly:** training length was raised from a fixed 12
epochs to early stopping on val NLL (patience 40, cap 400). This is a *convergence*
fix for an obviously undertrained model, not a change to any pre-registered pass
criterion — the three criteria in the rung-3 pre-registration are untouched, and the
evaluation set, bar, and protocol are identical. Both runs are kept; the 12-epoch
result is preserved rather than discarded.

### Run 2 (converged, early-stopped at epoch 163, best epoch 123) — `results/rung3.json`

**Verdict by the pre-registered conjunction: "RUNG 3 DID NOT HELP" — 2 of 3 criteria
missed.** That is the headline verdict and it stands. But the criteria conflated two
questions, and rung 3 answers them oppositely:

| Pre-registered criterion | Result |
|---|---|
| 1. beats rung-2 pose NLL | **FAIL** — −102.83 vs −113.29 |
| 2. coverage within ±3 pp at all four levels | **FAIL** — +7.46 pp @50%, +3.20 pp @80% (90/95 pass) |
| 3. EIG-greedy beats random by >2 SE | **PASS**, decisively — see below |

**What rung 3 did do (MEASURED, same 20,000 test pairs as rung 2):**

- **PCK@20 0.9425 vs bar 0.7557 → +18.7 pts.** Rung 2 on these same pairs was ~0.8208.
  A nonlinear mean head is worth **~+12 pts** over the linear-Gaussian one. MPJPE
  0.01073 vs rung 2's ~0.0207. Degeneracy guard passes (pred std 0.0208).
- **EIG-driven probing now works, at every budget:**

  | K | greedy | random | fixed | greedy−random | SE | t |
  |---|---|---|---|---|---|---|
  | 1 | 0.9425 | 0.9417 | 0.9419 | +0.00078 | 0.00025 | **+3.11** |
  | 3 | 0.9420 | 0.9404 | 0.9411 | +0.00166 | 0.00035 | **+4.80** |
  | 6 | 0.9412 | 0.9390 | 0.9380 | +0.00221 | 0.00044 | **+5.04** |
  | 9 | 0.9400 | 0.9383 | 0.9406 | +0.00176 | 0.00045 | **+3.91** |

- **Adaptivity is real and large:** at K=9, **19,653 distinct probe sets across 20,000
  samples** (modal set ≈0%). Rung 2 had exactly **1**, by construction.

**The central scientific finding of this work:** *the M4 negative at rung 2 was a
property of the model class, not of information-gain probing.* Once `Σ` depends on
`z_t`, so the probe set can vary per sample, EIG-driven selection beats random by
3–5 SE at every budget. Under a homoscedastic `Σ` it could not, because the
"EIG-optimal" set was frozen across all timesteps.

**Two caveats I am not allowed to hide behind, and one clarification:**

1. **The NLL comparison is not like-for-like, and that is my design's fault.** Rung 2
   carries a **full 30×30** pose covariance; my rung-3 head is **diagonal**. Joint
   multivariate NLL strongly rewards modelling cross-keypoint correlation, which a
   diagonal head structurally cannot. So −102.83 vs −113.29 does **not** show rung 3's
   marginals are worse. It is still a **FAIL** against the criterion as written — I am
   not rewriting the criterion — but the right next step is a low-rank + diagonal
   covariance head, not abandoning heteroscedasticity.
2. **Calibration got worse, not better** (+7.46 pp at the 50% level vs rung 2's
   +3.48 pp), in the same direction: over-covered at the centre, i.e. residuals more
   peaked than Gaussian. A Student-t or low-rank predictive is the obvious fix. As it
   stands, **rung 3's uncertainty is less trustworthy than rung 2's**, and no
   calibrated-uncertainty claim may be made from it.
3. **Clarification, to pre-empt a false contradiction:** rung 3's 0.9425 exceeds rung
   2's target-peeking upper bound of 0.8568. That is not inconsistent. That bound
   upper-bounded *probe-selection policies within the rung-2 model class*, holding the
   linear-Gaussian predictor fixed. A better predictor moves the whole curve; the bound
   was never a bound on all models.

### RETRACTION WITHIN THIS SESSION — the rung-3 EIG result did not replicate

The run above was **unseeded**: `LinearConfig::init` draws from Burn's global RNG,
which I had not seeded (my LCG covered masks and shuffles only). After seeding it and
rerunning across **three initialisations (2026, 7, 101)**:

| seed | no-probe PCK@20 | pose NLL | t(greedy−noprobe) @K=1,3,6,9 | t(greedy−random) @K=1,3,6,9 |
|---|---|---|---|---|
| *(all four seeds below; t are per-sample i.i.d., NOT cluster-corrected — see item 6)* | | | | |
|---|---|---|---|---|
| 2026 | 0.9424 | −97.17 | −1.42, −5.04, −8.54, −10.16 | +0.94, −0.56, −3.70, −1.67 |
| 7 | 0.9375 | −94.88 | −2.80, −5.02, −8.69, −9.78 | +0.03, −1.47, −2.56, −2.92 |
| 101 | 0.9452 | −102.02 | −3.24, −7.20, −12.73, −15.37 | +1.29, +1.53, −0.25, +0.39 |
| 555 | — | — | −4.10, −9.30, −13.48, −15.36 | — |

1. **The "EIG-greedy beats random by 3–5 SE" claim is RETRACTED.** It came from a
   single unseeded initialisation. Across three seeds the sign is not stable
   (K=3: −0.56 / −1.47 / **+1.53**; K=9: −1.67 / −2.92 / **+0.39**). **Criterion 3
   does not reproduce**, so rung 3 fails **all three** pre-registered criteria, not two.
   This is exactly the failure mode the charter warns about, caught by seeding and
   replication rather than by a single lucky run.
2. **"Probing hurts" is robust and is the real finding.**
   `any_budget_where_eig_greedy_beats_no_probe_by_2se` is **false on every seed**, and
   t(greedy−noprobe) is large and negative at every K ≥ 3 on every seed, growing
   monotonically worse with budget (to −15.4). Revealing more costs more.
3. **Point accuracy is robust:** no-probe PCK@20 = 0.9424 / 0.9375 / 0.9452
   (mean ≈0.942 vs bar 0.7557). The +18.7 pts is real and comes entirely from the
   nonlinear mean head on `z_t`.
4. **The NLL caveat is now MEASURED, not speculated.** Diagonalising rung 2's pose
   posterior (zeroing off-diagonals — the like-for-like comparison against rung 3's
   diagonal head) gives **−81.92**, versus rung 3's ≈−94.9 to −102.0. So
   heteroscedasticity *does* pay off on the marginals; rung 2's advantage on joint NLL
   came from its **full 30×30 covariance** capturing cross-keypoint correlation.
   Criterion 1 as written still **FAILS** — it is not being rewritten — but the fix is
   a low-rank + diagonal head, not abandoning heteroscedasticity.

**Net, corrected:** the machinery is built, unit-tested and validated, and the
nonlinear dynamics head is a large robust win on point accuracy. But **the charter's
central premise does not hold for this observation definition**: information-gain-driven
probing of time-averaged `z_{t+1}` bands does not beat not probing, in either model
class, on any seed. Rung 3 fails all three of its pre-registered criteria. The
"calibrated uncertainty" half of M3 is met only by rung 2 (3 of 4 coverage levels).

**Reproducibility note (M5):** rung-3 numbers are now seeded (`RUNG3_SEED`, default
2026) and reproducible. The earlier unseeded run is preserved as
`results/rung3_undertrained_12ep.json`; per-seed results are `results/rung3_seed*.json`.

## Superseded claims in the git history (read before trusting a commit message)

Commit messages are immutable, so falsified ones are superseded here rather than
rewritten. A reader running `git log` must not take these as the last word.

- **`5498dd7c` — "rung 3 heteroscedastic head — EIG probing works once Sigma depends on
  z_t". FALSIFIED by my own replication.** That commit's 3–5 SE win came from a single
  **unseeded** initialisation (Burn's `LinearConfig::init` draws from a global RNG my
  LCG never covered). Across seeds 2026 / 7 / 101 the sign is not stable, and against
  *no-probe* every policy loses. Superseding numbers are in items 6, 11–13 above.
- **"EIG is an overconfidence-seeking criterion"** (commit `8e3aab75` message body) —
  retracted in `2fbf21f4`; I had read the err/var ratio backwards. See item 11.

## Terminology: this policy is NOT adaptive

Called "adaptive sensing" in earlier notes; that is **wrong** and is corrected
everywhere. The variance head takes `(z_t, mask)` and deliberately **not** the revealed
values, so for a fixed `z_t` the entire greedy sequence is determined *before a single
probe is taken*. It is **open-loop per instance, instance-conditioned across instances**
— "per-sample probe-set selection".

- Krause, Singh & Guestrin, JMLR 9:235–284 (2008) §10: *"any closed-loop strategy which
  sequentially decides on the next location to measure, surprisingly, is equivalent to
  an open loop placement strategy."* Krause & Guestrin, ICML 2007 §4: *"any objective
  function depending only on the predictive variances cannot benefit from sequential
  strategies."* An equality, not a bound.
- It is also **not** a single static ranking: 19,653 distinct probe sets across 20,000
  samples (rung 2 had exactly 1). That measurement is what distinguishes the three
  cases — but note it *cannot* distinguish per-sample-fixed from closed-loop, since both
  produce ~20,000 distinct sets. **Not adaptive, not static, instance-conditioned.**

## No approximation guarantee is claimed, and none is available

Any implication that greedy EIG inherits a `(1−1/e)` bound would be **false**; none
appears in this artifact and none may be added.

- Krause et al. (2008) Remark 13: *"The information gain, IG(A) = I(A;U) is not
  submodular in A"*, with an explicit 3×3 Gaussian counterexample. Their `(1−1/e)` is
  for `I(A; V\A)` (field coverage), **not** `I(A; Y)` for a distinguished target — which
  is this case. Correlated CSI, where a band is useless alone but informative in
  combination, is exactly the non-submodular regime.
- Precision: **MI itself is submodular**; it is *monotonicity* that is approximate. The
  honest quantity is Das & Kempe's (ICML 2011) submodularity ratio γ, estimated
  empirically — not lower-bounded spectrally, which is near-vacuous for 540 correlated
  features.
- Chen, Hassani, Karbasi & Krause (COLT 2015): the MI criterion violates adaptive
  submodularity, so even a properly closed-loop version would not inherit the
  Golovin–Krause guarantee. Do not trade one unavailable guarantee for another.

**This selection study is empirical. It carries no bound.**

## Independent corroboration and prior art

- **Rung 2's null has direct published corroboration.** Khamaisi & Rodrigues,
  arXiv:2602.10823, *"Less is More: The Dilution Effect in Multi-Link Wireless Sensing"*
  — 9-node / 72-link mesh over 12 days: *"sophisticated link selection provided no
  significant advantage over random selection (p = 0.35)"*, the benefit coming from
  avoiding multi-link fusion rather than optimising which link to use. That is this
  result, independently, in a real WiFi-sensing deployment.
- **The efficiency result is corroboration, not novelty.** "Fewer subcarriers suffice
  for pose" is already published: Capozzi et al., IbPRIA 2025 (LNCS 15938),
  attention-rollout subcarrier importance with a random control. *Caveat: only the
  abstract was readable (SpringerLink 502'd), so its numbers are unread and no numeric
  comparison is made here.*
- **The gap statement**, with no mechanism attached in the literature: Meneghello et
  al., IEEE Comm. Mag. 2023 (arXiv:2212.13930) — *"the design of sensing applications
  should consider properly selecting the sub-channels that are the best for sensing
  purposes."*
- **The low-rank fix was reached twice independently** — from these results, and from
  the literature: Dorta et al., CVPR 2018, *"Structured Uncertainty Prediction
  Networks"*, `Σ = D + VVᵀ` at rank 4–8.
- **What is genuinely unclaimed:** budgeted, information-gain-driven, *sequential*,
  test-time CSI feature selection. Hard selection exists in EEG (Strypsteen & Bertrand
  2021, Gumbel-softmax channel selection) and has not been ported to CSI; the CSI pose
  subfield adds sensors rather than pruning them.
- **The strongest surviving claim is about evaluation practice.** Active-feature-
  acquisition work is evaluated almost exclusively by point accuracy — EDDI (ICML 2019),
  GSMRL (ICML 2021), Covert et al. (ICML 2023) and AFABench (2025) report accuracy/F1
  with **zero** calibration metrics; EDDI's regression curves use RMSE, which would show
  PCK improving here while hiding the NLL collapse entirely. *"AFA policies are
  evaluated exclusively by functionals of the mean; under a proper scoring rule the
  greedy policy is worse than random and the gap grows with budget"* is defensible on
  this evidence.

## NLL degradation decomposes into two named mechanisms (MEASURED)

From `results/rung3.json`, NLL from K=1 → K=9:

| policy | K=1 | K=9 | degradation |
|---|---|---|---|
| random | −102.503 | −100.882 | 1.621 |
| greedy | −101.823 | −95.634 | 6.189 |

**Both degrade, so both mechanisms are present and separable:** ~1.62 nats is the
baseline effect — **feedback covariate shift** over the mask distribution (Fannjiang,
Bates, Angelopoulos, Listgarten & Jordan, PNAS 119(43), 2022) — and the ~4.57-nat excess
is greedy-specific, the **optimizer's curse** (Smith & Winkler, *Management Science*
52(3):311–322, 2006). Roughly **26% baseline / 74% greedy-specific**.
Free follow-up: retrain the variance head on the policy's *own* induced mask
distribution rather than uniform random masks. That attacks the covariate-shift half at
zero modelling cost. It will **not** fix the open-loop property.

## Untested controls and named-but-unchecked degeneracies (for a later session)

- **PRIMARY untested control — homoscedastic same-backbone.** Hold *everything* fixed
  including the **trained mean head**, so μ and therefore PCK are identical by
  construction across arms; vary only the covariance, replacing per-sample
  `Σ(z_t, mask)` with a single global `Σ̂` from training residuals, in the **full 30×30
  form** (so heteroscedastic-vs-homoscedastic is not confounded with
  diagonal-vs-full). Adjudicate on **ENCE** — the only metric that directly tests
  whether per-sample σ is *informative* rather than merely calibrated; energy and
  variogram scores alongside, NLL secondary. **Decision rule:** if the heteroscedastic
  head does not beat that control on ENCE, per-sample variance carries no information.
- **Corruption avoidance (unchecked).** Files 487–499 are corrupt and file-clustered, so
  a selector could score by dodging corruption rather than by finding pose-informative
  bands. Check: mask corruption before selection and report both.
- **File fingerprinting (unchecked, likely weak).** A selector could pick features that
  identify the recording, letting the predictor memorise per-file pose statistics.
  Per-file mean pose is 0.7380 vs global 0.7264, so this channel is worth only ~+1.2
  points here. Direct test: a classifier on the selected subset predicting file ID.
- **Mask leakage — considered and DISCONFIRMED, no test needed.** The hypothesis was
  that the predictor reads pose from *which* features were selected (the greedy mask is
  a deterministic function of `z_t`, and `z_t` predicts pose), which would manufacture a
  spurious greedy-over-random win. That mechanism predicts a win; replication produced
  **no win**. The replication already did the work the proposed test would have done.
- **Anatomical plausibility (not run).** The dataset ships `SKELETON_CONNECTIONS`
  (14 bones); GT bone-length std across 345,373 complete windows is **0.00801**. Draw
  M=100 samples from the stored per-instance (μ, σ), compute the 14 bone lengths, take
  within-instance std per bone, average, divide by 0.00801. A ratio above ~1.3 means the
  diagonal head emits anatomically impossible poses — a failure marginal coverage
  structurally cannot see, and direct evidence for the low-rank fix.

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
