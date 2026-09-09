"""Fixture generation for the Rust anti-Goodhart harness. NOT a model or trainer.

Why this file is Python in a Rust-only ecosystem (deliberate, bounded deviation —
see STATE.md "Justified deviation"): reproducing the honest baseline requires
consuming a PyTorch .pth checkpoint and upstream's Dataset/split code, including
`file_mappings.pkl` (a Python pickle). Porting the 2.23M-param net to Burn purely to
re-derive an already-measured number would burn the guards budget for no epistemic
gain, and that net is not needed for the dynamics model.

This script writes DATA, never code paths the Rust harness depends on for logic:
  - split index arrays produced by upstream's own split function
  - the reference predictions of the retrained checkpoint
  - the ground truth EXACTLY as `calculate_pck` consumes it

The Rust harness is then validated two independent ways against these fixtures:
  (a) score (pred, gt) and match eval_retrained.json
  (b) load GT from .npy itself and match the dumped GT

Self-checks below hard-fail if the environment has drifted from the recorded
ground truth, so a silently-wrong fixture cannot be produced.
"""
import hashlib
import json
import os
import random
import sys

import numpy as np
import torch

BENCH = os.path.expanduser("~/wiflow-std-bench")
RESULTS = "/home/ruvultra/projects/ruview-worktrees/ruforecast-rust/benchmarks/wiflow-std/results"
OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/fixtures"
CKPT = f"{BENCH}/upstream/test/best_pose_model.pth"

# Expected values from results/eval_retrained.json (CITED ground truth).
EXPECT = {
    "test_full": {"samples": 54000, "pck@20": 0.9608815324571398, "mpjpe": 0.009834060806367133},
    "test_clean": {"samples": 52560, "pck@20": 0.9661454100405608, "mpjpe": 0.009432755044379373},
}

sys.path.insert(0, f"{BENCH}/upstream")

# mmap csi_windows.npy (15.5 GB) instead of loading it: the upstream Dataset calls a
# bare np.load. Patch is read-only and shape/semantics-identical.
_real_load = np.load


def _mmap_load(path, *a, **kw):
    if isinstance(path, str) and path.endswith("csi_windows.npy"):
        kw.setdefault("mmap_mode", "r")
    return _real_load(path, *a, **kw)


np.load = _mmap_load

from dataset import PreprocessedCSIKeypointsDataset, create_preprocessed_train_val_test_loaders  # noqa: E402
from models.pose_model import WiFlowPoseModel  # noqa: E402
from utils.metrics import calculate_mpjpe, calculate_pck  # noqa: E402

np.load = _real_load

os.makedirs(OUT, exist_ok=True)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

ds = PreprocessedCSIKeypointsDataset(
    data_dir=f"{BENCH}/preprocessed_csi_data", keypoint_scale=1000.0, enable_temporal_clean=True
)
train_loader, val_loader, test_loader = create_preprocessed_train_val_test_loaders(
    dataset=ds, batch_size=256, num_workers=2, random_seed=42
)

train_idx = np.asarray(train_loader.dataset.indices, dtype=np.int32)
val_idx = np.asarray(val_loader.dataset.indices, dtype=np.int32)
test_idx = np.asarray(test_loader.dataset.indices, dtype=np.int32)

assert len(test_idx) == 54000, f"test split is {len(test_idx)}, expected 54000"
assert len(train_idx) + len(val_idx) + len(test_idx) == 360000

# Corruption masks are defined over the FULL 360k window index space.
nan_mask = np.load(f"{RESULTS}/nan_windows_mask.npy")
big_mask = np.load(f"{RESULTS}/big_windows_mask.npy")
corrupt = nan_mask | big_mask
clean_sel = ~corrupt[test_idx]
assert int(clean_sel.sum()) == 52560, f"clean test is {int(clean_sel.sum())}, expected 52560"

device = torch.device("cuda")
model = WiFlowPoseModel(dropout=0.5).to(device)
state = torch.load(CKPT, map_location=device, weights_only=True)
renames = {"att.": "attention.", "final_conv.": "decoder."}
state = {
    next((new + k[len(old):] for old, new in renames.items() if k.startswith(old)), k): v
    for k, v in state.items()
}
model.load_state_dict(state, strict=True)
model.eval()

preds, gts = [], []
with torch.no_grad():
    for bx, by in test_loader:
        out = model(bx.to(device))
        preds.append(out.reshape(out.shape[0], 15, 2).float().cpu().numpy())
        gts.append(by.reshape(by.shape[0], 15, 2).float().numpy())

pred = np.concatenate(preds).astype(np.float32)
gt = np.concatenate(gts).astype(np.float32)
assert pred.shape == gt.shape == (54000, 15, 2), (pred.shape, gt.shape)


THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]


def score_flat_f32(p, g):
    """Upstream's own metric functions applied once over the whole split (float32)."""
    pt, gtt = torch.from_numpy(p), torch.from_numpy(g)
    pck = calculate_pck(pt, gtt, thresholds=THRESHOLDS)
    return {
        "samples": int(p.shape[0]),
        "mpjpe": float(calculate_mpjpe(pt, gtt)),
        **{f"pck@{int(t * 100)}": float(v) for t, v in pck.items()},
    }


def score_batched_f32(p, g, bs=256):
    """Replicate eval_retrained.py exactly: per-batch float32 means, size-weighted.

    This is the accumulation order that produced results/eval_retrained.json.
    """
    tot = {t: 0.0 for t in THRESHOLDS}
    tot_mpjpe, n = 0.0, 0
    for i in range(0, len(p), bs):
        pb, gb = torch.from_numpy(p[i:i + bs]), torch.from_numpy(g[i:i + bs])
        b = pb.shape[0]
        tot_mpjpe += calculate_mpjpe(pb, gb) * b
        pck = calculate_pck(pb, gb, thresholds=THRESHOLDS)
        for t in THRESHOLDS:
            tot[t] += pck[t] * b
        n += b
    return {
        "samples": n,
        "mpjpe": tot_mpjpe / n,
        **{f"pck@{int(t * 100)}": tot[t] / n for t in THRESHOLDS},
    }


def score_f64(p, g):
    """Canonical reference: identical protocol, float64 throughout.

    Neither float32 variant above is 'the' answer — both carry accumulation error.
    This is the value the Rust harness is held to.
    """
    p, g = p.astype(np.float64), g.astype(np.float64)
    torso = np.sqrt(((g[:, 2] - g[:, 12]) ** 2).sum(axis=1)).clip(min=0.01)
    d = np.sqrt(((p - g) ** 2).sum(axis=2))
    nd = d / torso[:, None]
    return {
        "samples": int(p.shape[0]),
        "mpjpe": float(d.mean()),
        **{f"pck@{int(t * 100)}": float((nd <= t).mean()) for t in THRESHOLDS},
    }


metrics = {}
for name, sel in (("test_full", slice(None)), ("test_clean", clean_sel)):
    metrics[name] = {
        "f64_canonical": score_f64(pred[sel], gt[sel]),
        "f32_batched_256": score_batched_f32(pred[sel], gt[sel]),
        "f32_flat": score_flat_f32(pred[sel], gt[sel]),
    }

# Self-check 1: reproduce the recorded honest baseline to within the MEASURED
# reproducibility floor of the forward pass.
#
# The floor is NOT assumed -- it was measured by scripts/diag_reproducibility_floor.py
# (raw: fixtures/reproducibility_floor.json). Toggling TF32 alone, with everything
# else identical, moves PCK@20 by 1.36e-5 and MPJPE by 5.7e-6; the recorded 2026-06-10
# value lies *between* the TF32-on and TF32-off results. So the forward pass is not
# bit-reproducible across cuDNN/TF32 configurations, and a bit-exact assertion here
# would be asserting something false.
#
# Tolerances are the measured floor rounded up one significant figure.
FLOOR = {"pck@20": 2e-5, "mpjpe": 1e-5}

for split, exp in EXPECT.items():
    for k, want in exp.items():
        got = metrics[split]["f32_batched_256"][k]
        if isinstance(want, int):
            assert got == want, f"{split}.{k}: got {got}, expected {want}"
        else:
            assert abs(got - want) < FLOOR[k], (
                f"{split}.{k}: got {got!r}, expected {want!r}, "
                f"delta {abs(got - want):.3e} exceeds measured floor {FLOOR[k]:.0e} "
                "-- this is larger than GPU numeric noise, so suspect a real protocol drift"
            )

# Self-check 2: float32 accumulation order contributes ~1e-9, three orders of
# magnitude below the forward-pass floor. Recorded so the two noise sources are
# never confused again.
noise = {}
for split in metrics:
    vals = [metrics[split][v]["pck@20"] for v in ("f64_canonical", "f32_batched_256", "f32_flat")]
    noise[split] = {
        "pck@20_accumulation_spread": max(vals) - min(vals),
        "pck@20_vs_recorded": abs(metrics[split]["f32_batched_256"]["pck@20"] - EXPECT[split]["pck@20"]),
    }
    assert noise[split]["pck@20_accumulation_spread"] < 1e-4, "accumulation spread unexpectedly large"

np.save(f"{OUT}/test_pred.npy", pred)
np.save(f"{OUT}/test_gt.npy", gt)
np.savez(f"{OUT}/split_indices.npz", train=train_idx, val=val_idx, test=test_idx)
np.save(f"{OUT}/test_clean_sel.npy", clean_sel)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


manifest = {
    "generated_by": "benchmarks/wiflow-dynamics/scripts/make_fixtures.py",
    "checkpoint": CKPT,
    "checkpoint_sha256": sha(CKPT),
    "dataset_dir": f"{BENCH}/preprocessed_csi_data",
    "protocol": {
        "split": "upstream create_preprocessed_train_val_test_loaders, random_seed=42, file-level 70/15/15",
        "pck": "torso-normalized, NECK_IDX=2 PELVIS_IDX=12, clamp(min=0.01), flat mean over batch x keypoints",
        "gt": "all_keypoints.npy (already 15-kp and already scaled); enable_temporal_clean=True",
    },
    "metrics_measured": metrics,
    "metrics_expected_cited": EXPECT,
    "float32_accumulation_noise": noise,
    "files": {f: sha(f"{OUT}/{f}") for f in sorted(os.listdir(OUT)) if not f.endswith(".json")},
}
with open(f"{OUT}/manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(json.dumps(metrics, indent=2))
print("\nAll self-checks PASSED. Fixtures written to", OUT)
