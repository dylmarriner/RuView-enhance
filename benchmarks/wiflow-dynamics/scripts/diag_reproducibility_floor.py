"""Measure the reproducibility floor of the WiFlow-STD forward pass.

Question this answers: results/eval_retrained.json records PCK@20 = 0.9608815 for a
specific checkpoint. Re-running the same checkpoint on the same box today gives
0.9608877 -- a 6.1e-6 absolute gap. Is that a protocol drift (a real defect) or
GPU numeric nondeterminism (an irreducible floor)?

MPJPE already argues for the latter: it is threshold-free and differs by 1.5e-4
relative, while three different accumulation orders of the SAME predictions agree to
5e-10. So the predictions themselves differ. This script tests the leading
hypothesis -- TF32 matmul/conv kernels, on by default for Ampere+ -- by running the
identical checkpoint over the identical test split under both settings.

Establishing this floor is itself an anti-Goodhart guard: no future 'improvement'
smaller than the floor is a real improvement.
"""
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
RECORDED = {"pck@20_full": 0.9608815324571398, "mpjpe_full": 0.009834060806367133}

sys.path.insert(0, f"{BENCH}/upstream")
_real_load = np.load
np.load = lambda p, *a, **k: (
    _real_load(p, *a, **{**k, "mmap_mode": "r"})
    if isinstance(p, str) and p.endswith("csi_windows.npy")
    else _real_load(p, *a, **k)
)
from dataset import PreprocessedCSIKeypointsDataset, create_preprocessed_train_val_test_loaders  # noqa: E402
from models.pose_model import WiFlowPoseModel  # noqa: E402

np.load = _real_load

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

ds = PreprocessedCSIKeypointsDataset(
    data_dir=f"{BENCH}/preprocessed_csi_data", keypoint_scale=1000.0, enable_temporal_clean=True
)
_, _, test_loader = create_preprocessed_train_val_test_loaders(
    dataset=ds, batch_size=256, num_workers=2, random_seed=42
)

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


def forward_all(tf32: bool):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    preds, gts = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            out = model(bx.to(device))
            preds.append(out.reshape(out.shape[0], 15, 2).float().cpu().numpy())
            gts.append(by.reshape(by.shape[0], 15, 2).float().numpy())
    return np.concatenate(preds).astype(np.float32), np.concatenate(gts).astype(np.float32)


def score(p, g):
    p, g = p.astype(np.float64), g.astype(np.float64)
    torso = np.sqrt(((g[:, 2] - g[:, 12]) ** 2).sum(axis=1)).clip(min=0.01)
    d = np.sqrt(((p - g) ** 2).sum(axis=2))
    nd = d / torso[:, None]
    return {"mpjpe": float(d.mean()), "pck@20": float((nd <= 0.2).mean())}


p_on, gt = forward_all(True)
p_off, gt2 = forward_all(False)
assert np.array_equal(gt, gt2), "ground truth differed between runs -- loader is nondeterministic"

s_on, s_off = score(p_on, gt), score(p_off, gt)
pred_absdiff = float(np.abs(p_on - p_off).max())
pred_reldiff = float(np.abs(p_on - p_off).mean() / (np.abs(p_on).mean() + 1e-12))

report = {
    "question": "is the 6.1e-6 PCK@20 gap vs eval_retrained.json protocol drift or numeric floor?",
    "recorded_2026_06_10": RECORDED,
    "tf32_on": s_on,
    "tf32_off": s_off,
    "tf32_on_vs_off": {
        "pred_max_abs_diff": pred_absdiff,
        "pred_mean_rel_diff": pred_reldiff,
        "pck@20_delta": abs(s_on["pck@20"] - s_off["pck@20"]),
        "mpjpe_delta": abs(s_on["mpjpe"] - s_off["mpjpe"]),
    },
    "gap_to_recorded": {
        "tf32_on_pck@20": abs(s_on["pck@20"] - RECORDED["pck@20_full"]),
        "tf32_off_pck@20": abs(s_off["pck@20"] - RECORDED["pck@20_full"]),
        "tf32_on_mpjpe": abs(s_on["mpjpe"] - RECORDED["mpjpe_full"]),
        "tf32_off_mpjpe": abs(s_off["mpjpe"] - RECORDED["mpjpe_full"]),
    },
}
os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/reproducibility_floor.json", "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
