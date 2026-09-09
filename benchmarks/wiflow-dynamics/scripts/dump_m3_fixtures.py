"""Dump the remaining fixtures M3 needs, so no Rust code ever touches the pickle.

Same justification as make_fixtures.py: data extraction, not model or training logic.
After this, the Rust side needs only .npy files.

Writes:
  train_idx.npy, val_idx.npy   -- .npy siblings of split_indices.npz (a zip, which
                                  the Rust reader deliberately does not support)
  all_gt_clean.npy             -- (360000, 30) f32: every window's ground-truth pose,
                                  zero-cleaned, indexed by WINDOW index
  window_to_file.npy           -- (360000,) i64, for the same-file pair guard
  pair_t.npy                   -- (P,) i64 window indices t such that (t, t+1) is a
                                  valid temporally-consecutive same-file pair
Also verifies globally (not just on file 0) that consecutive window indices within a
file are consecutive in time.
"""
import os
import pickle

import numpy as np

B = os.path.expanduser("~/wiflow-std-bench/preprocessed_csi_data/")
F = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/fixtures/"

wi = np.load(B + "window_info.npz")
w2f, w2fr = wi["window_to_file"], wi["window_to_frame"]
fi = np.load(B + "file_info.npz", allow_pickle=True)
fm = pickle.load(open(B + "file_mappings.pkl", "rb"))
kf = fi["keypoints_files"]

N = len(w2f)
assert N == 360000, N

# --- global temporal-adjacency check (the M3 design depends on this) -------------
same_file = w2f[:-1] == w2f[1:]
consecutive = w2fr[1:] == w2fr[:-1] + 1
assert np.all(consecutive[same_file]), (
    "within-file consecutive window indices are NOT consecutive in time -- "
    "next-observation dynamics must be redefined"
)
pair_t = np.nonzero(same_file)[0].astype(np.int64)
print(f"valid (t, t+1) pairs: {len(pair_t)} of {N - 1} adjacent index pairs")
assert len(pair_t) == N - 500, "expected exactly one broken pair per file boundary"

# --- full cleaned ground truth, indexed by WINDOW index --------------------------
ak = np.load(B + "all_keypoints.npy")
gfi = np.array([fm[kf[w2f[i]]]["start_idx"] + w2fr[i] for i in range(N)], dtype=np.int64)
gt = ak[gfi].copy()
nz = (gt[..., 0] != 0) | (gt[..., 1] != 0)
n_cleaned = 0
for i in range(N):
    m = nz[i]
    if m.any() and not m.all():
        gt[i][~m] = gt[i][m].mean(axis=0)
        n_cleaned += 1
print(f"zero-cleaned {n_cleaned} of {N} frames")

# Cross-check against the already-validated M2 test fixture.
ti = np.load(F + "test_idx.npy")
ref = np.load(F + "test_gt.npy")
assert np.array_equal(gt[ti], ref), "all_gt_clean disagrees with the validated M2 test GT"
print("cross-check vs M2 test_gt.npy: EXACT MATCH")

sp = np.load(F + "split_indices.npz")
np.save(F + "train_idx.npy", sp["train"].astype(np.int32))
np.save(F + "val_idx.npy", sp["val"].astype(np.int32))
np.save(F + "all_gt_clean.npy", gt.reshape(N, 30).astype(np.float32))
np.save(F + "window_to_file.npy", w2f.astype(np.int64))
np.save(F + "pair_t.npy", pair_t)
print("wrote M3 fixtures to", F)
