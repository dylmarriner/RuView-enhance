#!/usr/bin/env python3
"""Offline honest evaluation of webcam-supervised activity sessions.

Reads the session JSONL files the sensing server writes under
`<data-dir>/activity/sessions/` (one header line, then one window per line),
fits a standardised k-NN on the TRAIN sessions only and scores the separately
recorded TEST sessions. Reports accuracy against both baselines (1/num_classes
and always-predict-the-training-majority), the confusion matrix and a
leave-one-link-out ablation. It mirrors `evaluate()` in
`v2/crates/wifi-densepose-sensing-server/src/activity.rs`, so the numbers are a
reproducer for the server's `POST /api/v1/activity/evaluate`.

Method adapted from wifi-sensing-kit by Andris Gauracs (MIT licence),
https://github.com/andrisgauracs/wifi-sensing-kit (kit/evaluate.py).

Numbers are MEASURED only for the room/operator that recorded the sessions.
They are not a camera-grade or cross-environment claim.

Usage:
  python3 scripts/eval_activity.py data/activity/sessions [--k 7] [--json]
"""
import argparse
import collections
import glob
import json
import math
import os
import sys

FEATURES_PER_LINK = 156  # 52 mean + 52 std + 52 mean |delta|, must match activity.rs
MAX_TRAIN = 2000  # same even-stride caps as the server, so results agree
MAX_TEST = 1500
LABEL_OK = set("abcdefghijklmnopqrstuvwxyz0123456789_-")


def load_sessions(directory):
    sessions = []
    for path in sorted(glob.glob(os.path.join(directory, "*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            try:
                head = json.loads(fh.readline())
            except json.JSONDecodeError:
                continue
            if head.get("type") != "header" or head.get("features_per_link") != FEATURES_PER_LINK:
                continue
            if head.get("kind") not in ("train", "test") or not head.get("link_ids"):
                continue
            dim = len(head["link_ids"]) * FEATURES_PER_LINK
            rows = []
            for line in fh:
                try:
                    w = json.loads(line)
                except json.JSONDecodeError:
                    continue
                x, label = w.get("x"), w.get("label")
                ok_label = isinstance(label, str) and 0 < len(label) <= 32 and set(label) <= LABEL_OK
                if ok_label and isinstance(x, list) and len(x) == dim and all(map(math.isfinite, x)):
                    rows.append((label, x))
            sessions.append({"id": head["id"], "kind": head["kind"], "links": head["link_ids"], "rows": rows})
    return sessions


def stride(items, cap):
    n = len(items)
    return items if n <= cap else [items[i * n // cap] for i in range(cap)]


def knn_predict(train_x, train_y, x, k, n_labels):
    d = sorted(range(len(train_x)), key=lambda i: (sum((a - b) ** 2 for a, b in zip(train_x[i], x)), i))[:k]
    votes = [0] * n_labels
    for i in d:
        votes[train_y[i]] += 1
    return max(range(n_labels), key=lambda j: (votes[j], -j))  # lowest label index wins ties


def accuracy(train_x, train_y, test_x, test_y, k, n_labels, cols):
    tx = [[r[c] for c in cols] for r in train_x]
    hits = sum(
        knn_predict(tx, train_y, [r[c] for c in cols], k, n_labels) == y for r, y in zip(test_x, test_y)
    )
    return hits / len(test_y)


def evaluate(sessions, k):
    train_sessions = [s for s in sessions if s["kind"] == "train"]
    if not train_sessions:
        sys.exit("no training session found")
    ref = train_sessions[0]["links"]
    usable = [s for s in sessions if s["links"] == ref]
    excluded = [s["id"] for s in sessions if s["links"] != ref]
    tr = [r for s in usable if s["kind"] == "train" for r in s["rows"]]
    te = [r for s in usable if s["kind"] == "test" for r in s["rows"]]
    if not te:
        sys.exit("no test-session windows: record a separate test session first")
    if len(tr) < 10 or len({y for y, _ in tr}) < 2:
        sys.exit("need at least 10 training windows across 2+ labels")
    labels = sorted({y for y, _ in tr} | {y for y, _ in te})
    li = {l: i for i, l in enumerate(labels)}
    tr_s, te_s = stride(tr, MAX_TRAIN), stride(te, MAX_TEST)
    dim = len(ref) * FEATURES_PER_LINK
    n = len(tr_s)
    mean = [sum(r[1][c] for r in tr_s) / n for c in range(dim)]  # fitted on TRAIN only
    sd = [math.sqrt(sum((r[1][c] - mean[c]) ** 2 for r in tr_s) / n) + 1e-6 for c in range(dim)]
    z = lambda x: [(v - m) / s for v, m, s in zip(x, mean, sd)]
    trx, try_ = [z(r[1]) for r in tr_s], [li[r[0]] for r in tr_s]
    tex, tey = [z(r[1]) for r in te_s], [li[r[0]] for r in te_s]
    all_cols = list(range(dim))
    tx = trx
    pred = [knn_predict(tx, try_, x, k, len(labels)) for x in tex]
    acc = sum(p == t for p, t in zip(pred, tey)) / len(tey)
    conf = {l: collections.Counter() for l in labels}
    for p, t in zip(pred, tey):
        conf[labels[t]][labels[p]] += 1
    counts = collections.Counter(try_)
    maj = max(range(len(labels)), key=lambda j: (counts[j], -j))
    out = {
        "k": k, "links": ref, "n_train": len(tr), "n_test": len(te), "subsampled": len(tr_s) < len(tr) or len(te_s) < len(te),
        "excluded_sessions": excluded, "labels": labels, "accuracy": acc, "chance_baseline": 1 / len(labels),
        "majority_class": labels[maj], "majority_baseline": sum(t == maj for t in tey) / len(tey),
        "confusion": {l: dict(c) for l, c in conf.items() if sum(c.values())}, "link_importance": [],
    }
    if len(ref) >= 2:
        for i, link in enumerate(ref):
            lo = i * FEATURES_PER_LINK
            cols = [c for c in all_cols if c < lo or c >= lo + FEATURES_PER_LINK]
            a = accuracy(trx, try_, tex, tey, k, len(labels), cols)
            out["link_importance"].append({"link": link, "accuracy_without": a, "accuracy_drop": acc - a})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions_dir", help="<data-dir>/activity/sessions")
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = ap.parse_args()
    if not 1 <= args.k <= 51:
        sys.exit("--k must be in 1..51")
    res = evaluate(load_sessions(args.sessions_dir), args.k)
    if args.json:
        print(json.dumps(res, indent=2))
        return
    print(f"links: {', '.join(res['links'])}")
    print(f"train windows: {res['n_train']}   test windows: {res['n_test']}   k={res['k']}"
          + ("   (subsampled)" if res["subsampled"] else ""))
    if res["excluded_sessions"]:
        print(f"excluded (different link set): {', '.join(res['excluded_sessions'])}")
    print(f"\nTEST SESSION ACCURACY: {100 * res['accuracy']:.1f}%  [MEASURED, this room/operator only]")
    print(f"  chance (1/{len(res['labels'])} classes): {100 * res['chance_baseline']:.1f}%")
    print(f"  always '{res['majority_class']}' (train majority): {100 * res['majority_baseline']:.1f}%")
    w = max(len(l) for l in res["labels"])
    print("\n" + " " * (w + 2) + "  ".join(l[:7].rjust(7) for l in res["labels"]))
    for truth, row in res["confusion"].items():
        print(f"{truth:>{w}}  " + "  ".join(str(row.get(p, 0)).rjust(7) for p in res["labels"]))
    print("(rows: what the camera saw, columns: what WiFi guessed)")
    if res["link_importance"]:
        print("\nleave-one-link-out (accuracy without the link):")
        for x in sorted(res["link_importance"], key=lambda r: -r["accuracy_drop"]):
            print(f"  {x['link']:>8}: {100 * x['accuracy_without']:.1f}%  (drop {100 * x['accuracy_drop']:+.1f} pts)")


if __name__ == "__main__":
    main()
