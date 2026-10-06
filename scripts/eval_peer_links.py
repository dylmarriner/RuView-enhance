#!/usr/bin/env python3
"""Evaluate peer-link presence detection against labelled empty/occupied periods.

Input: the JSONL written by the Pi logger (one {"t": epoch_s, "d": <GET
/api/v1/peer-links>} record per line) plus labelled time ranges. Output, for a
sweep of thresholds: occupied-period detection rate and empty-period false
alarm rate for each room signal. Sets PEER_WANDER_THRESHOLD and
PEER_PRESENCE_FRACTION from measured data instead of starting guesses.

Usage:
  eval_peer_links.py links.jsonl --empty 1791300000-1791300600 \\
      --occupied 1791290000-1791299000 [--occupied ...]
"""
import argparse
import json
import sys


def parse_range(text):
    start, end = text.split("-")
    return float(start), float(end)


def load(path):
    records = []
    with open(path) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec.get("d"), dict) and rec["d"].get("links"):
                records.append(rec)
    return records


def label_of(t, empty, occupied):
    if any(a <= t <= b for a, b in empty):
        return "empty"
    if any(a <= t <= b for a, b in occupied):
        return "occupied"
    return None


def fraction(links, key, threshold):
    """Fraction of links whose `key` is >= threshold (links lacking it skipped)."""
    vals = [l[key] for l in links if l.get(key) is not None]
    return (sum(v >= threshold for v in vals) / len(vals)) if vals else None


def rates(samples, score):
    """(detection rate on occupied, false-alarm rate on empty) for a boolean score fn."""
    out = {}
    for lab in ("occupied", "empty"):
        hits = [score(s) for s in samples if s["label"] == lab]
        hits = [h for h in hits if h is not None]
        out[lab] = (sum(hits) / len(hits), len(hits)) if hits else (None, 0)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log")
    ap.add_argument("--empty", action="append", type=parse_range, default=[])
    ap.add_argument("--occupied", action="append", type=parse_range, default=[])
    args = ap.parse_args(argv)
    if not args.empty or not args.occupied:
        ap.error("need at least one --empty and one --occupied range")

    samples = []
    for rec in load(args.log):
        lab = label_of(rec["t"], args.empty, args.occupied)
        if lab:
            samples.append({"label": lab, "links": rec["d"]["links"], "room": rec["d"].get("room", {})})
    n_occ = sum(s["label"] == "occupied" for s in samples)
    print(f"samples: {len(samples)} ({n_occ} occupied, {len(samples) - n_occ} empty)")
    if not samples:
        return 1

    def show(name, score):
        r = rates(samples, score)
        det, n1 = r["occupied"]
        fa, n0 = r["empty"]
        fmt = lambda x: "  n/a" if x is None else f"{x:5.1%}"
        print(f"  {name:<34} detect {fmt(det)} (n={n1})   false-alarm {fmt(fa)} (n={n0})")

    print("Server room votes as logged:")
    show("presence (motion FSM)", lambda s: s["room"].get("presence"))
    show("still_presence (wander)", lambda s: s["room"].get("still_presence"))

    print("Sweep: fraction of links ACTIVE (motion FSM) >= f")
    for f in (0.1, 0.25, 0.4, 0.5):
        show(f"active fraction >= {f}", lambda s, f=f: (
            None if not s["links"] else
            sum(bool(l.get("active")) for l in s["links"]) / len(s["links"]) >= f))

    print("Sweep: wander threshold w, link fraction >= 0.25")
    for w in (0.05, 0.1, 0.15, 0.2, 0.3):
        show(f"wander >= {w}", lambda s, w=w: (
            None if (fr := fraction(s["links"], "wander", w)) is None else fr >= 0.25))
    return 0


if __name__ == "__main__":
    sys.exit(main())
