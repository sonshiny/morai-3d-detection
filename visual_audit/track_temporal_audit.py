#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Category 4 — track temporal audit.
For each scene pick >=3 tracks with the highest mean corr_dist. For each track,
multi-panel over frames (x=frame_id): v2 x & v3 x | v2 y & v3 y | corr_dist |
rel_speed. Mark teleport/discontinuity (large per-step jump) and note reversals.
Writes into visual_audit/track_temporal/. Read-only.
"""
import os
import sys
import json
import math
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _audit_common as ac

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "track_temporal")
os.makedirs(OUT, exist_ok=True)
SCENS = ["scen05", "scen77", "scen144"]
MIN_FRAMES = 8
TOP_TRACKS = 3
JUMP_SPEED_MPS = 25.0     # per-step apparent speed above this = suspicious jump/teleport
JUMP_MIN_M = 3.0          # and displacement at least this many meters

RED = "#d62020"
GREEN = "#1f9e1f"


def build_series(scen):
    rl, _ = ac.relspeed_lookup(scen)
    tmap = ac.scene_timing_map(scen)
    series = defaultdict(list)  # track_id -> list of dict
    for stem in sorted(tmap.keys()):
        v2 = ac.load_boxes(scen, stem, "v2")
        v3 = ac.load_boxes(scen, stem, "v3")
        for tid, b3 in v3.items():
            b2 = v2.get(tid)
            rlbox = rl.get(stem, {}).get(tid)
            series[tid].append({
                "frame_id": int(b3["frame_id"]),
                "timestamp": float(b3["timestamp"]),
                "v3x": float(b3["x"]), "v3y": float(b3["y"]),
                "v2x": (float(b2["x"]) if b2 else np.nan),
                "v2y": (float(b2["y"]) if b2 else np.nan),
                "corr_dist": float(b3.get("corr_dist", 0.0)),
                "rel_speed": (float(rlbox["rel_speed"]) if rlbox else np.nan),
                "class_id": b3["class_id"],
            })
    for tid in series:
        series[tid].sort(key=lambda d: d["frame_id"])
    return series


def detect_jumps(rows, xkey, ykey):
    """return list of frame_ids where per-step apparent speed exceeds threshold."""
    jumps = []
    for a, b in zip(rows[:-1], rows[1:]):
        dt = b["timestamp"] - a["timestamp"]
        if dt <= 0:
            continue
        if not (np.isfinite(a[xkey]) and np.isfinite(b[xkey])):
            continue
        disp = math.hypot(b[xkey] - a[xkey], b[ykey] - a[ykey])
        if disp >= JUMP_MIN_M and disp / dt >= JUMP_SPEED_MPS:
            jumps.append((b["frame_id"], disp, disp / dt))
    return jumps


REVERSAL_MIN_M = 0.05     # only count direction changes with >=5cm steps (ignore mm jitter)


def count_reversals(vals):
    """count sign reversals of first difference for steps >= REVERSAL_MIN_M."""
    d = np.diff(np.asarray(vals, dtype=np.float64))
    d = d[np.abs(d) > REVERSAL_MIN_M]
    if d.size < 2:
        return 0
    s = np.sign(d)
    return int(np.sum(s[1:] != s[:-1]))


def render(scen, tid, rows, rank, mean_cd, median_cd):
    fids = [r["frame_id"] for r in rows]
    v3x = [r["v3x"] for r in rows]; v3y = [r["v3y"] for r in rows]
    v2x = [r["v2x"] for r in rows]; v2y = [r["v2y"] for r in rows]
    cd = [r["corr_dist"] for r in rows]; rel = [r["rel_speed"] for r in rows]
    cls = ac.CLASS_NAME.get(rows[0]["class_id"], "?")

    jumps_v3 = detect_jumps(rows, "v3x", "v3y")
    jumps_v2 = detect_jumps(rows, "v2x", "v2y")
    rev_x = count_reversals(v3x); rev_y = count_reversals(v3y)
    rev_cd = count_reversals(cd)

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    axes[0].plot(fids, v2x, "-o", color=RED, ms=3, lw=1.2, label="v2 x")
    axes[0].plot(fids, v3x, "-o", color=GREEN, ms=3, lw=1.2, label="v3 x")
    axes[0].set_ylabel("x forward (m)"); axes[0].legend(fontsize=8, loc="best")

    axes[1].plot(fids, v2y, "-o", color=RED, ms=3, lw=1.2, label="v2 y")
    axes[1].plot(fids, v3y, "-o", color=GREEN, ms=3, lw=1.2, label="v3 y")
    axes[1].set_ylabel("y left (m)"); axes[1].legend(fontsize=8, loc="best")

    axes[2].plot(fids, cd, "-o", color="#7030a0", ms=3, lw=1.2)
    axes[2].set_ylabel("corr_dist (m)")
    axes[2].axhline(np.nanmean(cd), color="#7030a0", ls=":", alpha=0.6,
                    label=f"mean={np.nanmean(cd):.3f}")
    axes[2].legend(fontsize=8, loc="best")

    axes[3].plot(fids, rel, "-o", color="#c05000", ms=3, lw=1.2)
    axes[3].set_ylabel("rel_speed (m/s)"); axes[3].set_xlabel("frame_id")

    # mark jumps
    jump_fids = set(j[0] for j in jumps_v3) | set(j[0] for j in jumps_v2)
    for jf in jump_fids:
        for ax in axes:
            ax.axvline(jf, color="red", ls="--", alpha=0.5, lw=1.0)
    for ax in axes:
        ax.grid(True, ls=":", alpha=0.4)

    jtxt = "none"
    if jump_fids:
        jtxt = ", ".join(f"f{j[0]}({j[1]:.1f}m,{j[2]:.0f}m/s)" for j in
                         sorted(jumps_v3 + jumps_v2, key=lambda t: t[0]))
    fig.suptitle(
        f"{scen} track={tid} class={cls}  (scene-rank #{rank} by mean corr_dist)\n"
        f"n_frames={len(rows)}  mean_corr={mean_cd:.3f}  median_corr={median_cd:.3f} m  |  "
        f"jumps(>{JUMP_SPEED_MPS:.0f}m/s & >{JUMP_MIN_M:.0f}m): {jtxt}  |  "
        f"reversals(>{REVERSAL_MIN_M*100:.0f}cm) v3x/v3y/corr: {rev_x}/{rev_y}/{rev_cd}  "
        f"(red dashed = flagged jump)",
        fontsize=10.5, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fname = f"{scen}_t{tid}_rank{rank}_mcd{mean_cd:.3f}.png"
    out = os.path.join(OUT, fname)
    fig.savefig(out, dpi=105, bbox_inches="tight")
    plt.close(fig)
    return fname, list(jump_fids), rev_x, rev_y, rev_cd


def main():
    manifest = []
    for scen in SCENS:
        series = build_series(scen)
        stats = []
        for tid, rows in series.items():
            if len(rows) < MIN_FRAMES:
                continue
            cds = np.array([r["corr_dist"] for r in rows], dtype=np.float64)
            stats.append((tid, float(np.mean(cds)), float(np.median(cds)), len(rows)))
        stats.sort(key=lambda t: -t[1])
        chosen = stats[:TOP_TRACKS]
        print(f"[temporal] {scen}: tracks>= {MIN_FRAMES}f = {len(stats)}, "
              f"top{TOP_TRACKS} by mean corr_dist = "
              + ", ".join(f"t{t[0]}(mean={t[1]:.3f},n={t[3]})" for t in chosen))
        for rank, (tid, mean_cd, median_cd, n) in enumerate(chosen, 1):
            fname, jfids, rx, ry, rcd = render(scen, tid, series[tid], rank, mean_cd, median_cd)
            manifest.append({"scen": scen, "track_id": tid, "rank": rank,
                             "n_frames": n, "mean_corr_dist": round(mean_cd, 4),
                             "median_corr_dist": round(median_cd, 4),
                             "flagged_jump_frames": sorted(jfids),
                             "v3_reversals_x": rx, "v3_reversals_y": ry,
                             "corr_dist_reversals": rcd,
                             "file": fname})
            print(f"[temporal]   {fname}: jumps={sorted(jfids)} reversals x/y/corr={rx}/{ry}/{rcd}")
    with open(os.path.join(OUT, "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n[temporal] wrote {len(manifest)} figures")


if __name__ == "__main__":
    main()
