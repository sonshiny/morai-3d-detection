#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Category 3 — 50m boundary audit.
For EACH of the 13 entries in pretrain_verify/g_membership_audit.json
(12 v2_only + 1 v3_only), draw a BEV: the 50m ring, v2 center (red) and v3
center (green) of that track/frame, radial distances as text, and the box just
inside vs outside. Writes into visual_audit/boundary_50m/. Read-only.
"""
import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _audit_common as ac

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "boundary_50m")
os.makedirs(OUT, exist_ok=True)
AUDIT = os.path.join(ac.PROJECT_ROOT, "pretrain_verify", "g_membership_audit.json")

RED = "#ff3030"
GREEN = "#20e020"


def render(entry, idx):
    scen = entry["scen"]; fr = entry["frame_id"]; tid = entry["track_id"]
    kind = entry["kind"]
    stem = ac.frameid_to_stem(scen, fr, "v3") or ac.frameid_to_stem(scen, fr, "v2")
    v2 = ac.load_boxes(scen, stem, "v2") if stem else {}
    v3 = ac.load_boxes(scen, stem, "v3") if stem else {}
    b2 = v2.get(tid); b3 = v3.get(tid)
    r2 = entry.get("v2_radial"); r3 = entry.get("v3_radial")

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.2))
    for ax in axes:
        ax.plot(0, 0, marker=(3, 0, -90), color="w", markersize=13,
                markeredgecolor="k", zorder=6)
        ring = mpatches.Circle((0, 0), 50.0, fill=False, edgecolor="#00aaaa",
                               lw=1.6, ls="--", alpha=0.9, zorder=2)
        ax.add_patch(ring)
        if b2:
            ac.draw_bev_box(ax, b2, RED, lw=2.2, ls="--")
            ax.plot(b2["x"], b2["y"], "o", mfc="none", mec=RED, ms=13, mew=2.2,
                    zorder=7, label="v2 center")
        if b3:
            ac.draw_bev_box(ax, b3, GREEN, lw=1.6)
            ax.plot(b3["x"], b3["y"], "o", color=GREEN, ms=6, zorder=8, label="v3 center")
        if b2 and b3:
            ax.annotate("", xy=(b3["x"], b3["y"]), xytext=(b2["x"], b2["y"]),
                        arrowprops=dict(arrowstyle="-|>", color="w", lw=2.0))
        ax.set_aspect("equal"); ax.grid(True, color="#cccccc", ls=":", alpha=0.5)
        ax.set_xlabel("x forward (m)"); ax.set_ylabel("y left (m)")

    axes[0].set_xlim(-55, 55); axes[0].set_ylim(-55, 55)
    axes[0].set_title("full BEV (ego + 50m ring)", fontsize=10)
    axes[0].legend(loc="upper left", fontsize=8)

    # zoom on the box, keep some of the ring arc in view
    cx = (b3["x"] if b3 else (b2["x"] if b2 else 0.0))
    cy = (b3["y"] if b3 else (b2["y"] if b2 else 0.0))
    axes[1].set_xlim(cx - 4, cx + 4); axes[1].set_ylim(cy - 4, cy + 4)
    axes[1].set_title("zoom on boundary (±4 m)", fontsize=10)

    ins2 = "inside" if (r2 is not None and r2 <= 50.0) else "OUTSIDE"
    ins3 = "inside" if (r3 is not None and r3 <= 50.0) else "OUTSIDE"
    txt = (f"{scen}  frame={fr}  track={tid}  kind={kind}\n"
           f"v2_radial = {r2:.3f} m  -> {ins2} 50m\n"
           f"v3_radial = {r3:.3f} m  -> {ins3} 50m\n"
           f"(v2 kept in {'v2' if kind=='v2_only' else 'v3'} filter, dropped in the other)")
    fig.suptitle(txt, fontsize=11, y=1.06)
    fname = f"{idx:02d}_{scen}_f{fr}_t{tid}_{kind}.png"
    out = os.path.join(OUT, fname)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return fname, r2, r3, ins2, ins3


def main():
    with open(AUDIT, encoding="utf-8") as f:
        audit = json.load(f)
    diffs = audit["diffs"]
    manifest = []
    for i, entry in enumerate(diffs, 1):
        fname, r2, r3, ins2, ins3 = render(entry, i)
        manifest.append({"scen": entry["scen"], "frame_id": entry["frame_id"],
                         "track_id": entry["track_id"], "kind": entry["kind"],
                         "v2_radial": r2, "v3_radial": r3,
                         "v2_side": ins2, "v3_side": ins3, "file": fname})
        print(f"[boundary] {fname}: v2_r={r2:.3f}({ins2}) v3_r={r3:.3f}({ins3})")
    with open(os.path.join(OUT, "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n[boundary] wrote {len(manifest)} figures")


if __name__ == "__main__":
    main()
