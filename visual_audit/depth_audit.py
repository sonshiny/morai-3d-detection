#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Category 1 — depth audit.
For each scene x camera, pick ONE frame that has BOTH static structure and a
moving object (rel_speed>1 present) and depth points, preferring a frame where a
mover is visible in that camera. Save side-by-side: original | depth overlay.
Writes into visual_audit/depth/. Read-only against dataset.
"""
import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _audit_common as ac
import cv2

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "depth")
os.makedirs(OUT, exist_ok=True)
SCENS = ["scen05", "scen77", "scen144"]
CAMS = ["cam_front", "cam_front_left", "cam_front_right"]
MOVER_MIN = 1.0

# depth_gt (u,v) were generated in the 704x256 training-input coordinate system
# (generate_depth_gt.py uses scale_intrinsic_for_input -> IMG_WIDTH=704,IMG_HEIGHT=256,
# and cv2.resize is an anisotropic pure scale, no crop). To overlay on the original
# 1600x900 jpg we scale the points back up by these exact factors.
DEPTH_SRC_W, DEPTH_SRC_H = 704, 256
SX = ac.CAM_W / DEPTH_SRC_W    # 1600/704
SY = ac.CAM_H / DEPTH_SRC_H    # 900/256


def load_depth(scen, stem, cam):
    p = ac.depth_path(scen, stem, cam)
    if not os.path.isfile(p):
        return None
    a = np.load(p)
    if a.ndim != 2 or a.shape[0] == 0:
        return None
    return a  # [N,3] u,v,depth


def pick_frame(scen, cam, rl, boxes_cache):
    """Return (stem, n_movers_vis, n_depth, mover_tids) best candidate for scene/cam."""
    best = None
    for stem in sorted(rl.keys()):
        dep = load_depth(scen, stem, cam)
        if dep is None:
            continue
        movers = [tid for tid, b in rl[stem].items() if b["rel_speed"] > MOVER_MIN]
        if not movers:
            continue
        # is a mover visible in this cam?
        vis = []
        for tid in movers:
            b = rl[stem][tid]
            fake = {"x": b["x"], "y": b["y"], "z_center": 1.0}
            if ac.box_center_in_camera(fake, cam):
                vis.append(tid)
        score = (len(vis), dep.shape[0])
        cand = (stem, len(vis), dep.shape[0], vis if vis else movers)
        if best is None or score > (best[1], best[2]):
            best = cand
    return best


def render(scen, cam, stem, mover_tids, rl):
    img = cv2.cvtColor(cv2.imread(ac.image_path(scen, stem, cam)), cv2.COLOR_BGR2RGB)
    dep = load_depth(scen, stem, cam)
    u, v, d = dep[:, 0] * SX, dep[:, 1] * SY, dep[:, 2]  # 704x256 -> 1600x900

    fig, axes = plt.subplots(1, 2, figsize=(20, 6.2))
    axes[0].imshow(img)
    axes[0].set_title(f"{scen}/{cam}/{stem}  — original", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(img)
    order = np.argsort(-d)  # far first so near points draw on top
    sc = axes[1].scatter(u[order], v[order], c=d[order], s=4, cmap="turbo",
                         vmin=float(np.percentile(d, 2)), vmax=float(np.percentile(d, 98)),
                         alpha=0.7, linewidths=0)
    cb = fig.colorbar(sc, ax=axes[1], fraction=0.03, pad=0.01)
    cb.set_label("depth (m)")
    axes[1].set_xlim(0, ac.CAM_W)
    axes[1].set_ylim(ac.CAM_H, 0)
    n_mov = sum(1 for tid, b in rl[stem].items() if b["rel_speed"] > MOVER_MIN)
    mv = max((rl[stem][t]["rel_speed"] for t in rl[stem]), default=0.0)
    axes[1].set_title(f"depth_gt overlay  — {dep.shape[0]} pts | movers(rel>1)={n_mov} "
                      f"max rel_speed={mv:.1f} m/s", fontsize=11)
    axes[1].axis("off")
    fig.tight_layout()
    out = os.path.join(OUT, f"{scen}_{cam}_{stem}_depth.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out, dep.shape[0], n_mov, float(mv)


def main():
    manifest = []
    for scen in SCENS:
        rl, _ = ac.relspeed_lookup(scen)
        for cam in CAMS:
            pick = pick_frame(scen, cam, rl, None)
            if pick is None:
                print(f"[WARN] {scen}/{cam}: no frame with mover+depth found")
                continue
            stem, nvis, ndep, movers = pick
            out, ndep2, n_mov, mv = render(scen, cam, stem, movers, rl)
            manifest.append({"scen": scen, "cam": cam, "stem": stem,
                             "file": os.path.basename(out),
                             "n_depth_pts": ndep2, "n_movers": n_mov,
                             "max_rel_speed": mv, "mover_visible_in_cam": nvis})
            print(f"[depth] {scen}/{cam}/{stem}: depth_pts={ndep2} movers={n_mov} "
                  f"maxrel={mv:.1f} mover_vis_in_cam={nvis} -> {os.path.basename(out)}")
    with open(os.path.join(OUT, "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n[depth] wrote {len(manifest)} figures")


if __name__ == "__main__":
    main()
