#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Category 2 — v2 vs v3 correction audit (camera projection AND BEV).
Join v2/v3 by (scen, frame_id, track_id). Select:
  - Top 20 by v3 corr_dist (across all 3 scenes)
  - 10 boxes with corr_dist in [0.2, 0.4)
  - 10 boxes with corr_dist < 0.05
  - plus at least one of each category: stationary(rel<0.5), same-dir moving,
    oncoming(large rel), pedestrian(class 1).
Per selected box: figure with camera projection | BEV(50m ring) | BEV zoom.
  v2=RED, v3=GREEN, WHITE arrow v2->v3, BLUE=object vel, YELLOW=ego vel.
Writes into visual_audit/v2_vs_v3/. Read-only against dataset.
"""
import os
import sys
import json
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _audit_common as ac

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2_vs_v3")
os.makedirs(OUT, exist_ok=True)
SCENS = ["scen05", "scen77", "scen144"]
ARROW_SEC = 1.0          # velocity arrow = velocity(m/s) * this (m)

RED = "#ff3030"
GREEN = "#20e020"
WHITE = "#ffffff"
BLUE = "#3080ff"
YELLOW = "#ffd000"


def collect():
    """Return list of joined box records across all scenes."""
    recs = []
    for scen in SCENS:
        tmap = ac.scene_timing_map(scen)
        emap = ac.ego_world_velocity_map(scen)
        rl, _ = ac.relspeed_lookup(scen)
        # iterate stems from timing map
        for stem, meta in tmap.items():
            v2 = ac.load_boxes(scen, stem, "v2")
            v3 = ac.load_boxes(scen, stem, "v3")
            if not v3:
                continue
            timing = meta["timing"]
            for tid, b3 in v3.items():
                if tid not in v2:
                    continue
                b2 = v2[tid]
                rlbox = rl.get(stem, {}).get(tid)
                recs.append({
                    "scen": scen, "stem": stem, "frame_id": int(b3["frame_id"]),
                    "track_id": tid, "class_id": b3["class_id"],
                    "corr_dist": float(b3.get("corr_dist", 0.0)),
                    "corr_dx": float(b3.get("corr_dx", 0.0)),
                    "corr_dy": float(b3.get("corr_dy", 0.0)),
                    "correction_valid": int(float(b3.get("correction_valid", 1))),
                    "v2": b2, "v3": b3,
                    "ego_dt_ms": float(timing.get("ego_dt_ms", float("nan"))),
                    "obj_dt_ms": float(timing.get("obj_dt_ms", float("nan"))),
                    "rel_speed": (float(rlbox["rel_speed"]) if rlbox else None),
                    "obj_speed": (float(rlbox["obj_speed"]) if rlbox else None),
                    "ego_speed": (float(rlbox["ego_speed"]) if rlbox else None),
                    "ego_vel": emap.get(stem),
                })
    return recs


def categorize(r):
    cats = []
    cls = r["class_id"]
    rel = r["rel_speed"]
    obj = r["obj_speed"]
    if cls == 1:
        cats.append("pedestrian")
    if rel is not None:
        if rel < 0.5:
            cats.append("stationary")
        if cls == 0 and obj is not None and obj > 1.0 and rel < 2.0:
            cats.append("same_dir_vehicle")
        if rel > 6.0:
            cats.append("oncoming")
    return cats


def select(recs):
    valid = [r for r in recs if r["correction_valid"] == 1]
    by_cd = sorted(valid, key=lambda r: -r["corr_dist"])
    picked = {}  # (scen,frame,track) -> {rec, buckets:set}

    def add(r, bucket):
        k = (r["scen"], r["frame_id"], r["track_id"])
        if k not in picked:
            picked[k] = {"rec": r, "buckets": set()}
        picked[k]["buckets"].add(bucket)

    for r in by_cd[:20]:
        add(r, "top20")
    mid = [r for r in valid if 0.2 <= r["corr_dist"] < 0.4]
    mid = sorted(mid, key=lambda r: -r["corr_dist"])[:10]
    for r in mid:
        add(r, "mid_0.2_0.4")
    small = [r for r in valid if r["corr_dist"] < 0.05]
    # spread small ones across scenes/tracks: take distinct tracks
    seen_tr = set()
    small_sel = []
    for r in sorted(small, key=lambda r: r["corr_dist"]):
        tk = (r["scen"], r["track_id"])
        if tk in seen_tr:
            continue
        seen_tr.add(tk)
        small_sel.append(r)
        if len(small_sel) >= 10:
            break
    for r in small_sel:
        add(r, "small_lt0.05")

    # ensure categories present among current selection
    present = set()
    for v in picked.values():
        for c in categorize(v["rec"]):
            present.add(c)
    needed = ["stationary", "same_dir_vehicle", "oncoming", "pedestrian"]
    extra_notes = []
    for cat in needed:
        if cat in present:
            continue
        # find a valid rec in that category with the largest corr_dist for visibility
        cands = [r for r in valid if cat in categorize(r)]
        if not cands:
            extra_notes.append(f"category '{cat}': NONE found in dataset (valid corrections)")
            continue
        cands = sorted(cands, key=lambda r: -r["corr_dist"])
        add(cands[0], f"extra_{cat}")
        extra_notes.append(f"category '{cat}': added extra example "
                           f"{cands[0]['scen']} f{cands[0]['frame_id']} t{cands[0]['track_id']} "
                           f"corr_dist={cands[0]['corr_dist']:.3f}")
    return picked, extra_notes


def _arrow(ax, p0, p1, color, lw=2.2, alpha=1.0):
    if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))):
        return
    ax.annotate("", xy=(p1[0], p1[1]), xytext=(p0[0], p0[1]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, alpha=alpha,
                                shrinkA=0, shrinkB=0))


def render_camera(ax, r, cam):
    img = cv2.imread(ac.image_path(r["scen"], r["stem"], cam))
    if img is None:
        ax.text(0.5, 0.5, "no image", ha="center"); ax.axis("off"); return
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    b2, b3 = r["v2"], r["v3"]
    _, c2 = ac.draw_cuboid(ax, b2, cam, RED, lw=2.0)
    _, c3 = ac.draw_cuboid(ax, b3, cam, GREEN, lw=2.0)
    # white displacement arrow v2->v3 (projected centers)
    if c2 and c3:
        _arrow(ax, c2, c3, WHITE, lw=2.5)
    # blue object velocity arrow from v3 center
    b = b3
    vx, vy = float(b["vx_ego"]), float(b["vy_ego"])
    p_end = [b["x"] + vx * ARROW_SEC, b["y"] + vy * ARROW_SEC, b["z_center"]]
    us, vs, valid, _ = ac.project_ego_points([[b["x"], b["y"], b["z_center"]], p_end], cam)
    if valid[0] and valid[1]:
        _arrow(ax, (us[0], vs[0]), (us[1], vs[1]), BLUE, lw=2.2)
    ax.set_xlim(0, ac.CAM_W); ax.set_ylim(ac.CAM_H, 0); ax.axis("off")
    ax.set_title(f"camera: {cam}", fontsize=10)


def render_bev(ax, r, zoom=False):
    scen, stem = r["scen"], r["stem"]
    b2, b3 = r["v2"], r["v3"]
    # background: all v3 boxes in frame faint
    allv3 = ac.load_boxes(scen, stem, "v3")
    for tid, bb in allv3.items():
        ac.draw_bev_box(ax, bb, "#888888", lw=0.8, alpha=0.5)
    # ego
    ax.plot(0, 0, marker=(3, 0, -90), color="w", markersize=13, markeredgecolor="k", zorder=6)
    # 50m ring
    ring = mpatches.Circle((0, 0), 50.0, fill=False, edgecolor="#00cccc", lw=1.2, ls="--", alpha=0.8)
    ax.add_patch(ring)
    # highlighted boxes
    ac.draw_bev_box(ax, b2, RED, lw=2.2)
    ac.draw_bev_box(ax, b3, GREEN, lw=2.2)
    # white displacement arrow
    _arrow(ax, (b2["x"], b2["y"]), (b3["x"], b3["y"]), WHITE, lw=2.5)
    # blue object velocity (ego frame vx,vy)
    vx, vy = float(b3["vx_ego"]), float(b3["vy_ego"])
    _arrow(ax, (b3["x"], b3["y"]),
           (b3["x"] + vx * ARROW_SEC, b3["y"] + vy * ARROW_SEC), BLUE, lw=2.2)
    # yellow ego velocity in ego frame at origin
    ev = r["ego_vel"]
    if ev:
        vw = np.array([ev["vx_world"], ev["vy_world"]])
        vego = ac._rot2(-ev["yaw"]) @ vw
        _arrow(ax, (0, 0), (vego[0] * ARROW_SEC, vego[1] * ARROW_SEC), YELLOW, lw=2.4)

    ax.set_aspect("equal")
    ax.grid(True, color="#cccccc", ls=":", alpha=0.5)
    ax.set_xlabel("x forward (m)"); ax.set_ylabel("y left (m)")
    if zoom:
        cx, cy = b3["x"], b3["y"]
        ax.set_xlim(cx - 6, cx + 6); ax.set_ylim(cy - 6, cy + 6)
        ax.set_title("BEV zoom (±6 m)", fontsize=10)
    else:
        ax.set_xlim(-5, 60); ax.set_ylim(-32, 32)
        ax.set_title("BEV (ego frame, 50m ring)", fontsize=10)


def render(r, buckets):
    cam = ac.best_camera_for_box(r["v3"]) or "cam_front"
    fig = plt.figure(figsize=(21, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0], wspace=0.18)
    ax_cam = fig.add_subplot(gs[0, 0])
    ax_bev = fig.add_subplot(gs[0, 1])
    ax_zoom = fig.add_subplot(gs[0, 2])
    render_camera(ax_cam, r, cam)
    render_bev(ax_bev, r, zoom=False)
    render_bev(ax_zoom, r, zoom=True)

    rel = "n/a" if r["rel_speed"] is None else f"{r['rel_speed']:.2f}"
    obj = "n/a" if r["obj_speed"] is None else f"{r['obj_speed']:.2f}"
    egs = "n/a" if r["ego_speed"] is None else f"{r['ego_speed']:.2f}"
    cats = ",".join(categorize(r)) or "-"
    title = (f"{r['scen']} frame={r['frame_id']} track={r['track_id']} "
             f"class={ac.CLASS_NAME.get(r['class_id'],'?')}  |  corr_dist={r['corr_dist']:.3f} m "
             f"(dx={r['corr_dx']:.3f}, dy={r['corr_dy']:.3f})  |  "
             f"rel_speed={rel} obj_speed={obj} ego_speed={egs} m/s  |  "
             f"obj_dt={r['obj_dt_ms']:.0f}ms ego_dt={r['ego_dt_ms']:.0f}ms  |  "
             f"buckets=[{','.join(sorted(buckets))}] cats=[{cats}]")
    fig.suptitle(title, fontsize=10.5, y=1.02)
    # legend
    handles = [
        mpatches.Patch(color=RED, label="v2 box"),
        mpatches.Patch(color=GREEN, label="v3 box"),
        mpatches.Patch(color=WHITE, label="white: v2->v3 disp"),
        mpatches.Patch(color=BLUE, label="blue: object vel"),
        mpatches.Patch(color=YELLOW, label="yellow: ego vel"),
    ]
    ax_bev.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.85)

    fname = (f"{r['scen']}_f{r['frame_id']}_t{r['track_id']}_cd{r['corr_dist']:.3f}.png")
    out = os.path.join(OUT, fname)
    fig.savefig(out, dpi=105, bbox_inches="tight")
    plt.close(fig)
    return fname, cam


def main():
    recs = collect()
    print(f"[v2v3] joined boxes total: {len(recs)}")
    picked, extra_notes = select(recs)
    print(f"[v2v3] selected unique boxes: {len(picked)}")
    for n in extra_notes:
        print("   note:", n)

    manifest = []
    for (scen, fr, tid), v in sorted(picked.items(),
                                     key=lambda kv: -kv[1]["rec"]["corr_dist"]):
        r = v["rec"]
        fname, cam = render(r, v["buckets"])
        manifest.append({
            "scen": scen, "frame_id": fr, "track_id": tid,
            "class": ac.CLASS_NAME.get(r["class_id"], "?"),
            "corr_dist": round(r["corr_dist"], 4),
            "rel_speed": r["rel_speed"], "obj_speed": r["obj_speed"],
            "ego_speed": r["ego_speed"],
            "obj_dt_ms": round(r["obj_dt_ms"], 1), "ego_dt_ms": round(r["ego_dt_ms"], 1),
            "buckets": sorted(v["buckets"]), "categories": categorize(r),
            "camera": cam, "file": fname, "stem": r["stem"],
        })
        print(f"[v2v3] {fname}  buckets={sorted(v['buckets'])} cats={categorize(r)} cam={cam}")
    with open(os.path.join(OUT, "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"n": len(manifest), "extra_notes": extra_notes, "items": manifest},
                  f, indent=2, ensure_ascii=False)
    print(f"\n[v2v3] wrote {len(manifest)} figures")


if __name__ == "__main__":
    main()
