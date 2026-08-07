#!/usr/bin/env python3
"""Compare predicted 3D state with GT over consecutive frames.

This is a detector-state diagnostic, not a tracking evaluation.  Predictions
are independently matched to GT in every frame using class-aware Hungarian
matching in BEV.  GT track_id is used only to join the same GT object across
frames for plotting; predicted instance IDs are neither required nor scored.

The prediction NPZ must come from evaluate_ap.py, which performs chronological
streaming inference with the temporal memory enabled according to run_config.

Outputs
-------
  matches.csv             one row per GT per frame (misses are retained)
  summary.json            vehicle/pedestrian state errors and temporal errors
  timeline_summary.png    frame-level match/error curves for both classes
  tracks/*.png            GT vs prediction x/y/z/yaw/vx/vy/speed per GT object
  bev_comparison.mp4      GT/pred boxes, heading arrows and velocity arrows

Example
-------
  python analyze_temporal_state.py \
    --run-dir runs/p2_scratch_parity_20260801_30ep \
    --tag ep25_full --scenario scen08 --start-stem live_000150 \
    --num-frames 100
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402


CLASS_NAMES = {0: "vehicle", 1: "pedestrian"}
CLASS_SHORT = {0: "V", 1: "P"}
GT_COLOR = (255, 255, 0)       # BGR cyan
PRED_COLORS = {0: (40, 40, 255), 1: (0, 165, 255)}


def parse_args():
    p = argparse.ArgumentParser(
        description="연속 프레임에서 vehicle/pedestrian의 x/y/z/yaw/vx/vy를 GT와 비교")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--tag", required=True,
                   help="evaluate_ap.py 결과 tag (예: ep25_full)")
    p.add_argument("--dataset-root", default="/workspace/dataset")
    p.add_argument("--scenario", required=True, help="예: scen08")
    p.add_argument("--start-stem", default=None,
                   help="예: live_000150. 생략하면 선택 segment의 첫 프레임")
    p.add_argument("--segment", type=int, default=None,
                   help="segment index. 생략하면 start-stem이 속한 segment 또는 가장 긴 segment")
    p.add_argument("--num-frames", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=None,
                   help="NPZ 생성 당시 evaluate_ap batch size (기본 run_config batch_size)")
    p.add_argument("--raw-thresh", type=float, default=0.05,
                   help="배포 decode와 같은 raw classification prefilter")
    p.add_argument("--match-dist", type=float, default=2.0,
                   help="클래스별 GT-pred 최대 BEV 중심거리(m)")
    p.add_argument("--no-nms", action="store_true",
                   help="rotated NMS를 생략하고 pre-NMS 예측을 매칭")
    p.add_argument("--nms-iou", type=float, default=0.3)
    p.add_argument("--max-tracks-per-class", type=int, default=4)
    p.add_argument("--fps", type=float, default=0.0,
                   help="BEV 영상 FPS. 0이면 timestamp에서 자동 결정")
    p.add_argument("--out", default=None)
    return p.parse_args()


def wrap_angle(x):
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


def yaw_of(box):
    return float(math.atan2(float(box[6]), float(box[7])))


def score_of(raw, cns):
    return raw * cns


def ego_to_global(box, T):
    """Return global center, global yaw and global planar velocity."""
    center_h = np.array([box[0], box[1], box[2], 1.0], dtype=np.float64)
    center = (T @ center_h)[:3]
    ego_yaw = math.atan2(float(T[1, 0]), float(T[0, 0]))
    yaw = float(wrap_angle(yaw_of(box) + ego_yaw))
    vel3 = T[:3, :3] @ np.array([box[8], box[9], 0.0], dtype=np.float64)
    return center, yaw, vel3[:2]


def frame_slice(sorted_frames, frame_no):
    lo = int(np.searchsorted(sorted_frames, frame_no, side="left"))
    hi = int(np.searchsorted(sorted_frames, frame_no, side="right"))
    return slice(lo, hi)


def per_frame_match(pred_boxes, pred_labels, pred_scores,
                    gt_boxes, gt_labels, max_dist):
    """Class-aware one-to-one BEV matching; returns gt_idx -> pred_idx."""
    matches = {}
    for cls_id in CLASS_NAMES:
        pi = np.flatnonzero(pred_labels == cls_id)
        gi = np.flatnonzero(gt_labels == cls_id)
        if not len(pi) or not len(gi):
            continue
        dist = np.linalg.norm(
            pred_boxes[pi, None, :2] - gt_boxes[None, gi, :2], axis=-1)
        # Tiny score tie-break: geometry is primary; higher score wins equal distances.
        cost = dist - 1e-5 * pred_scores[pi, None]
        # Hungarian is otherwise forced to assign out-of-gate pairs, which can
        # steal a prediction from a valid <=max_dist pair in a crowded frame.
        cost[dist > max_dist] = 1e6
        pr, gr = linear_sum_assignment(cost)
        for a, b in zip(pr, gr):
            if dist[a, b] <= max_dist:
                matches[int(gi[b])] = int(pi[a])
    return matches


def _mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else None


def _p90(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, 90)) if values.size else None


def summarize(rows):
    result = {}
    for cls_id, cls_name in CLASS_NAMES.items():
        cr = [r for r in rows if r["class_id"] == cls_id]
        mr = [r for r in cr if r["matched"]]
        vr = [r for r in mr if r["velocity_valid"]]
        summary = {
            "gt_samples": len(cr),
            "matched_samples": len(mr),
            "match_rate": len(mr) / max(len(cr), 1),
            "position_mae_ego_m": {
                axis: _mean([abs(r[f"err_{axis}_ego"]) for r in mr])
                for axis in ("x", "y", "z")
            },
            "center_error_xy_m": {
                "mean": _mean([r["center_error_xy"] for r in mr]),
                "p90": _p90([r["center_error_xy"] for r in mr]),
            },
            "center_error_3d_m": {
                "mean": _mean([r["center_error_3d"] for r in mr]),
                "p90": _p90([r["center_error_3d"] for r in mr]),
            },
            "yaw_error_deg": {
                "mean": _mean([abs(r["yaw_error_rad"]) * 180.0 / np.pi for r in mr]),
                "p90": _p90([abs(r["yaw_error_rad"]) * 180.0 / np.pi for r in mr]),
            },
            "velocity_valid_matched": len(vr),
            "velocity_error_mps": {
                "vx_mae": _mean([abs(r["err_vx_ego"]) for r in vr]),
                "vy_mae": _mean([abs(r["err_vy_ego"]) for r in vr]),
                "vector_mean": _mean([r["velocity_error"] for r in vr]),
                "vector_p90": _p90([r["velocity_error"] for r in vr]),
                "speed_mae": _mean([abs(r["pred_speed"] - r["gt_speed"]) for r in vr]),
            },
        }

        # Consecutive matched samples of one GT object.  These compare changes,
        # not absolute values, so they reveal frame-to-frame jitter or lag.
        step_pos, step_yaw, step_vel = [], [], []
        by_obj = defaultdict(list)
        for r in mr:
            by_obj[r["object_key"]].append(r)
        for obj_rows in by_obj.values():
            obj_rows.sort(key=lambda r: r["frame_offset"])
            for a, b in zip(obj_rows, obj_rows[1:]):
                if b["frame_offset"] != a["frame_offset"] + 1:
                    continue
                if not (0.0 < b["timestamp"] - a["timestamp"] <= 0.5):
                    continue
                dp = ((b["pred_global"] - a["pred_global"])
                      - (b["gt_global"] - a["gt_global"]))
                step_pos.append(float(np.linalg.norm(dp)))
                pred_dyaw = wrap_angle(b["pred_yaw_global"] - a["pred_yaw_global"])
                gt_dyaw = wrap_angle(b["gt_yaw_global"] - a["gt_yaw_global"])
                step_yaw.append(abs(float(wrap_angle(pred_dyaw - gt_dyaw))))
                if a["velocity_valid"] and b["velocity_valid"]:
                    dv = ((b["pred_vel_global"] - a["pred_vel_global"])
                          - (b["gt_vel_global"] - a["gt_vel_global"]))
                    step_vel.append(float(np.linalg.norm(dv)))
        summary["temporal_delta_error"] = {
            "consecutive_pairs": len(step_pos),
            "center_step_mean_m": _mean(step_pos),
            "center_step_p90_m": _p90(step_pos),
            "yaw_step_mean_deg": (
                None if not step_yaw else _mean(step_yaw) * 180.0 / np.pi),
            "velocity_step_mean_mps": _mean(step_vel),
        }
        result[cls_name] = summary
    return result


def csv_value(v):
    if isinstance(v, np.ndarray):
        return json.dumps(v.tolist())
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def save_csv(path, rows):
    fields = [
        "scenario", "segment", "stem", "frame_id", "frame_offset", "timestamp",
        "class_id", "class_name", "object_key", "gt_track_id", "matched",
        "pred_score", "velocity_valid",
        "gt_x_ego", "pred_x_ego", "err_x_ego",
        "gt_y_ego", "pred_y_ego", "err_y_ego",
        "gt_z_ego", "pred_z_ego", "err_z_ego",
        "gt_yaw_ego_rad", "pred_yaw_ego_rad", "yaw_error_rad",
        "gt_vx_ego", "pred_vx_ego", "err_vx_ego",
        "gt_vy_ego", "pred_vy_ego", "err_vy_ego",
        "gt_speed", "pred_speed", "velocity_error",
        "center_error_xy", "center_error_3d",
        "gt_x_global", "pred_x_global", "gt_y_global", "pred_y_global",
        "gt_z_global", "pred_z_global", "gt_yaw_global", "pred_yaw_global",
        "gt_vx_global", "pred_vx_global", "gt_vy_global", "pred_vy_global",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {k: csv_value(row.get(k, "")) for k in fields}
            writer.writerow(out)


def _plot_pair(ax, t, gt, pred, title, ylabel):
    ax.plot(t, gt, color="#00bcd4", lw=2.0, label="GT")
    ax.plot(t, pred, color="#ff5252", lw=1.6, ls="--", label="Pred")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def save_track_plots(out_dir, rows, max_per_class):
    os.makedirs(out_dir, exist_ok=True)
    by_obj = defaultdict(list)
    for row in rows:
        by_obj[row["object_key"]].append(row)

    for cls_id, cls_name in CLASS_NAMES.items():
        candidates = []
        for key, rr in by_obj.items():
            if rr[0]["class_id"] != cls_id:
                continue
            n_match = sum(r["matched"] for r in rr)
            candidates.append((n_match, len(rr), key, rr))
        candidates.sort(reverse=True, key=lambda x: (x[0], x[1]))
        for rank, (_, _, key, rr) in enumerate(candidates[:max_per_class], 1):
            rr.sort(key=lambda r: r["timestamp"])
            t0 = rr[0]["timestamp"]
            t = np.array([r["timestamp"] - t0 for r in rr])

            def arr(name):
                return np.array([r[name] if r["matched"] else np.nan for r in rr], dtype=float)

            gt_yaw = np.unwrap(np.array([r["gt_yaw_global"] for r in rr])) * 180 / np.pi
            pred_yaw_raw = arr("pred_yaw_global")
            pred_yaw = np.full_like(pred_yaw_raw, np.nan)
            valid = np.isfinite(pred_yaw_raw)
            if valid.any():
                # Unwrap contiguous visible runs without bridging missed frames.
                starts = np.flatnonzero(valid & ~np.r_[False, valid[:-1]])
                ends = np.flatnonzero(valid & ~np.r_[valid[1:], False]) + 1
                for s, e in zip(starts, ends):
                    pred_yaw[s:e] = np.unwrap(pred_yaw_raw[s:e]) * 180 / np.pi

            fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
            _plot_pair(axes[0, 0], t, [r["gt_global"][0] for r in rr], arr("pred_x_global"), "Global x", "m")
            _plot_pair(axes[0, 1], t, [r["gt_global"][1] for r in rr], arr("pred_y_global"), "Global y", "m")
            _plot_pair(axes[1, 0], t, [r["gt_global"][2] for r in rr], arr("pred_z_global"), "Global z", "m")
            _plot_pair(axes[1, 1], t, gt_yaw, pred_yaw, "Global yaw", "degree")
            _plot_pair(axes[2, 0], t, [r["gt_vel_global"][0] for r in rr], arr("pred_vx_global"), "Global vx", "m/s")
            _plot_pair(axes[2, 1], t, [r["gt_vel_global"][1] for r in rr], arr("pred_vy_global"), "Global vy", "m/s")
            _plot_pair(axes[3, 0], t, [r["gt_speed"] for r in rr], arr("pred_speed"), "Speed", "m/s")
            axes[3, 1].plot(t, arr("center_error_3d"), color="#ffb300")
            axes[3, 1].set_title("3D center error (miss = gap)")
            axes[3, 1].set_ylabel("m")
            axes[3, 1].grid(alpha=0.25)
            for ax in axes[-1]:
                ax.set_xlabel("time (s)")
            axes[0, 0].legend(loc="best")
            matched = sum(r["matched"] for r in rr)
            fig.suptitle(
                f"{cls_name} GT object | matched {matched}/{len(rr)} | {key}",
                fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            safe = key.replace("/", "_").replace(":", "_").replace("#", "_")
            fig.savefig(os.path.join(out_dir, f"{cls_name}_{rank:02d}_{safe}.png"), dpi=140)
            plt.close(fig)


def save_timeline(path, rows, frame_meta):
    frames = [m["frame_offset"] for m in frame_meta]
    fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
    for cls_id, cls_name in CLASS_NAMES.items():
        color = "#1565c0" if cls_id == 0 else "#ef6c00"
        match_rate, center, yaw, speed = [], [], [], []
        for off in frames:
            fr = [r for r in rows if r["frame_offset"] == off and r["class_id"] == cls_id]
            mr = [r for r in fr if r["matched"]]
            vr = [r for r in mr if r["velocity_valid"]]
            match_rate.append(len(mr) / max(len(fr), 1) if fr else np.nan)
            center.append(_mean([r["center_error_3d"] for r in mr]))
            yaw.append(_mean([abs(r["yaw_error_rad"]) * 180 / np.pi for r in mr]))
            speed.append(_mean([abs(r["pred_speed"] - r["gt_speed"]) for r in vr]))
        for ax, vals, title in zip(
                axes, (match_rate, center, yaw, speed),
                ("GT match rate", "3D center error", "Yaw error", "Speed error")):
            ax.plot(frames, vals, label=cls_name, color=color, lw=1.6)
            ax.set_title(title)
            ax.grid(alpha=0.25)
    axes[0].set_ylabel("ratio")
    axes[1].set_ylabel("m")
    axes[2].set_ylabel("degree")
    axes[3].set_ylabel("m/s")
    axes[3].set_xlabel("scenario frame offset")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bev_point(x, y, width, height, xlim=(-10.0, 55.0), ylim=(-35.0, 35.0)):
    u = int((y - ylim[0]) / (ylim[1] - ylim[0]) * (width - 1))
    v = int((xlim[1] - x) / (xlim[1] - xlim[0]) * (height - 1))
    return u, v


def box_corners(box):
    x, y = float(box[0]), float(box[1])
    w, l = math.exp(float(box[3])), math.exp(float(box[4]))
    yaw = yaw_of(box)
    local = np.array([[l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2]])
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return local @ R.T + np.array([x, y])


def draw_state(canvas, box, color, label, solid=True):
    h, w = canvas.shape[:2]
    pts = np.array([bev_point(x, y, w, h) for x, y in box_corners(box)], np.int32)
    cv2.polylines(canvas, [pts], True, color, 2 if solid else 1, cv2.LINE_AA)
    center = bev_point(float(box[0]), float(box[1]), w, h)
    yaw = yaw_of(box)
    head = bev_point(float(box[0]) + 3.0 * math.cos(yaw),
                     float(box[1]) + 3.0 * math.sin(yaw), w, h)
    cv2.arrowedLine(canvas, center, head, color, 2, cv2.LINE_AA, tipLength=0.22)
    speed_end = bev_point(float(box[0]) + float(box[8]),
                          float(box[1]) + float(box[9]), w, h)
    cv2.arrowedLine(canvas, center, speed_end, (80, 255, 80), 1,
                    cv2.LINE_AA, tipLength=0.20)
    cv2.putText(canvas, label, (center[0] + 4, center[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


def save_bev_video(path, records_by_offset, frame_meta, fps):
    width, height = 1000, 900
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter를 열 수 없습니다: {path}")
    for meta in frame_meta:
        canvas = np.full((height, width, 3), 18, dtype=np.uint8)
        # 10 m grid
        for x in range(-10, 56, 10):
            a = bev_point(x, -35, width, height)
            b = bev_point(x, 35, width, height)
            cv2.line(canvas, a, b, (48, 48, 48), 1)
        for y in range(-30, 31, 10):
            a = bev_point(-10, y, width, height)
            b = bev_point(55, y, width, height)
            cv2.line(canvas, a, b, (48, 48, 48), 1)
        ego = bev_point(0, 0, width, height)
        cv2.arrowedLine(canvas, ego, bev_point(4, 0, width, height),
                        (255, 255, 255), 2, tipLength=0.2)

        rr = records_by_offset.get(meta["frame_offset"], [])
        for r in rr:
            gt_label = f"GT {CLASS_SHORT[r['class_id']]}"
            draw_state(canvas, r["gt_box"], GT_COLOR, gt_label)
            if r["matched"]:
                color = PRED_COLORS[r["class_id"]]
                pred_label = f"P {CLASS_SHORT[r['class_id']]} {r['pred_score']:.2f}"
                draw_state(canvas, r["pred_box"], color, pred_label)
                a = bev_point(r["gt_box"][0], r["gt_box"][1], width, height)
                b = bev_point(r["pred_box"][0], r["pred_box"][1], width, height)
                cv2.line(canvas, a, b, (120, 120, 120), 1, cv2.LINE_AA)

        matched = sum(r["matched"] for r in rr)
        title = (f"{meta['scenario']}/{meta['stem']}  frame={meta['frame_id']}  "
                 f"GT={len(rr)} matched={matched}")
        cv2.putText(canvas, title, (18, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(canvas, "cyan=GT  red=pred vehicle  orange=pred pedestrian",
                    (18, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, "long arrow=heading  green arrow=velocity (1m per 1m/s)",
                    (18, 79), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (180, 255, 180), 1, cv2.LINE_AA)
        writer.write(canvas)
    writer.release()


def main():
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)
    with open(os.path.join(run_dir, "run_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    anchor_dir = cfg["anchor_dir"]
    os.environ.setdefault("ANCHOR_XY_FILE", os.path.join(anchor_dir, "anchor_kmeans_xy.npy"))
    os.environ.setdefault("ANCHOR_FULL_FILE", os.path.join(anchor_dir, "anchor_kmeans_full.npy"))
    os.environ.setdefault("FILTER_VISIBLE", "0")

    # Delayed imports: train.py reads anchor env during import.
    from morai_dataset import MoraiTemporalDataset, StreamingGroupSampler
    from train import bev_nms_rotated

    pred_path = os.path.join(run_dir, "ap_eval", f"{args.tag}_preds.npz")
    if not os.path.isfile(pred_path):
        raise FileNotFoundError(
            f"{pred_path}\n먼저 evaluate_ap.py로 full streaming 평가를 실행해야 합니다.")

    dataset = MoraiTemporalDataset(
        dataset_root=args.dataset_root, split="val",
        val_scenarios=cfg["val_scenarios"], load_depth=False,
        gt_version=cfg.get("val_gt_version", "v3"),
    )
    batch_size = args.batch_size or int(cfg.get("batch_size", 4))
    sampler = StreamingGroupSampler(
        dataset, batch_size=batch_size, shuffle=False, seed=0,
        drop_uneven_tail=False)
    eval_order = [idx for batch in sampler for idx in batch]
    eval_frame_for_idx = {idx: frame for frame, idx in enumerate(eval_order)}

    scenario_items = [
        (idx, item) for idx, item in enumerate(dataset.items)
        if item["scen_name"] == args.scenario
    ]
    if not scenario_items:
        raise ValueError(
            f"val split에 {args.scenario}가 없습니다. 가능한 값: {cfg['val_scenarios']}")

    if args.start_stem:
        found = [(idx, it) for idx, it in scenario_items if it["stem"] == args.start_stem]
        if not found:
            raise ValueError(f"{args.scenario}/{args.start_stem}를 찾을 수 없습니다.")
        chosen_seg = found[0][1]["seg_idx"]
    elif args.segment is not None:
        chosen_seg = args.segment
    else:
        counts = defaultdict(int)
        for _, it in scenario_items:
            counts[it["seg_idx"]] += 1
        chosen_seg = max(counts, key=counts.get)

    if args.segment is not None and chosen_seg != args.segment:
        raise ValueError(
            f"start-stem은 segment {chosen_seg}이지만 --segment={args.segment}입니다.")
    segment_items = [(idx, it) for idx, it in scenario_items if it["seg_idx"] == chosen_seg]
    segment_items.sort(key=lambda pair: pair[1]["frame_offset"])
    start_pos = 0
    if args.start_stem:
        start_pos = next(i for i, (_, it) in enumerate(segment_items)
                         if it["stem"] == args.start_stem)
    selected = segment_items[start_pos:start_pos + args.num_frames]
    if not selected:
        raise RuntimeError("선택된 프레임이 없습니다.")

    # npz entries are compressed zip members.  Accessing dump["box"] inside
    # every frame would decompress the full ~300 MB member repeatedly, so load
    # each prediction array exactly once (roughly 400 MB RAM for a full val dump).
    with np.load(pred_path) as dump:
        pred_frames = dump["frame"]
        all_pred_boxes = dump["box"]
        all_pred_labels = dump["label"]
        all_pred_raw = dump["raw"]
        all_pred_cns = dump["cns"]
    if len(eval_order) != int(pred_frames.max()) + 1:
        raise RuntimeError(
            f"평가 순서 불일치: sampler={len(eval_order)}, dump_frames={int(pred_frames.max()) + 1}. "
            "NPZ 평가 때와 같은 --batch-size를 지정하세요.")
    if np.any(pred_frames[1:] < pred_frames[:-1]):
        raise RuntimeError("NPZ frame 배열이 정렬되어 있지 않습니다.")

    rows = []
    records_by_offset = defaultdict(list)
    frame_meta = []
    for ds_idx, item in selected:
        eval_frame = eval_frame_for_idx[ds_idx]
        sl = frame_slice(pred_frames, eval_frame)
        pred_boxes = all_pred_boxes[sl].astype(np.float64)
        pred_labels = all_pred_labels[sl].astype(np.int64)
        pred_raw = all_pred_raw[sl].astype(np.float64)
        pred_cns = all_pred_cns[sl].astype(np.float64)
        keep = pred_raw >= args.raw_thresh
        pred_boxes, pred_labels = pred_boxes[keep], pred_labels[keep]
        pred_scores = score_of(pred_raw[keep], pred_cns[keep])
        if not args.no_nms and len(pred_boxes):
            nms_keep = bev_nms_rotated(
                torch.from_numpy(pred_boxes).float(),
                torch.from_numpy(pred_scores).float(),
                torch.from_numpy(pred_labels).long(),
                iou_thresh=args.nms_iou,
            ).cpu().numpy()
            pred_boxes, pred_labels, pred_scores = (
                pred_boxes[nms_keep], pred_labels[nms_keep], pred_scores[nms_keep])

        gt_boxes_t, gt_labels_t, gt_ids_t, vel_valid_t, _, _ = dataset._load_labels_v2(
            item["scen_dir"], item["stem"])
        gt_boxes = gt_boxes_t.numpy().astype(np.float64)
        gt_labels = gt_labels_t.numpy().astype(np.int64)
        gt_ids = gt_ids_t.numpy().astype(np.int64)
        vel_valid = vel_valid_t.numpy() > 0.5
        info = dataset.scene_infos[args.scenario][item["stem"]]
        T = np.asarray(info["T_ego2global"], dtype=np.float64)
        timestamp = float(info["timestamp"])
        frame_id = int(info["frame_id"])
        matches = per_frame_match(
            pred_boxes, pred_labels, pred_scores,
            gt_boxes, gt_labels, args.match_dist)

        meta = {"scenario": args.scenario, "stem": item["stem"],
                "frame_id": frame_id, "frame_offset": item["frame_offset"],
                "timestamp": timestamp}
        frame_meta.append(meta)
        for gi, gt_box in enumerate(gt_boxes):
            cls_id = int(gt_labels[gi])
            gt_global, gt_yaw_global, gt_vel_global = ego_to_global(gt_box, T)
            pi = matches.get(gi)
            matched = pi is not None
            pred_box = pred_boxes[pi] if matched else np.full(11, np.nan)
            pred_global, pred_yaw_global, pred_vel_global = (
                ego_to_global(pred_box, T) if matched
                else (np.full(3, np.nan), np.nan, np.full(2, np.nan)))
            gt_yaw = yaw_of(gt_box)
            pred_yaw = yaw_of(pred_box) if matched else np.nan
            gt_speed = float(np.linalg.norm(gt_box[8:10]))
            pred_speed = float(np.linalg.norm(pred_box[8:10])) if matched else np.nan
            row = {
                **meta,
                "segment": chosen_seg,
                "class_id": cls_id,
                "class_name": CLASS_NAMES.get(cls_id, str(cls_id)),
                "object_key": f"{args.scenario}#seg{chosen_seg}:{cls_id}:{int(gt_ids[gi])}",
                "gt_track_id": int(gt_ids[gi]),
                "matched": bool(matched),
                "pred_score": float(pred_scores[pi]) if matched else np.nan,
                "velocity_valid": bool(vel_valid[gi]),
                "gt_box": gt_box,
                "pred_box": pred_box,
                "gt_global": gt_global,
                "pred_global": pred_global,
                "gt_yaw_global": gt_yaw_global,
                "pred_yaw_global": pred_yaw_global,
                "gt_vel_global": gt_vel_global,
                "pred_vel_global": pred_vel_global,
                "gt_x_ego": gt_box[0], "pred_x_ego": pred_box[0],
                "gt_y_ego": gt_box[1], "pred_y_ego": pred_box[1],
                "gt_z_ego": gt_box[2], "pred_z_ego": pred_box[2],
                "err_x_ego": pred_box[0] - gt_box[0],
                "err_y_ego": pred_box[1] - gt_box[1],
                "err_z_ego": pred_box[2] - gt_box[2],
                "gt_yaw_ego_rad": gt_yaw, "pred_yaw_ego_rad": pred_yaw,
                "yaw_error_rad": float(wrap_angle(pred_yaw - gt_yaw)) if matched else np.nan,
                "gt_vx_ego": gt_box[8], "pred_vx_ego": pred_box[8],
                "gt_vy_ego": gt_box[9], "pred_vy_ego": pred_box[9],
                "err_vx_ego": pred_box[8] - gt_box[8],
                "err_vy_ego": pred_box[9] - gt_box[9],
                "gt_speed": gt_speed, "pred_speed": pred_speed,
                "velocity_error": (float(np.linalg.norm(pred_box[8:10] - gt_box[8:10]))
                                   if matched else np.nan),
                "center_error_xy": (float(np.linalg.norm(pred_box[:2] - gt_box[:2]))
                                    if matched else np.nan),
                "center_error_3d": (float(np.linalg.norm(pred_box[:3] - gt_box[:3]))
                                    if matched else np.nan),
                "gt_x_global": gt_global[0], "pred_x_global": pred_global[0],
                "gt_y_global": gt_global[1], "pred_y_global": pred_global[1],
                "gt_z_global": gt_global[2], "pred_z_global": pred_global[2],
                "gt_vx_global": gt_vel_global[0], "pred_vx_global": pred_vel_global[0],
                "gt_vy_global": gt_vel_global[1], "pred_vy_global": pred_vel_global[1],
            }
            rows.append(row)
            records_by_offset[item["frame_offset"]].append(row)

    out_dir = os.path.abspath(args.out or os.path.join(
        run_dir, "temporal_state", f"{args.tag}_{args.scenario}_{selected[0][1]['stem']}"))
    os.makedirs(out_dir, exist_ok=True)
    save_csv(os.path.join(out_dir, "matches.csv"), rows)
    summary = {
        "meta": {
            "source_predictions": pred_path,
            "scenario": args.scenario,
            "segment": chosen_seg,
            "start_stem": selected[0][1]["stem"],
            "end_stem": selected[-1][1]["stem"],
            "frames": len(selected),
            "raw_threshold": args.raw_thresh,
            "rotated_nms": not args.no_nms,
            "match_distance_m": args.match_dist,
            "note": "GT track_id only joins GT rows; prediction tracking is not evaluated.",
            "vz_note": "predicted vz is omitted because current loss supervises only channels 0:10.",
        },
        "classes": summarize(rows),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    save_timeline(os.path.join(out_dir, "timeline_summary.png"), rows, frame_meta)
    save_track_plots(
        os.path.join(out_dir, "tracks"), rows, args.max_tracks_per_class)

    timestamps = np.array([m["timestamp"] for m in frame_meta], dtype=float)
    positive_dt = np.diff(timestamps)
    positive_dt = positive_dt[positive_dt > 0]
    fps = args.fps or (1.0 / np.median(positive_dt) if positive_dt.size else 10.0)
    fps = float(np.clip(fps, 1.0, 30.0))
    save_bev_video(
        os.path.join(out_dir, "bev_comparison.mp4"),
        records_by_offset, frame_meta, fps)

    print(f"[saved] {out_dir}")
    for cls_name, s in summary["classes"].items():
        print(
            f"  {cls_name:10s} match={s['matched_samples']}/{s['gt_samples']} "
            f"({s['match_rate']:.1%}) | center3d={s['center_error_3d_m']['mean']}m "
            f"| yaw={s['yaw_error_deg']['mean']}deg "
            f"| vel={s['velocity_error_mps']['vector_mean']}m/s")


if __name__ == "__main__":
    main()
