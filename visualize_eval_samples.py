#!/usr/bin/env python3
"""Render exact streaming-evaluation predictions as camera and BEV images.

The AP evaluator stores predictions after temporal streaming inference, but the
dump uses integer frame IDs.  This script reconstructs the deterministic
validation sampler order, joins those predictions to the source images and GT,
and writes two files per sample:

  <scenario>__<stem>__camera.png  three-camera projected 3D boxes
  <scenario>__<stem>__bev.png     ego-frame GT/prediction BEV boxes

Camera/BEV legend:
  cyan = GT, red = predicted vehicle, orange = predicted pedestrian
"""

import argparse
import json
import os

import cv2
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from camera_configs import CAM_ORDER
from morai_dataset import MoraiTemporalDataset, StreamingGroupSampler
from visualize_camera_proj import (
    CAM_DISPLAY,
    DET_COLORS_BEV,
    DET_COLORS_BGR,
    DET_NAMES,
    DET_SHORT,
    draw_cuboid_on_image,
    draw_rotated_box_bev,
    load_raw_image,
)


# Five deterministic, held-out validation frames spanning mixed ranges/classes.
# They are not selected by AP correctness, so qualitative failures remain visible.
DEFAULT_STEMS = (
    "scen08/live_000174",
    "scen09/live_000219",
    "scen51/live_000096",
    "scen72/live_000124",
    "scen149/live_000180",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--tag", default="ep25_full")
    p.add_argument("--dataset-root", default="/workspace/dataset")
    p.add_argument("--out", default=None)
    p.add_argument("--stems", nargs="*", default=None,
                   help="scenario/stem 목록. 생략하면 대표 held-out 5프레임")
    p.add_argument("--score-mode", choices=("final", "soft", "raw"),
                   default="final",
                   help="final=raw*centerness (AP 공식 score)")
    p.add_argument("--score-thresh", type=float, default=0.15,
                   help="시각화 전용 threshold (AP 계산에는 미사용)")
    p.add_argument("--match-dist", type=float, default=2.0)
    return p.parse_args()


def prediction_score(raw, cns, mode):
    if mode == "raw":
        return raw
    if mode == "soft":
        return raw * np.sqrt(np.clip(cns, 1e-6, 1.0))
    return raw * cns


def center_z_to_bottom_z(boxes):
    """Convert current v3/model boxes to the legacy projection helper contract.

    MoraiTemporalDataset and the detector use box[2]=z_center, while
    visualize_camera_proj.draw_cuboid_on_image intentionally retains the older
    box[2]=z_bottom convention.  Convert only for camera projection; BEV does
    not consume z.
    """
    converted = np.asarray(boxes).copy()
    if converted.size:
        converted[:, 2] -= np.exp(converted[:, 5]) * 0.5
    return converted


def match_counts(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, dist):
    """Per-frame class-aware greedy match, ordered by displayed score."""
    tp = 0
    used = np.zeros(len(gt_boxes), dtype=bool)
    for i in np.argsort(-pred_scores, kind="stable"):
        candidates = np.flatnonzero((gt_labels == pred_labels[i]) & ~used)
        if not len(candidates):
            continue
        d = np.linalg.norm(gt_boxes[candidates, :2] - pred_boxes[i, :2], axis=1)
        j = int(np.argmin(d))
        if d[j] <= dist:
            used[candidates[j]] = True
            tp += 1
    return {"tp": tp, "fp": int(len(pred_boxes) - tp),
            "fn": int(len(gt_boxes) - tp)}


def save_camera(path, title, scen_dir, stem, pred_boxes, pred_labels,
                pred_scores, gt_boxes, gt_labels):
    fig, axes = plt.subplots(1, len(CAM_ORDER), figsize=(18, 4.6))
    fig.patch.set_facecolor("#181818")
    pred_boxes_bottom = center_z_to_bottom_z(pred_boxes)
    gt_boxes_bottom = center_z_to_bottom_z(gt_boxes)
    for ax, cam_name in zip(axes, CAM_ORDER):
        image = load_raw_image(scen_dir, stem, cam_name)
        for box, cls_id in zip(gt_boxes_bottom, gt_labels):
            image = draw_cuboid_on_image(
                image, box, cam_name, (255, 255, 0),
                label=f"GT {DET_SHORT.get(int(cls_id), '?')}", thickness=2,
            )
        for box, cls_id, score in zip(
            pred_boxes_bottom, pred_labels, pred_scores
        ):
            image = draw_cuboid_on_image(
                image, box, cam_name,
                DET_COLORS_BGR.get(int(cls_id), (0, 0, 255)),
                label=f"P {DET_SHORT.get(int(cls_id), '?')} {score:.2f}",
                thickness=2,
            )
        ax.imshow(image)
        ax.set_title(CAM_DISPLAY.get(cam_name, cam_name), color="white")
        ax.axis("off")
    fig.suptitle(title, color="white", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def save_bev(path, title, pred_boxes, pred_labels, pred_scores,
             gt_boxes, gt_labels):
    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor("#181818")
    ax.set_facecolor("#080808")
    ax.plot(0, 0, marker=">", color="white", markersize=13, zorder=5)
    for box, cls_id in zip(gt_boxes, gt_labels):
        draw_rotated_box_bev(
            ax, box, color="cyan", lw=2.0,
            label=f"GT {DET_SHORT.get(int(cls_id), '?')}", alpha=0.9,
        )
    for box, cls_id, score in zip(pred_boxes, pred_labels, pred_scores):
        draw_rotated_box_bev(
            ax, box, color=DET_COLORS_BEV.get(int(cls_id), "red"), lw=2.2,
            label=f"P {DET_SHORT.get(int(cls_id), '?')} {score:.2f}",
        )
    ax.set_xlim(-5, 55)
    ax.set_ylim(-35, 35)
    ax.set_aspect("equal")
    ax.grid(True, color="gray", linestyle=":", alpha=0.35)
    ax.set_xlabel("x forward (m)", color="white")
    ax.set_ylabel("y left (m)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("gray")
    ax.set_title(title, color="white", fontsize=11)
    ax.legend(handles=(
        mlines.Line2D([], [], color="cyan", label="GT"),
        mlines.Line2D([], [], color="red", label="Pred vehicle"),
        mlines.Line2D([], [], color="orange", label="Pred pedestrian"),
    ), loc="upper right", facecolor="#202020", labelcolor="white")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)
    cfg_path = os.path.join(run_dir, "run_config.json")
    pred_path = os.path.join(run_dir, "ap_eval", f"{args.tag}_preds.npz")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(cfg_path)
    if not os.path.isfile(pred_path):
        raise FileNotFoundError(pred_path)

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    dataset = MoraiTemporalDataset(
        dataset_root=args.dataset_root,
        split="val",
        val_scenarios=cfg["val_scenarios"],
        load_depth=False,
        gt_version=cfg.get("val_gt_version", "v3"),
    )
    sampler = StreamingGroupSampler(
        dataset, batch_size=int(cfg.get("batch_size", 4)), shuffle=False,
        seed=0, drop_uneven_tail=False,
    )
    eval_order = [idx for batch in sampler for idx in batch]

    dump = np.load(pred_path)
    n_frames = int(dump["frame"].max()) + 1
    if len(eval_order) != n_frames:
        raise RuntimeError(
            f"평가 프레임 순서 불일치: sampler={len(eval_order)}, dump={n_frames}"
        )
    frame_by_stem = {
        f"{dataset.items[idx]['scen_name']}/{dataset.items[idx]['stem']}": frame_no
        for frame_no, idx in enumerate(eval_order)
    }

    selected = tuple(args.stems) if args.stems else DEFAULT_STEMS
    unknown = [stem for stem in selected if stem not in frame_by_stem]
    if unknown:
        raise ValueError(f"held-out split에서 stem을 찾을 수 없습니다: {unknown}")

    out_dir = os.path.abspath(
        args.out or os.path.join(run_dir, "qualitative", args.tag)
    )
    os.makedirs(out_dir, exist_ok=True)
    score_all = prediction_score(dump["raw"], dump["cns"], args.score_mode)
    records = []

    for full_stem in selected:
        frame_no = frame_by_stem[full_stem]
        ds_idx = eval_order[frame_no]
        item = dataset.items[ds_idx]
        sample = dataset[ds_idx]
        gt_boxes = sample["dynamic_gt_boxes"].numpy()
        gt_labels = sample["dynamic_gt_labels"].numpy()
        gt_roi = np.linalg.norm(gt_boxes[:, :2], axis=1) <= 50.0
        gt_boxes, gt_labels = gt_boxes[gt_roi], gt_labels[gt_roi]

        pred_idx = np.flatnonzero(
            (dump["frame"] == frame_no) & (score_all >= args.score_thresh)
        )
        pred_boxes = dump["box"][pred_idx]
        pred_labels = dump["label"][pred_idx]
        pred_scores = score_all[pred_idx]
        order = np.argsort(-pred_scores, kind="stable")
        pred_boxes, pred_labels, pred_scores = (
            pred_boxes[order], pred_labels[order], pred_scores[order]
        )

        counts = match_counts(
            pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels,
            args.match_dist,
        )
        summary = (
            f"{full_stem} | GT={len(gt_boxes)} Pred={len(pred_boxes)} | "
            f"TP/FP/FN@{args.match_dist:g}m="
            f"{counts['tp']}/{counts['fp']}/{counts['fn']} | "
            f"{args.score_mode}>={args.score_thresh:g}"
        )
        safe = full_stem.replace("/", "__")
        camera_path = os.path.join(out_dir, f"{safe}__camera.png")
        bev_path = os.path.join(out_dir, f"{safe}__bev.png")
        save_camera(
            camera_path, summary, item["scen_dir"], item["stem"],
            pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels,
        )
        save_bev(
            bev_path, summary, pred_boxes, pred_labels, pred_scores,
            gt_boxes, gt_labels,
        )
        record = {
            "frame_no": frame_no,
            "stem": full_stem,
            "gt": int(len(gt_boxes)),
            "pred": int(len(pred_boxes)),
            **counts,
            "camera": camera_path,
            "bev": bev_path,
        }
        records.append(record)
        print(f"[saved] {summary}")

    manifest = {
        "source": pred_path,
        "split": "held-out validation (used as test; repository has no test split)",
        "score_mode": args.score_mode,
        "score_threshold": args.score_thresh,
        "match_distance_m": args.match_dist,
        "samples": records,
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[done] {len(records)} samples, {len(records) * 2} images -> {out_dir}")
    print(f"[manifest] {manifest_path}")


if __name__ == "__main__":
    main()
