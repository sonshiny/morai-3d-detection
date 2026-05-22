#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms

from morai_dataset import MoraiDataset
from train import AutoNavModel

DET_NAMES = {
    0: 'vehicle',
    1: 'pedestrian',
}

DET_COLORS = {
    0: 'red',
    1: 'orange',
}

MAP_COLORS = {
    0: 'yellow',   # lane_boundary
    1: 'cyan',     # crosswalk
    2: 'white',    # road_boundary
}

MAP_NAMES = {
    0: 'lane_boundary',
    1: 'crosswalk',
    2: 'road_boundary',
}

DET_THRESH = 0.20
MAP_THRESH = 0.20


def bev_nms(boxes, scores, iou_thresh=0.3):
    if len(boxes) == 0:
        return []

    order = np.argsort(scores)[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        cx_i = boxes[i, 0]
        cy_i = boxes[i, 1]
        w_i = np.exp(boxes[i, 3])
        l_i = np.exp(boxes[i, 4])

        rest = order[1:]
        cx_r = boxes[rest, 0]
        cy_r = boxes[rest, 1]
        w_r = np.exp(boxes[rest, 3])
        l_r = np.exp(boxes[rest, 4])

        inter_x = np.maximum(
            0,
            np.minimum(cx_i + w_i / 2, cx_r + w_r / 2) -
            np.maximum(cx_i - w_i / 2, cx_r - w_r / 2)
        )
        inter_y = np.maximum(
            0,
            np.minimum(cy_i + l_i / 2, cy_r + l_r / 2) -
            np.maximum(cy_i - l_i / 2, cy_r - l_r / 2)
        )

        inter = inter_x * inter_y
        iou = inter / (w_i * l_i + w_r * l_r - inter + 1e-6)
        order = rest[iou < iou_thresh]

    return keep


def draw_rotated_box_bev(ax, x, y, w, l, sin_yaw, cos_yaw, color, lw=2.0, label=None):
    angle = np.degrees(np.arctan2(sin_yaw, cos_yaw))
    rect = patches.Rectangle(
        (x - l / 2, y - w / 2),
        l, w,
        angle=0,
        linewidth=lw,
        edgecolor=color,
        facecolor='none'
    )
    t = transforms.Affine2D().rotate_deg_around(x, y, angle) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)

    if label is not None:
        ax.text(x, y, label, color=color, fontsize=8)


def find_group_by_stem(dataset, stem):
    for i, g in enumerate(dataset.groups):
        if g['label_stem'] == stem:
            return i, g
    raise ValueError(f"stem not found in split: {stem}")


def load_model(weights_path, device):
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"[ERROR] weights not found: {weights_path}")

    model = AutoNavModel().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def get_prediction(sample, model, device, det_thresh=DET_THRESH, map_thresh=MAP_THRESH):
    images = sample['images'].unsqueeze(0).to(device)
    intrinsics = sample['intrinsics'].unsqueeze(0).to(device)
    extrinsics = sample['extrinsics'].unsqueeze(0).to(device)

    with torch.no_grad():
        det_logits, det_boxes, map_logits, map_lines = model(images, intrinsics, extrinsics)

    det_probs = torch.softmax(det_logits, dim=-1).cpu().numpy()   # [900, 3]
    det_boxes = det_boxes.cpu().numpy()                           # [900, 11]
    map_probs = torch.softmax(map_logits, dim=-1).cpu().numpy()   # [150, 4]
    map_lines = map_lines.cpu().numpy()                           # [150, 20, 2]

    # multi-class: 0=vehicle, 1=pedestrian, 2=bg
    fg_scores  = det_probs[:, :2]                  # [900, 2]
    best_cls   = np.argmax(fg_scores, axis=1)      # [900]
    best_score = np.max(fg_scores, axis=1)         # [900]
    keep_mask  = best_score > det_thresh

    boxes_cand = det_boxes[keep_mask]
    scr_cand   = best_score[keep_mask]
    cls_cand   = best_cls[keep_mask]

    # 클래스별 NMS
    if len(boxes_cand) > 0:
        keep_global = []
        for c in (0, 1):
            mask = cls_cand == c
            if not mask.any():
                continue
            local_idx = bev_nms(boxes_cand[mask], scr_cand[mask], iou_thresh=0.3)
            original_idx = np.where(mask)[0]
            keep_global.extend(original_idx[i] for i in local_idx)
        keep_global = np.array(keep_global, dtype=np.int64)
        boxes_keep = boxes_cand[keep_global]
        scr_keep   = scr_cand[keep_global]
        cls_keep   = cls_cand[keep_global]
    else:
        boxes_keep = boxes_cand
        scr_keep = scr_cand
        cls_keep = cls_cand

    pred_map = []
    for i in range(map_lines.shape[0]):
        bg_prob = float(map_probs[i, 3])
        if bg_prob > 0.5:
            continue

        cls_id = int(np.argmax(map_probs[i, :3]))
        conf = float(np.max(map_probs[i, :3]))
        if conf < map_thresh:
            continue

        pred_map.append((cls_id, conf, map_lines[i]))

    return {
        'det_boxes': boxes_keep,
        'det_scores': scr_keep,
        'det_labels': cls_keep,
        'map_items': pred_map,
    }


def visualize_one_gt(dataset, stem, out_dir="gt_vis"):
    os.makedirs(out_dir, exist_ok=True)

    idx, _ = find_group_by_stem(dataset, stem)
    sample = dataset[idx]

    gt_boxes = sample['dynamic_gt_boxes'].cpu().numpy()
    gt_labels = sample['dynamic_gt_labels'].cpu().numpy()
    gt_lines = sample['static_gt_polylines'].cpu().numpy()
    gt_line_labels = sample['static_gt_labels'].cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('black')
    ax.plot(0, 0, marker='>', color='white', markersize=16)

    # GT static
    map_count = 0
    for i in range(len(gt_lines)):
        cls_id = int(gt_line_labels[i])
        color = MAP_COLORS.get(cls_id, 'yellow')
        line = gt_lines[i]
        ax.plot(line[:, 0], line[:, 1], color=color, linewidth=2.0, alpha=0.9)
        map_count += 1

    # GT dynamic
    det_count = 0
    for i in range(len(gt_boxes)):
        box = gt_boxes[i]
        cls_id = int(gt_labels[i])

        w = float(np.exp(box[3]))
        l = float(np.exp(box[4]))

        draw_rotated_box_bev(
            ax,
            box[0], box[1],
            w, l,
            box[6], box[7],
            color='cyan',
            lw=1.8,
            label=DET_NAMES.get(cls_id, 'obj')
        )
        det_count += 1

    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(
        f"GT BEV — {stem} | det={det_count}, map={map_count}",
        color='white',
        fontsize=14
    )
    ax.set_xlabel("X (forward, m)", color='white')
    ax.set_ylabel("Y (left, m)", color='white')
    ax.tick_params(colors='white')
    ax.grid(True, color='gray', linestyle=':', alpha=0.4)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{stem}_gt_bev.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"[saved] {out_path}")


def visualize_one_pred(dataset, stem, model, device, out_dir="pred_vis",
                       det_thresh=DET_THRESH, map_thresh=MAP_THRESH):
    os.makedirs(out_dir, exist_ok=True)

    idx, _ = find_group_by_stem(dataset, stem)
    sample = dataset[idx]

    gt_boxes = sample['dynamic_gt_boxes'].cpu().numpy()
    gt_labels = sample['dynamic_gt_labels'].cpu().numpy()
    gt_lines = sample['static_gt_polylines'].cpu().numpy()
    gt_line_labels = sample['static_gt_labels'].cpu().numpy()

    pred = get_prediction(sample, model, device, det_thresh=det_thresh, map_thresh=map_thresh)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('black')
    ax.plot(0, 0, marker='>', color='white', markersize=16)

    # GT static
    for i in range(len(gt_lines)):
        cls_id = int(gt_line_labels[i])
        line = gt_lines[i]
        ax.plot(line[:, 0], line[:, 1], color='lime', linewidth=1.2, alpha=0.5)

    # Pred static
    pred_map_count = 0
    for cls_id, conf, line in pred['map_items']:
        color = MAP_COLORS.get(cls_id, 'yellow')
        ax.plot(
            line[:, 0], line[:, 1],
            color=color,
            linestyle='--',
            linewidth=2.0,
            alpha=0.9
        )
        pred_map_count += 1

    # GT dynamic
    for i in range(len(gt_boxes)):
        box = gt_boxes[i]
        w = float(np.exp(box[3]))
        l = float(np.exp(box[4]))

        draw_rotated_box_bev(
            ax,
            box[0], box[1],
            w, l,
            box[6], box[7],
            color='cyan',
            lw=1.4,
            label=None
        )

    # Pred dynamic
    pred_det_count = 0
    for box, cls_id, score in zip(pred['det_boxes'], pred['det_labels'], pred['det_scores']):
        w = float(np.exp(box[3]))
        l = float(np.exp(box[4]))

        draw_rotated_box_bev(
            ax,
            box[0], box[1],
            w, l,
            box[6], box[7],
            color=DET_COLORS.get(int(cls_id), 'red'),
            lw=2.0,
            label=f"{DET_NAMES.get(int(cls_id), 'obj')}:{score:.2f}"
        )
        pred_det_count += 1

    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(
        f"PRED BEV — {stem} | pred_det={pred_det_count}, pred_map={pred_map_count}",
        color='white',
        fontsize=14
    )
    ax.set_xlabel("X (forward, m)", color='white')
    ax.set_ylabel("Y (left, m)", color='white')
    ax.tick_params(colors='white')
    ax.grid(True, color='gray', linestyle=':', alpha=0.4)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{stem}_pred_bev.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"[saved] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['gt', 'pred'], default='gt')
    parser.add_argument('--weights', default='best_model.pth')
    parser.add_argument('--split', default='train', choices=['train', 'val'])
    parser.add_argument('--stem', default=None)
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--out', default=None)
    parser.add_argument('--det_thresh', type=float, default=DET_THRESH)
    parser.add_argument('--map_thresh', type=float, default=MAP_THRESH)
    args = parser.parse_args()

    dataset = MoraiDataset(dataset_dir='./dataset', split=args.split)
    all_stems = [g['label_stem'] for g in dataset.groups]

    if args.stem is not None:
        stems = [args.stem]
    else:
        if args.random:
            stems = random.sample(all_stems, min(args.n, len(all_stems)))
        else:
            stems = all_stems[:args.n]

    if args.out is None:
        args.out = 'gt_vis' if args.mode == 'gt' else 'pred_vis'

    if args.mode == 'gt':
        for stem in stems:
            visualize_one_gt(dataset, stem, out_dir=args.out)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.weights, device)

    for stem in stems:
        visualize_one_pred(
            dataset=dataset,
            stem=stem,
            model=model,
            device=device,
            out_dir=args.out,
            det_thresh=args.det_thresh,
            map_thresh=args.map_thresh
        )


if __name__ == "__main__":
    main()