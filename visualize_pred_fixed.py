#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms

from morai_dataset import MoraiDataset

DET_NAMES = {
    0: 'car',
    1: 'truck',
    2: 'bus',
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
    raise ValueError(f"stem not found in train split: {stem}")


def visualize_one_gt(dataset, stem, out_dir="gt_vis"):
    os.makedirs(out_dir, exist_ok=True)

    idx, group = find_group_by_stem(dataset, stem)
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


def main():
    dataset = MoraiDataset(dataset_dir='./dataset', split='train')

    # 앞쪽 몇 개만 먼저 확인
    stems = [g['label_stem'] for g in dataset.groups[:10]]

    # 랜덤으로 보고 싶으면 아래 한 줄로 바꿔도 됨
    # stems = random.sample([g['label_stem'] for g in dataset.groups], 10)

    for stem in stems:
        visualize_one_gt(dataset, stem, out_dir="gt_vis")


if __name__ == "__main__":
    main()