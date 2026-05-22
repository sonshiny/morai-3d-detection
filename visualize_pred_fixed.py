#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import torch

from morai_dataset import MoraiDataset
from train import AutoNavModel, decode_detections


DET_NAMES = {
    0: 'vehicle',
    1: 'pedestrian',
}

DET_COLORS = {
    0: 'red',
    1: 'orange',
}


def draw_rotated_box_bev(ax, box, color, lw=2.0, label=None, alpha=1.0):
    x, y = float(box[0]), float(box[1])
    w = float(np.exp(box[3]))
    l = float(np.exp(box[4]))
    angle = float(np.degrees(np.arctan2(box[6], box[7])))

    rect = patches.Rectangle(
        (x - l / 2, y - w / 2),
        l,
        w,
        angle=0,
        linewidth=lw,
        edgecolor=color,
        facecolor='none',
        alpha=alpha,
    )
    rect.set_transform(transforms.Affine2D().rotate_deg_around(x, y, angle) + ax.transData)
    ax.add_patch(rect)

    if label:
        ax.text(x, y, label, color=color, fontsize=8)


def load_model(weights_path, device):
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"[ERROR] weights not found: {weights_path}")

    model = AutoNavModel().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def find_index_by_stem(dataset, stem):
    for idx in range(len(dataset)):
        scen_dir, item_stem = dataset.items[idx]
        full_stem = f"{os.path.basename(scen_dir)}/{item_stem}"
        if stem in (item_stem, full_stem):
            return idx
    raise ValueError(f"stem not found in split: {stem}")


@torch.no_grad()
def predict(sample, model, device, det_thresh):
    images = sample['images'].unsqueeze(0).to(device)
    intrinsics = sample['intrinsics'].unsqueeze(0).to(device)
    extrinsics = sample['extrinsics'].unsqueeze(0).to(device)

    det_logits, det_boxes = model(images, intrinsics, extrinsics)
    boxes, labels, scores = decode_detections(
        det_logits[0],
        det_boxes[0],
        score_thresh=det_thresh,
    )
    return boxes.cpu().numpy(), labels.cpu().numpy(), scores.cpu().numpy()


def visualize_one(dataset, model, device, stem, out_dir, det_thresh):
    os.makedirs(out_dir, exist_ok=True)
    idx = find_index_by_stem(dataset, stem)
    sample = dataset[idx]

    gt_boxes = sample['dynamic_gt_boxes'].cpu().numpy()
    gt_labels = sample['dynamic_gt_labels'].cpu().numpy()
    pred_boxes, pred_labels, pred_scores = predict(sample, model, device, det_thresh)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('black')
    ax.plot(0, 0, marker='>', color='white', markersize=14)

    for box, cls_id in zip(gt_boxes, gt_labels):
        draw_rotated_box_bev(
            ax,
            box,
            color='cyan',
            lw=1.5,
            label=f"GT {DET_NAMES.get(int(cls_id), '?')}",
            alpha=0.8,
        )

    for box, cls_id, score in zip(pred_boxes, pred_labels, pred_scores):
        color = DET_COLORS.get(int(cls_id), 'red')
        draw_rotated_box_bev(
            ax,
            box,
            color=color,
            lw=2.0,
            label=f"{DET_NAMES.get(int(cls_id), '?')} {score:.2f}",
        )

    ax.set_xlim(-5, 65)
    ax.set_ylim(-35, 35)
    ax.set_aspect('equal')
    ax.grid(True, color='gray', linestyle=':', alpha=0.35)
    ax.set_xlabel('x forward (m)')
    ax.set_ylabel('y left (m)')
    ax.set_title(f"{sample['stem']} | GT={len(gt_boxes)} Pred={len(pred_boxes)}")

    safe_stem = sample['stem'].replace('/', '__')
    out_path = os.path.join(out_dir, f"{safe_stem}_pred_bev.png")
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"[visualize] saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', default='./dataset')
    parser.add_argument('--weights', default='best_model.pth')
    parser.add_argument('--split', default='val', choices=['train', 'val'])
    parser.add_argument('--stem', default=None)
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--out', default='./pred_vis')
    parser.add_argument('--det-thresh', type=float, default=0.25)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MoraiDataset(dataset_root=args.dataset_root, split=args.split)
    model = load_model(args.weights, device)

    if args.stem:
        stems = [args.stem]
    else:
        indices = random.sample(range(len(dataset)), min(args.n, len(dataset)))
        stems = []
        for idx in indices:
            scen_dir, item_stem = dataset.items[idx]
            stems.append(f"{os.path.basename(scen_dir)}/{item_stem}")

    for stem in stems:
        visualize_one(dataset, model, device, stem, args.out, args.det_thresh)


if __name__ == '__main__':
    main()
