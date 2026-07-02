#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v8 기준 시각화 스크립트
- decode_detections에 det_quality를 반드시 전달 (val 평가와 동일 기준으로 맞춤)
- 기본 모드 = softcalibrated (apply_quality=True, quality_power=0.5)
  -> train.py의 metric_modes['softcalibrated']와 동일 (= best_model.pth 선정 기준)
- NMS는 train.py의 bev_nms_axis_aligned가 decode_detections 내부에서 이미 적용됨
  (axis-aligned IoU 0.3 또는 center_dist < 1.5m 기준으로 억제)
"""

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

# train.py metric_modes와 동일하게 정의 (mode 이름으로 선택 가능하게)
QUALITY_MODES = {
    'calibrated':     dict(apply_quality=True,  quality_power=1.0),  # foreground * quality
    'softcalibrated': dict(apply_quality=True,  quality_power=0.5),  # foreground * sqrt(quality)  <- best_model.pth 선정 기준
    'raw':            dict(apply_quality=False, quality_power=1.0),  # foreground only
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
    ckpt = torch.load(weights_path, map_location=device)

    # full checkpoint(dict with 'model_state_dict')와 순수 state_dict 둘 다 지원
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)
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
def predict(sample, model, device, det_thresh, mode):
    images = sample['images'].unsqueeze(0).to(device)
    intrinsics = sample['intrinsics'].unsqueeze(0).to(device)
    extrinsics = sample['extrinsics'].unsqueeze(0).to(device)

    output = model(images, intrinsics, extrinsics)
    det_logits = output['det_cls']        # [B, N, num_classes]
    det_boxes  = output['det_box']        # [B, N, box_dim]
    det_quality = output.get('det_quality', None)  # [B, N, ...] or None

    quality_b = det_quality[0] if det_quality is not None else None
    mode_cfg = QUALITY_MODES[mode]

    boxes, labels, scores = decode_detections(
        det_logits[0],
        det_boxes[0],
        det_quality=quality_b,
        score_thresh=det_thresh,
        apply_quality=mode_cfg['apply_quality'],
        quality_power=mode_cfg['quality_power'],
        # nms_iou, pre_nms_topk은 train.py decode_detections 기본값 그대로 사용
    )
    return boxes.cpu().numpy(), labels.cpu().numpy(), scores.cpu().numpy()


def visualize_one(dataset, model, device, stem, out_dir, det_thresh, mode):
    os.makedirs(out_dir, exist_ok=True)
    idx = find_index_by_stem(dataset, stem)
    sample = dataset[idx]

    gt_boxes = sample['dynamic_gt_boxes'].cpu().numpy()
    gt_labels = sample['dynamic_gt_labels'].cpu().numpy()
    pred_boxes, pred_labels, pred_scores = predict(sample, model, device, det_thresh, mode)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('black')
    ax.plot(0, 0, marker='>', color='white', markersize=14)

    for box, cls_id in zip(gt_boxes, gt_labels):
        draw_rotated_box_bev(
            ax, box, color='cyan', lw=1.5,
            label=f"GT {DET_NAMES.get(int(cls_id), '?')}", alpha=0.8,
        )

    for box, cls_id, score in zip(pred_boxes, pred_labels, pred_scores):
        color = DET_COLORS.get(int(cls_id), 'red')
        draw_rotated_box_bev(
            ax, box, color=color, lw=2.0,
            label=f"{DET_NAMES.get(int(cls_id), '?')} {score:.2f}",
        )

    ax.set_xlim(-5, 65)
    ax.set_ylim(-35, 35)
    ax.set_aspect('equal')
    ax.grid(True, color='gray', linestyle=':', alpha=0.35)
    ax.set_xlabel('x forward (m)')
    ax.set_ylabel('y left (m)')
    ax.set_title(
        f"{sample['stem']} | mode={mode} thresh={det_thresh:.2f} | "
        f"GT={len(gt_boxes)} Pred={len(pred_boxes)}"
    )

    safe_stem = sample['stem'].replace('/', '__')
    out_path = os.path.join(out_dir, f"{safe_stem}_{mode}_t{det_thresh:.2f}_pred_bev.png")
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
    parser.add_argument('--det-thresh', type=float, default=0.15,
                         help='train.py의 best 선정 기준(softcalibrated f1@0.15)과 맞춘 기본값')
    parser.add_argument('--mode', default='softcalibrated',
                         choices=list(QUALITY_MODES.keys()),
                         help="quality 가중치 방식. 'softcalibrated'가 best_model.pth 선정 기준과 동일")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}")
    print(f"[mode] {args.mode} -> {QUALITY_MODES[args.mode]}")

    dataset = MoraiDataset(dataset_root=args.dataset_root, split=args.split)
    model = load_model(args.weights, device)
    print(f"[model] loaded: {args.weights}")

    if args.stem:
        stems = [args.stem]
    else:
        indices = random.sample(range(len(dataset)), min(args.n, len(dataset)))
        stems = []
        for idx in indices:
            scen_dir, item_stem = dataset.items[idx]
            stems.append(f"{os.path.basename(scen_dir)}/{item_stem}")

    print(f"[visualize] {len(stems)}개 샘플 시각화 시작")
    for stem in stems:
        try:
            visualize_one(dataset, model, device, stem, args.out, args.det_thresh, args.mode)
        except Exception as e:
            print(f"[ERROR] {stem}: {e}")

    print(f"[완료] 저장 위치: {args.out}/")


if __name__ == '__main__':
    main()