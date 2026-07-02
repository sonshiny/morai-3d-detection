#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
카메라 투영 + BEV 통합 시각화

상단: 3개 카메라 이미지에 예측 박스 (cuboid) 투영  — GT 없음
하단: BEV 맵  —  GT (cyan) + 예측 (red/orange)

사용:
  python visualize_camera_proj.py [--weights best_model.pth] [--n 5]
  python visualize_camera_proj.py --stem scen01/live_000042
"""

import argparse
import os
import random
import traceback

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

from camera_configs import EXTRINSICS as _EXTRINSICS, INTRINSICS as _INTRINSICS, CAM_ORDER
from morai_dataset import ORIG_IMG_HEIGHT, ORIG_IMG_WIDTH, MoraiDataset
from train import AutoNavModel, decode_detections


# ── 상수 ──────────────────────────────────────────────────────────────────
DET_NAMES      = {0: 'vehicle',  1: 'pedestrian'}
DET_SHORT      = {0: 'veh',      1: 'ped'}
DET_COLORS_BGR = {0: (0, 0, 255), 1: (0, 165, 255)}   # OpenCV BGR: red, orange
DET_COLORS_BEV = {0: 'red',       1: 'orange'}

QUALITY_MODES = {
    'calibrated':     dict(apply_quality=True,  quality_power=1.0),
    'softcalibrated': dict(apply_quality=True,  quality_power=0.5),
    'raw':            dict(apply_quality=False, quality_power=1.0),
}

# 큐보이드 12 엣지 (top 4 + bottom 4 + vertical 4)
BOX_EDGES = [
    (0, 2), (2, 6), (6, 4), (4, 0),
    (1, 3), (3, 7), (7, 5), (5, 1),
    (0, 1), (2, 3), (4, 5), (6, 7),
]

CAM_DISPLAY = {
    'cam_front_left':  'Front-Left',
    'cam_front':       'Front-Center',
    'cam_front_right': 'Front-Right',
}


# ── 이미지 로드 ────────────────────────────────────────────────────────────

def load_raw_image(scen_dir, stem, cam_name):
    """원본 해상도(1600×900) RGB 이미지."""
    path = os.path.join(scen_dir, 'images', cam_name, f'{stem}.jpg')
    img = cv2.imread(path)
    if img is None:
        return np.zeros((ORIG_IMG_HEIGHT, ORIG_IMG_WIDTH, 3), dtype=np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── 3D 박스 → 8 코너 ───────────────────────────────────────────────────────

def box_corners_ego(box):
    """anchor/GT 박스 [11] → ego 좌표계 8 코너 [8, 3]."""
    x, y, z_bottom = float(box[0]), float(box[1]), float(box[2])
    w     = float(np.exp(box[3]))
    l     = float(np.exp(box[4]))
    h     = float(np.exp(box[5]))
    sin_y = float(box[6])
    cos_y = float(box[7])
    z_c   = z_bottom + h * 0.5

    corners_local = np.array([
        [ l/2,  w/2,  h/2], [ l/2,  w/2, -h/2],
        [ l/2, -w/2,  h/2], [ l/2, -w/2, -h/2],
        [-l/2,  w/2,  h/2], [-l/2,  w/2, -h/2],
        [-l/2, -w/2,  h/2], [-l/2, -w/2, -h/2],
    ], dtype=np.float32)

    Rz = np.array([[cos_y, -sin_y, 0],
                   [sin_y,  cos_y, 0],
                   [0,      0,     1]], dtype=np.float32)
    return (Rz @ corners_local.T).T + np.array([x, y, z_c], dtype=np.float32)


# ── 카메라 투영 ────────────────────────────────────────────────────────────

def project_corners(box, cam_name):
    """
    박스 8 코너를 원본 해상도 이미지 픽셀 좌표로 투영.

    카메라 컨벤션 (inference.py / decoder.py 동일):
      depth = cam_x
      u = fx * (-cam_y) / depth + cx
      v = fy * (-cam_z) / depth + cy

    반환: us [8], vs [8], valid [8]  (depth > 0.1 인 점)
    """
    corners   = box_corners_ego(box)
    corners_h = np.hstack([corners, np.ones((8, 1), dtype=np.float32)])

    E       = _EXTRINSICS[cam_name]
    pts_cam = (E @ corners_h.T).T          # [8, 4]
    depth   = pts_cam[:, 0]
    valid   = depth > 0.1

    K  = _INTRINSICS[cam_name]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    us = np.full(8, np.nan, dtype=np.float32)
    vs = np.full(8, np.nan, dtype=np.float32)
    if valid.any():
        d        = depth[valid]
        us[valid] = fx * (-pts_cam[valid, 1]) / d + cx
        vs[valid] = fy * (-pts_cam[valid, 2]) / d + cy

    return us, vs, valid


def draw_cuboid_on_image(img_rgb, box, cam_name, color_bgr, label=None, thickness=2):
    """
    RGB 이미지에 3D 큐보이드를 그린 뒤 RGB로 반환.
    depth > 0.1 인 엣지만 그리고, cv2.line 이 화면 밖 선분을 자동 클립.
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    us, vs, valid = project_corners(box, cam_name)

    drew = False
    for a, b in BOX_EDGES:
        if not (valid[a] and valid[b]):
            continue
        p1 = (int(round(float(us[a]))), int(round(float(vs[a]))))
        p2 = (int(round(float(us[b]))), int(round(float(vs[b]))))
        cv2.line(img_bgr, p1, p2, color_bgr, thickness, cv2.LINE_AA)
        drew = True

    if label and drew:
        vis = valid & ~np.isnan(us)
        if vis.any():
            x_txt = int(np.nanmin(us[vis]))
            y_txt = int(np.nanmin(vs[vis]))
            cv2.putText(img_bgr, label,
                        (max(0, x_txt + 2), max(20, y_txt - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        color_bgr, 1, cv2.LINE_AA)

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ── BEV 박스 그리기 ────────────────────────────────────────────────────────

def draw_rotated_box_bev(ax, box, color, lw=2.0, label=None, alpha=1.0):
    x, y  = float(box[0]), float(box[1])
    w     = float(np.exp(box[3]))
    l     = float(np.exp(box[4]))
    angle = float(np.degrees(np.arctan2(float(box[6]), float(box[7]))))

    rect = patches.Rectangle(
        (x - l / 2, y - w / 2), l, w,
        linewidth=lw, edgecolor=color, facecolor='none', alpha=alpha,
    )
    rect.set_transform(
        transforms.Affine2D().rotate_deg_around(x, y, angle) + ax.transData
    )
    ax.add_patch(rect)
    if label:
        ax.text(x, y, label, color=color, fontsize=7)


# ── 모델 로드 ──────────────────────────────────────────────────────────────

def load_model(weights_path, device):
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"가중치 없음: {weights_path}")
    model = AutoNavModel().to(device)
    ckpt  = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        state = ckpt['model_state']
    elif isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# ── 추론 ──────────────────────────────────────────────────────────────────

def find_index_by_stem(dataset, stem):
    for idx in range(len(dataset)):
        scen_dir, item_stem = dataset.items[idx]
        full_stem = f"{os.path.basename(scen_dir)}/{item_stem}"
        if stem in (item_stem, full_stem):
            return idx
    raise ValueError(f"stem not found in split: {stem}")


@torch.no_grad()
def predict(sample, model, device, det_thresh, mode):
    images     = sample['images'].unsqueeze(0).to(device)
    intrinsics = sample['intrinsics'].unsqueeze(0).to(device)
    extrinsics = sample['extrinsics'].unsqueeze(0).to(device)

    output      = model(images, intrinsics, extrinsics)
    det_quality = output.get('det_quality', None)
    quality_b   = det_quality[0] if det_quality is not None else None
    mode_cfg    = QUALITY_MODES[mode]

    boxes, labels, scores = decode_detections(
        output['det_cls'][0],
        output['det_box'][0],
        det_quality=quality_b,
        score_thresh=det_thresh,
        apply_quality=mode_cfg['apply_quality'],
        quality_power=mode_cfg['quality_power'],
    )
    return boxes.cpu().numpy(), labels.cpu().numpy(), scores.cpu().numpy()


# ── 샘플 시각화 ────────────────────────────────────────────────────────────

def visualize_one(dataset, model, device, stem, out_dir, det_thresh, mode):
    os.makedirs(out_dir, exist_ok=True)

    idx              = find_index_by_stem(dataset, stem)
    sample           = dataset[idx]
    scen_dir, item_stem = dataset.items[idx]

    gt_boxes  = sample['dynamic_gt_boxes'].cpu().numpy()
    gt_labels = sample['dynamic_gt_labels'].cpu().numpy()
    pred_boxes, pred_labels, pred_scores = predict(
        sample, model, device, det_thresh, mode
    )

    # ── 카메라 이미지 로드 + 예측 큐보이드 투영 ───────────────────────────
    cam_imgs = []
    for cam_name in CAM_ORDER:
        img = load_raw_image(scen_dir, item_stem, cam_name)
        for box, cls_id, score in zip(pred_boxes, pred_labels, pred_scores):
            color_bgr = DET_COLORS_BGR.get(int(cls_id), (0, 0, 255))
            label     = f"{DET_SHORT.get(int(cls_id), '?')} {score:.2f}"
            img = draw_cuboid_on_image(img, box, cam_name, color_bgr, label=label)
        cam_imgs.append(img)

    # ── Figure 레이아웃 ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor('#1a1a1a')
    gs = GridSpec(2, 3, figure=fig,
                  height_ratios=[1, 1.15],
                  hspace=0.28, wspace=0.04)

    # 상단: 카메라 3개
    for ci, (cam_img, cam_name) in enumerate(zip(cam_imgs, CAM_ORDER)):
        ax = fig.add_subplot(gs[0, ci])
        ax.imshow(cam_img)
        ax.set_title(CAM_DISPLAY.get(cam_name, cam_name),
                     fontsize=11, color='white', pad=4)
        ax.axis('off')

    # 하단: BEV (3열 전체)
    ax_bev = fig.add_subplot(gs[1, :])
    ax_bev.set_facecolor('black')
    ax_bev.plot(0, 0, marker='>', color='white', markersize=12, zorder=5)

    for box, cls_id in zip(gt_boxes, gt_labels):
        draw_rotated_box_bev(
            ax_bev, box, color='cyan', lw=1.5,
            label=f"GT {DET_NAMES.get(int(cls_id), '?')}", alpha=0.8,
        )
    for box, cls_id, score in zip(pred_boxes, pred_labels, pred_scores):
        color = DET_COLORS_BEV.get(int(cls_id), 'red')
        draw_rotated_box_bev(
            ax_bev, box, color=color, lw=2.0,
            label=f"{DET_SHORT.get(int(cls_id), '?')} {score:.2f}",
        )

    ax_bev.set_xlim(-5, 65)
    ax_bev.set_ylim(-35, 35)
    ax_bev.set_aspect('equal')
    ax_bev.grid(True, color='gray', linestyle=':', alpha=0.35)
    ax_bev.set_xlabel('x forward (m)', color='white')
    ax_bev.set_ylabel('y left (m)', color='white')
    ax_bev.tick_params(colors='white')
    for spine in ax_bev.spines.values():
        spine.set_edgecolor('gray')
    ax_bev.set_title(
        f"BEV  |  GT={len(gt_boxes)}  Pred={len(pred_boxes)}"
        f"  |  thresh={det_thresh:.2f}  mode={mode}",
        color='white', fontsize=10,
    )

    fig.suptitle(sample['stem'], fontsize=12, color='white', y=0.995)

    safe_stem = sample['stem'].replace('/', '__')
    out_path  = os.path.join(
        out_dir, f"{safe_stem}_{mode}_t{det_thresh:.2f}_camproj.png"
    )
    fig.savefig(out_path, dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[visualize] saved: {out_path}")


# ── 진입점 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='카메라 투영 + BEV 통합 시각화')
    parser.add_argument('--dataset-root', default='./dataset')
    parser.add_argument('--weights',      default='best_model.pth')
    parser.add_argument('--split',        default='val', choices=['train', 'val'])
    parser.add_argument('--stem',         default=None,
                        help='특정 샘플 (예: scen01/live_000042). '
                             '없으면 --n 개 랜덤 선택')
    parser.add_argument('--n',            type=int, default=5)
    parser.add_argument('--out',          default='./pred_vis_cam')
    parser.add_argument('--det-thresh',   type=float, default=0.15,
                        help='softcalibrated f1@0.15 기준과 맞춘 기본값')
    parser.add_argument('--mode',         default='softcalibrated',
                        choices=list(QUALITY_MODES.keys()))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}")
    print(f"[mode]   {args.mode} -> {QUALITY_MODES[args.mode]}")

    dataset = MoraiDataset(dataset_root=args.dataset_root, split=args.split)
    model   = load_model(args.weights, device)
    print(f"[model]  loaded: {args.weights}\n")

    if args.stem:
        stems = [args.stem]
    else:
        indices = random.sample(range(len(dataset)), min(args.n, len(dataset)))
        stems = [
            f"{os.path.basename(dataset.items[i][0])}/{dataset.items[i][1]}"
            for i in indices
        ]

    print(f"[visualize] {len(stems)}개 샘플 처리")
    for stem in stems:
        try:
            visualize_one(
                dataset, model, device, stem, args.out, args.det_thresh, args.mode
            )
        except Exception as e:
            print(f"[ERROR] {stem}: {e}")
            traceback.print_exc()

    print(f"\n[완료] 저장 위치: {args.out}/")


if __name__ == '__main__':
    main()
