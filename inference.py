#!/usr/bin/env python3
"""
inference.py
학습된 모델로 추론 + BEV/카메라 시각화
"""

import os
import json
import argparse
import random
import numpy as np
import cv2
import torch

from train import AutoNavModel
from camera_configs import (INTRINSICS as _INTRINSICS,
                             EXTRINSICS as _EXTRINSICS,
                             CAM_ORDER)

DATASET_DIR  = './dataset'
SCORE_THRESH = 0.2
CLASS_NAMES  = {0: 'vehicle'}
CLASS_COLORS = {0: (0, 255, 0)}
IMG_SIZE     = 224


def bev_nms(boxes, scores, iou_thresh=0.3):
    if len(boxes) == 0:
        return []
    order = np.argsort(scores)[::-1]
    keep  = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        cx_i = boxes[i, 0];  cy_i = boxes[i, 1]
        w_i  = np.exp(boxes[i, 3]); l_i = np.exp(boxes[i, 4])
        rest = order[1:]
        cx_r = boxes[rest, 0]; cy_r = boxes[rest, 1]
        w_r  = np.exp(boxes[rest, 3]); l_r = np.exp(boxes[rest, 4])
        inter_x = np.maximum(
            0, np.minimum(cx_i+w_i/2, cx_r+w_r/2)
             - np.maximum(cx_i-w_i/2, cx_r-w_r/2))
        inter_y = np.maximum(
            0, np.minimum(cy_i+l_i/2, cy_r+l_r/2)
             - np.maximum(cy_i-l_i/2, cy_r-l_r/2))
        inter = inter_x * inter_y
        iou   = inter / (w_i*l_i + w_r*l_r - inter + 1e-6)
        order = rest[iou < iou_thresh]
    return keep


def project_box_to_cam(box_ego, cam_name, orig_w=640, orig_h=480):
    x, y, z  = box_ego[0], box_ego[1], box_ego[2]
    sx       = np.exp(box_ego[3])   # ln_w → w
    sy       = np.exp(box_ego[4])   # ln_l → l
    sz       = np.exp(box_ego[5])   # ln_h → h
    sin_y, cos_y = box_ego[6], box_ego[7]

    corners_local = np.array([
        [ sx/2,  sy/2,  sz/2], [ sx/2,  sy/2, -sz/2],
        [ sx/2, -sy/2,  sz/2], [ sx/2, -sy/2, -sz/2],
        [-sx/2,  sy/2,  sz/2], [-sx/2,  sy/2, -sz/2],
        [-sx/2, -sy/2,  sz/2], [-sx/2, -sy/2, -sz/2],
    ], dtype=np.float32)

    Rz = np.array([[cos_y, -sin_y, 0],
                   [sin_y,  cos_y, 0],
                   [    0,      0, 1]], dtype=np.float32)
    corners_ego = (Rz @ corners_local.T).T + np.array([x, y, z])

    E = _EXTRINSICS[cam_name]
    corners_h   = np.hstack([corners_ego,
                              np.ones((8, 1), dtype=np.float32)])
    corners_cam = (E @ corners_h.T).T

    depth = corners_cam[:, 0]   # X = depth
    valid = depth > 0.1
    if valid.sum() < 1:
        return None

    K  = _INTRINSICS[cam_name]
    fx = K[0, 0]; cx = K[0, 2]; cy = K[1, 2]

    d  = depth[valid]
    us = fx * (-corners_cam[valid, 1]) / d + cx
    vs = fx * (-corners_cam[valid, 2]) / d + cy

    u_min = max(0,      float(us.min()))
    u_max = min(orig_w, float(us.max()))
    v_min = max(0,      float(vs.min()))
    v_max = min(orig_h, float(vs.max()))

    if u_max - u_min < 2 or v_max - v_min < 2:
        return None
    if u_min >= orig_w or u_max <= 0 or v_min >= orig_h or v_max <= 0:
        return None

    return int(u_min), int(v_min), int(u_max), int(v_max)


def run_inference(weights_path, stems, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[디바이스] {device}")

    model = AutoNavModel().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f"[모델] {weights_path} 로드 완료\n")

    groups_path = os.path.join(DATASET_DIR, 'frame_groups.json')
    with open(groups_path) as f:
        all_groups = json.load(f)
    stem_to_group = {g['label_stem']: g for g in all_groups}

    img_dir = os.path.join(DATASET_DIR, 'images')
    lbl_dir = os.path.join(DATASET_DIR, 'labels_3d')

    for stem in stems:
        if stem not in stem_to_group:
            print(f"[SKIP] {stem}")
            continue

        group = stem_to_group[stem]
        cams  = group['cams']

        images     = torch.zeros(1, 6, 3, IMG_SIZE, IMG_SIZE)
        intrinsics = torch.zeros(1, 6, 3, 3)
        extrinsics = torch.zeros(1, 6, 4, 4)

        for ci, cam_name in enumerate(CAM_ORDER):
            intrinsics[0, ci] = torch.from_numpy(_INTRINSICS[cam_name])
            extrinsics[0, ci] = torch.from_numpy(_EXTRINSICS[cam_name])
            if cam_name in cams:
                path = os.path.join(img_dir, f"{cams[cam_name]}.jpg")
                img  = cv2.imread(path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    images[0, ci] = torch.from_numpy(img).permute(2,0,1).float()/255.

        with torch.no_grad():
            det_cls, det_box, map_cls, map_lines = model(
                images.to(device),
                intrinsics.to(device),
                extrinsics.to(device)
            )

        probs = det_cls.softmax(-1)          # [900, 2] = vehicle/background
        vehicle_scores = probs[:, 0]         # foreground score만 사용
        keep_mask = vehicle_scores > SCORE_THRESH

        boxes_cand = det_box[keep_mask].cpu().numpy()
        cls_cand   = np.zeros(len(boxes_cand), dtype=np.int64)   # 전부 vehicle class 0
        scr_cand   = vehicle_scores[keep_mask].cpu().numpy()

        if len(boxes_cand) > 0:
            nms_idx    = bev_nms(boxes_cand, scr_cand)
            boxes_keep = boxes_cand[nms_idx]
            cls_keep   = cls_cand[nms_idx]
            scr_keep   = scr_cand[nms_idx]
        else:
            boxes_keep = boxes_cand
            cls_keep   = cls_cand
            scr_keep   = scr_cand

        print(f"[{stem}] 예측: {len(boxes_keep)}개")

        gt_boxes_raw = []
        lbl_path = os.path.join(lbl_dir, f"{stem}.txt")
        if os.path.isfile(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 12:
                        gt_boxes_raw.append(list(map(float, parts[1:])))

        for cam_name in CAM_ORDER:
            if cam_name not in cams:
                continue
            path = os.path.join(img_dir, f"{cams[cam_name]}.jpg")
            img  = cv2.imread(path)
            if img is None:
                continue
            orig_h, orig_w = img.shape[:2]

            for gt_box in gt_boxes_raw:
                bbox = project_box_to_cam(gt_box, cam_name, orig_w, orig_h)
                if bbox:
                    cv2.rectangle(img, (bbox[0], bbox[1]),
                                  (bbox[2], bbox[3]), (255, 100, 0), 2)
                    cv2.putText(img, 'GT', (bbox[0]+2, bbox[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (255, 100, 0), 1, cv2.LINE_AA)

            for box, cls_id, score in zip(boxes_keep, cls_keep, scr_keep):
                bbox = project_box_to_cam(box, cam_name, orig_w, orig_h)
                if bbox:
                    color = CLASS_COLORS.get(int(cls_id), (0, 255, 0))
                    cv2.rectangle(img, (bbox[0], bbox[1]),
                                  (bbox[2], bbox[3]), color, 2)
                    label = f"{CLASS_NAMES.get(int(cls_id),'?')} {score:.2f}"
                    cv2.putText(img, label, (bbox[0]+2, bbox[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                color, 1, cv2.LINE_AA)

            out_name = f"{stem}_{cam_name}_infer.jpg"
            cv2.imwrite(os.path.join(out_dir, out_name), img)

        print(f"   → {out_dir}/{stem}_*.jpg")

    print(f"\n✅ 완료!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='best_model.pth')
    parser.add_argument('--n',       type=int, default=10)
    parser.add_argument('--stem',    default=None)
    parser.add_argument('--out',     default='./inference_results')
    args = parser.parse_args()

    if not os.path.isfile(args.weights):
        print(f"[ERROR] 가중치 없음: {args.weights}")
        exit(1)

    if args.stem:
        stems = [args.stem]
    else:
        with open(os.path.join(DATASET_DIR, 'frame_groups.json')) as f:
            all_groups = json.load(f)
        all_stems = [g['label_stem'] for g in all_groups]
        stems = random.sample(all_stems, min(args.n, len(all_stems)))

    run_inference(args.weights, stems, args.out)