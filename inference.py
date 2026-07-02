#!/usr/bin/env python3
"""
inference.py
학습된 모델로 추론 + BEV/카메라 시각화
"""

import os
import csv
import argparse
import random
import numpy as np
import cv2
import torch

from costmap import rasterize_detections, save_costmap
from train import AutoNavModel
from camera_configs import (INTRINSICS as _INTRINSICS,
                             EXTRINSICS as _EXTRINSICS,
                             CAM_ORDER)
from morai_dataset import IMG_HEIGHT, IMG_MEAN, IMG_STD, IMG_WIDTH, scale_intrinsic_for_input

DATASET_DIR  = '/data/dataset/scen01'
SCORE_THRESH = 0.05
PRE_NMS_TOPK = 300
CLASS_NAMES  = {0: 'vehicle',     1: 'pedestrian'}
CLASS_COLORS = {0: (0, 255, 0),   1: (0, 165, 255)}   # green, orange (BGR)


def load_model_weights(model, weights_path, device):
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        epoch = checkpoint.get('epoch')
        if epoch is not None:
            print(f"[모델] full checkpoint 감지: epoch={epoch}")
        return
    model.load_state_dict(checkpoint)


def combine_scores(fg_scores, quality, mode):
    if mode == 'raw':
        return fg_scores
    if mode == 'calibrated':
        return fg_scores * quality
    if mode == 'softcalibrated':
        return fg_scores * np.sqrt(np.clip(quality, 1e-6, 1.0))
    raise ValueError(f"unknown score mode: {mode}")


def bev_nms(boxes, scores, iou_thresh=0.3, center_dist_thresh=1.5):
    # BEV: ego x=전방(길이 l), ego y=좌(폭 w). 회전 무시 axis-aligned IoU.
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
            0, np.minimum(cx_i+l_i/2, cx_r+l_r/2)
             - np.maximum(cx_i-l_i/2, cx_r-l_r/2))
        inter_y = np.maximum(
            0, np.minimum(cy_i+w_i/2, cy_r+w_r/2)
             - np.maximum(cy_i-w_i/2, cy_r-w_r/2))
        inter = inter_x * inter_y
        iou   = inter / (w_i*l_i + w_r*l_r - inter + 1e-6)
        center_dist = np.sqrt((cx_r - cx_i) ** 2 + (cy_r - cy_i) ** 2)
        suppress = (iou >= iou_thresh) | (center_dist < center_dist_thresh)
        order = rest[~suppress]
    return keep


def _box_corners_ego(box_ego):
    x, y, z_bottom = box_ego[0], box_ego[1], box_ego[2]
    w       = np.exp(box_ego[3])   # ln_w → 폭   (lateral)
    l       = np.exp(box_ego[4])   # ln_l → 길이 (forward)
    h       = np.exp(box_ego[5])   # ln_h → 높이
    sin_y, cos_y = box_ego[6], box_ego[7]
    z_center = z_bottom + h * 0.5

    corners_local = np.array([
        [ l/2,  w/2,  h/2], [ l/2,  w/2, -h/2],
        [ l/2, -w/2,  h/2], [ l/2, -w/2, -h/2],
        [-l/2,  w/2,  h/2], [-l/2,  w/2, -h/2],
        [-l/2, -w/2,  h/2], [-l/2, -w/2, -h/2],
    ], dtype=np.float32)

    Rz = np.array([[cos_y, -sin_y, 0],
                   [sin_y,  cos_y, 0],
                   [    0,      0, 1]], dtype=np.float32)
    return (Rz @ corners_local.T).T + np.array([x, y, z_center])


def project_box_corners_to_cam(box_ego, cam_name, orig_w=1600, orig_h=900):
    corners_ego = _box_corners_ego(box_ego)

    E = _EXTRINSICS[cam_name]
    corners_h   = np.hstack([corners_ego,
                              np.ones((8, 1), dtype=np.float32)])
    corners_cam = (E @ corners_h.T).T

    depth = corners_cam[:, 0]   # X = depth
    valid = depth > 0.1
    if valid.sum() < 1:
        return None, None

    K  = _INTRINSICS[cam_name]
    fx = K[0, 0]; fy = K[1, 1]; cx = K[0, 2]; cy = K[1, 2]

    us = np.full(8, np.nan, dtype=np.float32)
    vs = np.full(8, np.nan, dtype=np.float32)
    d  = depth[valid]
    us[valid] = fx * (-corners_cam[valid, 1]) / d + cx
    vs[valid] = fy * (-corners_cam[valid, 2]) / d + cy

    in_frame = (
        valid &
        (us >= 0.0) & (us < float(orig_w)) &
        (vs >= 0.0) & (vs < float(orig_h))
    )
    if in_frame.sum() < 1:
        return None, None

    pts = np.stack([us, vs], axis=1)
    return pts, valid


def project_box_to_cam(box_ego, cam_name, orig_w=1600, orig_h=900):
    # local frame 컨벤션 (train.py / visualize_pred_fixed.py 와 동일):
    #   local x = 길이(전방), local y = 폭(좌), local z = 높이
    pts, valid = project_box_corners_to_cam(box_ego, cam_name, orig_w, orig_h)
    if pts is None:
        return None

    us = pts[valid, 0]
    vs = pts[valid, 1]

    u_min = max(0,      float(us.min()))
    u_max = min(orig_w, float(us.max()))
    v_min = max(0,      float(vs.min()))
    v_max = min(orig_h, float(vs.max()))

    if u_max - u_min < 2 or v_max - v_min < 2:
        return None
    if u_min >= orig_w or u_max <= 0 or v_min >= orig_h or v_max <= 0:
        return None

    return int(u_min), int(v_min), int(u_max), int(v_max)


def draw_cuboid(img, box_ego, cam_name, color, label=None, thickness=2):
    orig_h, orig_w = img.shape[:2]
    pts, valid = project_box_corners_to_cam(box_ego, cam_name, orig_w, orig_h)
    if pts is None or valid is None:
        return False

    edges = [
        (0, 2), (2, 6), (6, 4), (4, 0),  # top face
        (1, 3), (3, 7), (7, 5), (5, 1),  # bottom face
        (0, 1), (2, 3), (4, 5), (6, 7),  # verticals
    ]
    drew = False
    for a, b in edges:
        if not (valid[a] and valid[b]):
            continue
        p1 = tuple(np.round(pts[a]).astype(int))
        p2 = tuple(np.round(pts[b]).astype(int))
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
        drew = True

    if label and drew:
        valid_pts = pts[valid]
        x = int(np.nanmin(valid_pts[:, 0]))
        y = int(np.nanmin(valid_pts[:, 1]))
        cv2.putText(
            img,
            label,
            (max(0, x + 2), max(12, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    return drew


def load_gt_from_csv(csv_path):
    """CSV GT 파일에서 박스 리스트 반환. 각 항목: [x,y,z,ln_w,ln_l,ln_h,sin_yaw,cos_yaw,vx,vy,vz]"""
    boxes = []
    if not os.path.isfile(csv_path):
        return boxes
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            boxes.append([
                float(row['x']),       float(row['y']),       float(row['z']),
                float(row['ln_w']),    float(row['ln_l']),    float(row['ln_h']),
                float(row['sin_yaw']), float(row['cos_yaw']),
                float(row['vx']),      float(row['vy']),      float(row['vz']),
            ])
    return boxes


def run_inference(
    weights_path,
    stems,
    out_dir,
    score_mode='softcalibrated',
    score_thresh=SCORE_THRESH,
    draw_mode='cuboid',
):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[디바이스] {device}")

    model = AutoNavModel(use_temporal_memory=False).to(device)
    load_model_weights(model, weights_path, device)
    model.eval()
    print(f"[모델] {weights_path} 로드 완료")
    print(f"[score] mode={score_mode}, thresh={score_thresh:.3f}, draw={draw_mode}\n")

    img_root = os.path.join(DATASET_DIR, 'images')
    lbl_dir  = os.path.join(DATASET_DIR, 'labels_3d')

    for stem in stems:
        n_cams     = len(CAM_ORDER)
        images     = torch.zeros(1, n_cams, 3, IMG_HEIGHT, IMG_WIDTH)
        intrinsics = torch.zeros(1, n_cams, 3, 3)
        extrinsics = torch.zeros(1, n_cams, 4, 4)

        for ci, cam_name in enumerate(CAM_ORDER):
            intrinsics[0, ci] = torch.from_numpy(scale_intrinsic_for_input(_INTRINSICS[cam_name]))
            extrinsics[0, ci] = torch.from_numpy(_EXTRINSICS[cam_name])
            path = os.path.join(img_root, cam_name, f"{stem}.jpg")
            img  = cv2.imread(path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.
                images[0, ci] = (img_t - IMG_MEAN) / IMG_STD

        with torch.no_grad():
            model_out = model(
                images.to(device),
                intrinsics.to(device),
                extrinsics.to(device)
            )

        det_cls = model_out['det_cls'][0]
        det_box = model_out['det_box'][0]
        det_quality = model_out['det_quality'][0]

        # det_cls : [900, 2]  (0=vehicle, 1=pedestrian) — sigmoid 2-class, 배경 채널 없음
        # Detection confidence is the best per-class sigmoid probability.
        probs       = det_cls.sigmoid().cpu().numpy()
        fg_probs    = probs
        best_cls    = np.argmax(fg_probs, axis=1)     # [900]
        if det_quality.ndim == 1:
            centerness = det_quality
        else:
            centerness = det_quality[..., 0]
        quality     = torch.sigmoid(centerness).cpu().numpy()
        fg_score    = np.max(fg_probs, axis=1)
        best_score  = combine_scores(fg_score, quality, score_mode)
        keep_mask   = best_score > score_thresh

        boxes_cand = det_box.cpu().numpy()[keep_mask]
        cls_cand   = best_cls[keep_mask]
        scr_cand   = best_score[keep_mask]
        if len(scr_cand) > PRE_NMS_TOPK:
            topk = np.argsort(scr_cand)[-PRE_NMS_TOPK:]
            boxes_cand = boxes_cand[topk]
            cls_cand = cls_cand[topk]
            scr_cand = scr_cand[topk]

        # 클래스별 NMS
        if len(boxes_cand) > 0:
            keep_global = []
            for c in (0, 1):
                cls_mask = cls_cand == c
                if not cls_mask.any():
                    continue
                local_idx = bev_nms(boxes_cand[cls_mask], scr_cand[cls_mask])
                original_idx = np.where(cls_mask)[0]
                keep_global.extend(original_idx[i] for i in local_idx)
            keep_global = np.array(keep_global, dtype=np.int64)
            boxes_keep  = boxes_cand[keep_global]
            cls_keep    = cls_cand[keep_global]
            scr_keep    = scr_cand[keep_global]
        else:
            boxes_keep = boxes_cand
            cls_keep   = cls_cand
            scr_keep   = scr_cand

        n_veh = int((cls_keep == 0).sum())
        n_ped = int((cls_keep == 1).sum())
        print(f"[{stem}] 예측: vehicle {n_veh}, pedestrian {n_ped}")

        costmap = rasterize_detections(boxes_keep, cls_keep, scr_keep)
        safe_stem = stem.replace('/', '__')
        save_costmap(costmap, os.path.join(out_dir, "costmaps", safe_stem))

        gt_boxes_raw = load_gt_from_csv(os.path.join(lbl_dir, f"{stem}.csv"))

        for cam_name in CAM_ORDER:
            path = os.path.join(img_root, cam_name, f"{stem}.jpg")
            img  = cv2.imread(path)
            if img is None:
                continue
            orig_h, orig_w = img.shape[:2]

            for gt_box in gt_boxes_raw:
                if draw_mode in ('cuboid', 'both'):
                    draw_cuboid(img, gt_box, cam_name, (255, 100, 0), label='GT', thickness=2)
                if draw_mode in ('box2d', 'both'):
                    bbox = project_box_to_cam(gt_box, cam_name, orig_w, orig_h)
                else:
                    bbox = None
                if bbox is not None:
                    cv2.rectangle(img, (bbox[0], bbox[1]),
                                  (bbox[2], bbox[3]), (255, 100, 0), 2)
                    cv2.putText(img, 'GT', (bbox[0]+2, bbox[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (255, 100, 0), 1, cv2.LINE_AA)

            for box, cls_id, score in zip(boxes_keep, cls_keep, scr_keep):
                color = CLASS_COLORS.get(int(cls_id), (0, 255, 0))
                label = f"{CLASS_NAMES.get(int(cls_id), '?')} {score:.2f}"
                if draw_mode in ('cuboid', 'both'):
                    draw_cuboid(img, box, cam_name, color, label=label, thickness=2)
                if draw_mode in ('box2d', 'both'):
                    bbox = project_box_to_cam(box, cam_name, orig_w, orig_h)
                else:
                    bbox = None
                if bbox is not None:
                    cv2.rectangle(img, (bbox[0], bbox[1]),
                                  (bbox[2], bbox[3]), color, 2)
                    cv2.putText(img, label, (bbox[0]+2, bbox[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                color, 1, cv2.LINE_AA)

            out_name = f"{stem}_{cam_name}_infer.jpg"
            cv2.imwrite(os.path.join(out_dir, out_name), img)

        print(f"   → {out_dir}/{stem}_*.jpg")
        print(f"   → {out_dir}/costmaps/{safe_stem}.npy/.png")

    print(f"\n✅ 완료!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='best_model.pth')
    parser.add_argument('--n',       type=int, default=10)
    parser.add_argument('--stem',    default=None)
    parser.add_argument('--out',     default='./inference_results')
    parser.add_argument(
        '--score-mode',
        default='softcalibrated',
        choices=('raw', 'calibrated', 'softcalibrated'),
    )
    parser.add_argument('--score-thresh', type=float, default=SCORE_THRESH)
    parser.add_argument(
        '--draw-mode',
        default='cuboid',
        choices=('cuboid', 'box2d', 'both'),
    )
    args = parser.parse_args()

    if not os.path.isfile(args.weights):
        print(f"[ERROR] 가중치 없음: {args.weights}")
        exit(1)

    if args.stem:
        stems = [args.stem]
    else:
        lbl_dir   = os.path.join(DATASET_DIR, 'labels_3d')
        all_stems = sorted([
            os.path.splitext(f)[0]
            for f in os.listdir(lbl_dir)
            if f.endswith('.csv')
        ])
        stems = random.sample(all_stems, min(args.n, len(all_stems)))

    run_inference(
        args.weights,
        stems,
        args.out,
        args.score_mode,
        args.score_thresh,
        args.draw_mode,
    )
