#!/usr/bin/env python3
"""
=============================================================
  inference.py
  학습된 모델로 실제 이미지에 3D 예측 박스를 2D로 투영해서 시각화
=============================================================
사용법:
  python3 inference.py                              # dataset/images 에서 랜덤 10장
  python3 inference.py --n 20                       # 20장
  python3 inference.py --stem cam_front_00100       # 특정 프레임
  python3 inference.py --weights best_model.pth     # 가중치 파일 지정
=============================================================
"""

import os
import json
import argparse
import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from resnet_fpn import ResNet50_FPN, Bottleneck
from anchor_generator import generate_anchors
from decoder import FFNDecoder
from morai_dataset import _INTRINSICS, _EXTRINSICS, CAM_ORDER, CAMERA_CONFIGS

# 설정
DATASET_DIR  = './dataset'
SCORE_THRESH = 0.5    # 이 확률 이상인 예측만 표시
CLASS_NAMES  = {0: 'car', 1: 'truck', 2: 'bus'}
CLASS_COLORS = {0: (0, 255, 0), 1: (0, 165, 255), 2: (0, 0, 255)}
IMG_SIZE     = 224


# ===========================================================
# 모델 정의 (train.py와 동일)
# ===========================================================
class AutoNavModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet50_FPN(Bottleneck)
        self.decoder  = FFNDecoder()
        self.anchors  = generate_anchors()

    def forward(self, images, intrinsics, extrinsics):
        device = images.device
        self.anchors = self.anchors.to(device)
        N = self.anchors.shape[0]

        anchors_homo = torch.cat(
            [self.anchors, torch.ones(N, 1, device=device)], dim=-1
        )

        agg_features = torch.zeros(N, 256, device=device)
        valid_cams   = 0

        for cam_idx in range(6):
            cam_img = images[0, cam_idx]
            if cam_img.abs().sum() < 1e-6:
                continue

            features   = self.backbone(cam_img.unsqueeze(0))
            p2_feature = features[0]

            E = extrinsics[0, cam_idx]
            K = intrinsics[0, cam_idx]

            points_cam = (E @ anchors_homo.T).T
            points_2d  = (K @ points_cam[:, :3].T).T
            depth = points_2d[:, 2]
            u = points_2d[:, 0] / (depth + 1e-6)
            v = points_2d[:, 1] / (depth + 1e-6)

            valid_mask = depth > 0.1
            u_norm = (u / 640.0) * 2.0 - 1.0
            v_norm = (v / 480.0) * 2.0 - 1.0
            grid   = torch.stack([u_norm, v_norm], dim=-1).view(1, 1, N, 2)

            sampled = F.grid_sample(p2_feature, grid, align_corners=False)
            sampled = sampled.view(256, N).T
            mask    = valid_mask.float().unsqueeze(1)
            sampled = sampled * mask

            agg_features += sampled
            valid_cams   += 1

        if valid_cams > 0:
            agg_features = agg_features / valid_cams

        pred_classes, pred_boxes = self.decoder(agg_features)
        return pred_classes, pred_boxes


# ===========================================================
# 3D 박스 → 2D 투영 유틸
# MORAI 카메라 좌표계: X=전방(depth), Y=좌측, Z=상방
# u = fx * (-Y) / X + cx
# v = fy * (-Z) / X + cy
# ===========================================================
def project_box_to_cam(box_ego, cam_name, orig_w=640, orig_h=480):
    """
    Ego 좌표계 3D 박스 → 카메라 이미지상 2D 박스 (픽셀)
    box_ego: [x, y, z, size_x, size_y, size_z, sin, cos, vx, vy, vz]
    반환: (u_min, v_min, u_max, v_max) 또는 None
    """
    x, y, z = box_ego[0], box_ego[1], box_ego[2]
    # size_x = 차량 길이(전후), size_y = 너비(좌우), size_z = 높이
    sx, sy, sz = box_ego[3], box_ego[4], box_ego[5]
    sin_y, cos_y = box_ego[6], box_ego[7]

    # 8개 코너 (NPC 로컬 좌표, X=전후, Y=좌우, Z=상하)
    corners_local = np.array([
        [ sx/2,  sy/2,  sz/2],
        [ sx/2,  sy/2, -sz/2],
        [ sx/2, -sy/2,  sz/2],
        [ sx/2, -sy/2, -sz/2],
        [-sx/2,  sy/2,  sz/2],
        [-sx/2,  sy/2, -sz/2],
        [-sx/2, -sy/2,  sz/2],
        [-sx/2, -sy/2, -sz/2],
    ], dtype=np.float32)

    # NPC heading으로 회전 후 Ego 좌표계 위치로 이동
    Rz = np.array([[cos_y, -sin_y, 0],
                   [sin_y,  cos_y, 0],
                   [    0,      0, 1]], dtype=np.float32)
    corners_ego = (Rz @ corners_local.T).T + np.array([x, y, z])

    # Ego → 카메라 좌표 변환 (4x4 extrinsic)
    E = _EXTRINSICS[cam_name]          # [4, 4]
    corners_h   = np.hstack([corners_ego, np.ones((8, 1), dtype=np.float32)])
    corners_cam = (E @ corners_h.T).T  # [8, 4]

    # MORAI 카메라 좌표계: X=전방(depth), Y=좌, Z=상
    # depth = X축
    depth = corners_cam[:, 0]
    valid = depth > 0.1
    if valid.sum() < 1:
        return None

    # K에서 fx, cx, cy 추출
    K  = _INTRINSICS[cam_name]
    fx = K[0, 0]
    cx = K[0, 2]
    cy = K[1, 2]

    # 픽셀 투영 (MORAI 컨벤션)
    d  = depth[valid]
    us = fx * (-corners_cam[valid, 1]) / d + cx   # -Y → 오른쪽
    vs = fx * (-corners_cam[valid, 2]) / d + cy   # -Z → 아래

    u_min, u_max = float(us.min()), float(us.max())
    v_min, v_max = float(vs.min()), float(vs.max())

    # 이미지 영역 클리핑
    u_min = max(0, u_min);  u_max = min(orig_w, u_max)
    v_min = max(0, v_min);  v_max = min(orig_h, v_max)

    if u_max - u_min < 2 or v_max - v_min < 2:
        return None
    if u_min >= orig_w or u_max <= 0 or v_min >= orig_h or v_max <= 0:
        return None

    return int(u_min), int(v_min), int(u_max), int(v_max)


# ===========================================================
# 추론 및 시각화
# ===========================================================
def run_inference(weights_path, stems, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[디바이스] {device}")

    # 모델 로드
    model = AutoNavModel().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f"[모델] {weights_path} 로드 완료\n")

    # frame_groups.json 로드
    groups_path = os.path.join(DATASET_DIR, 'frame_groups.json')
    with open(groups_path) as f:
        all_groups = json.load(f)
    stem_to_group = {g['label_stem']: g for g in all_groups}

    img_dir = os.path.join(DATASET_DIR, 'images')
    lbl_dir = os.path.join(DATASET_DIR, 'labels_3d')

    for stem in stems:
        if stem not in stem_to_group:
            print(f"[SKIP] {stem} → frame_groups에 없음")
            continue

        group = stem_to_group[stem]
        cams  = group['cams']

        # ── 이미지 텐서 구성 ──────────────────────────────
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

        # ── 추론 ──────────────────────────────────────────
        with torch.no_grad():
            pred_cls, pred_box = model(
                images.to(device),
                intrinsics.to(device),
                extrinsics.to(device)
            )

        scores     = pred_cls.softmax(-1)                    # [900, 3]
        max_scores, max_cls = scores.max(dim=-1)             # [900]
        keep       = max_scores > SCORE_THRESH
        boxes_keep = pred_box[keep].cpu().numpy()            # [M, 11]
        cls_keep   = max_cls[keep].cpu().numpy()
        scr_keep   = max_scores[keep].cpu().numpy()

        print(f"[{stem}] 예측 박스: {keep.sum().item()}개 "
              f"(임계값 {SCORE_THRESH} 이상)")

        # ── GT 박스 로드 ──────────────────────────────────
        gt_boxes_raw = []
        lbl_path = os.path.join(lbl_dir, f"{stem}.txt")
        if os.path.isfile(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 12:
                        gt_boxes_raw.append(list(map(float, parts[1:])))

        # ── 카메라별 시각화 ───────────────────────────────
        for cam_name in CAM_ORDER:
            if cam_name not in cams:
                continue

            path = os.path.join(img_dir, f"{cams[cam_name]}.jpg")
            img  = cv2.imread(path)
            if img is None:
                continue

            orig_h, orig_w = img.shape[:2]

            # GT 박스 (파란색)
            for gt_box in gt_boxes_raw:
                bbox = project_box_to_cam(gt_box, cam_name, orig_w, orig_h)
                if bbox:
                    cv2.rectangle(img, (bbox[0], bbox[1]),
                                  (bbox[2], bbox[3]), (255, 100, 0), 2)
                    cv2.putText(img, 'GT', (bbox[0]+2, bbox[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (255, 100, 0), 1, cv2.LINE_AA)

            # 예측 박스 (초록/주황/빨강)
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

            # 범례
            cv2.putText(img, f"GT=Blue  Pred=Green",
                        (5, orig_h-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (200, 200, 200), 1, cv2.LINE_AA)

            out_name = f"{stem}_{cam_name}_infer.jpg"
            cv2.imwrite(os.path.join(out_dir, out_name), img)

        print(f"   → {out_dir}/{stem}_*.jpg 저장 완료")

    print(f"\n✅ 전체 완료! 결과 확인:")
    print(f"   eog {out_dir}/*.jpg")


# ===========================================================
# 실행 진입점
# ===========================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D Detection 추론 시각화')
    parser.add_argument('--weights',  default='best_model.pth')
    parser.add_argument('--n',        type=int, default=10, help='랜덤 샘플 수')
    parser.add_argument('--stem',     default=None, help='특정 프레임 지정')
    parser.add_argument('--out',      default='./inference_results')
    args = parser.parse_args()

    if not os.path.isfile(args.weights):
        print(f"[ERROR] 가중치 파일 없음: {args.weights}")
        exit(1)

    # 프레임 선택
    if args.stem:
        stems = [args.stem]
    else:
        groups_path = os.path.join(DATASET_DIR, 'frame_groups.json')
        with open(groups_path) as f:
            all_groups = json.load(f)
        all_stems = [g['label_stem'] for g in all_groups]
        stems = random.sample(all_stems, min(args.n, len(all_stems)))
        print(f"[랜덤 샘플] {len(stems)}개 프레임 선택")

    run_inference(args.weights, stems, args.out)