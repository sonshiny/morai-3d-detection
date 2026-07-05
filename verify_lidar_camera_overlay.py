#!/usr/bin/env python3
"""
verify_lidar_camera_overlay.py
==============================
라이다-카메라 좌표계 정합 육안 검증용 스크립트 (신규, 기존 파일 무수정).

지정 카메라(cam_front / cam_front_left / cam_front_right)의 EXTRINSICS·INTRINSICS로
scen54 라이다 포인트클라우드를 같은 stem의 카메라 이미지 위에 투영해, 점이
도로·차량 위에 정확히 얹히는지 확인하는 오버레이 PNG를 만든다.

[투영 파이프라인 — morai_dataset.box_visible_in_any_camera 와 "동일"]
새 변환 규약을 만들지 않는다. 이 코드베이스의 카메라 프레임은 표준 OpenCV(Z=depth)가
아니라  X=전방(depth), Y=좌, Z=상  이다 (camera_configs.py 주석 참조).

  box_visible_in_any_camera (morai_dataset.py:72-80) 픽셀 투영:
    pts   = E @ p_body                # camera-frame 4-vector
    depth = pts[0]                    # depth = X_cam  (Z 아님!)
    유효  = depth > min_depth         # 카메라 뒤/근접 컷은 X 기준
    u     = fx * (-Y_cam) / depth + cx
    v     = fy * (-Z_cam) / depth + cy

  → 본 스크립트의 project_body()가 이 식을 그대로 재사용한다. 라이다 점과 GT 박스
    코너 모두 같은 project_body()를 통과한다(경로 일원화).

[카메라 선택 — 요구사항 1]
  --camera 로 cam_front / cam_front_left / cam_front_right 중 선택.
  해당 카메라의 EXTRINSICS[camera]·INTRINSICS[camera] 를 그대로 쓴다.
  cam_front_left 는 yaw=+45(좌향) extrinsic. 회전 행렬은 camera_configs.py 가 이미
  계산해 둔 것을 import 만 하므로, 여기서 yaw 부호/euler 순서를 새로 다루지 않는다.

[차량 프레임 자동 선택 — 요구사항 2]
  labels_3d CSV 에서 class_id==0(vehicle) 박스를 찾아, 그 박스가 대상 카메라에
  in-frame 으로 투영되는 프레임 중 "가장 가까운(center depth 최소)" 프레임을 고른다.
  (GT 박스 코너 계산은 morai_dataset._box_corners_ego 를 import 재사용)

[의도적 선택 — 지시문 요구사항 근거]
  1) 뒤쪽 컷은 depth(=X_cam) <= min_depth. 표준 OpenCV 의 z<=0 이 아님.
  2) 원본 1600x900 jpg 위에 그리므로 스케일하지 않은 원본 INTRINSICS[camera] 사용
     (box_visible_in_any_camera 는 704x256 모델입력용 스케일 K를 쓰지만 그건 학습 좌표계).
"""

import os
import csv
import sys
import argparse

import cv2
import numpy as np

# box_visible_in_any_camera 와 "동일한" 파라미터/헬퍼 소스에서 import.
# (투영 행렬·박스 코너 로직을 새로 만들지 않는다 — 요구사항)
from camera_configs import INTRINSICS, EXTRINSICS, LIDAR_TO_BODY, CAM_ORDER
from morai_dataset import _box_corners_ego

# ------------------------------------------------------------------
# 상수
# ------------------------------------------------------------------
MIN_DEPTH = 0.1          # box_visible_in_any_camera 기본값과 동일한 뒤쪽/근접 컷
ORIG_W = 1600            # 원본 이미지 해상도(스케일 안 한 K가 가정하는 좌표계)
ORIG_H = 900
VEHICLE_CLASS_ID = 0     # class0 = vehicle

_HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(_HERE, "dataset")

_PREFERRED_OUT = "/mnt/user-data/outputs"
OUT_DIR = _PREFERRED_OUT if os.path.isdir(_PREFERRED_OUT) else os.path.join(_HERE, "verify_output")
os.makedirs(OUT_DIR, exist_ok=True)

# GT 박스 wireframe 엣지 (_box_corners_ego 코너 순서 기준)
#  0:[+l,+w,+h] 1:[+l,+w,-h] 2:[+l,-w,+h] 3:[+l,-w,-h]
#  4:[-l,+w,+h] 5:[-l,+w,-h] 6:[-l,-w,+h] 7:[-l,-w,-h]  8:center
_BOX_EDGES = [
    (0, 1), (1, 3), (3, 2), (2, 0),   # front face (x=+l/2)
    (4, 5), (5, 7), (7, 6), (6, 4),   # back face  (x=-l/2)
    (0, 4), (1, 5), (2, 6), (3, 7),   # connectors
]


def depth_to_bgr(depth, near=0.0, far=50.0):
    """가까우면 빨강, 멀면 파랑 (near=RED, far=BLUE). OpenCV BGR 반환."""
    t = np.clip((depth - near) / max(far - near, 1e-6), 0.0, 1.0)
    b = (255.0 * t).astype(np.uint8)
    r = (255.0 * (1.0 - t)).astype(np.uint8)
    g = np.zeros_like(b)
    return np.stack([b, g, r], axis=1)


def project_body(p_body_h, camera, K, min_depth=MIN_DEPTH):
    """
    body-frame 동차좌표 (N,4) -> 픽셀 (u,v), depth, valid.
    box_visible_in_any_camera 와 완전히 동일한 식/부호 규약.
    """
    E = EXTRINSICS[camera].astype(np.float64)
    p_cam = (E @ p_body_h.T).T                 # (N,4) camera-frame
    depth = p_cam[:, 0]                         # depth = X_cam
    y_cam = p_cam[:, 1]
    z_cam = p_cam[:, 2]
    u = K[0, 0] * (-y_cam) / (depth + 1e-6) + K[0, 2]
    v = K[1, 1] * (-z_cam) / (depth + 1e-6) + K[1, 2]
    valid = depth > min_depth
    return u, v, depth, valid


def lidar_to_body_h(xyz):
    """라이다 프레임 (N,3) -> body 프레임 동차좌표 (N,4). LIDAR_TO_BODY 재사용."""
    n = xyz.shape[0]
    p_lidar_h = np.concatenate([xyz.astype(np.float64), np.ones((n, 1))], axis=1)
    p_body = (LIDAR_TO_BODY.astype(np.float64) @ p_lidar_h.T).T
    return p_body  # (N,4)


def read_vehicle_boxes(csv_path):
    """CSV에서 class0(vehicle) 박스 vals 리스트 반환. _load_labels 와 동일 컬럼 사용."""
    boxes = []
    if not os.path.isfile(csv_path):
        return boxes
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(float(row["class_id"])) != VEHICLE_CLASS_ID:
                continue
            boxes.append([
                float(row["x"]), float(row["y"]), float(row["z"]),
                float(row["ln_w"]), float(row["ln_l"]), float(row["ln_h"]),
                float(row["sin_yaw"]), float(row["cos_yaw"]),
                float(row["vx"]), float(row["vy"]), float(row["vz"]),
            ])
    return boxes


def sensors_exist(scen_dir, stem, camera):
    return (
        os.path.exists(os.path.join(scen_dir, "lidar", stem + ".npy")) and
        os.path.exists(os.path.join(scen_dir, "images", camera, stem + ".jpg"))
    )


def select_vehicle_frame(scen_dir, camera, K, min_depth=MIN_DEPTH):
    """
    class0 차량이 대상 카메라에 in-frame 투영되는 프레임 중 가장 가까운 것 선택.
    반환: (stem, best_boxes, center_depth) 또는 (None, None, None).
    """
    lbl_dir = os.path.join(scen_dir, "labels_3d")
    if not os.path.isdir(lbl_dir):
        return None, None, None
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith(".csv"))

    best = None  # (center_depth, stem)
    fallback = None
    for stem in stems:
        if not sensors_exist(scen_dir, stem, camera):
            continue
        boxes = read_vehicle_boxes(os.path.join(lbl_dir, f"{stem}.csv"))
        if not boxes:
            continue
        if fallback is None:
            fallback = stem
        for vals in boxes:
            corners = _box_corners_ego(vals)                       # (9,3) body/ego frame
            corners_h = np.concatenate(
                [corners.astype(np.float64), np.ones((corners.shape[0], 1))], axis=1)
            u, v, depth, valid = project_body(corners_h, camera, K, min_depth)
            inframe = valid & (u >= 0) & (u < ORIG_W) & (v >= 0) & (v < ORIG_H)
            if not inframe.any():
                continue
            center_depth = float(depth[8])  # 코너 index 8 = center
            if center_depth <= min_depth:
                continue
            if best is None or center_depth < best[0]:
                best = (center_depth, stem)

    if best is not None:
        stem = best[1]
        return stem, read_vehicle_boxes(os.path.join(lbl_dir, f"{stem}.csv")), best[0]
    if fallback is not None:
        return fallback, read_vehicle_boxes(os.path.join(lbl_dir, f"{fallback}.csv")), None
    return None, None, None


def draw_gt_boxes(img, boxes, camera, K, min_depth=MIN_DEPTH):
    """GT 차량 박스를 초록 wireframe 으로 그림. project_body 재사용. 그린 박스 수 반환."""
    drawn = 0
    for vals in boxes:
        corners = _box_corners_ego(vals)
        corners_h = np.concatenate(
            [corners.astype(np.float64), np.ones((corners.shape[0], 1))], axis=1)
        u, v, depth, valid = project_body(corners_h, camera, K, min_depth)
        # 대상 카메라에 안 보이면 스킵
        inframe = valid & (u >= 0) & (u < img.shape[1]) & (v >= 0) & (v < img.shape[0])
        if not inframe.any():
            continue
        for a, b in _BOX_EDGES:
            if not (valid[a] and valid[b]):
                continue
            pa = (int(round(u[a])), int(round(v[a])))
            pb = (int(round(u[b])), int(round(v[b])))
            cv2.line(img, pa, pb, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        drawn += 1
    return drawn


def process(scen_dir, camera, stem, out_dir, min_depth=MIN_DEPTH, gt_boxes=None):
    """한 (camera, stem) 조합을 투영·오버레이·저장·리포트."""
    lidar_path = os.path.join(scen_dir, "lidar", stem + ".npy")
    image_path = os.path.join(scen_dir, "images", camera, stem + ".jpg")
    for p in (lidar_path, image_path):
        if not os.path.exists(p):
            print(f"[ERROR] 입력 없음: {p}")
            return False

    cloud = np.load(lidar_path)
    if cloud.ndim != 2 or cloud.shape[1] < 3:
        print(f"[ERROR] 예상과 다른 라이다 shape: {cloud.shape}")
        return False
    xyz = cloud[:, :3]
    n_total = xyz.shape[0]

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] 이미지 디코드 실패: {image_path}")
        return False
    h, w = img.shape[:2]
    if (w, h) != (ORIG_W, ORIG_H):
        print(f"[WARN] 이미지 {w}x{h} != 원본 가정 {ORIG_W}x{ORIG_H}. 원본 K 어긋날 수 있음.")

    # 원본 K (스케일 안 함 — 원본 jpg 위에 그림)
    K = INTRINSICS[camera].astype(np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # --- 라이다 투영 (box_visible 와 동일 경로) ---
    p_body = lidar_to_body_h(xyz)
    u, v, depth, valid = project_body(p_body, camera, K, min_depth)
    n_behind = int((~valid).sum())

    u_v, v_v, d_v = u[valid], v[valid], depth[valid]
    in_frame = (u_v >= 0) & (u_v < w) & (v_v >= 0) & (v_v < h)
    n_in = int(in_frame.sum())
    u_in, v_in, d_in = u_v[in_frame], v_v[in_frame], d_v[in_frame]

    # --- 오버레이 ---
    overlay = img.copy()
    if n_in > 0:
        order = np.argsort(-d_in)  # 먼 점 먼저 그려 가까운 점이 위로
        uu = u_in[order].astype(np.int32)
        vv = v_in[order].astype(np.int32)
        colors = depth_to_bgr(d_in[order])
        for pu, pv, col in zip(uu, vv, colors):
            cv2.circle(overlay, (int(pu), int(pv)), 2,
                       (int(col[0]), int(col[1]), int(col[2])), -1, lineType=cv2.LINE_AA)

    blended = cv2.addWeighted(overlay, 0.85, img, 0.15, 0.0)

    # GT 차량 박스(초록) — 라이다가 차량 위에 얹히는지 대조용
    n_gt_drawn = 0
    if gt_boxes:
        n_gt_drawn = draw_gt_boxes(blended, gt_boxes, camera, K, min_depth)

    cv2.putText(blended, f"{camera}  {stem}  near=RED far=BLUE (depth=X_cam)",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(blended, f"in-frame={n_in}/{n_total}  GTbox(green)={n_gt_drawn}",
                (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    out_path = os.path.join(out_dir, f"lidar_overlay_{camera}_{stem}.png")
    cv2.imwrite(out_path, blended)

    # --- 콘솔 리포트 ---
    print("=" * 70)
    print(f"  LiDAR -> {camera} 투영 검증   (stem={stem})")
    print("=" * 70)
    print(f"라이다   : {lidar_path}")
    print(f"이미지   : {image_path}  ({w}x{h})")
    print(f"K(원본)  : fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}  (스케일 안 함)")
    print(f"규약     : depth=X_cam,  u=fx*(-Y/X)+cx,  v=fy*(-Z/X)+cy,  컷 X>{min_depth}")
    print("-" * 70)
    print(f"전체 포인트 수                  : {n_total}")
    print(f"카메라 뒤/근접(X<=min_depth) 제거 : {n_behind}")
    print(f"프레임({w}x{h}) 안 투영 포인트     : {n_in}")
    print("-" * 70)
    if n_in > 0:
        u_ok = (0 <= u_in.min()) and (u_in.max() < w)
        v_ok = (0 <= v_in.min()) and (v_in.max() < h)
        print("투영 포인트 픽셀/깊이 범위 (in-frame):")
        print(f"  u    : min={u_in.min():8.2f}  max={u_in.max():8.2f}   (0~{w})")
        print(f"  v    : min={v_in.min():8.2f}  max={v_in.max():8.2f}   (0~{h})")
        print(f"  depth: min={d_in.min():7.2f}m max={d_in.max():7.2f}m  (X_cam)")
        print(f"  → u,v 이미지 범위 내 분포: {'OK' if (u_ok and v_ok) else 'CHECK'}")
    else:
        print("[WARN] 프레임 안 투영 포인트 0개. 좌표 규약/축 부호 재점검 필요.")
    if gt_boxes is not None:
        print(f"GT 차량 박스(class0)            : {len(gt_boxes)}개 중 {n_gt_drawn}개 in-frame 렌더")
    print("-" * 70)
    print(f"오버레이 PNG 저장: {out_path}")
    print("=" * 70)
    print()
    return True


def main():
    ap = argparse.ArgumentParser(description="LiDAR->카메라 투영 정합 검증")
    ap.add_argument("--camera", default="cam_front", choices=CAM_ORDER,
                    help="투영 대상 카메라 (기본 cam_front)")
    ap.add_argument("--scen", default="scen54", help="시나리오 폴더명 (기본 scen54)")
    ap.add_argument("--stem", default=None,
                    help="검증할 프레임 stem. 미지정 시 class0 차량이 가장 가깝게 보이는 프레임 자동 선택")
    ap.add_argument("--min-depth", type=float, default=MIN_DEPTH,
                    help=f"카메라 뒤/근접 컷 임계 (기본 {MIN_DEPTH})")
    args = ap.parse_args()

    scen_dir = os.path.join(DATASET_ROOT, args.scen)
    if not os.path.isdir(scen_dir):
        print(f"[ERROR] 시나리오 폴더 없음: {scen_dir}")
        sys.exit(1)

    K = INTRINSICS[args.camera].astype(np.float64)

    # 프레임 선택
    if args.stem is not None:
        stem = args.stem
        gt_boxes = read_vehicle_boxes(os.path.join(scen_dir, "labels_3d", f"{stem}.csv"))
        print(f"[frame] 지정 stem 사용: {stem} (class0 차량 {len(gt_boxes)}개)")
    else:
        stem, gt_boxes, cdep = select_vehicle_frame(scen_dir, args.camera, K, args.min_depth)
        if stem is None:
            print(f"[ERROR] {args.scen} 에서 class0 차량 프레임을 찾지 못함.")
            sys.exit(1)
        if cdep is not None:
            print(f"[frame] {args.camera} 에 차량이 가장 가깝게 보이는 프레임 자동 선택: "
                  f"{stem} (차량 center depth={cdep:.2f}m, class0 {len(gt_boxes)}개)")
        else:
            print(f"[frame] in-frame 차량 프레임 없어 fallback 선택: {stem}")
    print()

    ok = process(scen_dir, args.camera, stem, OUT_DIR, args.min_depth, gt_boxes)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
