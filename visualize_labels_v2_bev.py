#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_labels_v2_bev.py
==========================
전처리 산출물(labels_3d_v2)을 BEV(top-down)로 그려서 육안 검증하는 오프라인 뷰어.
기존 오프라인 도구들의 빈틈을 메운다:
  - morai_gt_bev_viewer.py 는 라이브(rospy) 전용이라 저장 파일을 못 본다.
  - verify_lidar_camera_overlay.py 는 원본 labels_3d(전처리 전)를 읽는다.
이 스크립트는 오직 labels_3d_v2 만 읽어, 전처리로 복구된 x,y,yaw_ego,vx,vy,track_id 를
BEV 로 렌더한다. occlusion/visibility sidecar 가 있으면 함께 겹쳐 가림 상태도 표시한다.
기존 파일은 읽기만 하고 수정하지 않는다.

표시 규약(ego frame: x=전방, y=좌):
  - 화면 위=전방(x+), 화면 왼쪽=좌(y+). ego 는 하단 중앙 삼각형.
  - 박스 색: vehicle=초록, pedestrian=주황.  단 num_lidar_pts==0 이면 빨강(완전가림 후보).
  - 박스에서 뻗는 선 = heading(yaw_ego), 노란 선 = 속도벡터(vx,vy).
  - 라벨: track_id / 거리 / npts(num_lidar_pts) / vis(visibility_level 0~3).

사용:
  python3 visualize_labels_v2_bev.py --scen scen02               # 무작위 12프레임
  python3 visualize_labels_v2_bev.py --scen scen02 --stems live_000000,live_000069
  python3 visualize_labels_v2_bev.py --scen scen02 --all         # 전 프레임
  python3 visualize_labels_v2_bev.py --scen scen02 --gif         # 추가로 애니메이션 gif
출력:  dataset/<scen>/bev_v2_vis/<stem>.png
"""
import os
import csv
import glob
import math
import argparse

import cv2
import numpy as np

# 검증된 투영 경로 재사용 (수정 금지)
from camera_configs import INTRINSICS, CAM_ORDER
from verify_lidar_camera_overlay import project_body, lidar_to_body_h, _BOX_EDGES

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(_HERE, "dataset")

ORIG_W, ORIG_H = 1600, 900

# BEV ROI (morai_3d_live 수집 범위와 동일)
X_MAX = 60.0            # 전방 m (화면 위)
Y_HALF = 30.0          # 좌우 m
PPM = 11               # pixel per meter
MARGIN = 40            # 상하좌우 여백(px)

CLASS_COLOR = {0: (60, 220, 60), 1: (0, 165, 255)}   # BGR: vehicle=green, ped=orange
CLASS_NAME = {0: "veh", 1: "ped"}


def _load_sidecar_occ(scen_dir, stem):
    """track_id -> num_lidar_pts"""
    p = os.path.join(scen_dir, "occlusion", stem + ".npy")
    if not os.path.isfile(p):
        return {}
    a = np.load(p)
    return {int(r[0]): int(r[1]) for r in a}


def _load_sidecar_vis(scen_dir, stem):
    """track_id -> visibility_level(0~3)"""
    p = os.path.join(scen_dir, "visibility", stem + ".npz")
    if not os.path.isfile(p):
        return {}
    z = np.load(p)
    if "track_id" not in z or "visibility_level" not in z:
        return {}
    return {int(t): int(v) for t, v in zip(z["track_id"], z["visibility_level"])}


def _read_boxes_v2(csv_path):
    out = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            out.append({
                "track_id": int(float(row["track_id"])),
                "class_id": int(float(row["class_id"])),
                "x": float(row["x"]), "y": float(row["y"]), "z": float(row["z_center"]),
                "w": float(row["w"]), "l": float(row["l"]), "h": float(row["h"]),
                "yaw": float(row["yaw_ego"]),
                "vx": float(row["vx_ego"]), "vy": float(row["vy_ego"]),
            })
    return out


def _corners_v2(b):
    """v2 박스 -> body/ego frame 8코너 동차좌표 (8,4). _BOX_EDGES 순서 규약과 일치.
    local: x=전방(l), y=좌(w), z=상(h). z 는 이미 중심(z_center)."""
    hl, hw, hh = b["l"] / 2, b["w"] / 2, b["h"] / 2
    local = np.array([
        [ hl,  hw,  hh], [ hl,  hw, -hh], [ hl, -hw,  hh], [ hl, -hw, -hh],
        [-hl,  hw,  hh], [-hl,  hw, -hh], [-hl, -hw,  hh], [-hl, -hw, -hh],
    ], np.float64)
    c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)
    world = local @ R.T + np.array([b["x"], b["y"], b["z"]], np.float64)
    return np.concatenate([world, np.ones((8, 1))], axis=1)


def render_cam(scen_dir, stem, camera="cam_front", draw_lidar=True):
    """전처리된 v2 3D 박스를 실제 카메라 사진에 투영해서 그린다 (verify 스타일)."""
    img_path = os.path.join(scen_dir, "images", camera, stem + ".jpg")
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    h, w = img.shape[:2]
    K = INTRINSICS[camera].astype(np.float64)

    # 라이다 (먼 점=파랑, 가까운 점=빨강)
    lidar_path = os.path.join(scen_dir, "lidar", stem + ".npy")
    if draw_lidar and os.path.isfile(lidar_path):
        cloud = np.load(lidar_path)
        if cloud.ndim == 2 and cloud.shape[1] >= 3:
            u, v, d, valid = project_body(lidar_to_body_h(cloud[:, :3]), camera, K)
            m = valid & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            uu, vv, dd = u[m], v[m], d[m]
            if dd.size:
                t = np.clip(dd / 40.0, 0, 1)          # 0~40m 정규화
                for i in np.argsort(-dd):             # 먼 점 먼저
                    col = (int(255 * t[i]), 0, int(255 * (1 - t[i])))  # BGR: 근=빨강 원=파랑
                    cv2.circle(img, (int(uu[i]), int(vv[i])), 1, col, -1)

    occ = _load_sidecar_occ(scen_dir, stem)
    vis = _load_sidecar_vis(scen_dir, stem)
    boxes = _read_boxes_v2(os.path.join(scen_dir, "labels_3d_v2", stem + ".csv"))

    drawn = 0
    for b in boxes:
        tid = b["track_id"]
        npts = occ.get(tid, None)
        occluded = (npts == 0)
        u, v, depth, valid = project_body(_corners_v2(b), camera, K)
        inframe = valid & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not inframe.any():
            continue
        color = (40, 40, 235) if occluded else CLASS_COLOR.get(b["class_id"], (200, 200, 200))
        for a, bb in _BOX_EDGES:
            if valid[a] and valid[bb]:
                cv2.line(img, (int(u[a]), int(v[a])), (int(u[bb]), int(v[bb])),
                         color, 2, cv2.LINE_AA)
        drawn += 1
        # 라벨 (박스 상단 근처)
        vu = int(u[inframe].mean()); vv2 = int(v[valid].min())
        vlvl = vis.get(tid, None)
        tag = f"#{tid} {CLASS_NAME.get(b['class_id'],'?')}"
        if npts is not None:
            tag += f" pts={npts}"
        if vlvl is not None:
            tag += f" vis={vlvl}"
        cv2.putText(img, tag, (max(0, vu - 40), max(14, vv2 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    hdr = f"[v2] {os.path.basename(scen_dir)} {stem} {camera}  boxes={drawn}  red=occluded(pts=0)"
    cv2.putText(img, hdr, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def _world_to_px(x, y):
    """ego frame(x 전방, y 좌) -> 이미지 픽셀. 위=전방, 왼쪽=좌."""
    col = MARGIN + int((Y_HALF - y) * PPM)      # y+(좌) -> 왼쪽
    rowpx = MARGIN + int((X_MAX - x) * PPM)     # x+(전방) -> 위
    return col, rowpx


def render_frame(scen_dir, stem):
    W = MARGIN * 2 + int(2 * Y_HALF * PPM)
    H = MARGIN * 2 + int(X_MAX * PPM)
    img = np.full((H, W, 3), 28, np.uint8)

    # --- 그리드 (10m 간격) ---
    for m in range(0, int(X_MAX) + 1, 10):
        _, ry = _world_to_px(m, 0)
        cv2.line(img, (MARGIN, ry), (W - MARGIN, ry), (55, 55, 55), 1)
        cv2.putText(img, f"{m}m", (4, ry + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    for m in range(-int(Y_HALF), int(Y_HALF) + 1, 10):
        cx, _ = _world_to_px(0, m)
        cv2.line(img, (cx, MARGIN), (cx, H - MARGIN), (55, 55, 55), 1)

    # --- ego (하단 중앙) ---
    ex, ey = _world_to_px(0, 0)
    cv2.drawMarker(img, (ex, ey), (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 16, 2)
    cv2.putText(img, "EGO", (ex - 16, ey + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    occ = _load_sidecar_occ(scen_dir, stem)
    vis = _load_sidecar_vis(scen_dir, stem)
    boxes = _read_boxes_v2(os.path.join(scen_dir, "labels_3d_v2", stem + ".csv"))

    for b in boxes:
        tid = b["track_id"]
        npts = occ.get(tid, None)
        vlvl = vis.get(tid, None)
        occluded = (npts == 0)
        color = (40, 40, 235) if occluded else CLASS_COLOR.get(b["class_id"], (200, 200, 200))

        # 회전 박스 4코너 (local: x=길이 l, y=폭 w)
        c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
        hl, hw = b["l"] / 2, b["w"] / 2
        local = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
        pts = []
        for lx, ly in local:
            wx = b["x"] + c * lx - s * ly
            wy = b["y"] + s * lx + c * ly
            pts.append(_world_to_px(wx, wy))
        pts = np.array(pts, np.int32)
        cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)

        # heading 선 (박스 앞면 방향)
        hx = b["x"] + c * hl
        hy = b["y"] + s * hl
        cv2.line(img, _world_to_px(b["x"], b["y"]), _world_to_px(hx, hy), color, 2, cv2.LINE_AA)

        # 속도 벡터 (노랑, 1초 예측 위치)
        vx1, vy1 = b["x"] + b["vx"], b["y"] + b["vy"]
        if abs(b["vx"]) + abs(b["vy"]) > 0.3:
            cv2.arrowedLine(img, _world_to_px(b["x"], b["y"]), _world_to_px(vx1, vy1),
                            (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.3)

        dist = math.hypot(b["x"], b["y"])
        cx, cy = _world_to_px(b["x"], b["y"])
        tag = f"#{tid} {CLASS_NAME.get(b['class_id'],'?')} {dist:.0f}m"
        if npts is not None:
            tag += f" pts={npts}"
        if vlvl is not None:
            tag += f" vis={vlvl}"
        cv2.putText(img, tag, (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (235, 235, 235), 1, cv2.LINE_AA)

    # 헤더
    n_occ = sum(1 for b in boxes if occ.get(b["track_id"], 1) == 0)
    hdr = f"{os.path.basename(scen_dir)}  {stem}  objs={len(boxes)}  occluded(pts=0)={n_occ}"
    cv2.putText(img, hdr, (MARGIN, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "red=fully occluded  yellow=velocity", (MARGIN, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--scen", required=True, help="시나리오명 (예: scen02)")
    ap.add_argument("--stems", default=None, help="쉼표구분 stem. 미지정시 --n 무작위 샘플")
    ap.add_argument("--n", type=int, default=12, help="무작위 샘플 프레임 수")
    ap.add_argument("--all", action="store_true", help="전 프레임 렌더")
    ap.add_argument("--gif", action="store_true", help="추가로 애니메이션 gif 저장")
    ap.add_argument("--view", default="cam", choices=["cam", "bev", "both"],
                    help="cam=사진에 3D박스 투영(기본), bev=탑뷰, both=나란히")
    ap.add_argument("--camera", default="cam_front", choices=CAM_ORDER, help="cam 뷰 대상 카메라")
    ap.add_argument("--no-lidar", action="store_true", help="cam 뷰에서 라이다 점 생략")
    args = ap.parse_args()

    scen_dir = os.path.join(args.root, args.scen)
    lbl_dir = os.path.join(scen_dir, "labels_3d_v2")
    if not os.path.isdir(lbl_dir):
        raise SystemExit(f"[ERROR] labels_3d_v2 없음: {lbl_dir} (먼저 preprocess_dataset.py 실행)")

    all_stems = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(lbl_dir, "*.csv")))
    if args.stems:
        stems = [s.strip() for s in args.stems.split(",")]
    elif args.all:
        stems = all_stems
    else:
        idx = np.linspace(0, len(all_stems) - 1, min(args.n, len(all_stems))).astype(int)
        stems = [all_stems[i] for i in idx]

    out_dir = os.path.join(scen_dir, {"cam": "cam_v2_vis", "bev": "bev_v2_vis", "both": "cam_bev_v2_vis"}[args.view])
    os.makedirs(out_dir, exist_ok=True)

    def make(stem):
        if args.view == "bev":
            return render_frame(scen_dir, stem)
        cam = render_cam(scen_dir, stem, args.camera, draw_lidar=not args.no_lidar)
        if args.view == "cam":
            return cam
        bev = render_frame(scen_dir, stem)                       # both: 사진 옆에 BEV
        bev = cv2.resize(bev, (cam.shape[0], cam.shape[0]))
        return np.concatenate([cam, bev], axis=1)

    frames = []
    for stem in stems:
        img = make(stem)
        outp = os.path.join(out_dir, stem + ".png")
        cv2.imwrite(outp, img)
        frames.append(img)
        print("saved", outp)

    if args.gif and frames:
        try:
            import imageio
            gif = os.path.join(out_dir, "bev_v2.gif")
            imageio.mimsave(gif, [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], fps=8)
            print("saved", gif)
        except ImportError:
            print("[warn] imageio 없음 -> gif 생략 (pip install imageio)")

    print(f"\n총 {len(stems)}프레임 -> {out_dir}")


if __name__ == "__main__":
    main()
