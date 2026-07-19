#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visibility_contact_sheet.py
===========================
generate_visibility_gt.py 산출물(visibility/<stem>.npz)을 육안 검증하는 contact sheet.

각 프레임마다 [cam_front_left | cam_front | cam_front_right] 원본 이미지 위에
GT 3D box wireframe 을 그리고, object별 그 카메라의 visible_ratio 를 표기한다.
색상은 visibility_level:
   visible(3)=초록  partial(2)=노랑  heavily_occluded(1)=주황  invisible(0)=빨강

또한 이 clean bag 은 동적 상호가림이 거의 없어(단일 주행), 알고리즘이 "앞 객체가
뒤 객체를 가릴 때 뒤 객체 visible_ratio 가 떨어지는지"를 실이미지 위에서 보여주기
위해, cam_front 실이미지에 near/far 두 박스를 합성 투영한 _synthetic_occlusion.jpg 도
생성한다(명확히 SYNTHETIC 표기).

사용:
  python3 visibility_contact_sheet.py --scen /home/autonav/visibility_test_dataset/scen02 --n 20
"""

import os
import csv
import argparse

import cv2
import numpy as np

from camera_configs import INTRINSICS, CAM_ORDER
from verify_lidar_camera_overlay import project_body, _BOX_EDGES
import generate_occlusion_gt as OCC
import generate_visibility_gt as V

# 화면 배치 순서(좌->우 자연스러운 뷰)
VIEW_ORDER = ["cam_front_left", "cam_front", "cam_front_right"]
CELL_W, CELL_H = 640, 360

LEVEL_COLOR = {  # BGR
    3: (60, 200, 60),     # visible - green
    2: (0, 215, 235),     # partial - yellow
    1: (0, 140, 255),     # heavily_occluded - orange
    0: (40, 40, 235),     # invisible - red
}
LEVEL_TXT = {3: "VIS", 2: "PART", 1: "HOCC", 0: "INV"}


def load_vis_npz(path):
    d = np.load(path, allow_pickle=True)
    if "track_id" not in d or d["track_id"].size == 0:
        return None
    return {k: d[k] for k in d.files}


def draw_boxes_on_cam(img, boxes, vis, cam):
    """img(1600x900 BGR)에 boxes wireframe + visible_ratio 표기. vis=npz dict(row순 정렬)."""
    K = INTRINSICS[cam].astype(np.float64)
    cam_key = {"cam_front": "front", "cam_front_left": "left", "cam_front_right": "right"}[cam]
    ratio_arr = vis["%s_visible_ratio" % cam_key]
    proj_arr = vis["%s_projected_px" % cam_key]
    level_arr = vis["visibility_level"]
    tid_arr = vis["track_id"]
    npts_arr = vis["num_lidar_pts"]
    for i, b in enumerate(boxes):
        corners = V.box_corners_from_v2(b)
        ch = np.concatenate([corners, np.ones((8, 1))], axis=1)
        u, v, depth, valid = project_body(ch, cam, K, min_depth=V.MIN_DEPTH)
        inb = valid & (u > -200) & (u < 1800) & (v > -200) & (v < 1100)
        if int(inb.sum()) < 2 or int(proj_arr[i]) == 0:
            continue
        lvl = int(level_arr[i])
        color = LEVEL_COLOR[lvl]
        for a, c in _BOX_EDGES:
            if not (valid[a] and valid[c]):
                continue
            pa = (int(round(u[a])), int(round(v[a])))
            pb = (int(round(u[c])), int(round(v[c])))
            cv2.line(img, pa, pb, color, 2, cv2.LINE_AA)
        # 라벨 위치: 박스 상단 코너 근처
        ok = valid & (u > -1e4) & (u < 1e4)
        if not ok.any():
            continue
        lx = int(np.clip(u[ok].min(), 2, 1500))
        ly = int(np.clip(v[ok].min() - 6, 14, 890))
        txt = "t%d %s %.2f n%d" % (int(tid_arr[i]), LEVEL_TXT[lvl], float(ratio_arr[i]), int(npts_arr[i]))
        cv2.putText(img, txt, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, txt, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return img


def make_frame_row(scen_dir, stem, vis):
    boxes = OCC.read_boxes_v2(os.path.join(scen_dir, "labels_3d_v2", stem + ".csv"))
    cells = []
    for cam in VIEW_ORDER:
        p = os.path.join(scen_dir, "images", cam, stem + ".jpg")
        img = cv2.imread(p)
        if img is None:
            img = np.zeros((900, 1600, 3), np.uint8)
        img = draw_boxes_on_cam(img, boxes, vis, cam)
        cv2.putText(img, "%s %s" % (cam, stem), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cells.append(cv2.resize(img, (CELL_W, CELL_H)))
    return cv2.hconcat(cells)


def select_frames(scen_dir, n, must_include):
    vis_dir = os.path.join(scen_dir, "visibility")
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(vis_dir) if f.endswith(".npz"))
    stems = [s for s in stems if load_vis_npz(os.path.join(vis_dir, s + ".npz")) is not None]
    picks = list(must_include)
    # 균등 샘플로 채움
    for i in np.linspace(0, len(stems) - 1, num=max(n - len(picks), 1)):
        s = stems[int(round(i))]
        if s not in picks:
            picks.append(s)
    # 순서 정렬, n개로 컷
    picks = sorted(set(picks))[:max(n, len(must_include))]
    return picks


def synthetic_occlusion_panel(scen_dir, out_path):
    """cam_front 실이미지 위에 near/far 두 박스를 합성 투영: 앞 박스가 뒤 박스를 가림."""
    # 실제 cam_front 이미지 하나
    p = os.path.join(scen_dir, "images", "cam_front", "live_000000.jpg")
    img = cv2.imread(p)
    if img is None:
        img = np.zeros((900, 1600, 3), np.uint8)
    near = {"track_id": 100, "class_id": 0, "cx": 7.0, "cy": 0.0, "cz": 0.8,
            "w": 2.2, "l": 4.2, "h": 1.7, "yaw": 0.0}
    far = {"track_id": 101, "class_id": 0, "cx": 20.0, "cy": 0.0, "cz": 0.8,
           "w": 2.0, "l": 4.0, "h": 1.6, "yaw": 0.0}
    recs = V.compute_frame_visibility([near, far], downscale=2)
    fake_vis = {
        "track_id": np.array([100, 101]),
        "front_visible_ratio": np.array([recs[0]["front_visible_ratio"], recs[1]["front_visible_ratio"]], np.float32),
        "front_projected_px": np.array([recs[0]["front_projected_px"], recs[1]["front_projected_px"]]),
        "visibility_level": np.array([recs[0]["visibility_level"], recs[1]["visibility_level"]]),
        "num_lidar_pts": np.array([-1, -1]),
        "left_visible_ratio": np.zeros(2, np.float32), "left_projected_px": np.zeros(2),
        "right_visible_ratio": np.zeros(2, np.float32), "right_projected_px": np.zeros(2),
    }
    img = draw_boxes_on_cam(img, [near, far], fake_vis, "cam_front")
    cv2.putText(img, "SYNTHETIC: near(t100) occludes far(t101)  far ratio=%.2f near ratio=%.2f" % (
        recs[1]["front_visible_ratio"], recs[0]["front_visible_ratio"]),
        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, img)
    return recs[0]["front_visible_ratio"], recs[1]["front_visible_ratio"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scen", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--per_sheet", type=int, default=5)
    ap.add_argument("--must", default="live_000164", help="반드시 포함할 stem(쉼표구분)")
    args = ap.parse_args()

    out_dir = os.path.join(args.scen, "_visibility_contact")
    os.makedirs(out_dir, exist_ok=True)
    must = [s for s in args.must.split(",") if s]
    picks = select_frames(args.scen, args.n, must)
    print("[contact] frames=%d -> %s" % (len(picks), picks))

    rows = []
    for stem in picks:
        vis = load_vis_npz(os.path.join(args.scen, "visibility", stem + ".npz"))
        rows.append(make_frame_row(args.scen, stem, vis))

    sheet_i = 0
    for i in range(0, len(rows), args.per_sheet):
        chunk = rows[i:i + args.per_sheet]
        sheet = cv2.vconcat(chunk)
        outp = os.path.join(out_dir, "contact_%02d.jpg" % sheet_i)
        cv2.imwrite(outp, sheet)
        print("  saved %s (%d frames)" % (outp, len(chunk)))
        sheet_i += 1

    syn = os.path.join(out_dir, "_synthetic_occlusion.jpg")
    nr, fr = synthetic_occlusion_panel(args.scen, syn)
    print("  saved %s (near_ratio=%.3f far_ratio=%.3f)" % (syn, nr, fr))
    print("[contact] DONE -> %s" % out_dir)


if __name__ == "__main__":
    main()
