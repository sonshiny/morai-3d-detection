#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_occlusion_gt.py
========================
각 GT 3D 박스에 대해 "박스 안에 들어온 LiDAR 포인트 수(num_lidar_pts)"를 계산해
비파괴적으로 sidecar 로 저장한다. 이는 nuScenes/KITTI 의 num_lidar_pts 와 같은
표준 occlusion(가시성) 신호로, 건물/차량/사람에 완전히 가려진 객체는 박스 안에
LiDAR 점이 거의/전혀 없다 → confidence 학습 노이즈가 되는 amodal GT 를 걸러낼 근거.

기존 파일은 절대 수정하지 않는다(생성 스크립트/verify 재사용만).
  - verify_lidar_camera_overlay.lidar_to_body_h : LiDAR→body(ego) 프레임 변환 (검증된 경로)
  - morai_dataset._box_corners_ego              : 좌표 규약 확인용 (직접 사용 X)

신규 산출물만 생성:
  dataset/<scen>/occlusion/<stem>.npy
    shape (M, 3) float32 = [track_id, num_lidar_pts, radial_dist(m)]  (CSV 행 순서, 전체 행)
    → loader 는 track_id 로 매칭한다(가시성 필터로 행이 빠져도 안전).

박스 안 포인트 판정(body/ego 프레임, x=forward=l, y=left=w, z=up=h):
  q = p - center ;  yaw 역회전 후  |x_l|<=l/2+m & |y_l|<=w/2+m & |z_l|<=h/2+m
"""

import os
import csv
import argparse
import time

import numpy as np

# 기존 검증 경로 재사용 (수정 금지)
from verify_lidar_camera_overlay import lidar_to_body_h

MIN_LIDAR_RANGE = 1.5   # generate_depth_gt.py 와 동일: 자차 근처 반사 제거
DEFAULT_MARGIN = 0.10   # 표면에 맺힌 점을 포함하기 위한 소폭 확장(m)

_HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(_HERE, "dataset")


def load_lidar_body(path, min_range=MIN_LIDAR_RANGE):
    """LiDAR npy → body(ego) 프레임 (N,3). generate_depth_gt.load_lidar_xyz 와 동일한 range 필터."""
    cloud = np.load(path)
    if cloud.ndim != 2 or cloud.shape[1] < 3:
        raise ValueError(f"예상과 다른 라이다 shape: {cloud.shape} ({path})")
    xyz = cloud[:, :3].astype(np.float64)
    ranges = np.linalg.norm(xyz, axis=1)
    xyz = xyz[ranges >= float(min_range)]
    p_body = lidar_to_body_h(xyz)[:, :3]   # (N,3)
    return p_body


def read_boxes_v2(csv_path):
    """labels_3d_v2 CSV → 박스 리스트. z 는 z_center(박스 중심) 사용."""
    boxes = []
    if not os.path.isfile(csv_path):
        return boxes
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            boxes.append({
                "track_id": int(float(row["track_id"])),
                "class_id": int(float(row["class_id"])),
                "cx": float(row["x"]),
                "cy": float(row["y"]),
                "cz": float(row["z_center"]),
                "w": float(row["w"]),
                "l": float(row["l"]),
                "h": float(row["h"]),
                "yaw": float(row["yaw_ego"]),
            })
    return boxes


def count_points_in_boxes(p_body, boxes, margin=DEFAULT_MARGIN):
    """
    p_body : (N,3) body-frame LiDAR
    boxes  : read_boxes_v2 결과
    반환   : (M,3) float32 = [track_id, num_lidar_pts, radial_dist]
    """
    out = np.zeros((len(boxes), 3), dtype=np.float32)
    if p_body.shape[0] == 0:
        for i, b in enumerate(boxes):
            out[i] = (b["track_id"], 0, float(np.hypot(b["cx"], b["cy"])))
        return out

    px, py, pz = p_body[:, 0], p_body[:, 1], p_body[:, 2]
    for i, b in enumerate(boxes):
        qx = px - b["cx"]
        qy = py - b["cy"]
        qz = pz - b["cz"]
        c = np.cos(b["yaw"])
        s = np.sin(b["yaw"])
        # world→box-local: R(-yaw) 적용
        x_l = c * qx + s * qy
        y_l = -s * qx + c * qy
        hl = b["l"] * 0.5 + margin      # box x (length, forward)
        hw = b["w"] * 0.5 + margin      # box y (width, left)
        hh = b["h"] * 0.5 + margin      # box z (height)
        inside = (
            (np.abs(x_l) <= hl) &
            (np.abs(y_l) <= hw) &
            (np.abs(qz) <= hh)
        )
        out[i] = (b["track_id"], int(inside.sum()), float(np.hypot(b["cx"], b["cy"])))
    return out


def discover_scens(dataset_root, only=None):
    scens = []
    for name in sorted(
        os.listdir(dataset_root),
        key=lambda s: int(s[4:]) if s.startswith("scen") and s[4:].isdigit() else 10 ** 9,
    ):
        if not (name.startswith("scen") and name[4:].isdigit()):
            continue
        if only and name not in only:
            continue
        scen_dir = os.path.join(dataset_root, name)
        if os.path.isdir(os.path.join(scen_dir, "labels_3d_v2")) and \
           os.path.isdir(os.path.join(scen_dir, "lidar")):
            scens.append(name)
    return scens


def process_scen(scen_dir, margin=DEFAULT_MARGIN, write=True):
    lbl_dir = os.path.join(scen_dir, "labels_3d_v2")
    lidar_dir = os.path.join(scen_dir, "lidar")
    occ_dir = os.path.join(scen_dir, "occlusion")
    if write:
        os.makedirs(occ_dir, exist_ok=True)

    stems = sorted(
        os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith(".csv")
    )

    n_boxes = 0
    n_missing_lidar = 0
    # (num_lidar_pts, radial_dist) 누적 — 통계용
    npts_all = []
    dist_all = []
    for stem in stems:
        boxes = read_boxes_v2(os.path.join(lbl_dir, stem + ".csv"))
        lidar_path = os.path.join(lidar_dir, stem + ".npy")
        if not os.path.isfile(lidar_path):
            n_missing_lidar += 1
            # LiDAR 없으면 npts=-1(=미상)로 기록 → loader 가 필터하지 않도록.
            out = np.array(
                [[b["track_id"], -1, float(np.hypot(b["cx"], b["cy"]))] for b in boxes],
                dtype=np.float32,
            ).reshape(-1, 3)
        else:
            p_body = load_lidar_body(lidar_path)
            out = count_points_in_boxes(p_body, boxes, margin=margin)
            if out.shape[0]:
                npts_all.append(out[:, 1])
                dist_all.append(out[:, 2])
        n_boxes += len(boxes)
        if write:
            np.save(os.path.join(occ_dir, stem + ".npy"), out)

    npts_all = np.concatenate(npts_all) if npts_all else np.zeros((0,), np.float32)
    dist_all = np.concatenate(dist_all) if dist_all else np.zeros((0,), np.float32)
    return {
        "stems": len(stems),
        "boxes": n_boxes,
        "missing_lidar": n_missing_lidar,
        "npts": npts_all,
        "dist": dist_all,
    }


DIST_BUCKETS = ((0.0, 20.0), (20.0, 40.0), (40.0, 1e9))
THRESHOLDS = (1, 3, 5, 10)


def print_report(scen, st):
    npts = st["npts"]
    dist = st["dist"]
    print("=" * 78)
    print(f"  occlusion GT: {scen} | frames={st['stems']} boxes={st['boxes']} "
          f"missing_lidar_frames={st['missing_lidar']}")
    print("=" * 78)
    if npts.size == 0:
        print("  (LiDAR 매칭된 박스 없음)")
        return
    print(f"  num_lidar_pts: min={npts.min():.0f} median={np.median(npts):.0f} "
          f"mean={npts.mean():.1f} max={npts.max():.0f}")
    print(f"  전체 박스 중 npts==0 : {(npts == 0).sum()} / {npts.size} "
          f"({100.0 * (npts == 0).mean():.1f}%)")
    print("  거리구간 × threshold 별 '드롭될 박스 비율' (npts < thr):")
    header = "    dist        n" + "".join(f"   <{t:<4d}" for t in THRESHOLDS)
    print(header)
    for lo, hi in DIST_BUCKETS:
        m = (dist >= lo) & (dist < hi)
        n = int(m.sum())
        if n == 0:
            continue
        row = f"    [{lo:4.0f}-{hi if hi < 1e8 else 999:>3.0f})  {n:6d}"
        for t in THRESHOLDS:
            frac = 100.0 * ((npts[m] < t).mean())
            row += f"  {frac:5.1f}%"
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scen", default=None, help="단일 scen (예: scen08)")
    ap.add_argument("--scens", default=None, help="쉼표구분 목록 (예: scen08,scen18)")
    ap.add_argument("--all", action="store_true", help="labels_3d_v2+lidar 있는 모든 scen")
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    ap.add_argument("--dry-run", action="store_true", help="저장하지 않고 통계만")
    args = ap.parse_args()

    only = None
    if args.scens:
        only = set(args.scens.split(","))
    elif args.scen:
        only = {args.scen}

    scens = discover_scens(DATASET_ROOT, only=None if args.all else only)
    if not scens:
        raise SystemExit("[ERROR] 처리 가능한 scen 없음 (labels_3d_v2 + lidar 필요)")

    t0 = time.time()
    agg_npts, agg_dist = [], []
    tot_boxes = tot_frames = tot_missing = 0
    for i, scen in enumerate(scens, 1):
        scen_dir = os.path.join(DATASET_ROOT, scen)
        print(f"[occlusion] ({i}/{len(scens)}) {scen}")
        st = process_scen(scen_dir, margin=args.margin, write=not args.dry_run)
        print_report(scen, st)
        if st["npts"].size:
            agg_npts.append(st["npts"])
            agg_dist.append(st["dist"])
        tot_boxes += st["boxes"]
        tot_frames += st["stems"]
        tot_missing += st["missing_lidar"]

    if len(scens) > 1 and agg_npts:
        st = {
            "stems": tot_frames, "boxes": tot_boxes, "missing_lidar": tot_missing,
            "npts": np.concatenate(agg_npts), "dist": np.concatenate(agg_dist),
        }
        print()
        print_report(f"ALL({len(scens)} scen)", st)
    print(f"\n[occlusion] 완료: {len(scens)} scen, {tot_frames} frames, "
          f"{time.time() - t0:.1f}s, write={not args.dry_run}")


if __name__ == "__main__":
    main()
