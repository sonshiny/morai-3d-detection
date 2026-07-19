#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diag_motion_timeline.py — scen17 센서 vs pose 모션 타임라인 대조

센서 모션: (a) cam_front 연속 프레임 pixel-diff
          (b) LiDAR 연속 프레임 2D 정합 이동량 (지면·근접 링 제거 후)
pose 모션: ego_pose 연속 프레임 이동량
셋을 프레임축에 나란히 출력 → 어긋나는 구간/시간 오프셋 확정.
"""

import os
import csv
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCEN = os.path.join(HERE, "dataset", "scen17")
IMG = os.path.join(SCEN, "images", "cam_front")
N = 251


def imgs_gray(stem):
    im = cv2.imread(os.path.join(IMG, stem + ".jpg"), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    return cv2.resize(im, (400, 225)).astype(np.float32)


def lidar_grid(stem, max_r=35.0):
    p = np.load(os.path.join(SCEN, "lidar", stem + ".npy"))[:, :3]
    r = np.hypot(p[:, 0], p[:, 1])
    # 근접 링/지면 제거: 반경 6~35m, z(라이다 기준) > -1.0 (지면 ~ -1.9)
    m = (r > 6.0) & (r < max_r) & (p[:, 2] > -1.0) & (p[:, 2] < 3.0)
    q = p[m]
    g = np.zeros((140, 140), dtype=np.float32)
    ij = ((q[:, :2] + max_r) / 0.5).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < 140) & (ij[:, 1] >= 0) & (ij[:, 1] < 140)
    np.add.at(g, (ij[ok, 0], ij[ok, 1]), 1.0)
    return np.minimum(g, 3.0)


def grid_shift(ga, gb, max_cells=16):
    best = (0, 0, -1.0)
    for dx in range(-max_cells, max_cells + 1):
        for dy in range(-max_cells, max_cells + 1):
            a = ga[max(0, dx):140 + min(0, dx), max(0, dy):140 + min(0, dy)]
            b = gb[max(0, -dx):140 - max(0, dx), max(0, -dy):140 - max(0, dy)]
            sc = float((a * b).sum())
            if sc > best[2]:
                best = (dx, dy, sc)
    # 정규화 매칭 품질: 자기상관 대비
    self_sc = float((ga * ga).sum()) ** 0.5 * float((gb * gb).sum()) ** 0.5
    q = best[2] / max(self_sc, 1e-6)
    return best[0] * 0.5, best[1] * 0.5, q


def main():
    # pose
    ego = []
    for i in range(N):
        p = os.path.join(SCEN, "ego_pose", f"live_{i:06d}.csv")
        with open(p) as fh:
            r = list(csv.DictReader(fh))[0]
        ego.append((float(r["timestamp"]), float(r["ego_x"]), float(r["ego_y"]),
                    float(r["ego_heading_deg"])))

    print(f"{'f':>4s} {'dt(s)':>6s} {'pose_d(m)':>9s} {'pose_hdg':>8s} "
          f"{'img_diff':>8s} {'lidar_d(m)':>10s} {'q':>5s}")
    prev_g = imgs_gray("live_000000")
    prev_l = lidar_grid("live_000000")
    out = []
    for i in range(1, N):
        a, b = f"live_{i-1:06d}", f"live_{i:06d}"
        g = imgs_gray(b)
        idf = float(np.abs(g - prev_g).mean()) if g is not None and prev_g is not None else np.nan
        lg = lidar_grid(b)
        sx, sy, q = grid_shift(prev_l, lg)
        ld = float(np.hypot(sx, sy))
        pd = float(np.hypot(ego[i][1] - ego[i-1][1], ego[i][2] - ego[i-1][2]))
        dt = ego[i][0] - ego[i-1][0]
        out.append((i, dt, pd, ego[i][3], idf, ld, q))
        prev_g, prev_l = g, lg

    for (i, dt, pd, hdg, idf, ld, q) in out:
        flag = ""
        if pd > 0.5 or ld > 0.5 or idf > 6.0:
            flag = "  <-- 모션"
        if i % 2 == 0 or flag:
            print(f"{i:4d} {dt:6.2f} {pd:9.2f} {hdg:8.2f} {idf:8.1f} {ld:10.1f} {q:5.2f}{flag}")

    # 요약: 센서 모션 시작/종료 프레임 vs pose 모션 프레임
    img_mov = [i for (i, dt, pd, h, idf, ld, q) in out if idf > 6.0]
    lid_mov = [i for (i, dt, pd, h, idf, ld, q) in out if ld >= 1.0]
    pos_mov = [i for (i, dt, pd, h, idf, ld, q) in out if pd > 0.5]
    print("\n요약:")
    print(f"  이미지 diff>6 프레임: {img_mov[:5]}...{img_mov[-5:] if len(img_mov)>5 else ''} (총 {len(img_mov)})")
    print(f"  LiDAR 이동>=1.0m 프레임: {lid_mov[:5]}...{lid_mov[-5:] if len(lid_mov)>5 else ''} (총 {len(lid_mov)})")
    print(f"  pose 이동>0.5m 프레임: {pos_mov[:5]}...{pos_mov[-5:] if len(pos_mov)>5 else ''} (총 {len(pos_mov)})")


if __name__ == "__main__":
    main()
