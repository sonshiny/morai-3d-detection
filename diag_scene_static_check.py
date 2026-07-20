#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scen17 센서 장면 정적 여부 확정: cam_front pixel-diff + 주행구간 LiDAR shift."""

import os
import csv
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCEN = os.path.join(HERE, "dataset", "scen17")
IMG = os.path.join(SCEN, "images", "cam_front")


def load(stem):
    im = cv2.imread(os.path.join(IMG, stem + ".jpg"), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(im, (400, 225)).astype(np.float32)


def diff(a, b):
    return float(np.abs(load(a) - load(b)).mean())


def main():
    pairs = [
        ("live_000020", "live_000021", "seg1 연속"),
        ("live_000020", "live_000050", "seg1 vs seg2 (텔레포트 전후)"),
        ("live_000050", "live_000190", "seg2 처음 vs 끝"),
        ("live_000050", "live_000230", "seg2 vs seg3(주행중)"),
        ("live_000210", "live_000213", "seg3 주행 연속(3프레임)"),
        ("live_000230", "live_000240", "seg3 주행 10프레임"),
        ("live_000000", "live_000250", "처음 vs 마지막"),
    ]
    print("cam_front 평균 |pixel diff| (0=동일):")
    for a, b, tag in pairs:
        print(f"  {a} vs {b} [{tag:26s}]: {diff(a, b):7.2f}")

    # 주행 구간 LiDAR 이동량
    def scene_shift(sa, sb, max_r=35.0):
        pa = np.load(os.path.join(SCEN, "lidar", sa + ".npy"))[:, :3]
        pb = np.load(os.path.join(SCEN, "lidar", sb + ".npy"))[:, :3]

        def grid(p):
            m = (np.abs(p[:, 0]) < max_r) & (np.abs(p[:, 1]) < max_r) & (p[:, 2] > -1.0)
            q = p[m]
            g = np.zeros((140, 140), dtype=np.float32)
            ij = ((q[:, :2] + max_r) / 0.5).astype(int)
            ok = (ij[:, 0] >= 0) & (ij[:, 0] < 140) & (ij[:, 1] >= 0) & (ij[:, 1] < 140)
            np.add.at(g, (ij[ok, 0], ij[ok, 1]), 1.0)
            return np.minimum(g, 5.0)

        ga, gb = grid(pa), grid(pb)
        best = (0, 0, -1.0)
        for dx in range(-14, 15):
            for dy in range(-14, 15):
                a = ga[max(0, dx):140 + min(0, dx), max(0, dy):140 + min(0, dy)]
                b = gb[max(0, -dx):140 - max(0, dx), max(0, -dy):140 - max(0, dy)]
                sc = float((a * b).sum())
                if sc > best[2]:
                    best = (dx, dy, sc)
        return best[0] * 0.5, best[1] * 0.5

    print("\n주행 구간(f205~250) LiDAR 장면 이동 vs ego pose 이동:")
    ego = {}
    for f in sorted(os.listdir(os.path.join(SCEN, "ego_pose"))):
        with open(os.path.join(SCEN, "ego_pose", f)) as fh:
            r = list(csv.DictReader(fh))[0]
        ego[os.path.splitext(f)[0]] = (float(r["ego_x"]), float(r["ego_y"]))
    for i in range(205, 248, 6):
        a, b = f"live_{i:06d}", f"live_{i + 6:06d}"
        if a not in ego or b not in ego:
            continue
        sx, sy = scene_shift(a, b)
        pd = np.hypot(ego[b][0] - ego[a][0], ego[b][1] - ego[a][1])
        print(f"  {a}->{b}: LiDAR |shift|={np.hypot(sx, sy):5.1f}m  pose 이동={pd:5.2f}m")

    # LiDAR 자체도 프레임 간 동일한지 (byte-level)
    import hashlib
    print("\nLiDAR npy 파일 해시 (동일 파일 반복 여부):")
    for i in (20, 50, 120, 210, 230, 245):
        p = os.path.join(SCEN, "lidar", f"live_{i:06d}.npy")
        if os.path.isfile(p):
            h = hashlib.md5(open(p, "rb").read()).hexdigest()[:10]
            n = np.load(p).shape[0]
            print(f"  live_{i:06d}: md5={h} npts={n}")


if __name__ == "__main__":
    main()
