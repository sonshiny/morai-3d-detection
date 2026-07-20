#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diag_world_to_ego_bug2.py — 가설별 GT 회전보정 vs LiDAR 점유 채점 (전 프레임)

gt = R(theta_true - theta_code) @ true  (dp는 heading 무관하게 옳으므로 순수 회전 오류)
→ true = R(theta_code - theta_true) @ gt,  theta_code = ego_heading_deg

각 가설 f(h)=theta_true 에 대해 true 위치를 복원하고, 그 자리(반경 2.3m,
지면+0.3~+2.4m 높이 밴드)에 LiDAR 점이 실제로 있는지 센다.
정답 가설만 모든 ego heading에서 일관되게 점유율이 높아야 한다.
"""

import os
import csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCEN = os.path.join(HERE, "dataset", "scen17")

HYPS = {
    "H0 theta=h (현행)":   lambda h: h,
    "H1 theta=-h":         lambda h: -h,
    "H2 theta=h+90":       lambda h: h + 90.0,
    "H3 theta=h-90":       lambda h: h - 90.0,
    "H4 theta=90-h (컴퍼스)": lambda h: 90.0 - h,
    "H5 theta=h+180":      lambda h: h + 180.0,
}

RADIUS = 2.3
MIN_PTS = 15


def read_ego(stem):
    with open(os.path.join(SCEN, "ego_pose", stem + ".csv")) as f:
        row = list(csv.DictReader(f))[0]
    return float(row["ego_heading_deg"])


def read_objs(stem):
    out = []
    with open(os.path.join(SCEN, "labels_3d", stem + ".csv")) as f:
        for row in csv.DictReader(f):
            out.append((float(row["x"]), float(row["y"]),
                        int(float(row["class_id"]))))
    return out


def ground_z_of(pts):
    """근거리(<15m) 낮은 z 점들의 최빈 z = 지면. z 히스토그램에서
    '가장 낮은 유의미한 피크'를 취한다(벽/건물 피크 배제)."""
    near = pts[np.hypot(pts[:, 0], pts[:, 1]) < 15.0]
    if len(near) < 100:
        near = pts
    zs = near[:, 2]
    hist, edges = np.histogram(zs, bins=120)
    thresh = max(hist.max() * 0.25, 30)
    for i in range(len(hist)):
        if hist[i] >= thresh:
            return 0.5 * (edges[i] + edges[i + 1])
    return np.percentile(zs, 5)


def main():
    stems = sorted(os.path.splitext(f)[0]
                   for f in os.listdir(os.path.join(SCEN, "labels_3d"))
                   if f.endswith(".csv"))

    score = {k: dict(hit=0, tot=0, pts=[]) for k in HYPS}
    headings = []
    gz_samples = []

    for stem in stems:
        lp = os.path.join(SCEN, "lidar", stem + ".npy")
        if not os.path.isfile(lp):
            continue
        objs = read_objs(stem)
        if not objs:
            continue
        h = read_ego(stem)
        headings.append(h)
        pts = np.load(lp)[:, :3] + np.array([1.92, 0.0, 1.35], dtype=np.float32)
        gz = ground_z_of(pts)
        gz_samples.append(gz)
        band = pts[(pts[:, 2] > gz + 0.3) & (pts[:, 2] < gz + 2.4)]

        for (gx, gy, cid) in objs:
            for name, f in HYPS.items():
                dth = np.radians(h - f(h))          # theta_code - theta_true
                c, s = np.cos(dth), np.sin(dth)
                tx = c * gx - s * gy
                ty = s * gx + c * gy
                d = np.hypot(band[:, 0] - tx, band[:, 1] - ty)
                n = int((d < RADIUS).sum())
                sc = score[name]
                sc["tot"] += 1
                sc["pts"].append(n)
                if n >= MIN_PTS:
                    sc["hit"] += 1

    print(f"프레임 수(라벨 있음): {len(headings)}, ego heading 범위: "
          f"{min(headings):.1f} ~ {max(headings):.1f} deg "
          f"(고유값 개수 근사: {len(set(round(x) for x in headings))})")
    print(f"ground_z(body) 중앙값: {np.median(gz_samples):.2f} "
          f"(범위 {min(gz_samples):.2f}~{max(gz_samples):.2f})")
    print()
    print(f"{'가설':26s} {'점유율(≥'+str(MIN_PTS)+'pts)':>16s} {'중앙 점수':>10s} {'평균 점수':>10s}")
    print("-" * 70)
    for name, sc in score.items():
        arr = np.array(sc["pts"])
        print(f"{name:26s} {sc['hit']:5d}/{sc['tot']:<5d} ({100.0*sc['hit']/sc['tot']:5.1f}%) "
              f"{np.median(arr):8.0f} {arr.mean():10.1f}")

    # heading 구간별 분해: heading 의존성이 있는 가설을 가려낸다
    print()
    print("heading 구간별 점유율 (정답 가설은 모든 구간에서 높아야 함):")
    hdr = "  " + f"{'구간':14s}" + "".join(f"{k.split()[0]:>8s}" for k in HYPS)
    print(hdr)
    edges = [-180, -90, 0, 45, 90, 135, 180, 270]
    # per-frame 재계산 (구간 분해용)
    bins = {k: {} for k in HYPS}
    tot_bin = {}
    idx = 0
    for stem in stems:
        lp = os.path.join(SCEN, "lidar", stem + ".npy")
        if not os.path.isfile(lp):
            continue
        objs = read_objs(stem)
        if not objs:
            continue
        h = read_ego(stem)
        b = None
        for i in range(len(edges) - 1):
            if edges[i] <= h < edges[i + 1]:
                b = (edges[i], edges[i + 1])
        pts = np.load(lp)[:, :3] + np.array([1.92, 0.0, 1.35], dtype=np.float32)
        gz = ground_z_of(pts)
        band = pts[(pts[:, 2] > gz + 0.3) & (pts[:, 2] < gz + 2.4)]
        for (gx, gy, cid) in objs:
            tot_bin[b] = tot_bin.get(b, 0) + 1
            for name, f in HYPS.items():
                dth = np.radians(h - f(h))
                c, s = np.cos(dth), np.sin(dth)
                tx, ty = c * gx - s * gy, s * gx + c * gy
                n = int((np.hypot(band[:, 0] - tx, band[:, 1] - ty) < RADIUS).sum())
                if n >= MIN_PTS:
                    bins[name][b] = bins[name].get(b, 0) + 1
    for b in sorted(tot_bin, key=lambda x: x[0]):
        line = f"  {str(b):14s}"
        for k in HYPS:
            line += f"{100.0*bins[k].get(b,0)/tot_bin[b]:7.1f}%"
        line += f"   (n={tot_bin[b]})"
        print(line)


if __name__ == "__main__":
    main()
