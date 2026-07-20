#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diag_world_to_ego_bug3.py — GT별 회전각 δ 전수 스윕

true = R(δ) @ gt 라 가정하고 δ ∈ [-180,180) 1° 스텝으로 LiDAR 점유 최대 δ* 탐색.
δ* 의 heading 의존성으로 규약 오류를 확정한다.
  δ* = θ_code - θ_true 이므로:
    θ_true = h      → δ* = 0
    θ_true = 90-h   → δ* = 2h-90   (h=80.9→+71.8, h=123.2→+156.5)
    θ_true = h-90   → δ* = +90
    θ_true = h+90   → δ* = -90
    θ_true = -h     → δ* = 2h      (h=80.9→+161.8, h=123.2→-113.5)
"""

import os
import csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCEN = os.path.join(HERE, "dataset", "scen17")

DELTAS = np.arange(-180, 180, 1.0)
R_MATCH = 2.0          # 점유 반경
MIN_STRONG = 40        # 이 미만이면 관측불가(유령)로 버림
MAX_RANGE = 30.0


def read_ego(stem):
    with open(os.path.join(SCEN, "ego_pose", stem + ".csv")) as f:
        row = list(csv.DictReader(f))[0]
    return float(row["ego_heading_deg"])


def read_vehicles(stem):
    out = []
    with open(os.path.join(SCEN, "labels_3d", stem + ".csv")) as f:
        for row in csv.DictReader(f):
            if int(float(row["class_id"])) != 0:
                continue
            out.append((float(row["x"]), float(row["y"])))
    return out


def ground_z_of(pts):
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

    results = []  # (h, delta_star, count, range, stem)
    cd, sd = np.cos(np.radians(DELTAS)), np.sin(np.radians(DELTAS))

    for stem in stems:
        lp = os.path.join(SCEN, "lidar", stem + ".npy")
        if not os.path.isfile(lp):
            continue
        vehs = [v for v in read_vehicles(stem) if np.hypot(*v) < MAX_RANGE]
        if not vehs:
            continue
        h = read_ego(stem)
        pts = np.load(lp)[:, :3] + np.array([1.92, 0.0, 1.35], dtype=np.float32)
        gz = ground_z_of(pts)
        if gz > 0.5:   # 지면 추정 실패 프레임은 스킵
            continue
        band = pts[(pts[:, 2] > gz + 0.3) & (pts[:, 2] < gz + 2.2)][:, :2]
        if len(band) < 50:
            continue

        for (gx, gy) in vehs:
            # 모든 δ 후보 위치 (360, 2)
            tx = cd * gx - sd * gy
            ty = sd * gx + cd * gy
            # 각 후보 위치 반경 내 점 수
            d2 = (band[None, :, 0] - tx[:, None]) ** 2 + (band[None, :, 1] - ty[:, None]) ** 2
            cnt = (d2 < R_MATCH ** 2).sum(axis=1)
            k = int(cnt.argmax())
            if cnt[k] < MIN_STRONG:
                continue
            results.append((h, float(DELTAS[k]), int(cnt[k]), float(np.hypot(gx, gy)), stem))

    print(f"유효 (GT차량, 강한 클러스터 매칭) 표본: {len(results)}")
    segA = [r for r in results if r[0] < 100]
    segB = [r for r in results if r[0] >= 100]
    for name, seg in (("segA h≈81", segA), ("segB h≈123", segB)):
        if not seg:
            continue
        ds = np.array([r[1] for r in seg])
        hs = np.array([r[0] for r in seg])
        # 원형 평균
        mean_d = np.degrees(np.arctan2(np.mean(np.sin(np.radians(ds))),
                                       np.mean(np.cos(np.radians(ds)))))
        print(f"\n[{name}] n={len(seg)}  ego h 평균={hs.mean():.2f}")
        print(f"  δ* 원형평균 = {mean_d:+.2f} deg")
        hist = {}
        for d in ds:
            b = int(round(d / 10.0) * 10)
            hist[b] = hist.get(b, 0) + 1
        for b in sorted(hist):
            print(f"    δ*≈{b:+4d}deg : {'#' * hist[b]} ({hist[b]})")
        hm = hs.mean()
        print(f"  예측: θ=h→0 | θ=90-h→{2*hm-90:+.1f} | θ=h-90→+90 | "
              f"θ=h+90→-90 | θ=-h→{((2*hm+180)%360)-180:+.1f}")

    # 상위 매칭 20건 출력
    print("\n강한 매칭 상위 20건 (count 순):")
    for h, d, c, rng, stem in sorted(results, key=lambda r: -r[2])[:20]:
        print(f"  {stem} h={h:7.2f} δ*={d:+7.1f} count={c:4d} range={rng:5.1f}")


if __name__ == "__main__":
    main()
