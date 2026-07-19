#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diag_ego_pose_timing.py — scen17 ego pose 시점 어긋남 검증 (오프라인)

① ego_pose 추이: 프레임별 ego_x/ego_y/heading, 프레임 간 이동량(=속도 proxy)
② sync_log dt: ego/obj의 src·recv 매칭 간격, ego_src_ts가 프레임마다 전진하는지
③ LiDAR 장면 이동량: 연속 프레임 점군의 2D 정합 오프셋 → ego 실제 이동 여부
   (pose가 멈췄는데 장면이 흐르면 = ego pose 시점/스트림 어긋남 확정)
"""

import os
import csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCEN = os.path.join(HERE, "dataset", "scen17")


def load_ego_all():
    rows = []
    d = os.path.join(SCEN, "ego_pose")
    for f in sorted(os.listdir(d)):
        if not f.endswith(".csv"):
            continue
        with open(os.path.join(d, f)) as fh:
            r = list(csv.DictReader(fh))[0]
        rows.append((os.path.splitext(f)[0], float(r["timestamp"]),
                     float(r["ego_x"]), float(r["ego_y"]),
                     float(r["ego_heading_deg"])))
    return rows


def scene_shift(stem_a, stem_b, max_r=35.0):
    """연속 두 LiDAR 프레임의 2D 이동량을 x/y 그리드 상호상관으로 추정."""
    pa = np.load(os.path.join(SCEN, "lidar", stem_a + ".npy"))[:, :3]
    pb = np.load(os.path.join(SCEN, "lidar", stem_b + ".npy"))[:, :3]

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
    for dx in range(-12, 13):
        for dy in range(-12, 13):
            a = ga[max(0, dx):140 + min(0, dx), max(0, dy):140 + min(0, dy)]
            b = gb[max(0, -dx):140 - max(0, dx), max(0, -dy):140 - max(0, dy)]
            sc = float((a * b).sum())
            if sc > best[2]:
                best = (dx, dy, sc)
    return best[0] * 0.5, best[1] * 0.5


def main():
    ego = load_ego_all()

    print("=" * 80)
    print("① ego pose 추이 (5프레임 간격) + 프레임 간 이동량")
    print("=" * 80)
    print(f"{'stem':14s} {'t(상대)':>9s} {'ego_x':>10s} {'ego_y':>10s} {'hdg':>8s} {'Δd(m)':>7s} {'Δhdg':>7s}")
    t0 = ego[0][1]
    prev = None
    for i, (stem, t, x, y, h) in enumerate(ego):
        dd = dh = 0.0
        if prev:
            dd = np.hypot(x - prev[2], y - prev[3])
            dh = h - prev[4]
        if i % 5 == 0 or dd > 1.0:
            print(f"{stem:14s} {t - t0:9.2f} {x:10.2f} {y:10.2f} {h:8.3f} {dd:7.2f} {dh:+7.3f}")
        prev = (stem, t, x, y, h)

    # 정지 구간 요약
    print()
    moves = [np.hypot(ego[i][2] - ego[i - 1][2], ego[i][3] - ego[i - 1][3])
             for i in range(1, len(ego))]
    moves = np.array(moves)
    print(f"프레임 간 ego 이동량: median={np.median(moves):.3f}m, "
          f"mean={moves.mean():.3f}m, max={moves.max():.2f}m, "
          f"0.05m 미만 비율={100.0 * (moves < 0.05).mean():.1f}%")

    print()
    print("=" * 80)
    print("② sync_log: ego/obj 매칭 시각 분석 (save 행만)")
    print("=" * 80)
    with open(os.path.join(SCEN, "sync_log.csv")) as f:
        sl = [r for r in csv.DictReader(f) if r["action"] == "save"]
    print(f"save 행 수: {len(sl)}")

    def fl(r, k):
        return float(r[k]) if r[k] not in ("", None) else np.nan

    ego_src = np.array([fl(r, "ego_src_ts") for r in sl])
    ego_recv = np.array([fl(r, "ego_recv_ts") for r in sl])
    ego_dt = np.array([fl(r, "ego_src_dt_ms") for r in sl])
    obj_dt = np.array([fl(r, "obj_src_dt_ms") for r in sl])
    ref_src = np.array([fl(r, "ref_src_ts") for r in sl])
    ref_recv = np.array([fl(r, "ref_recv_ts") for r in sl])

    print(f"ego_src_dt_ms : median={np.nanmedian(ego_dt):8.1f}  max={np.nanmax(ego_dt):10.1f}")
    print(f"obj_src_dt_ms : median={np.nanmedian(obj_dt):8.1f}  max={np.nanmax(obj_dt):10.1f}")
    dsrc = np.diff(ego_src)
    print(f"ego_src_ts 프레임 간 증분: median={np.nanmedian(dsrc):.3f}s  "
          f"0증분(같은 msg 재사용) 비율={100.0 * (np.abs(dsrc) < 1e-6).mean():.1f}%")
    print(f"ego_src_ts - ref_src_ts : median={np.nanmedian(ego_src - ref_src):+.3f}s "
          f"범위 [{np.nanmin(ego_src - ref_src):+.3f}, {np.nanmax(ego_src - ref_src):+.3f}]")
    print(f"ego_recv_ts - ref_recv_ts: median={np.nanmedian(ego_recv - ref_recv):+.3f}s "
          f"범위 [{np.nanmin(ego_recv - ref_recv):+.3f}, {np.nanmax(ego_recv - ref_recv):+.3f}]")
    print(f"ref_src vs ref_recv 차: median={np.nanmedian(ref_recv - ref_src):+.2f}s "
          f"(카메라 header가 딴 클럭이면 큼)")

    print()
    print("=" * 80)
    print("③ LiDAR 장면 이동량 vs ego pose 이동량 (프레임 40~70, 120~130)")
    print("=" * 80)
    stems = [e[0] for e in ego]
    pose = {e[0]: (e[2], e[3], e[4]) for e in ego}
    for lo, hi in ((40, 70), (118, 130)):
        print(f"-- 프레임 {lo}~{hi} --")
        for i in range(lo, hi, 3):
            a, b = f"live_{i:06d}", f"live_{i + 3:06d}"
            if a not in pose or b not in pose:
                continue
            if not (os.path.isfile(os.path.join(SCEN, "lidar", a + ".npy"))
                    and os.path.isfile(os.path.join(SCEN, "lidar", b + ".npy"))):
                continue
            sx, sy = scene_shift(a, b)
            pd = np.hypot(pose[b][0] - pose[a][0], pose[b][1] - pose[a][1])
            print(f"  {a}->{b}: LiDAR 장면이동=({sx:+5.1f},{sy:+5.1f})m |{np.hypot(sx,sy):5.1f}m|"
                  f"   ego pose 이동={pd:6.2f}m   hdg {pose[a][2]:.2f}->{pose[b][2]:.2f}")


if __name__ == "__main__":
    main()
