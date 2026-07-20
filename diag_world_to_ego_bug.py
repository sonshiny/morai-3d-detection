#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diag_world_to_ego_bug.py — world_to_ego 버그 오프라인 진단 (MORAI 불필요)

원리:
  world_to_ego에서 dp = obj_world - ego_world 는 heading과 무관하게 옳다.
  버그가 heading 규약(부호/오프셋)이라면 GT (x,y)는 실제 위치를 ego 원점 기준
  순수 회전시킨 값이다 → 거리 보존. LiDAR에서 실제 차량 클러스터를 찾아
  GT 대비 회전각을 재면 규약 오류가 그대로 드러난다.

  또한 여러 프레임에서 GT (x,y)를 각 가설의 역변환으로 월드에 되돌렸을 때
  "정지 차량이므로 월드좌표가 상수"가 되는 가설이 정답이다.
"""

import os
import csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCEN = os.path.join(HERE, "dataset", "scen17")

FRAMES = list(range(35, 66))


def read_ego(stem):
    with open(os.path.join(SCEN, "ego_pose", stem + ".csv")) as f:
        row = list(csv.DictReader(f))[0]
    return (float(row["ego_x"]), float(row["ego_y"]), float(row["ego_z"]),
            float(row["ego_heading_deg"]))


def read_vehicles(stem):
    out = []
    p = os.path.join(SCEN, "labels_3d", stem + ".csv")
    if not os.path.isfile(p):
        return out
    with open(p) as f:
        for row in csv.DictReader(f):
            if int(float(row["class_id"])) != 0:
                continue
            out.append(dict(x=float(row["x"]), y=float(row["y"]), z=float(row["z"]),
                            w=float(row["w"]), l=float(row["l"]), h=float(row["h"]),
                            sin=float(row["sin_yaw"]), cos=float(row["cos_yaw"]),
                            idx=row["object_index"]))
    return out


def lidar_clusters(stem):
    """전방 0~40m, |y|<25m, 지면 위 점을 2D 그리드 클러스터링해
    차량 후보(점수 충분, 면적 차량급) 클러스터 중심 반환."""
    pts = np.load(os.path.join(SCEN, "lidar", stem + ".npy"))[:, :3]
    # body frame으로 (LIDAR_TO_BODY: +[1.92, 0, 1.35])
    pts = pts + np.array([1.92, 0.0, 1.35], dtype=np.float32)
    m = (pts[:, 0] > 0.5) & (pts[:, 0] < 40) & (np.abs(pts[:, 1]) < 25)
    pts = pts[m]
    # 지면 제거: z 백분위 기반 (body z: 지면 ~ -0.7?) — 히스토그램으로 지면 높이 추정
    zs = pts[:, 2]
    hist, edges = np.histogram(zs, bins=100)
    ground_z = edges[np.argmax(hist)]
    obj = pts[(zs > ground_z + 0.35) & (zs < ground_z + 2.5)]

    # 0.5m 그리드 connected-component 클러스터링
    if len(obj) == 0:
        return [], ground_z
    cell = 0.5
    ij = np.floor(obj[:, :2] / cell).astype(np.int64)
    keys = {}
    for k, (i, j) in enumerate(map(tuple, ij)):
        keys.setdefault((i, j), []).append(k)
    # union-find over occupied cells (8-neighbor)
    parent = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for c in keys:
        parent[c] = c
    for (i, j) in keys:
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                n = (i + di, j + dj)
                if n in keys:
                    union((i, j), n)
    groups = {}
    for c in keys:
        groups.setdefault(find(c), []).extend(keys[c])
    clusters = []
    for idxs in groups.values():
        if len(idxs) < 30:
            continue
        p = obj[idxs]
        ext = p[:, :2].max(0) - p[:, :2].min(0)
        # 차량급 크기: 한 변 1~7m
        if max(ext) < 1.0 or max(ext) > 8.0:
            continue
        clusters.append(dict(center=p[:, :2].mean(0), npts=len(idxs),
                             ext=ext, zmax=p[:, 2].max()))
    clusters.sort(key=lambda c: -c["npts"])
    return clusters, ground_z


def main():
    np.set_printoptions(precision=3, suppress=True)

    print("=" * 78)
    print("A. 프레임별 ego pose / GT vehicle / LiDAR 차량 클러스터")
    print("=" * 78)
    per_frame = []
    for fi in FRAMES:
        stem = f"live_{fi:06d}"
        try:
            ex, ey, ez, eh = read_ego(stem)
        except FileNotFoundError:
            continue
        vehs = read_vehicles(stem)
        row = dict(fi=fi, ego=(ex, ey), eh=eh, vehs=vehs)
        per_frame.append(row)
    for r in per_frame[::5]:
        vs = "; ".join(f"[{v['idx']}] x={v['x']:.2f} y={v['y']:.2f}" for v in r["vehs"])
        print(f"f{r['fi']:03d} ego=({r['ego'][0]:.2f},{r['ego'][1]:.2f}) "
              f"hdg={r['eh']:.3f}  GT: {vs}")

    print()
    print("=" * 78)
    print("B. live_000050: LiDAR 클러스터 vs GT 위치 (거리 보존/회전각 측정)")
    print("=" * 78)
    stem = "live_000050"
    ex, ey, ez, eh = read_ego(stem)
    vehs = read_vehicles(stem)
    clusters, gz = lidar_clusters(stem)
    print(f"ground_z(body) ≈ {gz:.2f}")
    for c in clusters[:8]:
        cx, cy = c["center"]
        rng = np.hypot(cx, cy)
        brg = np.degrees(np.arctan2(cy, cx))
        print(f"  cluster n={c['npts']:5d} center=({cx:7.2f},{cy:7.2f}) "
              f"range={rng:6.2f} bearing={brg:+7.2f}deg ext=({c['ext'][0]:.1f}x{c['ext'][1]:.1f}) zmax={c['zmax']:.2f}")
    for v in vehs:
        rng = np.hypot(v["x"], v["y"])
        brg = np.degrees(np.arctan2(v["y"], v["x"]))
        print(f"  GT[{v['idx']}]  (x={v['x']:.2f}, y={v['y']:.2f}) range={rng:.2f} "
              f"bearing={brg:+.2f}deg  l={v['l']:.2f} w={v['w']:.2f}")
    print(f"  ego heading = {eh:.3f} deg")

    # 가장 가까운 range-매칭 클러스터와의 회전각
    if vehs and clusters:
        v = vehs[0]
        rng_gt = np.hypot(v["x"], v["y"])
        best = min(clusters, key=lambda c: abs(np.hypot(*c["center"]) - rng_gt))
        cx, cy = best["center"]
        dbrg = (np.degrees(np.arctan2(v["y"], v["x"])) -
                np.degrees(np.arctan2(cy, cx)))
        print(f"\n  → range 최근접 클러스터 ({cx:.2f},{cy:.2f}) vs GT: "
              f"range差={np.hypot(cx,cy)-rng_gt:+.2f}m, GT-실제 bearing差={dbrg:+.2f}deg")
        print(f"    참고: ego_heading={eh:.2f}, 2*heading={2*eh:.2f}, "
              f"90-heading 규약이면 예상差={eh-(90-eh):+.2f}, +90 규약이면 -90.00")

    print()
    print("=" * 78)
    print("C. 다중 프레임 월드좌표 상수성 테스트 (정지 차량 가정)")
    print("=" * 78)
    print("각 가설의 역변환으로 GT(x,y)->월드 복원. 표준편차가 0에 가까운 가설이 정답.")
    print("가설: math_theta = f(ego_heading_deg) ; 역변환 world = ego + R(+math_theta)@gt")

    hyps = {
        "H0 현행 (theta=h)": lambda h: h,
        "H1 부호반전 (theta=-h)": lambda h: -h,
        "H2 +90 (theta=h+90)": lambda h: h + 90.0,
        "H3 -90 (theta=h-90)": lambda h: h - 90.0,
        "H4 컴퍼스 (theta=90-h)": lambda h: 90.0 - h,
    }
    # 차량 트랙: 각 프레임 첫 vehicle (x~15 부근) 사용
    track = [(r["ego"], r["eh"], r["vehs"][0]) for r in per_frame if r["vehs"]]
    print(f"사용 프레임 수: {len(track)}, ego heading 범위: "
          f"{min(t[1] for t in track):.2f}~{max(t[1] for t in track):.2f} deg, "
          f"ego 이동거리: "
          f"{np.hypot(track[-1][0][0]-track[0][0][0], track[-1][0][1]-track[0][0][1]):.2f} m")
    for name, f in hyps.items():
        ws = []
        for (egx, egy), h, v in track:
            th = np.radians(f(h))
            c, s = np.cos(th), np.sin(th)
            wx = egx + c * v["x"] - s * v["y"]
            wy = egy + s * v["x"] + c * v["y"]
            ws.append((wx, wy))
        ws = np.array(ws)
        sd = ws.std(0)
        print(f"  {name:24s} world mean=({ws[:,0].mean():9.2f},{ws[:,1].mean():9.2f}) "
              f"std=({sd[0]:6.3f},{sd[1]:6.3f})  drift={np.hypot(*(ws[-1]-ws[0])):6.2f} m")

    print()
    print("=" * 78)
    print("D. 참고: GT rel_yaw (sin,cos) — 첫 프레임 vehicle")
    print("=" * 78)
    for (egx, egy), h, v in track[:3]:
        yaw = np.degrees(np.arctan2(v["sin"], v["cos"]))
        print(f"  ego_hdg={h:.2f} rel_yaw={yaw:+.2f} deg")


if __name__ == "__main__":
    main()
