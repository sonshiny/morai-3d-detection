#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_visibility_gt.py
=========================
각 GT 3D 박스를 3개 카메라에 rasterize 하고, 오브젝트-owner z-buffer 로
"카메라에서 실제로 보이는 픽셀 비율(visible_ratio)"을 계산해 비파괴적 sidecar 로 저장한다.

이것이 푸는 문제 / 못 푸는 문제
--------------------------------
* 푼다  : 동적 GT 객체끼리의 상호 가림(앞차가 뒤차를, 차가 보행자를 가림 등).
          모든 GT 3D box surface를 같은 카메라 평면에 rasterize 하고, 같은 픽셀에
          여러 object가 겹치면 depth가 가장 가까운 object만 그 픽셀의 owner가 된다.
          object별 visible_px / projected_px = visible_ratio.
* 못 푼다: 정적 구조물(벽/건물/가드레일/식생)에 의한 가림. 이 스크립트는 GT 3D box만
          rasterize 하므로 맵 mesh/depth/segmentation 없이는 정적 occluder를 모른다.
          따라서 벽 뒤 객체는 (다른 GT가 안 가리면) 여전히 visible로 나온다 — 한계로 명시.
          num_lidar_pts(보조 신호)가 이런 정적 가림에서 낮게 나오므로 교차 점검에 쓴다.

좌표/투영 규약은 전부 기존 검증 경로를 그대로 재사용한다(새로 만들지 않음):
  camera_configs.INTRINSICS / EXTRINSICS  (X=전방 depth, Y=좌, Z=상)
  verify_lidar_camera_overlay.project_body (box_visible_in_any_camera 와 동일 식)
  morai_dataset._box_corners_ego           (8코너 + _BOX_EDGES 순서)
  generate_occlusion_gt.{read_boxes_v2, load_lidar_body, count_points_in_boxes}
                                           (num_lidar_pts 보조필드 — helper 재사용)

핵심: LiDAR point count 로 visible/invisible 을 판정하지 않는다. visibility_level 은
      오직 z-buffer visible_ratio 로 결정하고, num_lidar_pts 는 보조 필드로만 저장한다.

산출물:  dataset/<scen>/visibility/<stem>.npz
  track_id[N] class_id[N]
  front_projected_px front_visible_px front_visible_ratio
  left_projected_px  left_visible_px  left_visible_ratio
  right_projected_px right_visible_px right_visible_ratio
  best_visible_ratio  best_visible_camera(int: 0=front,1=left,2=right,-1=none)
  truncation_ratio(best cam) num_lidar_pts visibility_level(int)
  + meta: cam_order, level_names, raster_downscale, image_wh

visibility_level (best_visible_ratio 기준):
  >= 0.50 : 3 visible
  0.10~0.50: 2 partial
  0.02~0.10: 1 heavily_occluded
  < 0.02  : 0 invisible

사용:
  python3 generate_visibility_gt.py --scen /home/autonav/visibility_test_dataset/scen02
  python3 generate_visibility_gt.py --root /home/autonav/visibility_test_dataset --all
"""

import os
import csv
import argparse

import numpy as np

from camera_configs import INTRINSICS, CAM_ORDER
from verify_lidar_camera_overlay import project_body
from morai_dataset import _box_corners_ego
import generate_occlusion_gt as OCC

# 데이터셋은 프로젝트 안 dataset/ (스크립트와 동일 위치) 에 있다.
_HERE = os.path.dirname(os.path.abspath(__file__))


# ---- 원본 이미지 해상도 (jpg = 1600x900). project_body 에 원본 INTRINSICS 사용. ----
ORIG_W = 1600
ORIG_H = 900

# rasterize 다운스케일 (속도용). ratio 는 해상도 불변이라 정밀도 영향 미미.
DEFAULT_DOWNSCALE = 2

MIN_DEPTH = 0.1   # project_body 기본 near-cut (카메라 뒤/근접)

# visibility_level 임계 (best_visible_ratio 기준)
LEVEL_NAMES = ["invisible", "heavily_occluded", "partial", "visible"]  # 0..3


def visibility_level(best_ratio):
    if best_ratio >= 0.50:
        return 3
    if best_ratio >= 0.10:
        return 2
    if best_ratio >= 0.02:
        return 1
    return 0


# _box_corners_ego 코너 순서로 정의한 6면 -> 12 삼각형.
#  0:+l+w+h 1:+l+w-h 2:+l-w+h 3:+l-w-h 4:-l+w+h 5:-l+w-h 6:-l-w+h 7:-l-w-h
_BOX_TRIS = [
    (0, 1, 3), (0, 3, 2),   # front  (x=+l/2)
    (4, 5, 7), (4, 7, 6),   # back   (x=-l/2)
    (0, 2, 6), (0, 6, 4),   # top    (z=+h/2)
    (1, 3, 7), (1, 7, 5),   # bottom (z=-h/2)
    (0, 1, 5), (0, 5, 4),   # left   (y=+w/2)
    (2, 3, 7), (2, 7, 6),   # right  (y=-w/2)
]


# ---------------------------------------------------------------------------
# box -> 8 corners (body/ego frame). labels_3d_v2 는 z_center 를 주므로
# _box_corners_ego 규약(box[2]=z_bottom, 내부에서 +h/2)에 맞춰 z_bottom 전달.
# ---------------------------------------------------------------------------
def box_corners_from_v2(b):
    h = float(b["h"])
    z_bottom = float(b["cz"]) - h * 0.5
    box_vec = [
        float(b["cx"]), float(b["cy"]), z_bottom,
        float(np.log(max(b["w"], 1e-6))),
        float(np.log(max(b["l"], 1e-6))),
        float(np.log(max(h, 1e-6))),
        float(np.sin(b["yaw"])), float(np.cos(b["yaw"])),
        0.0, 0.0, 0.0,
    ]
    return _box_corners_ego(box_vec)[:8].astype(np.float64)  # (8,3)


# ---------------------------------------------------------------------------
# 삼각형 rasterize (barycentric depth, per-object depth map 갱신, in-place z-min)
# ---------------------------------------------------------------------------
def _fill_tri(depth_buf, tx, ty, td, W, H):
    minx = int(np.floor(min(tx))); maxx = int(np.ceil(max(tx)))
    miny = int(np.floor(min(ty))); maxy = int(np.ceil(max(ty)))
    minx = max(minx, 0); miny = max(miny, 0)
    maxx = min(maxx, W - 1); maxy = min(maxy, H - 1)
    if maxx < minx or maxy < miny:
        return
    x0, x1, x2 = tx
    y0, y1, y2 = ty
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    if abs(area) < 1e-9:
        return
    xs = np.arange(minx, maxx + 1)
    ys = np.arange(miny, maxy + 1)
    px = xs[None, :] + 0.5
    py = ys[:, None] + 0.5
    w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) / area
    w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) / area
    w2 = 1.0 - w0 - w1
    eps = -1e-6
    inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
    if not inside.any():
        return
    depth = w0 * td[0] + w1 * td[1] + w2 * td[2]
    sub = depth_buf[miny:maxy + 1, minx:maxx + 1]
    upd = inside & (depth < sub)
    sub[upd] = depth[upd].astype(sub.dtype)


def rasterize_object_depth(ur, vr, depth, valid, W, H):
    """object 8코너의 raster 좌표(ur,vr)+depth로 per-object nearest-surface depth map.
    카메라 뒤 코너를 가진 삼각형은 스킵(near-clip 근사)."""
    buf = np.full((H, W), np.inf, dtype=np.float32)
    for a, b, c in _BOX_TRIS:
        if not (valid[a] and valid[b] and valid[c]):
            continue
        _fill_tri(buf, ur[[a, b, c]], vr[[a, b, c]], depth[[a, b, c]], W, H)
    return buf


# ---------------------------------------------------------------------------
# 볼록다각형 유틸 (truncation = 화면 밖으로 나간 projected 영역 비율)
# ---------------------------------------------------------------------------
def _convex_hull(pts):
    """Andrew monotone chain. pts: (M,2). 반환 (K,2) 시계/반시계 폐다각형(미폐)."""
    pts = sorted(set(map(tuple, np.round(pts, 3))))
    if len(pts) < 3:
        return np.array(pts, dtype=np.float64)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=np.float64)


def _poly_area(poly):
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]; y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clip_poly_rect(poly, W, H):
    """Sutherland-Hodgman: 다각형을 사각형 [0,W]x[0,H] 로 클립."""
    if len(poly) < 3:
        return poly

    def clip_edge(pts, inside_fn, inter_fn):
        out = []
        n = len(pts)
        for i in range(n):
            cur = pts[i]; prv = pts[i - 1]
            ci = inside_fn(cur); pi = inside_fn(prv)
            if ci:
                if not pi:
                    out.append(inter_fn(prv, cur))
                out.append(cur)
            elif pi:
                out.append(inter_fn(prv, cur))
        return out

    def inter(p, q, f):
        t = f(p) / (f(p) - f(q))
        return (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))

    pts = [tuple(p) for p in poly]
    # left x>=0
    pts = clip_edge(pts, lambda p: p[0] >= 0, lambda p, q: inter(p, q, lambda z: z[0]))
    if not pts:
        return np.zeros((0, 2))
    # right x<=W
    pts = clip_edge(pts, lambda p: p[0] <= W, lambda p, q: inter(p, q, lambda z: z[0] - W))
    if not pts:
        return np.zeros((0, 2))
    # bottom y>=0
    pts = clip_edge(pts, lambda p: p[1] >= 0, lambda p, q: inter(p, q, lambda z: z[1]))
    if not pts:
        return np.zeros((0, 2))
    # top y<=H
    pts = clip_edge(pts, lambda p: p[1] <= H, lambda p, q: inter(p, q, lambda z: z[1] - H))
    return np.array(pts, dtype=np.float64)


def truncation_ratio(ur, vr, valid, W, H):
    """카메라 앞(valid) 코너들의 projected convex hull 중 화면 밖 비율."""
    pts = np.stack([ur[valid], vr[valid]], axis=1)
    if pts.shape[0] < 3:
        return 0.0
    hull = _convex_hull(pts)
    a_full = _poly_area(hull)
    if a_full < 1e-9:
        return 0.0
    a_in = _poly_area(_clip_poly_rect(hull, W, H))
    return float(np.clip(1.0 - a_in / a_full, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 프레임 처리
# ---------------------------------------------------------------------------
def compute_frame_visibility(boxes, downscale=DEFAULT_DOWNSCALE):
    """
    boxes: read_boxes_v2 결과(list of dict). 반환: list of per-object dict.
    각 카메라마다 모든 object depth map을 구하고, argmin owner z-buffer로 visible_px 계산.
    """
    N = len(boxes)
    W = ORIG_W // downscale
    H = ORIG_H // downscale
    ds = float(downscale)

    # 결과 누적
    per_cam = {cam: {"proj": np.zeros(N, np.int64),
                     "vis": np.zeros(N, np.int64),
                     "trunc": np.zeros(N, np.float64)} for cam in CAM_ORDER}

    # 코너 미리 계산 (body frame)
    corners = [box_corners_from_v2(b) for b in boxes]  # each (8,3)

    for cam in CAM_ORDER:
        K = INTRINSICS[cam].astype(np.float64)
        obj_depths = []
        objs_meta = []
        for i in range(N):
            ch = np.concatenate([corners[i], np.ones((8, 1))], axis=1)
            u, v, depth, valid = project_body(ch, cam, K, min_depth=MIN_DEPTH)
            ur = u / ds
            vr = v / ds
            objs_meta.append((ur, vr, depth, valid))
            if not valid.any():
                obj_depths.append(None)
                per_cam[cam]["trunc"][i] = 1.0
                continue
            buf = rasterize_object_depth(ur, vr, depth, valid, W, H)
            obj_depths.append(buf)
            per_cam[cam]["proj"][i] = int(np.isfinite(buf).sum())
            per_cam[cam]["trunc"][i] = truncation_ratio(ur, vr, valid, W, H)

        # 글로벌 z-buffer owner (가장 가까운 object가 픽셀 소유)
        stack = np.full((max(N, 1), H, W), np.inf, dtype=np.float32)
        for i in range(N):
            if obj_depths[i] is not None:
                stack[i] = obj_depths[i]
        min_depth = stack.min(axis=0)
        covered = np.isfinite(min_depth)
        owner = np.argmin(stack, axis=0)
        for i in range(N):
            if obj_depths[i] is None or per_cam[cam]["proj"][i] == 0:
                continue
            vis = covered & (owner == i)
            per_cam[cam]["vis"][i] = int(vis.sum())

    # object별 취합
    results = []
    cam_key = {"cam_front": "front", "cam_front_left": "left", "cam_front_right": "right"}
    for i, b in enumerate(boxes):
        rec = {"track_id": int(b["track_id"]), "class_id": int(b["class_id"])}
        ratios = {}
        for cam in CAM_ORDER:
            proj = int(per_cam[cam]["proj"][i])
            vis = int(per_cam[cam]["vis"][i])
            ratio = (vis / proj) if proj > 0 else 0.0
            k = cam_key[cam]
            rec["%s_projected_px" % k] = proj
            rec["%s_visible_px" % k] = vis
            rec["%s_visible_ratio" % k] = float(ratio)
            ratios[cam] = (ratio, proj)
        # best = projected_px>0 인 카메라 중 visible_ratio 최대
        best_cam_idx = -1
        best_ratio = 0.0
        for ci, cam in enumerate(CAM_ORDER):
            r, proj = ratios[cam]
            if proj > 0 and r >= best_ratio:
                if r > best_ratio or best_cam_idx == -1:
                    best_ratio = r
                    best_cam_idx = ci
        rec["best_visible_ratio"] = float(best_ratio)
        rec["best_visible_camera"] = int(best_cam_idx)
        # truncation 은 best camera 기준(가장 신뢰하는 뷰)
        rec["truncation_ratio"] = float(
            per_cam[CAM_ORDER[best_cam_idx]]["trunc"][i] if best_cam_idx >= 0 else 1.0
        )
        rec["visibility_level"] = int(visibility_level(best_ratio))
        results.append(rec)
    return results


# ---------------------------------------------------------------------------
# 보조 신호: num_lidar_pts (generate_occlusion_gt helper 재사용). 판정에는 미사용.
# ---------------------------------------------------------------------------
def lidar_counts_for_boxes(scen_dir, stem, boxes):
    lidar_path = os.path.join(scen_dir, "lidar", stem + ".npy")
    if not os.path.isfile(lidar_path):
        return {int(b["track_id"]): -1 for b in boxes}
    p_body = OCC.load_lidar_body(lidar_path)
    out = OCC.count_points_in_boxes(p_body, boxes)  # (M,3) [tid, npts, dist]
    return {int(r[0]): int(r[1]) for r in out}


# ---------------------------------------------------------------------------
# scene 처리 + 저장
# ---------------------------------------------------------------------------
def process_scen(scen_dir, downscale=DEFAULT_DOWNSCALE, write=True, limit=None):
    lbl_dir = os.path.join(scen_dir, "labels_3d_v2")
    if not os.path.isdir(lbl_dir):
        raise FileNotFoundError("labels_3d_v2 없음: %s" % lbl_dir)
    vis_dir = os.path.join(scen_dir, "visibility")
    if write:
        os.makedirs(vis_dir, exist_ok=True)

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith(".csv"))
    if limit:
        stems = stems[:limit]

    agg = {"n_frames": 0, "n_objs": 0,
           "level_counts": np.zeros(4, np.int64),
           "best_ratios": [], "num_lidar": [], "contradict": []}

    for stem in stems:
        boxes = OCC.read_boxes_v2(os.path.join(lbl_dir, stem + ".csv"))
        if not boxes:
            if write:
                np.savez(os.path.join(vis_dir, stem + ".npz"),
                         track_id=np.zeros(0, np.int64))
            continue
        recs = compute_frame_visibility(boxes, downscale=downscale)
        npts_map = lidar_counts_for_boxes(scen_dir, stem, boxes)

        cols = {}
        keys_int = ["track_id", "class_id",
                    "front_projected_px", "front_visible_px",
                    "left_projected_px", "left_visible_px",
                    "right_projected_px", "right_visible_px",
                    "best_visible_camera", "num_lidar_pts", "visibility_level"]
        keys_float = ["front_visible_ratio", "left_visible_ratio", "right_visible_ratio",
                      "best_visible_ratio", "truncation_ratio"]
        for r in recs:
            r["num_lidar_pts"] = int(npts_map.get(r["track_id"], -1))
        for k in keys_int:
            cols[k] = np.array([r[k] for r in recs], dtype=np.int64)
        for k in keys_float:
            cols[k] = np.array([r[k] for r in recs], dtype=np.float32)

        if write:
            np.savez(
                os.path.join(vis_dir, stem + ".npz"),
                cam_order=np.array(CAM_ORDER),
                level_names=np.array(LEVEL_NAMES),
                raster_downscale=np.array([downscale]),
                image_wh=np.array([ORIG_W, ORIG_H]),
                **cols,
            )

        # 통계
        agg["n_frames"] += 1
        agg["n_objs"] += len(recs)
        for r in recs:
            agg["level_counts"][r["visibility_level"]] += 1
            agg["best_ratios"].append(r["best_visible_ratio"])
            agg["num_lidar"].append(r["num_lidar_pts"])
            # LiDAR vs z-buffer 모순: LiDAR 점 많은데 z-buffer로는 안 보임(정적가림 후보) 또는
            #                         LiDAR 점 0인데 z-buffer로는 잘 보임(맵 밖/미측정 후보)
            n = r["num_lidar_pts"]; br = r["best_visible_ratio"]
            if n >= 10 and br < 0.10:
                agg["contradict"].append((stem, r["track_id"], n, br, "lidar_many_zbuf_hidden"))
            elif n == 0 and br >= 0.50:
                agg["contradict"].append((stem, r["track_id"], n, br, "lidar_zero_zbuf_visible"))
    return agg


def print_report(scen_dir, agg):
    br = np.array(agg["best_ratios"], float) if agg["best_ratios"] else np.zeros(0)
    print("=" * 78)
    print("  visibility GT: %s" % scen_dir)
    print("  frames=%d objects=%d" % (agg["n_frames"], agg["n_objs"]))
    print("=" * 78)
    lc = agg["level_counts"]
    tot = max(int(lc.sum()), 1)
    for lvl in (3, 2, 1, 0):
        print("  level %d %-16s : %5d (%.1f%%)" % (lvl, LEVEL_NAMES[lvl], lc[lvl], 100.0 * lc[lvl] / tot))
    if br.size:
        print("  best_visible_ratio: min=%.3f median=%.3f mean=%.3f max=%.3f" % (
            br.min(), np.median(br), br.mean(), br.max()))
    con = agg["contradict"]
    print("  LiDAR vs z-buffer 모순 사례: %d" % len(con))
    for stem, tid, n, r, why in con[:15]:
        print("    %s tid=%d num_lidar=%d best_ratio=%.3f [%s]" % (stem, tid, n, r, why))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scen", default=None, help="단일 scen 디렉터리 경로")
    ap.add_argument("--root", default=os.path.join(_HERE, "dataset"))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--downscale", type=int, default=DEFAULT_DOWNSCALE)
    ap.add_argument("--limit", type=int, default=None, help="처리할 프레임 수 상한(디버그)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    scen_dirs = []
    if args.scen:
        scen_dirs = [args.scen]
    elif args.all:
        for name in sorted(os.listdir(args.root)):
            d = os.path.join(args.root, name)
            if os.path.isdir(os.path.join(d, "labels_3d_v2")):
                scen_dirs.append(d)
    else:
        raise SystemExit("--scen <dir> 또는 --all 필요")

    for d in scen_dirs:
        agg = process_scen(d, downscale=args.downscale, write=not args.dry_run, limit=args.limit)
        print_report(d, agg)


if __name__ == "__main__":
    main()
