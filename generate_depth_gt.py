#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_depth_gt.py
=====================
단일 scen에 대한 depth GT(sparse LiDAR projection) 생성 실험 스크립트.
SparseDrive 공식 코드(MultiScaleDepthMapGenerator, DenseDepthNet)의 depth GT 생성 방식을
그대로 따른다: LiDAR 포인트를 카메라에 투영하고, 같은 픽셀에 여러 점이 맺히면
z-buffer 방식으로 가장 가까운 depth만 남긴다.

기존 파일(morai_3d_live.py, camera_configs.py, morai_dataset.py,
verify_lidar_camera_overlay.py)은 import만 하고 절대 수정하지 않는다.
verify_lidar_camera_overlay.py 가 이미 검증해 둔 project_body()/lidar_to_body_h()를
그대로 재사용한다 — 이 스크립트에서 유일하게 다른 점은 K를 원본이 아니라
morai_dataset.scale_intrinsic_for_input()으로 704x256 학습 좌표계로 스케일한 것뿐이다
(그래야 depth GT가 모델 입력 해상도와 픽셀 단위로 정렬된다).

신규 산출물만 생성한다:
  dataset/<scen>/depth_gt/{cam_front,cam_front_left,cam_front_right}/<stem>.npy
  depth_gt_verify/*.jpg
기존 데이터는 읽기만 한다.
"""

import os
import csv
import random
import argparse
import time

import cv2
import numpy as np

from camera_configs import INTRINSICS, CAM_ORDER
from morai_dataset import (
    scale_intrinsic_for_input,
    IMG_WIDTH,
    IMG_HEIGHT,
    _box_corners_ego,
)
# verify_lidar_camera_overlay 는 import만 한다 (수정 금지, 재사용).
from verify_lidar_camera_overlay import (
    project_body,
    lidar_to_body_h,
    _BOX_EDGES,
)

MIN_DEPTH = 0.1
MAX_DEPTH = 60.0
MIN_LIDAR_RANGE = 1.5

_HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(_HERE, "dataset")

# GT 박스 클래스별 wireframe 색 (BGR)
_CLASS_COLOR = {0: (0, 255, 0), 1: (0, 255, 255)}   # 0=vehicle green, 1=pedestrian yellow


def scaled_K(cam):
    """box_visible_in_any_camera / 학습 좌표계와 동일한 704x256 스케일 K."""
    return scale_intrinsic_for_input(INTRINSICS[cam].astype(np.float64), IMG_WIDTH, IMG_HEIGHT)


def load_lidar_xyz(path, min_range=MIN_LIDAR_RANGE):
    cloud = np.load(path)
    if cloud.ndim != 2 or cloud.shape[1] < 3:
        raise ValueError(f"예상과 다른 라이다 shape: {cloud.shape} ({path})")
    xyz = cloud[:, :3].astype(np.float64)
    ranges = np.linalg.norm(xyz, axis=1)
    keep = ranges >= float(min_range)
    return xyz[keep], int(xyz.shape[0]), int((~keep).sum())


def project_and_zbuffer(xyz, cam):
    """
    LiDAR (N,3) -> 704x256 카메라 픽셀에 투영 후 z-buffer dedup.

    project_body()는 verify_lidar_camera_overlay.py 검증 경로 그대로 재사용
    (valid = depth > min_depth, u/v 부호·축 규약 동일). 여기서 바뀌는 건 K뿐.

    dedup: "depth 내림차순 정렬 후 기록"(먼 점 먼저 써서 가까운 점이 마지막에
    덮어씀)과 동일한 결과를 내도록, depth 오름차순 정렬 후 픽셀당 첫 occurrence
    (=가장 가까운 점)만 남기는 벡터화 방식을 쓴다. 최종적으로 각 픽셀에는
    가장 가까운 depth 1개만 남는다 (공식 z-buffer 로직과 동일한 최종 상태).

    반환: (out[N,3] float32 [u,v,depth], n_in_frame_before_dedup, n_after_dedup)
    """
    K = scaled_K(cam)
    p_body = lidar_to_body_h(xyz)
    u, v, depth, valid = project_body(p_body, cam, K, MIN_DEPTH)

    u, v, depth = u[valid], v[valid], depth[valid]
    in_frame = (u >= 0.0) & (u < IMG_WIDTH) & (v >= 0.0) & (v < IMG_HEIGHT)
    u, v, depth = u[in_frame], v[in_frame], depth[in_frame]

    n_before_dedup = int(u.shape[0])
    if n_before_dedup == 0:
        return np.zeros((0, 3), dtype=np.float32), 0, 0

    # max_depth는 필터가 아니라 clip. min_depth 하한은 project_body의 valid에서
    # 이미 depth > MIN_DEPTH 로 보장되므로 여기서는 상한만 실질적으로 작동한다.
    depth = np.clip(depth, MIN_DEPTH, MAX_DEPTH)

    ru = np.clip(np.round(u), 0, IMG_WIDTH - 1).astype(np.int64)
    rv = np.clip(np.round(v), 0, IMG_HEIGHT - 1).astype(np.int64)

    order = np.argsort(depth)  # 오름차순 = 가까운 점 먼저
    ru_s, rv_s, d_s = ru[order], rv[order], depth[order]
    keys = ru_s * IMG_HEIGHT + rv_s  # (ru,rv) 픽셀 고유 키

    _, first_idx = np.unique(keys, return_index=True)  # 정렬된 배열에서 픽셀당 첫 occurrence = 최근접 depth
    out = np.stack(
        [
            ru_s[first_idx].astype(np.float32),
            rv_s[first_idx].astype(np.float32),
            d_s[first_idx].astype(np.float32),
        ],
        axis=1,
    )
    n_after_dedup = int(out.shape[0])
    return out, n_before_dedup, n_after_dedup


def read_all_boxes(csv_path):
    """labels_3d_v2/labels_3d CSV 전체 행을 (class_id, vals) 리스트로 반환."""
    boxes = []
    if not os.path.isfile(csv_path):
        return boxes
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cls_id = int(float(row["class_id"]))
            if "z_center" in row:
                w, l, h = float(row["w"]), float(row["l"]), float(row["h"])
                z_bottom = float(row["z_center"]) - h * 0.5
                yaw = float(row["yaw_ego"])
                vals = [
                    float(row["x"]), float(row["y"]), z_bottom,
                    np.log(max(w, 1e-6)), np.log(max(l, 1e-6)), np.log(max(h, 1e-6)),
                    np.sin(yaw), np.cos(yaw),
                    float(row["vx_ego"]), float(row["vy_ego"]), float(row["vz"]),
                ]
            else:
                vals = [
                    float(row["x"]), float(row["y"]), float(row["z"]),
                    float(row["ln_w"]), float(row["ln_l"]), float(row["ln_h"]),
                    float(row["sin_yaw"]), float(row["cos_yaw"]),
                    float(row["vx"]), float(row["vy"]), float(row["vz"]),
                ]
            boxes.append((cls_id, vals))
    return boxes


def label_dir_for_scen(scen_dir):
    labels_v2 = os.path.join(scen_dir, "labels_3d_v2")
    if os.path.isdir(labels_v2):
        return labels_v2
    return os.path.join(scen_dir, "labels_3d")


def depth_to_bgr(depth, near=0.0, far=60.0):
    t = np.clip((depth - near) / max(far - near, 1e-6), 0.0, 1.0)
    b = (255.0 * t).astype(np.uint8)
    r = (255.0 * (1.0 - t)).astype(np.uint8)
    g = np.zeros_like(b)
    return np.stack([b, g, r], axis=1)


def draw_gt_wireframe(img, boxes, cam, K, min_depth=MIN_DEPTH):
    drawn = 0
    for cls_id, vals in boxes:
        corners = _box_corners_ego(vals)
        corners_h = np.concatenate(
            [corners.astype(np.float64), np.ones((corners.shape[0], 1))], axis=1
        )
        u, v, depth, valid = project_body(corners_h, cam, K, min_depth)
        inframe = valid & (u >= 0) & (u < img.shape[1]) & (v >= 0) & (v < img.shape[0])
        if not inframe.any():
            continue
        color = _CLASS_COLOR.get(cls_id, (255, 255, 255))
        for a, b in _BOX_EDGES:
            if not (valid[a] and valid[b]):
                continue
            pa = (int(round(u[a])), int(round(v[a])))
            pb = (int(round(u[b])), int(round(v[b])))
            cv2.line(img, pa, pb, color, 2, lineType=cv2.LINE_AA)
        drawn += 1
    return drawn


def process_scen(scen_dir, scen_name):
    lbl_dir = label_dir_for_scen(scen_dir)
    lidar_dir = os.path.join(scen_dir, "lidar")
    depth_gt_root = os.path.join(scen_dir, "depth_gt")

    stems = sorted(
        os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith(".csv")
    )

    for cam in CAM_ORDER:
        os.makedirs(os.path.join(depth_gt_root, cam), exist_ok=True)

    stats = {
        cam: {"n_points_per_frame": [], "n_before": 0, "n_after": 0}
        for cam in CAM_ORDER
    }
    all_depths = {cam: [] for cam in CAM_ORDER}
    n_missing_lidar = 0
    missing_lidar_stems = []
    frames_with_gt_boxes = []
    n_lidar_raw = 0
    n_lidar_range_removed = 0

    for stem in stems:
        lidar_path = os.path.join(lidar_dir, stem + ".npy")
        csv_path = os.path.join(lbl_dir, stem + ".csv")
        boxes = read_all_boxes(csv_path)
        if boxes:
            frames_with_gt_boxes.append(stem)

        if not os.path.isfile(lidar_path):
            n_missing_lidar += 1
            missing_lidar_stems.append(stem)
            print(f"[SKIP] lidar 없음: {scen_name}/{stem} -> depth_gt 생성 스킵")
            continue

        xyz, n_raw, n_range_removed = load_lidar_xyz(lidar_path)
        n_lidar_raw += n_raw
        n_lidar_range_removed += n_range_removed

        for cam in CAM_ORDER:
            out, n_before, n_after = project_and_zbuffer(xyz, cam)
            save_path = os.path.join(depth_gt_root, cam, stem + ".npy")
            np.save(save_path, out)

            stats[cam]["n_points_per_frame"].append(n_after)
            stats[cam]["n_before"] += n_before
            stats[cam]["n_after"] += n_after
            if out.shape[0] > 0:
                all_depths[cam].append(out[:, 2])

    range_stats = {
        "n_lidar_raw": n_lidar_raw,
        "n_lidar_range_removed": n_lidar_range_removed,
        "n_lidar_after_range": n_lidar_raw - n_lidar_range_removed,
    }
    return stems, stats, all_depths, n_missing_lidar, missing_lidar_stems, frames_with_gt_boxes, range_stats


def make_histogram_text(values, bins=10, lo=0.0, hi=60.0):
    if values.size == 0:
        return "  (포인트 없음)"
    hist, edges = np.histogram(values, bins=bins, range=(lo, hi))
    lines = []
    max_count = max(int(hist.max()), 1)
    for i in range(bins):
        bar = "#" * int(40 * hist[i] / max_count)
        lines.append(f"  [{edges[i]:5.1f}-{edges[i+1]:5.1f}m) {hist[i]:6d} {bar}")
    return "\n".join(lines)


def make_verify_overlays(scen_dir, scen_name, frames_with_gt_boxes, out_dir, num_frames=6, seed=20260707):
    rng = random.Random(seed)
    pool = list(frames_with_gt_boxes)
    rng.shuffle(pool)
    picked = pool[:num_frames]

    os.makedirs(out_dir, exist_ok=True)
    saved_paths = []

    for stem in picked:
        boxes = read_all_boxes(os.path.join(label_dir_for_scen(scen_dir), stem + ".csv"))
        for cam in CAM_ORDER:
            img_path = os.path.join(scen_dir, "images", cam, stem + ".jpg")
            depth_path = os.path.join(scen_dir, "depth_gt", cam, stem + ".npy")
            if not os.path.isfile(img_path) or not os.path.isfile(depth_path):
                print(f"[WARN] verify 스킵 (파일 없음): {scen_name}/{cam}/{stem}")
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] 이미지 디코드 실패: {img_path}")
                continue
            img_rs = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

            depth_pts = np.load(depth_path)
            overlay = img_rs.copy()
            if depth_pts.shape[0] > 0:
                order = np.argsort(-depth_pts[:, 2])  # 먼 점 먼저 그려 가까운 점이 위로
                colors = depth_to_bgr(depth_pts[order, 2])
                for (pu, pv, _), col in zip(depth_pts[order], colors):
                    cv2.circle(
                        overlay, (int(pu), int(pv)), 1,
                        (int(col[0]), int(col[1]), int(col[2])), -1, lineType=cv2.LINE_AA
                    )

            K = scaled_K(cam)
            n_drawn = draw_gt_wireframe(overlay, boxes, cam, K, MIN_DEPTH)

            cv2.putText(overlay, f"{cam} {stem} depth_gt(n={depth_pts.shape[0]}) GTbox={n_drawn}",
                        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            out_path = os.path.join(out_dir, f"depthgt_{scen_name}_{cam}_{stem}.jpg")
            cv2.imwrite(out_path, overlay)
            saved_paths.append(out_path)

    return picked, saved_paths


def discover_complete_scens(dataset_root):
    scens = []
    for name in sorted(os.listdir(dataset_root), key=lambda s: int(s[4:]) if s.startswith("scen") and s[4:].isdigit() else 10**9):
        if not (name.startswith("scen") and name[4:].isdigit()):
            continue
        scen_dir = os.path.join(dataset_root, name)
        required = [
            os.path.join(scen_dir, "labels_3d_v2"),
            os.path.join(scen_dir, "lidar"),
            os.path.join(scen_dir, "images", "cam_front"),
            os.path.join(scen_dir, "images", "cam_front_left"),
            os.path.join(scen_dir, "images", "cam_front_right"),
        ]
        if all(os.path.isdir(p) for p in required):
            scens.append(name)
    return scens


def print_report(scen_name, stems, stats, all_depths, n_missing_lidar, missing_stems, frames_with_gt, range_stats, picked, saved_paths):
    print()
    print("=" * 76)
    print(f"  depth GT 생성 완료: {scen_name}  (총 {len(stems)}프레임, GT박스 있는 프레임 {len(frames_with_gt)}개)")
    print("=" * 76)
    print(f"라이다 결측 프레임 수 : {n_missing_lidar} / {len(stems)}")
    if missing_stems:
        print(f"  결측 stem 예시: {missing_stems[:10]}")
    raw = range_stats["n_lidar_raw"]
    removed = range_stats["n_lidar_range_removed"]
    removed_ratio = (removed / raw * 100.0) if raw > 0 else 0.0
    print(f"LiDAR r<{MIN_LIDAR_RANGE:.1f}m 필터 제거 : {removed} / {raw} pts ({removed_ratio:.2f}%)")
    print("-" * 76)

    for cam in CAM_ORDER:
        n_frames_processed = len(stats[cam]["n_points_per_frame"])
        avg_pts = (
            sum(stats[cam]["n_points_per_frame"]) / n_frames_processed
            if n_frames_processed > 0 else 0.0
        )
        n_before = stats[cam]["n_before"]
        n_after = stats[cam]["n_after"]
        dedup_removed = n_before - n_after
        dedup_ratio = (dedup_removed / n_before * 100.0) if n_before > 0 else 0.0

        depths_cam = np.concatenate(all_depths[cam]) if all_depths[cam] else np.zeros((0,), dtype=np.float32)

        print(f"[{cam}]")
        print(f"  프레임당 평균 유효(dedup후) 포인트 수 : {avg_pts:.1f}")
        print(f"  dedup 전 in-frame 포인트 합계        : {n_before}")
        print(f"  dedup 후 포인트 합계                 : {n_after}")
        print(f"  dedup으로 제거된 비율                : {dedup_ratio:.2f}%  ({dedup_removed} pts)")
        if depths_cam.size > 0:
            print(f"  depth 분포: min={depths_cam.min():.2f}m  median={np.median(depths_cam):.2f}m  max={depths_cam.max():.2f}m")
            print(make_histogram_text(depths_cam))
        else:
            print("  depth 분포: (포인트 없음)")
        print("-" * 76)

    print(f"검증 오버레이 대상 프레임(무작위 {len(picked)}개): {picked}")
    if saved_paths:
        print(f"검증 오버레이 저장 경로: {os.path.dirname(saved_paths[0])}")
    for p in saved_paths:
        print(f"  - {p}")
    print("=" * 76)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scen", default="scen36")
    ap.add_argument("--all", action="store_true", help="labels_3d_v2/lidar/3캠이 모두 있는 모든 scen 처리")
    ap.add_argument("--num-verify-frames", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260707)
    args = ap.parse_args()

    scens = discover_complete_scens(DATASET_ROOT) if args.all else [args.scen]
    if not scens:
        raise SystemExit("[ERROR] 처리 가능한 scen을 찾지 못함")

    t0 = time.time()
    totals = {
        "frames": 0,
        "missing_lidar": 0,
        "lidar_raw": 0,
        "lidar_range_removed": 0,
        "cam_after": {cam: 0 for cam in CAM_ORDER},
    }

    for i, scen in enumerate(scens, 1):
        scen_dir = os.path.join(DATASET_ROOT, scen)
        if not os.path.isdir(scen_dir):
            raise SystemExit(f"[ERROR] scen 폴더 없음: {scen_dir}")

        print(f"[generate_depth_gt] ({i}/{len(scens)}) scen = {scen}  ({scen_dir})")
        stems, stats, all_depths, n_missing_lidar, missing_stems, frames_with_gt, range_stats = process_scen(scen_dir, scen)

        verify_out_dir = os.path.join(scen_dir, "depth_gt_verify")
        picked, saved_paths = make_verify_overlays(
            scen_dir, scen, frames_with_gt, verify_out_dir,
            num_frames=args.num_verify_frames, seed=args.seed + i - 1,
        )

        totals["frames"] += len(stems)
        totals["missing_lidar"] += n_missing_lidar
        totals["lidar_raw"] += range_stats["n_lidar_raw"]
        totals["lidar_range_removed"] += range_stats["n_lidar_range_removed"]
        for cam in CAM_ORDER:
            totals["cam_after"][cam] += stats[cam]["n_after"]

        if args.all:
            raw = range_stats["n_lidar_raw"]
            removed = range_stats["n_lidar_range_removed"]
            removed_ratio = (removed / raw * 100.0) if raw > 0 else 0.0
            cam_summary = ", ".join(f"{cam}={stats[cam]['n_after']}" for cam in CAM_ORDER)
            print(
                f"  -> frames={len(stems)} missing_lidar={n_missing_lidar} "
                f"range_removed={removed_ratio:.2f}% depth_pts({cam_summary}) "
                f"verify={len(saved_paths)} files"
            )
        else:
            print_report(scen, stems, stats, all_depths, n_missing_lidar, missing_stems, frames_with_gt, range_stats, picked, saved_paths)

    if args.all:
        raw = totals["lidar_raw"]
        removed = totals["lidar_range_removed"]
        removed_ratio = (removed / raw * 100.0) if raw > 0 else 0.0
        print()
        print("=" * 76)
        print(f"  전체 depth GT 생성 완료: {len(scens)}개 scen, {totals['frames']}프레임, {time.time() - t0:.1f}초")
        print("=" * 76)
        print(f"라이다 결측 프레임 수          : {totals['missing_lidar']} / {totals['frames']}")
        print(f"LiDAR r<{MIN_LIDAR_RANGE:.1f}m 필터 제거 : {removed} / {raw} pts ({removed_ratio:.2f}%)")
        for cam in CAM_ORDER:
            print(f"{cam} dedup 후 depth 포인트 합계 : {totals['cam_after'][cam]}")
        print("=" * 76)


if __name__ == "__main__":
    main()
