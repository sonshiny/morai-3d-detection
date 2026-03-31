#!/usr/bin/env python3
"""
mgeo_to_static_labels.py
좌표변환: morai_3d_label_generator.py와 동일한 수식 사용
출력 포맷: class_id pt1_x pt1_y pt2_x pt2_y ... pt20_x pt20_y (총 41개 값)
"""

import os
import json
import numpy as np
import rosbag
from scipy.interpolate import interp1d

MGEO_DIR    = './mgeo_data'
BAG_FILE    = 'o.bag'
DATASET_DIR = './dataset'
OUT_DIR     = os.path.join(DATASET_DIR, 'labels_static')
os.makedirs(OUT_DIR, exist_ok=True)

# static_decoder.py 앵커 범위와 일치: ±110m
MAX_RANGE       = 110.0
POINTS_PER_LINE = 20


def resample_polyline_2d(points_2d, num_points=20):
    points_2d = np.array(points_2d)
    if len(points_2d) < 2:
        return None
    distances = np.cumsum(
        np.sqrt(np.sum(np.diff(points_2d, axis=0) ** 2, axis=1))
    )
    distances = np.insert(distances, 0, 0)
    if distances[-1] == 0:
        return None
    alpha = np.linspace(0, distances[-1], num_points)
    resampled = np.zeros((num_points, 2))
    for i in range(2):
        interpolator = interp1d(distances, points_2d[:, i], kind='linear')
        resampled[:, i] = interpolator(alpha)
    return resampled


def load_mgeo_polylines(mgeo_dir):
    global_lines = []
    files_to_load = [
        ('lane_boundary_set.json', 0),
        ('crosswalk_set.json',     1),
    ]
    for file_name, class_id in files_to_load:
        path = os.path.join(mgeo_dir, file_name)
        if not os.path.exists(path):
            print(f"[WARN] {file_name} 없음 → 스킵")
            continue
        with open(path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"[ERROR] {file_name} 읽기 실패: {e}")
                continue

        items = data if isinstance(data, list) \
                else data.get('features', [data])

        for item in items:
            points = None
            if isinstance(item, dict):
                points = item.get('points')
                if points is None:
                    points = item.get('geometry', {}).get('coordinates')
            elif isinstance(item, list):
                points = item

            if points and isinstance(points, list) and len(points) >= 2:
                try:
                    pts_np = np.array(points)
                    if pts_np.ndim >= 2:
                        pts_2d = pts_np[:, :2]
                        global_lines.append(
                            {'class': class_id, 'points': pts_2d}
                        )
                except Exception:
                    continue

    print(f"[MGeo] 총 {len(global_lines)}개 폴리라인 로드 완료")
    return global_lines


def transform_to_ego_centric_2d(global_lines, ego_pos, ego_heading_deg,
                                  max_range=110.0):
    """
    world → ego 좌표 변환.
    morai_3d_label_generator.py와 동일한 수식 사용:
      x_e =  c*dp[0] + s*dp[1]
      y_e = -s*dp[0] + c*dp[1]
    (c=cos(yaw), s=sin(yaw), yaw=radians(ego_heading_deg))
    """
    ego_centric_lines = []

    yaw = np.radians(ego_heading_deg)
    c, s = np.cos(yaw), np.sin(yaw)

    for line in global_lines:
        pts = line['points']
        dx  = pts[:, 0] - ego_pos[0]
        dy  = pts[:, 1] - ego_pos[1]

        # 범위 필터: 모든 점이 max_range 밖이면 스킵
        dists = np.sqrt(dx ** 2 + dy ** 2)
        if np.min(dists) > max_range:
            continue

        # morai_3d_label_generator.py와 동일한 수식
        x_e =  c * dx + s * dy
        y_e = -s * dx + c * dy

        xy_ego = np.stack([x_e, y_e], axis=1)

        resampled = resample_polyline_2d(xy_ego, POINTS_PER_LINE)
        if resampled is not None:
            ego_centric_lines.append(
                {'class': line['class'], 'points': resampled}
            )

    return ego_centric_lines


def main():
    print("🚀 정적 맵 라벨 생성 시작...")
    global_lines = load_mgeo_polylines(MGEO_DIR)

    groups_path = os.path.join(DATASET_DIR, 'frame_groups.json')
    with open(groups_path, 'r') as f:
        groups = json.load(f)
    print(f"[frame_groups] {len(groups)}개 프레임 그룹 로드")

    print("[bag] Ego 메시지 로드 중...")
    bag = rosbag.Bag(BAG_FILE)
    ego_msgs = []
    for topic, msg, t in bag.read_messages(topics=['/Ego_topic']):
        ego_msgs.append({
            'ts':      t.to_sec(),
            'pos':     [msg.position.x, msg.position.y],
            'heading': msg.heading
        })
    bag.close()
    print(f"[bag] Ego 메시지 {len(ego_msgs)}개 로드 완료")

    written = 0
    empty   = 0

    for group in groups:
        best_ego = min(ego_msgs,
                       key=lambda e: abs(e['ts'] - group['ts']))
        ego_lines = transform_to_ego_centric_2d(
            global_lines,
            best_ego['pos'],
            best_ego['heading'],
            MAX_RANGE
        )

        out_file = os.path.join(OUT_DIR, f"{group['label_stem']}.txt")
        with open(out_file, 'w') as f:
            for line in ego_lines:
                pts_str = " ".join(
                    [f"{p[0]:.2f} {p[1]:.2f}" for p in line['points']]
                )
                f.write(f"{line['class']} {pts_str}\n")

        if ego_lines:
            written += 1
        else:
            empty += 1

    print(f"\n✅ 완료!")
    print(f"   라벨 생성됨  : {written}개 프레임")
    print(f"   빈 프레임    : {empty}개 (범위 밖)")
    print(f"   출력 폴더    : {os.path.abspath(OUT_DIR)}")


if __name__ == '__main__':
    main()