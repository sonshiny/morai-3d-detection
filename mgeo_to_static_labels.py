#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import bisect
import argparse
import numpy as np
import rosbag
from scipy.interpolate import interp1d

MGEO_DIR = './mgeo_data'
DATASET_DIR = '/data/dataset'
POINTS_PER_LINE = 20
MAX_RANGE = 110.0


def get_msg_time(msg, bag_time_sec):
    try:
        if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
            stamp = msg.header.stamp
            if hasattr(stamp, 'to_sec'):
                sec = float(stamp.to_sec())
                if sec > 0:
                    return sec
    except Exception:
        pass
    return float(bag_time_sec)


def resample_polyline_2d(points_2d, num_points=POINTS_PER_LINE):
    points_2d = np.array(points_2d, dtype=np.float32)
    if len(points_2d) < 2:
        return None

    seg = np.sqrt(np.sum(np.diff(points_2d, axis=0) ** 2, axis=1))
    distances = np.insert(np.cumsum(seg), 0, 0.0)
    if distances[-1] == 0:
        return None

    alpha = np.linspace(0.0, distances[-1], num_points)
    resampled = np.zeros((num_points, 2), dtype=np.float32)
    for i in range(2):
        interpolator = interp1d(distances, points_2d[:, i], kind='linear')
        resampled[:, i] = interpolator(alpha)
    return resampled


def load_mgeo_polylines(mgeo_dir):
    files_to_load = [
        ('lane_boundary_set.json', 0),
        ('crosswalk_set.json', 1),
        ('road_boundary_set.json', 2),
    ]

    global_lines = []
    for file_name, class_id in files_to_load:
        path = os.path.join(mgeo_dir, file_name)
        if not os.path.exists(path):
            print(f'[WARN] {file_name} 없음 → 스킵')
            continue

        with open(path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f'[ERROR] {file_name} 읽기 실패: {e}')
                continue

        items = data if isinstance(data, list) else data.get('features', [data])

        loaded = 0
        for item in items:
            points = None

            if isinstance(item, dict):
                points = item.get('points')
                if points is None:
                    points = item.get('geometry', {}).get('coordinates')
            elif isinstance(item, list):
                points = item

            if points is None:
                continue

            try:
                pts_np = np.array(points, dtype=np.float32)
                if pts_np.ndim < 2 or pts_np.shape[0] < 2:
                    continue
                pts_2d = pts_np[:, :2]
                global_lines.append({'class': class_id, 'points': pts_2d})
                loaded += 1
            except Exception:
                continue

        print(f'[MGeo] {file_name:24s}: {loaded:,}개 로드')

    print(f'[MGeo] 총 {len(global_lines):,}개 폴리라인 로드 완료')
    return global_lines


def transform_to_ego_centric_2d(global_lines, ego_pos, ego_heading_deg, max_range=MAX_RANGE):
    ego_centric_lines = []

    yaw = np.radians(ego_heading_deg)
    c, s = np.cos(yaw), np.sin(yaw)

    for line in global_lines:
        pts = line['points']
        dx = pts[:, 0] - ego_pos[0]
        dy = pts[:, 1] - ego_pos[1]

        dists = np.sqrt(dx ** 2 + dy ** 2)
        if np.min(dists) > max_range:
            continue

        x_e = c * dx + s * dy
        y_e = -s * dx + c * dy
        xy_ego = np.stack([x_e, y_e], axis=1)

        resampled = resample_polyline_2d(xy_ego, POINTS_PER_LINE)
        if resampled is not None:
            ego_centric_lines.append({'class': line['class'], 'points': resampled})

    return ego_centric_lines


def load_ego_cache(bag_path):
    ts_list = []
    ego_list = []
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=['/Ego_topic']):
            ts = get_msg_time(msg, t.to_sec())
            ts_list.append(ts)
            ego_list.append({
                'ts': ts,
                'pos': [msg.position.x, msg.position.y],
                'heading': msg.heading,
            })
    return ts_list, ego_list


def find_closest_ego(ts_list, ego_list, target_ts, max_gap=0.10):
    if not ts_list:
        return None

    idx = bisect.bisect_left(ts_list, target_ts)
    candidates = []
    if idx < len(ts_list):
        candidates.append(idx)
    if idx - 1 >= 0:
        candidates.append(idx - 1)

    if not candidates:
        return None

    best_idx = min(candidates, key=lambda i: abs(ts_list[i] - target_ts))
    if abs(ts_list[best_idx] - target_ts) > max_gap:
        return None
    return ego_list[best_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mgeo_dir', default=MGEO_DIR)
    parser.add_argument('--dataset_dir', '-d', default=DATASET_DIR)
    parser.add_argument('--bag_dir', default='.',
                        help='frame_groups.json 안의 bag_file들이 있는 폴더')
    parser.add_argument('--max_sync_gap', type=float, default=0.10)
    args = parser.parse_args()

    out_dir = os.path.join(args.dataset_dir, 'labels_static')
    os.makedirs(out_dir, exist_ok=True)

    groups_path = os.path.join(args.dataset_dir, 'frame_groups.json')
    if not os.path.isfile(groups_path):
        raise FileNotFoundError(
            f'[ERROR] {groups_path} 없음. 먼저 python3 build_frame_groups.py 를 실행하세요.'
        )

    print('🚀 정적 맵 라벨 생성 시작...')
    global_lines = load_mgeo_polylines(args.mgeo_dir)

    with open(groups_path, 'r', encoding='utf-8') as f:
        groups = json.load(f)
    print(f'[frame_groups] {len(groups):,}개 프레임 그룹 로드')

    ego_cache = {}
    written = 0
    empty = 0
    sync_fail = 0

    for i, group in enumerate(groups):
        bag_file = group.get('bag_file')
        if not bag_file:
            bag_key = group.get('bag_key')
            if not bag_key:
                sync_fail += 1
                continue
            bag_file = f'{bag_key}.bag'

        if bag_file not in ego_cache:
            bag_path = os.path.join(args.bag_dir, bag_file)
            if not os.path.isfile(bag_path):
                raise FileNotFoundError(f'[ERROR] bag 파일 없음: {bag_path}')
            ts_list, ego_list = load_ego_cache(bag_path)
            ego_cache[bag_file] = (ts_list, ego_list)
            print(f'[bag] {bag_file}: Ego 메시지 {len(ts_list):,}개 로드')

        ts_list, ego_list = ego_cache[bag_file]
        best_ego = find_closest_ego(ts_list, ego_list, group['ts'], args.max_sync_gap)
        if best_ego is None:
            sync_fail += 1
            continue

        ego_lines = transform_to_ego_centric_2d(
            global_lines,
            best_ego['pos'],
            best_ego['heading'],
            MAX_RANGE
        )

        out_file = os.path.join(out_dir, f"{group['label_stem']}.txt")
        with open(out_file, 'w', encoding='utf-8') as f:
            for line in ego_lines:
                pts_str = ' '.join(f'{p[0]:.2f} {p[1]:.2f}' for p in line['points'])
                f.write(f"{line['class']} {pts_str}\n")

        if ego_lines:
            written += 1
        else:
            empty += 1

        if i % 500 == 0:
            print(f'  진행: {i:5d} / {len(groups):5d}')

    print('\n✅ 완료!')
    print(f'   라벨 생성됨  : {written:,}개 프레임')
    print(f'   빈 프레임    : {empty:,}개 (범위 밖)')
    print(f'   동기화 실패  : {sync_fail:,}개')
    print(f'   출력 폴더    : {os.path.abspath(out_dir)}')


if __name__ == '__main__':
    main()
