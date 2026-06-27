#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_frame_groups.py
멀티-bag 통합 데이터셋용 frame_groups.json 생성기.

핵심 변경점:
- 여러 bag을 한 번에 처리
- stem에 bag_key를 포함
- cam_front 기준 label_stem 생성
- 현재 bag 범위 안에서만 타임스탬프 동기화

출력:
  dataset/frame_groups.json

사용법:
  python3 build_frame_groups.py
  python3 build_frame_groups.py scenario1.bag scenario3.bag
"""

import os
import re
import sys
import json
import bisect
import argparse
import rosbag

DEFAULT_BAGS = [
    'scenario1.bag', 'scenario1_1.bag', 'scenario1_1.2.bag', 'scenario1_1.3.bag',
    'scenario3.bag', 'scenario322.bag', 'scenario4.bag', 'scenario6.bag',
    'scenario7.bag', 'scenario7.bag_2.bag', 'scenario7.2_5.bag'
]

CAM_TOPICS = {
    '/morai/cam_front':       'cam_front',
    '/morai/cam_front_left':  'cam_front_left',
    '/morai/cam_front_right': 'cam_front_right',
}

SYNC_THRESHOLD = 0.05


def bag_to_key(bag_path: str) -> str:
    base = os.path.basename(bag_path)
    return base[:-4] if base.endswith('.bag') else base


def closest_index(ts_list, target_sec, max_gap=SYNC_THRESHOLD):
    if not ts_list:
        return None
    idx = bisect.bisect_left(ts_list, target_sec)
    candidates = []
    if idx < len(ts_list):
        candidates.append(idx)
    if idx - 1 >= 0:
        candidates.append(idx - 1)
    if not candidates:
        return None
    best_idx = min(candidates, key=lambda i: abs(ts_list[i] - target_sec))
    if abs(ts_list[best_idx] - target_sec) > max_gap:
        return None
    return best_idx


def build_camera_timestamp_map(bag_path: str):
    cam_ts = {cam_name: [] for cam_name in CAM_TOPICS.values()}
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=list(CAM_TOPICS.keys())):
            cam_name = CAM_TOPICS.get(topic)
            if cam_name is not None:
                cam_ts[cam_name].append(t.to_sec())
    return cam_ts


def collect_existing_stems(img_dir: str, lbl_dir: str, bag_key: str):
    bag_pat = re.compile(rf'^(?P<cam>.+)__{re.escape(bag_key)}__(?P<idx>\d{{5}})$')

    img_stems = set()
    for fname in os.listdir(img_dir):
        if not fname.endswith('.jpg'):
            continue
        stem = os.path.splitext(fname)[0]
        if bag_pat.match(stem):
            img_stems.add(stem)

    lbl_stems = set()
    if os.path.isdir(lbl_dir):
        for fname in os.listdir(lbl_dir):
            if not fname.endswith('.txt'):
                continue
            stem = os.path.splitext(fname)[0]
            if bag_pat.match(stem):
                lbl_stems.add(stem)

    return img_stems, lbl_stems


def build_groups_for_bag(bag_path: str, dataset_dir: str):
    img_dir = os.path.join(dataset_dir, 'images')
    lbl_dir = os.path.join(dataset_dir, 'labels_3d')
    bag_key = bag_to_key(bag_path)

    print('=' * 72)
    print(f'[frame_groups] {bag_path}  → bag_key={bag_key}')
    print('=' * 72)

    cam_ts = build_camera_timestamp_map(bag_path)
    for cam, ts_list in cam_ts.items():
        print(f'  - {cam:16s}: {len(ts_list):6d} 프레임')

    img_stems, lbl_stems = collect_existing_stems(img_dir, lbl_dir, bag_key)
    print(f'[이미지 stem] {len(img_stems):,} 개')
    print(f'[라벨 stem]   {len(lbl_stems):,} 개')

    groups = []
    skip_count = 0

    front_ts = cam_ts['cam_front']
    for fidx, ts_ref in enumerate(front_ts):
        front_stem = f'cam_front__{bag_key}__{fidx:05d}'

        if front_stem not in img_stems:
            skip_count += 1
            continue
        if front_stem not in lbl_stems:
            skip_count += 1
            continue

        group = {
            'ts': ts_ref,
            'bag_key': bag_key,
            'bag_file': os.path.basename(bag_path),
            'front_idx': fidx,
            'cams': {'cam_front': front_stem},
            'label_stem': front_stem,
        }

        for cam_name, ts_list in cam_ts.items():
            if cam_name == 'cam_front':
                continue

            best_idx = closest_index(ts_list, ts_ref, SYNC_THRESHOLD)
            if best_idx is None:
                continue

            stem_cam = f'{cam_name}__{bag_key}__{best_idx:05d}'
            if stem_cam in img_stems:
                group['cams'][cam_name] = stem_cam

        groups.append(group)

    print(f'[생성 그룹] {len(groups):,} 개')
    print(f'[스킵 프레임] {skip_count:,} 개')

    cam_coverage = {cam: 0 for cam in cam_ts.keys()}
    for g in groups:
        for cam in g['cams']:
            cam_coverage[cam] += 1

    for cam, cnt in cam_coverage.items():
        pct = (cnt / len(groups) * 100.0) if groups else 0.0
        print(f'  - coverage {cam:16s}: {cnt:6d} / {len(groups):6d} ({pct:5.1f}%)')

    print()
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bags', nargs='*', help='.bag 파일 경로들 (생략 시 기본 목록 전체)')
    parser.add_argument('--dataset_dir', '-d', default='/data/dataset')
    args = parser.parse_args()

    bag_paths = args.bags if args.bags else DEFAULT_BAGS

    for bag_path in bag_paths:
        if not os.path.isfile(bag_path):
            print(f'[ERROR] bag 파일 없음: {bag_path}')
            sys.exit(1)

    all_groups = []
    for bag_path in bag_paths:
        all_groups.extend(build_groups_for_bag(bag_path, args.dataset_dir))

    out_path = os.path.join(args.dataset_dir, 'frame_groups.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_groups, f, indent=2, ensure_ascii=False)

    print('=' * 72)
    print('✅ frame_groups.json 생성 완료')
    print(f'[출력] {os.path.abspath(out_path)}')
    print(f'[총 그룹 수] {len(all_groups):,}')
    print('=' * 72)


if __name__ == '__main__':
    main()
