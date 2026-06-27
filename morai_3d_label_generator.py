#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
morai_3d_label_generator.py
멀티-bag 통합 데이터셋용 3D GT 라벨 생성기 (single vehicle class)

핵심:
- 현재 bag에 해당하는 cam_front 이미지들만 처리
- image stem: cam_front__{bag_key}__{frame_idx:05d}
- 동적 클래스는 단일 vehicle(0)
- timestamp는 msg.header.stamp 우선, 없으면 bag time 사용
- sync gap은 기본 0.10초

사용법:
  python3 morai_3d_label_generator.py
  python3 morai_3d_label_generator.py scenario1.bag
  python3 morai_3d_label_generator.py scenario1.bag scenario3.bag
  python3 morai_3d_label_generator.py --max_sync_gap 0.15
"""

import os
import re
import sys
import bisect
import argparse
import numpy as np
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

MAX_RANGE_XY = 50.0
MAX_RANGE_Z = 3.0

VALID_NPC_TYPES = {1}
VEHICLE_CLASS_ID = 0

OFFSET_RATIOS = {
    1: 0.2,
}


def bag_to_key(bag_path: str) -> str:
    base = os.path.basename(bag_path)
    return base[:-4] if base.endswith('.bag') else base


def get_msg_time(msg, bag_time_sec):
    """
    ROS message header.stamp가 있으면 그걸 우선 사용하고,
    없으면 bag time(t.to_sec()) 사용.
    """
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


def find_closest(ts_list, msg_list, target_sec, max_gap):
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
    return msg_list[best_idx]


def world_to_ego(npc_pos, npc_heading_deg, npc_velocity,
                 ego_pos, ego_heading_deg, npc_length, offset_ratio):
    offset_dist = npc_length * offset_ratio
    npc_yaw_rad = np.radians(npc_heading_deg)
    npc_pos_corrected = np.array([
        npc_pos[0] + offset_dist * np.cos(npc_yaw_rad),
        npc_pos[1] + offset_dist * np.sin(npc_yaw_rad),
        npc_pos[2]
    ], dtype=np.float32)

    dp = npc_pos_corrected - np.array(ego_pos, dtype=np.float32)
    yaw = np.radians(ego_heading_deg)
    c, s = np.cos(yaw), np.sin(yaw)

    x_e = c * dp[0] + s * dp[1]
    y_e = -s * dp[0] + c * dp[1]
    z_e = dp[2]

    rel_yaw_raw = np.radians(npc_heading_deg - ego_heading_deg) - np.pi / 2.0
    rel_yaw = np.arctan2(np.sin(rel_yaw_raw), np.cos(rel_yaw_raw))

    vx_w, vy_w, vz_w = npc_velocity
    vx_e = c * vx_w + s * vy_w
    vy_e = -s * vx_w + c * vy_w
    vz_e = vz_w

    return np.array([x_e, y_e, z_e], dtype=np.float32), rel_yaw, \
        np.array([vx_e, vy_e, vz_e], dtype=np.float32)


def load_front_image_stems(img_dir: str, bag_key: str):
    pattern = re.compile(rf'^cam_front__{re.escape(bag_key)}__(\d{{5}})\.jpg$')
    items = []
    for fname in os.listdir(img_dir):
        match = pattern.match(fname)
        if not match:
            continue
        frame_idx = int(match.group(1))
        stem = os.path.splitext(fname)[0]
        items.append((frame_idx, stem))
    items.sort(key=lambda x: x[0])
    return items


def build_camera_timestamp_map(bag_path: str):
    ts_map = {cam_name: [] for cam_name in CAM_TOPICS.values()}
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=list(CAM_TOPICS.keys())):
            cam_name = CAM_TOPICS.get(topic)
            if cam_name is None:
                continue
            ts = get_msg_time(msg, t.to_sec())
            ts_map[cam_name].append(ts)
    return ts_map


def generate_for_bag(bag_path: str, dataset_dir: str, global_offset_ratio, max_sync_gap: float):
    bag_key = bag_to_key(bag_path)
    img_dir = os.path.join(dataset_dir, 'images')
    lbl_dir = os.path.join(dataset_dir, 'labels_3d')
    os.makedirs(lbl_dir, exist_ok=True)

    print('=' * 72)
    print('  MORAI bag → 3D 라벨 생성기 (single vehicle class)')
    print(f'  bag_file            = {os.path.basename(bag_path)}')
    print(f'  bag_key             = {bag_key}')
    print(f'  global_offset_ratio = {global_offset_ratio}')
    print(f'  max_sync_gap        = {max_sync_gap}')
    print('=' * 72)

    front_images = load_front_image_stems(img_dir, bag_key)
    if not front_images:
        print(f'[ERROR] 현재 bag에 해당하는 cam_front 이미지가 없습니다: bag_key={bag_key}')
        return False

    print(f'[cam_front 이미지] {len(front_images):,} 개')

    print('\n[1/3] Ego / Object 토픽 로드 중...')
    ego_ts, ego_msgs = [], []
    obj_ts, obj_msgs = [], []

    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=['/Ego_topic', '/Object_topic']):
            ts = get_msg_time(msg, t.to_sec())
            if topic == '/Ego_topic':
                ego_ts.append(ts)
                ego_msgs.append(msg)
            elif topic == '/Object_topic':
                obj_ts.append(ts)
                obj_msgs.append(msg)

    print(f'   Ego   : {len(ego_ts):,} 개')
    print(f'   Object: {len(obj_ts):,} 개')

    print('\n[2/3] 카메라 타임스탬프 추출 중...')
    cam_ts_map = build_camera_timestamp_map(bag_path)
    front_ts = cam_ts_map['cam_front']
    print(f'   cam_front ts: {len(front_ts):,} 개')

    print('\n[3/3] 3D 라벨 생성 중...')
    total_labels = 0
    total_empty = 0
    sync_fail = 0
    written = 0

    for frame_idx, stem in front_images:
        if frame_idx >= len(front_ts):
            sync_fail += 1
            continue

        ts = front_ts[frame_idx]
        ego_msg = find_closest(ego_ts, ego_msgs, ts, max_sync_gap)
        obj_msg = find_closest(obj_ts, obj_msgs, ts, max_sync_gap)

        if ego_msg is None or obj_msg is None:
            sync_fail += 1
            continue

        ego_pos = [ego_msg.position.x, ego_msg.position.y, ego_msg.position.z]
        ego_heading = ego_msg.heading

        lines = []
        for npc in obj_msg.npc_list:
            if npc.type not in VALID_NPC_TYPES:
                continue

            npc_pos = [npc.position.x, npc.position.y, npc.position.z]
            npc_vel = [npc.velocity.x, npc.velocity.y, npc.velocity.z]
            npc_size = [npc.size.x, npc.size.y, npc.size.z]
            npc_length = npc_size[1]

            ratio = global_offset_ratio if global_offset_ratio is not None \
                else OFFSET_RATIOS.get(npc.type, 0.2)

            pos_ego, rel_yaw, vel_ego = world_to_ego(
                npc_pos, npc.heading, npc_vel,
                ego_pos, ego_heading,
                npc_length, ratio
            )

            if (abs(pos_ego[0]) > MAX_RANGE_XY or
                abs(pos_ego[1]) > MAX_RANGE_XY or
                abs(pos_ego[2]) > MAX_RANGE_Z):
                continue

            cls_id = VEHICLE_CLASS_ID
            sin_yaw = float(np.sin(rel_yaw))
            cos_yaw = float(np.cos(rel_yaw))
            ln_w = float(np.log(max(npc_size[0], 0.01)))
            ln_l = float(np.log(max(npc_size[1], 0.01)))
            ln_h = float(np.log(max(npc_size[2], 0.01)))
            z_center = float(pos_ego[2])

            line = (
                f'{cls_id} '
                f'{float(pos_ego[0]):.4f} {float(pos_ego[1]):.4f} {z_center:.4f} '
                f'{ln_w:.4f} {ln_l:.4f} {ln_h:.4f} '
                f'{sin_yaw:.4f} {cos_yaw:.4f} '
                f'{float(vel_ego[0]):.4f} {float(vel_ego[1]):.4f} {float(vel_ego[2]):.4f}'
            )
            lines.append(line)

        lbl_path = os.path.join(lbl_dir, f'{stem}.txt')
        with open(lbl_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        written += 1
        if lines:
            total_labels += len(lines)
        else:
            total_empty += 1

        if frame_idx % 100 == 0:
            print(f'   [cam_front] frame {frame_idx:05d} | vehicle {len(lines)}개')

    print('\n✅ 완료!')
    print(f'   bag                : {os.path.basename(bag_path)}')
    print(f'   라벨 파일 생성     : {written:,} 개')
    print(f'   총 vehicle 라벨    : {total_labels:,} 개')
    print(f'   빈 프레임          : {total_empty:,} 개')
    print(f'   동기화 실패        : {sync_fail:,} 개')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bags', nargs='*', help='.bag 파일 경로들 (생략 시 기본 목록 전체)')
    parser.add_argument('--dataset_dir', '-d', default='/data/dataset')
    parser.add_argument('--offset_ratio', type=float, default=None,
                        help='NPC 후축→중심 보정 비율 (기본: 차종별 0.2)')
    parser.add_argument('--max_sync_gap', type=float, default=0.10,
                        help='Ego/Object와 cam_front 동기화 허용 오차(초)')
    args = parser.parse_args()

    bag_paths = args.bags if args.bags else DEFAULT_BAGS

    for bag_path in bag_paths:
        if not os.path.isfile(bag_path):
            print(f'[ERROR] bag 파일 없음: {bag_path}')
            sys.exit(1)

    ok = True
    for bag_path in bag_paths:
        ok = generate_for_bag(
            bag_path=bag_path,
            dataset_dir=args.dataset_dir,
            global_offset_ratio=args.offset_ratio,
            max_sync_gap=args.max_sync_gap
        ) and ok

    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
