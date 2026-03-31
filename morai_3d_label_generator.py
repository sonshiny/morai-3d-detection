#!/usr/bin/env python3
"""
=============================================================
  MORAI Simulation → 3D Label Generator (Ego 좌표계 기준)
=============================================================
출력 라벨 형식 (txt, 한 줄 = NPC 1대):
  class_id  x  y  z  ln_w  ln_l  ln_h  sin_yaw  cos_yaw  vx  vy  vz
  (총 12개 값, 공백 구분)

  - x, y, z      : Ego 좌표계 기준 NPC 중심 위치 (m)
  - ln_w/ln_l/ln_h: NPC 크기의 자연로그값 (SparseDrive 논문 포맷)
  - sin/cos_yaw  : NPC heading - Ego heading (상대 각도)
  - vx, vy, vz   : Ego 좌표계 기준 NPC 속도 (m/s)

NPC 위치 보정:
  MORAI의 npc.position은 후축(rear axle) 기준.
  offset_ratio로 차량 길이의 일정 비율만큼 전방 보정.
  예) offset_ratio=0.2, L=4.5m → 0.9m 전방으로 보정

사용법:
  python morai_3d_label_generator.py o.bag
  python morai_3d_label_generator.py o.bag --offset_ratio 0.2
  python morai_3d_label_generator.py o.bag --dataset_dir ./dataset
=============================================================
"""

import os
import sys
import argparse
import numpy as np
import rosbag

# 최대 타임스탬프 허용 오차 (초)
MAX_SYNC_GAP_SEC = 0.05

# NPC 타입 → 클래스 ID
NPC_TYPE_MAP = {1: 0, 2: 1, 3: 2}  # car=0, truck=1, bus=2

# 검출 범위 필터 (Ego 좌표계 기준)
MAX_RANGE_XY = 50.0
MAX_RANGE_Z  =  3.0

# NPC 타입별 기본 offset_ratio (경험값, --offset_ratio 인자로 덮어쓸 수 있음)
# 차종마다 후축~중심 비율이 다름
OFFSET_RATIOS = {
    1: 0.2,   # car
    2: 0.2,   # truck
    3: 0.2,   # bus
}


# ===========================================================
# 좌표 변환
# ===========================================================
def world_to_ego(npc_pos, npc_heading_deg, npc_velocity,
                 ego_pos, ego_heading_deg, npc_length, offset_ratio):
    """
    world 좌표 → ego 좌표 변환.
    offset_ratio: NPC 후축→중심 보정 (npc.position이 후축 기준이므로)
    """
    # 1. NPC 위치를 후축→중심으로 보정 (world 좌표계에서)
    offset_dist = npc_length * offset_ratio
    npc_yaw_rad = np.radians(npc_heading_deg)
    npc_pos_corrected = np.array([
        npc_pos[0] + offset_dist * np.cos(npc_yaw_rad),
        npc_pos[1] + offset_dist * np.sin(npc_yaw_rad),
        npc_pos[2]
    ])

    # 2. world → ego 변환
    dp = npc_pos_corrected - np.array(ego_pos)
    yaw = np.radians(ego_heading_deg)
    c, s = np.cos(yaw), np.sin(yaw)

    # ROS Standard: X=앞(Front), Y=왼(Left)
    x_e =  c * dp[0] + s * dp[1]
    y_e = -s * dp[0] + c * dp[1]
    z_e = dp[2]

    # 상대 heading (-90도 보정 적용)
    rel_yaw_raw = np.radians(npc_heading_deg - ego_heading_deg) - np.pi / 2
    rel_yaw = np.arctan2(np.sin(rel_yaw_raw), np.cos(rel_yaw_raw))

    vx_w, vy_w, vz_w = npc_velocity
    vx_e =  c * vx_w + s * vy_w
    vy_e = -s * vx_w + c * vy_w
    vz_e = vz_w

    return (np.array([x_e, y_e, z_e]),
            rel_yaw,
            np.array([vx_e, vy_e, vz_e]))


# ===========================================================
# 타임스탬프 동기화
# ===========================================================
def find_closest(msg_list, target_sec):
    if not msg_list:
        return None
    idx = min(range(len(msg_list)),
              key=lambda i: abs(msg_list[i][0] - target_sec))
    if abs(msg_list[idx][0] - target_sec) > MAX_SYNC_GAP_SEC:
        return None
    return msg_list[idx][1]


def build_image_timestamp_map(bag_path, cam_topics):
    ts_map = {t: [] for t in cam_topics}
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=cam_topics):
            if topic in ts_map:
                ts_map[topic].append(t.to_sec())
    return ts_map


# ===========================================================
# 메인
# ===========================================================
def generate_3d_labels(bag_path, dataset_dir, global_offset_ratio):
    print("=" * 60)
    print("  MORAI bag → 3D 라벨 생성기 (Ego 좌표계)")
    print(f"  global_offset_ratio = {global_offset_ratio}")
    print("=" * 60)

    img_dir = os.path.join(dataset_dir, 'images')
    lbl_dir = os.path.join(dataset_dir, 'labels_3d')
    os.makedirs(lbl_dir, exist_ok=True)

    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    if not img_files:
        print("[ERROR] images/ 폴더에 jpg 파일이 없습니다!")
        sys.exit(1)
    print(f"[이미지 파일] {len(img_files):,} 개")

    cam_topics_set = set()
    for f in img_files:
        stem = os.path.splitext(f)[0]
        parts = stem.rsplit('_', 1)
        if len(parts) == 2:
            cam_topics_set.add('/morai/' + parts[0])
    cam_topics = sorted(cam_topics_set)
    print(f"[카메라 토픽] {cam_topics}")

    print("\n[1/3] Ego / Object 토픽 로드 중...")
    ego_msgs = []
    obj_msgs = []
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(
                topics=['/Ego_topic', '/Object_topic']):
            ts = t.to_sec()
            if topic == '/Ego_topic':
                ego_msgs.append((ts, msg))
            elif topic == '/Object_topic':
                obj_msgs.append((ts, msg))
    print(f"   Ego  : {len(ego_msgs):,} 개")
    print(f"   Object: {len(obj_msgs):,} 개")

    print("\n[2/3] 카메라 타임스탬프 추출 중...")
    ts_map = build_image_timestamp_map(bag_path, cam_topics)

    print("\n[3/3] 3D 라벨 생성 중...")
    total_labels = 0
    total_empty  = 0
    sync_fail    = 0

    for img_file in img_files:
        stem  = os.path.splitext(img_file)[0]
        parts = stem.rsplit('_', 1)
        if len(parts) != 2:
            continue
        cam_short = parts[0]
        fidx      = int(parts[1])
        topic     = '/morai/' + cam_short

        if topic not in ts_map or fidx >= len(ts_map[topic]):
            sync_fail += 1
            continue

        ts = ts_map[topic][fidx]

        ego_msg = find_closest(ego_msgs, ts)
        obj_msg = find_closest(obj_msgs, ts)
        if ego_msg is None or obj_msg is None:
            sync_fail += 1
            continue

        ego_pos     = [ego_msg.position.x,
                       ego_msg.position.y,
                       ego_msg.position.z]
        ego_heading = ego_msg.heading

        lines = []
        for npc in obj_msg.npc_list:
            npc_pos  = [npc.position.x, npc.position.y, npc.position.z]
            npc_vel  = [npc.velocity.x, npc.velocity.y, npc.velocity.z]
            npc_size = [npc.size.x, npc.size.y, npc.size.z]  # w(x), l(y), h(z)
            npc_length = npc_size[1]  # MORAI: size.y = 길이(앞뒤)

            # offset_ratio: 인자로 넘어온 값 우선, 없으면 차종별 기본값
            ratio = global_offset_ratio if global_offset_ratio is not None \
                    else OFFSET_RATIOS.get(npc.type, 0.2)

            pos_ego, rel_yaw, vel_ego = world_to_ego(
                npc_pos, npc.heading, npc_vel,
                ego_pos, ego_heading,
                npc_length, ratio
            )

            # 범위 필터
            if (abs(pos_ego[0]) > MAX_RANGE_XY or
                abs(pos_ego[1]) > MAX_RANGE_XY or
                abs(pos_ego[2]) > MAX_RANGE_Z):
                continue

            cls_id  = NPC_TYPE_MAP.get(npc.type, 0)
            sin_yaw = float(np.sin(rel_yaw))
            cos_yaw = float(np.cos(rel_yaw))

            # SparseDrive 논문 포맷: 크기를 ln 스케일로 저장
            ln_w = float(np.log(max(npc_size[0], 0.01)))
            ln_l = float(np.log(max(npc_size[1], 0.01)))
            ln_h = float(np.log(max(npc_size[2], 0.01)))

            # z 좌표: 지면 기준 [0, H] → 중심은 H/2
            # pos_ego[2]는 world z차이 → 바닥 기준으로 보정
            z_center = pos_ego[2]

            line = (
                f"{cls_id} "
                f"{pos_ego[0]:.4f} {pos_ego[1]:.4f} {z_center:.4f} "
                f"{ln_w:.4f} {ln_l:.4f} {ln_h:.4f} "
                f"{sin_yaw:.4f} {cos_yaw:.4f} "
                f"{vel_ego[0]:.4f} {vel_ego[1]:.4f} {vel_ego[2]:.4f}"
            )
            lines.append(line)

        lbl_path = os.path.join(lbl_dir, f"{stem}.txt")
        with open(lbl_path, 'w') as f:
            f.write('\n'.join(lines))

        if lines:
            total_labels += len(lines)
        else:
            total_empty += 1

        if fidx % 100 == 0 and cam_short == 'cam_front':
            print(f"   [cam_front] frame {fidx:05d} | NPC {len(lines)}개")

    print("\n✅ 완료!")
    print(f"   총 NPC 라벨  : {total_labels:,} 개")
    print(f"   빈 프레임    : {total_empty:,} 개")
    print(f"   동기화 실패  : {sync_fail:,} 개")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bag', help='.bag 파일 경로')
    parser.add_argument('--dataset_dir', '-d', default='./dataset')
    parser.add_argument('--offset_ratio', type=float, default=None,
                        help='NPC 후축→중심 보정 비율 (기본: 차종별 0.2)')
    args = parser.parse_args()

    if not os.path.isfile(args.bag):
        print(f"[ERROR] bag 파일 없음: {args.bag}")
        sys.exit(1)

    generate_3d_labels(args.bag, args.dataset_dir, args.offset_ratio)