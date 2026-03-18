#!/usr/bin/env python3
"""
=============================================================
  MORAI Simulation → 3D Label Generator (Ego 좌표계 기준)
=============================================================
출력 라벨 형식 (txt, 한 줄 = NPC 1대):
  class_id  x  y  z  w  l  h  sin_yaw  cos_yaw  vx  vy  vz
  (총 12개 값, 공백 구분)

  - x, y, z   : Ego 좌표계 기준 NPC 위치 (m)
  - w, l, h   : NPC 크기 (너비, 길이, 높이)
  - sin/cos   : NPC heading - Ego heading (상대 각도)
  - vx, vy, vz: Ego 좌표계 기준 NPC 속도 (m/s)

사용법:
  python morai_3d_label_generator.py /path/to/your.bag
  python morai_3d_label_generator.py /path/to/your.bag --dataset_dir ./dataset
=============================================================
"""

import os
import sys
import argparse
import numpy as np
import rosbag

# 최대 타임스탬프 허용 오차 (초)
MAX_SYNC_GAP_SEC = 0.1

# NPC 타입 → 클래스 ID
NPC_TYPE_MAP = {1: 0, 2: 1, 3: 2}  # car=0, truck=1, bus=2

# 검출 범위 필터 (Ego 좌표계 기준)
MAX_RANGE_XY = 50.0   # 앞뒤 좌우 50m 이내만
MAX_RANGE_Z  =  3.0   # 높이 3m 이내만


# ===========================================================
# 좌표 변환 유틸
# ===========================================================
def world_to_ego(npc_pos, npc_heading_deg, npc_velocity,
                 ego_pos, ego_heading_deg):
    """
    월드 좌표계 → Ego 좌표계 변환

    반환:
      pos_ego  : [x, y, z] Ego 기준 NPC 위치
      rel_yaw  : NPC heading - Ego heading (라디안)
      vel_ego  : [vx, vy, vz] Ego 기준 NPC 속도
    """
    # ── 위치 변환 ──────────────────────────────────────────
    dp = np.array(npc_pos) - np.array(ego_pos)
    ego_yaw = np.radians(ego_heading_deg)
    cos_y, sin_y = np.cos(ego_yaw), np.sin(ego_yaw)

    # Ego 차량 방향 기준으로 회전 (Z축 역회전)
    x_e =  cos_y * dp[0] + sin_y * dp[1]
    y_e = -sin_y * dp[0] + cos_y * dp[1]
    z_e = dp[2]

    # ── 상대 heading ───────────────────────────────────────
    rel_yaw = np.radians(npc_heading_deg - ego_heading_deg)

    # ── 속도 변환 ──────────────────────────────────────────
    vx_w, vy_w, vz_w = npc_velocity
    vx_e =  cos_y * vx_w + sin_y * vy_w
    vy_e = -sin_y * vx_w + cos_y * vy_w
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


# ===========================================================
# 이미지 파일명 → 타임스탬프 매핑 (bag에서 추출)
# ===========================================================
def build_image_timestamp_map(bag_path, cam_topics):
    """
    각 카메라 토픽의 프레임 순서별 타임스탬프를 추출.
    반환: {topic: [ts0, ts1, ts2, ...]}
    """
    ts_map = {t: [] for t in cam_topics}
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=cam_topics):
            if topic in ts_map:
                ts_map[topic].append(t.to_sec())
    return ts_map


# ===========================================================
# 메인 처리 함수
# ===========================================================
def generate_3d_labels(bag_path, dataset_dir):
    print("=" * 60)
    print("  MORAI bag → 3D 라벨 생성기 (Ego 좌표계)")
    print("=" * 60)
    print(f"[입력] {bag_path}")
    print(f"[출력] {dataset_dir}/labels_3d/")
    print()

    img_dir = os.path.join(dataset_dir, 'images')
    lbl_dir = os.path.join(dataset_dir, 'labels_3d')
    os.makedirs(lbl_dir, exist_ok=True)

    # 이미지 파일 목록 (이미 존재하는 것 기준)
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    if not img_files:
        print("[ERROR] images/ 폴더에 jpg 파일이 없습니다!")
        sys.exit(1)
    print(f"[이미지 파일] {len(img_files):,} 개 발견")

    # 카메라 토픽 목록 추출 (파일명에서)
    cam_topics_set = set()
    for f in img_files:
        stem = os.path.splitext(f)[0]
        # 예: cam_front_00123 → /morai/cam_front
        parts = stem.rsplit('_', 1)
        if len(parts) == 2:
            cam_topics_set.add('/morai/' + parts[0])
    cam_topics = sorted(cam_topics_set)
    print(f"[카메라 토픽] {cam_topics}")

    # ── Ego / Object 메시지 사전 로드 ────────────────────────
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

    print(f"   Ego 메시지   : {len(ego_msgs):,} 개")
    print(f"   Object 메시지: {len(obj_msgs):,} 개")

    # ── 카메라별 타임스탬프 추출 ────────────────────────────
    print("\n[2/3] 카메라 타임스탬프 추출 중...")
    ts_map = build_image_timestamp_map(bag_path, cam_topics)
    for t, tsl in ts_map.items():
        print(f"   {t}: {len(tsl):,} 프레임")

    # ── 3D 라벨 생성 ─────────────────────────────────────
    print("\n[3/3] 3D 라벨 생성 중...")

    total_labels  = 0
    total_empty   = 0
    sync_fail     = 0
    frame_cnt     = {}   # {cam_short_name: count}

    for img_file in img_files:
        stem     = os.path.splitext(img_file)[0]           # cam_front_00123
        parts    = stem.rsplit('_', 1)
        if len(parts) != 2:
            continue
        cam_short = parts[0]                               # cam_front
        fidx      = int(parts[1])                          # 123
        topic     = '/morai/' + cam_short                  # /morai/cam_front

        if topic not in ts_map or fidx >= len(ts_map[topic]):
            sync_fail += 1
            continue

        ts = ts_map[topic][fidx]

        # 동기화
        ego_msg = find_closest(ego_msgs, ts)
        obj_msg = find_closest(obj_msgs, ts)
        if ego_msg is None or obj_msg is None:
            sync_fail += 1
            continue

        ego_pos     = [ego_msg.position.x,
                       ego_msg.position.y,
                       ego_msg.position.z]
        ego_heading = ego_msg.heading

        # NPC 3D 라벨 생성
        lines = []
        for npc in obj_msg.npc_list:
            npc_pos = [npc.position.x, npc.position.y, npc.position.z]
            npc_vel = [npc.velocity.x, npc.velocity.y, npc.velocity.z]
            npc_size = [npc.size.x, npc.size.y, npc.size.z]   # w, l, h

            pos_ego, rel_yaw, vel_ego = world_to_ego(
                npc_pos, npc.heading, npc_vel,
                ego_pos, ego_heading
            )

            # 범위 필터 (너무 멀거나 높은 건 제외)
            if (abs(pos_ego[0]) > MAX_RANGE_XY or
                abs(pos_ego[1]) > MAX_RANGE_XY or
                abs(pos_ego[2]) > MAX_RANGE_Z):
                continue

            cls_id  = NPC_TYPE_MAP.get(npc.type, 0)
            sin_yaw = float(np.sin(rel_yaw))
            cos_yaw = float(np.cos(rel_yaw))

            line = (
                f"{cls_id} "
                f"{pos_ego[0]:.4f} {pos_ego[1]:.4f} {pos_ego[2]:.4f} "
                f"{npc_size[0]:.4f} {npc_size[1]:.4f} {npc_size[2]:.4f} "
                f"{sin_yaw:.4f} {cos_yaw:.4f} "
                f"{vel_ego[0]:.4f} {vel_ego[1]:.4f} {vel_ego[2]:.4f}"
            )
            lines.append(line)

        # 저장
        lbl_path = os.path.join(lbl_dir, f"{stem}.txt")
        with open(lbl_path, 'w') as f:
            f.write('\n'.join(lines))

        if lines:
            total_labels += len(lines)
        else:
            total_empty += 1

        if fidx % 100 == 0 and cam_short == 'cam_front':
            print(f"   [cam_front] frame {fidx:05d} | "
                  f"NPC {len(lines)}개 | "
                  f"ego({ego_pos[0]:.1f},{ego_pos[1]:.1f})")

    # 결과 요약
    print("\n✅ 완료!")
    print(f"   총 라벨 파일  : {len(img_files):,} 개")
    print(f"   총 NPC 라벨   : {total_labels:,} 개")
    print(f"   빈 프레임     : {total_empty:,} 개 (NPC 범위 밖)")
    print(f"   동기화 실패   : {sync_fail:,} 개")
    print(f"\n   📁 labels_3d/ → {os.path.abspath(lbl_dir)}")
    print("\n🚀 다음 단계: morai_dataset.py 에서 labels_3d/ 를 읽도록 설정!")


# ===========================================================
# 실행 진입점
# ===========================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MORAI bag → 3D 라벨 생성기'
    )
    parser.add_argument('bag',
                        help='.bag 파일 경로')
    parser.add_argument('--dataset_dir', '-d',
                        default='./dataset',
                        help='dataset 폴더 경로 (기본: ./dataset)')
    args = parser.parse_args()

    if not os.path.isfile(args.bag):
        print(f"[ERROR] bag 파일을 찾을 수 없습니다: {args.bag}")
        sys.exit(1)

    generate_3d_labels(args.bag, args.dataset_dir)
