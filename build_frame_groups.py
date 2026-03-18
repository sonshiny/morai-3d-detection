#!/usr/bin/env python3
"""
=============================================================
  build_frame_groups.py
  bag 파일에서 카메라별 타임스탬프를 읽어
  같은 시간대의 6대 카메라를 1묶음으로 매핑하는
  frame_groups.json을 생성합니다.

사용법:
  python build_frame_groups.py /path/to/your.bag
  python build_frame_groups.py /path/to/your.bag --dataset_dir ./dataset
=============================================================
출력 예시 (frame_groups.json):
[
  {
    "ts": 1773824525.975,
    "cams": {
      "cam_front":       "cam_front_00000",
      "cam_back":        "cam_back_00000",
      "cam_front_left":  "cam_front_left_00000",
      "cam_front_right": "cam_front_right_00000",
      "cam_back_left":   "cam_back_left_00000",
      "cam_back_right":  "cam_back_right_00000"
    },
    "label_stem": "cam_front_00000"
  },
  ...
]
=============================================================
"""

import os
import sys
import json
import argparse
import rosbag

CAM_TOPICS = {
    '/morai/cam_front':       'cam_front',
    '/morai/cam_back':        'cam_back',
    '/morai/cam_front_left':  'cam_front_left',
    '/morai/cam_front_right': 'cam_front_right',
    '/morai/cam_back':        'cam_back',
    '/morai/cam_back_left':   'cam_back_left',
    '/morai/cam_back_right':  'cam_back_right',
}

# 그룹 매핑 허용 오차 (초)
SYNC_THRESHOLD = 0.05


def build_groups(bag_path, dataset_dir):
    print("=" * 60)
    print("  Frame Group Builder")
    print("=" * 60)
    print(f"[입력] {bag_path}")
    print(f"[출력] {dataset_dir}/frame_groups.json\n")

    img_dir = os.path.join(dataset_dir, 'images')
    lbl_dir = os.path.join(dataset_dir, 'labels_3d')

    # ── 1. 카메라별 타임스탬프 추출 ──────────────────────────
    print("[1/3] bag에서 카메라 타임스탬프 추출 중...")
    cam_ts = {v: [] for v in CAM_TOPICS.values()}  # {cam_name: [ts0, ts1, ...]}

    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=list(CAM_TOPICS.keys())):
            cam_name = CAM_TOPICS.get(topic)
            if cam_name:
                cam_ts[cam_name].append(t.to_sec())

    for cam, tsl in cam_ts.items():
        print(f"   {cam}: {len(tsl):,} 프레임")

    # ── 2. 이미지 파일 존재 확인 ─────────────────────────────
    print("\n[2/3] 이미지 파일 목록 확인 중...")
    existing = set(
        os.path.splitext(f)[0]
        for f in os.listdir(img_dir)
        if f.endswith('.jpg')
    )
    print(f"   총 {len(existing):,} 개 이미지 파일 발견")

    # ── 3. cam_front 기준으로 그룹 매핑 ─────────────────────
    print("\n[3/3] 타임스탬프 기반 그룹 매핑 중...")

    groups     = []
    skip_count = 0

    # cam_front 타임스탬프를 기준으로 그룹 생성
    for fidx, ts_ref in enumerate(cam_ts['cam_front']):
        stem_ref = f"cam_front_{fidx:05d}"

        # 이미지 파일 없으면 스킵
        if stem_ref not in existing:
            skip_count += 1
            continue

        # 라벨 파일 없으면 스킵
        if not os.path.isfile(os.path.join(lbl_dir, f"{stem_ref}.txt")):
            skip_count += 1
            continue

        group = {
            "ts":         ts_ref,
            "cams":       {},
            "label_stem": stem_ref   # GT는 cam_front 기준 (Ego 좌표계이므로 동일)
        }

        # cam_front는 무조건 포함
        group["cams"]["cam_front"] = stem_ref

        # 나머지 5대 카메라: ts_ref와 가장 가까운 프레임 찾기
        for cam_name, ts_list in cam_ts.items():
            if cam_name == 'cam_front' or not ts_list:
                continue

            # 가장 가까운 프레임 인덱스 찾기
            best_idx = min(range(len(ts_list)),
                           key=lambda i: abs(ts_list[i] - ts_ref))
            best_gap = abs(ts_list[best_idx] - ts_ref)

            if best_gap > SYNC_THRESHOLD:
                continue   # 너무 멀면 이 카메라는 None

            stem_cam = f"{cam_name}_{best_idx:05d}"
            if stem_cam in existing:
                group["cams"][cam_name] = stem_cam

        groups.append(group)

    print(f"   생성된 그룹 수: {len(groups):,} 개")
    print(f"   스킵된 프레임 : {skip_count:,} 개")

    # 카메라별 커버리지 통계
    cam_coverage = {cam: 0 for cam in cam_ts.keys()}
    for g in groups:
        for cam in g["cams"]:
            cam_coverage[cam] += 1
    print("\n   카메라별 커버리지:")
    for cam, cnt in cam_coverage.items():
        pct = cnt / len(groups) * 100 if groups else 0
        print(f"   {cam:20s}: {cnt:,} / {len(groups):,} ({pct:.1f}%)")

    # ── 저장 ─────────────────────────────────────────────────
    out_path = os.path.join(dataset_dir, 'frame_groups.json')
    with open(out_path, 'w') as f:
        json.dump(groups, f, indent=2)

    print(f"\n✅ 완료! → {out_path}")
    print("🚀 다음: python3 morai_dataset.py 로 데이터 로더 테스트!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Frame Group Builder')
    parser.add_argument('bag', help='.bag 파일 경로')
    parser.add_argument('--dataset_dir', '-d', default='./dataset')
    args = parser.parse_args()

    if not os.path.isfile(args.bag):
        print(f"[ERROR] bag 파일을 찾을 수 없습니다: {args.bag}")
        sys.exit(1)

    build_groups(args.bag, args.dataset_dir)
