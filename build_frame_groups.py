#!/usr/bin/env python3
"""
build_frame_groups.py
bag 파일에서 6개 카메라를 시간 기준으로 동기화 → frame_groups.json 생성

사용법:
  python build_frame_groups.py o.bag
  python build_frame_groups.py o.bag --dataset_dir ./dataset
"""

import os
import sys
import json
import argparse
import rosbag

# cam_back 중복 제거
CAM_TOPICS = {
    '/morai/cam_front':       'cam_front',
    '/morai/cam_front_left':  'cam_front_left',
    '/morai/cam_front_right': 'cam_front_right',
    '/morai/cam_back':        'cam_back',
    '/morai/cam_back_left':   'cam_back_left',
    '/morai/cam_back_right':  'cam_back_right',
}

SYNC_THRESHOLD = 0.05  # 초


def build_groups(bag_path, dataset_dir):
    print("=" * 60)
    print("  Frame Group Builder")
    print("=" * 60)
    print(f"[입력] {bag_path}")
    print(f"[출력] {dataset_dir}/frame_groups.json\n")

    img_dir = os.path.join(dataset_dir, 'images')
    lbl_dir = os.path.join(dataset_dir, 'labels_3d')

    # 1. 카메라별 타임스탬프 추출
    print("[1/3] 카메라 타임스탬프 추출 중...")
    cam_ts = {v: [] for v in CAM_TOPICS.values()}

    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=list(CAM_TOPICS.keys())):
            cam_name = CAM_TOPICS.get(topic)
            if cam_name:
                cam_ts[cam_name].append(t.to_sec())

    for cam, tsl in cam_ts.items():
        print(f"   {cam}: {len(tsl):,} 프레임")

    # 2. 이미지 파일 존재 확인
    print("\n[2/3] 이미지 파일 목록 확인 중...")
    existing = set(
        os.path.splitext(f)[0]
        for f in os.listdir(img_dir)
        if f.endswith('.jpg')
    )
    print(f"   총 {len(existing):,} 개 이미지 파일")

    # 3. cam_front 기준 그룹 매핑
    print("\n[3/3] 타임스탬프 기반 그룹 매핑 중...")
    groups     = []
    skip_count = 0

    for fidx, ts_ref in enumerate(cam_ts['cam_front']):
        stem_ref = f"cam_front_{fidx:05d}"

        if stem_ref not in existing:
            skip_count += 1
            continue

        if not os.path.isfile(os.path.join(lbl_dir, f"{stem_ref}.txt")):
            skip_count += 1
            continue

        group = {
            "ts":         ts_ref,
            "cams":       {"cam_front": stem_ref},
            "label_stem": stem_ref,
        }

        for cam_name, ts_list in cam_ts.items():
            if cam_name == 'cam_front' or not ts_list:
                continue
            best_idx = min(range(len(ts_list)),
                           key=lambda i: abs(ts_list[i] - ts_ref))
            if abs(ts_list[best_idx] - ts_ref) > SYNC_THRESHOLD:
                continue
            stem_cam = f"{cam_name}_{best_idx:05d}"
            if stem_cam in existing:
                group["cams"][cam_name] = stem_cam

        groups.append(group)

    print(f"   생성된 그룹 수: {len(groups):,}")
    print(f"   스킵된 프레임 : {skip_count:,}")

    cam_coverage = {cam: 0 for cam in cam_ts.keys()}
    for g in groups:
        for cam in g["cams"]:
            cam_coverage[cam] += 1
    print("\n   카메라별 커버리지:")
    for cam, cnt in cam_coverage.items():
        pct = cnt / len(groups) * 100 if groups else 0
        print(f"   {cam:20s}: {cnt:,} / {len(groups):,} ({pct:.1f}%)")

    out_path = os.path.join(dataset_dir, 'frame_groups.json')
    with open(out_path, 'w') as f:
        json.dump(groups, f, indent=2)

    print(f"\n✅ 완료! → {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bag', help='.bag 파일 경로')
    parser.add_argument('--dataset_dir', '-d', default='./dataset')
    args = parser.parse_args()

    if not os.path.isfile(args.bag):
        print(f"[ERROR] bag 파일 없음: {args.bag}")
        sys.exit(1)

    build_groups(args.bag, args.dataset_dir)