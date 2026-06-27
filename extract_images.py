#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_images.py
여러 MORAI bag 파일에서 6개 카메라 이미지를 추출해 dataset/images 에 저장한다.

파일명 규칙:
  {cam_name}__{bag_key}__{frame_idx:05d}.jpg

예:
  cam_front__scenario1__00000.jpg
  cam_back_right__scenario7.bag_2__00363.jpg

사용법:
  python3 extract_images.py
  python3 extract_images.py scenario1.bag scenario3.bag
"""

import os
import sys
import json
import cv2
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

DATASET_DIR = '/data/dataset'
IMG_DIR = os.path.join(DATASET_DIR, 'images')
MANIFEST_PATH = os.path.join(DATASET_DIR, 'image_manifest.json')


def bag_to_key(bag_path: str) -> str:
    base = os.path.basename(bag_path)
    return base[:-4] if base.endswith('.bag') else base


def decode_compressed_image(msg):
    """
    sensor_msgs/CompressedImage → BGR np.ndarray
    cv_bridge를 쓰지 않고 OpenCV로 직접 디코딩
    """
    np_arr = np.frombuffer(msg.data, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('cv2.imdecode 실패')
    return img


def decode_raw_image(msg):
    """
    sensor_msgs/Image → np.ndarray
    MORAI에서 흔한 encoding 위주로 처리
    """
    h = msg.height
    w = msg.width
    enc = str(msg.encoding).lower()

    data = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ('bgr8',):
        expected = h * w * 3
        if data.size != expected:
            raise ValueError(f'bgr8 크기 불일치: got={data.size}, expected={expected}')
        return data.reshape(h, w, 3)

    elif enc in ('rgb8',):
        expected = h * w * 3
        if data.size != expected:
            raise ValueError(f'rgb8 크기 불일치: got={data.size}, expected={expected}')
        img = data.reshape(h, w, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    elif enc in ('bgra8',):
        expected = h * w * 4
        if data.size != expected:
            raise ValueError(f'bgra8 크기 불일치: got={data.size}, expected={expected}')
        img = data.reshape(h, w, 4)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    elif enc in ('rgba8',):
        expected = h * w * 4
        if data.size != expected:
            raise ValueError(f'rgba8 크기 불일치: got={data.size}, expected={expected}')
        img = data.reshape(h, w, 4)
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    elif enc in ('mono8', '8uc1'):
        expected = h * w
        if data.size != expected:
            raise ValueError(f'mono8 크기 불일치: got={data.size}, expected={expected}')
        img = data.reshape(h, w)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    else:
        raise ValueError(f'지원하지 않는 raw encoding: {msg.encoding}')


def decode_ros_image(msg):
    # sensor_msgs/Image
    if hasattr(msg, 'encoding') and hasattr(msg, 'height') and hasattr(msg, 'width'):
        return decode_raw_image(msg)

    # sensor_msgs/CompressedImage
    if hasattr(msg, 'format') and hasattr(msg, 'data'):
        return decode_compressed_image(msg)

    raise TypeError(f'지원하지 않는 메시지 타입: {type(msg)}')


def extract_one_bag(bag_path: str):
    bag_key = bag_to_key(bag_path)
    counters = {cam: 0 for cam in CAM_TOPICS.values()}
    errors = 0

    print('=' * 72)
    print(f'[추출 시작] {bag_path}  → bag_key={bag_key}')
    print('=' * 72)

    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=list(CAM_TOPICS.keys())):
            cam_name = CAM_TOPICS[topic]
            frame_idx = counters[cam_name]
            stem = f'{cam_name}__{bag_key}__{frame_idx:05d}'
            out_path = os.path.join(IMG_DIR, f'{stem}.jpg')

            try:
                img = decode_ros_image(msg)
                ok = cv2.imwrite(out_path, img)
                if not ok:
                    raise RuntimeError(f'cv2.imwrite 실패: {out_path}')
                counters[cam_name] += 1
            except Exception as e:
                errors += 1
                print(f'[ERROR] {bag_path} | {topic} | frame={frame_idx} | {e}')

    total = sum(counters.values())
    print(f'[완료] {bag_path}')
    for cam_name in CAM_TOPICS.values():
        print(f'  - {cam_name:16s}: {counters[cam_name]:6d}')
    print(f'  - total           : {total:6d}')
    print(f'  - errors          : {errors:6d}')
    print()

    return {
        'bag_file': os.path.basename(bag_path),
        'bag_key': bag_key,
        'counts': counters,
        'total': total,
        'errors': errors,
    }


def main():
    bag_paths = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_BAGS

    os.makedirs(IMG_DIR, exist_ok=True)

    for bag_path in bag_paths:
        if not os.path.isfile(bag_path):
            print(f'[ERROR] bag 파일 없음: {bag_path}')
            sys.exit(1)

    manifest = {
        'images_dir': os.path.abspath(IMG_DIR),
        'bags': [],
    }

    grand_total = 0
    grand_errors = 0
    grand_counts = {cam: 0 for cam in CAM_TOPICS.values()}

    for bag_path in bag_paths:
        info = extract_one_bag(bag_path)
        manifest['bags'].append(info)
        grand_total += info['total']
        grand_errors += info['errors']
        for cam, cnt in info['counts'].items():
            grand_counts[cam] += cnt

    manifest['grand_counts'] = grand_counts
    manifest['grand_total'] = grand_total
    manifest['grand_errors'] = grand_errors

    with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print('=' * 72)
    print('전체 완료!')
    print(f'[이미지 폴더] {os.path.abspath(IMG_DIR)}')
    print(f'[매니페스트] {os.path.abspath(MANIFEST_PATH)}')
    for cam_name in CAM_TOPICS.values():
        print(f'  - {cam_name:16s}: {grand_counts[cam_name]:6d}')
    print(f'  - total           : {grand_total:6d}')
    print(f'  - errors          : {grand_errors:6d}')
    print('=' * 72)


if __name__ == '__main__':
    main()
