#!/usr/bin/env python3
"""
=============================================================
  MORAI Simulation → YOLO Label Generator
  bag 파일에서 이미지 + NPC 3D 박스를 읽어 YOLO 라벨 생성
=============================================================
사용법:
  python morai_label_generator.py /path/to/your.bag
  python morai_label_generator.py /path/to/your.bag --output ./dataset --visualize

출력 구조:
  dataset/
  ├── images/   ← 원본 이미지 (.jpg)
  ├── labels/   ← YOLO 라벨 (.txt)
  └── visualize/ ← 박스 시각화 이미지 (--visualize 옵션 시)
=============================================================
"""

import os
import sys
import argparse
import numpy as np
import cv2
import rosbag
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation

# ===========================================================
# ① 카메라 설정 (제공된 값 그대로)
# ===========================================================
CAMERA_CONFIGS = {
    '/morai/cam_back': {
        'offset_xyz': [-0.35,  0.00, 1.23],   # 차량 중심 기준 (m)
        'rpy_deg':    [  0.0,   5.0, 180.0],  # roll, pitch, yaw (도)
        'width': 640, 'height': 480, 'fov_h_deg': 90,
        'short_name': 'cam_back',
    },
    '/morai/cam_back_left': {
        'offset_xyz': [ 0.57,  0.85, 1.30],
        'rpy_deg':    [  0.0,   5.0, 125.0],
        'width': 640, 'height': 480, 'fov_h_deg': 90,
        'short_name': 'cam_back_left',
    },
    '/morai/cam_back_right': {
        'offset_xyz': [ 0.57, -0.85, 1.30],
        'rpy_deg':    [  0.0,   5.0, 235.0],
        'width': 640, 'height': 480, 'fov_h_deg': 90,
        'short_name': 'cam_back_right',
    },
    '/morai/cam_front': {
        'offset_xyz': [ 1.88,  0.00, 1.35],
        'rpy_deg':    [  0.0,  15.0,   0.0],
        'width': 640, 'height': 480, 'fov_h_deg': 90,
        'short_name': 'cam_front',
    },
    '/morai/cam_front_left': {
        'offset_xyz': [ 1.40,  0.85, 1.35],
        'rpy_deg':    [  0.0,   5.0,  55.0],
        'width': 640, 'height': 480, 'fov_h_deg': 90,
        'short_name': 'cam_front_left',
    },
    '/morai/cam_front_right': {
        'offset_xyz': [ 1.40, -0.85, 1.35],
        'rpy_deg':    [  0.0,   5.0, 305.0],
        'width': 640, 'height': 480, 'fov_h_deg': 90,
        'short_name': 'cam_front_right',
    },
}

# 클래스 매핑 (MORAI NPC type → YOLO class_id)
NPC_TYPE_MAP = {
    1: 0,   # 승용차 → class 0
    2: 1,   # 트럭   → class 1
    3: 2,   # 버스   → class 2
}
CLASS_NAMES = {0: 'car', 1: 'truck', 2: 'bus'}

# 최대 타임스탬프 허용 오차 (초) — 이보다 멀면 동기화 실패로 처리
MAX_SYNC_GAP_SEC = 0.1


# ===========================================================
# ② 카메라 내부 파라미터 계산 (Intrinsic)
# ===========================================================
def compute_intrinsic(width, height, fov_h_deg):
    """
    핀홀 카메라 모델 기반 내부 파라미터 계산
      fx = (W/2) / tan(fov/2)
    """
    fov_rad = np.radians(fov_h_deg)
    fx = (width  / 2.0) / np.tan(fov_rad / 2.0)
    fy = fx                 # 정방형 픽셀 가정
    cx = width  / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


# ===========================================================
# ③ 카메라 외부 파라미터 사전 계산 (Extrinsic)
# ===========================================================
def preprocess_cameras(configs):
    """
    모든 카메라의 intrinsic + extrinsic 회전 행렬을 미리 계산.

    좌표계 관례:
      - 차량 바디 프레임 : X=전방, Y=좌측, Z=상방 (RHR)
      - 카메라 바디 프레임 : 차량 바디와 동일 관례, rpy_deg로 회전
      - 이미지 프레임    : U=오른쪽, V=아래 (OpenCV)

    R_cam_in_body = Rz(yaw) * Ry(pitch) * Rx(roll)
      → 차량 바디 → 카메라 바디 변환:  p_cam = R_cam_in_body^T @ (p_body - offset)
    """
    processed = {}
    for topic, cfg in configs.items():
        roll, pitch, yaw = cfg['rpy_deg']
        w, h, fov = cfg['width'], cfg['height'], cfg['fov_h_deg']

        fx, fy, cx, cy = compute_intrinsic(w, h, fov)

        # scipy ZYX 오일러: Rz(yaw) * Ry(pitch) * Rx(roll)
        R_cam_in_body = Rotation.from_euler(
            'ZYX', [yaw, pitch, roll], degrees=True
        ).as_matrix()

        # 차량 바디 → 카메라 바디 (역방향)
        R_body_to_cam = R_cam_in_body.T

        processed[topic] = {
            'offset':        np.array(cfg['offset_xyz'], dtype=float),
            'R_body_to_cam': R_body_to_cam,
            'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
            'width': w, 'height': h,
            'short_name': cfg['short_name'],
        }
    return processed


# ===========================================================
# ④ 좌표 변환 유틸
# ===========================================================
def world_to_body(p_world, ego_pos, ego_heading_deg):
    """
    월드 좌표 → Ego 차량 바디 좌표
      1) 평행 이동 (ego 기준으로 이동)
      2) Ego heading(yaw)만큼 역회전
         (MORAI heading: 반시계 방향 양수, 북쪽=0 or x축=0 여부는
          시각화로 확인 권장)
    """
    dp = p_world - ego_pos
    yaw = np.radians(ego_heading_deg)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    x_b =  cos_y * dp[0] + sin_y * dp[1]
    y_b = -sin_y * dp[0] + cos_y * dp[1]
    z_b = dp[2]
    return np.array([x_b, y_b, z_b])


def body_to_cam(p_body, cam_cfg):
    """
    Ego 차량 바디 좌표 → 카메라 좌표 (MORAI 카메라 바디 프레임)
    """
    p_rel = p_body - cam_cfg['offset']
    return cam_cfg['R_body_to_cam'] @ p_rel


def cam_to_pixel(p_cam, cam_cfg):
    """
    카메라 좌표 (X=전방, Y=좌, Z=상) → 이미지 픽셀 (u, v)

      depth    = p_cam[0]   (전방, 양수여야 카메라 앞쪽)
      right    = -p_cam[1]  (Y=좌 → 부호 반전 → 우측)
      down     = -p_cam[2]  (Z=상 → 부호 반전 → 하방)

      u = fx * right / depth + cx
      v = fy * down  / depth + cy
    """
    depth = p_cam[0]
    if depth <= 0.3:          # 카메라 뒤쪽 또는 너무 가까움
        return None
    u = cam_cfg['fx'] * (-p_cam[1]) / depth + cam_cfg['cx']
    v = cam_cfg['fy'] * (-p_cam[2]) / depth + cam_cfg['cy']
    return u, v


# ===========================================================
# ⑤ NPC 3D 바운딩 박스 → 8개 코너 (월드 좌표)
# ===========================================================
def get_3d_corners_world(npc_pos, npc_size, npc_heading_deg):
    """
    NPC 중심(position)과 크기(size), 방향(heading)을 받아
    8개 모서리를 월드 좌표로 반환.

    NPC 로컬 좌표:
      X=앞뒤(길이), Y=좌우(너비), Z=상하(높이)
    """
    L = npc_size[0]   # 길이 (앞뒤)
    W = npc_size[1]   # 너비 (좌우)
    H = npc_size[2]   # 높이

    # 로컬 8개 코너
    corners_local = np.array([
        [ L/2,  W/2,  H/2],
        [ L/2,  W/2, -H/2],
        [ L/2, -W/2,  H/2],
        [ L/2, -W/2, -H/2],
        [-L/2,  W/2,  H/2],
        [-L/2,  W/2, -H/2],
        [-L/2, -W/2,  H/2],
        [-L/2, -W/2, -H/2],
    ], dtype=float)

    # NPC heading으로 회전 (Z축 yaw)
    yaw = np.radians(npc_heading_deg)
    Rz = np.array([
        [ np.cos(yaw), -np.sin(yaw), 0],
        [ np.sin(yaw),  np.cos(yaw), 0],
        [           0,            0, 1],
    ])

    corners_world = (Rz @ corners_local.T).T + np.array(npc_pos)
    return corners_world   # shape: (8, 3)


# ===========================================================
# ⑥ NPC → 2D 바운딩 박스 (픽셀)
# ===========================================================
def npc_to_bbox2d(npc, ego_pos, ego_heading_deg, cam_cfg):
    """
    NPC 1개를 2D 바운딩 박스로 투영.
    반환: (x_min, y_min, x_max, y_max) 픽셀 좌표  또는  None
    """
    npc_pos  = np.array([npc.position.x, npc.position.y, npc.position.z])
    npc_size = [npc.size.x, npc.size.y, npc.size.z]

    corners_world = get_3d_corners_world(npc_pos, npc_size, npc.heading)

    us, vs = [], []
    for corner in corners_world:
        p_body = world_to_body(corner, ego_pos, ego_heading_deg)
        p_cam  = body_to_cam(p_body, cam_cfg)
        result = cam_to_pixel(p_cam, cam_cfg)
        if result is not None:
            us.append(result[0])
            vs.append(result[1])

    if len(us) < 1:
        return None   # 모든 코너가 카메라 뒤쪽

    W, H = cam_cfg['width'], cam_cfg['height']
    x_min = max(0,  min(us))
    x_max = min(W,  max(us))
    y_min = max(0,  min(vs))
    y_max = min(H,  max(vs))

    # 유효한 박스 크기 체크
    if x_max - x_min < 2 or y_max - y_min < 2:
        return None
    # 이미지 영역 완전 밖
    if x_max <= 0 or x_min >= W or y_max <= 0 or y_min >= H:
        return None

    return x_min, y_min, x_max, y_max


# ===========================================================
# ⑦ YOLO 형식으로 변환
# ===========================================================
def to_yolo_fmt(x_min, y_min, x_max, y_max, img_w, img_h, class_id=0):
    """픽셀 좌표 → YOLO 정규화 포맷 (0~1)"""
    cx = ((x_min + x_max) / 2) / img_w
    cy = ((y_min + y_max) / 2) / img_h
    bw = (x_max - x_min)       / img_w
    bh = (y_max - y_min)       / img_h
    return class_id, cx, cy, bw, bh


# ===========================================================
# ⑧ 타임스탬프 동기화 (가장 가까운 메시지 찾기)
# ===========================================================
def find_closest(msg_list, target_sec):
    """(timestamp, msg) 리스트에서 target_sec에 가장 가까운 것 반환"""
    if not msg_list:
        return None, None
    idx = min(range(len(msg_list)), key=lambda i: abs(msg_list[i][0] - target_sec))
    gap = abs(msg_list[idx][0] - target_sec)
    if gap > MAX_SYNC_GAP_SEC:
        return None, None
    return msg_list[idx][0], msg_list[idx][1]


# ===========================================================
# ⑨ 메인 처리 함수
# ===========================================================
def process_bag(bag_path, output_dir, visualize=False):
    print("=" * 60)
    print("  MORAI bag → YOLO 라벨 생성기")
    print("=" * 60)
    print(f"[입력] {bag_path}")
    print(f"[출력] {output_dir}")
    print(f"[시각화] {'ON' if visualize else 'OFF'}")
    print()

    bridge      = CvBridge()
    cam_configs = preprocess_cameras(CAMERA_CONFIGS)
    cam_topics  = list(CAMERA_CONFIGS.keys())

    # 출력 폴더 생성
    img_dir = os.path.join(output_dir, 'images')
    lbl_dir = os.path.join(output_dir, 'labels')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    if visualize:
        viz_dir = os.path.join(output_dir, 'visualize')
        os.makedirs(viz_dir, exist_ok=True)

    # ── Ego / Object 메시지 전체 사전 로드 ──────────────────
    print("[1/3] Ego / Object 토픽 사전 로드 중...")
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
    if not ego_msgs or not obj_msgs:
        print("[ERROR] Ego 또는 Object 토픽이 bag에 없습니다!")
        sys.exit(1)

    # ── 카메라 이미지 처리 ──────────────────────────────────
    print("\n[2/3] 이미지 투영 및 라벨 생성 중...")

    frame_cnt   = {t: 0 for t in cam_topics}
    total_boxes = 0
    total_imgs  = 0
    sync_fail   = 0

    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=cam_topics):
            if topic not in cam_configs:
                continue

            cam_cfg  = cam_configs[topic]
            cam_name = cam_cfg['short_name']
            fidx     = frame_cnt[topic]
            frame_cnt[topic] += 1
            ts = t.to_sec()
            # 이미지 디코딩 (CompressedImage 대응)
            try:
                import numpy as np
                buf = np.frombuffer(msg.data, dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("imdecode 결과 None")
            except Exception as e:
                print(f"[WARN] 이미지 디코딩 실패 ({cam_name} #{fidx}): {e}")
                continue

            # 동기화
            _, ego_msg = find_closest(ego_msgs, ts)
            _, obj_msg = find_closest(obj_msgs, ts)
            if ego_msg is None or obj_msg is None:
                sync_fail += 1
                continue

            ego_pos     = np.array([ego_msg.position.x,
                                    ego_msg.position.y,
                                    ego_msg.position.z])
            ego_heading = ego_msg.heading

            # NPC 투영
            yolo_lines = []
            viz_img    = img.copy() if visualize else None

            for npc in obj_msg.npc_list:
                bbox = npc_to_bbox2d(npc, ego_pos, ego_heading, cam_cfg)
                if bbox is None:
                    continue

                x_min, y_min, x_max, y_max = bbox
                class_id = NPC_TYPE_MAP.get(npc.type, 0)
                _, cx, cy, bw, bh = to_yolo_fmt(
                    x_min, y_min, x_max, y_max,
                    cam_cfg['width'], cam_cfg['height'], class_id
                )
                yolo_lines.append(
                    f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                )

                if visualize:
                    color = [(0,255,0),(0,165,255),(0,0,255)][min(class_id,2)]
                    cv2.rectangle(viz_img,
                                  (int(x_min), int(y_min)),
                                  (int(x_max), int(y_max)),
                                  color, 2)
                    label_txt = f"{CLASS_NAMES.get(class_id,'?')} {npc.unique_id}"
                    cv2.putText(viz_img, label_txt,
                                (int(x_min)+2, int(y_min)-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                                cv2.LINE_AA)

            # 파일 저장
            stem     = f"{cam_name}_{fidx:05d}"
            img_path = os.path.join(img_dir, f"{stem}.jpg")
            lbl_path = os.path.join(lbl_dir, f"{stem}.txt")

            cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            with open(lbl_path, 'w') as f:
                f.write('\n'.join(yolo_lines))

            if visualize:
                cv2.imwrite(os.path.join(viz_dir, f"{stem}_viz.jpg"), viz_img)

            total_boxes += len(yolo_lines)
            total_imgs  += 1

            if fidx % 100 == 0:
                print(f"   [{cam_name}] frame {fidx:05d} | "
                      f"NPC {len(yolo_lines)}개 박스 | "
                      f"ego({ego_pos[0]:.1f},{ego_pos[1]:.1f}) "
                      f"hdg={ego_heading:.1f}°")

    # ── 결과 요약 ────────────────────────────────────────────
    print("\n[3/3] 완료!")
    print(f"   처리 이미지  : {total_imgs:,} 장")
    print(f"   총 라벨 박스 : {total_boxes:,} 개")
    print(f"   동기화 실패  : {sync_fail:,} 프레임")
    print(f"\n   📁 images/   → {img_dir}")
    print(f"   📁 labels/   → {lbl_dir}")
    if visualize:
        print(f"   📁 visualize/ → {viz_dir}")

    # YOLO dataset.yaml 자동 생성
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write(f"train: images\n")
        f.write(f"val:   images\n\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {list(CLASS_NAMES.values())}\n")
    print(f"\n   📄 dataset.yaml → {yaml_path}")
    print("\n✅ 이제 YOLO 학습 바로 시작 가능!")


# ===========================================================
# ⑩ 실행 진입점
# ===========================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MORAI bag → YOLO 라벨 생성기',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python morai_label_generator.py data.bag
  python morai_label_generator.py data.bag -o ./dataset -v
        """
    )
    parser.add_argument('bag',
                        help='.bag 파일 경로')
    parser.add_argument('--output', '-o',
                        default='./dataset',
                        help='출력 폴더 (기본: ./dataset)')
    parser.add_argument('--visualize', '-v',
                        action='store_true',
                        help='박스 시각화 이미지 저장')
    args = parser.parse_args()

    if not os.path.isfile(args.bag):
        print(f"[ERROR] bag 파일을 찾을 수 없습니다: {args.bag}")
        sys.exit(1)

    process_bag(args.bag, args.output, visualize=args.visualize)
