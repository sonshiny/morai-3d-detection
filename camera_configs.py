#!/usr/bin/env python3
"""
camera_configs.py
=================
MORAI 센서 명세 기반 3카메라 외부/내부파라미터.
모든 파일이 이 파일 하나에서 import해서 사용한다.

카메라 좌표계 (train.py / inference.py 공통):
  X = 전방(depth), Y = 좌, Z = 상
  → u = fx * (-Y/X) + cx
  → v = fx * (-Z/X) + cy

외부파라미터 회전 규칙:
  - yaw  : body Z축 기준 수평 회전 (양수 = 좌향)
  - pitch: 카메라 로컬 Y축 기준 상하 회전 (양수 = 하향)
  - R_cam_to_body = Rz(yaw) @ Ry(pitch)  [intrinsic ZY]
  - E = [R_body_to_cam | -R_body_to_cam @ t]
"""

import numpy as np
from scipy.spatial.transform import Rotation

# ===========================================================
# 카메라 명세 (MORAI 센서 설정 기준, 10Hz 공통)
# ===========================================================
_CAM_DATA = {
    'cam_front': {
        'translation': [1.92, 0.0, 1.21],
        'pitch_deg':   3.0,
        'yaw_deg':     0.0,
    },
    'cam_front_right': {
        'translation': [1.92, -0.56, 1.21],
        'pitch_deg':   3.0,
        'yaw_deg':    -45.0,
    },
    'cam_front_left': {
        'translation': [1.92, 0.56, 1.21],
        'pitch_deg':   3.0,
        'yaw_deg':     45.0,
    },
}

# ===========================================================
# 카메라 내부파라미터 (1600x900, FOV 70도)
# ===========================================================
_CAM_W   = 1600
_CAM_H   = 900
_FOV_DEG = 70.0

def _compute_intrinsic(w=_CAM_W, h=_CAM_H, fov_h_deg=_FOV_DEG):
    fov_rad = np.radians(fov_h_deg)
    fx = (w / 2.0) / np.tan(fov_rad / 2.0)
    return np.array([[fx,  0, w / 2.0],
                     [ 0, fx, h / 2.0],
                     [ 0,  0,       1]], dtype=np.float32)

# ===========================================================
# 외부파라미터 계산
# ===========================================================
def _compute_extrinsic(translation, pitch_deg, yaw_deg):
    R_cam_to_body = Rotation.from_euler(
        'ZY', [yaw_deg, pitch_deg], degrees=True
    ).as_matrix()
    R_body_to_cam = R_cam_to_body.T
    t = np.array(translation, dtype=np.float32)
    E = np.eye(4, dtype=np.float32)
    E[:3, :3] = R_body_to_cam
    E[:3,  3] = -R_body_to_cam @ t
    return E

# ===========================================================
# 공개 API
# ===========================================================
CAM_ORDER = ['cam_front', 'cam_front_left', 'cam_front_right']

INTRINSICS = {cam: _compute_intrinsic() for cam in CAM_ORDER}

EXTRINSICS = {
    cam: _compute_extrinsic(
        _CAM_DATA[cam]['translation'],
        _CAM_DATA[cam]['pitch_deg'],
        _CAM_DATA[cam]['yaw_deg'],
    )
    for cam in CAM_ORDER
}

CAM_POSITIONS = {
    cam: np.array(_CAM_DATA[cam]['translation'], dtype=np.float32)
    for cam in CAM_ORDER
}

# ===========================================================
# 라이다 외부파라미터 (카메라와 독립, CAM_ORDER에는 넣지 않는다)
# MORAI /lidar3D 포인트가 REP-103(X전방, Y좌, Z상)으로 퍼블리시된다는
# 가정. cam_front에 스캔을 투영해 실증 검증 필요.
# ===========================================================
LIDAR_TRANSLATION = [1.92, 0.0, 1.35]

def _compute_lidar_to_body(translation):
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = np.array(translation, dtype=np.float32)
    return T

LIDAR_TO_BODY = _compute_lidar_to_body(LIDAR_TRANSLATION)  # lidar-frame point -> body-frame (rot=0, 평행이동만)

# ===========================================================
# 검증 (직접 실행 시)
# ===========================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  camera_configs.py 검증")
    print("=" * 60)

    for cam in CAM_ORDER:
        K = INTRINSICS[cam]
        E = EXTRINSICS[cam]
        t = CAM_POSITIONS[cam]
        print(f"\n[{cam}]")
        print(f"  위치 (body): x={t[0]:.3f}, y={t[1]:.3f}, z={t[2]:.3f}")
        print(f"  fx={K[0,0]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
        print(f"  R_body_to_cam:\n{E[:3,:3]}")

    print("\n" + "=" * 60)
    print("  투영 테스트: 전방 20m 지점 (x=20, y=0, z=0)")
    print("=" * 60)
    p_body = np.array([20.0, 0.0, 0.0, 1.0], dtype=np.float32)
    for cam in CAM_ORDER:
        E = EXTRINSICS[cam]
        K = INTRINSICS[cam]
        p_cam = E @ p_body
        depth = p_cam[0]
        if depth > 0.1:
            u = K[0, 0] * (-p_cam[1]) / depth + K[0, 2]
            v = K[0, 0] * (-p_cam[2]) / depth + K[1, 2]
            in_frame = (0 <= u <= _CAM_W) and (0 <= v <= _CAM_H)
            print(f"  {cam:20s}: depth={depth:.1f}m  u={u:.0f}  v={v:.0f}  "
                  f"{'✅ 프레임 안' if in_frame else '❌ 프레임 밖'}")
        else:
            print(f"  {cam:20s}: depth={depth:.1f}m  (카메라 뒤쪽)")
