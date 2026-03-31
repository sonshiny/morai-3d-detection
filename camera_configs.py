#!/usr/bin/env python3
"""
camera_configs.py
=================
/tf 토픽에서 추출한 정확한 카메라 외부파라미터.
모든 파일이 이 파일 하나에서 import해서 사용한다.

ROS tf 쿼터니언 변환 원칙:
  - /tf의 쿼터니언은 cam→body 방향 (child=cam, parent=base_link)
  - R_body_to_cam = R_cam_to_body.T
  - E (4x4 extrinsic) = [R_body_to_cam | -R_body_to_cam @ t_cam_in_body]

카메라 좌표계 (train.py / inference.py 공통):
  X = 전방(depth), Y = 좌, Z = 상
  → u = fx * (-Y/X) + cx
  → v = fx * (-Z/X) + cy
"""

import numpy as np
from scipy.spatial.transform import Rotation

# ===========================================================
# /tf 원본 데이터 (절대 수정 금지 — 시뮬레이터 ground truth)
# 2026-03-31 최신 tf 에코 기준으로 업데이트
# ===========================================================
_TF_DATA = {
    'cam_front': {
        'translation': [1.8799999952316284, 0.0, 1.350000023841858],
        'quaternion':  [0.0, 0.13052621483802795, 0.0, 0.9914448857307434],
    },
    'cam_back': {
        'translation': [-0.3499999940395355, 0.0, 1.350000023841858],
        'quaternion':  [-0.13052621483802795, -5.705481420648084e-09,
                        -0.9914448857307434,  -4.333743319762107e-08],
    },
    'cam_front_left': {
        'translation': [1.399999976158142, 0.8500000238418579, 1.350000023841858],
        'quaternion':  [-0.020141197368502617, -0.038690872490406036,
                        -0.46130919456481934,  -0.8861665725708008],
    },
    'cam_front_right': {
        'translation': [1.399999976158142, -0.8500000238418579, 1.350000023841858],
        'quaternion':  [-0.020141197368502617,  0.038690872490406036,
                        -0.46130919456481934,   0.8861665725708008],
    },
    'cam_back_left': {
        'translation': [0.5699999928474426, 0.8500000238418579, 1.350000023841858],
        'quaternion':  [-0.038690872490406036, -0.020141197368502617,
                        -0.8861665725708008,   -0.46130919456481934],
    },
    'cam_back_right': {
        'translation': [0.5699999928474426, -0.8500000238418579, 1.350000023841858],
        'quaternion':  [-0.038690872490406036,  0.02014119364321232,
                        -0.8861665725708008,    0.46130913496017456],
    },
}

# ===========================================================
# 카메라 내부파라미터 (FOV 90도, 640x480 고정)
# ===========================================================
_CAM_W   = 640
_CAM_H   = 480
_FOV_DEG = 90.0

def _compute_intrinsic(w=_CAM_W, h=_CAM_H, fov_h_deg=_FOV_DEG):
    fov_rad = np.radians(fov_h_deg)
    fx = (w / 2.0) / np.tan(fov_rad / 2.0)
    return np.array([[fx,  0, w / 2.0],
                     [ 0, fx, h / 2.0],
                     [ 0,  0,       1]], dtype=np.float32)

# ===========================================================
# 외부파라미터 계산
# /tf 쿼터니언: cam→body 방향
# R_cam_to_body = Rotation.from_quat([x,y,z,w]).as_matrix()
# R_body_to_cam = R_cam_to_body.T
# E[:3,:3] = R_body_to_cam
# E[:3, 3] = -R_body_to_cam @ t   (t = 카메라 위치 in body frame)
# ===========================================================
def _compute_extrinsic(translation, quaternion):
    x, y, z, w = quaternion
    R_cam_to_body = Rotation.from_quat([x, y, z, w]).as_matrix()
    R_body_to_cam = R_cam_to_body.T

    t = np.array(translation, dtype=np.float32)
    E = np.eye(4, dtype=np.float32)
    E[:3, :3] = R_body_to_cam
    E[:3,  3] = -R_body_to_cam @ t
    return E

# ===========================================================
# 공개 API
# ===========================================================
CAM_ORDER = [
    'cam_front', 'cam_front_left', 'cam_front_right',
    'cam_back',  'cam_back_left',  'cam_back_right',
]

INTRINSICS = {cam: _compute_intrinsic() for cam in CAM_ORDER}

EXTRINSICS = {
    cam: _compute_extrinsic(
        _TF_DATA[cam]['translation'],
        _TF_DATA[cam]['quaternion']
    )
    for cam in CAM_ORDER
}

CAM_POSITIONS = {
    cam: np.array(_TF_DATA[cam]['translation'], dtype=np.float32)
    for cam in CAM_ORDER
}

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