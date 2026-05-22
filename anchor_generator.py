import torch
import numpy as np
import os
import math

NUM_ANCHORS = 900
ANCHOR_XY_FILE = 'anchor_kmeans_xy.npy'
ANCHOR_FULL_FILE = 'anchor_kmeans_full.npy'


def generate_anchors():
    """
    3D 위치 앵커: [900, 3]
    K-means 클러스터링으로 GT가 자주 등장하는 위치에 앵커 배치
    """
    kmeans_path = os.path.join(os.path.dirname(__file__), ANCHOR_XY_FILE)

    if os.path.isfile(kmeans_path):
        centers_xy = np.load(kmeans_path)
        if centers_xy.shape != (NUM_ANCHORS, 2):
            print(f"[anchor_generator] ⚠️  K-means shape 이상: {centers_xy.shape}, 균일 그리드로 fallback")
        else:
            z = np.zeros((NUM_ANCHORS, 1), dtype=np.float32)
            anchors_np = np.hstack([centers_xy.astype(np.float32), z])
            return torch.from_numpy(anchors_np)

    # K-means 파일 없으면 3전방 카메라 ROI에 맞춘 grid fallback
    print("[anchor_generator] K-means 파일 없음 → 전방 ROI 균일 그리드 사용")
    x = torch.linspace(0, 60, 30)
    y = torch.linspace(-30, 30, 30)
    z = torch.tensor([0.0])
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    return torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)


def generate_anchors_full():
    """
    SparseDrive 11차원 앵커: [900, 11]
    {x, y, z, ln_w, ln_l, ln_h, sin_yaw, cos_yaw, vx, vy, vz}

    ⚠️ GT(morai_dataset / morai_3d_live.py)와 동일한 순서: ln_w → ln_l → ln_h

    Default 값:
      - 차량 크기: w=1.8m, l=4.5m, h=1.6m (sedan 평균)
      - yaw: -π/2 (GT의 동일 방향 NPC 오프셋과 일치)
      - velocity: 0
    """
    full_path = os.path.join(os.path.dirname(__file__), ANCHOR_FULL_FILE)
    if os.path.isfile(full_path):
        anchors_full = np.load(full_path)
        if anchors_full.shape == (NUM_ANCHORS, 11):
            return torch.from_numpy(anchors_full.astype(np.float32))
        print(f"[anchor_generator] ⚠️  full K-means shape 이상: {anchors_full.shape}, XY anchor로 fallback")

    anchors_3d = generate_anchors()
    N = anchors_3d.shape[0]

    # 현실적인 차량 크기 default (log-space)
    ln_w = math.log(1.8)   # 폭   1.8m
    ln_l = math.log(4.5)   # 길이 4.5m
    ln_h = math.log(1.6)   # 높이 1.6m

    # 동일 방향 NPC와 일치 (GT의 -π/2 오프셋)
    sin_yaw = -1.0
    cos_yaw = 0.0

    defaults = torch.tensor(
        [ln_w, ln_l, ln_h, sin_yaw, cos_yaw, 0.0, 0.0, 0.0]
    ).unsqueeze(0).expand(N, -1)
    return torch.cat([anchors_3d, defaults], dim=-1)


if __name__ == "__main__":
    a3  = generate_anchors()
    a11 = generate_anchors_full()
    print(f"앵커 수: {a3.shape[0]}")
    print(f"x 범위: {a3[:,0].min():.1f} ~ {a3[:,0].max():.1f}")
    print(f"y 범위: {a3[:,1].min():.1f} ~ {a3[:,1].max():.1f}")
    print(f"\n11차원 default:")
    print(f"  w  : {torch.exp(a11[0, 3]).item():.2f}m")
    print(f"  l  : {torch.exp(a11[0, 4]).item():.2f}m")
    print(f"  h  : {torch.exp(a11[0, 5]).item():.2f}m")
    sin_y = a11[0, 6].item()
    cos_y = a11[0, 7].item()
    yaw_rad = math.atan2(sin_y, cos_y)
    print(f"  yaw: {yaw_rad:.3f} rad ({math.degrees(yaw_rad):.1f}°)")
