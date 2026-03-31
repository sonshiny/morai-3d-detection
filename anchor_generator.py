import torch
import numpy as np
import os


def generate_anchors():
    """
    3D 위치 앵커: [900, 3] (이미지 투영용)

    SparseDrive 논문 방식: K-means 클러스터링으로 초기화
    → GT가 자주 등장하는 위치에 앵커 배치
    → Hungarian Matching 안정성 대폭 향상

    균일 그리드 대비 K-means의 장점:
    - 차량이 자주 있는 위치(도로 위)에 앵커 집중
    - 차량이 없는 위치(건물, 하늘)에 앵커 낭비 없음
    - Hungarian Matching이 매 step마다 일관된 앵커에 매칭
    """
    kmeans_path = os.path.join(os.path.dirname(__file__), 'anchor_kmeans_xy.npy')

    if os.path.isfile(kmeans_path):
        # K-means 앵커 로드 (논문 방식)
        centers_xy = np.load(kmeans_path)   # [900, 2]
        z = np.zeros((900, 1), dtype=np.float32)  # z=0 고정 (GT z: -0.2~0.1)
        anchors_np = np.hstack([centers_xy.astype(np.float32), z])  # [900, 3]
        return torch.from_numpy(anchors_np)
    else:
        # K-means 파일 없으면 균일 그리드 fallback
        print("[anchor_generator] K-means 파일 없음 → 균일 그리드 사용")
        x = torch.linspace(-50, 50, 30)
        y = torch.linspace(-50, 50, 30)
        z = torch.tensor([0.0])
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
        return torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)


def generate_anchors_full():
    """
    SparseDrive 논문 기반 11차원 앵커: [900, 11]
    {x, y, z, ln_w, ln_h, ln_l, sin_yaw, cos_yaw, vx, vy, vz}
    논문 Section 2.1.1: 나머지 파라미터 {1, 1, 1, 0, 1, 0, 0, 0}
    """
    anchors_3d = generate_anchors()   # [900, 3]
    N = anchors_3d.shape[0]
    defaults = torch.tensor(
        [1., 1., 1., 0., 1., 0., 0., 0.]
    ).unsqueeze(0).expand(N, -1)
    return torch.cat([anchors_3d, defaults], dim=-1)   # [900, 11]


if __name__ == "__main__":
    a3  = generate_anchors()
    a11 = generate_anchors_full()
    print(f"앵커 수: {a3.shape[0]}")
    print(f"x 범위: {a3[:,0].min():.1f} ~ {a3[:,0].max():.1f}")
    print(f"y 범위: {a3[:,1].min():.1f} ~ {a3[:,1].max():.1f}")
    print(f"z 값:   {a3[:,2].unique()}")