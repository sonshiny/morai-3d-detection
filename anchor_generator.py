import torch


def generate_anchors():
    """3D 위치만 반환: [900, 3] (이미지 투영용)"""
    x = torch.linspace(-50, 50, 15)
    y = torch.linspace(-50, 50, 15)
    z = torch.linspace(-3, 3, 4)

    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    anchors = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
    return anchors


def generate_anchors_full():
    """
    SparseDrive 논문 기반 11차원 앵커 생성: [900, 11]
    {x, y, z, ln_w, ln_h, ln_l, sin_yaw, cos_yaw, vx, vy, vz}

    - x, y, z: 균일 그리드 (향후 K-means로 교체 가능)
    - ln_w, ln_h, ln_l: 고정값 1.0 (실제 크기 e^1 ≈ 2.7m)
    - sin_yaw=0, cos_yaw=1: 정면 방향
    - vx, vy, vz: 0 (정지 상태)
    """
    anchors_3d = generate_anchors()  # [900, 3]
    N = anchors_3d.shape[0]
    # 논문 Section 2.1.1: {1, 1, 1, 0, 1, 0, 0, 0}
    defaults = torch.tensor([1., 1., 1., 0., 1., 0., 0., 0.]).unsqueeze(0).expand(N, -1)
    return torch.cat([anchors_3d, defaults], dim=-1)  # [900, 11]


if __name__ == "__main__":
    anchors_3d = generate_anchors()
    anchors_full = generate_anchors_full()
    print(f"3D 앵커: {anchors_3d.shape}")        # [900, 3]
    print(f"11D 앵커: {anchors_full.shape}")      # [900, 11]
    print(f"첫 번째 11D 앵커: {anchors_full[0]}") # [x, y, z, 1, 1, 1, 0, 1, 0, 0, 0]
