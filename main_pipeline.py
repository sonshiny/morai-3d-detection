import torch
import torch.nn.functional as F
from anchor_generator import generate_anchors
from resnet_fpn import ResNet50_FPN, Bottleneck  # 드디어 도서관 모델을 불러옵니다!

def project_3d_to_2d(points_3d, intrinsic, extrinsic, img_H, img_W):
    # (이전과 완전히 동일한 투영 함수입니다)
    N = points_3d.shape[0]
    points_3d_homo = torch.cat([points_3d, torch.ones(N, 1, device=points_3d.device)], dim=-1)
    points_cam = (extrinsic @ points_3d_homo.T).T
    valid_z = points_cam[:, 2] > 0
    points_2d_homo = (intrinsic @ points_cam[:, :3].T).T
    u = points_2d_homo[:, 0] / points_2d_homo[:, 2]
    v = points_2d_homo[:, 1] / points_2d_homo[:, 2]
    points_2d = torch.stack([u, v], dim=-1)
    valid_u = (u >= 0) & (u < img_W)
    valid_v = (v >= 0) & (v < img_H)
    valid_mask = valid_z & valid_u & valid_v
    return points_2d, valid_mask

if __name__ == "__main__":
    print("🚀 [최종 E2E 테스트] 이미지 -> 도서관 -> 앵커 투영 -> 단서 획득\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 앵커 생성 및 가짜 카메라로 투영 (이전 단계)
    anchors = generate_anchors().to(device)
    IMG_H, IMG_W = 224, 224
    dummy_intrinsic = torch.tensor([[1000.0, 0.0, 112.0], [0.0, 1000.0, 112.0], [0.0, 0.0, 1.0]], device=device)
    dummy_extrinsic = torch.eye(4, device=device)
    dummy_extrinsic[2, 3] = 5.0

    points_2d, valid_mask = project_3d_to_2d(anchors, dummy_intrinsic, dummy_extrinsic, IMG_H, IMG_W)

    # =========================================================================
    # 🌟 [NEW] 여기서부터가 진짜 핵심 (Sampling) 입니다!
    # =========================================================================
    
    # 2. ResNet-FPN 모델 가동 (가짜 이미지 1장 투입)
    print("1. ResNet-FPN 도서관 가동 중...")
    model = ResNet50_FPN(Bottleneck).to(device)
    dummy_image = torch.randn(1, 3, IMG_H, IMG_W).to(device) # 카메라 1대 이미지
    
    # 모델 통과 -> 4가지 해상도(Multi-scale) 도서관 획득
    features = model(dummy_image) 
    p2_feature = features[0] # 가장 해상도가 높은 P2 [1, 256, 56, 56] 사용
    print(f"2. P2 특징 도서관 준비 완료: {p2_feature.shape}")

    # 3. 좌표 정규화 (현업 엔지니어들도 제일 많이 실수하는 부분!)
    # 파이토치의 F.grid_sample 함수는 픽셀 좌표(0~224)를 못 알아듣습니다. 
    # 무조건 -1.0(왼쪽 끝) ~ 1.0(오른쪽 끝) 사이의 비율로 바꿔줘야 합니다.
    u_norm = (points_2d[:, 0] / IMG_W) * 2.0 - 1.0
    v_norm = (points_2d[:, 1] / IMG_H) * 2.0 - 1.0
    grid = torch.stack([u_norm, v_norm], dim=-1) # [900, 2]

    # 형태 맞추기: [배치=1, 높이=1, 너비=900, 좌표=2]
    grid = grid.view(1, 1, -1, 2)

    # 4. 단서 빼오기 (Grid Sampling)
    # p2_feature(도서관)에서 grid(탐정 위치)에 있는 256개 데이터를 빨아들입니다.
    sampled_features = F.grid_sample(p2_feature, grid, align_corners=False) # [1, 256, 1, 900]
    
    # 우리가 보기 편하게 [900명 탐정, 256개 단서] 형태로 정리
    sampled_features = sampled_features.view(256, -1).T # [900, 256]
    print(f"3. 탐정들이 단서 획득 완료: {sampled_features.shape}")

    # 5. 카메라 밖의 엉뚱한 곳(쓰레기 값)을 쳐다본 탐정들의 단서는 0으로 리셋
    sampled_features[~valid_mask] = 0.0

    # 6. 결과 확인!
    valid_count = valid_mask.sum().item()
    print(f"\n✅ 렌즈 안에 들어온 앵커 수: {valid_count}개")
    
    if valid_count > 0:
        # 렌즈 안에 들어온 첫 번째 앵커의 번호 찾기
        first_valid_idx = torch.where(valid_mask)[0][0].item()
        print(f"✅ 유효한 첫 번째 앵커(번호 {first_valid_idx})가 뽑아온 단서(특징) 256개 중 앞 5개:")
        print(sampled_features[first_valid_idx][:5])
        
    print("\n🎉 축하합니다! [이미지 -> 도서관 -> 투영 -> 단서 획득] 전체 파이프라인 관통 성공!")
