import torch
import torch.nn as nn

class TemporalDecoder(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        # 과거 특징과 현재 특징을 융합해주는 인공신경망
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def ego_motion_compensation(self, past_anchors, ego_translation, ego_rotation):
        """
        past_anchors: [900, 3] 과거 프레임의 3D 앵커 좌표 (x, y, z)
        ego_translation: [3] 내 차가 과거->현재까지 이동한 거리 (x, y, z)
        ego_rotation: [3, 3] 내 차가 과거->현재까지 회전한 각도 (행렬)
        """
        # 1. 과거의 앵커들을 현재 내 차의 위치 기준으로 끌고 옴 (수학적 보정)
        # 공식: (과거좌표 - 이동거리) * 회전역행렬
        compensated_anchors = past_anchors - ego_translation
        compensated_anchors = (ego_rotation.T @ compensated_anchors.T).T
        
        return compensated_anchors

    def forward(self, current_features, past_features, current_anchors, past_anchors, ego_trans, ego_rot, track_threshold=0.5):
        """
        current_features, past_features: [900, 256] 탐정들이 찾은 특징들
        """
        # 1. 과거 앵커 위치를 현재 내 차 기준으로 보정
        aligned_past_anchors = self.ego_motion_compensation(past_anchors, ego_trans, ego_rot)
        
        # 2. (개념적) 과거 특징과 현재 특징 융합
        # 실제 논문에서는 Attention을 쓰지만, 여기서는 직관적으로 이어붙여(concat) 융합합니다.
        fused_features = torch.cat([current_features, past_features], dim=-1) # [900, 512]
        updated_features = self.fusion_net(fused_features) # [900, 256]
        
        # 3. 트래킹 (ID 부여 로직 예시)
        # 임계값(threshold)을 넘는 의미 있는 객체만 남깁니다.
        # (실제로는 FFN의 분류 결과 확률값이 track_threshold를 넘는지 확인합니다)
        track_ids = torch.arange(900) # 초기 ID 부여 (0~899)
        
        return updated_features, aligned_past_anchors, track_ids

if __name__ == "__main__":
    print("🚀 시간적 디코더(자차 이동 보정) 테스트 시작!")
    
    # 가짜 데이터 준비
    dummy_past_anchors = torch.randn(900, 3)     # 1초 전 앵커 위치
    dummy_current_anchors = torch.randn(900, 3)  # 현재 앵커 위치
    dummy_past_features = torch.randn(900, 256)
    dummy_current_features = torch.randn(900, 256)
    
    # 모라이(MORAI)에서 받아올 자차 이동 정보 (예: 차가 앞으로 2.5m 직진함)
    ego_translation = torch.tensor([2.5, 0.0, 0.0]) 
    ego_rotation = torch.eye(3) # 회전은 안 했다고 가정
    
    decoder = TemporalDecoder()
    
    # 실행
    updated_feat, aligned_anchors, ids = decoder(
        dummy_current_features, dummy_past_features, 
        dummy_current_anchors, dummy_past_anchors, 
        ego_translation, ego_rotation
    )
    
    print(f"✅ 과거 앵커 보정 완료: {aligned_anchors.shape}")
    print(f"✅ 시간적 특징 융합 완료: {updated_feat.shape}")
    # 과거 앵커의 X좌표가 내 차가 이동한 만큼(-2.5) 뒤로 밀렸는지 확인!
    print(f"비교 - 과거 원본 X: {dummy_past_anchors[0][0]:.2f} -> 보정된 X: {aligned_anchors[0][0]:.2f}")
