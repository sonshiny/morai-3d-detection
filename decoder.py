import math  # prior bias 계산(-log((1-π)/π))에 필요
import torch
import torch.nn as nn


class FFNDecoder(nn.Module):
    """
    Detection decoder.
    num_classes includes background. Current project uses:
      0=vehicle, 1=pedestrian, 2=background
    """
    def __init__(self, hidden_dim=256, num_classes=2):  
        super(FFNDecoder, self).__init__()

        # 1. 분류기 (vehicle / pedestrian / background)
        self.cls_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

        # 2. 3D 바운딩 박스 회귀 (11개의 특성 예측)
        # x, y, z, w, l, h, sin_yaw, cos_yaw, vx, vy, vz
        self.reg_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 11)
        )

        # 3. box quality/confidence 보정 branch
        #    channel 0: centerness, channel 1: yawness (official SparseDrive style)
        self.quality_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

        # RetinaNet 스타일 prior bias 초기화: 초기 예측을 "기본 background"로 만들어 cls loss 폭발 방지
        prior_prob = 0.1  # 0.01은 객체점수 0.22가 bg 0.77 못넘김 → 0.1로 완화
        bias_value = -math.log((1.0 - prior_prob) / prior_prob)  # 객체 클래스 bias ≈ -2.197
        # cls_branch의 마지막 nn.Linear를 인덱스 하드코딩 없이 동적으로 탐색
        last_cls_linear = [m for m in self.cls_branch if isinstance(m, nn.Linear)][-1]
        nn.init.constant_(last_cls_linear.bias, bias_value)        # 모든 클래스 bias를 낮게 시작
        last_cls_linear.bias.data[num_classes - 1] = 0.0           # background(마지막 인덱스) bias만 0으로 복원
        # K-means anchor가 이미 GT 중심을 잘 덮고 있으므로, refinement는
        # 처음에 anchor를 망가뜨리지 않도록 0 offset에서 시작한다.
        last_reg_linear = [m for m in self.reg_branch if isinstance(m, nn.Linear)][-1]
        nn.init.constant_(last_reg_linear.weight, 0.0)
        nn.init.constant_(last_reg_linear.bias, 0.0)
        # Official SparseDrive does not push quality logits strongly negative at init.
        # Keep centerness/yawness near sigmoid(0)=0.5 so calibrated scores are
        # observable from the first epochs.
        nn.init.constant_(self.quality_branch[-1].bias, 0.0)

    def forward(self, sampled_features):
        # sampled_features: [900, 256]
        class_preds = self.cls_branch(sampled_features)
        box_preds   = self.reg_branch(sampled_features)  # [900, 11]
        quality     = self.quality_branch(sampled_features)
        return class_preds, box_preds, quality


if __name__ == "__main__":
    dummy_features = torch.randn(900, 256)
    decoder = FFNDecoder()
    class_out, box_out, quality_out = decoder(dummy_features)
    print("✅ FFN 디코더 테스트 성공!")
    print(f"분류 결과 크기: {class_out.shape}")
    print(f"박스 예측 크기: {box_out.shape}   → [900, 11]")
    print(f"품질 예측 크기: {quality_out.shape}   → [900, 2]")
