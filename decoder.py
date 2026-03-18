import torch
import torch.nn as nn

class FFNDecoder(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=3):  # 3 = car, truck, bus
        super(FFNDecoder, self).__init__()
        
        # 1. 분류기 (어떤 차량인가?)
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

    def forward(self, sampled_features):
        # sampled_features: [900, 256]
        class_preds = self.cls_branch(sampled_features)  # [900, 3]
        box_preds   = self.reg_branch(sampled_features)  # [900, 11]
        return class_preds, box_preds

if __name__ == "__main__":
    dummy_features = torch.randn(900, 256)
    decoder = FFNDecoder()
    class_out, box_out = decoder(dummy_features)
    print("✅ FFN 디코더 테스트 성공!")
    print(f"분류 결과 크기: {class_out.shape}  → [900, 3]")
    print(f"박스 예측 크기: {box_out.shape}   → [900, 11]")