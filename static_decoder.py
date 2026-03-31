import torch
import torch.nn as nn

# 1. 정적 폴리라인 앵커 생성기
def generate_polyline_anchors(num_lines=150, points_per_line=20):
    import os
    import numpy as np
    kmeans_path = os.path.join(os.path.dirname(__file__), 'map_kmeans_centers.npy')
    if os.path.isfile(kmeans_path):
        centers = np.load(kmeans_path)  # [150, 2]
        anchors = torch.zeros(num_lines, points_per_line, 2)
        for i in range(num_lines):
            cx, cy = float(centers[i, 0]), float(centers[i, 1])
            anchors[i, :, 0] = torch.linspace(cx - 5.0, cx + 5.0, points_per_line)
            anchors[i, :, 1] = cy
        return anchors
    else:
        print("[anchor] K-means 파일 없음 → 균일 그리드 사용")
        anchors = torch.zeros(num_lines, points_per_line, 2)
        for i in range(num_lines):
            y_pos = -110 + (220 / num_lines) * i
            anchors[i, :, 0] = torch.linspace(-110, 110, points_per_line)
            anchors[i, :, 1] = y_pos
        return anchors


# 2. 정적 맵 정보 해독기 (FFN)
class StaticMapDecoder(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=3, points_per_line=20):
        super(StaticMapDecoder, self).__init__()
        self.points_per_line = points_per_line

        # 분류기: 차선경계(0) / 횡단보도(1) / 도로경계(2)
        self.cls_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes+1)
        )

        # 회귀: 20개 점 * 2D(x,y) = 40
        self.reg_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, points_per_line * 2)
        )

    def forward(self, sampled_features):
        # sampled_features: [150, 256]
        class_preds = self.cls_branch(sampled_features)         # [150, 3]
        line_preds = self.reg_branch(sampled_features)          # [150, 40]
        line_preds = line_preds.view(-1, self.points_per_line, 2)  # [150, 20, 2]
        return class_preds, line_preds


if __name__ == "__main__":
    print("🚀 정적 맵 폴리라인 디코더 테스트\n")

    poly_anchors = generate_polyline_anchors()
    print(f"폴리라인 앵커: {poly_anchors.shape}")
    print(f"앵커 x 범위: {poly_anchors[:,:,0].min():.1f} ~ {poly_anchors[:,:,0].max():.1f}")
    print(f"앵커 y 범위: {poly_anchors[:,:,1].min():.1f} ~ {poly_anchors[:,:,1].max():.1f}")

    dummy_features = torch.randn(150, 256)
    map_decoder = StaticMapDecoder()
    class_out, line_out = map_decoder(dummy_features)

    print(f"\n클래스 예측: {class_out.shape}")
    print(f"폴리라인 예측: {line_out.shape}")