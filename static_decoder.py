import torch
import torch.nn as nn

# 1. 정적 폴리라인 앵커 생성기
def generate_polyline_anchors(num_lines=150, points_per_line=20):
    import os
    import numpy as np
    kmeans_path = os.path.join(os.path.dirname(__file__), 'map_kmeans_centers.npy')
    if os.path.isfile(kmeans_path):
        centers = np.load(kmeans_path)
        if centers.shape != (num_lines, 2):
            print(f"[map anchor] ⚠️  K-means shape 이상: {centers.shape}, fallback 사용")
        else:
            # K-means 중심 주위로 폴리라인 펼침 (전방 방향, x축으로 30m)
            anchors = torch.zeros(num_lines, points_per_line, 2)
            for i in range(num_lines):
                cx, cy = float(centers[i, 0]), float(centers[i, 1])
                anchors[i, :, 0] = torch.linspace(cx - 15.0, cx + 15.0, points_per_line)
                anchors[i, :, 1] = cy
            return anchors

    # Fallback: 격자 폴리라인 (전방위 분포)
    print("[map anchor] K-means 파일 없음 → 격자 폴리라인 사용")
    anchors = torch.zeros(num_lines, points_per_line, 2)
    n_per_row = 15
    n_rows = num_lines // n_per_row + 1
    idx = 0
    for r in range(n_rows):
        for c in range(n_per_row):
            if idx >= num_lines:
                break
            y_pos = -50 + (100 / n_per_row) * c
            x_start = -50 + (100 / n_rows) * r
            anchors[idx, :, 0] = torch.linspace(x_start, x_start + 15, points_per_line)
            anchors[idx, :, 1] = y_pos
            idx += 1
    return anchors


# 2. 정적 맵 디코더 (FFN)
class StaticMapDecoder(nn.Module):
    """
    클래스 출력: num_classes + 1 (배경 포함)
    예: num_classes=3 (lane_boundary, crosswalk, road_boundary) + 1 (background) = 4
    """
    def __init__(self, hidden_dim=256, num_classes=3, points_per_line=20):
        super(StaticMapDecoder, self).__init__()
        self.points_per_line = points_per_line

        self.cls_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes + 1)
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
        class_preds = self.cls_branch(sampled_features)            # [150, 4]
        line_preds = self.reg_branch(sampled_features)              # [150, 40]
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

    print(f"\n클래스 예측: {class_out.shape}  → [150, 4] (배경 포함)")
    print(f"폴리라인 예측: {line_out.shape}")
