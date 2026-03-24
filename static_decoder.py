import torch
import torch.nn as nn

# 1. 정적 폴리라인 앵커 생성기
def generate_polyline_anchors(num_lines=100, points_per_line=20):
    """
    100개의 차선/경계선 앵커를 생성합니다.
    각 선은 20개의 점으로 이루어져 있습니다.
    
    수정: SparseDrive 논문 L_m ∈ R^{N_m × N_p × 2}에 따라
    3D(x,y,z) → 2D(x,y)로 변경. 도로 위 요소이므로 z=0 가정.
    """
    # [100, 20, 2] 형태의 텐서 생성 (100개 선, 각 20개 점, x/y 좌표)
    anchors = torch.zeros(num_lines, points_per_line, 2)
    
    for i in range(num_lines):
        y_pos = (i / num_lines) * 100 - 50 
        anchors[i, :, 0] = torch.linspace(-50, 50, points_per_line)  # X
        anchors[i, :, 1] = y_pos                                      # Y
        
    return anchors

# 2. 정적 맵 정보 해독기 (FFN)
class StaticMapDecoder(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=3, points_per_line=20):
        super(StaticMapDecoder, self).__init__()
        self.points_per_line = points_per_line
        
        # 1. 분류기: 이 선은 3가지 중 무엇인가?
        # (0: 차선경계, 1: 보행자횡단보도, 2: 도로경계)
        self.cls_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # 2. 회귀: 20개 점의 위치(x, y)를 보정
        # 수정: 20개 점 * 2D 좌표(x, y) = 40 (기존 60에서 변경)
        self.reg_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, points_per_line * 2)  # ← 3에서 2로 변경!
        )

    def forward(self, sampled_features):
        # sampled_features: [100, 256]
        
        # 1. 분류 → [100, 3]
        class_preds = self.cls_branch(sampled_features) 
        
        # 2. 폴리라인 보정 → [100, 40] → [100, 20, 2]
        line_preds = self.reg_branch(sampled_features)   
        line_preds = line_preds.view(-1, self.points_per_line, 2)  # ← 3에서 2로 변경!
        
        return class_preds, line_preds

if __name__ == "__main__":
    print("🚀 정적 맵 폴리라인 디코더 테스트 (2D 버전)\n")
    
    poly_anchors = generate_polyline_anchors()
    print(f"1. 폴리라인 앵커: {poly_anchors.shape} (100개 선, 20개 점, 2D좌표)")
    
    dummy_features = torch.randn(100, 256)
    map_decoder = StaticMapDecoder()
    class_out, line_out = map_decoder(dummy_features)
    
    print(f"\n2. 결과:")
    print(f"✅ 클래스 예측: {class_out.shape} (100개 선의 3가지 클래스)")
    print(f"✅ 폴리라인 예측: {line_out.shape} (100개 선의 20개 점 2D좌표)")