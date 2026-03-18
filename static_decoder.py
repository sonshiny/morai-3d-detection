import torch
import torch.nn as nn

# 1. 정적 폴리라인 앵커 생성기
def generate_polyline_anchors(num_lines=100, points_per_line=20):
    """
    100개의 차선/경계선 앵커를 생성합니다.
    각 선은 20개의 점으로 이루어져 있습니다.
    """
    # [100, 20, 3] 형태의 텐서 생성 (100개 선, 각 20개 점, x/y/z 좌표)
    # 초기에는 도로(자차 주변 -50m ~ 50m)에 대략적인 격자 형태로 선을 흩뿌려놓습니다.
    # (실제 학습 시에는 K-means나 훈련 데이터 통계를 기반으로 초기화합니다)
    anchors = torch.zeros(num_lines, points_per_line, 3)
    
    # 예시: 임의로 x축, y축 방향으로 뻗은 선들을 초기화
    for i in range(num_lines):
        # x좌표는 -50 ~ 50 사이를 20등분, y좌표는 선마다 다르게 배치
        y_pos = (i / num_lines) * 100 - 50 
        anchors[i, :, 0] = torch.linspace(-50, 50, points_per_line) # X
        anchors[i, :, 1] = y_pos                                    # Y
        anchors[i, :, 2] = 0.0                                      # Z (도로 바닥이므로 0)
        
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
        
        # 2. 회귀(Regression): 20개 점의 위치(x, y, z)를 얼마나 깎고 다듬을 것인가?
        # 출력 크기: 20개 점 * 3D 좌표(x, y, z) = 60
        self.reg_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, points_per_line * 3)
        )

    def forward(self, sampled_features):
        # sampled_features: [100, 256] 
        # (100개의 폴리라인 앵커가 도서관에서 수집해 온 256개의 특징)
        
        # 1. 어떤 종류의 선인지 예측 (차선? 횡단보도? 도로경계?) -> [100, 3]
        class_preds = self.cls_branch(sampled_features) 
        
        # 2. 구불구불한 선의 모양을 정확하게 보정 -> [100, 60]
        # 결과를 다시 [100, 20, 3] 형태로 예쁘게 접어줍니다.
        line_preds = self.reg_branch(sampled_features)   
        line_preds = line_preds.view(-1, self.points_per_line, 3) 
        
        return class_preds, line_preds

if __name__ == "__main__":
    print("🚀 정적 맵(Static Map) 폴리라인 디코더 테스트 시작!\n")
    
    # 1. 100개의 폴리라인 앵커 생성 (각 선마다 20개의 점)
    poly_anchors = generate_polyline_anchors()
    print(f"1. 폴리라인 앵커 생성 완료: {poly_anchors.shape} (100개 선, 20개 점, 3D좌표)")
    
    # 2. 가짜 데이터 준비 (100개의 선이 특징 도서관에서 단서를 뽑아왔다고 가정)
    dummy_features = torch.randn(100, 256)
    
    # 3. 디코더 가동!
    map_decoder = StaticMapDecoder()
    class_out, line_out = map_decoder(dummy_features)
    
    print(f"\n2. 최종 결과물 도출 완료!")
    print(f"✅ 클래스(분류) 예측 크기: {class_out.shape} (100개 선의 3가지 클래스 확률)")
    print(f"✅ 폴리라인(형태) 예측 크기: {line_out.shape} (100개 선의 보정된 20개 점 3D좌표)")
