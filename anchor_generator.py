import torch

def generate_anchors():
    # 1. 그리드 설정 (피드백 주신 분의 가이드 반영)
    # x(전후) 15개, y(좌우) 15개, z(높이) 4개 -> 총 15 * 15 * 4 = 900개
    x = torch.linspace(-50, 50, 15)
    y = torch.linspace(-50, 50, 15)
    z = torch.linspace(-3, 3, 4)

    # 2. 3D 공간에 격자(Meshgrid) 만들기
    # meshgrid는 x, y, z의 모든 조합을 만들어줍니다.
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    
    # 3. [900, 3] 형태로 모양 변경 (각 점의 x, y, z 좌표)
    anchors = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
    
    return anchors

if __name__ == "__main__":
    anchors = generate_anchors()
    print(f"🚀 900개의 앵커 생성 완료!")
    print(f"앵커 텐서 크기: {anchors.shape}") # [900, 3]
    print(f"첫 번째 앵커 좌표: {anchors[0]}") # 맨 구석 점
    print(f"마지막 앵커 좌표: {anchors[-1]}") # 반대편 구석 점
