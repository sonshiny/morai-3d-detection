import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from morai_dataset import MoraiDataset, morai_collate_fn
from resnet_fpn import ResNet50_FPN, Bottleneck
from anchor_generator import generate_anchors, generate_anchors_full
from decoder import FFNDecoder
from static_decoder import StaticMapDecoder, generate_polyline_anchors
from loss_calculator import CustomLoss
from torch.utils.data import DataLoader

CAM_ORDER = ['cam_front', 'cam_front_left', 'cam_front_right',
             'cam_back',  'cam_back_left',  'cam_back_right']


# ===========================================================
# 멀티스케일 샘플링 함수
# ===========================================================
def sample_from_multiscale(features_list, grid_2d, valid_mask, N):
    """
    p2~p5 모든 스케일에서 특징을 샘플링하고 합산.
    
    기존: p2(56x56) 한 스케일에서만 샘플링
    변경: p2(56x56) + p3(28x28) + p4(14x14) + p5(7x7) 전부에서 샘플링
    
    효과: 가까운 큰 차는 p2에서, 먼 작은 차는 p4/p5에서 잘 잡힘
    
    features_list: [p2, p3, p4, p5] 각각 [1, 256, H, W]
    grid_2d:       [1, 1, N, 2] 정규화된 좌표 (-1 ~ 1)
    valid_mask:    [N] bool
    """
    combined = torch.zeros(N, 256, device=features_list[0].device)
    
    for feat in features_list:
        sampled = F.grid_sample(feat, grid_2d, align_corners=False)
        sampled = sampled.view(256, N).T           # [N, 256]
        mask = valid_mask.float().unsqueeze(1)      # [N, 1]
        sampled = sampled * mask
        combined = combined + sampled
    
    # 4개 스케일의 평균
    combined = combined / len(features_list)
    return combined


class AutoNavModel(nn.Module):
    """
    수정사항:
    1. 멀티스케일 특징 활용 (p2~p5 전부)
    2. static_decoder 연결 (정적 맵: 차선/횡단보도/도로경계)
    3. decoder 출력 num_classes=4 (배경 포함)
    4. 앵커 오프셋 메커니즘 (논문 Section 2.6)
       - 디코더가 절대 좌표 대신 앵커 대비 오프셋을 예측
       - final_box = anchor + offset → 학습 안정성 대폭 향상
    """
    def __init__(self):
        super().__init__()
        # 공유 인코더
        self.backbone = ResNet50_FPN(Bottleneck)

        # 동적 객체 디코더 (Detection)
        self.det_decoder = FFNDecoder(num_classes=4)        # car/truck/bus/background
        self.det_anchors_3d = generate_anchors()            # [900, 3] 이미지 투영용
        self.det_anchors_full = generate_anchors_full()     # [900, 11] 오프셋 기준

        # 정적 맵 디코더 (Online Mapping)
        self.map_decoder = StaticMapDecoder(num_classes=3)  # 차선경계/횡단보도/도로경계
        self.map_anchors = generate_polyline_anchors()      # [100, 20, 2]

    def _sample_features(self, features_list, anchors_3d, intrinsic, extrinsic, N):
        """
        3D 앵커를 2D로 투영하고 멀티스케일에서 특징 샘플링.
        Detection과 Online Mapping 둘 다 이 함수를 공유.
        """
        device = features_list[0].device
        
        # 동차 좌표로 변환
        if anchors_3d.dim() == 2:
            # Detection: [N, 3] → [N, 4]
            anchors_homo = torch.cat(
                [anchors_3d, torch.ones(N, 1, device=device)], dim=-1
            )
        else:
            # 이미 [N, 4]면 그대로
            anchors_homo = anchors_3d
        
        # 앵커 → 카메라 좌표 → 이미지 좌표
        points_cam = (extrinsic @ anchors_homo.T).T          # [N, 4]
        points_2d  = (intrinsic @ points_cam[:, :3].T).T     # [N, 3]
        depth = points_2d[:, 2]
        u = points_2d[:, 0] / (depth + 1e-6)
        v = points_2d[:, 1] / (depth + 1e-6)

        valid_mask = depth > 0.1

        # 정규화 좌표 (원본 해상도 기준)
        u_norm = (u / 640.0) * 2.0 - 1.0
        v_norm = (v / 480.0) * 2.0 - 1.0
        grid = torch.stack([u_norm, v_norm], dim=-1).view(1, 1, N, 2)

        # 멀티스케일 샘플링
        sampled = sample_from_multiscale(features_list, grid, valid_mask, N)
        return sampled, valid_mask

    def forward(self, images, intrinsics, extrinsics):
        """
        images     : [1, 6, 3, 224, 224]
        intrinsics : [1, 6, 3, 3]
        extrinsics : [1, 6, 4, 4]
        
        반환:
            det_classes  : [900, 4]   동적 객체 분류 (배경 포함)
            det_boxes    : [900, 11]  동적 객체 3D 박스
            map_classes  : [100, 3]   정적 맵 분류
            map_lines    : [100, 20, 3] 정적 맵 폴리라인
        """
        device = images.device
        self.det_anchors_3d = self.det_anchors_3d.to(device)
        self.det_anchors_full = self.det_anchors_full.to(device)
        self.map_anchors = self.map_anchors.to(device)

        N_det = self.det_anchors_3d.shape[0]  # 900
        N_map = self.map_anchors.shape[0]     # 100

        # Detection 앵커: [900, 4] 동차 좌표 (이미지 투영용)
        det_anchors_homo = torch.cat(
            [self.det_anchors_3d, torch.ones(N_det, 1, device=device)], dim=-1
        )
        
        # Mapping 앵커: [100, 20, 2] → 중심점 [100, 2] → z=0 추가 → [100, 4]
        map_centers = self.map_anchors.mean(dim=1)  # 20개 점의 평균 = 폴리라인 중심 [100, 2]
        map_centers_3d = torch.cat(
            [map_centers, torch.zeros(N_map, 1, device=device)], dim=-1
        )  # z=0 추가 → [100, 3]
        map_centers_homo = torch.cat(
            [map_centers_3d, torch.ones(N_map, 1, device=device)], dim=-1
        )  # 동차 좌표 → [100, 4]
        
        # 특징 수집 버퍼
        det_agg_features = torch.zeros(N_det, 256, device=device)
        map_agg_features = torch.zeros(N_map, 256, device=device)
        valid_cams = 0

        for cam_idx in range(6):
            cam_img = images[0, cam_idx]
            if cam_img.abs().sum() < 1e-6:
                continue

            # 1. 멀티스케일 특징맵 추출 (p2, p3, p4, p5)
            features_list = self.backbone(cam_img.unsqueeze(0))

            E = extrinsics[0, cam_idx]
            K = intrinsics[0, cam_idx]

            # 2. Detection 앵커 샘플링 (멀티스케일)
            det_points_cam = (E @ det_anchors_homo.T).T
            det_points_2d  = (K @ det_points_cam[:, :3].T).T
            det_depth = det_points_2d[:, 2]
            det_u = det_points_2d[:, 0] / (det_depth + 1e-6)
            det_v = det_points_2d[:, 1] / (det_depth + 1e-6)
            det_valid = det_depth > 0.1
            
            det_u_norm = (det_u / 640.0) * 2.0 - 1.0
            det_v_norm = (det_v / 480.0) * 2.0 - 1.0
            det_grid = torch.stack([det_u_norm, det_v_norm], dim=-1).view(1, 1, N_det, 2)
            
            det_sampled = sample_from_multiscale(features_list, det_grid, det_valid, N_det)

            # 3. Mapping 앵커 샘플링 (멀티스케일)
            map_points_cam = (E @ map_centers_homo.T).T
            map_points_2d  = (K @ map_points_cam[:, :3].T).T
            map_depth = map_points_2d[:, 2]
            map_u = map_points_2d[:, 0] / (map_depth + 1e-6)
            map_v = map_points_2d[:, 1] / (map_depth + 1e-6)
            map_valid = map_depth > 0.1
            
            map_u_norm = (map_u / 640.0) * 2.0 - 1.0
            map_v_norm = (map_v / 480.0) * 2.0 - 1.0
            map_grid = torch.stack([map_u_norm, map_v_norm], dim=-1).view(1, 1, N_map, 2)
            
            map_sampled = sample_from_multiscale(features_list, map_grid, map_valid, N_map)

            det_agg_features += det_sampled
            map_agg_features += map_sampled
            valid_cams += 1

        if valid_cams > 0:
            det_agg_features = det_agg_features / valid_cams
            map_agg_features = map_agg_features / valid_cams

        # 4. 디코더 통과 + 앵커 오프셋 메커니즘
        # 디코더는 '오프셋'을 예측하고, 앵커에 더해서 최종 예측을 만든다
        # 이전: 절대 좌표 직접 예측 (학습 매우 어려움)
        # 지금: anchor + small_offset (학습 안정적)
        det_classes, det_offsets = self.det_decoder(det_agg_features)
        det_boxes = self.det_anchors_full + det_offsets  # [900, 11]

        map_classes, map_offsets = self.map_decoder(map_agg_features)
        map_lines = self.map_anchors + map_offsets       # [100, 20, 2]

        return det_classes, det_boxes, map_classes, map_lines


# ===========================================================
# 정적 맵 Loss (간단 버전)
# ===========================================================
# 정적 맵 폴리라인 정규화 스케일
# MORAI 데이터 기준 좌표 범위가 ~[-60, +210]이므로 60m으로 정규화
POLYLINE_SCALE = 60.0

class StaticMapLoss(nn.Module):
    """
    정적 맵 디코더의 Loss.
    GT가 없으면 0을 반환 (아직 라벨이 없는 경우 대비).
    GT가 있으면 분류 + 폴리라인 회귀 Loss 계산.

    수정사항:
    - 폴리라인 좌표를 POLYLINE_SCALE로 정규화 (이전: 정규화 없음 → L1 ~50)
    - 논문 Section 2.9 가중치: λ_map_cls=1.0, λ_map_reg=10.0
    """
    def __init__(self):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, pred_classes, pred_lines, gt_classes, gt_lines):
        """
        pred_classes: [100, 3]
        pred_lines:   [100, 20, 2]    ← 2D (x,y)
        gt_classes:    [M]             (없으면 빈 텐서)
        gt_lines:      [M, 20, 2]     (없으면 빈 텐서)
        """
        device = pred_classes.device

        if gt_classes is None or gt_classes.shape[0] == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero

        # 간단한 매칭: GT 개수만큼 앞쪽 앵커에 매칭 (향후 Hungarian으로 개선)
        M = min(gt_classes.shape[0], pred_classes.shape[0])

        loss_cls = self.cls_loss(pred_classes[:M], gt_classes[:M])
        # 정규화된 L1 Loss (이전: 정규화 없어서 loss ~50 → 지금: ~0.5)
        loss_reg = F.l1_loss(pred_lines[:M] / POLYLINE_SCALE,
                             gt_lines[:M] / POLYLINE_SCALE)

        # 논문 Section 2.9: λ_map_cls=1.0, λ_map_reg=10.0
        return 1.0 * loss_cls + 10.0 * loss_reg


# ===========================================================
# 학습 루프
# ===========================================================
if __name__ == "__main__":
    print("SparseDrive 인지 모듈 학습을 시작합니다!")
    print("   - Detection: Focal Loss + 배경 클래스 (2.0*cls + 0.25*reg)")
    print("   - Online Mapping: 정규화된 L1 Loss (1.0*cls + 10.0*reg)")
    print("   - 앵커 오프셋 메커니즘 (anchor + offset)")
    print("   - Backbone LR 0.1x, CosineAnnealing\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[디바이스] {device}\n")

    model      = AutoNavModel().to(device)
    dataset    = MoraiDataset(dataset_dir='./dataset', split='train')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True,
                            collate_fn=morai_collate_fn, num_workers=2)
    
    # Detection Loss (Focal Loss + 배경)
    det_criterion = CustomLoss(num_classes=3).to(device)

    # Static Map Loss (GT 없으면 자동으로 0 반환)
    map_criterion = StaticMapLoss().to(device)

    # 논문 Section 2.10: AdamW + Cosine Annealing
    # Backbone LR을 낮게 설정 (논문: backbone_lr_scale=0.1~0.5)
    backbone_params = list(model.backbone.parameters())
    backbone_ids = set(id(p) for p in backbone_params)
    other_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 4e-5},   # backbone: 4e-4 * 0.1
        {'params': other_params,    'lr': 4e-4},    # 나머지: 4e-4
    ], weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)

    num_epochs   = 200
    best_loss    = float('inf')
    loss_history = []

    for epoch in range(num_epochs):
        model.train()
        print(f"========== [Epoch {epoch+1}/{num_epochs}] ==========")
        epoch_loss = 0.0

        for step, batch in enumerate(dataloader):
            images     = batch['images'].to(device)
            intrinsics = batch['intrinsics'].to(device)
            extrinsics = batch['extrinsics'].to(device)

            batch_loss     = 0.0
            batch_det_loss = 0.0
            batch_map_loss = 0.0
            n = len(batch['dynamic_gt_boxes'])

            for i in range(n):
                gt_boxes   = batch['dynamic_gt_boxes'][i].to(device)
                gt_classes = batch['dynamic_gt_labels'][i].to(device)

                # 정적 맵 GT (아직 라벨이 없으면 빈 텐서)
                static_gt_classes = batch.get('static_gt_labels', [None])[i] if 'static_gt_labels' in batch else None
                static_gt_lines   = batch.get('static_gt_polylines', [None])[i] if 'static_gt_polylines' in batch else None
                
                if static_gt_classes is not None:
                    static_gt_classes = static_gt_classes.to(device)
                if static_gt_lines is not None:
                    static_gt_lines = static_gt_lines.to(device)

                # Forward (Detection + Mapping 동시)
                det_classes, det_boxes, map_classes, map_lines = model(
                    images[i:i+1],
                    intrinsics[i:i+1],
                    extrinsics[i:i+1]
                )

                # Detection Loss
                det_loss, cls_loss, box_loss = det_criterion(
                    det_classes, det_boxes, gt_classes, gt_boxes
                )

                # Static Map Loss (GT 없으면 0)
                map_loss = map_criterion(
                    map_classes, map_lines, static_gt_classes, static_gt_lines
                )

                # SparseDrive 논문과 동일: L = Ldet + Lmap
                total_loss = det_loss + map_loss

                batch_loss     += total_loss
                batch_det_loss += det_loss.item()
                batch_map_loss += map_loss.item()

            batch_loss = batch_loss / n

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += batch_loss.item()

            if step % 10 == 0:
                print(f"  Step {step:03d} | Loss: {batch_loss.item():.4f} "
                      f"(Det: {batch_det_loss/n:.4f}, "
                      f"Map: {batch_map_loss/n:.4f})")

        scheduler.step()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"\n🚀 Epoch {epoch+1} 완료! 평균 Loss: {avg_loss:.4f} "
              f"| LR: {scheduler.get_last_lr()[0]:.2e}\n")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  💾 Best 모델 저장! Loss: {best_loss:.4f}\n")

    print("🎉 학습 완료!")
    torch.save(model.state_dict(), "morai_autonav_weights.pth")
    print("💾 최종 모델 저장 완료: morai_autonav_weights.pth")