import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from morai_dataset import MoraiDataset, morai_collate_fn
from resnet_fpn import ResNet50_FPN, Bottleneck
from anchor_generator import generate_anchors, generate_anchors_full
from decoder import FFNDecoder
from static_decoder import StaticMapDecoder, generate_polyline_anchors
from loss_calculator import CustomLoss, MapHungarianMatcher
from torch.utils.data import DataLoader

CAM_ORDER = ['cam_front', 'cam_front_left', 'cam_front_right',
             'cam_back',  'cam_back_left',  'cam_back_right']


# ===========================================================
# 멀티스케일 샘플링 함수
# ===========================================================
def sample_from_multiscale(features_list, grid_2d, valid_mask, N):
    combined = torch.zeros(N, 256, device=features_list[0].device)
    for feat in features_list:
        sampled = F.grid_sample(feat, grid_2d, align_corners=False)
        sampled = sampled.view(256, N).T
        mask = valid_mask.float().unsqueeze(1)
        sampled = sampled * mask
        combined = combined + sampled
    combined = combined / len(features_list)
    return combined


class AutoNavModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet50_FPN(Bottleneck)
        self.det_decoder = FFNDecoder(num_classes=4)
        self.det_anchors_3d = generate_anchors()
        self.det_anchors_full = generate_anchors_full()
        self.map_decoder = StaticMapDecoder(num_classes=3)
        self.map_anchors = generate_polyline_anchors()      # [150, 20, 2]

    def forward(self, images, intrinsics, extrinsics):
        device = images.device
        self.det_anchors_3d = self.det_anchors_3d.to(device)
        self.det_anchors_full = self.det_anchors_full.to(device)
        self.map_anchors = self.map_anchors.to(device)

        N_det = self.det_anchors_3d.shape[0]   # 900
        N_map = self.map_anchors.shape[0]      # 150

        det_anchors_homo = torch.cat(
            [self.det_anchors_3d, torch.ones(N_det, 1, device=device)], dim=-1
        )
        map_centers = self.map_anchors.mean(dim=1)
        map_centers_3d = torch.cat(
            [map_centers, torch.zeros(N_map, 1, device=device)], dim=-1
        )
        map_centers_homo = torch.cat(
            [map_centers_3d, torch.ones(N_map, 1, device=device)], dim=-1
        )

        det_agg_features = torch.zeros(N_det, 256, device=device)
        map_agg_features = torch.zeros(N_map, 256, device=device)
        valid_cams = 0

        for cam_idx in range(6):
            cam_img = images[0, cam_idx]
            if cam_img.abs().sum() < 1e-6:
                continue

            features_list = self.backbone(cam_img.unsqueeze(0))
            E = extrinsics[0, cam_idx]
            K = intrinsics[0, cam_idx]

            det_points_cam = (E @ det_anchors_homo.T).T
            det_depth = det_points_cam[:, 0]
            det_u = K[0,0] * (-det_points_cam[:, 1]) / (det_depth + 1e-6) + K[0,2]
            det_v = K[0,0] * (-det_points_cam[:, 2]) / (det_depth + 1e-6) + K[1,2]
            det_valid = det_depth > 0.1
            det_u_norm = (det_u / 640.0) * 2.0 - 1.0
            det_v_norm = (det_v / 480.0) * 2.0 - 1.0
            det_grid = torch.stack([det_u_norm, det_v_norm], dim=-1).view(1, 1, N_det, 2)
            det_sampled = sample_from_multiscale(features_list, det_grid, det_valid, N_det)

            map_points_cam = (E @ map_centers_homo.T).T
            map_depth = map_points_cam[:, 0]
            map_u = K[0,0] * (-map_points_cam[:, 1]) / (map_depth + 1e-6) + K[0,2]
            map_v = K[0,0] * (-map_points_cam[:, 2]) / (map_depth + 1e-6) + K[1,2]
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

        det_classes, det_offsets = self.det_decoder(det_agg_features)
        det_boxes = self.det_anchors_full + det_offsets

        map_classes, map_offsets = self.map_decoder(map_agg_features)
        map_lines = self.map_anchors + map_offsets

        return det_classes, det_boxes, map_classes, map_lines


# ===========================================================
# 정적 맵 Loss
# ===========================================================
POLYLINE_SCALE = 110.0

class StaticMapLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bg_class = 3
        self.matcher = MapHungarianMatcher(cost_class=2.0, cost_line=5.0)

    def forward(self, pred_classes, pred_lines, gt_classes, gt_lines):
        device = pred_classes.device
        num_anchors = pred_classes.shape[0]

        if gt_classes is None or gt_classes.shape[0] == 0:
            target = torch.full((num_anchors,), self.bg_class,
                                dtype=torch.long, device=device)
            return F.cross_entropy(pred_classes, target) * 0.1

        pred_idx, gt_idx = self.matcher(
            pred_classes, pred_lines, gt_classes, gt_lines,
            polyline_scale=POLYLINE_SCALE
        )
        pred_idx = pred_idx.to(device)
        gt_idx = gt_idx.to(device)

        target = torch.full((num_anchors,), self.bg_class,
                            dtype=torch.long, device=device)
        target[pred_idx] = gt_classes[gt_idx]

        loss_cls = F.cross_entropy(pred_classes, target)
        loss_reg = F.l1_loss(
            pred_lines[pred_idx] / POLYLINE_SCALE,
            gt_lines[gt_idx] / POLYLINE_SCALE
        )
        return 1.0 * loss_cls + 10.0 * loss_reg


# ===========================================================
# 학습 루프
# ===========================================================
if __name__ == "__main__":
    print("SparseDrive 인지 모듈 학습을 시작합니다!")
    print("   - Detection: Focal Loss + 배경 클래스 (2.0*cls + 0.25*reg)")
    print("   - Online Mapping: 배경 클래스 포함 (1.0*cls + 10.0*reg)")
    print("   - Early Stopping: Det Loss < 0.05 시 자동 중단")
    print("   - 10 epoch마다 체크포인트 저장\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[디바이스] {device}\n")

    model      = AutoNavModel().to(device)
    dataset    = MoraiDataset(dataset_dir='./dataset', split='train')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True,
                            collate_fn=morai_collate_fn, num_workers=2)

    det_criterion = CustomLoss(num_classes=3).to(device)
    map_criterion = StaticMapLoss().to(device)

    backbone_params = list(model.backbone.parameters())
    backbone_ids = set(id(p) for p in backbone_params)
    other_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 4e-5},
        {'params': other_params,    'lr': 4e-4},
    ], weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    num_epochs = 100
    best_loss  = float('inf')
    # Early Stopping: Det Loss 이 임계값 이하로 내려가면 중단
    DET_EARLY_STOP = 0.05

    for epoch in range(num_epochs):
        model.train()
        print(f"========== [Epoch {epoch+1}/{num_epochs}] ==========")
        epoch_loss     = 0.0
        epoch_det_loss = 0.0
        epoch_map_loss = 0.0

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

                static_gt_classes = batch.get('static_gt_labels', [None])[i] if 'static_gt_labels' in batch else None
                static_gt_lines   = batch.get('static_gt_polylines', [None])[i] if 'static_gt_polylines' in batch else None

                if static_gt_classes is not None:
                    static_gt_classes = static_gt_classes.to(device)
                if static_gt_lines is not None:
                    static_gt_lines = static_gt_lines.to(device)

                det_classes, det_boxes, map_classes, map_lines = model(
                    images[i:i+1], intrinsics[i:i+1], extrinsics[i:i+1]
                )

                det_loss, cls_loss, box_loss = det_criterion(
                    det_classes, det_boxes, gt_classes, gt_boxes
                )
                map_loss = map_criterion(
                    map_classes, map_lines, static_gt_classes, static_gt_lines
                )

                total_loss = det_loss + map_loss
                batch_loss     += total_loss
                batch_det_loss += det_loss.item()
                batch_map_loss += map_loss.item()

            batch_loss = batch_loss / n

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss     += batch_loss.item()
            epoch_det_loss += batch_det_loss / n
            epoch_map_loss += batch_map_loss / n

            if step % 10 == 0:
                print(f"  Step {step:03d} | Loss: {batch_loss.item():.4f} "
                      f"(Det: {batch_det_loss/n:.4f}, Map: {batch_map_loss/n:.4f})")

        scheduler.step()

        avg_loss     = epoch_loss     / len(dataloader)
        avg_det_loss = epoch_det_loss / len(dataloader)
        avg_map_loss = epoch_map_loss / len(dataloader)

        print(f"\n🚀 Epoch {epoch+1} 완료! "
              f"평균 Loss: {avg_loss:.4f} "
              f"(Det: {avg_det_loss:.4f}, Map: {avg_map_loss:.4f}) "
              f"| LR: {scheduler.get_last_lr()[0]:.2e}\n")

        # Best 모델 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  💾 Best 모델 저장! Loss: {best_loss:.4f}\n")

        # 10 epoch마다 체크포인트 저장
        if (epoch + 1) % 10 == 0:
            ckpt_path = f"checkpoint_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  📌 체크포인트 저장: {ckpt_path}\n")

        # Early Stopping: Det Loss 오버피팅 감지
        if avg_det_loss < DET_EARLY_STOP:
            print(f"⚠️  Early Stopping! Det Loss {avg_det_loss:.4f} < {DET_EARLY_STOP}")
            print(f"   마지막 체크포인트: checkpoint_epoch{((epoch)//10)*10}.pth 사용 권장\n")
            break

    print("🎉 학습 완료!")
    torch.save(model.state_dict(), "morai_autonav_weights.pth")
    print("💾 최종 모델 저장 완료: morai_autonav_weights.pth")