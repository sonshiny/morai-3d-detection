import os
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
        self.det_decoder = FFNDecoder(num_classes=2)
        self.det_anchors_3d = generate_anchors()
        self.det_anchors_full = generate_anchors_full()
        self.map_decoder = StaticMapDecoder(num_classes=3)
        self.map_anchors = generate_polyline_anchors()      # [150, 20, 2]

    def forward(self, images, intrinsics, extrinsics):
        """
        images     : [B, 6, 3, H, W]
        intrinsics : [B, 6, 3, 3]
        extrinsics : [B, 6, 4, 4]
        반환       : det_classes [B,900,2], det_boxes [B,900,11],
                     map_classes [B,150,4], map_lines [B,150,20,2]
        """
        device = images.device
        B = images.shape[0]

        self.det_anchors_3d  = self.det_anchors_3d.to(device)
        self.det_anchors_full = self.det_anchors_full.to(device)
        self.map_anchors     = self.map_anchors.to(device)

        N_det = self.det_anchors_3d.shape[0]    # 900
        N_map = self.map_anchors.shape[0]       # 150

        det_anchors_homo = torch.cat(
            [self.det_anchors_3d, torch.ones(N_det, 1, device=device)], dim=-1
        )  # [900, 4]
        map_centers = self.map_anchors.mean(dim=1)  # [150, 2]
        map_centers_3d = torch.cat(
            [map_centers, torch.zeros(N_map, 1, device=device)], dim=-1
        )  # [150, 3]
        map_centers_homo = torch.cat(
            [map_centers_3d, torch.ones(N_map, 1, device=device)], dim=-1
        )  # [150, 4]

        # 배치별 출력 누적
        batch_det_classes = []
        batch_det_boxes   = []
        batch_map_classes = []
        batch_map_lines   = []

        for b in range(B):
            # ── 핵심 수정: 6카메라를 한 번에 backbone 통과 (BN batch_size=6) ──
            cam_imgs = images[b]                          # [6, 3, H, W]
            all_features = self.backbone(cam_imgs)        # list of [6, C, H, W]

            det_agg = torch.zeros(N_det, 256, device=device)
            map_agg = torch.zeros(N_map, 256, device=device)
            valid_cams = 0

            for cam_idx in range(6):
                cam_img = images[b, cam_idx]
                if cam_img.abs().sum() < 1e-6:
                    continue

                # 해당 카메라 슬라이스만 꺼냄
                features_list = [f[cam_idx:cam_idx+1] for f in all_features]

                E = extrinsics[b, cam_idx]   # [4, 4]
                K = intrinsics[b, cam_idx]   # [3, 3]

                # Detection 앵커 투영
                det_pts = (E @ det_anchors_homo.T).T     # [900, 4]
                det_depth = det_pts[:, 0]
                det_u = K[0,0] * (-det_pts[:, 1]) / (det_depth + 1e-6) + K[0,2]
                det_v = K[0,0] * (-det_pts[:, 2]) / (det_depth + 1e-6) + K[1,2]
                det_valid = det_depth > 0.1
                det_u_n = (det_u / 640.0) * 2.0 - 1.0
                det_v_n = (det_v / 480.0) * 2.0 - 1.0
                det_grid = torch.stack([det_u_n, det_v_n], dim=-1).view(1, 1, N_det, 2)
                det_sampled = sample_from_multiscale(features_list, det_grid, det_valid, N_det)

                # Map 앵커 투영
                map_pts = (E @ map_centers_homo.T).T     # [150, 4]
                map_depth = map_pts[:, 0]
                map_u = K[0,0] * (-map_pts[:, 1]) / (map_depth + 1e-6) + K[0,2]
                map_v = K[0,0] * (-map_pts[:, 2]) / (map_depth + 1e-6) + K[1,2]
                map_valid = map_depth > 0.1
                map_u_n = (map_u / 640.0) * 2.0 - 1.0
                map_v_n = (map_v / 480.0) * 2.0 - 1.0
                map_grid = torch.stack([map_u_n, map_v_n], dim=-1).view(1, 1, N_map, 2)
                map_sampled = sample_from_multiscale(features_list, map_grid, map_valid, N_map)

                det_agg += det_sampled
                map_agg += map_sampled
                valid_cams += 1

            if valid_cams > 0:
                det_agg = det_agg / valid_cams
                map_agg = map_agg / valid_cams

            det_cls, det_off = self.det_decoder(det_agg)          # [900,2], [900,11]
            det_box = self.det_anchors_full + det_off              # [900,11]

            map_cls, map_off = self.map_decoder(map_agg)          # [150,4], [150,40]
            map_line = self.map_anchors + map_off                  # [150,20,2]

            batch_det_classes.append(det_cls)
            batch_det_boxes.append(det_box)
            batch_map_classes.append(map_cls)
            batch_map_lines.append(map_line)

        # [B, 900, 2], [B, 900, 11], [B, 150, 4], [B, 150, 20, 2]
        return (torch.stack(batch_det_classes),
                torch.stack(batch_det_boxes),
                torch.stack(batch_map_classes),
                torch.stack(batch_map_lines))


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
        gt_idx   = gt_idx.to(device)

        target = torch.full((num_anchors,), self.bg_class,
                            dtype=torch.long, device=device)
        target[pred_idx] = gt_classes[gt_idx]

        loss_cls = F.cross_entropy(pred_classes, target)
        loss_reg = F.l1_loss(
            pred_lines[pred_idx] / POLYLINE_SCALE,
            gt_lines[gt_idx]     / POLYLINE_SCALE
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

    model = AutoNavModel().to(device)

    # ── checkpoint resume ──────────────────────────────────
    resume_ckpt = 'checkpoint_epoch30.pth'
    if os.path.isfile(resume_ckpt):
        model.load_state_dict(torch.load(resume_ckpt, map_location=device))
        start_epoch = 30
        print(f"[resume] {resume_ckpt} 로드 완료 → epoch {start_epoch+1}부터 이어서 학습\n")
    else:
        start_epoch = 0
        print("[resume] 체크포인트 없음 → 처음부터 학습\n")

    dataset    = MoraiDataset(dataset_dir='./dataset', split='train')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True,
                            collate_fn=morai_collate_fn, num_workers=2)

    det_criterion = CustomLoss(num_classes=1).to(device)
    map_criterion = StaticMapLoss().to(device)

    backbone_params = list(model.backbone.parameters())
    backbone_ids    = set(id(p) for p in backbone_params)
    other_params    = [p for p in model.parameters() if id(p) not in backbone_ids]

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 4e-5},
        {'params': other_params,    'lr': 4e-4},
    ], weight_decay=1e-3)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1e-6
    )
    # resume 시 scheduler 상태 맞추기
    for _ in range(start_epoch):
        scheduler.step()

    num_epochs     = 100
    best_loss      = float('inf')
    DET_EARLY_STOP = 0.05

    for epoch in range(start_epoch, num_epochs):
        model.train()
        print(f"========== [Epoch {epoch+1}/{num_epochs}] ==========")
        epoch_loss     = 0.0
        epoch_det_loss = 0.0
        epoch_map_loss = 0.0

        for step, batch in enumerate(dataloader):
            images     = batch['images'].to(device)       # [B, 6, 3, H, W]
            intrinsics = batch['intrinsics'].to(device)   # [B, 6, 3, 3]
            extrinsics = batch['extrinsics'].to(device)   # [B, 6, 4, 4]
            n = images.shape[0]

            # ── 핵심 수정: 배치 한 번 forward ──────────────
            det_classes_b, det_boxes_b, map_classes_b, map_lines_b = model(
                images, intrinsics, extrinsics
            )
            # det_classes_b : [B, 900, 2]
            # det_boxes_b   : [B, 900, 11]
            # map_classes_b : [B, 150, 4]
            # map_lines_b   : [B, 150, 20, 2]

            batch_loss     = 0.0
            batch_det_loss = 0.0
            batch_map_loss = 0.0

            for i in range(n):
                gt_boxes   = batch['dynamic_gt_boxes'][i].to(device)
                gt_classes = batch['dynamic_gt_labels'][i].to(device)

                static_gt_classes = batch['static_gt_labels'][i].to(device) \
                    if 'static_gt_labels' in batch else None
                static_gt_lines = batch['static_gt_polylines'][i].to(device) \
                    if 'static_gt_polylines' in batch else None

                det_loss, cls_loss, box_loss = det_criterion(
                    det_classes_b[i], det_boxes_b[i], gt_classes, gt_boxes
                )
                map_loss = map_criterion(
                    map_classes_b[i], map_lines_b[i],
                    static_gt_classes, static_gt_lines
                )

                total_i     = det_loss + map_loss
                batch_loss     += total_i
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

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  💾 Best 모델 저장! Loss: {best_loss:.4f}\n")

        if (epoch + 1) % 10 == 0:
            ckpt_path = f"checkpoint_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  📌 체크포인트 저장: {ckpt_path}\n")

        if avg_det_loss < DET_EARLY_STOP:
            print(f"⚠️  Early Stopping! Det Loss {avg_det_loss:.4f} < {DET_EARLY_STOP}")
            break

    print("🎉 학습 완료!")
    torch.save(model.state_dict(), "morai_autonav_weights.pth")
    print("💾 최종 모델 저장 완료: morai_autonav_weights.pth")