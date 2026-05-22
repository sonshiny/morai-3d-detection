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


# ===========================================================
# [P0-#1] 멀티스케일 샘플링 — per-point visible mask 반환
# ===========================================================
def sample_from_multiscale(features_list, grid_2d, valid_mask, N):
    """
    features_list : list of [1, 256, H, W]  (각 스케일)
    grid_2d       : [1, 1, N, 2]            (normalized -1~1)
    valid_mask    : [N] bool                (depth > 0.1)
    N             : 점 개수
    반환          : [N, 256] sampled feature (invalid는 0)
    """
    combined = torch.zeros(N, 256, device=features_list[0].device)
    for feat in features_list:
        sampled = F.grid_sample(feat, grid_2d, align_corners=False)
        sampled = sampled.view(256, N).T            # [N, 256]
        sampled = sampled * valid_mask.float().unsqueeze(1)
        combined = combined + sampled
    combined = combined / len(features_list)        # 4 스케일 평균
    return combined


# ===========================================================
# [P0-#2] anchor 박스 → 5개 키포인트 (중심 + BEV 4 corner)
# ===========================================================
def generate_5_keypoints(anchors_full):
    """
    anchors_full : [N, 11] = x,y,z,ln_w,ln_l,ln_h,sin_yaw,cos_yaw,vx,vy,vz
    반환         : [N, 5, 3] = (중심, FL, FR, RL, RR)
                   각 점은 (x,y,z) ego 좌표
    """
    device = anchors_full.device
    N = anchors_full.shape[0]

    xyz = anchors_full[:, 0:3]           # [N, 3]
    w = torch.exp(anchors_full[:, 3])    # [N]  (좌우 폭)
    l = torch.exp(anchors_full[:, 4])    # [N]  (전후 길이)
    sin_y = anchors_full[:, 6]           # [N]
    cos_y = anchors_full[:, 7]           # [N]

    half_l = (l * 0.5).unsqueeze(-1)     # [N, 1]
    half_w = (w * 0.5).unsqueeze(-1)     # [N, 1]

    # BEV 평면 4 꼭짓점 (local frame: x=forward, y=left)
    corners_local = torch.stack([
        torch.cat([ half_l,  half_w], dim=-1),   # FL
        torch.cat([ half_l, -half_w], dim=-1),   # FR
        torch.cat([-half_l,  half_w], dim=-1),   # RL
        torch.cat([-half_l, -half_w], dim=-1),   # RR
    ], dim=1)                                    # [N, 4, 2]

    # yaw 회전 적용 (rel_yaw 정의는 GT와 같음)
    cos_e = cos_y.unsqueeze(-1).unsqueeze(-1)    # [N, 1, 1]
    sin_e = sin_y.unsqueeze(-1).unsqueeze(-1)    # [N, 1, 1]
    x_l = corners_local[..., 0:1]                # [N, 4, 1]
    y_l = corners_local[..., 1:2]                # [N, 4, 1]
    x_r = cos_e * x_l - sin_e * y_l              # [N, 4, 1]
    y_r = sin_e * x_l + cos_e * y_l              # [N, 4, 1]
    corners_rot = torch.cat([x_r, y_r], dim=-1)  # [N, 4, 2]

    # ego 좌표로 평행이동 + z 추가
    corners_xy = corners_rot + xyz[:, 0:2].unsqueeze(1)              # [N, 4, 2]
    corners_z  = xyz[:, 2:3].unsqueeze(1).expand(-1, 4, 1)           # [N, 4, 1]
    corners_3d = torch.cat([corners_xy, corners_z], dim=-1)          # [N, 4, 3]

    # 중심 + 4 corner = 5 keypoints
    center_3d = xyz.unsqueeze(1)                                     # [N, 1, 3]
    keypoints = torch.cat([center_3d, corners_3d], dim=1)            # [N, 5, 3]
    return keypoints


# ===========================================================
# 모델
# ===========================================================
class AutoNavModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet50_FPN(Bottleneck)
        # num_classes=3 → 출력 logits 3개: 0=vehicle, 1=pedestrian, 2=background
        self.det_decoder = FFNDecoder(num_classes=3)
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
        N_CAMS = images.shape[1]

        self.det_anchors_3d   = self.det_anchors_3d.to(device)
        self.det_anchors_full = self.det_anchors_full.to(device)
        self.map_anchors      = self.map_anchors.to(device)

        N_det = self.det_anchors_full.shape[0]    # 900
        N_map = self.map_anchors.shape[0]         # 150
        N_kp  = 5                                  # [P0-#2] 키포인트 개수
        N_pp  = self.map_anchors.shape[1]         # 20 (polyline points)

        # 배치별 출력 누적
        batch_det_classes = []
        batch_det_boxes   = []
        batch_map_classes = []
        batch_map_lines   = []

        for b in range(B):
            # 6 카메라를 한 번에 backbone 통과
            cam_imgs = images[b]                            # [6, 3, H, W]
            all_features = self.backbone(cam_imgs)          # list of [6, C, H, W]

            # [P0-#2] 동적 anchor → 5개 키포인트
            det_keypoints = generate_5_keypoints(self.det_anchors_full)   # [900, 5, 3]
            det_kp_flat = det_keypoints.view(N_det * N_kp, 3)             # [4500, 3]
            det_kp_homo = torch.cat(
                [det_kp_flat, torch.ones(N_det * N_kp, 1, device=device)],
                dim=-1
            )                                                              # [4500, 4]

            # [P0-#3] 정적 map polyline 20개 점 모두 사용
            map_pts_xy = self.map_anchors                                  # [150, 20, 2]
            map_pts_z  = torch.zeros(N_map, N_pp, 1, device=device)
            map_pts_3d = torch.cat([map_pts_xy, map_pts_z], dim=-1)        # [150, 20, 3]
            map_pts_flat = map_pts_3d.view(N_map * N_pp, 3)                # [3000, 3]
            map_pts_homo = torch.cat(
                [map_pts_flat, torch.ones(N_map * N_pp, 1, device=device)],
                dim=-1
            )                                                              # [3000, 4]

            # [P0-#1] anchor별 visible 카메라 카운트 (전체 카메라 수 아님)
            det_agg = torch.zeros(N_det, 256, device=device)
            map_agg = torch.zeros(N_map, 256, device=device)
            det_visible_cams = torch.zeros(N_det, device=device)
            map_visible_cams = torch.zeros(N_map, device=device)

            for cam_idx in range(N_CAMS):
                cam_img = images[b, cam_idx]
                if cam_img.abs().sum() < 1e-6:
                    continue

                features_list = [f[cam_idx:cam_idx+1] for f in all_features]
                E = extrinsics[b, cam_idx]   # [4, 4]
                K = intrinsics[b, cam_idx]   # [3, 3]
                fx = K[0, 0]
                cx = K[0, 2]
                cy = K[1, 2]

                # ─── Detection 키포인트 투영 ──────────────────────
                det_pts = (E @ det_kp_homo.T).T               # [4500, 4]
                det_depth = det_pts[:, 0]
                det_u = fx * (-det_pts[:, 1]) / (det_depth + 1e-6) + cx
                det_v = fx * (-det_pts[:, 2]) / (det_depth + 1e-6) + cy
                det_kp_valid = det_depth > 0.1                # [4500]
                det_u_n = (det_u / 1600.0) * 2.0 - 1.0
                det_v_n = (det_v / 900.0) * 2.0 - 1.0
                det_grid = torch.stack([det_u_n, det_v_n], dim=-1).view(1, 1, N_det * N_kp, 2)
                det_sampled_kp = sample_from_multiscale(
                    features_list, det_grid, det_kp_valid, N_det * N_kp
                )                                              # [4500, 256]

                # 5개 키포인트 평균 (이 카메라에서 보이는 점만)
                kp_mask = det_kp_valid.view(N_det, N_kp).float()        # [900, 5]
                kp_count = kp_mask.sum(dim=1)                            # [900]
                anchor_visible_in_cam = (kp_count > 0).float()           # [900]

                det_sampled_kp_view = det_sampled_kp.view(N_det, N_kp, 256)
                det_sampled_sum = det_sampled_kp_view.sum(dim=1)         # [900, 256]
                det_sampled = det_sampled_sum / kp_count.clamp(min=1).unsqueeze(-1)
                # 이 카메라에 안 보이면 0
                det_sampled = det_sampled * anchor_visible_in_cam.unsqueeze(-1)

                det_agg = det_agg + det_sampled
                det_visible_cams = det_visible_cams + anchor_visible_in_cam

                # ─── Map polyline 점 투영 ────────────────────────
                map_pts_cam = (E @ map_pts_homo.T).T          # [3000, 4]
                map_depth = map_pts_cam[:, 0]
                map_u = fx * (-map_pts_cam[:, 1]) / (map_depth + 1e-6) + cx
                map_v = fx * (-map_pts_cam[:, 2]) / (map_depth + 1e-6) + cy
                map_pt_valid = map_depth > 0.1                # [3000]
                map_u_n = (map_u / 1600.0) * 2.0 - 1.0
                map_v_n = (map_v / 900.0) * 2.0 - 1.0
                map_grid = torch.stack([map_u_n, map_v_n], dim=-1).view(1, 1, N_map * N_pp, 2)
                map_sampled_pt = sample_from_multiscale(
                    features_list, map_grid, map_pt_valid, N_map * N_pp
                )                                              # [3000, 256]

                # 20개 점 평균 (이 카메라에서 보이는 점만)
                pt_mask = map_pt_valid.view(N_map, N_pp).float()         # [150, 20]
                pt_count = pt_mask.sum(dim=1)                             # [150]
                map_visible_in_cam = (pt_count > 0).float()               # [150]

                map_sampled_pt_view = map_sampled_pt.view(N_map, N_pp, 256)
                map_sampled_sum = map_sampled_pt_view.sum(dim=1)          # [150, 256]
                map_sampled = map_sampled_sum / pt_count.clamp(min=1).unsqueeze(-1)
                map_sampled = map_sampled * map_visible_in_cam.unsqueeze(-1)

                map_agg = map_agg + map_sampled
                map_visible_cams = map_visible_cams + map_visible_in_cam

            # [P0-#1] anchor별 visible 카메라 수로 정규화
            det_agg = det_agg / det_visible_cams.clamp(min=1).unsqueeze(-1)
            map_agg = map_agg / map_visible_cams.clamp(min=1).unsqueeze(-1)

            # FFN 디코더
            det_cls, det_off = self.det_decoder(det_agg)        # [900,2], [900,11]
            map_cls, map_off = self.map_decoder(map_agg)        # [150,4], [150,40]

            # anchor + offset
            det_box = self.det_anchors_full + det_off            # [900, 11]

            # [P0-#4] sin/cos 자리 unit norm 정규화
            sin_cos = det_box[..., 6:8]
            norm = sin_cos.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            sin_cos = sin_cos / norm
            det_box = torch.cat([
                det_box[..., :6],
                sin_cos,
                det_box[..., 8:],
            ], dim=-1)                                           # [900, 11]

            map_line = self.map_anchors + map_off                # [150, 20, 2]

            batch_det_classes.append(det_cls)
            batch_det_boxes.append(det_box)
            batch_map_classes.append(map_cls)
            batch_map_lines.append(map_line)

        return (torch.stack(batch_det_classes),
                torch.stack(batch_det_boxes),
                torch.stack(batch_map_classes),
                torch.stack(batch_map_lines))


# ===========================================================
# Val 헬퍼 — Recall@thr & 검증 루프
# ===========================================================
@torch.no_grad()
def compute_recall_at(det_classes, det_boxes, gt_boxes, threshold=2.0, bg_idx=2):
    """
    1 sample의 Recall@threshold(m):
    foreground(argmax != bg) 예측 중 각 GT마다 가장 가까운 xy 거리가 threshold 미만인 비율.
    GT 없으면 None.
    """
    if gt_boxes.numel() == 0:
        return None
    preds = det_classes.argmax(dim=-1)              # [900]
    fg_mask = preds != bg_idx                       # [900]
    if fg_mask.sum() == 0:
        return 0.0
    fg_xy = det_boxes[fg_mask, :2]                  # [N_fg, 2]
    gt_xy = gt_boxes[:, :2]                         # [N_gt, 2]
    dmin = torch.cdist(gt_xy, fg_xy).min(dim=1).values
    return (dmin < threshold).float().mean().item()


@torch.no_grad()
def validate(model, loader, criterion, device, compute_metric=False, recall_thr=2.0):
    """
    val set 전체에 대해 평균 detection loss를 계산하고,
    compute_metric=True면 Recall@recall_thr도 같이 평균.
    반환: (val_loss, val_recall_or_None)
    """
    model.eval()
    loss_sum = 0.0
    n_batches = 0
    recalls = []

    for batch in loader:
        images     = batch['images'].to(device)
        intrinsics = batch['intrinsics'].to(device)
        extrinsics = batch['extrinsics'].to(device)
        n = images.shape[0]

        det_classes_b, det_boxes_b, _, _ = model(images, intrinsics, extrinsics)

        sample_loss = 0.0
        for i in range(n):
            gt_boxes   = batch['dynamic_gt_boxes'][i].to(device)
            gt_classes = batch['dynamic_gt_labels'][i].to(device)
            det_loss, _, _ = criterion(
                det_classes_b[i], det_boxes_b[i], gt_classes, gt_boxes
            )
            sample_loss += det_loss.item()

            if compute_metric:
                r = compute_recall_at(
                    det_classes_b[i], det_boxes_b[i], gt_boxes, threshold=recall_thr
                )
                if r is not None:
                    recalls.append(r)

        loss_sum += sample_loss / n
        n_batches += 1

    avg_loss = loss_sum / max(n_batches, 1)
    avg_recall = (sum(recalls) / len(recalls)) if recalls else None
    return avg_loss, avg_recall


# ===========================================================
# 학습 루프 (동적 객체만 학습 — 정적 맵 head는 forward에서만 통과)
# ===========================================================
if __name__ == "__main__":
    # ─── Config ────────────────────────────────────────
    # val로 뺄 시나리오 이름. None이면 알파벳 정렬 마지막 1개를 자동 사용.
    # ⚠️ make_kmeans.py 의 --val-scenarios와 반드시 일치시킬 것.
    VAL_SCENARIOS = None

    NUM_EPOCHS           = 100
    BATCH_SIZE           = 8
    EARLY_STOP_PATIENCE  = 10
    EARLY_STOP_MIN_DELTA = 1e-4
    METRIC_EVERY         = 5      # N epoch마다 val Recall@thr 추가 계산
    RECALL_THR           = 2.0    # Recall@2m
    # ───────────────────────────────────────────────────

    print("SparseDrive 인지 모듈 학습 시작! [동적 객체 전용]")
    print(f"   - val 시나리오: {'auto(last)' if VAL_SCENARIOS is None else VAL_SCENARIOS}")
    print(f"   - best 기준   : val loss")
    print(f"   - early stop  : patience={EARLY_STOP_PATIENCE}, min_delta={EARLY_STOP_MIN_DELTA}")
    print(f"   - metric      : Recall@{RECALL_THR}m, 매 {METRIC_EVERY} epoch\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[디바이스] {device}\n")

    model = AutoNavModel().to(device)
    start_epoch = 0
    print("[학습] 처음부터 학습\n")

    train_ds = MoraiDataset(dataset_root='./dataset', split='train', val_scenarios=VAL_SCENARIOS)
    val_ds   = MoraiDataset(dataset_root='./dataset', split='val',   val_scenarios=VAL_SCENARIOS)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=morai_collate_fn, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=morai_collate_fn, num_workers=2)

    # num_classes=2: vehicle, pedestrian (bg_class=2)
    det_criterion = CustomLoss(num_classes=2).to(device)

    backbone_params = list(model.backbone.parameters())
    backbone_ids    = set(id(p) for p in backbone_params)
    other_params    = [p for p in model.parameters() if id(p) not in backbone_ids]

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 4e-5},
        {'params': other_params,    'lr': 4e-4},
    ], weight_decay=1e-3)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    best_val_loss     = float('inf')
    epochs_no_improve = 0

    for epoch in range(start_epoch, NUM_EPOCHS):
        # ─── Train ───────────────────────────────────────
        model.train()
        print(f"\n========== [Epoch {epoch+1}/{NUM_EPOCHS}] ==========")
        train_loss_sum = 0.0

        for step, batch in enumerate(train_loader):
            images     = batch['images'].to(device)
            intrinsics = batch['intrinsics'].to(device)
            extrinsics = batch['extrinsics'].to(device)
            n = images.shape[0]

            det_classes_b, det_boxes_b, _, _ = model(
                images, intrinsics, extrinsics
            )

            batch_loss = 0.0
            for i in range(n):
                gt_boxes   = batch['dynamic_gt_boxes'][i].to(device)
                gt_classes = batch['dynamic_gt_labels'][i].to(device)
                det_loss, _, _ = det_criterion(
                    det_classes_b[i], det_boxes_b[i], gt_classes, gt_boxes
                )
                batch_loss = batch_loss + det_loss
            batch_loss = batch_loss / n

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += batch_loss.item()

            if step % 10 == 0:
                print(f"  [train] Step {step:03d} | Det Loss: {batch_loss.item():.4f}")

        scheduler.step()
        train_loss = train_loss_sum / len(train_loader)

        # ─── Val ─────────────────────────────────────────
        compute_metric = ((epoch + 1) % METRIC_EVERY == 0)
        val_loss, val_recall = validate(
            model, val_loader, det_criterion, device,
            compute_metric=compute_metric, recall_thr=RECALL_THR,
        )

        lr = scheduler.get_last_lr()[0]
        msg = (f"\n📊 Epoch {epoch+1} | "
               f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr:.2e}")
        if val_recall is not None:
            msg += f"\n   └─ Val Recall@{RECALL_THR}m: {val_recall:.4f}"
        print(msg)

        # ─── Best save (val loss 기준) ───────────────────
        if val_loss < best_val_loss - EARLY_STOP_MIN_DELTA:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(f"   💾 Best 저장! Val Loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"   ⏳ Val 개선 없음 ({epochs_no_improve}/{EARLY_STOP_PATIENCE})")

        # ─── 정기 체크포인트 ──────────────────────────────
        if (epoch + 1) % 10 == 0:
            ckpt_path = f"checkpoint_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"   📌 체크포인트 저장: {ckpt_path}")

        # ─── Early stop (val 기준) ───────────────────────
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n⚠️  Early Stopping! "
                  f"Val Loss가 {EARLY_STOP_PATIENCE} epoch 동안 개선 없음.")
            break

    print("\n🎉 학습 완료!")
    torch.save(model.state_dict(), "morai_autonav_weights.pth")
    print(f"💾 최종 모델 저장: morai_autonav_weights.pth")
    print(f"📊 Best Val Loss: {best_val_loss:.4f}")
