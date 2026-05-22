import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from morai_dataset import MoraiDataset, morai_collate_fn
from resnet_fpn import ResNet50_FPN, Bottleneck
from anchor_generator import NUM_ANCHORS, generate_anchors_full
from decoder import FFNDecoder
from loss_calculator import CustomLoss
from make_kmeans import (
    DEFAULT_FULL_OUT,
    DEFAULT_K,
    DEFAULT_META_OUT,
    DEFAULT_XY_OUT,
    ensure_kmeans_files,
)
from torch.utils.data import DataLoader


# ===========================================================
# [P0-#1] 멀티스케일 샘플링 — per-point visible mask 반환
# ===========================================================
def sample_from_multiscale(features_list, grid_2d, valid_mask, N, level_logits=None):
    """
    features_list : list of [1, 256, H, W]  (각 스케일)
    grid_2d       : [1, 1, N, 2]            (normalized -1~1)
    valid_mask    : [N] bool                (depth > 0.1)
    N             : 점 개수
    반환          : [N, 256] sampled feature (invalid는 0)
    """
    combined = torch.zeros(N, 256, device=features_list[0].device)
    if level_logits is None:
        level_weights = torch.full(
            (len(features_list),),
            1.0 / len(features_list),
            device=features_list[0].device,
            dtype=features_list[0].dtype,
        )
    else:
        level_weights = torch.softmax(level_logits[:len(features_list)], dim=0)

    for level_idx, feat in enumerate(features_list):
        sampled = F.grid_sample(feat, grid_2d, align_corners=False)
        sampled = sampled.view(256, N).T            # [N, 256]
        sampled = sampled * valid_mask.float().unsqueeze(1)
        combined = combined + level_weights[level_idx] * sampled
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


class CameraFeatureFusion(nn.Module):
    """
    Lightweight learned camera fusion for the 3 front cameras.
    Invalid cameras receive zero weight, so anchors outside every camera stay zero.
    """
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, camera_features, camera_visible, query_feature):
        """
        camera_features : [C, N, D]
        camera_visible  : [C, N] float/bool
        query_feature   : [N, D]
        """
        C, N, _ = camera_features.shape
        query = query_feature.unsqueeze(0).expand(C, N, -1)
        logits = self.score_net(torch.cat([camera_features, query], dim=-1)).squeeze(-1)

        visible = camera_visible.float()
        logits = logits.masked_fill(visible <= 0, -1e4)
        weights = torch.softmax(logits, dim=0) * visible
        weights = weights / weights.sum(dim=0, keepdim=True).clamp(min=1e-6)
        return (camera_features * weights.unsqueeze(-1)).sum(dim=0)


# ===========================================================
# 모델
# ===========================================================
class AutoNavModel(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=2):
        super().__init__()
        anchors_full = generate_anchors_full()
        if anchors_full.shape != (NUM_ANCHORS, 11):
            raise ValueError(f"anchors_full shape 이상: {anchors_full.shape}")

        self.backbone = ResNet50_FPN(Bottleneck)
        self.det_decoder = FFNDecoder(hidden_dim=hidden_dim, num_classes=num_classes + 1)
        self.camera_fusion = CameraFeatureFusion(hidden_dim=hidden_dim)
        self.instance_feature = nn.Parameter(torch.empty(NUM_ANCHORS, hidden_dim))
        self.level_logits = nn.Parameter(torch.zeros(4))
        self.register_buffer('det_anchors_full', anchors_full)
        nn.init.normal_(self.instance_feature, std=0.01)

    def forward(self, images, intrinsics, extrinsics):
        """
        images     : [B, 3, 3, H, W]
        intrinsics : [B, 3, 3, 3] resized-input coordinate system
        extrinsics : [B, 3, 4, 4]
        반환       : det_classes [B,900,3], det_boxes [B,900,11]
        """
        device = images.device
        B = images.shape[0]
        N_CAMS = images.shape[1]
        image_h = images.shape[-2]
        image_w = images.shape[-1]

        N_det = self.det_anchors_full.shape[0]    # 900
        N_kp  = 5                                  # [P0-#2] 키포인트 개수

        # 배치별 출력 누적
        batch_det_classes = []
        batch_det_boxes   = []

        for b in range(B):
            cam_imgs = images[b]                            # [3, 3, H, W]
            all_features = self.backbone(cam_imgs)          # list of [3, C, H, W]

            # [P0-#2] 동적 anchor → 5개 키포인트
            det_keypoints = generate_5_keypoints(self.det_anchors_full)   # [900, 5, 3]
            det_kp_flat = det_keypoints.view(N_det * N_kp, 3)             # [4500, 3]
            det_kp_homo = torch.cat(
                [det_kp_flat, torch.ones(N_det * N_kp, 1, device=device)],
                dim=-1
            )                                                              # [4500, 4]

            det_camera_features = []
            det_camera_visible = []

            for cam_idx in range(N_CAMS):
                cam_img = images[b, cam_idx]
                if cam_img.abs().sum() < 1e-6:
                    continue

                features_list = [f[cam_idx:cam_idx+1] for f in all_features]
                E = extrinsics[b, cam_idx]   # [4, 4]
                K = intrinsics[b, cam_idx]   # [3, 3]
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]

                # ─── Detection 키포인트 투영 ──────────────────────
                det_pts = (E @ det_kp_homo.T).T               # [4500, 4]
                det_depth = det_pts[:, 0]
                det_u = fx * (-det_pts[:, 1]) / (det_depth + 1e-6) + cx
                det_v = fy * (-det_pts[:, 2]) / (det_depth + 1e-6) + cy
                det_kp_valid = (
                    (det_depth > 0.1) &
                    (det_u >= 0.0) & (det_u < float(image_w)) &
                    (det_v >= 0.0) & (det_v < float(image_h))
                )                                             # [4500]
                det_u_n = ((det_u + 0.5) / float(image_w)) * 2.0 - 1.0
                det_v_n = ((det_v + 0.5) / float(image_h)) * 2.0 - 1.0
                det_grid = torch.stack([det_u_n, det_v_n], dim=-1).view(1, 1, N_det * N_kp, 2)
                det_sampled_kp = sample_from_multiscale(
                    features_list, det_grid, det_kp_valid, N_det * N_kp, self.level_logits
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

                det_camera_features.append(det_sampled)
                det_camera_visible.append(anchor_visible_in_cam)

            if det_camera_features:
                det_camera_features = torch.stack(det_camera_features, dim=0)  # [C,900,256]
                det_camera_visible = torch.stack(det_camera_visible, dim=0)    # [C,900]
                det_agg = self.camera_fusion(
                    det_camera_features,
                    det_camera_visible,
                    self.instance_feature,
                )
            else:
                det_agg = torch.zeros(N_det, 256, device=device)

            # FFN 디코더
            det_feat = det_agg + self.instance_feature
            det_cls, det_off = self.det_decoder(det_feat)       # [900,3], [900,11]

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

            batch_det_classes.append(det_cls)
            batch_det_boxes.append(det_box)

        return torch.stack(batch_det_classes), torch.stack(batch_det_boxes)


# ===========================================================
# Val 헬퍼 — score/class-aware Precision & Recall
# ===========================================================
@torch.no_grad()
def bev_nms_axis_aligned(boxes, scores, labels, iou_thresh=0.3):
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    keep_all = []
    for cls_id in labels.unique():
        cls_idx = torch.where(labels == cls_id)[0]
        cls_scores = scores[cls_idx]
        order = torch.argsort(cls_scores, descending=True)

        while order.numel() > 0:
            current_local = order[0]
            current_idx = cls_idx[current_local]
            keep_all.append(current_idx)
            if order.numel() == 1:
                break

            rest_local = order[1:]
            rest_idx = cls_idx[rest_local]

            cur = boxes[current_idx]
            rest = boxes[rest_idx]

            cur_w = torch.exp(cur[3])
            cur_l = torch.exp(cur[4])
            rest_w = torch.exp(rest[:, 3])
            rest_l = torch.exp(rest[:, 4])

            inter_x = (
                torch.minimum(cur[0] + cur_l * 0.5, rest[:, 0] + rest_l * 0.5) -
                torch.maximum(cur[0] - cur_l * 0.5, rest[:, 0] - rest_l * 0.5)
            ).clamp(min=0)
            inter_y = (
                torch.minimum(cur[1] + cur_w * 0.5, rest[:, 1] + rest_w * 0.5) -
                torch.maximum(cur[1] - cur_w * 0.5, rest[:, 1] - rest_w * 0.5)
            ).clamp(min=0)
            inter = inter_x * inter_y
            union = cur_w * cur_l + rest_w * rest_l - inter + 1e-6
            order = rest_local[(inter / union) < iou_thresh]

    if not keep_all:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    return torch.stack(keep_all)


@torch.no_grad()
def decode_detections(det_classes, det_boxes, score_thresh=0.25, bg_idx=2, nms_iou=0.3):
    probs = det_classes.softmax(dim=-1)
    scores, labels_all = probs.max(dim=-1)
    keep = (labels_all != bg_idx) & (scores >= score_thresh)
    if keep.sum() == 0:
        empty_long = torch.zeros(0, dtype=torch.long, device=det_boxes.device)
        empty_float = torch.zeros(0, dtype=det_boxes.dtype, device=det_boxes.device)
        return det_boxes[:0], empty_long, empty_float

    boxes = det_boxes[keep]
    labels = labels_all[keep]
    scores = scores[keep]
    keep_nms = bev_nms_axis_aligned(boxes, scores, labels, iou_thresh=nms_iou)
    return boxes[keep_nms], labels[keep_nms], scores[keep_nms]


@torch.no_grad()
def compute_detection_counts(
    det_classes,
    det_boxes,
    gt_classes,
    gt_boxes,
    distance_thr=2.0,
    score_thresh=0.25,
):
    pred_boxes, pred_labels, pred_scores = decode_detections(
        det_classes, det_boxes, score_thresh=score_thresh
    )

    if gt_boxes.numel() == 0:
        return 0, int(pred_boxes.shape[0]), 0

    gt_classes = gt_classes.long().view(-1)
    matched_gt = torch.zeros(gt_boxes.shape[0], dtype=torch.bool, device=gt_boxes.device)
    tp = 0
    fp = 0

    order = torch.argsort(pred_scores, descending=True)
    for pred_idx in order:
        same_cls = gt_classes == pred_labels[pred_idx]
        candidates = torch.where(same_cls & (~matched_gt))[0]
        if candidates.numel() == 0:
            fp += 1
            continue

        distances = torch.norm(gt_boxes[candidates, :2] - pred_boxes[pred_idx, :2], dim=-1)
        best_dist, best_local = distances.min(dim=0)
        if best_dist <= distance_thr:
            matched_gt[candidates[best_local]] = True
            tp += 1
        else:
            fp += 1

    fn = int((~matched_gt).sum().item())
    return tp, fp, fn


@torch.no_grad()
def validate(model, loader, criterion, device, compute_metric=False, recall_thr=2.0):
    """
    val set 전체에 대해 평균 detection loss를 계산하고,
    compute_metric=True면 score/class-aware Precision/Recall도 계산.
    반환: (val_loss, metrics_or_None)
    """
    model.eval()
    loss_sum = 0.0
    n_batches = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for batch in loader:
        images     = batch['images'].to(device)
        intrinsics = batch['intrinsics'].to(device)
        extrinsics = batch['extrinsics'].to(device)
        n = images.shape[0]

        det_classes_b, det_boxes_b = model(images, intrinsics, extrinsics)

        sample_loss = 0.0
        for i in range(n):
            gt_boxes   = batch['dynamic_gt_boxes'][i].to(device)
            gt_classes = batch['dynamic_gt_labels'][i].to(device)
            det_loss, _, _ = criterion(
                det_classes_b[i], det_boxes_b[i], gt_classes, gt_boxes
            )
            sample_loss += det_loss.item()

            if compute_metric:
                tp, fp, fn = compute_detection_counts(
                    det_classes_b[i],
                    det_boxes_b[i],
                    gt_classes,
                    gt_boxes,
                    distance_thr=recall_thr,
                )
                total_tp += tp
                total_fp += fp
                total_fn += fn

        loss_sum += sample_loss / n
        n_batches += 1

    avg_loss = loss_sum / max(n_batches, 1)
    if not compute_metric:
        return avg_loss, None

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    return avg_loss, {'precision': precision, 'recall': recall}


# ===========================================================
# 학습 루프 (3전방 카메라 동적 객체 detection 전용)
# ===========================================================
if __name__ == "__main__":
    # ─── Config ────────────────────────────────────────
    DATASET_ROOT = './dataset'
    # val로 뺄 시나리오 이름. None이면 알파벳 정렬 마지막 5개를 자동 사용.
    # ⚠️ make_kmeans.py 의 --val-scenarios와 반드시 일치시킬 것.
    VAL_SCENARIOS = None

    NUM_EPOCHS           = 100
    BATCH_SIZE           = 8
    EARLY_STOP_PATIENCE  = 10
    EARLY_STOP_MIN_DELTA = 1e-4
    METRIC_EVERY         = 5      # N epoch마다 val Precision/Recall 추가 계산
    RECALL_THR           = 2.0    # distance match threshold
    KMEANS_K             = DEFAULT_K
    FORCE_REMAKE_KMEANS  = True
    # ───────────────────────────────────────────────────

    print("SparseDrive-style 3-camera detection 학습 시작! [vehicle + pedestrian]")
    print(f"   - val 시나리오: {'auto(last 5)' if VAL_SCENARIOS is None else VAL_SCENARIOS}")
    print(f"   - kmeans anchor: train split only, K={KMEANS_K}, always remake")
    print(f"   - best 기준   : val loss")
    print(f"   - early stop  : patience={EARLY_STOP_PATIENCE}, min_delta={EARLY_STOP_MIN_DELTA}")
    print(f"   - metric      : Precision/Recall@{RECALL_THR}m, 매 {METRIC_EVERY} epoch\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[디바이스] {device}\n")

    ensure_kmeans_files(
        dataset_root=DATASET_ROOT,
        val_scenarios=VAL_SCENARIOS,
        k=KMEANS_K,
        xy_out=DEFAULT_XY_OUT,
        full_out=DEFAULT_FULL_OUT,
        meta_out=DEFAULT_META_OUT,
        force=FORCE_REMAKE_KMEANS,
    )

    model = AutoNavModel().to(device)
    start_epoch = 0
    print("[학습] 처음부터 학습\n")

    train_ds = MoraiDataset(dataset_root=DATASET_ROOT, split='train', val_scenarios=VAL_SCENARIOS)
    val_ds   = MoraiDataset(dataset_root=DATASET_ROOT, split='val',   val_scenarios=VAL_SCENARIOS)

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

            det_classes_b, det_boxes_b = model(images, intrinsics, extrinsics)

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
        val_loss, val_metrics = validate(
            model, val_loader, det_criterion, device,
            compute_metric=compute_metric, recall_thr=RECALL_THR,
        )

        lr = scheduler.get_last_lr()[0]
        msg = (f"\n📊 Epoch {epoch+1} | "
               f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr:.2e}")
        if val_metrics is not None:
            msg += (
                f"\n   └─ Val P/R@{RECALL_THR}m: "
                f"{val_metrics['precision']:.4f}/{val_metrics['recall']:.4f}"
            )
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
