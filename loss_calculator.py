import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

BOX_SCALE = [50., 50., 3.,
             5.,  5., 3.,
             1.,  1.,
             15., 15., 3.]   # velocity는 m/s (labels_3d_v2 기준; 도심 ~15m/s 스케일)

# SparseDrive 방식: Hungarian matching cost에는 velocity를 넣지 않고(8채널),
# regression loss에는 vx,vy까지 포함(10채널). vz(ch10)는 GT가 z 지터 유래라 제외.
MATCH_CHANNELS = 8   # x,y,z,ln_w,ln_l,ln_h,sin_yaw,cos_yaw
REG_CHANNELS = 10    # + vx,vy


class FocalLoss(nn.Module):
    """RetinaNet(Lin et al.) sigmoid focal loss.

    각 클래스 채널을 독립적인 binary 분류로 보고 sigmoid focal을 적용한다.
    배경 클래스는 별도 채널이 없으며, 모든 채널이 0인 anchor가 곧 배경이다.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_logits, target, num_fg=None):
        # pred_logits: [N, num_classes], target: [N, num_classes] multi-label one-hot (float)
        p = torch.sigmoid(pred_logits)
        pt = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        ce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce
        if num_fg is not None:
            return focal_loss.sum() / num_fg.clamp(min=1).float()
        return focal_loss.sum()


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1.0, cost_bbox=5.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox

    @torch.no_grad()
    def forward(self, pred_classes, pred_boxes, gt_classes, gt_boxes):
        """
        pred_classes : [900, num_classes]
        pred_boxes   : [900, 11]
        gt_classes   : [N]
        gt_boxes     : [N, 11]
        """
        device = pred_classes.device

        if gt_boxes is None or gt_classes is None:
            empty = torch.zeros(0, dtype=torch.int64, device=device)
            return empty, empty

        gt_boxes = gt_boxes.to(device=device, dtype=pred_boxes.dtype)

        if not torch.is_tensor(gt_classes):
            gt_classes = torch.as_tensor(gt_classes, device=device)
        else:
            gt_classes = gt_classes.to(device=device)

        if gt_classes.ndim > 1:
            gt_classes = gt_classes.squeeze(-1)

        gt_classes = gt_classes.long().view(-1)

        if gt_boxes.shape[0] == 0 or gt_classes.shape[0] == 0:
            empty = torch.zeros(0, dtype=torch.int64, device=device)
            return empty, empty

        scale = torch.tensor(BOX_SCALE, device=device, dtype=pred_boxes.dtype)

        out_prob = pred_classes.sigmoid()
        gt_classes = gt_classes.clamp(min=0, max=pred_classes.shape[-1] - 1)
        cost_class = -out_prob[:, gt_classes]

        pred_norm = pred_boxes[:, :MATCH_CHANNELS] / scale[:MATCH_CHANNELS]
        gt_norm = gt_boxes[:, :MATCH_CHANNELS] / scale[:MATCH_CHANNELS]
        cost_bbox = torch.cdist(pred_norm, gt_norm, p=1)

        C = self.cost_class * cost_class + self.cost_bbox * cost_bbox
        # NaN/Inf 방어: 예측 발산으로 cost 에 비유한값이 생기면 linear_sum_assignment 가
        # 'matrix contains invalid numeric entries' 로 죽는다. 유한값으로 치환해 crash 만 막고,
        # 실제 그 배치의 스킵은 상위 train loop 의 loss 유한성 검사(torch.isfinite)가 담당한다.
        if not torch.isfinite(C).all():
            C = torch.nan_to_num(C, nan=1e6, posinf=1e6, neginf=-1e6)
        C = C.detach().cpu().numpy()

        pred_indices, gt_indices = linear_sum_assignment(C)
        return (
            torch.as_tensor(pred_indices, dtype=torch.int64, device=device),
            torch.as_tensor(gt_indices, dtype=torch.int64, device=device),
        )


class MapHungarianMatcher(nn.Module):
    def __init__(self, cost_class=2.0, cost_line=5.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_line = cost_line

    @torch.no_grad()
    def forward(self, pred_classes, pred_lines, gt_classes, gt_lines, polyline_scale=60.0):
        """
        pred_classes : [100, num_classes]
        pred_lines   : [100, 20, 2]
        gt_classes   : [N]
        gt_lines     : [N, 20, 2]
        """
        device = pred_classes.device

        if gt_classes is None or gt_lines is None:
            empty = torch.zeros(0, dtype=torch.int64, device=device)
            return empty, empty

        gt_lines = gt_lines.to(device=device, dtype=pred_lines.dtype)

        if not torch.is_tensor(gt_classes):
            gt_classes = torch.as_tensor(gt_classes, device=device)
        else:
            gt_classes = gt_classes.to(device=device)

        if gt_classes.ndim > 1:
            gt_classes = gt_classes.squeeze(-1)

        gt_classes = gt_classes.long().view(-1)

        if gt_classes.shape[0] == 0 or gt_lines.shape[0] == 0:
            empty = torch.zeros(0, dtype=torch.int64, device=device)
            return empty, empty

        num_queries = pred_classes.shape[0]

        out_prob = pred_classes.softmax(-1)
        gt_classes = gt_classes.clamp(min=0, max=pred_classes.shape[-1] - 1)
        cost_class = -out_prob[:, gt_classes]

        pred_lines_flat = (pred_lines / polyline_scale).reshape(num_queries, -1)
        gt_lines_flat = (gt_lines / polyline_scale).reshape(gt_classes.shape[0], -1)
        cost_line = torch.cdist(pred_lines_flat, gt_lines_flat, p=1)

        C = self.cost_class * cost_class + self.cost_line * cost_line
        C = C.detach().cpu().numpy()

        pred_indices, gt_indices = linear_sum_assignment(C)
        return (
            torch.as_tensor(pred_indices, dtype=torch.int64, device=device),
            torch.as_tensor(gt_indices, dtype=torch.int64, device=device),
        )


class CustomLoss(nn.Module):
    def __init__(
        self,
        num_classes=1,
        bg_weight=0.1,
        quality_weight=0.2,
        bg_quality_weight=0.0,
        quality_distance_scale=4.0,
        yawness_weight=0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.bg_weight = bg_weight
        self.quality_weight = quality_weight
        self.bg_quality_weight = bg_quality_weight
        self.quality_distance_scale = quality_distance_scale
        self.yawness_weight = yawness_weight

    def quality_loss(self, pred_quality, pred_boxes, gt_boxes, pred_idx, gt_idx):
        if pred_quality is None:
            return pred_boxes.new_tensor(0.0)

        if pred_quality.ndim == 1:
            pred_quality = pred_quality.unsqueeze(-1)

        centerness = pred_quality[:, 0]
        cns_target = torch.zeros_like(centerness)
        cns_weight = torch.full_like(centerness, self.bg_quality_weight)

        has_yawness = pred_quality.shape[-1] > 1
        if has_yawness:
            yawness = pred_quality[:, 1]
            yns_target = torch.zeros_like(yawness)
            yns_weight = torch.full_like(yawness, self.bg_quality_weight)

        if len(pred_idx) > 0:
            distance = torch.norm(
                pred_boxes[pred_idx, :2].detach() - gt_boxes[gt_idx, :2].detach(),
                dim=-1,
            )
            cns = torch.exp(-distance / self.quality_distance_scale).clamp(0.0, 1.0)
            cns_target[pred_idx] = cns.to(dtype=cns_target.dtype)
            cns_weight[pred_idx] = 1.0

            if has_yawness:
                yaw_cos = F.cosine_similarity(
                    pred_boxes[pred_idx, 6:8].detach(),
                    gt_boxes[gt_idx, 6:8].detach(),
                    dim=-1,
                )
                yns_target[pred_idx] = (yaw_cos > 0).to(dtype=yns_target.dtype)
                yns_weight[pred_idx] = 1.0

        cns_loss = F.binary_cross_entropy_with_logits(
            centerness,
            cns_target,
            weight=cns_weight,
            reduction='sum',
        )
        cns_loss = cns_loss / cns_weight.sum().clamp(min=1.0)
        if not has_yawness:
            return cns_loss

        yns_loss = F.binary_cross_entropy_with_logits(
            yawness,
            yns_target,
            weight=yns_weight,
            reduction='sum',
        )
        yns_loss = yns_loss / yns_weight.sum().clamp(min=1.0)
        return cns_loss + self.yawness_weight * yns_loss

    def forward(self, pred_classes, pred_boxes, gt_classes, gt_boxes, pred_quality=None,
                velocity_valid=None):
        """
        pred_classes: [900, 2]
        pred_boxes:   [900, 11]
        pred_quality: [900, 2] optional; [:,0]=centerness, [:,1]=yawness
        gt_classes:   [N]
        gt_boxes:     [N, 11]
        velocity_valid: [N] optional float mask (1=신뢰, 0=불확실). vx,vy 채널(ch8,9)에만
                        element-wise 로 적용. None 이면 모든 채널 유지(기존 동작과 동일).
                        matcher/cost 에는 영향 없음(위치·치수·yaw 8채널만 매칭).
        """
        device = pred_classes.device
        num_anchors = pred_classes.shape[0]

        gt_boxes = gt_boxes.to(device=device, dtype=pred_boxes.dtype)

        if not torch.is_tensor(gt_classes):
            gt_classes = torch.as_tensor(gt_classes, device=device)
        else:
            gt_classes = gt_classes.to(device=device)

        if gt_classes.ndim > 1:
            gt_classes = gt_classes.squeeze(-1)

        gt_classes = gt_classes.long().view(-1)

        if gt_boxes.shape[0] == 0 or gt_classes.shape[0] == 0:
            # GT가 없으면 모든 anchor가 배경 → 전 채널 0인 one-hot target.
            target = torch.zeros(
                (num_anchors, self.num_classes),
                dtype=pred_classes.dtype,
                device=device
            )
            num_fg = torch.tensor(0, device=device)
            loss_class = self.focal_loss(pred_classes, target, num_fg=num_fg) * self.bg_weight
            loss_quality = self.quality_loss(
                pred_quality,
                pred_boxes,
                gt_boxes,
                torch.zeros(0, dtype=torch.long, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
            ) * self.bg_weight
            zero = torch.tensor(0.0, device=device)
            return loss_class + self.quality_weight * loss_quality, loss_class, zero, loss_quality

        pred_idx, gt_idx = self.matcher(pred_classes, pred_boxes, gt_classes, gt_boxes)

        # sigmoid focal: [num_anchors, num_classes] multi-label one-hot.
        # 매칭 안 된 anchor는 전 채널 0(=배경), 매칭된 anchor만 해당 클래스 채널을 1.0으로 set.
        target = torch.zeros(
            (num_anchors, self.num_classes),
            dtype=pred_classes.dtype,
            device=device
        )
        target[pred_idx, gt_classes[gt_idx]] = 1.0

        num_fg = torch.tensor(len(pred_idx), device=device)
        loss_class = self.focal_loss(pred_classes, target, num_fg=num_fg)

        if len(pred_idx) == 0:
            loss_bbox = torch.tensor(0.0, device=device)
        else:
            scale = torch.tensor(BOX_SCALE, device=device, dtype=pred_boxes.dtype)
            diff = (
                pred_boxes[pred_idx, :REG_CHANNELS] / scale[:REG_CHANNELS]
                - gt_boxes[gt_idx, :REG_CHANNELS] / scale[:REG_CHANNELS]
            ).abs()
            # element-wise L1. x/y/z/dims/yaw(ch0-7)는 모든 matched box에서 유지,
            # vx/vy(ch8,9=MATCH_CHANNELS:REG_CHANNELS)만 velocity_valid로 mask.
            # denominator 는 기존 F.l1_loss(mean)와 같은 전체 element 수를 유지한다.
            # weight.sum()으로 나누면 invalid velocity 비율이 높을수록 x/y/dim/yaw의
            # gradient까지 최대 REG_CHANNELS/MATCH_CHANNELS 배 증폭된다. 여기서는
            # "vx/vy 감독만 제거하고 나머지 채널은 현행 유지"가 목적이므로, 마스킹된
            # velocity 항만 0이 되고 다른 채널의 절대 gradient는 바뀌지 않아야 한다.
            weight = torch.ones_like(diff)
            if velocity_valid is not None:
                vv = velocity_valid.to(device=device, dtype=diff.dtype).view(-1)
                vv = vv[gt_idx].clamp(0.0, 1.0)
                weight[:, MATCH_CHANNELS:REG_CHANNELS] = vv.unsqueeze(1)
            loss_bbox = (diff * weight).sum() / max(diff.numel(), 1)

        loss_quality = self.quality_loss(pred_quality, pred_boxes, gt_boxes, pred_idx, gt_idx)

        total_loss = 2.0 * loss_class + 2.0 * loss_bbox + self.quality_weight * loss_quality
        return total_loss, loss_class, loss_bbox, loss_quality


# ═══════════════════════════════════════════════════════════════════
# Phase 2: 공식 SparseDrive Stage-1 parity loss 계약
#
# 합의 계약 (2026-07 Phase 1/2 분석):
#   matching      : 전체 900 query ↔ GT
#                   cls cost = focal pos-neg × 2.0 (공식 target.py._cls_cost)
#                   box cost = |Δ|·[2,2,2,.5,.5,.5,0,0,0,0]×0.25 (metric 단위,
#                              yaw·velocity 명시적 제외, BOX_SCALE 미사용)
#   num_pos       : matched 수 — cls threshold 적용 **전** (공식 head 435행)
#   cls loss      : 모든 query, sigmoid focal α.25 γ2 × 2.0, ÷ batch num_pos
#   reg/qual mask : matched ∧ sigmoid(cls).max > 0.05 (공식 cls_threshold_to_reg)
#   reg loss      : L1·[2,2,2,1,1,1,1,1,1,1]×0.25, ÷ batch num_pos, vz 제외,
#                   velocity_valid 는 vx/vy 채널에만 적용 (의도적 MORAI 이탈)
#   cns loss      : BCE(cns_logit, exp(−‖Δxyz‖₂)) — detach 없음(공식과 동일;
#                   gradient 가 target 경유로 box 에도 흐름), ÷ batch num_pos
#   yns loss      : GaussianFocal(sigmoid(yns), cos_sim>0), ÷ batch num_pos
#   decoder 결합  : 6개 동일 가중 합산 (batch num_pos 정규화는 train.py 집계 담당)
#
# 이 클래스들은 frame 단위 **비정규화 합**을 반환한다. 프레임별 GT 수로 나누면
# per-object gradient 가 GT 밀도에 반비례하는 왜곡이 생기므로(기존 계약의 문제),
# 정규화는 반드시 배치 전체 num_pos 로 한 번만 수행해야 한다
# → train.py compute_parity_detection_loss 참조.
# ═══════════════════════════════════════════════════════════════════

PARITY_MATCH_WEIGHTS = [2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]  # ×0.25
PARITY_REG_WEIGHTS = [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]    # ×0.25
PARITY_CLS_WEIGHT = 2.0
PARITY_BOX_WEIGHT = 0.25
PARITY_CLS_THRESHOLD_TO_REG = 0.05


class ParityHungarianMatcher(nn.Module):
    """공식 SparseBox3DTarget.sample 의 cost 계약 재현."""

    def __init__(self, alpha=0.25, gamma=2.0, eps=1e-12):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.register_buffer(
            "match_w", torch.tensor(PARITY_MATCH_WEIGHTS) * PARITY_BOX_WEIGHT)

    @torch.no_grad()
    def forward(self, pred_classes, pred_boxes, gt_classes, gt_boxes):
        device = pred_classes.device
        if gt_boxes is None or gt_boxes.shape[0] == 0:
            empty = torch.zeros(0, dtype=torch.int64, device=device)
            return empty, empty

        p = pred_classes.sigmoid()                                    # [N,C]
        neg = -(1 - p + self.eps).log() * (1 - self.alpha) * p.pow(self.gamma)
        pos = -(p + self.eps).log() * self.alpha * (1 - p).pow(self.gamma)
        cost_cls = (pos - neg)[:, gt_classes] * PARITY_CLS_WEIGHT     # [N,M]

        w = self.match_w.to(device=device, dtype=pred_boxes.dtype)    # [10]
        diff = (pred_boxes[:, None, :10] - gt_boxes[None, :, :10]).abs()
        cost_box = (diff * w).sum(-1)                                 # [N,M]

        C = cost_cls + cost_box
        if not torch.isfinite(C).all():
            C = torch.nan_to_num(C, nan=1e6, posinf=1e6, neginf=-1e6)
        pred_idx, gt_idx = linear_sum_assignment(C.detach().cpu().numpy())
        return (
            torch.as_tensor(pred_idx, dtype=torch.int64, device=device),
            torch.as_tensor(gt_idx, dtype=torch.int64, device=device),
        )


def gaussian_focal_loss_sum(pred_sigmoid, target, alpha=2.0, gamma=4.0, eps=1e-12):
    """mmdet GaussianFocalLoss (합만 반환, 정규화는 호출부).
    binary target(0/1)에서 (1-t)^γ 항은 neg 에서 1이 된다.
    ⚠️ eps 는 mmdet 과 동일하게 log **안**에 더한다. clamp(max=1-eps) 방식은
    fp32 에서 1-1e-12 가 1.0 으로 반올림되어 saturated sigmoid(=1.0)에서
    log(0)=-inf → NaN 이 된다(실측: BCE 로 학습된 epoch20 yns head)."""
    p = pred_sigmoid
    pos = -(p + eps).log() * (1 - p).pow(alpha) * target
    neg = -(1 - p + eps).log() * p.pow(alpha) * (1 - target).pow(gamma)
    return (pos + neg).sum()


class ParityLoss(nn.Module):
    """공식 Stage-1 detection loss 계약 — frame 단위 비정규화 합을 반환.

    반환 dict:
      cls_sum   : focal 합 × 2.0 (모든 query)
      box_sum   : weighted L1 합 × 0.25 (gate 통과 positive)
      cns_sum   : centerness BCE 합 (gate 통과 positive, detach 없음)
      yns_sum   : yawness GaussianFocal 합 (gate 통과 positive)
      num_pos   : matched GT 수 (gate 적용 전 — 정규화 분모)
      num_gate  : gate(cls>0.05) 통과 positive 수 (진단용)
    """

    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = ParityHungarianMatcher()
        self.register_buffer(
            "reg_w", torch.tensor(PARITY_REG_WEIGHTS) * PARITY_BOX_WEIGHT)

    def forward(self, pred_classes, pred_boxes, gt_classes, gt_boxes,
                pred_quality=None, velocity_valid=None):
        device = pred_classes.device
        zero = pred_classes.new_tensor(0.0)

        gt_boxes = gt_boxes.to(device=device, dtype=pred_boxes.dtype)
        if not torch.is_tensor(gt_classes):
            gt_classes = torch.as_tensor(gt_classes, device=device)
        gt_classes = gt_classes.to(device=device).long().view(-1)

        target = torch.zeros(
            (pred_classes.shape[0], self.num_classes),
            dtype=pred_classes.dtype, device=device)

        if gt_boxes.shape[0] == 0 or gt_classes.shape[0] == 0:
            # 전부 배경: focal neg 합만. num_pos 는 0 (배치 집계에서 분모 합산).
            p = torch.sigmoid(pred_classes)
            ce = F.binary_cross_entropy_with_logits(pred_classes, target, reduction='none')
            focal = 0.75 * p.pow(2.0) * ce      # α=0.25, target=0 → (1-α)·p^γ·CE
            return {"cls_sum": PARITY_CLS_WEIGHT * focal.sum(),
                    "box_sum": zero, "cns_sum": zero, "yns_sum": zero,
                    "num_pos": 0, "num_gate": 0}

        pred_idx, gt_idx = self.matcher(pred_classes, pred_boxes, gt_classes, gt_boxes)
        num_pos = int(len(pred_idx))
        target[pred_idx, gt_classes[gt_idx]] = 1.0

        # cls: sigmoid focal 합 (정규화 없음 — 배치 집계에서 ÷num_pos)
        p = torch.sigmoid(pred_classes)
        pt = p * target + (1 - p) * (1 - target)
        alpha_t = 0.25 * target + 0.75 * (1 - target)
        ce = F.binary_cross_entropy_with_logits(pred_classes, target, reduction='none')
        cls_sum = PARITY_CLS_WEIGHT * (alpha_t * (1 - pt).pow(2.0) * ce).sum()

        # reg/quality gate: matched ∧ cls>0.05 (공식 head 438-442행. 분모는 pre-gate num_pos)
        with torch.no_grad():
            gate = p[pred_idx].max(dim=-1).values > PARITY_CLS_THRESHOLD_TO_REG
        pi, gi = pred_idx[gate], gt_idx[gate]
        num_gate = int(gate.sum())

        if num_gate == 0:
            return {"cls_sum": cls_sum, "box_sum": zero, "cns_sum": zero,
                    "yns_sum": zero, "num_pos": num_pos, "num_gate": 0}

        # box: L1 · reg_w (vz 제외 10채널), velocity_valid 는 vx/vy 에만
        w = self.reg_w.to(device=device, dtype=pred_boxes.dtype).clone()  # [10]
        diff = (pred_boxes[pi, :10] - gt_boxes[gi, :10]).abs()            # [K,10]
        wmat = w.unsqueeze(0).expand_as(diff).clone()
        if velocity_valid is not None:
            vv = velocity_valid.to(device=device, dtype=diff.dtype).view(-1)
            wmat[:, 8:10] = wmat[:, 8:10] * vv[gi].clamp(0.0, 1.0).unsqueeze(1)
        box_sum = (diff * wmat).sum()

        cns_sum = zero
        yns_sum = zero
        if pred_quality is not None:
            if pred_quality.ndim == 1:
                pred_quality = pred_quality.unsqueeze(-1)
            # centerness target: exp(−‖Δxyz‖₂) — 공식과 동일하게 detach 하지 않는다.
            d3d = torch.norm(pred_boxes[pi, :3] - gt_boxes[gi, :3], dim=-1)
            cns_target = torch.exp(-d3d)
            cns_sum = F.binary_cross_entropy_with_logits(
                pred_quality[pi, 0], cns_target, reduction='sum')
            if pred_quality.shape[-1] > 1:
                yns_target = (F.cosine_similarity(
                    pred_boxes[pi, 6:8], gt_boxes[gi, 6:8], dim=-1) > 0).float()
                yns_sum = gaussian_focal_loss_sum(
                    torch.sigmoid(pred_quality[pi, 1]), yns_target)

        return {"cls_sum": cls_sum, "box_sum": box_sum, "cns_sum": cns_sum,
                "yns_sum": yns_sum, "num_pos": num_pos, "num_gate": num_gate}


if __name__ == "__main__":
    print("🚀 Focal Loss + 배경 클래스 Loss 테스트!\n")

    dummy_pred_classes = torch.randn(900, 2)
    dummy_pred_boxes = torch.randn(900, 11)
    dummy_pred_quality = torch.randn(900)
    dummy_gt_classes = torch.randint(0, 1, (5,), dtype=torch.long)
    dummy_gt_boxes = torch.randn(5, 11)

    criterion = CustomLoss(num_classes=1)
    total_loss, cls_loss, box_loss, quality_loss = criterion(
        dummy_pred_classes, dummy_pred_boxes,
        dummy_gt_classes, dummy_gt_boxes, dummy_pred_quality
    )
    print(f"✅ 분류 Loss (Focal) : {cls_loss.item():.4f}")
    print(f"✅ 박스 Loss         : {box_loss.item():.4f}")
    print(f"✅ 품질 Loss         : {quality_loss.item():.4f}")
    print(f"🔥 총합 Loss         : {total_loss.item():.4f}")

    empty_gt_classes = torch.zeros(0, dtype=torch.long)
    empty_gt_boxes = torch.zeros(0, 11)
    total_loss2, _, _, _ = criterion(
        dummy_pred_classes, dummy_pred_boxes,
        empty_gt_classes, empty_gt_boxes, dummy_pred_quality
    )
    print(f"\n✅ 빈 GT (전부 배경) Loss: {total_loss2.item():.4f}")
    print("\n🎉 Focal Loss + 배경 클래스 테스트 통과!")
