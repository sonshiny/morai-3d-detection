import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

# 정규화 스케일 (전역 상수로 관리)
BOX_SCALE = [50., 50., 3.,   # x, y, z
              5.,  5., 3.,   # w, l, h
              1.,  1.,       # sin, cos
             30., 30., 5.]   # vx, vy, vz


# ===========================================================
# Focal Loss (SparseDrive 논문과 동일)
# ===========================================================
class FocalLoss(nn.Module):
    """
    배경 클래스까지 포함하는 Focal Loss.
    쉬운 샘플(이미 잘 맞추는 것)의 Loss를 줄이고,
    어려운 샘플(아직 못 맞추는 것)에 집중하게 만든다.
    
    기존 CrossEntropyLoss와의 차이:
    - CE: 모든 샘플에 동일한 가중치
    - Focal: 잘 맞추는 샘플은 가중치↓, 못 맞추는 샘플은 가중치↑
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha   # 클래스 불균형 보정 (배경이 압도적으로 많으니까)
        self.gamma = gamma   # 쉬운 샘플 억제 강도

    def forward(self, pred_logits, target):
        """
        pred_logits: [N, num_classes+1]  (배경 포함)
        target:      [N]                 (0~2=객체, 3=배경)
        """
        ce_loss = F.cross_entropy(pred_logits, target, reduction='none')
        pt = torch.exp(-ce_loss)  # 맞출 확률
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1.0, cost_bbox=5.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox  = cost_bbox

    @torch.no_grad()
    def forward(self, pred_classes, pred_boxes, gt_classes, gt_boxes):
        """
        pred_classes : [900, num_classes+1]  (배경 포함)
        pred_boxes   : [900, 11]
        gt_classes   : [N]
        gt_boxes     : [N, 11]
        """
        if gt_boxes.shape[0] == 0:
            device = pred_classes.device
            return (torch.zeros(0, dtype=torch.int64, device=device),
                    torch.zeros(0, dtype=torch.int64, device=device))

        device = pred_boxes.device
        scale  = torch.tensor(BOX_SCALE, device=device)

        # 분류 비용: 배경 제외한 객체 클래스만 사용
        out_prob   = pred_classes.softmax(-1)
        cost_class = -out_prob[:, gt_classes]   # GT 클래스에 대한 확률

        # 박스 비용: 정규화된 값으로 계산
        pred_norm  = pred_boxes / scale
        gt_norm    = gt_boxes   / scale
        cost_bbox  = torch.cdist(pred_norm, gt_norm, p=1)

        C = self.cost_class * cost_class + self.cost_bbox * cost_bbox
        C = C.cpu().numpy()

        pred_indices, gt_indices = linear_sum_assignment(C)
        return (torch.as_tensor(pred_indices, dtype=torch.int64),
                torch.as_tensor(gt_indices,   dtype=torch.int64))


class CustomLoss(nn.Module):
    """
    수정된 Loss:
    1. Focal Loss로 교체 (배경 클래스 포함)
    2. 매칭 안 된 앵커를 배경(no-object)으로 학습
    
    이전 코드와의 차이:
    - 이전: 매칭된 5~10개만 Loss → 890개 앵커는 학습 신호 0
    - 지금: 전체 900개 앵커에 Loss → 배경 앵커도 "여기엔 없다"를 학습
    """
    def __init__(self, num_classes=3, bg_weight=0.1):
        super().__init__()
        self.num_classes = num_classes   # 객체 클래스 수 (car, truck, bus)
        self.bg_class    = num_classes   # 배경 클래스 ID = 3
        self.matcher     = HungarianMatcher()
        self.focal_loss  = FocalLoss(alpha=0.25, gamma=2.0)
        self.bg_weight   = bg_weight    # 배경 Loss 가중치 (너무 크면 전부 배경으로 예측)

    def forward(self, pred_classes, pred_boxes, gt_classes, gt_boxes):
        """
        pred_classes: [900, num_classes+1]  ← 디코더 출력이 4로 바뀌어야 함!
        pred_boxes:   [900, 11]
        gt_classes:    [N]   (0=car, 1=truck, 2=bus)
        gt_boxes:      [N, 11]
        """
        device = pred_classes.device
        num_anchors = pred_classes.shape[0]  # 900

        # ── GT가 비어있는 경우: 전부 배경 ──────────────────
        if gt_boxes.shape[0] == 0:
            target = torch.full((num_anchors,), self.bg_class,
                                dtype=torch.long, device=device)
            loss_class = self.focal_loss(pred_classes, target) * self.bg_weight
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return loss_class, loss_class, zero

        # ── Hungarian Matching ────────────────────────────
        pred_idx, gt_idx = self.matcher(pred_classes, pred_boxes,
                                        gt_classes, gt_boxes)
        pred_idx = pred_idx.to(device)
        gt_idx   = gt_idx.to(device)

        # ── 분류 Loss (전체 900개 앵커) ───────────────────
        # 기본: 전부 배경으로 설정
        target = torch.full((num_anchors,), self.bg_class,
                            dtype=torch.long, device=device)
        # 매칭된 앵커만 실제 GT 클래스로 덮어쓰기
        target[pred_idx] = gt_classes[gt_idx]

        loss_class = self.focal_loss(pred_classes, target)

        # ── 박스 Loss (매칭된 앵커만) ─────────────────────
        scale     = torch.tensor(BOX_SCALE, device=device)
        loss_bbox = F.l1_loss(pred_boxes[pred_idx] / scale,
                              gt_boxes[gt_idx]     / scale)

        total_loss = loss_class + 5.0 * loss_bbox
        return total_loss, loss_class, loss_bbox


if __name__ == "__main__":
    print("🚀 Focal Loss + 배경 클래스 Loss 테스트!\n")

    # num_classes+1 = 4 (car, truck, bus, background)
    dummy_pred_classes = torch.randn(900, 4)   # ← 3이 아니라 4!
    dummy_pred_boxes   = torch.randn(900, 11)
    dummy_gt_classes   = torch.randint(0, 3, (5,))
    dummy_gt_boxes     = torch.randn(5, 11)

    criterion = CustomLoss(num_classes=3)
    total_loss, cls_loss, box_loss = criterion(
        dummy_pred_classes, dummy_pred_boxes,
        dummy_gt_classes,   dummy_gt_boxes
    )
    print(f"✅ 분류 Loss (Focal) : {cls_loss.item():.4f}")
    print(f"✅ 박스 Loss         : {box_loss.item():.4f}")
    print(f"🔥 총합 Loss         : {total_loss.item():.4f}")

    # 빈 GT 테스트 (전부 배경)
    empty_gt_classes = torch.zeros(0, dtype=torch.long)
    empty_gt_boxes   = torch.zeros(0, 11)
    total_loss2, _, _ = criterion(
        dummy_pred_classes, dummy_pred_boxes,
        empty_gt_classes, empty_gt_boxes
    )
    print(f"\n✅ 빈 GT (전부 배경) Loss: {total_loss2.item():.4f}")
    print("\n🎉 Focal Loss + 배경 클래스 테스트 통과!")