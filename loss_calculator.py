import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

# 정규화 스케일 (전역 상수로 관리)
BOX_SCALE = [50., 50., 3.,   # x, y, z
              5.,  5., 3.,   # w, l, h
              1.,  1.,       # sin, cos
             30., 30., 5.]   # vx, vy, vz

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1.0, cost_bbox=5.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox  = cost_bbox

    @torch.no_grad()
    def forward(self, pred_classes, pred_boxes, gt_classes, gt_boxes):
        """
        pred_classes : [900, 3]
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

        out_prob   = pred_classes.softmax(-1)
        cost_class = -out_prob[:, gt_classes]

        # ✅ 버그2 수정: 매칭 비용도 정규화된 값으로 계산
        pred_norm  = pred_boxes / scale    # [900, 11]
        gt_norm    = gt_boxes   / scale    # [N, 11]
        cost_bbox  = torch.cdist(pred_norm, gt_norm, p=1)  # [900, N]

        C = self.cost_class * cost_class + self.cost_bbox * cost_bbox
        C = C.cpu().numpy()

        pred_indices, gt_indices = linear_sum_assignment(C)
        return (torch.as_tensor(pred_indices, dtype=torch.int64),
                torch.as_tensor(gt_indices,   dtype=torch.int64))


class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.matcher = HungarianMatcher()
        self.loss_ce = nn.CrossEntropyLoss()

    def forward(self, pred_classes, pred_boxes, gt_classes, gt_boxes):
        device = pred_classes.device

        if gt_boxes.shape[0] == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, zero, zero

        pred_idx, gt_idx = self.matcher(pred_classes, pred_boxes,
                                        gt_classes,   gt_boxes)
        pred_idx = pred_idx.to(device)
        gt_idx   = gt_idx.to(device)

        # 분류 Loss
        loss_class = self.loss_ce(pred_classes[pred_idx], gt_classes[gt_idx])

        # 박스 Loss (정규화된 값으로 비교)
        scale     = torch.tensor(BOX_SCALE, device=device)
        loss_bbox = F.l1_loss(pred_boxes[pred_idx] / scale,
                              gt_boxes[gt_idx]     / scale)

        total_loss = loss_class + 5.0 * loss_bbox
        return total_loss, loss_class, loss_bbox


if __name__ == "__main__":
    print("🚀 Loss 채점 테스트!\n")
    dummy_pred_classes = torch.randn(900, 3)
    dummy_pred_boxes   = torch.randn(900, 11)
    dummy_gt_classes   = torch.randint(0, 3, (5,))
    dummy_gt_boxes     = torch.randn(5, 11)

    criterion = CustomLoss()
    total_loss, cls_loss, box_loss = criterion(
        dummy_pred_classes, dummy_pred_boxes,
        dummy_gt_classes,   dummy_gt_boxes
    )
    print(f"✅ 분류 Loss : {cls_loss.item():.4f}")
    print(f"✅ 박스 Loss : {box_loss.item():.4f}")
    print(f"🔥 총합 Loss : {total_loss.item():.4f}")