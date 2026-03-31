import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

BOX_SCALE = [50., 50., 3.,
             5.,  5., 3.,
             1.,  1.,
             30., 30., 5.]


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_logits, target, num_fg=None):
        ce_loss = F.cross_entropy(pred_logits, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if num_fg is not None:
            return focal_loss.sum() / num_fg.clamp(min=1).float()
        return focal_loss.mean()


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1.0, cost_bbox=5.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox

    @torch.no_grad()
    def forward(self, pred_classes, pred_boxes, gt_classes, gt_boxes):
        """
        pred_classes : [900, num_classes+1]
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

        out_prob = pred_classes.softmax(-1)
        gt_classes = gt_classes.clamp(min=0, max=pred_classes.shape[-1] - 1)
        cost_class = -out_prob[:, gt_classes]

        pred_norm = pred_boxes / scale
        gt_norm = gt_boxes / scale
        cost_bbox = torch.cdist(pred_norm, gt_norm, p=1)

        C = self.cost_class * cost_class + self.cost_bbox * cost_bbox
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
    def __init__(self, num_classes=1, bg_weight=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.bg_class = num_classes
        self.matcher = HungarianMatcher()
        self.focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
        self.bg_weight = bg_weight

    def forward(self, pred_classes, pred_boxes, gt_classes, gt_boxes):
        """
        pred_classes: [900, 2]
        pred_boxes:   [900, 11]
        gt_classes:   [N]
        gt_boxes:     [N, 11]
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
            target = torch.full(
                (num_anchors,),
                self.bg_class,
                dtype=torch.long,
                device=device
            )
            loss_class = self.focal_loss(pred_classes, target) * self.bg_weight
            zero = torch.tensor(0.0, device=device)
            return loss_class, loss_class, zero

        pred_idx, gt_idx = self.matcher(pred_classes, pred_boxes, gt_classes, gt_boxes)

        target = torch.full(
            (num_anchors,),
            self.bg_class,
            dtype=torch.long,
            device=device
        )
        target[pred_idx] = gt_classes[gt_idx]

        num_fg = torch.tensor(len(pred_idx), device=device)
        loss_class = self.focal_loss(pred_classes, target, num_fg=num_fg)

        if len(pred_idx) == 0:
            loss_bbox = torch.tensor(0.0, device=device)
        else:
            scale = torch.tensor(BOX_SCALE, device=device, dtype=pred_boxes.dtype)
            loss_bbox = F.l1_loss(
                pred_boxes[pred_idx] / scale,
                gt_boxes[gt_idx] / scale
            )

        total_loss = 2.0 * loss_class + 0.25 * loss_bbox
        return total_loss, loss_class, loss_bbox


if __name__ == "__main__":
    print("🚀 Focal Loss + 배경 클래스 Loss 테스트!\n")

    dummy_pred_classes = torch.randn(900, 2)
    dummy_pred_boxes = torch.randn(900, 11)
    dummy_gt_classes = torch.randint(0, 1, (5,), dtype=torch.long)
    dummy_gt_boxes = torch.randn(5, 11)

    criterion = CustomLoss(num_classes=1)
    total_loss, cls_loss, box_loss = criterion(
        dummy_pred_classes, dummy_pred_boxes,
        dummy_gt_classes, dummy_gt_boxes
    )
    print(f"✅ 분류 Loss (Focal) : {cls_loss.item():.4f}")
    print(f"✅ 박스 Loss         : {box_loss.item():.4f}")
    print(f"🔥 총합 Loss         : {total_loss.item():.4f}")

    empty_gt_classes = torch.zeros(0, dtype=torch.long)
    empty_gt_boxes = torch.zeros(0, 11)
    total_loss2, _, _ = criterion(
        dummy_pred_classes, dummy_pred_boxes,
        empty_gt_classes, empty_gt_boxes
    )
    print(f"\n✅ 빈 GT (전부 배경) Loss: {total_loss2.item():.4f}")
    print("\n🎉 Focal Loss + 배경 클래스 테스트 통과!")