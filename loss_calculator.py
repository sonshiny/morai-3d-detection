import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

BOX_SCALE = [50., 50., 3.,
             5.,  5., 3.,
             1.,  1.,
             30., 30., 5.]


class FocalLoss(nn.Module):
    def __init__(self, fg_alpha=0.75, bg_alpha=0.25, gamma=2.0):
        super().__init__()
        self.fg_alpha = fg_alpha
        self.bg_alpha = bg_alpha
        self.gamma = gamma

    def forward(self, pred_logits, target, num_fg=None):
        ce_loss = F.cross_entropy(pred_logits, target, reduction='none')
        pt = torch.exp(-ce_loss)
        bg_class = pred_logits.shape[-1] - 1
        alpha_t = torch.where(
            target == bg_class,
            ce_loss.new_tensor(self.bg_alpha),
            ce_loss.new_tensor(self.fg_alpha),
        )
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
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
    def __init__(
        self,
        num_classes=1,
        bg_weight=0.1,
        quality_weight=0.05,
        bg_quality_weight=0.0,
        quality_distance_scale=4.0,
        yawness_weight=0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.bg_class = num_classes
        self.matcher = HungarianMatcher()
        self.focal_loss = FocalLoss(fg_alpha=0.75, bg_alpha=0.25, gamma=2.0)
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

    def forward(self, pred_classes, pred_boxes, gt_classes, gt_boxes, pred_quality=None):
        """
        pred_classes: [900, 2]
        pred_boxes:   [900, 11]
        pred_quality: [900, 2] optional; [:,0]=centerness, [:,1]=yawness
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

        loss_quality = self.quality_loss(pred_quality, pred_boxes, gt_boxes, pred_idx, gt_idx)

        total_loss = 2.0 * loss_class + 2.0 * loss_bbox + self.quality_weight * loss_quality
        return total_loss, loss_class, loss_bbox, loss_quality


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
