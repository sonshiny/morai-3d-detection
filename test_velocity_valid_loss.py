#!/usr/bin/env python3
"""
test_velocity_valid_loss.py
===========================
P2 velocity-validity guard 의 loss 동작 검증:
  1) baseline 동일성: velocity_valid=None 과 모두-1(raw) 은 기존 F.l1_loss(mean)과 수치적으로 동일.
  2) gradient masking: velocity_valid=0 인 GT 에 매칭된 pred 는 vx,vy(ch8,9) gradient 가 0,
     위치/치수/yaw 채널은 정상 gradient. matcher 는 velocity_valid 와 무관(불변).
"""
import torch
import torch.nn.functional as F

from loss_calculator import CustomLoss, BOX_SCALE, MATCH_CHANNELS, REG_CHANNELS


def _mk():
    torch.manual_seed(0)
    N_anchor = 50
    pred_classes = torch.randn(N_anchor, 2)
    pred_boxes = torch.randn(N_anchor, 11)
    pred_quality = torch.randn(N_anchor, 2)
    gt_classes = torch.tensor([0, 1], dtype=torch.long)
    gt_boxes = torch.randn(2, 11)
    return pred_classes, pred_boxes, pred_quality, gt_classes, gt_boxes


def test_baseline_equivalence():
    print("[baseline equivalence: None == all-ones == old F.l1_loss]")
    crit = CustomLoss(num_classes=2, quality_weight=0.2)
    pc, pb, pq, gc, gb = _mk()

    _, _, box_none, _ = crit(pc, pb, gc, gb, pq, velocity_valid=None)
    _, _, box_ones, _ = crit(pc, pb, gc, gb, pq, velocity_valid=torch.ones(gb.shape[0]))

    # 기존 공식 직접 재현(같은 matcher 인덱스 사용)
    pred_idx, gt_idx = crit.matcher(pc, pb, gc, gb)
    scale = torch.tensor(BOX_SCALE)
    old = F.l1_loss(pb[pred_idx, :REG_CHANNELS] / scale[:REG_CHANNELS],
                    gb[gt_idx, :REG_CHANNELS] / scale[:REG_CHANNELS])

    print(f"  box_loss None={box_none.item():.8f}  ones={box_ones.item():.8f}  old={old.item():.8f}")
    assert torch.allclose(box_none, box_ones, atol=1e-7), "None != all-ones"
    assert torch.allclose(box_none, old, atol=1e-7), "masked-path != old F.l1_loss"
    print("  -> PASS (all-raw is numerically identical to baseline)\n")


def test_gradient_masking():
    print("[gradient masking: velocity_valid=0 masks only vx,vy grad]")
    crit = CustomLoss(num_classes=2, quality_weight=0.2)
    pc, pb0, pq, gc, gb = _mk()

    # 어떤 pred 가 어느 GT 에 매칭되는지(velocity_valid 와 무관) 확인
    pred_idx, gt_idx = crit.matcher(pc, pb0, gc, gb)
    # gt 0 = valid(1), gt 1 = masked(0)
    vv = torch.tensor([1.0, 0.0])
    # gt_idx 순서 -> 매칭된 pred anchor
    pred_for_gt = {int(g): int(p) for p, g in zip(pred_idx.tolist(), gt_idx.tolist())}
    p_valid = pred_for_gt[0]
    p_masked = pred_for_gt[1]

    pb = pb0.clone().detach().requires_grad_(True)
    total, _, _, _ = crit(pc, pb, gc, gb, pq, velocity_valid=vv)
    total.backward()
    g = pb.grad

    gv = g[p_masked, MATCH_CHANNELS:REG_CHANNELS].abs().sum().item()   # masked box vx,vy grad
    gv_xy = g[p_masked, 0:2].abs().sum().item()                       # masked box x,y grad
    gvalid = g[p_valid, MATCH_CHANNELS:REG_CHANNELS].abs().sum().item()  # valid box vx,vy grad

    print(f"  masked GT->pred[{p_masked}]: |grad vx,vy|={gv:.3e}  |grad x,y|={gv_xy:.3e}")
    print(f"  valid  GT->pred[{p_valid}]: |grad vx,vy|={gvalid:.3e}")
    assert gv == 0.0, f"masked box vx,vy grad must be 0, got {gv}"
    assert gv_xy > 0.0, "masked box x,y grad must be nonzero (position not masked)"
    assert gvalid > 0.0, "valid box vx,vy grad must be nonzero"
    print("  -> PASS (only vx,vy of masked box is zeroed; position/dims/yaw intact)\n")


def test_clean_channel_gradient_invariance():
    """velocity mask가 invalid GT의 깨끗한 회귀 채널 gradient를 재가중하지 않아야 한다."""
    print("[clean-channel invariance: masking velocity must not rescale x/y/dims/yaw]")
    crit = CustomLoss(num_classes=2, quality_weight=0.2)
    pc, pb0, pq, gc, gb = _mk()
    pred_idx, gt_idx = crit.matcher(pc, pb0, gc, gb)
    pred_for_gt = {int(g): int(p) for p, g in zip(pred_idx.tolist(), gt_idx.tolist())}
    p_masked = pred_for_gt[1]

    pb_all = pb0.clone().detach().requires_grad_(True)
    total_all, _, _, _ = crit(
        pc, pb_all, gc, gb, pq, velocity_valid=torch.ones(gb.shape[0]))
    total_all.backward()

    pb_mask = pb0.clone().detach().requires_grad_(True)
    total_mask, _, _, _ = crit(
        pc, pb_mask, gc, gb, pq, velocity_valid=torch.tensor([1.0, 0.0]))
    total_mask.backward()

    clean = slice(0, MATCH_CHANNELS)
    g_all = pb_all.grad[p_masked, clean]
    g_mask = pb_mask.grad[p_masked, clean]
    max_diff = (g_all - g_mask).abs().max().item()
    print(f"  masked GT clean-channel max|grad(mask)-grad(all-valid)|={max_diff:.3e}")
    assert torch.equal(g_all, g_mask), (
        "velocity mask changed clean-channel gradients; denominator must remain baseline-sized")
    print("  -> PASS (clean-channel gradients are bitwise identical)\n")


def test_matcher_unaffected():
    print("[matcher invariance: velocity_valid does not change matching]")
    crit = CustomLoss(num_classes=2, quality_weight=0.2)
    pc, pb, pq, gc, gb = _mk()
    i0, j0 = crit.matcher(pc, pb, gc, gb)
    # matcher 는 velocity_valid 인자를 받지 않으므로 호출 자체가 동일. 결정성만 확인.
    i1, j1 = crit.matcher(pc, pb, gc, gb)
    assert torch.equal(i0, i1) and torch.equal(j0, j1), "matcher not deterministic"
    print("  -> PASS (matcher uses 8 channels only; velocity_valid never enters cost)\n")


if __name__ == "__main__":
    test_baseline_equivalence()
    test_gradient_masking()
    test_clean_channel_gradient_invariance()
    test_matcher_unaffected()
    print("ALL VELOCITY-VALID LOSS TESTS PASSED")
