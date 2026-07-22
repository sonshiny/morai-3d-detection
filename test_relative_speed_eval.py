#!/usr/bin/env python3
"""
test_relative_speed_eval.py  (P3)
=================================
합성 검증:
  1) bin 경계: 반열림 [lo,hi) 규약 + inf 마지막 bin.
  2) ego 속도 local linear regression: 등속 ego 를 정확히 복원, epoch 경계 미교차, 경계 프레임 표시.
  3) 상대속도 정의: |v_obj_world - v_ego_world| (ego frame→world 회전 포함).
  4) 매칭: greedy·class-aware·score 내림차순·2.0m 반경 (train.py by-distance 규약).
  5) unmatched prediction 은 recall(GT-conditioned)에 영향 없이 별도로 카운트(precision 아님).
"""
import math
import os
import numpy as np
import pytest

from preprocess_dataset import _rot2
from eval_relative_speed import (
    bin_index, local_linear_velocity, _greedy_match_frame, evaluate_relative_speed,
    load_filtered_gt_with_relspeed, DEFAULT_BIN_EDGES,
)

E = DEFAULT_BIN_EDGES   # [0,1,3,6,8,inf]
REPO = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(REPO, "dataset")
_have_data = os.path.isdir(os.path.join(DATASET, "scen144", "labels_3d_v3"))
needs_data = pytest.mark.skipif(not _have_data, reason="./dataset scen144 v3 없음")


def test_bin_boundaries():
    print("[bin boundaries: half-open [lo,hi)]")
    assert bin_index(0.0, E) == 0
    assert bin_index(0.999, E) == 0
    assert bin_index(1.0, E) == 1        # 경계는 상위 bin
    assert bin_index(2.999, E) == 1
    assert bin_index(3.0, E) == 2
    assert bin_index(6.0, E) == 3
    assert bin_index(7.999, E) == 3
    assert bin_index(8.0, E) == 4        # 마지막 bin [8,inf)
    assert bin_index(1e6, E) == 4
    print("  -> PASS\n")


def test_ego_velocity_local_regression():
    print("[ego local-linreg: recover constant velocity, block epoch cross, flag boundary]")
    # 등속 ego: x=2t, y=-3t, 10프레임 @ dt=0.1, 단일 epoch
    ts = [round(0.1 * i, 6) for i in range(10)]
    xs = [2.0 * t for t in ts]
    ys = [-3.0 * t for t in ts]
    ep = ["0"] * 10
    vx, vy, a = local_linear_velocity(ts, xs, ys, ep, 5, halfwidth=3)
    assert abs(vx - 2.0) < 1e-9 and abs(vy + 3.0) < 1e-9, f"got {vx},{vy}"
    assert a["method"] == "linreg" and a["n_used"] == 7 and not a["is_boundary"]
    # 경계 프레임(i=0): 윈도 축소 → is_boundary=True, 그래도 등속이면 기울기 정확
    vx0, vy0, a0 = local_linear_velocity(ts, xs, ys, ep, 0, halfwidth=3)
    assert abs(vx0 - 2.0) < 1e-9 and a0["is_boundary"] and a0["n_used"] == 4
    print(f"  interior n_used={a['n_used']} boundary i=0 n_used={a0['n_used']} (flagged)")

    # epoch 경계: i=5 가 epoch 'A'(0..4)/'B'(5..9) 사이 → A 표본만 사용해야 하며 B(텔레포트) 미혼입
    ep2 = ["A"] * 5 + ["B"] * 5
    xs2 = [2.0 * t for t in ts[:5]] + [1000.0 + 2.0 * (t - ts[5]) for t in ts[5:]]
    vxb, vyb, ab = local_linear_velocity(ts, xs2, ys, ep2, 5, halfwidth=3)
    # i=5 는 epoch B; 윈도 [2..8]∩B = {5,6,7,8} → 전부 B, 기울기 2.0 (텔레포트 점프 미포함)
    assert abs(vxb - 2.0) < 1e-6, f"epoch cross leaked: vx={vxb}"
    assert ab["n_used"] == 4 and ab["is_boundary"]
    print(f"  epoch B frame uses only B samples n_used={ab['n_used']} vx={vxb:.3f} (no teleport blend)\n")


def test_relative_speed_definition():
    print("[relative speed = |v_obj_world - v_ego_world|]")
    yaw = math.radians(30.0)
    R = _rot2(yaw)
    # object ego-frame velocity
    v_ego_obj = np.array([5.0, 1.0])
    v_obj_world = R @ v_ego_obj
    v_ego_world = np.array([4.0, 0.0])
    rel = math.hypot(*(v_obj_world - v_ego_world))
    # 독립 재계산
    exp = math.hypot(R[0, 0] * 5 + R[0, 1] * 1 - 4.0, R[1, 0] * 5 + R[1, 1] * 1 - 0.0)
    assert abs(rel - exp) < 1e-12
    # ego 와 object 가 동일 world 속도면 상대속도 0 (동방향 등속)
    v_ego_obj2 = R.T @ v_ego_world      # world→ego 로 역회전
    v_obj_world2 = R @ v_ego_obj2
    assert math.hypot(*(v_obj_world2 - v_ego_world)) < 1e-12
    print("  -> PASS (rotation + same-world-velocity → rel 0)\n")


def _pred(xs, ys, labels, scores):
    return {"x": np.array(xs, float), "y": np.array(ys, float),
            "label": np.array(labels, int), "score": np.array(scores, float)}


def test_matching_greedy_class_aware_radius():
    print("[matching: greedy, class-aware, 2.0m, score order]")
    gt = [{"x": 0.0, "y": 0.0, "class_id": 0, "track_id": 1, "rel_speed": 0.5},
          {"x": 10.0, "y": 0.0, "class_id": 1, "track_id": 2, "rel_speed": 4.0}]
    # pred A near gt0 class0 (1.9m, in), pred B near gt1 but wrong class (no match),
    # pred C near gt1 class1 (2.1m, out of radius)
    pred = _pred([1.9, 10.0, 12.1], [0.0, 0.0, 0.0], [0, 0, 1], [0.9, 0.8, 0.7])
    matched, mdist = _greedy_match_frame(gt, pred, dist_thr=2.0)
    assert matched[0] is True and abs(mdist[0] - 1.9) < 1e-9, "gt0 should match pred A"
    assert matched[1] is False, "gt1 must NOT match (wrong class / out of radius)"
    print(f"  gt0 matched@{mdist[0]:.2f}m, gt1 unmatched (class/radius) -> PASS")

    # score 순서: 더 가까운 pred 라도 낮은 score pred 가 먼저 GT 를 가져가면 greedy 반영
    gt2 = [{"x": 0.0, "y": 0.0, "class_id": 0, "track_id": 1, "rel_speed": 0.5}]
    pred2 = _pred([1.5, 0.2], [0.0, 0.0], [0, 0], [0.95, 0.5])  # high-score far(1.5) 먼저
    m2, d2 = _greedy_match_frame(gt2, pred2, 2.0)
    assert m2[0] and abs(d2[0] - 1.5) < 1e-9, "greedy: highest score claims GT first"
    print(f"  greedy score-order: GT taken by high-score pred @{d2[0]:.2f}m -> PASS\n")


def test_unmatched_prediction_excluded_from_recall():
    print("[unmatched prediction: counted separately, recall unaffected]")
    frames = [{"scen": "s", "stem": "f0",
               "boxes": [{"x": 0.0, "y": 0.0, "class_id": 0, "track_id": 1, "rel_speed": 0.5}]}]
    # 2 preds: one matches GT, one is a false positive far away (no GT there)
    preds = {("s", "f0"): _pred([0.1, 50.0], [0.0, 0.0], [0, 0], [0.9, 0.8])}
    res = evaluate_relative_speed(frames, preds, DEFAULT_BIN_EDGES, dist_thr=2.0)
    b0 = res["bins"][0]
    assert b0["n_gt"] == 1 and b0["n_matched"] == 1 and b0["recall"] == 1.0
    assert res["n_pred_total"] == 2 and res["n_pred_unmatched"] == 1
    # unmatched prediction 은 어떤 bin recall 도 바꾸지 않는다
    assert all((r["recall"] in (None, 1.0) or r["n_gt"] == 0) for r in res["bins"])
    assert "precision" not in res["note"].lower() or "아님" in res["note"]
    print(f"  recall(bin0)={b0['recall']} n_pred_unmatched={res['n_pred_unmatched']} "
          "→ FP not folded into recall -> PASS\n")


def test_evaluate_fail_fast_on_empty():
    print("[fail-fast: empty frame/GT/pred must raise, not silently 0/0]")
    frames = [{"scen": "s", "stem": "live_0",
               "boxes": [{"x": 0.0, "y": 0.0, "class_id": 0, "track_id": 1, "rel_speed": 0.5}]}]
    preds = {("s", "live_0"): _pred([0.1], [0.0], [0], [0.9])}
    ev = evaluate_relative_speed(frames, preds, DEFAULT_BIN_EDGES, require_nonempty=True)
    assert ev["total_gt"] == 1 and ev["n_pred_total"] == 1 and ev["n_frames_evaluated"] == 1

    def _raises(fr, pr):
        try:
            evaluate_relative_speed(fr, pr, DEFAULT_BIN_EDGES, require_nonempty=True)
        except ValueError:
            return True
        return False

    assert _raises([], preds), "empty frames must fail-fast"
    assert _raises(frames, {}), "0 prediction must fail-fast (join 실패 시나리오)"
    assert _raises([{"scen": "s", "stem": "live_0", "boxes": []}], preds), "0 GT must fail-fast"
    print("  -> PASS (empty frame/GT/pred all raise)\n")


@needs_data
def test_filtered_gt_uses_production_membership():
    print("[filtered GT == production validation filter membership + rel_speed attached]")
    frames, audit, n_missing = load_filtered_gt_with_relspeed(DATASET, "scen144", "v3")
    n = sum(len(f["boxes"]) for f in frames)
    # membership 감사: scen144 v3 natural filter = 4834 (g_membership_audit)
    assert n == 4834, f"scen144 v3 filtered box 기대 4834, got {n}"
    assert n_missing == 0, f"filtered box 중 rel_speed 매칭 실패 {n_missing}건"
    assert all(f["stem"].startswith("live_") for f in frames), "stem 은 base(live_N) 여야 함"
    # raw all-box(5534) 보다 적어야(필터 적용 증거)
    assert n < 5534
    print(f"  scen144 v3 filtered={n} (<5534 raw), rel_speed missing={n_missing} -> PASS\n")


if __name__ == "__main__":
    test_bin_boundaries()
    test_ego_velocity_local_regression()
    test_relative_speed_definition()
    test_matching_greedy_class_aware_radius()
    test_unmatched_prediction_excluded_from_recall()
    test_evaluate_fail_fast_on_empty()
    if _have_data:
        test_filtered_gt_uses_production_membership()
    print("ALL P3 RELATIVE-SPEED TESTS PASSED")
