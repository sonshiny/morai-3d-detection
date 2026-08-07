#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_metric_sweep_equivalence.py
================================
검증 metric 가속 함수 compute_metrics_sweep 가, 기존 방식
(compute_detection_counts_by_class 를 mode×threshold 반복 호출한 합산)과
**완전히 동일한** (tp, fp, fn) 을 mode/threshold/class 전 조합에서 산출하는지 검증한다.

핵심: eval 코드 최적화라 결과가 조금이라도 달라지면 best-model 선택/early-stop 이
오염된다. 그래서 랜덤 입력 다수(빈 GT/빈 예측/큰 규모 포함)에서 bit-단위 일치를 assert.

실행:  python3 test_metric_sweep_equivalence.py
       python3 -m pytest -q test_metric_sweep_equivalence.py
"""
import sys
from unittest.mock import MagicMock

# train.py 는 최상단에서 wandb 를 import 하므로, 테스트 환경에선 mock 으로 대체.
sys.modules.setdefault("wandb", MagicMock())

import torch  # noqa: E402
import train  # noqa: E402

# 검증 루프와 동일한 계약
METRIC_MODES = {
    "calibrated": (True, 1.0),
    "softcalibrated": (True, 0.5),
    "raw": (False, 1.0),
}
SCORE_THRESHOLDS = (0.01, 0.03, 0.05, 0.10, 0.15, 0.25)
CLASS_IDS = (0, 1)
DIST_THR = 2.0


def _old_reference(det_cls, det_box, gt_cls, gt_box, det_q):
    """기존 검증 루프(train.py:1778-1804)와 동일한 계산 경로."""
    result = {}
    for mode, (apply_q, q_pow) in METRIC_MODES.items():
        result[mode] = {}
        for thr in SCORE_THRESHOLDS:
            per_cls = train.compute_detection_counts_by_class(
                det_cls, det_box, gt_cls, gt_box,
                det_quality=det_q, distance_thr=DIST_THR,
                score_thresh=thr, apply_quality=apply_q,
                quality_power=q_pow, class_ids=CLASS_IDS,
            )
            result[mode][thr] = {c: tuple(per_cls[c]) for c in CLASS_IDS}
    return result


def _new_sweep(det_cls, det_box, gt_cls, gt_box, det_q):
    swept = train.compute_metrics_sweep(
        det_cls, det_box, gt_cls, gt_box, det_q,
        METRIC_MODES, SCORE_THRESHOLDS,
        distance_thr=DIST_THR, class_ids=CLASS_IDS,
    )
    return {m: {t: {c: tuple(swept[m][t][c]) for c in CLASS_IDS}
                for t in SCORE_THRESHOLDS} for m in METRIC_MODES}


def _make_case(seed, n_pred, n_gt):
    g = torch.Generator().manual_seed(seed)
    # det_cls: [N,2] 로짓, det_box: [N,11] (x,y 는 앞 2열), det_quality: [N,2]
    det_cls = torch.randn(n_pred, 2, generator=g)
    det_box = torch.randn(n_pred, 11, generator=g)
    det_box[:, :2] *= 25.0                       # x,y 를 ±25m 정도로
    det_q = torch.randn(n_pred, 2, generator=g)
    gt_box = torch.randn(n_gt, 11, generator=g)
    gt_box[:, :2] *= 25.0
    gt_cls = torch.randint(0, 2, (n_gt,), generator=g)
    # TP 가 실제로 생기도록 일부 GT 를 일부 예측 근처에 배치
    k = min(n_pred, n_gt) // 2
    if k > 0:
        gt_box[:k, :2] = det_box[:k, :2] + 0.3 * torch.randn(k, 2, generator=g)
        gt_cls[:k] = det_cls[:k].argmax(dim=-1)
    return det_cls, det_box, gt_cls, gt_box, det_q


def run_all():
    cases = [
        (1, 50, 20), (2, 200, 40), (3, 400, 60), (4, 30, 30),
        (5, 500, 5), (6, 5, 50),          # 불균형
        (7, 100, 0),                       # GT 없음
        (8, 0, 20),                        # 예측 없음
        (9, 0, 0),                         # 둘 다 없음
        (10, 350, 350), (11, 1000, 100),   # 대규모 (top-k 300 초과)
    ]
    n_checks = 0
    for seed, n_pred, n_gt in cases:
        det_cls, det_box, gt_cls, gt_box, det_q = _make_case(seed, n_pred, n_gt)
        old = _old_reference(det_cls, det_box, gt_cls, gt_box, det_q)
        new = _new_sweep(det_cls, det_box, gt_cls, gt_box, det_q)
        for m in METRIC_MODES:
            for t in SCORE_THRESHOLDS:
                for c in CLASS_IDS:
                    n_checks += 1
                    assert old[m][t][c] == new[m][t][c], (
                        f"불일치 seed={seed} n_pred={n_pred} n_gt={n_gt} "
                        f"mode={m} thr={t} cls={c}: old={old[m][t][c]} new={new[m][t][c]}"
                    )
        # overall(클래스 합)도 동일한지
        for m in METRIC_MODES:
            for t in SCORE_THRESHOLDS:
                o = tuple(sum(old[m][t][c][k] for c in CLASS_IDS) for k in range(3))
                nw = tuple(sum(new[m][t][c][k] for c in CLASS_IDS) for k in range(3))
                assert o == nw, f"overall 불일치 seed={seed} mode={m} thr={t}: {o} vs {nw}"
    print(f"[PASS] 모든 케이스 일치 — {len(cases)} cases × {len(METRIC_MODES)}modes × "
          f"{len(SCORE_THRESHOLDS)}thr × {len(CLASS_IDS)}cls = {n_checks} 검사 통과")


def test_metric_sweep_equivalence():
    run_all()


# ─────────────────────────────────────────────────────────────────────────────
# 멀티프로세싱(joblib) 경로 등가: _val_frame_metric 를 직렬 vs 병렬로 누적한 결과가
# 완전히 동일한지 검증한다. (joblib 은 입력 순서대로 결과를 돌려주고 병합은 합산이라
#  bit-단위 동일해야 함.)
RECALL_THR = 2.0


def _fresh_acc():
    n_buckets = len(train.DISTANCE_BUCKETS)
    return {
        "mc": {m: {t: {"tp": 0, "fp": 0, "fn": 0} for t in SCORE_THRESHOLDS} for m in METRIC_MODES},
        "mcc": {m: {t: {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASS_IDS}
                    for t in SCORE_THRESHOLDS} for m in METRIC_MODES},
        "dist": {i: {"tp": 0, "fp": 0, "fn": 0, "dist_sum": 0.0, "dist_n": 0} for i in range(n_buckets)},
        "ss": {"quality": 0.0, "raw": 0.0, "soft": 0.0},
        "sm": {"quality": 0.0, "raw": 0.0, "soft": 0.0},
        "sc": 0,
    }


def _merge(acc, result):
    stats, swept, dist = result
    acc["ss"]["quality"] += stats["quality_sum"]
    acc["ss"]["raw"] += stats["raw_sum"]
    acc["ss"]["soft"] += stats["soft_sum"]
    acc["sm"]["quality"] = max(acc["sm"]["quality"], stats["quality_max"])
    acc["sm"]["raw"] = max(acc["sm"]["raw"], stats["raw_max"])
    acc["sm"]["soft"] = max(acc["sm"]["soft"], stats["soft_max"])
    acc["sc"] += stats["count"]
    for m in METRIC_MODES:
        for t in SCORE_THRESHOLDS:
            for c in CLASS_IDS:
                tp, fp, fn = swept[m][t][c]
                acc["mcc"][m][t][c]["tp"] += tp
                acc["mcc"][m][t][c]["fp"] += fp
                acc["mcc"][m][t][c]["fn"] += fn
                acc["mc"][m][t]["tp"] += tp
                acc["mc"][m][t]["fp"] += fp
                acc["mc"][m][t]["fn"] += fn
    for b, counts in dist.items():
        for k in ("tp", "fp", "fn", "dist_sum", "dist_n"):
            acc["dist"][b][k] += counts[k]


def run_parallel_equivalence():
    from joblib import Parallel, delayed
    frames = []
    for seed, n_pred, n_gt in [(20, 300, 40), (21, 500, 60), (22, 30, 30),
                               (23, 900, 20), (24, 100, 0), (25, 0, 30)]:
        dc, dbx, gc, gb, dq = _make_case(seed, n_pred, n_gt)
        frames.append((dc, dbx, dq, gc, gb))   # _val_frame_metric 시그니처 순서

    acc_serial = _fresh_acc()
    for (dc, dbx, dq, gc, gb) in frames:
        _merge(acc_serial, train._val_frame_metric(dc, dbx, dq, gc, gb, RECALL_THR, CLASS_IDS))

    results = Parallel(n_jobs=3, prefer="processes")(
        delayed(train._val_frame_metric)(dc, dbx, dq, gc, gb, RECALL_THR, CLASS_IDS)
        for (dc, dbx, dq, gc, gb) in frames
    )
    acc_par = _fresh_acc()
    for r in results:
        _merge(acc_par, r)

    assert acc_serial == acc_par, "직렬 vs 병렬 누적 결과 불일치"
    print(f"[PASS] 멀티프로세싱 등가 — {len(frames)} frames 직렬==병렬(joblib) 누적 완전 일치")


def test_val_frame_metric_parallel_equivalence():
    run_parallel_equivalence()


if __name__ == "__main__":
    run_all()
    run_parallel_equivalence()
