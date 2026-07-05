#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_distance.py
================
이미 학습된 best_model.pth를 로드해서 validate()를 딱 한 번 호출하고,
v10에서 추가된 by_distance(ego 방사거리 구간별 P/R/F1 + 매칭쌍 평균 center
distance)와 softcalibrated@0.15 전체 지표만 출력하는 단발성 진단 스크립트.

이 스크립트는 학습을 하지 않는다:
  - optimizer / loss.backward() / epoch loop / scaler / wandb 전부 없음.
  - best_model.pth의 순수 state_dict만 로드하는 추론(eval) 전용이다.

train.py는 절대 수정하지 않고 import만 한다. import 시 train.py의
`if __name__ == "__main__":` 블록은 실행되지 않으므로(학습/wandb.init/kmeans
생성은 전부 그 안에 있음) 여기서 train을 import해도 학습이 트리거되지 않는다.
"""

import os
import sys

import torch

# ───────────────────────────────────────────────────────────
# train.py의 "모듈 레벨" 정의를 그대로 재사용한다 (재정의 금지).
#   - AutoNavModel / validate / DISTANCE_BUCKETS : train.py에서 직접 정의
#   - MoraiDataset / morai_collate_fn            : train.py가 morai_dataset에서 import (재-export)
#   - CustomLoss                                 : train.py가 loss_calculator에서 import (재-export)
#   - DataLoader                                 : train.py가 torch.utils.data에서 import (재-export)
# 위 심볼들은 전부 train 모듈의 속성이므로 아래 한 줄로 train과 "동일 객체"를 얻는다.
# ───────────────────────────────────────────────────────────
from train import (
    AutoNavModel,
    MoraiDataset,
    morai_collate_fn,
    CustomLoss,
    DataLoader,
    validate,
    DISTANCE_BUCKETS,
)

# ───────────────────────────────────────────────────────────
# ⚠️ 아래 설정값들은 train.py에서 import가 "불가능"하다.
# train.py에서 이들은 모듈 전역이 아니라 `if __name__ == "__main__":`
# 블록(train.py 1646행~) 안의 지역 변수로 정의돼 있어, import 시에는
# 존재하지 않는다. 따라서 train.py 원본 라인의 값을 그대로 복제하고,
# 각 줄에 출처 라인을 주석으로 남긴다. train.py에서 값이 바뀌면 여기도
# 함께 맞춰야 한다 (drift 주의).
# ───────────────────────────────────────────────────────────
DATASET_ROOT        = './dataset'      # train.py:1648
VAL_SCENARIOS       = None             # train.py:1651
BATCH_SIZE          = 4                # train.py:1654
USE_TEMPORAL_MEMORY = False            # train.py:1663
NUM_TEMP_INSTANCES  = 600              # train.py:1664
BEST_MODEL_PATH     = "best_model.pth"  # train.py:1671 (save_best_with_epoch가 저장한 순수 state_dict)


def main():
    # ─── device (train.py:1708과 동일) ───────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # ─── sanity check: best_model.pth 존재 확인 (요구사항 6) ─
    if not os.path.isfile(BEST_MODEL_PATH):
        print(f"[ERROR] 체크포인트 파일이 없습니다: {os.path.abspath(BEST_MODEL_PATH)}")
        print("        train.py로 학습해 best_model.pth를 먼저 생성하세요.")
        sys.exit(1)

    # ─── 모델 생성 (train.py:1721-1726과 동일 인자) ────────
    model = AutoNavModel(
        num_decoder_layers=6,
        pretrained_backbone=True,
        use_temporal_memory=USE_TEMPORAL_MEMORY,
        num_temp_instances=NUM_TEMP_INSTANCES,
    ).to(device)

    # ─── best_model.pth (순수 state_dict) 로드 ────────────
    state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    print(f"[load] state_dict 로드 완료: {os.path.abspath(BEST_MODEL_PATH)}")

    # ─── val 데이터셋/로더 (train.py:1731, 1750-1751과 동일) ─
    val_ds = MoraiDataset(
        dataset_root=DATASET_ROOT,
        split='val',
        val_scenarios=VAL_SCENARIOS,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=morai_collate_fn,
        num_workers=0,
    )

    # ─── criterion (train.py:1755와 동일) ────────────────
    # validate()가 loss 계산에 criterion을 쓰므로 필요 (역전파는 하지 않음).
    det_criterion = CustomLoss(num_classes=2, quality_weight=0.2).to(device)

    # ─── validate() 단 1회 호출 (recall_thr는 기본 2.0 = train RECALL_THR) ─
    print("\n[validate] compute_metric=True 로 1회 평가 실행...\n")
    val_loss, metrics = validate(
        model, val_loader, det_criterion, device, compute_metric=True
    )

    if metrics is None:
        print("[ERROR] metrics가 None입니다 (compute_metric=True인데도 반환 안 됨).")
        sys.exit(1)

    # ─── 출력 ────────────────────────────────────────────
    print("=" * 72)
    print(f"  Val loss: {val_loss:.4f}")

    # softcalibrated @ score>=0.15 전체 (거리 구분 없음)
    overall = metrics.get('by_mode', {}).get('softcalibrated', {}).get(0.15)
    print("-" * 72)
    print("  [softcalibrated @ score>=0.15] 전체 (거리 무관)")
    if overall is not None:
        print(
            f"    P/R/F1 = {overall['precision']:.4f} / "
            f"{overall['recall']:.4f} / {overall['f1']:.4f}"
            f"   (tp={overall['tp']}, fp={overall['fp']}, fn={overall['fn']})"
        )
    else:
        print("    (by_mode['softcalibrated'][0.15] 없음)")

    # by_distance: ego 방사거리 3구간별 P/R/F1 + 매칭쌍 평균 center distance
    by_distance = metrics.get('by_distance', {})
    print("-" * 72)
    print("  [softcalibrated @ score>=0.15] ego 방사거리 구간별 (v10 by_distance)")
    print(f"    {'구간':>12} | {'P':>7} | {'R':>7} | {'F1':>7} | {'mean_cdist':>10} | tp/fp/fn")
    if by_distance:
        # DISTANCE_BUCKETS 순서대로 출력 (train.py와 동일 정의를 import해 사용)
        for (lo, hi) in DISTANCE_BUCKETS:
            dm = by_distance.get((lo, hi))
            if dm is None:
                continue
            cdist = dm['mean_center_dist']
            cdist_str = f"{cdist:.3f} m" if cdist is not None else "n/a"
            label = f"[{lo:.0f}-{hi:.0f}m)"
            print(
                f"    {label:>12} | "
                f"{dm['precision']:.4f} | {dm['recall']:.4f} | {dm['f1']:.4f} | "
                f"{cdist_str:>10} | {dm['tp']}/{dm['fp']}/{dm['fn']}"
            )
    else:
        print("    (by_distance 없음 — train.py의 v10 로깅 확장이 적용됐는지 확인)")
    print("=" * 72)


if __name__ == "__main__":
    main()
