#!/usr/bin/env bash
# Phase 2 repair pilot from epoch20 — A/B 한쪽 팔 실행.
#
# 질문: "기존(잘못된) objective 로 20 epoch 학습된 모델을 새 공식 계약이 교정하는가?"
# 설계 (2026-07-30 합의):
#   Control  : LOSS_CONTRACT=legacy  — 기존 계약 + optimizer reset (더-학습 효과 대조군)
#   Treatment: LOSS_CONTRACT=parity  — 공식 Stage-1 계약 + optimizer reset
#   공통     : INIT_WEIGHTS=checkpoint_epoch20.pth (weight 만), scheduler/step 초기화,
#              같은 SEED/데이터/anchor/augmentation, NUM_EPOCHS=2, 새 RUN_DIR.
#   기준선   : epoch20 final_nonms mAP 0.0491 (runs/prod_v3_depth_20260726_1332/ap_eval)
#
# 사용: ARM=control|parity ./scripts/pilot_repair.sh
# 필수 env: DATASET_ROOT, ANCHOR_DIR, VAL_SCENARIOS, INIT_WEIGHTS, RUN_DIR
source "$(dirname "$0")/_lib.sh"
REPO="$(repo_root)"
need_dir "$DATASET_ROOT"
: "${ARM:?ARM=control|parity 지정 필요}"
: "${VAL_SCENARIOS:?VAL_SCENARIOS 지정 필요}"
: "${ANCHOR_DIR:?ANCHOR_DIR 지정 필요}"
: "${INIT_WEIGHTS:?INIT_WEIGHTS(epoch20 checkpoint 경로) 지정 필요}"
: "${RUN_DIR:?RUN_DIR 지정 필요(팔별 신규 디렉터리)}"
need_file "$INIT_WEIGHTS"
need_file "$ANCHOR_DIR/anchor_kmeans_full.npy"
[ -e "$RUN_DIR" ] && die "RUN_DIR 이 이미 존재합니다(팔별 신규 디렉터리 필수): $RUN_DIR"

case "$ARM" in
  control) CONTRACT=legacy ;;
  parity)  CONTRACT=parity ;;
  *) die "ARM 은 control 또는 parity: $ARM" ;;
esac

EPOCHS="${NUM_EPOCHS:-2}"
info "REPAIR PILOT [$ARM] contract=$CONTRACT init=$INIT_WEIGHTS epochs=$EPOCHS → $RUN_DIR"
run env DATASET_ROOT="$DATASET_ROOT" VAL_SCENARIOS="$VAL_SCENARIOS" ANCHOR_DIR="$ANCHOR_DIR" \
  TRAIN_GT_VERSION=v3 VAL_GT_VERSION=v3 USE_DENSE_DEPTH="${USE_DENSE_DEPTH:-1}" SEED="${SEED:-0}" \
  USE_TEMPORAL_MEMORY=1 STREAMING_SAMPLER=1 TEMP_GNN_MODE=gated FILTER_VISIBLE=0 \
  BATCH_SIZE="${BATCH_SIZE:-4}" GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}" \
  NUM_WORKERS="${NUM_WORKERS:-4}" USE_AMP="${USE_AMP:-0}" ALLOW_TF32="${ALLOW_TF32:-1}" \
  RESUME=0 FORCE_REMAKE_KMEANS=0 NUM_EPOCHS="$EPOCHS" \
  LOSS_CONTRACT="$CONTRACT" INIT_WEIGHTS="$INIT_WEIGHTS" \
  EARLY_STOP_PATIENCE=100 \
  MAX_STEPS_PER_EPOCH=0 FAST_VAL_MAX_FRAMES=0 VALIDATE_EVERY_EPOCHS=1 \
  RUN_DIR="$RUN_DIR" WANDB_MODE=disabled \
  python3 "$REPO/train.py"
