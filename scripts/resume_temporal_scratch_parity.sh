#!/usr/bin/env bash
# parity temporal scratch run (train_temporal_scratch_parity.sh) 전용 재개.
# 원 run 과 동일한 loss/optimizer/scheduler 계약을 명시적으로 전달하고,
# train.py 의 resume fail-fast 가 checkpoint 의 loss_contract/optim_contract/
# num_epochs/warmup 을 비교해 불일치 시 즉시 중단한다.
#
# 사용: RUN_DIR=runs/p2_scratch_parity_<date> ./scripts/resume_temporal_scratch_parity.sh
source "$(dirname "$0")/_lib.sh"
REPO="$(repo_root)"
need_dir "$DATASET_ROOT"
: "${VAL_SCENARIOS:?VAL_SCENARIOS 지정 필요}"
: "${ANCHOR_DIR:?ANCHOR_DIR 지정 필요}"
: "${RUN_DIR:?RUN_DIR 지정 필요(원 run 과 동일)}"
need_file "$RUN_DIR/last_checkpoint.pth"
[ "${CONFIRM_PRODUCTION:-0}" = "1" ] || die "CONFIRM_PRODUCTION=1 없이는 재개하지 않습니다."
[ -z "${TRANSFER_FROM_V10:-}" ] || die "TRANSFER_FROM_V10 을 해제하세요 (resume 과 상호배타)."
[ -z "${INIT_WEIGHTS:-}" ] || die "INIT_WEIGHTS 를 해제하세요 (resume 과 상호배타)."
# 중복 실행 방지: 같은 GPU/RUN_DIR 에 트레이너 2개가 뜨면 checkpoint 를 서로 덮어써
# 손상된다 (2026-08-02 실사고 — 수동 재개와 자동 재개가 겹쳐 2개 동시 실행됨).
if pgrep -f "python3 $REPO/train.py" > /dev/null; then
  die "train.py 가 이미 실행 중입니다 (PID: $(pgrep -f "python3 $REPO/train.py" | tr '\n' ' ')). 중복 실행 금지."
fi
EPOCHS="${NUM_EPOCHS:-30}"   # ⚠️ 원 run 과 동일해야 함 — cosine 길이 계약, 불일치 시 fail-fast

info "RESUME parity scratch: $RUN_DIR/last_checkpoint.pth (epochs=$EPOCHS)"
run env DATASET_ROOT="$DATASET_ROOT" VAL_SCENARIOS="$VAL_SCENARIOS" ANCHOR_DIR="$ANCHOR_DIR" \
  TRAIN_GT_VERSION=v3 VAL_GT_VERSION=v3 USE_DENSE_DEPTH="${USE_DENSE_DEPTH:-1}" SEED="${SEED:-0}" \
  USE_TEMPORAL_MEMORY=1 STREAMING_SAMPLER=1 TEMP_GNN_MODE=gated FILTER_VISIBLE=0 \
  BATCH_SIZE="${BATCH_SIZE:-4}" GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}" \
  NUM_WORKERS="${NUM_WORKERS:-4}" USE_AMP="${USE_AMP:-0}" ALLOW_TF32="${ALLOW_TF32:-1}" \
  RESUME=auto TRANSFER_FROM_V10= INIT_WEIGHTS= STOP_AFTER_UPDATES=0 \
  FORCE_REMAKE_KMEANS=0 NUM_EPOCHS="$EPOCHS" \
  LOSS_CONTRACT=parity OPTIM_CONTRACT=parity CLS_PRIOR_PROB=0.01 PREFLIGHT_UPDATES=500 \
  CHECKPOINT_EVERY_EPOCHS=5 EARLY_STOP_PATIENCE=100 \
  MAX_STEPS_PER_EPOCH=0 FAST_VAL_MAX_FRAMES=0 VALIDATE_EVERY_EPOCHS=1 \
  RUN_DIR="$RUN_DIR" WANDB_MODE=disabled \
  python3 "$REPO/train.py"
