#!/usr/bin/env bash
# 같은 RUN_DIR 의 last_checkpoint.pth 에서 production 학습을 재개한다.
# train.py 가 checkpoint 의 split/anchor/GT 정합을 검사해 다른 run 이면 fail-fast 한다.
# 필수 env: DATASET_ROOT, ANCHOR_DIR, VAL_SCENARIOS, RUN_DIR (원래 run 과 동일해야 함)
source "$(dirname "$0")/_lib.sh"
REPO="$(repo_root)"
need_dir "$DATASET_ROOT"
: "${VAL_SCENARIOS:?VAL_SCENARIOS 지정 필요}"
: "${ANCHOR_DIR:?ANCHOR_DIR 지정 필요}"
: "${RUN_DIR:?RUN_DIR 지정 필요}"
need_file "$RUN_DIR/last_checkpoint.pth"
[ "${CONFIRM_PRODUCTION:-0}" = "1" ] || die "CONFIRM_PRODUCTION=1 없이는 재개하지 않습니다."
DEPTH="${USE_DENSE_DEPTH:-0}"; EPOCHS="${NUM_EPOCHS:-100}"
info "RESUME: $RUN_DIR/last_checkpoint.pth (depth=$DEPTH epochs=$EPOCHS)"
run env DATASET_ROOT="$DATASET_ROOT" VAL_SCENARIOS="$VAL_SCENARIOS" ANCHOR_DIR="$ANCHOR_DIR" \
  TRAIN_GT_VERSION=v3 VAL_GT_VERSION=v3 USE_DENSE_DEPTH="$DEPTH" SEED="${SEED:-0}" \
  RESUME=auto FORCE_REMAKE_KMEANS=0 NUM_EPOCHS="$EPOCHS" \
  RUN_DIR="$RUN_DIR" WANDB_MODE="${WANDB_MODE:-disabled}" \
  python3 "$REPO/train.py"
