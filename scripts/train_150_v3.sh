#!/usr/bin/env bash
# 150-scene production 학습(v3). ⚠️ 실제 장시간 학습을 시작한다.
# 사고 방지: CONFIRM_PRODUCTION=1 이 없으면 실행하지 않는다.
# 필수 env: DATASET_ROOT, ANCHOR_DIR(최종 split anchor), VAL_SCENARIOS(승인된 val), RUN_DIR
# 정책: primary=depth OFF(LiDAR-camera 동적 동기화 미해결). depth ON 은 위험 감수 하 실험용.
source "$(dirname "$0")/_lib.sh"
REPO="$(repo_root)"
need_dir "$DATASET_ROOT"
: "${VAL_SCENARIOS:?VAL_SCENARIOS 지정 필요}"
: "${ANCHOR_DIR:?ANCHOR_DIR(최종 split anchor) 지정 필요}"
: "${RUN_DIR:?RUN_DIR 지정 필요(run 격리)}"
need_file "$ANCHOR_DIR/anchor_kmeans_full.npy"
[ "${CONFIRM_PRODUCTION:-0}" = "1" ] || die "CONFIRM_PRODUCTION=1 없이는 production 학습을 시작하지 않습니다."
DEPTH="${USE_DENSE_DEPTH:-0}"     # 권장 기본 0
EPOCHS="${NUM_EPOCHS:-100}"
info "PRODUCTION train: val=[$VAL_SCENARIOS] anchor=$ANCHOR_DIR depth=$DEPTH epochs=$EPOCHS → $RUN_DIR"
run env DATASET_ROOT="$DATASET_ROOT" VAL_SCENARIOS="$VAL_SCENARIOS" ANCHOR_DIR="$ANCHOR_DIR" \
  TRAIN_GT_VERSION=v3 VAL_GT_VERSION=v3 USE_DENSE_DEPTH="$DEPTH" SEED="${SEED:-0}" \
  RESUME="${RESUME:-0}" FORCE_REMAKE_KMEANS=0 NUM_EPOCHS="$EPOCHS" \
  RUN_DIR="$RUN_DIR" WANDB_MODE="${WANDB_MODE:-disabled}" \
  python3 "$REPO/train.py"
