#!/usr/bin/env bash
# 전체 split 짧은 학습 전 preflight(100~500 optimizer step). production 학습이 아니다.
# 전체 loader walk + anchor 정합 + NaN/Inf/OOM/throughput 확인용.
# 필수 env: DATASET_ROOT, ANCHOR_DIR(최종 split anchor), VAL_SCENARIOS(승인된 val)
# 선택: RUN_DIR(기본 runs/preflight_150), STEPS(dl-step, 기본 400), DEPTH(0/1, 기본 0)
# 이 wrapper는 production과 같은 SparseDrive-style temporal 경로를 강제한다.
source "$(dirname "$0")/_lib.sh"
REPO="$(repo_root)"
need_dir "$DATASET_ROOT"
: "${VAL_SCENARIOS:?VAL_SCENARIOS 지정 필요}"
: "${ANCHOR_DIR:?ANCHOR_DIR(최종 split anchor) 지정 필요}"
need_file "$ANCHOR_DIR/anchor_kmeans_full.npy"
RUN_DIR="${RUN_DIR:-$REPO/runs/preflight_150}"
STEPS="${STEPS:-400}"; DEPTH="${DEPTH:-0}"
TEMPORAL="${USE_TEMPORAL_MEMORY:-1}"
STREAMING="${STREAMING_SAMPLER:-1}"
TEMP_MODE="${TEMP_GNN_MODE:-gated}"
[ "$TEMPORAL" = "1" ] || die "preflight는 USE_TEMPORAL_MEMORY=1 이어야 합니다."
[ "$STREAMING" = "1" ] || die "preflight는 STREAMING_SAMPLER=1 이어야 합니다."
[ "$TEMP_MODE" = "gated" ] || die "production preflight TEMP_GNN_MODE는 gated 이어야 합니다."
info "preflight: val=[$VAL_SCENARIOS] anchor=$ANCHOR_DIR steps=$STEPS depth=$DEPTH temporal=gated → $RUN_DIR"
run env DATASET_ROOT="$DATASET_ROOT" VAL_SCENARIOS="$VAL_SCENARIOS" ANCHOR_DIR="$ANCHOR_DIR" \
  TRAIN_GT_VERSION=v3 VAL_GT_VERSION=v3 USE_DENSE_DEPTH="$DEPTH" SEED="${SEED:-0}" \
  USE_TEMPORAL_MEMORY=1 STREAMING_SAMPLER=1 TEMP_GNN_MODE=gated \
  BATCH_SIZE="${BATCH_SIZE:-4}" GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}" \
  NUM_WORKERS="${NUM_WORKERS:-0}" USE_AMP="${USE_AMP:-0}" ALLOW_TF32="${ALLOW_TF32:-0}" \
  RESUME=0 FORCE_REMAKE_KMEANS=0 EARLY_STOP_PATIENCE=999999 \
  NUM_EPOCHS=1 MAX_STEPS_PER_EPOCH="$STEPS" FAST_VAL_MAX_FRAMES="${VAL_MON:-200}" \
  VALIDATE_EVERY_EPOCHS=1 \
  RUN_DIR="$RUN_DIR" WANDB_MODE="${WANDB_MODE:-disabled}" \
  python3 "$REPO/train.py"
info "preflight 완료. run_config.json/throughput.jsonl 에 실제 temporal 처리량이 기록되었습니다."
