#!/usr/bin/env bash
# RTX 5070 등 실제 production PC에서 temporal 경로 처리량과 100-epoch budget을 실측한다.
# 학습을 시작하지 않으며, 짧은 1-epoch/MAX_STEPS preflight만 수행한다.
source "$(dirname "$0")/_lib.sh"
REPO="$(repo_root)"
need_cmd nvidia-smi
need_dir "$DATASET_ROOT"
: "${VAL_SCENARIOS:?VAL_SCENARIOS 지정 필요}"
: "${ANCHOR_DIR:?ANCHOR_DIR(최종 split anchor) 지정 필요}"
need_file "$ANCHOR_DIR/anchor_kmeans_full.npy"

RUN_DIR="${RUN_DIR:-$REPO/runs/temporal_benchmark_fp32_w0}"
STEPS="${STEPS:-400}"
VAL_MON="${VAL_MON:-200}"
EPOCHS_FOR_BUDGET="${EPOCHS_FOR_BUDGET:-100}"
BATCH="${BATCH_SIZE:-4}"
ACCUM="${GRAD_ACCUM_STEPS:-2}"
WORKERS="${NUM_WORKERS:-0}"
AMP="${USE_AMP:-0}"
TF32="${ALLOW_TF32:-0}"

[ "$STEPS" -ge 100 ] || die "timing 안정성을 위해 STEPS>=100 이어야 합니다."
[ $((BATCH * ACCUM)) -eq 8 ] || die "비교 시 effective batch=8을 유지하세요: BATCH_SIZE*GRAD_ACCUM_STEPS"
[ ! -e "$RUN_DIR/run_config.json" ] || die "기존 benchmark RUN_DIR 재사용 금지: $RUN_DIR"
mkdir -p "$RUN_DIR"

python3 "$REPO/scripts/check_gpu_environment.py" > "$RUN_DIR/torch_gpu_check.json"
info "GPU/driver inventory → $RUN_DIR/hardware.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit \
  --format=csv,noheader > "$RUN_DIR/hardware.txt"
nvidia-smi >> "$RUN_DIR/hardware.txt"

info "temporal benchmark: steps=$STEPS B=$BATCH accum=$ACCUM workers=$WORKERS amp=$AMP tf32=$TF32"
RUN_DIR="$RUN_DIR" STEPS="$STEPS" VAL_MON="$VAL_MON" DEPTH=0 \
BATCH_SIZE="$BATCH" GRAD_ACCUM_STEPS="$ACCUM" NUM_WORKERS="$WORKERS" \
USE_AMP="$AMP" ALLOW_TF32="$TF32" \
  "$REPO/scripts/preflight_150.sh"

python3 "$REPO/scripts/estimate_temporal_budget.py" \
  --run-dir "$RUN_DIR" --epochs "$EPOCHS_FOR_BUDGET" --cadences 1,5,10 \
  --output "$RUN_DIR/temporal_budget.json"
info "완료: $RUN_DIR/{torch_gpu_check.json,hardware.txt,run_config.json,throughput.jsonl,temporal_budget.json}"
