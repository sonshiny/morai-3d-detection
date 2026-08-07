#!/usr/bin/env bash
# Phase 2 최종 재학습: parity 계약 temporal perception detector — epoch 1부터 SCRATCH.
#
# 근거 (2026-07-31 합의): 공식 SparseDrive Stage-1 은 temporal=True 로 scratch 학습하며
# single-frame A0 선행 단계가 없다 (num_single_frame_decoder=1 은 모델 내부 처리 순서).
# repair pilot 에서 parity 계약이 temporal ON 으로 mAP 0.5365 달성 — 계약 유효성 검증 완료.
#
# 실행 계약:
#   LOSS_CONTRACT=parity, CLS_PRIOR_PROB=0.01 (공식 focal init), INIT_WEIGHTS 금지(scratch),
#   temporal+streaming+gated GNN, CHECKPOINT_EVERY_EPOCHS=5 (epoch 5/10/.../30 AP 평가용)
#
# ① preflight (300~500 update 점검, 장기 학습 전 필수):
#   CONFIRM_PRODUCTION=1 PREFLIGHT=1 RUN_DIR=runs/p2_scratch_preflight_<date> \
#     ./scripts/train_temporal_scratch_parity.sh
#   → preflight.csv 점검: NaN/Inf 없음, clip 발동이 지속 100% 아님, num_pos 정상,
#     bank/temp_gate 정상, 이후 본 학습과 다른 RUN_DIR 로 ② 실행.
# ② 본 학습 (30 epoch, epoch 20 중간 판정):
#   CONFIRM_PRODUCTION=1 RUN_DIR=runs/p2_scratch_parity_<date> \
#     ./scripts/train_temporal_scratch_parity.sh
source "$(dirname "$0")/_lib.sh"
REPO="$(repo_root)"
need_dir "$DATASET_ROOT"
: "${VAL_SCENARIOS:?VAL_SCENARIOS 지정 필요}"
: "${ANCHOR_DIR:?ANCHOR_DIR 지정 필요}"
: "${RUN_DIR:?RUN_DIR 지정 필요(신규 디렉터리)}"
need_file "$ANCHOR_DIR/anchor_kmeans_full.npy"
[ "${CONFIRM_PRODUCTION:-0}" = "1" ] || die "CONFIRM_PRODUCTION=1 없이는 시작하지 않습니다."
[ -e "$RUN_DIR" ] && die "RUN_DIR 이 이미 존재합니다: $RUN_DIR"
[ -z "${INIT_WEIGHTS:-}" ] || die "scratch 학습입니다 — INIT_WEIGHTS 를 지정하면 안 됩니다."
# env 상속 누수 방지: TRANSFER_FROM_V10 이 셸에 남아 있으면 train.py 가 transfer mode 로
# 진입해 기존 가중치를 로드한다. 아래 env 라인에서도 명시적으로 비운다(이중 방어).
[ -z "${TRANSFER_FROM_V10:-}" ] || die "scratch 학습입니다 — TRANSFER_FROM_V10 을 해제하세요."
if pgrep -f "python3 $REPO/train.py" > /dev/null; then
  die "train.py 가 이미 실행 중입니다 — 중복 실행 금지."
fi
[ -z "${RESUME:-}" ] || [ "${RESUME}" = "0" ] || die "신규 시작 스크립트입니다 — resume 은 resume_temporal_scratch_parity.sh 를 사용하세요."

if [ "${PREFLIGHT:-0}" = "1" ]; then
  # ⚠️ scheduler horizon 은 본 학습(30ep)과 동일하게 유지하고 STOP_AFTER_UPDATES 로만
  # 조기 종료한다. 1-epoch horizon 으로 줄이면 후반이 cosine 급감쇠 구간이 되어
  # grad 안정화와 LR 강제 하락을 구분할 수 없다 (2026-07-31 리뷰).
  EPOCHS="${NUM_EPOCHS:-30}"; MAX_STEPS=0; FAST_VAL=0
  STOP_AFTER="${STOP_AFTER_UPDATES:-1500}"
  PF_UPDATES="$STOP_AFTER"
  info "PREFLIGHT: 30ep-horizon, $STOP_AFTER update 에서 조기 종료 → $RUN_DIR"
else
  EPOCHS="${NUM_EPOCHS:-30}"; MAX_STEPS=0; FAST_VAL=0
  STOP_AFTER=0
  PF_UPDATES=500
  # 합의 계약은 30 epoch (preflight GO 판정도 30ep cosine 궤적 기준).
  # 셸에 남은 NUM_EPOCHS(예: 기존 production 의 100)가 조용히 상속되는 사고 방지 —
  # 다른 epoch 수는 ALLOW_NONSTANDARD_EPOCHS=1 로만 허용.
  if [ "$EPOCHS" != "30" ] && [ "${ALLOW_NONSTANDARD_EPOCHS:-0}" != "1" ]; then
    die "NUM_EPOCHS=$EPOCHS — 합의 계약은 30. 의도적이면 ALLOW_NONSTANDARD_EPOCHS=1 지정."
  fi
  info "SCRATCH parity temporal: epochs=$EPOCHS ckpt_every=5 → $RUN_DIR"
fi

run env DATASET_ROOT="$DATASET_ROOT" VAL_SCENARIOS="$VAL_SCENARIOS" ANCHOR_DIR="$ANCHOR_DIR" \
  TRAIN_GT_VERSION=v3 VAL_GT_VERSION=v3 USE_DENSE_DEPTH="${USE_DENSE_DEPTH:-1}" SEED="${SEED:-0}" \
  USE_TEMPORAL_MEMORY=1 STREAMING_SAMPLER=1 TEMP_GNN_MODE=gated FILTER_VISIBLE=0 \
  BATCH_SIZE="${BATCH_SIZE:-4}" GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}" \
  NUM_WORKERS="${NUM_WORKERS:-4}" USE_AMP="${USE_AMP:-0}" ALLOW_TF32="${ALLOW_TF32:-1}" \
  RESUME=0 TRANSFER_FROM_V10= FORCE_REMAKE_KMEANS=0 NUM_EPOCHS="$EPOCHS" \
  STOP_AFTER_UPDATES="$STOP_AFTER" \
  LOSS_CONTRACT=parity OPTIM_CONTRACT=parity CLS_PRIOR_PROB=0.01 PREFLIGHT_UPDATES="$PF_UPDATES" \
  CHECKPOINT_EVERY_EPOCHS=5 EARLY_STOP_PATIENCE=100 \
  MAX_STEPS_PER_EPOCH="$MAX_STEPS" FAST_VAL_MAX_FRAMES="$FAST_VAL" VALIDATE_EVERY_EPOCHS=1 \
  RUN_DIR="$RUN_DIR" WANDB_MODE=disabled \
  python3 "$REPO/train.py"
