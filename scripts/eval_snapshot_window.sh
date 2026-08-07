#!/usr/bin/env bash
# 학습 일시중단 → 스냅샷 AP 평가(순차) → 학습 재개.
#
# 근거 (2026-08-01 리뷰): RTX 5070 12GB 에서 학습(8.7GB)과 full 평가(~2GB+context)의
# 동시 실행은 OOM 위험 → epoch checkpoint 저장 직후 학습을 종료하고 순차 평가 후
# resume 한다. 6일 학습에서 회당 ~50분의 중단은 안정성과 맞바꿀 가치가 있다.
#
# 사용: EPOCH=5 RUN_DIR=runs/p2_scratch_parity_20260801_30ep ./scripts/eval_snapshot_window.sh
# 필수 env: DATASET_ROOT, ANCHOR_DIR, VAL_SCENARIOS, RUN_DIR, EPOCH
source "$(dirname "$0")/_lib.sh"
REPO="$(repo_root)"
need_dir "$DATASET_ROOT"
: "${VAL_SCENARIOS:?VAL_SCENARIOS 지정 필요}"
: "${ANCHOR_DIR:?ANCHOR_DIR 지정 필요}"
: "${RUN_DIR:?RUN_DIR 지정 필요}"
: "${EPOCH:?EPOCH 지정 필요 (예: 5)}"
CKPT="$RUN_DIR/checkpoint_epoch$EPOCH.pth"
need_file "$CKPT"
need_file "$RUN_DIR/last_checkpoint.pth"

# ① 학습 프로세스 종료 (epoch$EPOCH 은 이미 저장돼 있으므로 손실은 진행 중 epoch 의 앞부분뿐)
if pgrep -f "python3 $REPO/train.py" > /dev/null; then
  info "학습 일시중단 (resume 지점: last_checkpoint)"
  pkill -f "python3 $REPO/train.py" || true
  for _ in $(seq 1 30); do pgrep -f "python3 $REPO/train.py" > /dev/null || break; sleep 2; done
  pgrep -f "python3 $REPO/train.py" > /dev/null && die "학습 프로세스가 종료되지 않았습니다."
else
  info "실행 중인 학습 없음 → 평가만 수행"
fi

# ② 순차 평가: subset(preflight 0.157 과 동일 3,000 frames) → full(판정표)
mkdir -p "$RUN_DIR/ap_eval"
info "subset 평가 (3,000 frames)"
run python "$REPO/evaluate_ap.py" --run-dir "$RUN_DIR" --ckpt "checkpoint_epoch$EPOCH.pth" \
  --tag "ep${EPOCH}_sub3k" --max-frames 3000 --no-diag --num-workers 4 \
  > "$RUN_DIR/ap_eval/ep${EPOCH}_sub3k.log" 2>&1 || die "subset 평가 실패"
info "full 평가 (26,503 frames + 진단)"
run python "$REPO/evaluate_ap.py" --run-dir "$RUN_DIR" --ckpt "checkpoint_epoch$EPOCH.pth" \
  --tag "ep${EPOCH}_full" --num-workers 4 \
  > "$RUN_DIR/ap_eval/ep${EPOCH}_full.log" 2>&1 || die "full 평가 실패"

# ③ 학습 재개 — setsid 로 완전 분리(터미널/세션 종료에도 생존).
#    AUTO_RESUME=0 이면 재개하지 않고 중단 상태로 종료(판정 후 사람이 결정).
if [ "${AUTO_RESUME:-1}" != "1" ]; then
  info "AUTO_RESUME=0 → 학습 재개하지 않음. 평가 결과: $RUN_DIR/ap_eval/ep${EPOCH}_{sub3k,full}.json"
  info "재개 시: CONFIRM_PRODUCTION=1 RUN_DIR=$RUN_DIR (+ 데이터 env) ./scripts/resume_temporal_scratch_parity.sh"
  exit 0
fi
info "학습 재개 (resume_temporal_scratch_parity.sh)"
setsid env DATASET_ROOT="$DATASET_ROOT" ANCHOR_DIR="$ANCHOR_DIR" VAL_SCENARIOS="$VAL_SCENARIOS" \
  CONFIRM_PRODUCTION=1 RUN_DIR="$RUN_DIR" PYTHONUNBUFFERED=1 \
  nohup "$REPO/scripts/resume_temporal_scratch_parity.sh" \
  >> "$RUN_DIR.log" 2>&1 < /dev/null &
sleep 20
pgrep -f "python3 $REPO/train.py" > /dev/null || die "재개 실패 — $RUN_DIR.log 확인 필요"
info "재개 확인 완료. 평가 결과: $RUN_DIR/ap_eval/ep${EPOCH}_{sub3k,full}.json"
