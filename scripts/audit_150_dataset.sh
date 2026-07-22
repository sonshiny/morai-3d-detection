#!/usr/bin/env bash
# 150-scene 원본 데이터 사전 감사(report-only). 데이터를 절대 수정하지 않는다.
# 사용: DATASET_ROOT=/data/dataset scripts/audit_150_dataset.sh [--out audit.json]
source "$(dirname "$0")/_lib.sh"
REPO="$(repo_root)"
need_dir "$DATASET_ROOT"
OUT="${AUDIT_OUT:-$REPO/pretrain_verify/audit_150.json}"
info "감사 대상 DATASET_ROOT=$DATASET_ROOT → $OUT"
run python3 "$REPO/scripts/audit_150_dataset.py" --root "$DATASET_ROOT" --out "$OUT" "$@"
