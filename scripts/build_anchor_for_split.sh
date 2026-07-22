#!/usr/bin/env bash
# 승인된 최종 split 의 v3 train scene 으로 K=900 anchor 를 1회 생성한다.
# ⚠️ 3-scene 대표 anchor 를 150-scene production 에 쓰지 마라 — 반드시 최종 split 으로 재생성.
# 필수 env:
#   DATASET_ROOT   : 데이터 루트
#   VAL_SCENARIOS  : 승인된 val(+test) scene 쉼표구분 (train = 전체 - 이 목록)
#   ANCHOR_DIR     : 출력 디렉터리(예: anchors/v3_full_split_k900)
# 선택: K(기본 900), SEED(기본 42), SPLIT_APPROVED(1 이어야 실행)
source "$(dirname "$0")/_lib.sh"
REPO="$(repo_root)"
need_dir "$DATASET_ROOT"
: "${VAL_SCENARIOS:?VAL_SCENARIOS(쉼표구분)를 지정하세요}"
: "${ANCHOR_DIR:?ANCHOR_DIR 출력 디렉터리를 지정하세요}"
[ "${SPLIT_APPROVED:-0}" = "1" ] || die "SPLIT_APPROVED=1 이 아니면 실행하지 않습니다(최종 split 승인 필요)."
K="${K:-900}"; SEED="${SEED:-42}"
guard_overwrite "$ANCHOR_DIR/anchor_kmeans_full.npy"
run mkdir -p "$ANCHOR_DIR"
VS=$(echo "$VAL_SCENARIOS" | tr ',' ' ')
info "anchor 생성: gt=v3 K=$K seed=$SEED val=[$VS] → $ANCHOR_DIR"
run python3 "$REPO/make_kmeans.py" --dataset-root "$DATASET_ROOT" --val-scenarios $VS \
   --gt-version v3 --k "$K" --seed "$SEED" --force \
   --out "$ANCHOR_DIR/anchor_kmeans_xy.npy" --full-out "$ANCHOR_DIR/anchor_kmeans_full.npy" \
   --meta-out "$ANCHOR_DIR/anchor_kmeans_meta.json" --split-out "$ANCHOR_DIR/dataset_split.json"
info "meta 의 train_scenarios/val_scenarios/input_label_sha256/anchor SHA 를 확인하세요."
info "학습 시 ANCHOR_DIR=$ANCHOR_DIR 로 지정하면 train.py 가 split/K/seed/GT/input-hash 정합을 fail-fast 검증합니다."
