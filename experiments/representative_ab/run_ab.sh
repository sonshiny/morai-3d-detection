#!/usr/bin/env bash
# 대표 3-fold A/B: baseline(train v2 / val v3) vs candidate(train v3 / val v3).
# primary = USE_DENSE_DEPTH=0, P1 OFF. fold별 v3 anchor 1회 생성 + 공유 init.
# 고정·정합 budget(baseline/candidate 동일 update 수, early-stop 비활성).
set -euo pipefail
cd /home/autonav/projects/morai-3d-detection
export WANDB_MODE=disabled DATASET_ROOT=./dataset MPLBACKEND=Agg

AB=experiments/representative_ab
# 고정 budget(env override 가능): 3 epoch × 300 dl-step cap = 900 dl-step = 450 optimizer update/run.
NUM_EPOCHS="${AB_EPOCHS:-3}"
MAX_STEPS="${AB_MAX_STEPS:-300}"
VALMON="${AB_VAL_MON:-150}"          # 학습 중 monitoring val subset(최종 평가는 eval_run.py 전체)
SEED="${AB_SEED:-0}"
export EARLY_STOP_PATIENCE=999999    # 조기 종료 비활성 → baseline/candidate update 수 동일

run_one () {  # $1=fold $2=val_scen $3=role(baseline|candidate) $4=train_gt
  local fold="$1" val="$2" role="$3" tgt="$4"
  local fdir="$AB/$fold"
  local adir="$fdir/anchor"
  local afull="$adir/anchor_kmeans_full.npy"
  local rdir="$fdir/$role"
  local init="$fdir/init_seed${SEED}.pth"
  echo "=============================================================="
  echo "[$fold/$role] train_gt=$tgt val=$val  budget=${NUM_EPOCHS}ep x ${MAX_STEPS} dl-step (=$((NUM_EPOCHS*MAX_STEPS/2)) updates) seed=$SEED depth=OFF"
  echo "=============================================================="
  if [ -f "$rdir/last_checkpoint.pth" ] && [ "${FORCE_RETRAIN:-0}" != "1" ]; then
    echo "[$fold/$role] last_checkpoint.pth 존재 → 학습 skip(resumable)."
  else
    RUN_DIR="$rdir" OUTPUT_CKPT_DIR="$rdir" \
    VAL_SCENARIOS="$val" TRAIN_GT_VERSION="$tgt" VAL_GT_VERSION=v3 \
    ANCHOR_DIR="$adir" ANCHOR_SEED=42 RESUME="$init" FORCE_REMAKE_KMEANS=0 \
    SEED="$SEED" USE_DENSE_DEPTH=0 USE_TEMPORAL_MEMORY=0 \
    NUM_EPOCHS="$NUM_EPOCHS" MAX_STEPS_PER_EPOCH="$MAX_STEPS" \
    FAST_VAL_MAX_FRAMES="$VALMON" VAL_EVERY_STEPS=0 \
    python3 train.py > "$rdir.train.log" 2>&1
    echo "[$fold/$role] train done."
  fi
  if [ -f "$rdir/eval_metrics.json" ] && [ "${FORCE_REEVAL:-0}" != "1" ]; then
    echo "[$fold/$role] eval_metrics.json 존재 → eval skip."
  else
    echo "[$fold/$role] eval (full val)…"
    python3 "$AB/eval_run.py" --run-dir "$rdir" --val-scen "$val" --val-gt v3 \
       --anchor-full "$afull" --ckpt last_checkpoint.pth > "$rdir.eval.log" 2>&1
    tail -2 "$rdir.eval.log"
  fi
}

do_fold () {  # $1=fold $2=val_scen
  local fold="$1" val="$2"
  local fdir="$AB/$fold"; local adir="$fdir/anchor"
  mkdir -p "$adir" "$fdir/baseline" "$fdir/candidate"
  echo "########## FOLD $fold  val=$val ##########"
  if [ -f "$adir/anchor_kmeans_full.npy" ] && [ -f "$adir/anchor_kmeans_meta.json" ]; then
    echo "[$fold] anchor 존재 → 생성 skip."
  else
    echo "[$fold] generate v3 anchor (train = all - $val)…"
    python3 make_kmeans.py --dataset-root ./dataset --val-scenarios "$val" \
       --gt-version v3 --k 900 --seed 42 --force \
       --out "$adir/anchor_kmeans_xy.npy" --full-out "$adir/anchor_kmeans_full.npy" \
       --meta-out "$adir/anchor_kmeans_meta.json" --split-out "$adir/dataset_split.json" \
       > "$fdir/anchor.log" 2>&1
    grep -E "label_dir|split:|train boxes" "$fdir/anchor.log" | head -3
  fi
  if [ -f "$fdir/init_seed${SEED}.pth" ]; then
    echo "[$fold] init 존재 → 생성 skip."
  else
    echo "[$fold] build shared init (seed=$SEED, fold anchor)…"
    ANCHOR_FULL_FILE="$fdir/anchor/anchor_kmeans_full.npy" \
    ANCHOR_XY_FILE="$fdir/anchor/anchor_kmeans_xy.npy" \
       python3 "$AB/build_init.py" "$fdir/init_seed${SEED}.pth" "$SEED"
  fi
  run_one "$fold" "$val" baseline v2
  run_one "$fold" "$val" candidate v3
}

# fold C(가장 작음) 먼저 → 빠른 피드백. 이름은 val scene 으로 명확히.
do_fold foldC_val_scen144 scen144
do_fold foldB_val_scen77  scen77
do_fold foldA_val_scen05  scen05
echo "ALL 6 A/B RUNS DONE"
