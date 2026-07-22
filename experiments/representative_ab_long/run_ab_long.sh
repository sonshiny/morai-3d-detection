#!/usr/bin/env bash
# ================================================================================
# 장기(수렴) 대표 3-fold × 2-seed A/B runner.
#   baseline = TRAIN v2 / VAL v3   ·   candidate = TRAIN v3 / VAL v3
#   primary = USE_DENSE_DEPTH=0, P1 OFF, matcher/anchor(K=900,seed42)/2.0m 불변.
#
# 파일럿(experiments/representative_ab, 300-update)과 **완전히 분리된 root** 를 쓴다.
# 파일럿 checkpoint 를 절대 읽지 않는다(별도 root + run_meta collision guard).
#
# 각 run 의 격리 경로:
#   $AB_RUN_ROOT/anchors/<fold>/                     (anchor: seed 무관 → fold 당 1회, 두 seed 공유)
#   $AB_RUN_ROOT/seed<SEED>/<fold>/init_seed<SEED>.pth
#   $AB_RUN_ROOT/seed<SEED>/<fold>/{baseline,candidate}/   (RUN_DIR)
#
# 필수/주요 env (기본값):
#   AB_RUN_ROOT=experiments/representative_ab_long
#   AB_SEED=0            (0 또는 1 — 모델/학습 seed만; anchor seed 는 항상 42)
#   AB_EPOCHS=100        (production 기본 schedule; 임의 축소 금지)
#   AB_MAX_STEPS=0       (0 = full train loader, epoch 을 자르지 않음)
#   AB_VAL_MON=0         (학습중 monitoring val frame 수; 0=full. ★비용 결정 레버★)
#   AB_FOLDS="A B C"     (실행할 fold)
#   FORCE_RETRAIN=0 FORCE_REEVAL=0   (1 이면 존재해도 재실행)
# 최종 full-val 평가는 eval_run.py(fast single-forward full-val)로 수행.
# ================================================================================
set -euo pipefail
cd /home/autonav/projects/morai-3d-detection
export WANDB_MODE=disabled DATASET_ROOT=./dataset MPLBACKEND=Agg

AB_RUN_ROOT="${AB_RUN_ROOT:-experiments/representative_ab_long}"
SEED="${AB_SEED:-0}"
NUM_EPOCHS="${AB_EPOCHS:-100}"
MAX_STEPS="${AB_MAX_STEPS:-0}"        # 0 → MAX_STEPS_PER_EPOCH=0 (full loader)
VALMON="${AB_VAL_MON:-0}"             # 0 → FAST_VAL_MAX_FRAMES=0 (full monitoring val)
FOLDS="${AB_FOLDS:-A B C}"
KMEANS_K="${AB_KMEANS_K:-900}"
ANCHOR_SEED=42
CODE_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo nogit)"
CLEAN_WT="$(git diff --quiet 2>/dev/null && echo clean || echo dirty)"

if [ "$SEED" != "0" ] && [ "$SEED" != "1" ]; then
  echo "FATAL: AB_SEED must be 0 or 1 (got $SEED)"; exit 2
fi

sha() { sha256sum "$1" 2>/dev/null | cut -c1-16; }

# fold 정의: name / val_scen / anchor train scenes(정보용)
fold_val() { case "$1" in A) echo scen05;; B) echo scen77;; C) echo scen144;; *) echo "";; esac; }

# ---- run_meta collision guard --------------------------------------------------
# 의도한 config 를 run_meta.json 과 대조. 불일치면 fail-fast(자동 덮어쓰기/resume 금지).
guard_meta() { # $1=dir  $2=key=val ...  (writes if absent, else must match)
  local dir="$1"; shift
  local meta="$dir/run_meta.json"
  local want; want="$(printf '%s\n' "$@" | sort)"
  if [ -f "$meta" ]; then
    local have; have="$(python3 - "$meta" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print("\n".join(sorted(f"{k}={v}" for k,v in d.items())))
PY
)"
    if [ "$have" != "$want" ]; then
      echo "FATAL collision: $meta 의 기존 config 와 의도가 불일치 → 중단(수동 확인 필요)."
      echo "--- have ---"; echo "$have"; echo "--- want ---"; echo "$want"; exit 3
    fi
  else
    mkdir -p "$dir"
    python3 - "$meta" "$@" <<'PY'
import json,sys
meta=sys.argv[1]; kv=dict(a.split("=",1) for a in sys.argv[2:])
json.dump(kv, open(meta,"w"), indent=2, sort_keys=True)
PY
  fi
}

run_one() { # $1=fold $2=val $3=role $4=train_gt $5=fdir $6=adir $7=afull $8=init
  local fold="$1" val="$2" role="$3" tgt="$4" fdir="$5" adir="$6" afull="$7" init="$8"
  local rdir="$fdir/$role"
  mkdir -p "$rdir"
  local asha isha; asha="$(sha "$afull")"; isha="$(sha "$init")"
  guard_meta "$rdir" "seed=$SEED" "fold=$fold" "role=$role" "train_gt=$tgt" \
    "val_scen=$val" "val_gt=v3" "num_epochs=$NUM_EPOCHS" "max_steps=$MAX_STEPS" \
    "anchor_sha=$asha" "init_sha=$isha" "code_commit=$CODE_COMMIT" "kmeans_k=$KMEANS_K"
  echo "=============================================================="
  echo "[seed$SEED/$fold/$role] train_gt=$tgt val=$val epochs=$NUM_EPOCHS max_steps=$MAX_STEPS valmon=$VALMON depth=OFF"
  echo "  anchor=$asha init=$isha commit=${CODE_COMMIT:0:10}($CLEAN_WT)"
  echo "=============================================================="
  if [ -f "$rdir/last_checkpoint.pth" ] && [ "${FORCE_RETRAIN:-0}" != "1" ]; then
    echo "[seed$SEED/$fold/$role] last_checkpoint.pth 존재 → resume/skip(guard 통과)."
    RESUME_ARG=auto
  else
    RESUME_ARG="$init"   # shared init 에서 새로 시작(파일럿 ckpt 아님)
  fi
  RUN_DIR="$rdir" OUTPUT_CKPT_DIR="$rdir" \
  VAL_SCENARIOS="$val" TRAIN_GT_VERSION="$tgt" VAL_GT_VERSION=v3 \
  ANCHOR_DIR="$adir" ANCHOR_SEED="$ANCHOR_SEED" RESUME="$RESUME_ARG" FORCE_REMAKE_KMEANS=0 \
  SEED="$SEED" USE_DENSE_DEPTH=0 USE_TEMPORAL_MEMORY=0 \
  NUM_EPOCHS="$NUM_EPOCHS" MAX_STEPS_PER_EPOCH="$MAX_STEPS" \
  FAST_VAL_MAX_FRAMES="$VALMON" VAL_EVERY_STEPS=0 EARLY_STOP_PATIENCE=999999 \
  python3 train.py > "$rdir/train.log" 2>&1
  echo "[seed$SEED/$fold/$role] train done → final full-val eval…"
  python3 experiments/representative_ab/eval_run.py --run-dir "$rdir" --val-scen "$val" \
    --val-gt v3 --anchor-full "$afull" --ckpt last_checkpoint.pth > "$rdir/eval.log" 2>&1 \
    || { echo "eval FAILED (see $rdir/eval.log)"; exit 4; }
  tail -1 "$rdir/eval.log" || true
}

do_fold() { # $1=fold-letter
  local fold="$1"; local val; val="$(fold_val "$fold")"
  [ -z "$val" ] && { echo "unknown fold $fold"; exit 2; }
  local foldname="fold${fold}_val_${val}"
  local adir="$AB_RUN_ROOT/anchors/$foldname"
  local afull="$adir/anchor_kmeans_full.npy"
  local sroot="$AB_RUN_ROOT/seed$SEED/$foldname"
  local init="$sroot/init_seed${SEED}.pth"
  mkdir -p "$adir" "$sroot"
  echo "########## seed$SEED FOLD $foldname (val=$val) ##########"
  # anchor: seed 무관 → fold 당 1회(seed0 이 만들고 seed1 이 재사용). K=900 seed42.
  if [ -f "$afull" ] && [ -f "$adir/anchor_kmeans_meta.json" ]; then
    echo "[$foldname] anchor 존재 → 생성 skip (seed 공유)."
  else
    echo "[$foldname] v3 anchor 생성 (train=all-$val, K=$KMEANS_K seed=$ANCHOR_SEED)…"
    python3 make_kmeans.py --dataset-root ./dataset --val-scenarios "$val" \
      --gt-version v3 --k "$KMEANS_K" --seed "$ANCHOR_SEED" --force \
      --out "$adir/anchor_kmeans_xy.npy" --full-out "$afull" \
      --meta-out "$adir/anchor_kmeans_meta.json" --split-out "$adir/dataset_split.json" \
      > "$adir/anchor.log" 2>&1
    grep -E "label_dir|split:|train boxes" "$adir/anchor.log" | head -3 || true
  fi
  # shared init per seed (fold anchor). baseline/candidate 가 공유.
  if [ -f "$init" ]; then
    echo "[$foldname] seed$SEED init 존재 → skip."
  else
    echo "[$foldname] seed$SEED shared init build…"
    ANCHOR_FULL_FILE="$afull" ANCHOR_XY_FILE="$adir/anchor_kmeans_xy.npy" \
      python3 experiments/representative_ab/build_init.py "$init" "$SEED"
  fi
  run_one "$fold" "$val" baseline v2 "$sroot" "$adir" "$afull" "$init"
  run_one "$fold" "$val" candidate v3 "$sroot" "$adir" "$afull" "$init"
}

echo "=== LONG A/B  seed=$SEED  root=$AB_RUN_ROOT  epochs=$NUM_EPOCHS  folds=[$FOLDS]  commit=${CODE_COMMIT:0:10}($CLEAN_WT) ==="
for f in $FOLDS; do do_fold "$f"; done
echo "=== seed$SEED A/B DONE (folds: $FOLDS) ==="
