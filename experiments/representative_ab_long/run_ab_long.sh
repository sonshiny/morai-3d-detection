#!/usr/bin/env bash
# ================================================================================
# 장기(수렴) 대표 3-fold × 2-seed A/B runner (S2: subset150 monitoring + 종료후 full eval).
#   baseline = TRAIN v2 / VAL v3   ·   candidate = TRAIN v3 / VAL v3
#   primary = USE_DENSE_DEPTH=0, P1 OFF, matcher/anchor(K=900,seed42)/2.0m 불변.
#
# 파일럿(experiments/representative_ab, 300-update)과 **완전 분리된 root**. 파일럿 checkpoint 미접근.
#   $AB_RUN_ROOT/anchors/<fold>/                     anchor(seed 무관 → fold당 1회, 두 seed 공유)
#   $AB_RUN_ROOT/seed<SEED>/<fold>/init_seed<SEED>.pth
#   $AB_RUN_ROOT/seed<SEED>/<fold>/{baseline,candidate}/   (RUN_DIR)
#
# env (기본):
#   AB_RUN_ROOT=experiments/representative_ab_long · AB_SEED=0(0/1) · AB_EPOCHS=100
#   AB_MAX_STEPS=0(full loader) · AB_VAL_MON=subset150|150|full|0 (monitoring val frame 수)
#   AB_FOLDS="A B C" · AB_MIN_DISK_GB=50 · FORCE_RETRAIN=0 FORCE_REEVAL=0
# subset150 = 수렴/장애 monitoring 전용. **성능 판정은 종료후 full v3 validation(eval_run.py)**.
# 최종 eval: last_checkpoint.pth(=epoch100/final, primary) + best_model.pth(monitor-best, secondary),
#   둘 다 full v3 → eval_metrics.json / eval_metrics_monitorbest.json.
# ================================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO"
export WANDB_MODE=disabled DATASET_ROOT="${DATASET_ROOT:-$REPO/dataset}" MPLBACKEND=Agg

AB_RUN_ROOT="${AB_RUN_ROOT:-experiments/representative_ab_long}"
SEED="${AB_SEED:-0}"
NUM_EPOCHS="${AB_EPOCHS:-100}"
MAX_STEPS="${AB_MAX_STEPS:-0}"
FOLDS="${AB_FOLDS:-A B C}"
KMEANS_K="${AB_KMEANS_K:-900}"
ANCHOR_SEED=42
MIN_DISK_GB="${AB_MIN_DISK_GB:-50}"
STATUS="$AB_RUN_ROOT/RUN_STATUS.json"

# AB_VAL_MON 파싱: "subset150"/"150" → 150 ; "full"/"0"/"" → 0(full)
_raw_valmon="${AB_VAL_MON:-0}"
VALMON="$(printf '%s' "$_raw_valmon" | grep -oE '[0-9]+' | head -1)"
case "$_raw_valmon" in full|FULL) VALMON=0;; esac
VALMON="${VALMON:-0}"

CODE_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo nogit)"
CLEAN_WT="$(git diff --quiet 2>/dev/null && echo clean || echo dirty)"
[ "$SEED" = "0" ] || [ "$SEED" = "1" ] || { echo "FATAL: AB_SEED must be 0/1 (got $SEED)"; exit 2; }

sha() { sha256sum "$1" 2>/dev/null | cut -c1-16; }
now() { date +%Y-%m-%dT%H:%M:%S 2>/dev/null || echo unknown; }
free_gb() { df -P . | awk 'NR==2{printf "%d", $4/1024/1024}'; }

# ---- 원자적 RUN_STATUS.json 갱신 --------------------------------------------
status_set() { # $1=state $2=fold $3=role $4=note
  mkdir -p "$AB_RUN_ROOT"
  python3 - "$STATUS" "$1" "$SEED" "$2" "$3" "$4" "$CODE_COMMIT" "$CLEAN_WT" "$(now)" "$(free_gb)" "$NUM_EPOCHS" "$VALMON" <<'PY'
import json,sys,os
p,state,seed,fold,role,note,commit,cwt,ts,disk,ep,vm=sys.argv[1:13]
d={}
if os.path.exists(p):
    try: d=json.load(open(p))
    except Exception: d={}
d.update({"state":state,"seed":int(seed),"fold":fold,"role":role,"note":note,
          "code_commit":commit,"worktree":cwt,"updated":ts,"disk_free_gb":int(disk),
          "num_epochs":int(ep),"val_mon_frames":int(vm)})
hist=d.get("history",[]); hist.append({"ts":ts,"state":state,"fold":fold,"role":role,"note":note})
d["history"]=hist[-200:]
tmp=p+".tmp"; json.dump(d,open(tmp,"w"),indent=2); os.replace(tmp,p)
PY
}

fail() { echo "FATAL[$1]: $2"; status_set failed "${3:-}" "${4:-}" "$1: $2"; exit "${5:-1}"; }

fold_val() { case "$1" in A) echo scen05;; B) echo scen77;; C) echo scen144;; *) echo "";; esac; }

# ---- collision guard: 의도 config 와 run_meta.json 불일치 시 fail-fast ----------
guard_meta() { local dir="$1"; shift; local meta="$dir/run_meta.json"
  local want; want="$(printf '%s\n' "$@" | sort)"
  if [ -f "$meta" ]; then
    local have; have="$(python3 - "$meta" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); print("\n".join(sorted(f"{k}={v}" for k,v in d.items())))
PY
)"
    [ "$have" = "$want" ] || { echo "--- have ---"; echo "$have"; echo "--- want ---"; echo "$want"; \
      fail collision "run_meta 불일치(수동 확인; 삭제/덮어쓰기 금지): $meta" "" "" 3; }
  else mkdir -p "$dir"; python3 - "$meta" "$@" <<'PY'
import json,sys; json.dump(dict(a.split("=",1) for a in sys.argv[2:]),open(sys.argv[1],"w"),indent=2,sort_keys=True)
PY
  fi
}

# ---- 학습 후 NaN/Inf 감시(로그) ------------------------------------------------
check_finite() { # $1=train.log
  if grep -qiE 'loss[^0-9]*(nan|inf)|Val:[[:space:]]*(nan|inf)|nan\b|-?inf\b' "$1"; then
    grep -iE 'nan|inf' "$1" | tail -3; return 1; fi; return 0; }

# ---- eval 결과 join>0 확인(schema: val_frames/overall.softcalibrated[0.15]/p3_filtered.bins) ----
check_join() { # $1=metrics.json
  python3 - "$1" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
vf=m.get("val_frames",0)
ov=m.get("overall",{}).get("softcalibrated",{}).get("0.15",{})
gt=ov.get("tp",0)+ov.get("fn",0); pred=ov.get("tp",0)+ov.get("fp",0)
bins=m.get("p3_filtered",{}).get("bins",[])
p3gt=sum(b.get("n_gt",0) for b in bins); p3m=sum(b.get("n_matched",0) for b in bins)
print("JOIN val_frames=%s overall_gt=%s overall_pred=%s p3_n_gt=%s p3_matched=%s"%(vf,gt,pred,p3gt,p3m))
bad=[]
if not vf: bad.append("val_frames=0")
if gt<=0: bad.append("overall_GT=0")
if pred<=0: bad.append("overall_pred=0")
if p3gt<=0: bad.append("p3_n_gt=0")
if bad: sys.exit("FATAL join: "+", ".join(bad))
PY
}

run_one() { # $1=fold $2=val $3=role $4=train_gt $5=fdir $6=adir $7=afull $8=init
  local fold="$1" val="$2" role="$3" tgt="$4" fdir="$5" adir="$6" afull="$7" init="$8"
  local rdir="$fdir/$role"; mkdir -p "$rdir"
  local asha isha; asha="$(sha "$afull")"; isha="$(sha "$init")"
  guard_meta "$rdir" "seed=$SEED" "fold=$fold" "role=$role" "train_gt=$tgt" "val_scen=$val" \
    "val_gt=v3" "num_epochs=$NUM_EPOCHS" "max_steps=$MAX_STEPS" "val_mon=$VALMON" \
    "anchor_sha=$asha" "init_sha=$isha" "code_commit=$CODE_COMMIT" "kmeans_k=$KMEANS_K"
  status_set running "$fold" "$role" "train start ep=$NUM_EPOCHS valmon=$VALMON"
  local disk; disk="$(free_gb)"; [ "$disk" -ge "$MIN_DISK_GB" ] || fail disk "free ${disk}GB < ${MIN_DISK_GB}GB" "$fold" "$role" 5
  echo "=== [seed$SEED/$fold/$role] train_gt=$tgt val=$val ep=$NUM_EPOCHS max_steps=$MAX_STEPS valmon=$VALMON depth=OFF"
  echo "    anchor=$asha init=$isha commit=${CODE_COMMIT:0:10}($CLEAN_WT) disk=${disk}GB"
  local resume_arg
  if [ -f "$rdir/last_checkpoint.pth" ] && [ "${FORCE_RETRAIN:-0}" != "1" ]; then
    status_set resumed "$fold" "$role" "resume from last_checkpoint (guard passed)"; resume_arg=auto
  else resume_arg="$init"; fi
  RUN_DIR="$rdir" OUTPUT_CKPT_DIR="$rdir" \
  VAL_SCENARIOS="$val" TRAIN_GT_VERSION="$tgt" VAL_GT_VERSION=v3 \
  ANCHOR_DIR="$adir" ANCHOR_SEED="$ANCHOR_SEED" RESUME="$resume_arg" FORCE_REMAKE_KMEANS=0 \
  SEED="$SEED" USE_DENSE_DEPTH=0 USE_TEMPORAL_MEMORY=0 \
  NUM_EPOCHS="$NUM_EPOCHS" MAX_STEPS_PER_EPOCH="$MAX_STEPS" \
  FAST_VAL_MAX_FRAMES="$VALMON" VAL_EVERY_STEPS=0 EARLY_STOP_PATIENCE=999999 \
  python3 train.py > "$rdir/train.log" 2>&1
  local rc=$?
  [ $rc -eq 0 ] || fail train "train.py rc=$rc (see $rdir/train.log)" "$fold" "$role" $rc
  check_finite "$rdir/train.log" || fail nan "NaN/Inf in $rdir/train.log" "$fold" "$role" 6
  echo "[seed$SEED/$fold/$role] train done → full v3 eval (final + monitor-best)…"
  # eval_run.py 는 항상 <run_dir>/eval_metrics.json 에 기록 → 각 eval 후 named 파일로 보존.
  # primary: final/epoch100 = last_checkpoint.pth
  python3 experiments/representative_ab/eval_run.py --run-dir "$rdir" --val-scen "$val" \
    --val-gt v3 --anchor-full "$afull" --ckpt last_checkpoint.pth > "$rdir/eval_final.log" 2>&1 \
    || fail eval "final eval rc (see $rdir/eval_final.log)" "$fold" "$role" 4
  cp -f "$rdir/eval_metrics.json" "$rdir/eval_metrics_final.json"
  check_join "$rdir/eval_metrics_final.json" || fail join "final eval join=0" "$fold" "$role" 7
  # secondary: monitor-best = best_model.pth (있으면; 실패해도 non-fatal)
  if [ -f "$rdir/best_model.pth" ]; then
    if python3 experiments/representative_ab/eval_run.py --run-dir "$rdir" --val-scen "$val" \
        --val-gt v3 --anchor-full "$afull" --ckpt best_model.pth > "$rdir/eval_monitorbest.log" 2>&1; then
      cp -f "$rdir/eval_metrics.json" "$rdir/eval_metrics_monitorbest.json"
    else echo "[warn] monitor-best eval 실패(비치명): $rdir/eval_monitorbest.log"; fi
  fi
  # canonical eval_metrics.json = final(primary) 로 복원
  cp -f "$rdir/eval_metrics_final.json" "$rdir/eval_metrics.json"
  status_set completed "$fold" "$role" "train+eval done rc=0"
  echo "[seed$SEED/$fold/$role] DONE"; tail -1 "$rdir/eval_final.log" 2>/dev/null || true
}

do_fold() { local fold="$1"; local val; val="$(fold_val "$fold")"
  [ -n "$val" ] || fail cfg "unknown fold $fold" 2
  local foldname="fold${fold}_val_${val}"
  local adir="$AB_RUN_ROOT/anchors/$foldname"; local afull="$adir/anchor_kmeans_full.npy"
  local sroot="$AB_RUN_ROOT/seed$SEED/$foldname"; local init="$sroot/init_seed${SEED}.pth"
  mkdir -p "$adir" "$sroot"
  echo "########## seed$SEED FOLD $foldname (val=$val) ##########"
  # anchor: seed 무관 → fold당 1회(K=900 seed42). subset frame-list hash provenance.
  if [ -f "$afull" ] && [ -f "$adir/anchor_kmeans_meta.json" ]; then echo "[$foldname] anchor 존재 → skip."
  else
    echo "[$foldname] v3 anchor 생성(train=all-$val, K=$KMEANS_K seed=$ANCHOR_SEED)…"
    python3 make_kmeans.py --dataset-root "$DATASET_ROOT" --val-scenarios "$val" --gt-version v3 \
      --k "$KMEANS_K" --seed "$ANCHOR_SEED" --force --out "$adir/anchor_kmeans_xy.npy" \
      --full-out "$afull" --meta-out "$adir/anchor_kmeans_meta.json" \
      --split-out "$adir/dataset_split.json" > "$adir/anchor.log" 2>&1 \
      || fail anchor "make_kmeans rc (see $adir/anchor.log)" "$fold" "" 8
  fi
  if [ "$VALMON" -gt 0 ] && [ ! -f "$adir/monitor_subset.json" ]; then
    python3 experiments/representative_ab_long/compute_val_subset_hash.py "$val" "$VALMON" \
      "$adir/monitor_subset.json" v3 || fail subset "subset hash fail" "$fold" "" 9
    grep subset_hash "$adir/anchor.log" 2>/dev/null || true
  fi
  # shared init per seed
  if [ -f "$init" ]; then echo "[$foldname] seed$SEED init 존재 → skip."
  else
    echo "[$foldname] seed$SEED shared init build…"
    ANCHOR_FULL_FILE="$afull" ANCHOR_XY_FILE="$adir/anchor_kmeans_xy.npy" \
      python3 experiments/representative_ab/build_init.py "$init" "$SEED" \
      || fail init "build_init rc" "$fold" "" 10
  fi
  run_one "$fold" "$val" baseline v2 "$sroot" "$adir" "$afull" "$init"
  run_one "$fold" "$val" candidate v3 "$sroot" "$adir" "$afull" "$init"
}

echo "=== LONG A/B S2  seed=$SEED root=$AB_RUN_ROOT ep=$NUM_EPOCHS valmon=$VALMON folds=[$FOLDS] commit=${CODE_COMMIT:0:10}($CLEAN_WT) ==="
status_set running start start "seed$SEED launch"
for f in $FOLDS; do do_fold "$f"; done
status_set completed done done "seed$SEED all folds done"
echo "=== seed$SEED A/B DONE (folds: $FOLDS) ==="
