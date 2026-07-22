#!/usr/bin/env bash
# 전체 scene 의 labels_3d_v3 / scene_info_v3.json / timing_correction_report.json 생성.
# 원본(labels_3d, ego_pose, sync_log)은 수정하지 않는다(versioned 산출만).
# DRY_RUN=1 이면 --report-only(파일 생성 없이 통계만).
# 사용: DATASET_ROOT=/data/dataset scripts/build_v3_all.sh [--scen scenXX ...]
source "$(dirname "$0")/_lib.sh"
REPO="$(repo_root)"
need_dir "$DATASET_ROOT"
need_file "$REPO/correct_source_time.py"
EXTRA=()
[ "$DRY" = "1" ] && EXTRA+=(--report-only)
info "v3 생성: root=$DATASET_ROOT (dry=$DRY)"
run python3 "$REPO/correct_source_time.py" --root "$DATASET_ROOT" "${EXTRA[@]}" "$@"
# correction_valid=0 집계 fail-fast
info "correction_valid=0 집계 검사…"
python3 - "$DATASET_ROOT" <<'PY'
import json,os,sys,glob
root=sys.argv[1]; bad=0; tot=0
for rep in glob.glob(os.path.join(root,"scen*","timing_correction_report.json")):
    d=json.load(open(rep)); inv=int(d.get("n_correction_invalid",0)); tot+=1
    if inv: print(f"[WARN] {os.path.basename(os.path.dirname(rep))}: correction_valid=0 {inv}건"); bad+=inv
print(f"[v3] reports={tot} invalid_boxes={bad}")
sys.exit(1 if bad else 0)
PY
