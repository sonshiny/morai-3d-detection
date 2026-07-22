#!/usr/bin/env bash
# 전체 scene 의 depth_gt 생성 + 전수검사(generate_depth_gt.py --all).
# 주의: generate_depth_gt.py 는 preprocess_dataset.DATASET_ROOT(=<repo>/dataset)을 사용한다.
#   따라서 dataset 을 <repo>/dataset 에 두거나 symlink 하라(핸드오프 문서 참조).
# 사용: scripts/build_depth_all.sh   (DRY_RUN=1 이면 실행 명령만 출력)
source "$(dirname "$0")/_lib.sh"
REPO="$(repo_root)"
need_file "$REPO/generate_depth_gt.py"
if [ ! -d "$REPO/dataset" ]; then
  warn "<repo>/dataset 없음. generate_depth_gt.py 는 <repo>/dataset 를 기대한다."
  warn "  ln -s \"$DATASET_ROOT\" \"$REPO/dataset\"  (또는 dataset 을 그 위치에 배치)"
  die "dataset 배치/symlink 후 재실행"
fi
info "depth_gt 생성(--all) : <repo>/dataset"
run python3 "$REPO/generate_depth_gt.py" --all "$@"
info "생성 후 depth_gt 파일 수 = labels_3d_v2 프레임 × 3(cam) 인지 확인하세요(핸드오프 §10)."
