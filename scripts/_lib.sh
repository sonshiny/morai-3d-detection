#!/usr/bin/env bash
# 공통 헬퍼. 각 wrapper 가 source 한다. 절대경로 하드코딩 금지 — 전부 env 기반.
set -euo pipefail

: "${DATASET_ROOT:?DATASET_ROOT 를 지정하세요 (예: export DATASET_ROOT=/data/dataset)}"

die()  { echo "[FATAL] $*" >&2; exit 1; }
info() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*" >&2; }

need_dir()  { [ -d "$1" ] || die "디렉터리 없음: $1"; }
need_file() { [ -f "$1" ] || die "파일 없음: $1"; }
need_cmd()  { command -v "$1" >/dev/null 2>&1 || die "명령 없음: $1"; }

# destructive overwrite 방지: 이미 있으면 FORCE=1 아닐 때 중단
guard_overwrite() {  # $1 = path
  if [ -e "$1" ] && [ "${FORCE:-0}" != "1" ]; then
    die "이미 존재: $1  (덮어쓰려면 FORCE=1)"
  fi
}

DRY="${DRY_RUN:-0}"
run() {  # dry-run 지원
  if [ "$DRY" = "1" ]; then echo "[DRY] $*"; else echo "[RUN] $*"; "$@"; fi
}

repo_root() { cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd; }
