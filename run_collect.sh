#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIM_WS_ROOT="${AIM_WS_ROOT:-/home/jang/aim_ws}"
RUNNER_SH="$AIM_WS_ROOT/aim_scenario_runner/run.sh"
DATASET_ROOT="${DATASET_ROOT:-/home/jang/dataset}"

if [ -f "$AIM_WS_ROOT/devel/setup.bash" ]; then
  # shellcheck disable=SC1091
  source "$AIM_WS_ROOT/devel/setup.bash"
fi

cd "$PROJECT_DIR"

collector_pid=""

cleanup() {
  if [ -n "$collector_pid" ] && kill -0 "$collector_pid" 2>/dev/null; then
    kill "$collector_pid" 2>/dev/null || true
    wait "$collector_pid" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

./venv/bin/python morai_3d_live.py --dataset_root "$DATASET_ROOT" &
collector_pid="$!"

for _ in $(seq 1 50); do
  if ! kill -0 "$collector_pid" 2>/dev/null; then
    wait "$collector_pid"
  fi

  if rostopic list 2>/dev/null | grep -qx "/dataset_control"; then
    break
  fi

  sleep 0.2
done

"$RUNNER_SH" "$@"
