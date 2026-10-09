#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_python="${SO101_RL_PYTHON:-/path/to/isaac/env_isaaclab/bin/python}"
if [[ ! -x "$task_python" ]]; then
  echo "Isaac Python을 찾을 수 없습니다: $task_python" >&2
  exit 1
fi
export PYTHONUNBUFFERED=1
task_status="$(mktemp /tmp/so101-rl-status.XXXXXXXX)"
trap 'rm -f -- "$task_status"' EXIT
export SO101_RL_STATUS="$task_status"
# Kit's immediate shutdown can mask a Python exception with process status 0.
# Require our own completion marker after artifacts have been persisted.
task_code=0
"$task_python" "$task_root/run.py" "$@" || task_code=$?
if [[ "$task_code" -ne 0 ]]; then exit "$task_code"; fi
if [[ "$(cat "$task_status")" != complete ]]; then
  echo 'SO101 RL 실행이 완료되지 않았습니다. 출력 폴더의 error.json 또는 실행 로그를 확인하세요.' >&2
  exit 1
fi
