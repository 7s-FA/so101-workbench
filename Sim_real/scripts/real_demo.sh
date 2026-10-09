#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
mkdir -p outputs/sim_real_demo
exec 9>outputs/sim_real_demo/run.lock
flock -n 9 || { echo "Sim_real 실행 또는 검사가 이미 진행 중입니다." >&2; exit 1; }
if ! "$project_root/.venv-io/bin/python" scripts/build_real_plan.py >outputs/sim_real_demo/startup_plan.log 2>&1; then
    echo '동작 계획 준비에 실패했습니다. outputs/sim_real_demo/startup_plan.log를 확인하세요.' >&2
    exit 1
fi
"$project_root/.venv-io/bin/python" scripts/real_verification.py
REAL_STATUS_FILE="$(mktemp)"
export REAL_STATUS_FILE
trap 'rm -f -- "$REAL_STATUS_FILE"' EXIT
"$project_root/.venv-io/bin/python" scripts/run_real_sim.py "$@"
if [[ "$(cat -- "$REAL_STATUS_FILE")" != ok ]]; then
    echo '시뮬레이션이 정상 완료되지 않았습니다. outputs/sim_real_demo/runtime_report.json을 확인하세요.' >&2
    exit 1
fi
