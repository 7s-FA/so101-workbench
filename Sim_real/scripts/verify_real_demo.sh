#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}" OPENBLAS_NUM_THREADS=1
mkdir -p outputs/sim_real_demo
exec 9>outputs/sim_real_demo/run.lock
flock -n 9 || { echo 'Sim_real 실행 또는 검사가 이미 진행 중입니다.' >&2; exit 1; }
exec "$project_root/.venv-io/bin/python" scripts/verify_real_demo.py
