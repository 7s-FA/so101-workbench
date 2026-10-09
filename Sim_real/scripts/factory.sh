#!/usr/bin/env bash
set -eo pipefail
ARM_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ARM_PROJECT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ARM_PROJECT"
for option in "$@"; do
    if [[ "$option" == --help || "$option" == -h ]]; then
        exec "${ISAAC_PYTHON:-/path/to/isaac/env_isaaclab/bin/python}" scripts/factory_sim.py "$@"
    fi
done
# Isaac may close the host with exit code zero after a Python exception.
# A private per-run marker prevents that exception from becoming a success.
FACTORY_STATUS_FILE="$(mktemp)"
export FACTORY_STATUS_FILE
trap 'rm -f -- "$FACTORY_STATUS_FILE"' EXIT
run_status=0
"${ISAAC_PYTHON:-/path/to/isaac/env_isaaclab/bin/python}" scripts/factory_sim.py "$@" || run_status=$?
if [[ $run_status -ne 0 ]]; then exit "$run_status"; fi
if [[ "$(cat -- "$FACTORY_STATUS_FILE")" != ok ]]; then
    echo '시뮬레이션이 정상 완료되지 않았습니다. 로그와 최신 report.json을 확인하세요.' >&2
    exit 1
fi
