#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"
exec 9>"$project_root/.session.lock"
flock -n 9 || { echo 'SO-101 모방 학습가 이미 실행 중입니다.' >&2; exit 1; }
exec python3 "$project_root/scripts/run.py" "$@"
