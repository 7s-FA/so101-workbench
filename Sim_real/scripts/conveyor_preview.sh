#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"
"$project_root/.venv-io/bin/python" "$project_root/scripts/prepare_conveyor_assets.py"
mkdir -p "$project_root/outputs/conveyor_preview_latest"
# Keep a blocked preview reviewable, but never reuse a stale pass report.
if ! "$project_root/.venv-io/bin/python" "$project_root/scripts/check_conveyor_hardware.py"; then
    echo "기구 검사 미통과: 이미지는 확인용으로 생성하며 합격으로 표시하지 않습니다." >&2
fi
if ! "$project_root/.venv-io/bin/python" "$project_root/scripts/check_conveyor_floorplan.py"; then
    echo "평면 배치 검사 미통과: 동선·회전 공간 보고서를 확인하세요." >&2
fi
if ! "$project_root/.venv-io/bin/python" "$project_root/scripts/check_overhead_assembly.py"; then
    echo "카메라 연결부 검사 미통과: 확인용 이미지만 생성합니다." >&2
fi
PREVIEW_STATUS_FILE="$(mktemp)"
export PREVIEW_STATUS_FILE
trap 'rm -f -- "$PREVIEW_STATUS_FILE"' EXIT
/path/to/isaac/env_isaaclab/bin/python "$project_root/scripts/conveyor_preview.py" --headless --output "$project_root/outputs/conveyor_preview_latest" "$@"

if [[ "$(cat -- "$PREVIEW_STATUS_FILE")" != ok ]]; then
    echo '새 장면 생성에 실패했습니다. 이전 이미지를 새 결과로 사용하지 마세요.' >&2
    exit 1
fi
