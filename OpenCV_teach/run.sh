#!/usr/bin/env bash
set -Eeuo pipefail
unset PYTHONPATH PYTHONHOME
OPENCV_APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OPENCV_PYTHON="${OPENCV_TEACH_PYTHON:-$HOME/venv/il/bin/python}"
[[ -x "$OPENCV_PYTHON" ]] || { echo 'il 환경을 찾을 수 없습니다: ~/venv/il' >&2; exit 1; }
if [[ -f "$(dirname -- "$OPENCV_PYTHON")/activate" ]]; then
    source "$(dirname -- "$OPENCV_PYTHON")/activate"
fi
cd "$OPENCV_APP_DIR"
if [[ "${1:-}" == --calibrate ]]; then
    shift
    exec "$OPENCV_PYTHON" "$OPENCV_APP_DIR/robot_setup.py" --profile "$OPENCV_APP_DIR/follower.json" --calibrate follower "$@"
fi
if [[ "${1:-}" == --pair ]]; then
    shift
    exec "$OPENCV_PYTHON" "$OPENCV_APP_DIR/teach_left_arm.py" --profile "$OPENCV_APP_DIR/robot.json" "$@"
fi
if [[ $# -eq 0 ]]; then
    exec "$OPENCV_PYTHON" "$OPENCV_APP_DIR/teach_left_arm.py" --choose-mode
fi
exec "$OPENCV_PYTHON" "$OPENCV_APP_DIR/teach_left_arm.py" --follower-only --profile "$OPENCV_APP_DIR/follower.json" "$@"
