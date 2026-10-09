#!/usr/bin/env bash
set -Eeuo pipefail
OPENCV_APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OPENCV_PYTHON="${OPENCV_TEACH_PYTHON:-$HOME/venv/il/bin/python}"
exec "$OPENCV_PYTHON" "$OPENCV_APP_DIR/robot_setup.py" --profile "$OPENCV_APP_DIR/follower.json" "$@"
