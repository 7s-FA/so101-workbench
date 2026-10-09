#!/usr/bin/env bash
set -Eeuo pipefail
OPENCV_APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OPENCV_PYTHON="${OPENCV_TEACH_PYTHON:-$HOME/venv/il/bin/python}"
[[ -x "$OPENCV_PYTHON" ]] || { echo 'il 환경이 필요합니다: ~/venv/il' >&2; exit 1; }
"$OPENCV_PYTHON" -m pip install -r "$OPENCV_APP_DIR/requirements-il.txt"
"$OPENCV_PYTHON" -m pip check
"$OPENCV_PYTHON" -c 'import tkinter, cv2, mujoco, scipy; from importlib.metadata import version; assert version("lerobot") in ("0.4.4", "0.6.1"); print("il GUI 환경 확인 완료")'

"$OPENCV_PYTHON" "$OPENCV_APP_DIR/install_app.py"
