#!/usr/bin/env bash
set -Eeuo pipefail
OPENCV_APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$OPENCV_APP_DIR/install.sh"
