#!/usr/bin/env bash
set -euo pipefail
teach_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$teach_root/config/follower.json" ]]; then
  exec "$teach_root/start.sh" --legacy-omx --follower "$@"
fi
exec "$teach_root/start.sh" --legacy-omx --follower --visual-confirmed \
  --calibration "$teach_root/config/follower_visual_checked.json" "$@"
