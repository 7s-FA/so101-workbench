#!/usr/bin/env bash
set -eo pipefail
ARM_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ARM_PROJECT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ARM_PROJECT"
exec python3 scripts/monitor.py "$@"
