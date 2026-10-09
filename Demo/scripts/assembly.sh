#!/usr/bin/env bash
set -eo pipefail
ARM_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ARM_MODE="${1:-manual}"
if [ "$#" -gt 0 ]; then shift; fi
case "$ARM_MODE" in
  manual) exec "$ARM_PROJECT/scripts/sim.sh" --scene assembly --live-cameras --mode manual "$@" ;;
  record)
    if [ "$#" -eq 0 ]; then echo 'Usage: assembly.sh record NEW_EPISODE.jsonl [options]' >&2; exit 2; fi
    ARM_EPISODE="$1"; shift
    exec "$ARM_PROJECT/scripts/sim.sh" --scene assembly --live-cameras --mode manual --record "$ARM_EPISODE" "$@" ;;
  replay)
    if [ "$#" -eq 0 ]; then echo 'Usage: assembly.sh replay EPISODE.jsonl [options]' >&2; exit 2; fi
    ARM_EPISODE="$1"; shift
    exec "$ARM_PROJECT/scripts/sim.sh" --scene assembly --live-cameras --mode replay --episode "$ARM_EPISODE" "$@" ;;
  *) echo 'Choose manual, record, or replay.' >&2; exit 2 ;;
esac
