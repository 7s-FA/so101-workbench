#!/usr/bin/env bash
set -eo pipefail
ARM_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ARM_PROJECT/scripts/factory.sh" --demo --no-runtime-geometry "$@"
