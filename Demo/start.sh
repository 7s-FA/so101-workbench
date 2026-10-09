#!/usr/bin/env bash
set -euo pipefail
version_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$version_root/scripts/demo.sh" "$@"
