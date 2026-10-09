#!/usr/bin/env bash
set -euo pipefail
teach_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$teach_root/../Sim_real/.venv-io/bin/python" "$teach_root/scripts/torque_off.py" "$@"
