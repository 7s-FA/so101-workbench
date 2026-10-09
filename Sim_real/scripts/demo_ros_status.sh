#!/usr/bin/env bash
set -eo pipefail
ARM_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
exec "$ARM_PROJECT/.venv-io/bin/python" "$ARM_PROJECT/scripts/demo_ros_status.py"
