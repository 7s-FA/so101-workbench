#!/usr/bin/env bash
set -eo pipefail
ARM_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export PYTHONPATH="$ARM_PROJECT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ARM_PROJECT"
exec "$ARM_PROJECT/.venv-io/bin/python" scripts/voltage_monitor.py "$@"
