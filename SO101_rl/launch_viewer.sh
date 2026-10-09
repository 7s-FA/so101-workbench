#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$task_root"
exec 9>"$task_root/.viewer.lock"
if ! flock -n 9; then
  zenity --info --title='완제품 B 시뮬' --text='시뮬이 이미 실행 중이거나 준비 중입니다.' --width=360
  exit 0
fi
task_output="$task_root/outputs/viewer_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$task_output"
notify-send '완제품 B 시뮬' 'Isaac Sim 장면을 준비합니다. 잠시 후 시뮬 창이 열립니다.' 2>/dev/null || true
if ! bash "$task_root/run.sh" --mode view --output "$task_output" >"$task_output/launch.log" 2>&1; then
  zenity --error --title='완제품 B 시뮬 실행 실패' --text="시뮬을 시작하지 못했습니다.\n실행 기록: $task_output/launch.log" --width=520
  exit 1
fi
