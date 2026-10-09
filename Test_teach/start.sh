#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"
exec 9>"$project_root/.session.lock"
flock -n 9 || { echo 'Test_teach가 이미 실행 중입니다.' >&2; exit 1; }
export PYTHONPATH="$project_root/../Sim_real${PYTHONPATH:+:$PYTHONPATH}"
export TEST_TEACH_RUN_ID="$(date +%s%N)-$$"
python3 "$project_root/scripts/run.py" "$@"
mode=gui
plan=false
observed=false
legacy=false
for arg in "$@"; do
 [[ "$arg" != --legacy-omx ]] || legacy=true
 [[ "$arg" != --verify ]] || mode=trial
 [[ "$arg" != --preview ]] || mode=preview
 [[ "$arg" != --ui-smoke ]] || mode=ui_smoke
 [[ "$arg" != --follower-plan ]] || plan=true
 [[ "$arg" != --observed-pose ]] || observed=true
 [[ "$arg" != --observe-smoke ]] || mode=observe_smoke
 [[ "$arg" != --follower-read-smoke ]] || mode=follower_read_smoke
 [[ "$arg" != --mode-switch-smoke ]] || mode=mode_switch_smoke
done
if [[ "$mode" == trial && "$plan" == true ]]; then mode=follower_plan; fi
if [[ "$mode" == preview && "$observed" == true ]]; then mode=pose_comparison; fi
if [[ "$mode" != gui ]]; then
 python3 - "$mode" "$legacy" <<'PY'
import json,os,sys
from pathlib import Path
mode=sys.argv[1];legacy=sys.argv[2]=='true'
if not legacy:
    folder={'preview':'so101_preview','trial':'so101_trial','ui_smoke':'so101_ui_smoke'}[mode]
    d=json.loads((Path('outputs')/folder/'report.json').read_text())
    expected={'preview':'so101_layout_preview_passed','trial':'so101_physics_trial_passed','ui_smoke':'so101_ui_smoke_passed'}[mode]
    raise SystemExit(0 if d.get('run_id')==os.environ['TEST_TEACH_RUN_ID'] and d['status']==expected else 2)
d=json.loads((Path('outputs')/mode/'report.json').read_text())
expected='layout_preview_only' if mode in ('preview','pose_comparison') else 'simulation_trial_passed'
if mode=='observe_smoke':expected='observation_ui_smoke_passed'
if mode=='follower_read_smoke':expected='follower_read_smoke_passed'
if mode=='mode_switch_smoke':expected='mode_switch_smoke_passed'
raise SystemExit(0 if d.get('run_id')==os.environ['TEST_TEACH_RUN_ID'] and d['status']==expected else 2)
PY
fi
