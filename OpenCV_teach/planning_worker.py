"""Isolate CPU-heavy height IK from Tk and the motor heartbeat thread."""
from pathlib import Path
import json
import os
import subprocess
import sys


def compile_steps_isolated(steps,plan,station,observations,initial,**options):
    from vision_teaching import compile_steps
    if not any(t.get('upper_strategy')=='height_first' for t in plan.get('targets',{}).values()):
        return compile_steps(steps,plan,station,observations,initial,**options)
    payload={'args':[steps,plan,station,observations,initial],'options':options}
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    try:
        result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker'],
            input=json.dumps(payload,allow_nan=False),capture_output=True,text=True,env=env,timeout=60)
    except subprocess.TimeoutExpired as exc:raise ValueError('경로 계산 시간이 초과됐습니다. 로봇 이동은 시작하지 않았습니다.') from exc
    try:message=json.loads(result.stdout)
    except (ValueError,TypeError) as exc:raise ValueError('경로 계산 프로세스가 정상 결과를 반환하지 못했습니다.') from exc
    if result.returncode or 'error' in message:raise ValueError(message.get('error','경로 계산 실패'))
    return message['steps']


def main():
    from vision_teaching import compile_steps
    try:
        payload=json.load(sys.stdin)
        result=compile_steps(*payload['args'],**payload.get('options',{}))
        print(json.dumps({'steps':result},allow_nan=False));return 0
    except Exception as exc:
        print(json.dumps({'error':str(exc)},ensure_ascii=False));return 1

if __name__=='__main__':raise SystemExit(main())
