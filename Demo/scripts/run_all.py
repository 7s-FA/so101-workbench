#!/usr/bin/env python3
"""Start and stop this project's three processes as one foreground session."""
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = [('so101', ['scripts/voltage.sh', '--teleop']),
            ('omx', ['scripts/haptics.sh', '--enable-force']),
            ('simulation', ['scripts/assembly.sh', 'manual'])]

def stop_children(children):
    for sig, timeout in ((signal.SIGINT, 8), (signal.SIGTERM, 3), (signal.SIGKILL, 2)):
        running = [p for p in children if p.poll() is None]
        if not running: break
        for p in running:
            try: os.killpg(p.pid, sig)
            except ProcessLookupError: pass
        deadline = time.monotonic()+timeout
        for p in running:
            try: p.wait(timeout=max(0, deadline-time.monotonic()))
            except subprocess.TimeoutExpired: pass

def main():
    output = ROOT/'outputs'
    output.mkdir(exist_ok=True)
    with (output/'run_all.lock').open('w') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('이미 통합 실행 중입니다. 기존 터미널에서 Ctrl+C로 종료하세요.', flush=True)
            return 1
        stopped = False
        def halt(*_):
            nonlocal stopped
            stopped = True
        for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, halt)
        children, logs = [], []
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from arm_sim.monitor import ensure_monitor
        monitor=ensure_monitor()
        folder = output/'runs'/time.strftime('%Y%m%d_%H%M%S')
        folder.mkdir(parents=True, exist_ok=True)
        status = 0
        try:
            for name, command in COMMANDS:
                if stopped: break
                log = (folder/(name+'.log')).open('w')
                logs.append(log)
                children.append(subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, env={**os.environ, 'PYTHONUNBUFFERED':'1'}))
            print(f'시작 중 · 로그: {folder}\n통합 모니터: http://127.0.0.1:8767/\n종료: Ctrl+C (세 프로그램 함께 종료)', flush=True)
            while not stopped:
                for (name, _), child in zip(COMMANDS, children):
                    if child.poll() is not None:
                        print(f'{name} 종료됨 (코드 {child.returncode}). 나머지 프로그램도 종료합니다. 로그: {folder}', flush=True)
                        status = child.returncode or 0
                        stopped = True
                        break
                if not stopped: time.sleep(.2)
        finally:
            stop_children(children)
            if monitor is not None:monitor.close()
            for log in logs: log.close()
            print('통합 실행 종료.', flush=True)
        return status

if __name__ == '__main__':
    raise SystemExit(main())
