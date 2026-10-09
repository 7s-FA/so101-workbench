#!/usr/bin/env python3
"""Keep the complete simulator log while showing concise console status/errors."""
import os,re,signal,subprocess,sys
from pathlib import Path

class ConsoleFilter:
    def __init__(self,verbose=False,verification=False):
        self.verbose=verbose;self.verification=verification;self.traceback=False
    def feed(self,line):
        clean=re.sub(r'\x1b\[[0-9;]*m','',line)
        if self.verbose:
            print(line,end='',flush=True);return
        if 'SIM REAL READY ' in clean:
            url=clean.split('SIM REAL READY ',1)[1].split()[0]
            print('검증용 시뮬레이션 준비 완료' if self.verification else f'준비 완료 · {url} 에서 시연 시작을 누르세요.',flush=True)
        elif 'SIM REAL COMPLETE ' in clean:
            print('시뮬레이션 실행을 마쳤습니다.',flush=True)
        elif 'Traceback (most recent call last)' in clean:
            self.traceback=True;print(clean,end='',flush=True)
        elif self.traceback or re.search(r'\[(?:Error|Fatal)\]|\b(?:[\w.]*Error|[\w.]*Exception):',clean,re.I):
            print(clean,end='',flush=True)
            if not clean.strip() or re.match(r'^[\w.]+(?:Error|Exception):',clean):self.traceback=False
        elif re.search(r'\b(?:Warning|\w*Warning)\b',clean,re.I):
            return

def run(command,log_path,verbose=False):
    display=ConsoleFilter(verbose,'--verify' in command)
    child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',bufsize=1,start_new_session=True,env={**os.environ,'PYTHONUNBUFFERED':'1'})
    def stop(signum,_frame):
        if child.poll() is None:
            try:os.killpg(child.pid,signum)
            except ProcessLookupError:pass
    previous={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        with Path(log_path).open('w',encoding='utf-8') as log:
            for line in child.stdout:
                log.write(line);log.flush();display.feed(line)
        code=child.wait()
        if code:print(f'시뮬레이션이 종료 코드 {code}로 끝났습니다. 상세 로그: {log_path}',file=sys.stderr)
        return code if code>=0 else 128-code
    finally:
        child.stdout.close()
        if child.poll() is None:
            stop(signal.SIGTERM,None);child.wait()
        for s,handler in previous.items():signal.signal(s,handler)

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1];args=sys.argv[1:];verbose='--show-warnings' in args
    args=[a for a in args if a!='--show-warnings'];log=root/'outputs/sim_real_demo/terminal.log';log.parent.mkdir(parents=True,exist_ok=True)
    print(f'시뮬레이션 준비 중입니다. 상세 로그: {log}',flush=True)
    raise SystemExit(run(['/path/to/isaac/env_isaaclab/bin/python',str(root/'scripts/real_sim.py'),'--headless',*args],log,verbose))
