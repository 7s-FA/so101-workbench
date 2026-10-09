"""Read-only USB process: no target, torque, hold, calibration or EEPROM writes."""
import argparse
import json
import select
import signal
import sys
import time
from pathlib import Path
from .bus import Bus
from .observation import PoseObservation


class ReadOnlyBus(Bus):
    # Even an accidental use of an inherited mutation is refused.
    def write(self,*_):raise RuntimeError('Observation mode prohibits motor writes')
    def positions(self,*_):raise RuntimeError('Observation mode prohibits position commands')
    def enable(self,*_):raise RuntimeError('Observation mode prohibits torque changes')
    def hold(self):pass


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--port',required=True);p.add_argument('--mapping',type=Path,required=True)
    args=p.parse_args();cfg=json.loads(args.mapping.read_text());PoseObservation(cfg)
    if Path(args.port).resolve()!=Path(cfg['port']).resolve():raise ValueError('Mapping belongs to another device')
    bus=None
    def stop(*_):raise SystemExit(0)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        bus=ReadOnlyBus(args.port);info=bus.inspect()
        if any(m['torque']!=0 for m in info):raise RuntimeError('Torque is ON. Support the arm and release it before manual observation. No torque setting was changed.')
        signature=[{k:m[k] for k in ('id','model','mode','drive','offset')} for m in info]
        if signature!=cfg['motor_config']:raise RuntimeError('Motor configuration differs from the pose mapping')
        print(json.dumps({'ok':True,'state':'read_only','motors':info}),flush=True)
        while True:
            if not select.select([sys.stdin],[],[],2)[0]:continue
            line=sys.stdin.readline()
            if not line:break
            request=json.loads(line)
            if request.get('op')=='close':break
            if request.get('op')!='read':raise ValueError('Only read is allowed in observation mode')
            if not 0<=time.monotonic()-request['sent']<=.5:raise ValueError('Stale observation request')
            started=time.monotonic()
            sample=bus.sample()
            torque=[bus.read(m['id'],64,1) for m in info]
            if any(torque):raise RuntimeError('Torque became ON; manual observation stopped')
            print(json.dumps(dict(ok=True,state='read_only',sample=sample,torque=torque,
                received=time.monotonic(),read_ms=(time.monotonic()-started)*1000)),flush=True)
    except Exception as e:
        print(json.dumps({'ok':False,'error':str(e)}),flush=True);return 2
    finally:
        if bus:bus.close()
    return 0


if __name__=='__main__':sys.exit(main())
