#!/usr/bin/env python3
"""Read-only, bounded-rate OpenGL viewer in its own process (no motor imports)."""
import argparse
import http.client
import io
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import urllib.request

os.environ.setdefault('MUJOCO_GL','egl')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PIL import Image
from studio.pose_renderer import PoseRenderer
from studio.store import atomic_json

RECONNECT_LIMIT = 15.0


def main():
    p=argparse.ArgumentParser();p.add_argument('--server',required=True);p.add_argument('--fps',type=float,default=15)
    p.add_argument('--seconds',type=float,default=0);args=p.parse_args()
    if not 1<=args.fps<=30:raise ValueError('Display fps must be between 1 and 30')
    stopping=threading.Event()
    def stop(*_):
        stopping.set()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    out=ROOT/'outputs';out.mkdir(exist_ok=True);began=time.monotonic();last_status=0.;last_render=0.
    renderer=PoseRenderer();ready=False;lost_at=None;retry_delay=.25
    try:
        atomic_json(out/'sim_status.json',dict(state='loading',renderer='mujoco',time=time.time()))
        while not stopping.is_set():
            tick=time.monotonic()
            if args.seconds and tick-began>=args.seconds:break
            try:
                with urllib.request.urlopen(args.server+'/api/sim-state',timeout=1) as response:state=json.load(response)
            except (OSError, http.client.HTTPException, json.JSONDecodeError) as error:
                if stopping.is_set():break
                now=time.monotonic()
                if lost_at is None:lost_at=now
                if now-lost_at>=RECONNECT_LIMIT:
                    raise RuntimeError('자세 화면의 서버 연결을 15초 동안 복구하지 못했습니다.') from error
                # A reconnecting frame is never served as a fresh pose by the server.
                atomic_json(out/'sim_status.json',dict(state='reconnecting',renderer='mujoco',time=time.time(),
                    frames=renderer.frames,error=str(error)))
                stopping.wait(retry_delay);retry_delay=min(2.,retry_delay*2)
                continue
            if stopping.is_set():break
            recovered=lost_at is not None
            lost_at=None;retry_delay=.25
            stamp=time.perf_counter();pixels=renderer.render(state['display_angles'],state.get('display_robot','so101'))
            if pixels is not None:
                last_render=(time.perf_counter()-stamp)*1000
                buffer=io.BytesIO();Image.fromarray(pixels).save(buffer,format='JPEG',quality=85)
                temp=out/'sim.jpg.tmp';temp.write_bytes(buffer.getvalue());temp.replace(out/'sim.jpg');ready=True
            if pixels is not None or recovered or tick-last_status>=.5:
                atomic_json(out/'sim_status.json',dict(state='ready' if ready else 'loading',renderer='mujoco',time=time.time(),
                    frames=renderer.frames,source=state['display_source'],robot_type=state.get('display_robot','so101'),
                    joint_angles=state['display_angles'],render_ms=last_render,max_fps=args.fps,physics=False,object_tracking=False))
                last_status=tick
            stopping.wait(max(0,1/args.fps-(time.monotonic()-tick)))
        atomic_json(out/'sim_status.json',dict(state='stopped',renderer='mujoco',time=time.time(),frames=renderer.frames))
    except Exception as error:
        atomic_json(out/'sim_status.json',dict(state='error',renderer='mujoco',error=str(error),time=time.time()))
        raise
    finally:renderer.close()


if __name__=='__main__':main()
