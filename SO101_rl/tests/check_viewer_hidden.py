"""Finite viewer controls/render check inside an unmapped Xephyr parent."""
import os
import argparse
import signal
import numpy as np
from pathlib import Path
import subprocess
import time
import tkinter as tk
from PIL import ImageGrab

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group()
mode.add_argument('--capture-only',action='store_true');mode.add_argument('--jig-render-only',action='store_true');args=parser.parse_args()
out=ROOT/'outputs'/('viewer_camera_check' if args.capture_only else 'viewer_jig_render_check' if args.jig_render_only else 'viewer_ui_check');out.mkdir(parents=True,exist_ok=True)
for name in ('viewer_ready','viewer_captured','viewer_smoke.json','error.json'):
    (out/name).unlink(missing_ok=True)
parent=tk.Tk();parent.withdraw();parent.overrideredirect(True);parent.attributes('-alpha',0.)
parent.geometry('1600x1100-4000-4000');parent.deiconify();parent.lower();parent.update()
number=next(n for n in range(90,110) if not Path(f'/tmp/.X{n}-lock').exists() and not Path(f'/tmp/.X11-unix/X{n}').exists())
display=':'+str(number);child=None
log=(out/'xephyr.log').open('w')
server=subprocess.Popen(['Xephyr',display,'-parent',str(parent.winfo_id()),'-screen','1600x1100','-ac','-noreset','-nolisten','tcp'],stdout=log,stderr=log)
try:
    for _ in range(50):
        if server.poll() is not None:raise RuntimeError('Xephyr failed')
        if Path(f'/tmp/.X11-unix/X{number}').exists():break
        parent.update();time.sleep(.1)
    with (out/'viewer.log').open('w') as stream:
        child=subprocess.Popen(['bash',str(ROOT/'run.sh'),'--mode','view','--viewer-capture' if args.capture_only else '--viewer-jig-render' if args.jig_render_only else '--viewer-smoke','--output',str(out),
            '--kit_args','--/app/window/width=1500 --/app/window/height=1000 --/app/window/x=0 --/app/window/y=0'],
            env={**os.environ,'DISPLAY':display},stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+240;captured=False
        while child.poll() is None:
            parent.update()
            if (out/'viewer_ready').exists() and not captured:
                screenshot=ImageGrab.grab(xdisplay=display);screenshot.save(out/'viewer_ui.png')
                if args.capture_only and np.mean(np.asarray(screenshot.crop((80,180,1000,850)))>30)<.02:
                    raise RuntimeError('상단 카메라 렌더가 비어 있습니다.')
                (out/'viewer_captured').write_text('captured');captured=True
            if time.monotonic()>deadline:raise RuntimeError('Viewer check exceeded 240 seconds')
            time.sleep(.1)
        if child.returncode or not captured:raise RuntimeError('Viewer failed; see viewer.log')
    print('Hidden registered-camera rendering check passed.' if args.capture_only else 'Hidden jig restore/render capture completed.' if args.jig_render_only else
          'Hidden viewer check passed: play, pause, reset, jig correction, teaching controller and rendering.')
finally:
    if child and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=10)
        except subprocess.TimeoutExpired:pass
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        child.wait()
    server.terminate();server.wait(timeout=5);log.close();parent.destroy()
