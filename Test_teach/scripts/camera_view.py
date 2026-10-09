#!/usr/bin/env python3
"""Explicit UVC device, 15 FPS preview, no recording/server or motor access."""
import argparse
import json
import os
import signal
import time
from pathlib import Path
import cv2

p=argparse.ArgumentParser()
p.add_argument('--device',required=True)
p.add_argument('--parent',type=int)
p.add_argument('--status-file',type=Path)
p.add_argument('--seconds',type=float,default=0)
p.add_argument('--test-file',action='store_true',help='Use a local video file for preview verification')
args=p.parse_args()
if not args.test_file and not args.device.startswith('/dev/'):
    p.error('Use an explicit /dev/v4l/by-id/... or /dev/videoN capture device')
def status(state, **values):
    if args.status_file:
        args.status_file.parent.mkdir(parents=True,exist_ok=True)
        temporary=args.status_file.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(state=state,device=args.device,pid=os.getpid(),updated_at=time.time(),**values)))
        temporary.replace(args.status_file)

running=True
status('connecting')
def stop(*_):
    global running
    running=False
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
cap=cv2.VideoCapture(args.device,cv2.CAP_ANY if args.test_file else cv2.CAP_V4L2)
try:
    if not cap.isOpened():raise RuntimeError(f'Cannot open camera: {args.device}. Check USB and whether another app is using it.')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    cap.set(cv2.CAP_PROP_FPS,15);cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    title='OMX gripper camera - LIVE (15 FPS preview)'
    cv2.namedWindow(title,cv2.WINDOW_NORMAL);cv2.resizeWindow(title,800,600)
    started=time.monotonic();frames=0;last_status=0.
    while running and (not args.seconds or time.monotonic()-started<args.seconds):
        if args.parent and os.getppid()!=args.parent:break
        tick=time.monotonic();ok,frame=cap.read()
        if not ok:
            raise RuntimeError('Camera frame read failed. Check USB and use Restart camera.')
        cv2.putText(frame,'LIVE | '+time.strftime('%H:%M:%S'),(12,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,0),2)
        cv2.imshow(title,frame);frames+=1
        if frames==1:
            cv2.setWindowProperty(title,cv2.WND_PROP_TOPMOST,1)
            cv2.setWindowProperty(title,cv2.WND_PROP_TOPMOST,0)
        if time.monotonic()-last_status>.25:
            status('live',frames=frames,fps=frames/max(.001,time.monotonic()-started),width=frame.shape[1],height=frame.shape[0])
            last_status=time.monotonic()
        if cv2.waitKey(1)&255 in (27,ord('q')):break
        if cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:break
        time.sleep(max(0,1/15-(time.monotonic()-tick)))
    print(f'CAMERA_PREVIEW frames={frames}',flush=True)
    status('closed',frames=frames)
except Exception as error:
    status('error',message=str(error))
    raise
finally:
    cap.release();cv2.destroyAllWindows()
