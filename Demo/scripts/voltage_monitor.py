#!/usr/bin/env python3
"""Local SO-101 voltage dashboard. Serial operations are exclusively READ."""
import argparse
import csv
import fcntl
import json
import signal
import subprocess
import termios
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import scservo_sdk as sdk
from arm_sim.common import ROOT
from arm_sim.voltage import VoltageState

parser=argparse.ArgumentParser()
parser.add_argument('--port',default='/dev/serial/by-id/usb-YOUR_ROBOT-if00')
parser.add_argument('--http-port',type=int,default=8765)
parser.add_argument('--rate',type=float,default=60.)
parser.add_argument('--teleop',action='store_true',help='Forward healthy SO-101 positions to simulation with automatic HOLD/recovery')
args=parser.parse_args()
if not 1<=args.rate<=120:parser.error('--rate must be between 1 and 120')
state=VoltageState();lock=threading.RLock();stop=threading.Event()
meta={'connected':False,'message':'장치 연결 대기','rate':0.,'requested_rate':args.rate,'port':args.port}
folder=ROOT/'outputs/voltage';folder.mkdir(parents=True,exist_ok=True)
log=folder/('so101_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.csv')
stream=log.open('x',newline='',buffering=1)
writer=csv.writer(stream)
writer.writerow(['unix_time','elapsed_s','motor_id','voltage_V','temperature_C','status_register','packet_error','communication_result','lower_V','upper_V'])

def open_bus():
    # Refuse a port held by a teleoperation process; two readers corrupt replies.
    owner=subprocess.run(['fuser',str(Path(args.port).resolve())],capture_output=True,text=True)
    if owner.returncode==0 and owner.stdout.strip():
        raise RuntimeError('SO-101 포트를 다른 프로그램이 사용 중입니다. 해당 입력 프로그램을 종료하세요.')
    p=sdk.PortHandler(args.port)
    try:
        if not p.openPort() or not p.setBaudRate(1000000):raise RuntimeError('포트를 열 수 없습니다')
        # TIOCEXCL prevents later non-cooperating readers, including leader.py.
        fcntl.ioctl(p.ser.fileno(),termios.TIOCEXCL)
        return p
    except Exception:
        if p.ser is not None:p.closePort()
        raise

def sampler():
    packet=sdk.PacketHandler(0)
    bridge=None
    if args.teleop:
        try:
            from arm_sim.telemetry_bridge import TelemetryBridge
            bridge=TelemetryBridge(args.port)
        except Exception as exc:
            with lock:meta['teleop']={'tracking':False,'reason':str(exc)}
            # Monitoring remains available even if calibration is missing.
    while not stop.is_set():
        p=None
        try:
            p=open_bus()
            for i in range(1,7):
                lo,comm,_=packet.read1ByteTxRx(p,i,15)
                hi,comm2,_=packet.read1ByteTxRx(p,i,14)
                with lock:state.limits(i,round(lo*.1,1) if comm==0 else None,round(hi*.1,1) if comm2==0 else None)
            with lock:meta.update(connected=True,message='읽기 전용 수집 중')
            cycles=0;rate_start=time.monotonic();last_flush=rate_start;last_limits=rate_start
            while not stop.is_set():
                tick=time.monotonic();valid_count=0;frames={}
                for i in range(1,7):
                    if stop.is_set():break
                    data,comm,error=packet.readTxRx(p,i,56,15)
                    valid=comm==0 and len(data)==15;valid_count+=valid
                    frames[i]=(data,comm,error)
                    with lock:
                        state.update(i,data,comm,error)
                        m=state.motors[i]
                        writer.writerow([time.time(),round(state.clock()-state.start,6),i,
                            round(data[6]*.1,1) if valid else '',data[7] if valid else '',
                            data[9] if valid else '',error,comm,m['lower'],m['upper']])
                if bridge is not None:
                    with lock:limits={i:(m['lower'],m['upper']) for i,m in state.motors.items()}
                    teleop=bridge.cycle(frames,limits)
                    with lock:meta['teleop']=teleop
                cycles+=1
                now=time.monotonic()
                if valid_count and now-last_limits>2:
                    for i in range(1,7):
                        with lock:missing=state.motors[i]['lower'] is None or state.motors[i]['upper'] is None
                        if missing:
                            lo,c1,_=packet.read1ByteTxRx(p,i,15)
                            hi,c2,_=packet.read1ByteTxRx(p,i,14)
                            with lock:state.limits(i,round(lo*.1,1) if c1==0 else None,round(hi*.1,1) if c2==0 else None)
                    last_limits=now
                with lock:
                    meta['connected']=valid_count>0
                    meta['message']='읽기 전용 수집 중' if valid_count==6 else f'{valid_count}/6 모터 응답 · 전원/연결 확인'
                    if now-rate_start>=1:
                        meta['rate']=cycles/(now-rate_start);cycles=0;rate_start=now
                    if now-last_flush>=1:stream.flush();last_flush=now
                stop.wait(max(0,1/args.rate-(time.monotonic()-tick)))
        except Exception as exc:
            if bridge is not None:
                with lock:limits={i:(m['lower'],m['upper']) for i,m in state.motors.items()}
                teleop=bridge.cycle({},limits)
                with lock:meta['teleop']=teleop
            with lock:meta.update(connected=False,message=str(exc),rate=0.)
        finally:
            if p is not None:
                try:
                    fcntl.ioctl(p.ser.fileno(),termios.TIOCNXCL)
                except Exception:pass
                p.closePort()
        stop.wait(1.)
    if bridge is not None:bridge.close()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        route=self.path.split('?',1)[0]
        if route=='/':body=(ROOT/'web/voltage.html').read_bytes();ctype='text/html; charset=utf-8'
        elif route=='/api/state':
            with lock:body=json.dumps({**state.snapshot(),**meta,'csv':str(log)},ensure_ascii=False,allow_nan=False).encode()
            ctype='application/json; charset=utf-8'
        elif route=='/download':
            with lock:stream.flush();size=log.stat().st_size
            self.send_response(200);self.send_header('Content-Type','text/csv; charset=utf-8')
            self.send_header('Content-Disposition',f'attachment; filename="{log.name}"');self.send_header('Content-Length',str(size));self.end_headers()
            with log.open('rb') as f:
                while size>0:
                    chunk=f.read(min(size,65536))
                    if not chunk:break
                    self.wfile.write(chunk);size-=len(chunk)
            return
        else:self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',ctype)
        self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def log_message(self,*_):pass

server=ThreadingHTTPServer(('127.0.0.1',args.http_port),Handler)
def shutdown(*_):
    stop.set();threading.Thread(target=server.shutdown,daemon=True).start()
signal.signal(signal.SIGINT,shutdown);signal.signal(signal.SIGTERM,shutdown)
worker=threading.Thread(target=sampler,daemon=True);worker.start()
print(f'VOLTAGE MONITOR http://127.0.0.1:{args.http_port} | CSV: {log}',flush=True)
try:server.serve_forever()
finally:
    stop.set();worker.join(timeout=5);server.server_close();stream.close()
