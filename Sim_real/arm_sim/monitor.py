"""Loopback monitor: hardware read-only, with optional virtual demo controls."""
import copy
import json
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen, Request
from arm_sim.common import ROOT
from arm_sim.demo_commands import parse_command

LABELS={'top':'전체 공정 · 탑뷰','perspective':'전체 공정 · 사선뷰','overview':'전체 공정',
        'depth_a':'SO-101 A · 손목 RGB','depth_c':'SO-101 C · 손목 RGB',
        'depth_a_depth':'SO-101 A · 깊이','depth_c_depth':'SO-101 C · 깊이',
        'wrist_b':'OMX B · 그리퍼 RGB','wrist_d':'OMX D · 그리퍼 RGB',
        'top_b':'OMX B · 상부','top_d':'OMX D · 상부',
        'so101':'SO-101 · 손목 RGB','omx':'OMX · 손목 RGB','wrist':'SO-101 · 손목 RGB'}

def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)

def fetch_json(url):
    try:
        with urlopen(url,timeout=.4) as response:
            return json.loads(response.read(4*1024*1024))
    except (OSError,ValueError):return None


def normalize_so(raw,cache_age=0.):
    source=raw if isinstance(raw,dict) else {}
    motors=[]
    for m in source.get('motors',[]):
        age=m.get('age');age=age+cache_age if finite(age) else None
        live=bool(raw is not None and not m.get('stale',True) and age is not None and 0<=age<=.75)
        out={key:m.get(key) for key in ('id','name','minimum','maximum','lower','upper','error','error_samples','comm_errors','history')}
        out.update(stale=not live,age=age)
        for key in ('voltage','temperature','load_raw'):
            value=m.get(key);out[key]=value if live and finite(value) else None
        motors.append(out)
    return {'available':raw is not None,'connected':any(not m['stale'] for m in motors),
            'message':source.get('message','수집 프로그램 연결 대기'),'rate':source.get('rate'),
            'elapsed':source.get('elapsed'),'teleop':source.get('teleop'),'motors':motors,
            'events':source.get('events',[]),'scope':'SO-101 리더 1대 · 모터 6개'}


def read_omx(path,now=None):
    now=time.time() if now is None else now
    try:
        raw=json.loads(path.read_text())
        if not isinstance(raw,dict):raise ValueError('Expected an OMX status object')
        stamp=raw.get('timestamp',path.stat().st_mtime)
        age=now-stamp;live=finite(age) and 0<=age<=3.
    except (OSError,ValueError,TypeError):raw={};age=None;live=False
    result={'connected':live,'age':age,'scope':'OMX 리더 · 그리퍼 ID 6',
            'message':'수집 중' if live else '수집 프로그램 연결 대기'}
    for key in ('voltage','temperature','feedback_state','force_enabled','voltage_blocked','samples'):
        result[key]=raw.get(key) if live else None
    result['motor']=raw.get('motor',{}) if live else {}
    return result


class MonitorServer:
    def __init__(self,port=8767,camera_url='http://127.0.0.1:8766',voltage_url='http://127.0.0.1:8765',omx_path=None):
        self.camera_url=camera_url;self.voltage_url=voltage_url
        self.omx_path=Path(omx_path or ROOT/'outputs/omx_haptics_status.json')
        self.lock=threading.Lock();self.stop=threading.Event();self.cached=None
        owner=self
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path!='/api/demo/control':self.send_error(404);return
                if self.headers.get('Origin') not in (None,'http://127.0.0.1:'+str(owner.port),'http://localhost:'+str(owner.port)):
                    self.send_error(403);return
                try:
                    length=int(self.headers.get('Content-Length','0'))
                    if not 0<length<=128:raise ValueError('Invalid length')
                    command=parse_command(json.loads(self.rfile.read(length)))
                    snapshot=owner.snapshot()
                    if not snapshot['simulation_available'] or snapshot['simulation'].get('mode')!='demo':
                        self.send_error(409,'Demo is not connected');return
                    data=json.dumps(command if isinstance(command,dict) else {'command':command}).encode()
                    with urlopen(Request(owner.camera_url+'/demo/control',data=data,headers={'Content-Type':'application/json'}),timeout=1.) as response:
                        response.read()
                    return self.send_json({'queued':command})
                except (ValueError,AttributeError):self.send_error(400);return
                except OSError:self.send_error(503,'Demo unavailable');return
            def do_GET(self):
                route=self.path.split('?',1)[0]
                if route in ('/','/diagnostics'):
                    page='workcell.html' if route=='/' else 'monitor.html'
                    return self.send((ROOT/'web'/page).read_bytes(),'text/html; charset=utf-8')
                if route in ('/monitor.js','/monitor.css','/workcell.js','/workcell.css'):
                    mime='text/javascript' if route.endswith('.js') else 'text/css'
                    return self.send((ROOT/'web'/route[1:]).read_bytes(),mime+'; charset=utf-8')
                if route=='/api/health':return self.send_json({'application':'final-arm-monitor','version':1})
                if route=='/api/state':return self.send_json(owner.snapshot())
                if route.startswith('/frame/'):
                    name=route[7:]
                    valid=owner.camera_live(name)
                    if not re.fullmatch(r'[a-zA-Z0-9_]+',name) or not valid:
                        self.send_error(503,'Camera unavailable');return
                    try:
                        with urlopen(owner.camera_url+'/frame/'+name,timeout=1.) as response:data=response.read(3*1024*1024)
                        return self.send(data,'image/jpeg')
                    except OSError:self.send_error(503,'Camera unavailable');return
                if route=='/download/voltage':
                    try:
                        with urlopen(owner.voltage_url+'/download',timeout=2.) as response:
                            self.send_response(200);self.send_header('Content-Type','text/csv')
                            self.send_header('Content-Disposition','attachment; filename="so101-voltage.csv"');self.end_headers()
                            while chunk:=response.read(65536):self.wfile.write(chunk)
                    except (OSError,BrokenPipeError):pass
                    return
                self.send_error(404)
            def send_json(self,value):
                self.send(json.dumps(value,ensure_ascii=False,allow_nan=False).encode(),'application/json; charset=utf-8')
            def send(self,data,mime):
                self.send_response(200);self.send_header('Content-Type',mime)
                self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)))
                self.send_header('X-Content-Type-Options','nosniff');self.end_headers()
                try:self.wfile.write(data)
                except (BrokenPipeError,ConnectionResetError):pass
            def log_message(self,*_):pass
        self.server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
        self.port=self.server.server_address[1]
        self.http=threading.Thread(target=self.server.serve_forever,kwargs={'poll_interval':.1},daemon=True)
        self.worker=threading.Thread(target=self.poll,daemon=True)
        self.http.start();self.worker.start()

    def poll(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            while not self.stop.is_set():
                started=time.monotonic()
                camera=pool.submit(fetch_json,self.camera_url+'/state')
                voltage=pool.submit(fetch_json,self.voltage_url+'/api/state')
                cam=camera.result();so=voltage.result()
                # Request-start age is conservative even if the other source is slow.
                with self.lock:self.cached=(cam,so,started)
                self.stop.wait(.2)

    def camera_live(self,name):
        # Image requests must not copy voltage histories or reread diagnostic files.
        with self.lock:
            if self.cached is None:return False
            cam,_,stamp=self.cached
            value=cam.get(name) if isinstance(cam,dict) else None
            age=value.get('age') if isinstance(value,dict) else None
            return finite(age) and 0<=age+time.monotonic()-stamp<2.

    def snapshot(self):
        with self.lock:cache=copy.deepcopy(self.cached)
        cam,so,stamp=cache if cache is not None else (None,None,time.monotonic())
        age=time.monotonic()-stamp;cameras=[]
        if isinstance(cam,dict):
            for name,value in cam.items():
                if name.startswith('_') or not isinstance(value,dict) or not re.fullmatch(r'[a-zA-Z0-9_]+',name):continue
                a=value.get('age');a=a+age if finite(a) else None
                cameras.append({'id':name,'label':LABELS.get(name,name),'age':a,'live':a is not None and 0<=a<2.,
                                'role':'overview' if name in ('top','perspective','overview') else 'wrist'})
        return {'time':time.time(),'cameras':cameras,'simulation':cam.get('_meta',{}) if isinstance(cam,dict) else {},
                'simulation_available':any(c['live'] for c in cameras),
                'so101':normalize_so(so,age),'omx':read_omx(self.omx_path),
                'read_only':True}

    def close(self):
        self.stop.set();self.server.shutdown();self.server.server_close()
        self.http.join(timeout=2);self.worker.join(timeout=2)


def ensure_monitor(port=8767):
    try:return MonitorServer(port=port)
    except OSError:
        health=fetch_json(f'http://127.0.0.1:{port}/api/health')
        if isinstance(health,dict) and health.get('application')=='final-arm-monitor':return None
        raise RuntimeError(f'통합 모니터 포트 {port}를 다른 프로그램이 사용 중입니다')
