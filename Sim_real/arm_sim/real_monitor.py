"""Simulation-only monitor: no voltage collector or motor telemetry access."""
import json,time,threading
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from arm_sim.common import ROOT
from arm_sim.demo_commands import parse_command
class RealMonitor:
    def __init__(self,stream,port=8767):
        self.stream=stream;owner=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def send(self,data,mime,status=200):
                self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers()
                try:self.wfile.write(data)
                except (BrokenPipeError,ConnectionResetError):pass
            def do_GET(self):
                p=self.path.split('?',1)[0]
                if p=='/api/state':
                    with stream.lock:
                        cams=[{'id':n,'label':n,'live':n in stream.frames and time.monotonic()-stream.frames[n][1]<3} for n in stream.names]
                        state={'simulation_available':bool(stream.metadata),'simulation':stream.metadata,'cameras':cams}
                    return self.send(json.dumps(state).encode(),'application/json')
                if p.startswith('/frame/'):
                    with stream.lock:frame=stream.frames.get(p[7:])
                    return self.send(frame[0],'image/jpeg') if frame else self.send_error(503)
                files={'/':'workcell.html','/workcell.js':'workcell.js','/workcell.css':'workcell.css','/monitor.css':'monitor.css'}
                if p not in files:return self.send_error(404)
                file=ROOT/'web'/files[p];mime='text/javascript' if p.endswith('.js') else 'text/css' if p.endswith('.css') else 'text/html'
                self.send(file.read_bytes(),mime+'; charset=utf-8')
            def do_POST(self):
                if self.path!='/api/demo/control':return self.send_error(404)
                if self.headers.get('Origin') not in (None,f'http://127.0.0.1:{port}',f'http://localhost:{port}'):return self.send_error(403)
                try:
                    length=int(self.headers.get('Content-Length','0'))
                    if not 0<length<=128:raise ValueError()
                    command=parse_command(json.loads(self.rfile.read(length)))
                    if isinstance(command,dict) and command['command']=='start_mode':raise ValueError()
                    with stream.lock:
                        if len(stream.commands)>=16:return self.send_error(429)
                        stream.commands.append(command)
                    self.send(b'{"queued":true}','application/json')
                except (ValueError,AttributeError):self.send_error(400)
        self.server=ThreadingHTTPServer(('127.0.0.1',port),Handler);self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
    def close(self):self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)
