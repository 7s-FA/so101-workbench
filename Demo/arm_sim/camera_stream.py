"""Loopback-only latest-frame viewer; HTTP never accesses simulator objects."""
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from PIL import Image
from arm_sim.demo_commands import parse_command

class CameraStream:
    def __init__(self, names, port=8766, demo_control=False):
        self.frames = {}
        self.names = list(names)
        self.metadata = {}
        self.demo_control = demo_control
        self.commands = []
        self.lock = threading.Lock()
        page = (Path(__file__).resolve().parents[1]/'web/cameras.html').read_bytes()
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path!='/demo/control' or not owner.demo_control:
                    self.send_error(404);return
                try:
                    length=int(self.headers.get('Content-Length','0'))
                    if not 0<length<=128:raise ValueError('Invalid length')
                    if self.headers.get('Origin') not in (None,'http://127.0.0.1:'+str(port),'http://localhost:'+str(port)):
                        self.send_error(403);return
                    command=parse_command(json.loads(self.rfile.read(length)))
                except (ValueError,AttributeError):self.send_error(400);return
                with owner.lock:
                    if len(owner.commands)>=16:self.send_error(429);return
                    owner.commands.append(command)
                self.send_response(204);self.end_headers()
            def do_GET(self):
                path = self.path.split('?')[0]
                if path == '/':
                    data, mime = page, 'text/html; charset=utf-8'
                elif path == '/state':
                    with owner.lock:
                        data = json.dumps({**{name: {'age': time.monotonic()-owner.frames[name][1] if name in owner.frames else None} for name in owner.names},'_meta':owner.metadata}).encode()
                    mime = 'application/json'
                elif path.startswith('/frame/') and path[7:] in owner.names:
                    with owner.lock:
                        frame = owner.frames.get(path[7:])
                    if frame is None:
                        self.send_error(503, 'Warming up'); return
                    data, mime = frame[0], 'image/jpeg'
                else:
                    self.send_error(404); return
                self.send_response(200)
                self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                try: self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError): pass
            def log_message(self, *_): pass
        self.server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
    def update(self, name, rgba):
        if rgba is None or not rgba.size: return
        buffer = io.BytesIO()
        picture=Image.fromarray(rgba.astype('uint8')).convert('RGB')
        picture.thumbnail((960,960))
        picture.save(buffer, format='JPEG', quality=80)
        with self.lock: self.frames[name] = (buffer.getvalue(), time.monotonic())
    def set_metadata(self, **values):
        # Fail at the producer instead of silently breaking every HTTP poll.
        json.dumps(values,allow_nan=False)
        with self.lock:self.metadata=dict(values)
    def pop_commands(self):
        with self.lock:
            result=self.commands[:];self.commands.clear();return result
    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
