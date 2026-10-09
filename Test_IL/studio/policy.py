"""Non-blocking interface to an isolated ACT inference worker (no USB access)."""
import base64
import json
import queue
import subprocess
import threading
import time
from .config import ROOT, IL_PYTHON
from .cameras import learning_image


class PolicyWorker:
    def __init__(self, checkpoint, device, log_path, options=None):
        self.ready=False;self.ready_info={};self.error=None;self.result=None;self.busy=False;self.closed=False
        self.requests=queue.Queue(maxsize=1);self.log_path=log_path
        with log_path.open('w') as log:
            self.process=subprocess.Popen([str(IL_PYTHON),str(ROOT/'scripts/predict_act.py'),str(checkpoint),device,json.dumps(options or {})],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
    def run(self):
        try:
            response=json.loads(self.process.stdout.readline())
            if response.get('kind')!='ready':raise RuntimeError(response.get('error','모델을 준비하지 못했습니다.'))
            self.ready_info=response;self.ready=True
            while not self.closed:
                item=self.requests.get()
                if item is None:return
                frames=item.pop('frames')
                item['cameras']={k:base64.b64encode(learning_image(v)[1]).decode('ascii') for k,v in frames.items()}
                self.process.stdin.write(json.dumps(item)+'\n');self.process.stdin.flush()
                response=json.loads(self.process.stdout.readline())
                if 'error' in response:raise RuntimeError(response['error'])
                response['received_at']=time.monotonic();self.result=response;self.busy=False
        except Exception as error:
            if not self.closed:self.error=str(error)+' · '+str(self.log_path)
        finally:
            self.ready=False;self.busy=False
    def submit(self, state, frames, acquired):
        if not self.ready or self.busy:return False
        self.busy=True
        self.requests.put_nowait(dict(state=list(state),acquired=acquired,
            observation_at=min([acquired]+[v.get('acquired',acquired) for v in frames.values()]),
            frames=frames))
        return True
    def close(self):
        self.closed=True;self.ready=False
        if self.process.poll() is None:self.process.terminate()
        # Cleanup must not delay the control-owner heartbeat while it holds motors.
        def reap():
            try:self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
            try:self.requests.put_nowait(None)
            except queue.Full:pass
            self.thread.join(timeout=1)
            for stream in (self.process.stdin,self.process.stdout):stream.close()
        threading.Thread(target=reap,daemon=True).start()
