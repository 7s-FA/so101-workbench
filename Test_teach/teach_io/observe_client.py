"""Isaac-side read-only client. Stops on a failed/stale response; no auto reconnect."""
import json
import select
import subprocess
import time
from pathlib import Path
from .client import ROOT, IO_PYTHON


class Observer:
    def __init__(self,port,mapping):
        self.process=subprocess.Popen([str(IO_PYTHON),'-m','teach_io.observe_worker','--port',str(port),
            '--mapping',str(mapping)],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
        self.latest=None;self.last_poll=0.;self.samples=0
        try:self.initial=self.receive(5)
        except Exception:self.close();raise

    def receive(self,timeout=.5):
        if not select.select([self.process.stdout],[],[],timeout)[0]:raise RuntimeError('USB observation timed out')
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('USB observation process ended')
        reply=json.loads(line)
        if not reply.get('ok'):raise RuntimeError(reply.get('error','Observation error'))
        return reply

    def update(self):
        if time.monotonic()-self.last_poll>=1/30:
            self.process.stdin.write(json.dumps(dict(op='read',sent=time.monotonic()))+'\n');self.process.stdin.flush()
            self.latest=self.receive();self.last_poll=time.monotonic();self.samples+=1
        if self.latest and time.monotonic()-self.latest['received']>.5:raise RuntimeError('Observation is stale')
        return self.latest

    def close(self):
        if not self.process.stdin.closed:self.process.stdin.close()
        try:self.process.wait(timeout=.6)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:self.process.wait(timeout=.5)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        self.process.stdout.close()
