"""Isaac-side blocking RPC. Never buffers commands or starts torque on import."""
import json
import select
import subprocess
import time
from pathlib import Path
from .entry import EntryPoseChanged, ENTRY_SPEED, entry_duration

ROOT=Path(__file__).resolve().parents[1]
IO_PYTHON=ROOT.parent/'Sim_real/.venv-io/bin/python'


class Follower:
    def __init__(self, port, calibration, visual_confirmed=False, fast_feedback=False, reference_control=False, display_only_scene=False, entry_speed=ENTRY_SPEED, sustained_temperature=False):
        if type(sustained_temperature) is not bool or sustained_temperature and not reference_control:
            raise ValueError('Sustained temperature checks require reference control')
        entry_duration([0]*6,[0]*6,entry_speed)
        if entry_speed != ENTRY_SPEED and not reference_control:
            raise ValueError('Custom entry speed requires reference control')
        if display_only_scene and not reference_control:
            raise ValueError('Display-only scene requires reference control')
        command=[str(IO_PYTHON),'-m','teach_io.worker','--port',port,'--calibration',str(calibration)]
        if visual_confirmed:command.append('--visual-confirmed')
        if fast_feedback:command.append('--fast-feedback')
        if reference_control:command.append('--reference-control')
        if sustained_temperature:command.append('--sustained-temperature')
        if display_only_scene:command.append('--display-only-scene')
        if entry_speed != ENTRY_SPEED:command.extend(['--entry-speed',str(entry_speed)])
        self.process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            text=True,bufsize=1)
        self.armed=False
        self.last_poll=0.
        self.latest={}
        self.connection_error=None
        try:self.latest=self.receive(8)
        except Exception:
            self.close();raise

    def receive(self, timeout=.25):
        if not select.select([self.process.stdout],[],[],timeout)[0]:
            raise RuntimeError('Follower response timed out; motor watchdog remains active')
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('Follower connection closed')
        reply=json.loads(line)
        if not reply.get('ok'):
            if reply.get('code')=='entry_pose_changed':raise EntryPoseChanged(reply['error'])
            raise RuntimeError(reply.get('error','Follower fault'))
        return reply

    def request(self, op, **data):
        if self.connection_error:
            raise RuntimeError(self.connection_error)
        try:
            if self.process.poll() is not None:
                raise BrokenPipeError('Follower process exited')
            self.process.stdin.write(json.dumps(dict(op=op,sent=time.monotonic(),**data))+'\n')
            self.process.stdin.flush()
            self.latest=self.receive(5 if op=='arm' else .25)
        except EntryPoseChanged:
            raise  # Worker is still read-only and accepts a fresh pose.
        except Exception as error:
            self.connection_error=('팔 제어 연결이 종료되었습니다. 장치 연결을 해제한 뒤 다시 연결해 주세요.'
                                   if isinstance(error,(BrokenPipeError,EOFError,OSError)) else str(error))
            # Ownership/torque is deliberately retained until explicit release.
            # Never retry a write or implicitly restart this worker after a fault.
            raise RuntimeError(self.connection_error) from error
        self.last_poll=time.monotonic()
        return self.latest

    def arm(self, target, path, compensation_enabled=True, reuse_compensation=False, evaluation=True, entry=None):
        self.request('arm',target=list(target[:6]),path=path,evaluation=evaluation,entry=entry,
            compensation_enabled=compensation_enabled,reuse_compensation=reuse_compensation)
        self.armed=True

    def update(self, target, context='unknown', phase='unknown', trial_active=False):
        if time.monotonic()-self.last_poll>=.02:
            self.request('target' if self.armed else 'read',target=list(target[:6]),context=context,phase=phase,trial_active=trial_active)
        return self.latest

    def close(self):
        self.armed=False
        if self.process.poll() is None:
            try:
                self.request('hold')
                self.process.wait(timeout=.5)
            except Exception:
                self.process.terminate()
                try:self.process.wait(timeout=.5)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        for stream in (self.process.stdin,self.process.stdout):
            try:stream.close()
            except OSError:pass  # A dead pipe must not block explicit torque release.
