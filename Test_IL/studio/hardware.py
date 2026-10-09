"""SO-101 RPC client. Device startup is read-only; motion requires explicit arm()."""
import json
from pathlib import Path
import select
import subprocess
import sys
import time
from .config import ROOT,TEACH,IL_PYTHON
from .robot import Mapping,profile,reference_config,OCV,NAMES
sys.path.insert(0,str(TEACH))
from teach_io.lifecycle import notify
from teach_io.entry import validate_entry, EntryPoseChanged
IL_ENTRY_SPEED=.5


def mapping():return Mapping()


def audit_entry(start,goal):
    plan=dict(start=list(start),goal=list(goal),collision_checked=False,scene_policy='display_only')
    validate_entry(plan,start,goal,mapping(),allow_start_residual=True,require_collision_check=False)
    return plan


class Follower:
    def __init__(self,port,leader=None):
        command=[str(IL_PYTHON),'-m','studio.so101_worker','--port',port]
        if leader:command+=['--leader',leader]
        self.armed=False;self.latest={};self.process=None
        self.process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
        try:self.latest=self.receive(10)
        except BaseException:
            self.close(release=False);raise

    def receive(self,timeout):
        if not select.select([self.process.stdout],[],[],timeout)[0]:raise RuntimeError('SO-101 응답 지연. USB 연결을 확인하세요.')
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('SO-101 제어 연결이 종료되었습니다. 다시 연결하세요.')
        result=json.loads(line);self.latest=result
        if result.get('retryable')=='entry_pose_changed':raise EntryPoseChanged(result['error'])
        if not result.get('ok'):raise RuntimeError(result.get('error','SO-101 제어 오류'))
        return result

    def request(self,op,**data):
        if self.process.poll() is not None:raise RuntimeError('SO-101 제어 연결이 종료되었습니다. 다시 연결하세요.')
        self.process.stdin.write(json.dumps(dict(op=op,sent=time.monotonic(),**data),allow_nan=False)+'\n');self.process.stdin.flush()
        result=self.receive(10 if op in ('arm','release','close') else 1.)
        if op=='arm':self.armed=True
        if op in ('release','close'):self.armed=False
        return result

    def close(self,release=True):
        if self.process is None:return
        if self.process.poll() is None:
            try:self.request('close',release=release);self.process.wait(timeout=1)
            except Exception:
                self.process.terminate()
                try:self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=1)
        self.armed=False
        for stream in (self.process.stdin,self.process.stdout):
            try:stream.close()
            except OSError:pass


class Hardware:
    def __init__(self,port,leader_port=None):
        self.port=port;self.leader=leader_port or None;self.owned=False
        self.leader_cfg=reference_config();self.follower_mapping=mapping()
        self.follower=Follower(port,leader_port)
    def read(self):return self.follower.request('read')
    def target(self):
        result=self.read()
        if result.get('leader_target') is None:raise ValueError('SO-101 리더를 연결하세요.')
        self.requested_action=list(result.get('leader_angles') or result['leader_target'])
        return self.follower_mapping.clamp_target(result['leader_target']),result['leader_ticks']
    def arm(self,goal,path,entry):
        notify('motor_port',port=self.port);self.owned=True
        self.follower.request('arm',target=list(goal),entry=entry)
    def update(self,goal,active=False):return self.follower.request('target',target=list(goal))
    def hold(self):return self.follower.request('hold')
    def release_torque(self):
        result=self.follower.request('release');self.owned=False;notify('released',port=self.port)
        # Restart only after an explicit release, clearing a latched health fault.
        self.follower.close(release=False);self.follower=Follower(self.port,self.leader)
        return self.read()
    def close(self,release=True):
        required=release and (self.owned or self.follower.armed)
        self.follower.close(release=release)
        if required:
            if not release_port(self.port):raise RuntimeError('SO-101 모터 힘 해제를 확인하지 못했습니다.')
            self.owned=False;notify('released',port=self.port)


def release_port(port):
    """Supervisor fallback: correct SO-101 protocol, no goals/calibration writes."""
    try:
        sys.path.insert(0,str(OCV))
        from teach_left_arm import Hardware as TeachingHardware
        from diagnose_shoulder_lift import ensure_ports_available
        cfg=profile()
        if Path(port).resolve()!=Path(cfg['follower_port']).resolve():raise ValueError('등록된 SO-101 USB가 아닙니다.')
        ensure_ports_available([port]);hw=TeachingHardware(cfg['follower_profile'],follower_only=True)
        hw.follower.bus.connect();hw.motion_attempted=True
        try:
            errors=hw.close()
            if errors:raise RuntimeError('; '.join(errors))
        finally:
            if hw.follower.bus.is_connected:hw.follower.bus.disconnect(disable_torque=False)
        return True
    except Exception as error:
        print('SO-101 모터 힘 해제 확인 실패: '+str(error),file=sys.stderr,flush=True)
        return False
