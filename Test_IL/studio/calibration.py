"""Official SO-101 calibration workflow, reusing the OpenCV teaching worker."""
import copy
import hashlib
import json
from pathlib import Path
import queue
import re
import sys
import time
from types import SimpleNamespace

from .config import ROOT
from .robot import OCV, NAMES, check_calibration, profile, reference_config
from .store import atomic_json

GUIDE='https://huggingface.co/docs/lerobot/v0.4.4/en/so101#calibrate'


def validate_selection(role,port,name):
    if role not in ('follower','leader'):raise ValueError('팔로워 또는 리더를 선택하세요.')
    if not isinstance(port,str) or Path(port).parent!=Path('/dev/serial/by-id') or not Path(port).name.startswith('usb-1a86_'):
        raise ValueError('SO-101 USB 포트를 선택하세요.')
    if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',name):
        raise ValueError('파일 이름은 영문·숫자·밑줄·하이픈 1~64자로 입력하세요.')
    other=profile().get(('leader' if role=='follower' else 'follower')+'_port')
    if other and Path(other).resolve()==Path(port).resolve():raise ValueError('리더와 팔로워는 다른 USB를 사용해야 합니다.')


def catalog():
    cfg=profile();paths=set()
    for root in (ROOT/'config/calibration',OCV/'calibration',Path.home()/'.cache/huggingface/lerobot/calibration'):
        if root.exists():paths.update(root.rglob('*.json'))
    for role in ('follower','leader'):
        if cfg.get(role+'_calibration'):paths.add(Path(cfg[role+'_calibration']))
    rows=[]
    for path in sorted(paths):
        try:cal=check_calibration(json.loads(path.read_text()))
        except (OSError,ValueError,TypeError,KeyError):continue
        rows.append(dict(id=hashlib.sha256(str(path).encode()).hexdigest()[:20],name=path.stem,path=str(path),calibration=cal,
                         active_roles=[r for r in ('follower','leader') if cfg.get(r+'_calibration')==str(path)]))
    return rows


def register(role,port,name,calibration):
    """Select a local profile. No bus, motor or original external file writes."""
    validate_selection(role,port,name);calibration=check_calibration(calibration)
    cfg=profile();previous=None;old_port=cfg.get(role+'_port')
    old_file=cfg.get(role+'_calibration')
    if old_file:
        try:previous=check_calibration(json.loads(Path(old_file).read_text()))
        except (OSError,ValueError):pass
    directory=ROOT/'config/calibration'/('il_'+role);directory.mkdir(parents=True,exist_ok=True)
    destination=directory/(name+'.json')
    backup=ROOT/'config/calibration_backups';backup.mkdir(exist_ok=True)
    atomic_json(backup/(str(time.time_ns())+'.json'),dict(robot=cfg,role=role,previous_calibration=previous,
        destination=str(destination),previous_destination=json.loads(destination.read_text()) if destination.exists() else None))
    atomic_json(destination,calibration)
    cfg[role+'_port']=port;cfg[role+'_calibration']=str(destination)
    if role=='follower':
        wrapper=ROOT/'config/il_follower.json'
        atomic_json(wrapper,dict(schema=1,robot_type='so101',mode='follower_only',pair_name='il_follower',side='right',
                                 follower_id=name,follower_port=port))
        cfg['follower_profile']=str(wrapper)
    atomic_json(ROOT/'config/robot.json',cfg)
    atomic_json(ROOT/'config/lerobot_calibration.json',reference_config())
    changed=previous!=calibration or port!=old_port
    return dict(ok=True,role=role,path=str(destination),coordinate_changed=changed)


class CalibrationManager:
    def __init__(self):
        self.worker=None;self.applied=False
        self.status=dict(state='idle',busy=False,message='팔과 파일을 선택하세요.',positions=None,saved=None,error=None)

    def start(self,role,port,name,preset=None):
        if self.status['busy']:raise ValueError('진행 중인 보정을 먼저 끝내세요.')
        validate_selection(role,port,name)
        if port not in {str(p) for p in Path('/dev/serial/by-id').glob('usb-1a86_*')}:raise ValueError('선택한 USB가 연결되어 있지 않습니다.')
        sys.path.insert(0,str(OCV))
        from motor_calibration import CalibrationWorker,make_arm
        directory=ROOT/'config/calibration'/('il_'+role);directory.mkdir(parents=True,exist_ok=True)
        recovery=ROOT/'outputs/motor_calibration'/('recovery_'+str(time.time_ns())+'.json')
        recovery.parent.mkdir(parents=True,exist_ok=True)
        backend=SimpleNamespace(args=SimpleNamespace(calibration_dir=directory))
        self.worker=CalibrationWorker(lambda:make_arm(backend,role,port,name),recovery,preset=preset)
        self.applied=False
        self.status=dict(state='CONNECTING',busy=True,method='file' if preset is not None else 'new',role=role,port=port,name=name,positions=None,saved=None,error=None,
            message='선택한 팔의 모터 힘을 끄고 보정을 준비합니다.',recovery=str(recovery))
        self.worker.thread.start()

    def command(self,action):
        if not self.worker or not self.status['busy']:raise ValueError('진행 중인 보정이 없습니다.')
        self.status.pop('notice',None)
        if action=='cancel':self.worker.cancel.set();self.status['message']='이전 모터 보정을 복원하고 있습니다.';return
        allowed={'MIDPOINT':('middle',),'RANGE':('review',),'REVIEW':('resume','save')}
        if action not in allowed.get(self.worker.state,()):raise ValueError('현재 보정 단계에서 사용할 수 없는 버튼입니다.')
        self.worker.commands.put(action)

    def poll(self):
        if not self.worker:return
        was_busy=self.status['busy']
        while True:
            try:kind,value=self.worker.events.get_nowait()
            except queue.Empty:break
            if kind=='state':self.status['state']=value;self.status.pop('notice',None)
            elif kind=='positions':self.status['positions']=value
            elif kind=='saved':self.status['saved']=value
            elif kind=='error':self.status['error']=value;self.status['message']=value
            else:
                self.status['message']=value
                if self.worker.state in ('RANGE','REVIEW'):self.status['notice']=value
        self.status['busy']=not self.worker.finished.is_set()
        if was_busy and not self.status['busy']:self.status['completed_at']=time.time()

    def snapshot(self):return copy.deepcopy(self.status)

    def close(self):
        if self.worker and not self.worker.finished.is_set():
            self.worker.cancel.set();self.worker.thread.join(5)
            if self.worker.thread.is_alive():raise RuntimeError('모터 보정 정리가 끝나지 않았습니다. 복원 파일을 확인하세요: '+self.status['recovery'])
        self.poll()
