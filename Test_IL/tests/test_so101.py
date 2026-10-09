"""SO-101/STS3215 regressions. All buses and images here are fake."""
import copy,json,math,sys,time
from pathlib import Path
from types import SimpleNamespace
import pytest
from studio.robot import Mapping,reference_config,pose_to_angles,angles_to_pose,check_calibration,identity,profile
from studio.reference import ReferenceCodec
from studio.config import ROOT,load,fingerprint,TEACH
from studio.engine import Engine,validate_replay
from studio.so101_worker import Session
from temperature_guard import TemperatureGuard,TemperatureFault
from test_studio import frames,sample
from studio.store import Store,atomic_json


def test_degrees_and_gripper_match_latest_teaching_and_lerobot():
    cfg=reference_config();mapping=Mapping(cfg);codec=ReferenceCodec(cfg)
    assert mapping.cfg['ids']==list(range(1,7))
    q=pose_to_angles([10.,-30.,20.,45.,90.,35.])
    ticks=mapping.to_ticks(q);v=codec.normalize(ticks)
    assert v==pytest.approx([10.,-30.,20.,45.,90.,35.],abs=.09)
    assert mapping.to_angles(codec.follower_ticks(v))==pytest.approx(q,abs=.002)
    assert angles_to_pose(q)['gripper.pos']==pytest.approx(35)


def test_different_leader_ranges_map_angles_not_encoder_numbers():
    cfg=reference_config();cfg['leader']=copy.deepcopy(cfg['follower'])
    cfg['leader']['shoulder_pan'].update(range_min=800,range_max=3200)
    cfg['leader']['gripper'].update(range_min=1000,range_max=2000)
    codec=ReferenceCodec(cfg);m=Mapping(cfg)
    ticks=[2000,*[round(m.zero[i]) for i in range(1,5)],1500]
    q=codec.leader_target(ticks,m)
    assert abs(q[0])<.002 and q[5]==pytest.approx(1.74533*.5,abs=.002)
    assert m.to_ticks(q)[0]!=2000
    ticks[0]=4095
    assert codec.leader_target(ticks,m)[0]==m.cfg['limits'][0][1]


def test_missing_leader_and_omx_calibration_never_silently_used():
    codec=ReferenceCodec()
    if not codec.cfg.get('leader'):
        with pytest.raises(ValueError,match='리더 보정'):codec.normalize([2048]*6,'leader')
    with pytest.raises(ValueError):check_calibration(json.loads((ROOT/'config/legacy_omx/lerobot_calibration.json').read_text())['follower'])


class FakeBackend:
    def __init__(self,clock):
        self.clock=clock;self.q=pose_to_angles([0,0,0,0,0,30]);self.writes=[];self.motion_attempted=False
        self.temperature_guard=TemperatureGuard(clock=clock);self.temperature_sample={};self.heat=25
        self.follower=SimpleNamespace(bus=SimpleNamespace(is_connected=True,disconnect=lambda **kw:None))
    def connect(self):return angles_to_pose(self.q)
    def read(self):return None,angles_to_pose(self.q)
    def check_temperature_watchdog(self):self.temperature_guard.check_timeouts()
    def health(self,active):
        self.temperature_guard.observe('shoulder_lift','health/block',self.heat,active=active)
        self.temperature_guard.observe('shoulder_lift','health/single',self.heat,active=active)
        self.temperature_sample={'temperatures_raw':{'shoulder_lift':self.heat}}
    def activate(self):self.motion_attempted=True;self.writes.append('activate');return angles_to_pose(self.q)
    def send(self,pose):self.writes.append(pose.copy());self.q=pose_to_angles(pose);return pose
    def project_target(self,pose):return angles_to_pose(Mapping().clamp_target(pose_to_angles(pose)))
    def close(self):self.writes.append('off');return []


def test_readonly_start_entry_direct_follow_and_action_recording():
    now=[1.];backend=FakeBackend(lambda:now[0]);session=Session(profile()['follower_port'],backend=backend,clock=lambda:now[0])
    assert backend.writes==[] and session.armed is False
    start=session.read()['angles'];goal=start.copy();goal[1]=.1
    session.arm(goal,dict(start=start,goal=goal,scene_policy='display_only'))
    for _ in range(180):now[0]+=.02;result=session.update(goal)
    assert result['aligned'] and session.armed
    target=goal.copy();target[0]=.8;target[1]=-.5
    now[0]+=.02;result=session.update(target)
    assert result['tracking']['command_angles']==pytest.approx(target)
    assert result['tracking']['desired_angles']==pytest.approx(target)
    session.close();assert backend.writes[-1]=='off'


def test_same_raw_temperature_duration_and_reset_as_opencv():
    now=[1.];backend=FakeBackend(lambda:now[0]);session=Session(profile()['follower_port'],backend=backend,clock=lambda:now[0])
    session.armed=True;backend.heat=66
    for _ in range(10):now[0]+=.5;result=session.read()
    assert result['temperature']['unit']=='raw' and result['temperature']['pending']
    backend.heat=25;now[0]+=.5
    assert session.read()['temperature']['pending']==[]
    backend.heat=65
    with pytest.raises(TemperatureFault):
        for _ in range(12):now[0]+=.5;session.read()
    session.close()


def test_so101_and_omx_storage_and_model_compatibility(tmp_path):
    e=Engine(load(),tmp_path)
    legacy={'fingerprint':fingerprint(),'units':'rad','robot_type':'omx','synthetic':False,'control_profile':'lerobot_reference'}
    with pytest.raises(ValueError):validate_replay(legacy,[sample(0),sample(.03)],fingerprint(),Mapping())
    e.store.begin(legacy)
    for i in range(2):e.store.append(sample(i/30),frames())
    e.store.save('Success')
    with pytest.raises(ValueError,match='다른 로봇'):e.store.ensure_task_robot(None,'so101')
    e.select(1);e.mode='preview';e.publish()
    assert e.public['display_robot']=='omx' and e.public['robot_type']=='so101'
    new=e.learning.tasks.save('SO101 이동');e.store.ensure_task_robot(new['id'],'so101')
    e.close()


def test_latest_model_has_wrist_uvc_overhead_d435_and_tpu():
    import xml.etree.ElementTree as ET
    scene=json.loads((ROOT/'config/scene.json').read_text());tree=ET.parse(scene['model_source'])
    wrist=[v.get('name','') for v in tree.findall("link[@name='gripper_link']/visual")]
    base=[v.get('name','') for v in tree.findall("link[@name='base_link']/visual")]
    assert 'wrist_xlerobot_mount' in wrist and 'wrist_32x32_camera' in wrist
    assert 'overhead_d435' in base and 'overhead_top_d435' in base
    assert 'd435_body' not in wrist and 'd435_mount' not in wrist
    assert 'xlerobot_fixed_finger' in wrist
    assert scene['parts']=={} and scene['jigs_enabled'] is False
    reference=json.loads((TEACH/'config/so101_scene.json').read_text())
    assert scene['table']['size']==[v/1000 for v in reference['table']['size_mm']]
    safe_program=json.loads(Path(profile()['default_pose_program']).read_text())
    safe_pose=next(s['pose'] for s in safe_program['steps'] if s.get('safe_boundary')=='start')
    # The OpenCV program can be edited independently after CAD generation.
    from studio.safe_pose import load_safe_pose
    safe=load_safe_pose()
    assert safe['raw_angles']==pytest.approx(pose_to_angles(safe_pose))
    assert safe['reference']['pose']==safe_pose
    assert scene['base'][2]-scene['table']['height']==pytest.approx(.0074)
    assert tree.find("joint[@name='overhead_rgb_optical_frame_joint']") is not None


def test_stop_cancels_initial_usb_interpolation(tmp_path):
    now=[1.];backend=FakeBackend(lambda:now[0]);session=Session(profile()['follower_port'],backend=backend,clock=lambda:now[0])
    start=session.read()['angles'];goal=start.copy();goal[0]=.8
    session.arm(goal,dict(start=start,goal=goal))
    now[0]+=.1;session.update(goal)
    held=backend.q.copy();session.hold();assert session.entry is None
    now[0]+=.1;session.update(held)
    assert backend.q==pytest.approx(held)
    e=Engine(load(),tmp_path)
    calls=[]
    e.hardware=SimpleNamespace(follower=SimpleNamespace(armed=True),leader=None,
        hold=lambda:calls.append('hold') or dict(angles=held),close=lambda **kw:None)
    e.mode='preparing';e.after_prepare='teleop'
    e.hold();assert calls==['hold'] and e.mode=='holding' and e.entry is None
    e.close();session.close()


def test_rpc_watchdog_and_exit_release_only_owned_fake_session(tmp_path):
    import subprocess,select
    trace=tmp_path/'events.jsonl'
    program=r'''
import json,sys
from pathlib import Path
from types import SimpleNamespace
from studio import so101_worker as module
trace=Path(sys.argv.pop(1))
class Fake:
 def __init__(self,*args):self.latest={'ok':True,'angles':[0]*6};self.armed=False;self.error=None
 def event(self,name):
  with trace.open('a') as f:f.write(json.dumps(name)+'\n')
 def read(self):self.event('read');return self.latest
 def arm(self,*a):self.armed=True;self.event('arm');return self.latest
 def update(self,*a):self.event('target');return self.latest
 def hold(self):self.event('hold')
 def close(self,release=True):self.event('release' if release else 'disconnect');self.armed=False
 def release(self):self.event('release');self.armed=False
module.Session=Fake
module.main()
'''
    p=subprocess.Popen([sys.executable,'-c',program,str(trace),'--port','/dev/fake'],cwd=ROOT,
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    def read():
        assert select.select([p.stdout],[],[],8)[0]
        return json.loads(p.stdout.readline())
    def send(op,**kw):
        p.stdin.write(json.dumps(dict(op=op,sent=time.monotonic(),**kw))+'\n');p.stdin.flush();return read()
    try:
        assert read()['ok']
        assert send('arm',target=[0]*6,entry={})['ok']
        time.sleep(.7)
        assert send('target',target=[0]*6)['ok'] is False
        assert send('release')['ok']
        assert send('close',release=False)['ok'];p.wait(timeout=3)
        events=[json.loads(x) for x in trace.read_text().splitlines()]
        assert 'hold' in events and 'target' not in events and events[-2:]==['release','disconnect']
    finally:
        if p.poll() is None:p.kill();p.wait()
        for f in (p.stdin,p.stdout,p.stderr):f.close()


def test_release_failure_is_reported_not_claimed_success():
    now=[1.];backend=FakeBackend(lambda:now[0]);session=Session(profile()['follower_port'],backend=backend,clock=lambda:now[0])
    session.owned=True;session.armed=True;backend.close=lambda:['motor off readback failed']
    with pytest.raises(RuntimeError,match='readback'):session.close()
    assert session.owned and session.armed


def test_leader_registration_validates_without_opening_bus(tmp_path,monkeypatch):
    import studio.robot as module
    root=tmp_path/'app';(root/'config').mkdir(parents=True)
    cfg=profile();atomic_json(root/'config/robot.json',cfg)
    monkeypatch.setattr(module,'ROOT',root)
    leader=copy.deepcopy(reference_config()['follower']);leader['shoulder_pan']['homing_offset']=2
    port='/dev/serial/by-id/usb-YOUR_ROBOT-if00'
    with pytest.raises(ValueError):module.register_leader(cfg['follower_port'],leader)
    assert module.register_leader(port,leader)['ok']
    assert json.loads((root/'config/so101_leader.json').read_text())==leader
    assert module.reference_config()['leader']==leader


def test_camera_roles_are_swapped_before_any_usb_open(tmp_path,monkeypatch):
    from studio import engine as module
    e=Engine(load(),tmp_path)
    monkeypatch.setattr(module.Path,'glob',lambda *a:iter([Path('/dev/serial/by-id/fake')]))
    calls=[];monkeypatch.setattr(module,'Hardware',lambda *a:calls.append(a))
    with pytest.raises(ValueError,match='상단 카메라'):
        e.operation('connect',{'follower':'/dev/serial/by-id/fake','cameras':{'table':'/dev/video0'}})
    with pytest.raises(ValueError,match='32×32'):
        e.operation('connect',{'follower':'/dev/serial/by-id/fake','cameras':{'gripper':'realsense:123'}})
    assert calls==[]
    e.close()
