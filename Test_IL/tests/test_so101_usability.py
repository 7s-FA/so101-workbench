"""Real IO is replaced; exercise the actual SO-101 command and admission paths."""
import io
import json
import math
import time
from types import SimpleNamespace as NS
import pytest
from studio.config import load
from studio.engine import Engine
from studio.execution import options,ChunkTimeline
from studio.reference import ReferenceCodec
from studio.robot import pose_to_angles
from studio import hardware,so101_worker
from teach_io.entry import EntryPoseChanged
from test_so101 import FakeBackend


def policy_engine(tmp_path,mode='original'):
    e=Engine(load(),tmp_path)
    e.policy_worker=NS(error=None,busy=True,result=None,close=lambda:None)
    e.policy_job={'fps':30};e.model_options=options({'mode':mode});e.model_codec=ReferenceCodec()
    e.model_clock=9.;e.model_seconds=30.;e.model_submit_at=10.;e.frames=lambda:{}
    if mode=='prefetch':e.model_timeline=ChunkTimeline(30,e.model_options)
    return e


@pytest.mark.parametrize('mode',['original','ensemble','prefetch'])
def test_policy_keeps_degrees_and_only_clamps_calibrated_limits(tmp_path,mode):
    e=policy_engine(tmp_path,mode)
    try:
        for index,(angle,grip) in enumerate([(150.,30.),(-150.,30.),(400.,150.),(-400.,-20.)]):
            now=10.+index*.1;target=[0.,0.,0.,0.,angle,grip]
            e.policy_worker.result=dict(action=target,actions=[target]*20,acquired=now,received_at=now)
            e.update_model(now,1/30)
            lo,hi=e.mapping.cfg['limits'][4]
            expected=max(lo,min(hi,math.radians(angle)))
            assert e.goal[4]==pytest.approx(expected,abs=.002)
            assert e.goal[5]==pytest.approx(max(0,min(100,grip))*1.74533/100,abs=.002)
    finally:e.close()


def test_reference_inference_latency_is_not_a_motor_fault(tmp_path):
    e=policy_engine(tmp_path)
    try:
        e.policy_worker.result=dict(action=[0,0,0,0,150,30],acquired=10.,received_at=11.2)
        e.update_model(11.2,1/30)
        assert math.degrees(e.goal[4])==pytest.approx(150,abs=.1)
        assert e.model_metrics['observation_age_ms']==pytest.approx(1200)
    finally:e.close()


def test_busy_inference_holds_last_target_but_missing_worker_still_errors(tmp_path):
    e=policy_engine(tmp_path);before=e.goal.copy()
    try:
        e.update_model(12.,1/30)
        assert e.goal==before and e.model_metrics['waiting_for_inference']
        e.policy_worker.busy=False
        with pytest.raises(RuntimeError,match='모델 응답'):e.update_model(12.1,1/30)
    finally:e.close()


def test_usb_pose_refresh_does_not_latch_fault_or_activate(monkeypatch):
    fake=NS(latest=dict(ok=True,angles=[0]*6),error=None,armed=False)
    calls=[]
    def moved(*args):calls.append('arm_check');raise EntryPoseChanged('moved before activation')
    fake.arm=moved;fake.read=lambda:dict(ok=True,angles=[.01]*6)
    fake.hold=lambda:calls.append('hold');fake.close=lambda **kw:calls.append('close')
    requests=[dict(op='arm',target=[0]*6,entry={},sent=time.monotonic()),dict(op='read',sent=time.monotonic()),dict(op='close')]
    incoming=io.StringIO(''.join(json.dumps(v)+'\n' for v in requests));out=io.StringIO()
    monkeypatch.setattr(so101_worker,'Session',lambda *args:fake)
    monkeypatch.setattr(so101_worker.sys,'argv',['worker','--port','offline'])
    monkeypatch.setattr(so101_worker.sys,'stdin',incoming);monkeypatch.setattr(so101_worker.sys,'stdout',out)
    monkeypatch.setattr(so101_worker.select,'select',lambda *args:([incoming],[],[]))
    monkeypatch.setattr(so101_worker.signal,'signal',lambda *args:None)
    so101_worker.main()
    replies=[json.loads(line) for line in out.getvalue().splitlines()]
    assert replies[1]['retryable']=='entry_pose_changed' and replies[2]['ok']
    assert fake.error is None and calls==['arm_check','close']


def test_rpc_preserves_retryable_pose_change_type(monkeypatch):
    f=hardware.Follower.__new__(hardware.Follower)
    stream=io.StringIO(json.dumps(dict(ok=False,retryable='entry_pose_changed',error='changed'))+'\n')
    f.process=NS(stdout=stream)
    monkeypatch.setattr(hardware.select,'select',lambda *args:([stream],[],[]))
    with pytest.raises(EntryPoseChanged):f.receive(1)


def test_start_ramp_finishes_with_load_residual_and_keeps_monitoring():
    class LoadedBackend(FakeBackend):
        def send(self,pose):
            self.writes.append(pose.copy());self.q=pose_to_angles(pose);self.q[1]-=math.radians(8)
            return pose
    now=[1.];backend=LoadedBackend(lambda:now[0])
    session=so101_worker.Session(so101_worker.profile()['follower_port'],backend=backend,clock=lambda:now[0])
    start=session.read()['angles'];goal=start.copy();goal[1]=.4
    try:
        session.arm(goal,dict(start=start,goal=goal))
        for _ in range(120):now[0]+=.02;result=session.update(goal)
        assert result['aligned'] and session.entry is None
        assert abs(result['angles'][1]-goal[1])>math.radians(6)
        backend.heat=66
        with pytest.raises(Exception,match='temperature|온도|Temperature'):
            for _ in range(30):now[0]+=.2;session.read()
    finally:session.close()


def test_already_enabled_ramp_does_not_require_six_degree_settle(tmp_path):
    e=Engine(load(),tmp_path);now=time.monotonic();q=e.measured.copy();goal=q.copy();goal[1]+=.2
    e.hardware=NS(leader=None,follower=NS(armed=True),update=lambda *a,**kw:dict(angles=q))
    e.mode='preparing';e.after_prepare='idle';e.initial_arm=False;e.heartbeat=now
    e.entry=dict(start=q,goal=goal,began=now-3,duration=2)
    try:
        e.update(now,1/60)
        assert e.mode=='holding'
    finally:e.hardware=None;e.close()
