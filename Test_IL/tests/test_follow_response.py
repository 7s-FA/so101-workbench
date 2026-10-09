import math
from types import SimpleNamespace
import pytest
from studio.legacy_omx import Hardware, mapping
from studio.reference import ReferenceCodec
from studio.engine import Engine
from studio.config import load
from teach_io.bus import Bus


@pytest.mark.parametrize('comm', [-3001, -3002])
def test_leader_position_retry_discards_partial_pose(comm):
    h=Hardware.__new__(Hardware);calls=[]
    def read(port,motor,address):
        calls.append(motor)
        assert address==132
        if len(calls)==3:return 0,comm,0
        return (100+motor if len(calls)<=2 else 200+motor),0,0
    h.leader=SimpleNamespace(port=None,packet=SimpleNamespace(read4ByteTxRx=read))
    assert h.leader_ticks()==list(range(201,207))
    assert calls==[1,2,3,1,2,3,4,5,6]


@pytest.mark.parametrize('comm,status,attempts', [(-3001,0,3),(0,128,1),(-1001,0,1)])
def test_leader_position_retry_is_bounded_and_keeps_motor_faults(comm,status,attempts,monkeypatch):
    monkeypatch.setattr('studio.legacy_omx.time.monotonic',lambda:0.)
    h=Hardware.__new__(Hardware);calls=[]
    def read(port,motor,address):
        calls.append(motor)
        return 0,comm,status
    h.leader=SimpleNamespace(port=None,packet=SimpleNamespace(read4ByteTxRx=read))
    with pytest.raises(RuntimeError,match='리더암 ID 1'):
        h.leader_ticks()
    assert len(calls)==attempts


def test_leader_position_retry_does_not_wait_indefinitely(monkeypatch):
    clock=iter([0.,.09]);monkeypatch.setattr('studio.legacy_omx.time.monotonic',lambda:next(clock))
    h=Hardware.__new__(Hardware);calls=[]
    def read(port,motor,address):
        calls.append(motor)
        return 0,-3001,0
    h.leader=SimpleNamespace(port=None,packet=SimpleNamespace(read4ByteTxRx=read))
    with pytest.raises(RuntimeError):h.leader_ticks()
    assert calls==[1]


def test_leader_position_signed_ticks_are_preserved():
    h=Hardware.__new__(Hardware)
    h.leader=SimpleNamespace(port=None,packet=SimpleNamespace(
        read4ByteTxRx=lambda port,motor,address:(0xffffffff if motor==3 else 2048,0,0)))
    assert h.leader_ticks()==[2048,2048,-1,2048,2048,2048]


def test_every_control_step_exchanges_fresh_feedback():
    calls=[]
    h=Hardware.__new__(Hardware)
    h.follower=SimpleNamespace(request=lambda op,**kw:calls.append((op,kw)) or {'angles':[0]*6})
    assert h.update([0]*6)['angles']==[0]*6
    assert len(calls)==1 and calls[0][0]=='target'


def test_leader_outside_range_saturates_instead_of_aborting(tmp_path):
    h=Hardware.__new__(Hardware);h.codec=ReferenceCodec(__import__('json').loads((__import__('studio.config',fromlist=['ROOT']).ROOT/'config/legacy_omx/lerobot_calibration.json').read_text()));h.follower_mapping=mapping()
    h.leader_ticks=lambda:[5000,-100,5000,2048,2050,1500]
    target,raw=h.target()
    e=Engine(load(),tmp_path);e.mapping=mapping();target=e.limit_goal(target)
    e.mapping.to_ticks(target)
    assert target[1]==pytest.approx(math.radians(-120))
    assert target[2]==pytest.approx(math.radians(90))
    assert abs(target[4])<.01
    e.close()


def test_teleop_sends_reference_goal_without_software_rate_or_gap_cap(tmp_path):
    e=Engine(load(),tmp_path);e.mode='teleop';e.ping()
    measured=[0,0,math.radians(90),0,0,.2]
    e.measured=measured.copy();e.goal=measured.copy();calls=[]
    class F:
        armed=True
    class H:
        follower=F();leader=True
        def target(self):return [0,0,math.radians(40),.1,0,.2],[2048]*6
        def update(self,q,active=False):
            calls.append(list(q))
            return dict(angles=measured,tracking={'pace':1},motors=[])
        def close(self,release=True):pass
    e.hardware=H()
    import time
    for _ in range(8):e.update(time.monotonic(),.05)
    assert all(q[2]==pytest.approx(math.radians(40)) for q in calls)
    assert calls[-1][3]==pytest.approx(.1) # other joints still follow
    e.close()


class Packet:
    def __init__(self,short=False,error=0):self.calls=0;self.short=short;self.error=error
    def readTxRx(self,port,motor,address,size):
        self.calls+=1
        b=bytearray(23)
        for offset,n,value in [(0,2,-100),(2,2,-10),(4,4,-2),(8,4,2048),(16,4,2050),(20,2,120 if motor<=13 else 50),(22,1,25)]:
            b[offset:offset+n]=(value & ((1<<(8*n))-1)).to_bytes(n,'little')
        return list(b[:10] if self.short else b),0,self.error


def packed_bus(**kw):
    b=Bus.__new__(Bus);b.packet=Packet(**kw);b.port=None;b.read=lambda m,a,n:0
    return b


def test_packed_feedback_preserves_health_and_values():
    b=packed_bus();rows=b.sample(detailed=True,packed=True)
    assert b.packet.calls==6
    assert rows[0]['ticks']==2048 and rows[0]['load_percent']==-1
    assert rows[3]['current_ma']==-10 and rows[3]['temperature']==25
    assert rows[0]['trajectory_ticks']==2050 and rows[0]['pwm_percent']==pytest.approx(-11.3)
    assert rows[0]['velocity_rad_s']<0


@pytest.mark.parametrize('kw',[{'short':True},{'error':128}])
def test_packed_feedback_rejects_partial_or_motor_error(kw):
    with pytest.raises(RuntimeError):packed_bus(**kw).sample(detailed=True,packed=True)


def test_reference_goals_with_slow_follower_feedback(tmp_path):
    from teach_io.tracking import ReferenceTracking
    e=Engine(load(),tmp_path)
    measured=[0.,0.,math.radians(80),0.,0.,.2]
    e.measured=measured.copy();e.goal=measured.copy();e.mode='teleop'
    tracker=ReferenceTracking(e.mapping,measured,0.)
    clock=0.;sent=measured.copy()
    class H:
        follower=SimpleNamespace(armed=True);leader=True
        def target(self):return [0.,0.,math.radians(35),0.,0.,.2],[2048]*6
        def update(self,q,active=False):
            nonlocal sent
            # Deterministic 150 ms servo response, rather than instant ideal following.
            measured[:]=[a+(b-a)*(1/30)/.15 for a,b in zip(measured,sent)]
            sent,status=tracker.update(q,measured,[{'pwm_percent':20}]*6,clock,correction=False,context='unknown',phase='unknown')
            return dict(angles=list(measured),tracking=status,motors=[])
        def close(self,release=True):pass
    e.hardware=H()
    for i in range(1,181):
        clock=i/30;e.heartbeat=clock;e.update(clock,1/30)
    assert math.degrees(measured[2])==pytest.approx(35,abs=.5)
    e.close()


def test_persistent_blockage_and_large_disturbance_still_stop(tmp_path):
    from teach_io.tracking import Tracking
    from studio.engine import Engine
    e=Engine(load(),tmp_path);q=[0.,0.,.5,0.,0.,.2]
    tracker=Tracking(e.mapping,q,0.,compensation_enabled=False,evaluation=True)
    goal=q.copy();goal[2]-=math.radians(6.5)
    with pytest.raises(ValueError,match='no tracking recovery'):
        for i in range(1,220):tracker.update(goal,q,[{'pwm_percent':20}]*6,i/30,correction=False)
    tracker=Tracking(e.mapping,q,0.,compensation_enabled=False,evaluation=True)
    displaced=q.copy();displaced[2]+=.3
    with pytest.raises(ValueError,match='target lag'):tracker.update(q,displaced,[{'pwm_percent':20}]*6,.03)
    e.close()
