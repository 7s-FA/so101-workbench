import math
import time
import pytest
from studio.engine import Engine as SOEngine

class Engine(SOEngine):
    """Historical OMX regressions; not the active hardware profile."""
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw);self.mapping=mapping()

from studio.config import load
from studio.legacy_omx import mapping
from teach_io.entry import validate_entry
from teach_io.tracking import ReferenceTracking


def test_hold_does_not_turn_measured_overshoot_into_invalid_goal(tmp_path):
    e=Engine(load(),tmp_path)
    e.measured=[0,0,math.radians(90.966796875),0,0,-.006]
    e.hold()
    e.mapping.to_ticks(e.goal)
    assert e.goal[2]==math.radians(90) and e.goal[5]==0
    assert e.measured[2]>math.radians(90)
    e.close()


def test_limit_goal_respects_physical_limits_even_with_wider_calibration(tmp_path):
    e=Engine(load(),tmp_path)
    e.mapping.cfg['limits'][2]=[-3,3]
    assert e.limit_goal([0,0,2,0,0,.2])[2]==math.radians(90)
    with pytest.raises(ValueError):e.limit_goal([0,0,float('nan'),0,0,.2])
    e.close()


def test_small_observation_residual_is_not_an_unbounded_target():
    m=mapping();start=[0,0,math.radians(90.966796875),0,0,0];goal=[0,0,math.radians(70),0,0,.2]
    plan=dict(start=start,goal=goal,collision_checked=True)
    assert validate_entry(plan,start,goal,m,allow_start_residual=True)==start
    with pytest.raises(ValueError):validate_entry(plan,start,goal,m)
    far=start.copy();far[2]=math.radians(101)
    with pytest.raises(ValueError):validate_entry(dict(plan,start=far),far,goal,m,allow_start_residual=True)


def test_reference_boundary_does_not_stop_other_joints():
    m=mapping();q=[0,0,math.radians(90.97),0,0,.2]
    tracker=ReferenceTracking(m,q,0)
    target=[.2,.1,math.radians(110),.3,0,-.1]
    command,status=tracker.update(target,q,[{'pwm_percent':20}]*6,.02)
    assert command[2]==math.radians(90) and command[5]==0
    assert command[:2]==target[:2] and command[3]==target[3]
    assert status['desired_angles']==command
    m.to_ticks(command)


def test_prepare_uses_observed_geometry_but_bounded_goal(tmp_path):
    from types import SimpleNamespace
    e=Engine(load(),tmp_path);e.hardware=SimpleNamespace(follower=SimpleNamespace(armed=True),close=lambda **kw:None)
    e.measured=[0,0,math.radians(90.97),0,0,.2]
    observed=e.measured.copy();requests=[]
    old_pool=e.pool
    e.pool=SimpleNamespace(submit=lambda fn,start,goal:requests.append((start,goal)),shutdown=lambda **kw:None)
    e.prepare(observed,'discard_reset')
    assert requests[0][0]==observed
    assert requests[0][1][2]==math.radians(90)
    assert e.goal[2]==math.radians(90)
    e.close();old_pool.shutdown()


@pytest.mark.parametrize("shoulder_residual", [False, True])
def test_reference_worker_handles_boundary_entry_and_live_overshoot(tmp_path, shoulder_residual):
    import json,runpy,subprocess,select,sys
    from studio.config import TEACH,ROOT
    fake=runpy.run_path(str(TEACH/'tests/test_worker_protocol.py'))['FAKE']
    fake=fake.replace('self.q=[2048]*6','self.q=[2048,2048,3083,2048,2048,2200]')
    if shoulder_residual:
        fake=fake.replace('def positions(self,ticks):self.q=ticks',
                          'def positions(self,ticks):self.q=list(ticks);self.q[1]+=36')
    cfg=json.loads((ROOT/'config/legacy_omx/follower_calibration.json').read_text());cfg.update(port='/dev/fake',verified=True)
    path=tmp_path/'calibration.json';path.write_text(json.dumps(cfg))
    p=subprocess.Popen([sys.executable,'-c',fake,'--port','/dev/fake','--calibration',str(path),'--reference-control'],cwd=TEACH,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    def receive():
        assert select.select([p.stdout],[],[],3)[0]
        return json.loads(p.stdout.readline())
    def send(op,**kw):
        p.stdin.write(json.dumps(dict(op=op,sent=time.monotonic(),**kw))+'\n');p.stdin.flush()
        r=receive();assert r.get('ok'),r;return r
    try:
        assert receive()['ok']
        start=send('read')['angles'];goal=mapping().clamp_target(start)
        send('arm',target=goal,path=[goal],entry=dict(start=start,goal=goal,collision_checked=True),compensation_enabled=False)
        for _ in range(300):
            time.sleep(.01);r=send('target',target=goal)
            assert r['tracking']['command_angles'][2]<=math.radians(90)
            if r['aligned']:break
        assert r['aligned']
        leader=goal.copy();leader[2]=math.radians(110);leader[0]=.2
        r=send('target',target=leader)
        assert r['tracking']['command_angles'][2]==math.radians(90)
        assert r['tracking']['command_angles'][0]==.2
    finally:
        if p.poll() is None:
            p.send_signal(__import__('signal').SIGINT);p.wait(timeout=3)
        for f in (p.stdin,p.stdout,p.stderr):f.close()


def test_logged_folded_pose_has_ten_degree_observation_margin():
    m=mapping()
    q=[math.radians(v) for v in [-.62,-115.75,99.32,89.65,-2.9,19.78]]
    assert m.validate_observation(q)==q
    goal=m.clamp_target(q)
    assert goal[2]==math.radians(90)
    m.to_ticks(goal)
    plan=dict(start=q,goal=goal,collision_checked=True)
    assert validate_entry(plan,q,goal,m,allow_start_residual=True)==q
    edge=q.copy();edge[2]=math.radians(100)
    assert m.validate_observation(edge)==edge
    with pytest.raises(ValueError):m.to_ticks(q)
    outside=q.copy();outside[2]=math.radians(100.1)
    with pytest.raises(ValueError):m.validate_observation(outside)
    bad=q.copy();bad[2]=float('nan')
    with pytest.raises(ValueError):m.validate_observation(bad)
