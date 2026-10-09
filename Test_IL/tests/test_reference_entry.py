"""Regressions for measured load residuals; never accesses real hardware."""
import math
import pytest
from studio.legacy_omx import mapping
from teach_io.entry import CurrentPoseEntry
from teach_io.tracking import ReferenceTracking


def test_logged_shoulder_residual_and_gripper_contact_finish_reference_entry():
    goal=[math.radians(v) for v in [-1.14,-120,90,87.28,-2.81,19.95]]
    measured=[math.radians(v) for v in [-.53,-116.81,90.35,88.07,-2.90,40]]
    entry=CurrentPoseEntry(measured,goal,0,reference=True)
    tracker=ReferenceTracking(mapping(),measured,0)
    for i in range(1,600):
        now=i*.02
        target=entry.advance(goal,now)
        _,status=tracker.update(target,measured,[dict(pwm_percent=32,velocity_rad_s=0)]*6,now)
        if entry.reached(measured,stationary=status['compensation']['stationary'],effort=status['pwm_percent']):
            break
    else:pytest.fail('Stable recorded 3.19 degree shoulder residual should not abort entry')
    assert status['pace']==1 and status['profile']=='lerobot_reference'
    legacy=CurrentPoseEntry(measured,goal,0);legacy.elapsed=legacy.duration
    assert not legacy.reached(measured,stationary=True,effort=32)


def test_reference_entry_keeps_large_error_and_moving_arm_blocked():
    q=[0,0,0,0,0,.2]
    entry=CurrentPoseEntry(q,q,0,reference=True);entry.elapsed=entry.duration
    assert not entry.reached(q,stationary=False,effort=20)
    assert not entry.reached(q,stationary=True,effort=95)
    far=q.copy();far[1]=math.radians(7)
    assert not entry.reached(far,stationary=True,effort=20)
    near=q.copy();near[1]=math.radians(5.9)
    assert entry.reached(near,stationary=True,effort=65)


def test_reference_motion_reports_stationary_only_after_stable_feedback():
    q=[0,0,0,0,0,.2];t=ReferenceTracking(mapping(),q,0)
    motors=[dict(pwm_percent=30,velocity_rad_s=0)]*6
    _,status=t.update(q,q,motors,.02)
    assert not status['compensation']['stationary']
    for i in range(2,32):_,status=t.update(q,q,motors,i*.02)
    assert status['compensation']['stationary']
    moved=q.copy();moved[1]=math.radians(1)
    _,status=t.update(moved,moved,motors,.64)
    assert not status['compensation']['stationary']


def test_already_enabled_reset_uses_same_residual_acceptance(tmp_path):
    import time
    from types import SimpleNamespace
    from studio.engine import Engine
    from studio.config import load
    e=Engine(load(),tmp_path)
    q=[0,0,0,0,0,.2];measured=q.copy();measured[1]=math.radians(3.19)
    e.measured=measured;e.goal=q.copy();now=time.monotonic();e.heartbeat=now
    e.mode='preparing';e.initial_arm=False;e.after_prepare='discard_reset'
    e.entry=dict(start=q.copy(),goal=q.copy(),began=now-3,duration=2)
    e.hardware=SimpleNamespace(follower=SimpleNamespace(armed=True),leader=None,
        update=lambda *a,**kw:dict(angles=measured,tracking=dict(pwm_percent=32,compensation=dict(stationary=True))),
        close=lambda **kw:None)
    try:
        e.update(now,.02)
        assert e.mode=='holding'
        assert e.measured==measured
    finally:e.close()
