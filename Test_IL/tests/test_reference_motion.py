import math
import pytest
from studio.hardware import mapping
from studio.engine import validate_replay
from studio.config import fingerprint,load
from teach_io.tracking import ReferenceTracking
from teach_io.worker import Guard


def test_reference_target_transmits_without_old_lag_or_rate_cutoff():
    m=mapping();q=[0,0,0,0,0,.2];goal=[.5,-.4,.6,.2,-.3,.6]
    t=ReferenceTracking(m,q,0)
    result,status=t.update(goal,q,[{'pwm_percent':20}]*6,1/60)
    assert result==goal and status['pace']==1
    assert status['profile']=='lerobot_reference'
    assert status['compensation']['enabled'] is False
    g=Guard(m);g.last=q;g.last_time=0
    assert g.check(result,q,1/60,reference=True)==m.to_ticks(goal)
    with pytest.raises(ValueError):g.check([0,0,10,0,0,.2],q,2/60,reference=True)
    with pytest.raises(ValueError,match='Stale'):g.check(goal,q,1,reference=True)


def test_reference_keeps_stall_and_overload_protection():
    m=mapping();q=[0,0,0,0,0,.2];goal=[0,0,.5,0,0,.2]
    t=ReferenceTracking(m,q,0)
    with pytest.raises(ValueError,match='no measured movement'):
        for i in range(1,400):t.update(goal,q,[{'pwm_percent':20}]*6,i/60)
    t=ReferenceTracking(m,q,0)
    with pytest.raises(ValueError,match='output effort'):
        for i in range(1,140):t.update(goal,q,[{'pwm_percent':95}]*6,i/60)


def test_reference_recording_can_replay_its_actual_direct_targets():
    meta={'robot_type':'so101','synthetic':False,'fingerprint':fingerprint(),'units':'rad','control_profile':'lerobot_reference'}
    rows=[{'t':0.,'action':[0,0,0,0,0,.2]},{'t':1/30,'action':[.5,-.4,.6,.2,-.3,.6]}]
    validate_replay(meta,rows,fingerprint(),mapping())
    rows[1]['action'][2]=10
    with pytest.raises(ValueError):validate_replay(meta,rows,fingerprint(),mapping())
    assert load()['teleop_fps']==60 and load()['record_fps']==30
