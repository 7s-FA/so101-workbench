import copy
import json
from pathlib import Path
import queue
import threading
import time
import sys
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from studio import calibration as c,robot
from studio.config import load
from studio.engine import Engine
from studio.server import create_app
sys.path.insert(0,str(robot.OCV))


@pytest.fixture
def isolated(tmp_path,monkeypatch):
    cfg=robot.profile();cal=robot.reference_config()['follower'];root=tmp_path/'app';(root/'config').mkdir(parents=True)
    (root/'config/robot.json').write_text(json.dumps(cfg))
    monkeypatch.setattr(robot,'ROOT',root);monkeypatch.setattr(c,'ROOT',root)
    return root,cfg,copy.deepcopy(cal)


def test_selecting_follower_file_uses_local_wrapper_and_preserves_source(isolated):
    root,cfg,cal=isolated;source=Path(cfg['follower_calibration']);before=source.read_bytes()
    cal['shoulder_pan']['homing_offset']-=1
    result=c.register('follower',cfg['follower_port'],'selected_arm',cal)
    assert result['coordinate_changed'] and source.read_bytes()==before
    from teaching_profile import load_profile
    args,_=load_profile(robot.profile()['follower_profile'],follower_only=True)
    assert args.calibration_dir/(args.follower_id+'.json')==Path(result['path'])
    assert robot.reference_config()['follower']==cal
    assert robot.profile()['leader_calibration']==cfg['leader_calibration']
    assert any(row['path']==result['path'] for row in c.catalog())
    assert c.register('follower',cfg['follower_port'],'selected_arm',cal)['coordinate_changed'] is False


def test_registration_validation_happens_before_file_changes(isolated):
    root,cfg,cal=isolated;before=(root/'config/robot.json').read_bytes()
    for role,port,name in [('leader',cfg['follower_port'],'same'),('wrong',cfg['follower_port'],'x'),('follower',cfg['follower_port'],'../../outside')]:
        with pytest.raises(ValueError):c.register(role,port,name,cal)
    cal['gripper']['range_max']=cal['gripper']['range_min']
    with pytest.raises(ValueError):c.register('follower',cfg['follower_port'],'invalid',cal)
    assert (root/'config/robot.json').read_bytes()==before


def test_new_motor_frame_refreshes_mapping_and_preserves_other_role(isolated,tmp_path):
    root,cfg,cal=isolated;e=Engine(load(),tmp_path/'data')
    try:
        old_leader=e.motor_reference['leader']
        old=e.fingerprint;cal['shoulder_lift']['homing_offset']+=1
        e.operation('calibration_select',dict(role='follower',port=cfg['follower_port'],name='new_arm',calibration=cal))
        assert e.fingerprint!=old and e.motor_reference['follower']==cal
        assert e.motor_reference['leader']==old_leader and e.hardware is None
        e.calibration.status['busy']=True
        with pytest.raises(ValueError,match='보정'):e.operation('connect',{})
        with pytest.raises(ValueError,match='보정'):e.operation('teleop',{})
    finally:e.calibration.status['busy']=False;e.close()


def test_lost_ui_cancels_calibration_and_does_not_open_control_bus(tmp_path):
    e=Engine(load(),tmp_path);cancel=threading.Event()
    try:
        e.calibration.worker=NS(finished=threading.Event(),events=queue.Queue(),cancel=cancel,state='RANGE')
        e.calibration.status['busy']=True;e.heartbeat=time.monotonic()-3
        e.update(time.monotonic(),.02)
        assert cancel.is_set() and e.hardware is None
        e.operation('stop',{});assert cancel.is_set()
    finally:e.calibration.worker=None;e.close()


def test_test_mode_cannot_calibrate_or_register_files(tmp_path):
    with TestClient(create_app(data_root=tmp_path,launch_sim=False,test_mode=True)) as client:
        assert client.get('/api/calibration').status_code==200
        assert client.get('/api/calibration/reference').status_code==200
        for op in ('calibration_start','calibration_select','calibration_apply','calibration_command'):
            response=client.post('/api/command/'+op,headers={'X-Studio-Command':'1'},json={})
            assert response.status_code==409


def test_manager_applies_selected_file_through_shared_official_worker(isolated,monkeypatch):
    import motor_calibration
    from test_motor_calibration import FakeBus
    root,cfg,cal=isolated;bus=FakeBus();original=copy.deepcopy(bus.calibration)
    destination=root/'config/calibration/il_follower/test_saved.json';destination.parent.mkdir(parents=True)
    arm=NS(id='test_saved',bus=bus,calibration_fpath=destination,calibration=original)
    monkeypatch.setattr(motor_calibration,'make_arm',lambda *args:arm)
    original_glob=Path.glob
    monkeypatch.setattr(Path,'glob',lambda p,pattern:iter([Path(cfg['follower_port'])]) if str(p)=='/dev/serial/by-id' else original_glob(p,pattern))
    manager=c.CalibrationManager()
    manager.start('follower',cfg['follower_port'],'test_saved',preset=cal)
    assert manager.worker.finished.wait(3)
    manager.poll()
    assert manager.status['state']=='DONE' and manager.status['error'] is None
    assert manager.status['saved']['calibration']==cal
    assert not bus.is_connected and ('middle',) not in bus.calls
    assert json.loads(destination.read_text())==cal
    recovery=json.loads(Path(manager.status['recovery']).read_text())
    assert recovery['previous_eeprom']==motor_calibration.serializable(original)
    manager.close()
