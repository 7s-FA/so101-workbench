"""SO-101 uses official motor coordinates without a second model-zero layer."""
import copy
import math
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from studio.config import load
from studio.engine import Engine
from studio.hardware import Hardware
from studio.robot import Mapping, pose_to_angles
from studio.server import create_app
from test_studio import sample,frames


def test_follower_target_keeps_calibrated_coordinates_and_range_clamp():
    hardware=Hardware.__new__(Hardware);hardware.follower_mapping=Mapping()
    calibrated=pose_to_angles([10,-20,30,40,50,35]);calibrated[0]=100
    calls=[]
    hardware.follower=NS(request=lambda op:(calls.append(op) or dict(
        leader_target=calibrated,leader_angles=[-1]*6,leader_ticks=[1234]*6)))
    target,ticks=hardware.target()
    assert target==hardware.follower_mapping.clamp_target(calibrated)
    assert target[4]==pytest.approx(math.radians(50)) and ticks==[1234]*6 and calls==['read']


def test_recorded_angles_are_not_shifted_by_old_model_metadata(tmp_path):
    engine=Engine(load(),tmp_path)
    try:
        raw=pose_to_angles([0,-20,30,40,50,35])
        old=dict(roles={'follower':{'offsets_deg':[0,0,0,0,-90]}})
        engine.store.begin(dict(robot_type='so101',model_zero=old))
        for t in (0,.03):engine.store.append(sample(t,raw),frames())
        engine.store.save('Success');engine.select(1)
        original=copy.deepcopy(engine.episode_meta)
        engine.mode='preview';engine.publish()
        assert engine.public['display_angles']==raw
        assert engine.samples[0]['action']==raw and engine.samples[0]['state']==raw
        assert engine.episode_meta==original
    finally:engine.close()


def test_removed_adjustment_api_is_unavailable_but_motor_calibration_remains(tmp_path):
    with TestClient(create_app(load(),tmp_path,launch_sim=False,test_mode=True)) as client:
        assert client.get('/api/joint-zero').status_code==404
        assert client.get('/static/joint_zero.js').status_code==404
        for operation in ('zero_save','zero_preview','zero_preview_clear'):
            response=client.post('/api/command/'+operation,headers={'X-Studio-Command':'1'},json={})
            assert response.status_code==409
        assert 'joint_zero' not in client.get('/api/status').json()
        assert client.get('/api/calibration').status_code==200
