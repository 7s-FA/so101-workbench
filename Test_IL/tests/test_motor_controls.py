"""Control semantics and release recovery; all motors/USB are test doubles."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from studio.config import load
from studio.engine import Engine
from studio.legacy_omx import Hardware
from test_studio import FakeHardware, FakeCamera, frames, sample


def connected(tmp_path, armed=True):
    e=Engine(load(),tmp_path)
    h=FakeHardware(e.measured)
    h.follower=SimpleNamespace(armed=armed)
    h.owned=armed
    def release():
        h.follower.armed=False;h.owned=False
        return dict(angles=h.q,torque_on=False,release_required=False)
    h.release_torque=Mock(side_effect=release)
    e.hardware=h;e.latest=dict(torque_on=armed)
    e.mode='holding' if armed else 'idle'
    return e,h


def test_stop_is_hold_and_release_requires_support(tmp_path):
    e,h=connected(tmp_path)
    try:
        e.mode='teleop';e.operation('stop',{})
        assert e.mode=='holding' and h.follower.armed
        with pytest.raises(ValueError,match='Support'):e.operation('release_torque',{})
        h.release_torque.assert_not_called()
        e.operation('release_torque',{'supported':True})
        assert e.mode=='idle' and e.hardware is h and not h.follower.armed
        assert e.torque_status()==dict(state='off',release_required=False)
    finally:e.close()


def test_release_keeps_cameras_and_recording_for_save(tmp_path):
    e,h=connected(tmp_path);c=FakeCamera();c.close=Mock();e.cameras={'gripper':c}
    try:
        e.store.begin({})
        for i in range(3):e.store.append(sample(i/30),frames())
        e.mode='recording'
        e.operation('release_torque',{'supported':True})
        assert e.store.draft and e.store.count==3
        assert e.cameras['gripper'] is c and h.leader
        c.close.assert_not_called()
        assert not e.after_prepare and e.pending is None
    finally:e.close()


def test_external_torque_is_visible_and_disconnect_requires_support(tmp_path):
    e,h=connected(tmp_path,False);e.latest={'torque_on':True}
    try:
        e.publish();assert e.public['torque']['state']=='on'
        with pytest.raises(ValueError,match='Support'):e.operation('disconnect',{})
        assert e.hardware is h
        with pytest.raises(ValueError,match='모터 힘'):e.operation('model_load',{})
    finally:e.close()


def test_failed_release_keeps_recovery_and_does_not_claim_torque_off(tmp_path):
    e,h=connected(tmp_path)
    h.release_torque.side_effect=RuntimeError('no readback')
    try:
        with pytest.raises(RuntimeError):e.operation('release_torque',{'supported':True})
        e.publish()
        assert e.mode=='fault' and e.hardware is h
        assert e.public['torque']==dict(state='unknown',release_required=True)
        h.close=Mock(side_effect=RuntimeError('release failed'))
        with pytest.raises(RuntimeError):e.operation('disconnect',{'supported':True})
        assert e.hardware is h
    finally:
        h.close=Mock();e.close()


def test_hardware_release_closes_owner_before_verifying_and_only_reopens_read_only(monkeypatch):
    h=Hardware.__new__(Hardware);h.owned=True;h.port='fixture';h.leader=object();events=[]
    old=Mock();old.close.side_effect=lambda:events.append('close')
    h.follower=old
    h.release_motors=Mock(side_effect=lambda:events.append('release+verify'))
    new=Mock();new.request.return_value=dict(angles=[0]*6,torque_on=False,release_required=False)
    h.open_follower=Mock(side_effect=lambda:events.append('read-only reopen') or new)
    assert not h.release_torque()['torque_on']
    assert events==['close','release+verify','read-only reopen']
    new.arm.assert_not_called()
    new.request.assert_called_once_with('read')


def test_no_automatic_reopen_after_failed_release():
    h=Hardware.__new__(Hardware);h.owned=True;h.follower=Mock()
    h.release_motors=Mock(side_effect=RuntimeError('failed'));h.open_follower=Mock()
    with pytest.raises(RuntimeError):h.release_torque()
    h.open_follower.assert_not_called();assert h.owned


def test_release_readback_must_be_off():
    h=Hardware.__new__(Hardware);h.owned=True;h.follower=Mock()
    h.release_motors=Mock();h.open_follower=Mock(return_value=Mock())
    h.read=Mock(return_value=dict(torque_on=True))
    with pytest.raises(RuntimeError,match='not confirmed'):h.release_torque()
    assert h.owned
