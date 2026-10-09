"""Display-only admission and failed transport regressions; no USB access."""
import io
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from studio.hardware import audit_entry, mapping
from teach_io.client import Follower
from teach_io.entry import validate_entry, EntryPoseChanged


def test_il_admission_is_numeric_only_and_legacy_still_requires_geometry():
    q=[0.,0.,0.,0.,0.,.5]
    plan=audit_entry(q,q)
    assert plan['collision_checked'] is False
    plan['pairs']=[['link7','table'],['link6','table']]
    assert validate_entry(plan,q,q,mapping(),require_collision_check=False)==q
    with pytest.raises(ValueError,match='collision-checked'):
        validate_entry(plan,q,q,mapping())
    with pytest.raises(ValueError):audit_entry(q,[float('nan')]+q[1:])
    with pytest.raises(ValueError):audit_entry(q,[100.]+q[1:])


@pytest.mark.parametrize('exited',[False,True])
def test_failed_pipe_is_latched_without_retry_or_losing_torque_ownership(exited):
    follower=Follower.__new__(Follower)
    pipe=Mock();pipe.write.side_effect=BrokenPipeError('broken pipe')
    follower.process=SimpleNamespace(poll=lambda:2 if exited else None,stdin=pipe)
    follower.armed=True;follower.connection_error=None
    for _ in range(3):
        with pytest.raises(RuntimeError,match='다시 연결'):
            follower.request('target',target=[0]*6)
    assert pipe.write.call_count==(0 if exited else 1)
    assert follower.armed  # Explicit disconnect must still require support/release.


def test_changed_read_only_pose_does_not_poison_the_connection():
    follower=Follower.__new__(Follower)
    follower.process=SimpleNamespace(poll=lambda:None,stdin=io.StringIO())
    follower.connection_error=None;follower.armed=False
    follower.receive=Mock(side_effect=[EntryPoseChanged('changed'),{'ok':True,'angles':[0]*6}])
    with pytest.raises(EntryPoseChanged):follower.request('arm')
    assert follower.request('read')['ok']
    assert follower.connection_error is None


def test_dead_pipe_cleanup_finishes_so_release_can_continue():
    follower=Follower.__new__(Follower)
    stdin=Mock();stdin.close.side_effect=BrokenPipeError('closed during flush')
    stdout=Mock()
    follower.process=SimpleNamespace(poll=lambda:2,stdin=stdin,stdout=stdout)
    follower.armed=True
    follower.close()
    assert not follower.armed
    stdout.close.assert_called_once()
