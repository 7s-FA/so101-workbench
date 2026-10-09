"""Default visual pose follows the OpenCV snapshot, never a motor command."""
import json
import math

import pytest

from studio.config import load, fingerprint
from studio.engine import Engine
from studio.robot import NAMES, profile, pose_to_angles
from studio.safe_pose import load_safe_pose
from test_studio import FakeHardware


def test_opencv_reference_uses_saved_frame_and_reload_sees_new_pose(tmp_path):
    source=tmp_path/'safe.json'
    pose=dict(zip((n+'.pos' for n in NAMES),[0,-55,15,85,90,15]))
    doc=dict(steps=[dict(safe_boundary='start',pose=pose),dict(safe_boundary='end',pose=pose)],
             vision_plan=dict(preview_tool=dict(joint_offsets_deg=[0,0,0,0,-90])))
    source.write_text(json.dumps(doc))
    safe=load_safe_pose(source)
    assert safe['raw_angles'][4]==pytest.approx(math.pi/2)
    assert safe['model_angles'][4]==pytest.approx(0)
    assert safe['model_angles'][5]==safe['raw_angles'][5]
    pose['elbow_flex.pos']=20
    source.write_text(json.dumps(doc))
    assert load_safe_pose(source)['model_angles'][2]==pytest.approx(math.radians(20))
    assert load_safe_pose(source)['reference']['source_sha256']!=safe['reference']['source_sha256']
    invalid=json.loads(source.read_text());invalid['steps'][-1]['pose']['elbow_flex.pos']=25
    source.write_text(json.dumps(invalid))
    with pytest.raises(ValueError,match='안전자세'):load_safe_pose(source)


def test_default_pose_cannot_overwrite_live_targets_or_calibration(tmp_path):
    signature=fingerprint()
    e=Engine(load(),tmp_path/'data')
    try:
        safe=load_safe_pose(profile()['default_pose_program'])
        assert e.public['display_source']=='reference_pose'
        assert e.public['display_angles']==safe['model_angles']
        assert e.reference_pose==safe['raw_angles']
        assert e.hardware is None and fingerprint()==signature
        real=pose_to_angles([1,-20,25,30,40,35]);e.hardware=FakeHardware(real)
        e.measured=real.copy();e.goal=real.copy();e.publish()
        assert e.public['display_source']=='live_measured'
        assert e.public['display_angles']==real
        assert e.goal==real and not e.hardware.commands and fingerprint()==signature
    finally:e.close()
