import os
os.environ.setdefault('MUJOCO_GL','egl')
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from studio.config import ROOT
from studio.pose_renderer import PoseRenderer
from studio.safe_pose import load_safe_pose
from studio.robot import NAMES


@pytest.mark.parametrize('kind',['so101','omx'])
def test_rendered_link_transforms_match_original_urdf_without_physics(tmp_path,kind,monkeypatch):
    monkeypatch.setattr(mujoco,'mj_step',lambda *args:pytest.fail('Physics must not run'))
    renderer=PoseRenderer(480,320,tmp_path)
    q=[.2,-.4,.3,.7,-.25,.5]
    try:
        image=renderer.render(q,kind)
        assert image.shape==(320,480,3) and image.std()>15
        model,data,_=renderer.models[kind]
        assert data.time==0 and data.ncon==0 and not np.any(model.geom_contype)
        cfg=json.loads((ROOT/('config/scene.json' if kind=='so101' else 'config/legacy_omx/scene.json')).read_text())
        root=ET.parse(cfg['model_source']).getroot()
        names=NAMES if kind=='so101' else ['joint1','joint2','joint3','joint4','joint5','gripper_joint_1','gripper_joint_2']
        angles=dict(zip(names,q+([-q[5]] if kind=='omx' else [])))
        child_names={j.find('child').get('link') for j in root.findall('joint')}
        base=next(l.get('name') for l in root.findall('link') if l.get('name') not in child_names)
        transform=np.eye(4);transform[:3,3]=cfg['base'];frames={base:transform};pending=list(root.findall('joint'))
        while pending:
            for j in pending[:]:
                parent=j.find('parent').get('link')
                if parent not in frames:continue
                origin=j.find('origin');local=np.eye(4)
                if origin is not None:
                    local[:3,3]=list(map(float,origin.get('xyz','0 0 0').split()))
                    local[:3,:3]=Rotation.from_euler('xyz',list(map(float,origin.get('rpy','0 0 0').split()))).as_matrix()
                if j.get('type')=='revolute':
                    axis=np.array(list(map(float,j.find('axis').get('xyz').split())))
                    local[:3,:3]=local[:3,:3]@Rotation.from_rotvec(axis*angles[j.get('name')]).as_matrix()
                frames[j.find('child').get('link')]=frames[parent]@local;pending.remove(j)
        for name,expected in frames.items():
            actual=data.body(name)
            assert actual.xpos==pytest.approx(expected[:3,3],abs=2e-6)
            assert actual.xmat.reshape(3,3)==pytest.approx(expected[:3,:3],abs=2e-5)
        assert renderer.render(q,kind) is None and renderer.frames==1
        q[1]+=.2;assert renderer.render(q,kind) is not None and data.time==0
    finally:renderer.close()
