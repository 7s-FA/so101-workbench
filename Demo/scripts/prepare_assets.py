#!/usr/bin/env python3
"""Resolve upstream mesh paths; retain source geometry and document adaptations."""
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sources = {
    'so101': ROOT/'vendor/so_arm100/Simulation/SO101/so101_new_calib_camera.urdf',
    'omx': ROOT/'vendor/open_manipulator/open_manipulator_description/urdf/omx_f/omx_f.urdf',
}
names = {
    'so101': ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper'],
    'omx': ['joint1','joint2','joint3','joint4','joint5','gripper_joint_1','gripper_joint_2'],
}
models = {}
for kind, source in sources.items():
    tree = ET.parse(source)
    root = tree.getroot()
    for mesh in root.iter('mesh'):
        path = mesh.attrib['filename']
        if path.startswith('package://open_manipulator_description/'):
            resolved = ROOT/'vendor/open_manipulator/open_manipulator_description'/path.split('package://open_manipulator_description/')[1]
        else:
            resolved = source.parent/path
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        mesh.set('filename', str(resolved.resolve()))
    # Simulation-only copy: no hardware plugins, no physical follower connection.
    for node in list(root):
        if node.tag in ('ros2_control','gazebo','transmission'):
            root.remove(node)
    # Drive both OMX fingers explicitly to make mimic behavior deterministic in 5.1.
    for joint in root.findall('joint'):
        mimic = joint.find('mimic')
        if mimic is not None:
            joint.remove(mimic)
    joints = {j.attrib['name']:j for j in root.findall('joint')}
    limits = [[float(joints[n].find('limit').get('lower')),float(joints[n].find('limit').get('upper'))] for n in names[kind]]
    if kind == 'omx':
        # Upstream +/-2pi fingers are not meaningful for a pinch gripper.
        limits[-2:] = [[0.,1.1],[-1.1,0.]]
        for n, (lo,hi) in zip(names[kind], limits):
            joints[n].find('limit').set('lower',str(lo))
            joints[n].find('limit').set('upper',str(hi))
    out = ROOT/'assets/generated'/f'{kind}.urdf'
    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out, encoding='utf-8', xml_declaration=True)
    models[kind] = dict(joints=names[kind], limits=limits, home=([0.,0.,0.,0.,0.,0.5] if kind=='so101' else [0.,-0.4,0.3,0.3,0.,0.5,-0.5]),
                        mimic=(kind=='omx'), max_command_speed=3.0,
                        urdf=str(out.relative_to(ROOT)), position=[0.,-0.45 if kind=='so101' else 0.45,0.03])
models['omx']['gripper_drive']=dict(stiffness=8.,damping=.2,effort_limit_sim=.2,velocity_limit_sim=.6)
(ROOT/'config/models.json').write_text(json.dumps(models,indent=2)+'\n')
revisions = {}
for name in ['so_arm100','open_manipulator','so101_workshop','lerobot']:
    revisions[name] = subprocess.check_output(['git','-C',str(ROOT/'vendor'/name),'rev-parse','HEAD'],text=True).strip()
(ROOT/'config/source_revisions.json').write_text(json.dumps(revisions,indent=2)+'\n')
print('Prepared camera SO-101 and OMX-F URDFs:', list(models))
