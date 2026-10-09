"""Read-only triangle contact checks in millimetres for registration trials."""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import fcl
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import Reference,JOINTS,read_json,atomic_json
from scene_assets import mesh,collision_model
from alignment import aligned_scene


def frames(root,angles):
    known={'base_link':np.eye(4)};q=dict(zip(JOINTS,angles));pending=list(root.findall('joint'))
    while pending:
        for joint in list(pending):
            parent=joint.find('parent').get('link')
            if parent not in known:continue
            origin=joint.find('origin');tf=np.eye(4)
            if origin is not None:
                tf[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')*1000
                tf[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
            if joint.get('type')!='fixed':
                turn=np.eye(4);turn[:3,:3]=Rotation.from_rotvec(np.fromstring(joint.find('axis').get('xyz'),sep=' ')*q[joint.get('name')]).as_matrix();tf=tf@turn
            known[joint.find('child').get('link')]=known[parent]@tf;pending.remove(joint)
    return known


def check(scene,ref,offset):
    scene=aligned_scene(scene,ref,offset);root=ET.parse(scene['robot']).getroot();rows=[]
    fingers=[]
    for link in root.findall('link'):
        for col in link.findall('collision'):
            name=col.get('name','')
            if name not in ('xlerobot_fixed_finger','xlerobot_moving_finger','razer_fixed_pad','razer_moving_pad'):continue
            geo=col.find('geometry/mesh');value=mesh(geo.get('filename'))
            value.apply_scale(np.fromstring(geo.get('scale','1 1 1'),sep=' ')*1000)
            origin=col.find('origin');tf=np.eye(4)
            tf[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')*1000
            tf[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
            value.apply_transform(tf);fingers.append((link.get('name'),name,collision_model(value)))
    for kind,part in scene['parts'].items():
        value=mesh(part['mesh']);value.apply_scale(1000);tf=np.asarray(part['initial_pose_mm'])
        body=fcl.CollisionObject(collision_model(value),fcl.Transform(tf[:3,:3],tf[:3,3]))
        for index in [part['grasp_step']-1,part['grasp_step']]:
            poses=frames(root,ref.mapping.angles(ref.steps[index]['ticks']));gaps={}
            for link,name,model in fingers:
                t=poses[link];other=fcl.CollisionObject(model,fcl.Transform(t[:3,:3],t[:3,3]))
                gaps[name]=float(fcl.distance(other,body,fcl.DistanceRequest(),fcl.DistanceResult()))
            rows.append(dict(part=kind,step=index+1,gaps_mm=gaps))
    return rows


if __name__=='__main__':
    ref=Reference();scene=read_json(sys.argv[1])
    for distance in [0,6,9,12]:
        offset=[.87255274*distance,.48851993*distance,0]
        print('OFFSET',np.round(offset,3).tolist(),check(scene,ref,offset),flush=True)
