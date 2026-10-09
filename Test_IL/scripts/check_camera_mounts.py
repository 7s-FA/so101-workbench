"""Mesh topology, attachment interfaces, camera clearance and gripper sweep checks."""
from pathlib import Path
import json,sys,math,xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import fcl
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'assets/camera_mounts'

def obj(mesh):
    m=fcl.BVHModel();m.beginModel(len(mesh.vertices),len(mesh.faces));m.addSubModel(mesh.vertices,mesh.faces);m.endModel();return fcl.CollisionObject(m)
def clearance(a,b):
    aa,bb=obj(a),obj(b)
    contacts=fcl.collide(aa,bb,fcl.CollisionRequest(num_max_contacts=1),fcl.CollisionResult())
    d=fcl.distance(aa,bb,fcl.DistanceRequest(),fcl.DistanceResult())
    return dict(intersections=int(contacts),distance_mm=float(d))
def origin(e):
    m=np.eye(4)
    if e is not None:
        m[:3,:3]=Rotation.from_euler('xyz',list(map(float,e.get('rpy','0 0 0').split()))).as_matrix();m[:3,3]=np.array(list(map(float,e.get('xyz','0 0 0').split())))*1000
    return m

def visual_mesh(v):
    e=v.find('geometry/mesh');m=trimesh.load(e.get('filename'),force='mesh');m.apply_scale(np.array(list(map(float,e.get('scale','1 1 1').split())))*1000);m.apply_transform(origin(v.find('origin')));return m

def check():
    data=json.loads((OUT/'assembly.json').read_text());metrics={};failures=[]
    for row in data['print_parts']:
        m=trimesh.load(OUT/row['file'],force='mesh')
        if not m.is_watertight or len(m.split())!=1 or m.volume<=0:failures.append(row['file']+' topology')
    meshes={}
    for item in data['attachments']:
        m=trimesh.load(item['file'],force='mesh');m.apply_transform(np.array(item['transform_mm']));meshes[item['name']]=m
    for a,b in [('overhead_top_d435','overhead_d435'),('wrist_xlerobot_mount','wrist_32x32_camera')]:
        metrics[a+'__'+b]=clearance(meshes[a],meshes[b])
    if metrics['wrist_xlerobot_mount__wrist_32x32_camera']['intersections']:failures.append('Wrist PCB/back components collide with bracket')
    if data['top_interface_max_surface_delta_mm']>.01:failures.append('Original socket shape changed')
    if metrics['overhead_top_d435__overhead_d435']['distance_mm']<.45:failures.append('D435 rear clearance')
    # The PCB standoffs intentionally meet the PCB mounting face; report separately.
    tree=ET.parse(ROOT.parent/'Test_teach/assets/so101/so101_teach.urdf')
    fixed=trimesh.util.concatenate([visual_mesh(v) for v in tree.findall("link[@name='gripper_link']/visual") if v.find('geometry/mesh') is not None])
    for name in ('wrist_xlerobot_mount','wrist_32x32_camera'):
        c=clearance(meshes[name],fixed);metrics[name+'__fixed_gripper']=c
        if c['intersections']:failures.append(name+' intersects fixed gripper')
    moving=trimesh.util.concatenate([visual_mesh(v) for v in tree.findall("link[@name='moving_jaw_so101_v1_link']/visual") if v.find('geometry/mesh') is not None])
    joint=tree.find("joint[@name='gripper']");base=origin(joint.find('origin'));axis=np.array(list(map(float,joint.find('axis').get('xyz').split())))
    sweep=[]
    for deg in range(0,101,5):
        tf=base.copy();tf[:3,:3]=base[:3,:3]@Rotation.from_rotvec(axis*math.radians(deg)).as_matrix()
        part=moving.copy();part.apply_transform(tf)
        row={'opening_deg':deg}
        for name in ('wrist_xlerobot_mount','wrist_32x32_camera'):
            c=clearance(meshes[name],part);row[name]=c
            if c['intersections']:failures.append(f'{name} intersects moving jaw at {deg}')
        sweep.append(row)
    # Static ready-pose check against the rest of the arm, in the same URDF frame.
    scene=json.loads((ROOT/'config/scene.json').read_text())
    names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']
    angles=dict(zip(names,np.radians(scene['start_angles_deg'])))
    transforms={'base_link':np.eye(4)};pending=list(tree.findall('joint'))
    while pending:
        progress=False
        for j in pending[:]:
            parent=j.find('parent').get('link')
            if parent not in transforms:continue
            tf=origin(j.find('origin'))
            if j.get('type')=='revolute':tf[:3,:3]=tf[:3,:3]@Rotation.from_rotvec(np.array(list(map(float,j.find('axis').get('xyz').split())))*angles[j.get('name')]).as_matrix()
            transforms[j.find('child').get('link')]=transforms[parent]@tf;pending.remove(j);progress=True
        if not progress:raise ValueError('URDF tree cycle')
    static={}
    for name in ('wrist_xlerobot_mount','wrist_32x32_camera'):
        part=meshes[name].copy();part.apply_transform(transforms['gripper_link'])
        rows={}
        for link in tree.findall('link'):
            link_name=link.get('name')
            if link_name in ('gripper_link','moving_jaw_so101_v1_link'):continue
            visuals=[visual_mesh(v) for v in link.findall('visual') if v.find('geometry/mesh') is not None]
            if not visuals:continue
            other=trimesh.util.concatenate(visuals);other.apply_transform(transforms[link_name])
            c=clearance(part,other);rows[link_name]=c
            if c['intersections']:failures.append(name+' intersects '+link_name+' at ready pose')
        for attachment in data['attachments']:
            other_name=attachment['name']
            if attachment['link']!='base_link':continue
            c=clearance(part,meshes[other_name]);rows[other_name]=c
            if c['intersections']:failures.append(name+' intersects '+other_name+' at ready pose')
        static[name]=rows
    report=dict(metrics=metrics,gripper_sweep=sweep,ready_pose=static,failures=failures,physical_fit_verified=False,
        scope='21 gripper positions, body/camera static geometry. Not full-arm motion, structural FEA, real hole fit or cable bend validation.',
        optical_delta_mm=data['rgb_position_delta_mm'],optical_delta_deg=data['optical_angle_delta_deg'],
        unchanged_socket_surface_delta_mm=data['top_interface_max_surface_delta_mm'])
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if failures:raise SystemExit(1)
if __name__=='__main__':check()
