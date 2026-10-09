#!/usr/bin/env python3
"""Offline mesh audit of the saved start pose and entry motion; no motor IO."""
import argparse,json,sys,math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parent/'Sim_real')]
from teach_io.mapping import Mapping
from arm_sim.demo_kinematics import ArmKinematics
from arm_sim.conveyor_hardware import clamp_boxes
parser=argparse.ArgumentParser()
parser.add_argument('--config',type=Path,default=ROOT/'config/scene.json')
parser.add_argument('--report',type=Path,default=ROOT/'reports/start_pose_mesh_audit.json')
args=parser.parse_args()
cfg=json.loads(args.config.read_text())
model=json.loads((ROOT.parent/'Sim_real/assets/generated/conveyor_demo/models.json').read_text())['omx']
kin=ArmKinematics(model,dict(position=[0,0],yaw=0,base_plate_thickness=0,demo_pitch_degrees=90),cfg['table']['height'])
mapping=Mapping(json.loads((ROOT/'config/follower_visual_checked.json').read_text()),allow_visual_confirmation=True)
start=np.radians(cfg['start_angles_deg']);end=kin.solve(cfg['rest_point'],start);end[4]=end[0]
root=ET.parse(cfg['model_source']).getroot();joints=root.findall('joint')
def origin(e):
 t=np.eye(4)
 if e is not None:
  t[:3,3]=np.fromstring(e.get('xyz','0 0 0'),sep=' ');t[:3,:3]=Rotation.from_euler('xyz',np.fromstring(e.get('rpy','0 0 0'),sep=' ')).as_matrix()
 return t
meshes={}
for link in root.findall('link'):
 parts=[]
 for c in link.findall('collision'):
  e=c.find('geometry/mesh');m=trimesh.load(e.get('filename'),force='mesh');m.apply_scale(np.fromstring(e.get('scale','1 1 1'),sep=' '));m.apply_transform(origin(c.find('origin')));parts.append(m)
 if parts:meshes[link.get('name')]=trimesh.util.concatenate(parts)
robot=trimesh.collision.CollisionManager();world=trimesh.collision.CollisionManager()
for name,mesh in meshes.items():robot.add_object(name,mesh)
def obstacle(name,mesh,xyz):
 tf=np.eye(4);tf[:3,3]=xyz;world.add_object(name,mesh,tf)
z=cfg['table']['height'];obstacle('table',trimesh.creation.box([*cfg['table']['size'],.03]),[*cfg['table']['center'],z-.015])
fixed={'table_height':z,'arms':{'omx':{'model':'omx','position':[0,0]}},'tables':[{'id':'table',**cfg['table']}],
'arm_fixtures':{'omx':{'table':'table','side':'left','clamp_side':'left','clamp_offsets':[-.065,.065],'clamp_pad_depth':.018}}}
for name,p,s in clamp_boxes(fixed,'omx'):obstacle('clamp_'+name,trimesh.creation.box(s),p)
for name,p in cfg['parts'].items():
 for area in ('source','destination'):
  mesh=trimesh.creation.box([.068,.068,.0035]) if name=='cap' else trimesh.load(p['jig'],force='mesh')
  obstacle(area+'_'+name,mesh,[*p[area],z+(.00175 if name=='cap' else 0)])
 mesh=trimesh.load(p['mesh'],force='mesh');obstacle(name,mesh,[p['source'][0]+p['offset_xy'][0],p['source'][1]+p['offset_xy'][1],z+p['seating_z']+.0005])
excluded={frozenset((j.find('parent').get('link'),j.find('child').get('link'))) for j in joints};excluded.add(frozenset(('link6','link7')))
result={'start_angles_deg':cfg['start_angles_deg'],'scope':'Saved start, joint-space entry, gripper opening and first-part approach (before descent). Nominal URDF collision meshes; unmodeled cable/camera and real clearances are not verified.','samples':0,'faults':[],'min_tool_height_above_table_m':1.}
first=next(iter(cfg['parts'].values()))
src=np.array([first['source'][0]+first['offset_xy'][0],first['source'][1]+first['offset_xy'][1],z+first['seating_z']+.0005+first['grasp_height']])
approach=np.array([src[0],src[1],max(.325,src[2]+.06)])
for phase in ('entry','open','first_approach'):
 for u in np.linspace(0,1,201):
  q=np.r_[start,-start[5]]
  if phase=='entry':q[:5]=start[:5]+(end-start[:5])*u
  elif phase=='open':q[:5]=end;q[5]=start[5]+(.75-start[5])*u;q[6]=-q[5]
  else:
   q[:5]=kin.solve(np.asarray(cfg['rest_point'])+(approach-np.asarray(cfg['rest_point']))*u,end);q[4]=q[0];q[5:]=[.75,-.75]
  mapping.to_ticks(q[:6].tolist());values=dict(zip(model['joints'],q));tf={'link0':np.eye(4)};tf['link0'][:3,3]=cfg['base']
  pending=list(joints)
  while pending:
   progressed=False
   for j in pending[:]:
    parent=j.find('parent').get('link');child=j.find('child').get('link')
    if parent not in tf:continue
    t=origin(j.find('origin'))
    if j.get('type')!='fixed':
     r=np.eye(4);r[:3,:3]=Rotation.from_rotvec(np.fromstring(j.find('axis').get('xyz'),sep=' ')*values[j.get('name')]).as_matrix();t=t@r
    tf[child]=tf[parent]@t;pending.remove(j);progressed=True
   if not progressed:raise RuntimeError('Disconnected URDF')
  for name in meshes:robot.set_transform(name,tf[name])
  _,pairs=robot.in_collision_internal(return_names=True)
  bad=[list(p) for p in pairs if frozenset(p) not in excluded]
  _,pairs=robot.in_collision_other(world,return_names=True)
  bad.extend([list(p) for p in pairs if p[0]!='link0'])
  if bad:result['faults'].append(dict(phase=phase,u=float(u),pairs=bad))
  result['samples']+=1;result['min_tool_height_above_table_m']=min(result['min_tool_height_above_table_m'],float(kin.position(q)[2]-z))
result['status']='clear' if not result['faults'] else 'blocked'
args.report.write_text(json.dumps(result,indent=2)+'\n')
print(result['status'],'samples',result['samples'],'faults',len(result['faults']));print(result['faults'][:2])
raise SystemExit(bool(result['faults']))
