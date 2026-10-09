import json,sys,math,xml.etree.ElementTree as E
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
root=Path(__file__).resolve().parents[1];sys.path[:0]=[str(root),str(root.parent/'Sim_real')]
from arm_sim.demo_kinematics import ArmKinematics
cfg=json.loads((root/'config/scene.json').read_text());model=json.loads((root.parent/'Sim_real/assets/generated/conveyor_demo/models.json').read_text())['omx']
kin=ArmKinematics(model,dict(position=[0,0],yaw=0,base_plate_thickness=0,demo_pitch_degrees=90),cfg['table']['height'])
r=E.parse(cfg['model_source']).getroot();js=r.findall('joint')
def tf(q):
 q=list(q)+[-q[5]];vals=dict(zip(model['joints'],q));out={'link0':np.eye(4)}
 for j in js:
  e=j.find('origin');t=np.eye(4);t[:3,3]=np.fromstring(e.get('xyz','0 0 0'),sep=' ');t[:3,:3]=R.from_euler('xyz',np.fromstring(e.get('rpy','0 0 0'),sep=' ')).as_matrix()
  if j.get('type')!='fixed':
   rot=np.eye(4);rot[:3,:3]=R.from_rotvec(np.fromstring(j.find('axis').get('xyz'),sep=' ')*vals[j.get('name')]).as_matrix();t=t@rot
  out[j.find('child').get('link')]=out[j.find('parent').get('link')]@t
 return out
inertial=[(l.get('name'),float(l.find('inertial/mass').get('value')),np.r_[np.fromstring(l.find('inertial/origin').get('xyz'),sep=' '),1]) for l in r.findall('link') if l.find('inertial') is not None]
def potential(q):
 transforms=tf(q);return sum(m*9.81*(transforms[n]@p)[2] for n,m,p in inertial)
def metrics(deg):
 q=np.radians(deg);eps=1e-5;g=[]
 for i in range(5):
  dq=np.zeros(6);dq[i]=eps;g.append((potential(q+dq)-potential(q-dq))/(2*eps))
 return dict(angles_deg=deg,gravity_nm=g,tool_xyz=kin.position(q).tolist(),score=sum(abs(v) for v in g))
if __name__=='__main__':
 before=json.loads((root/'reports/upright_start_before.json').read_text())
 result=dict(scope='Nominal URDF mass and COM, numerical derivative of gravitational potential. Does not include unmodeled camera/cables/payload or certify physical loads.',
             previous=metrics(before['start_angles_deg']),upright=metrics(cfg['start_angles_deg']))
 (root/'reports/upright_start_load.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
