"""Check pace, parked task boundaries and differential-drive motion quality."""
import hashlib,json,pickle,math
import numpy as np
import xml.etree.ElementTree as ET
from arm_sim.common import ROOT
wheel_spacing={}
for kind in ('burger','waffle'):
 tree=ET.parse(ROOT/f'assets/factory/turtlebot3_description/urdf/turtlebot3_{kind}.urdf')
 sides=[float(j.find('origin').get('xyz').split()[1]) for j in tree.iter('joint') if j.get('name').endswith(('wheel_left_joint','wheel_right_joint'))]
 wheel_spacing[kind]=max(sides)-min(sides)
out=ROOT/'outputs/sim_real_demo';blob=(out/'plan.pkl').read_bytes();p=pickle.loads(blob);before=pickle.loads((out/'motion_revision/before.pkl').read_bytes());failures=[]
def summarize(plan):
 result={'duration_s':plan.duration,'arms':{},'vehicles':{}}
 for n,tr in plan.arms.items():
  times=np.arange(0,plan.duration,.04);qs=np.array([tr.at(t) for t in times]);velocity=np.diff(qs[:,:5],axis=0)/.04;acceleration=np.diff(velocity,axis=0)/.04;result['arms'][n]={'joint_travel_rad':float(np.abs(np.diff(qs[:,:5],axis=0)).sum()),'max_joint_speed_rad_s':float(abs(velocity).max()),'max_joint_acceleration_rad_s2':float(abs(acceleration).max())}
 for n,tr in plan.vehicles.items():
  times=np.arange(0,plan.duration,.02);q=np.array([tr.at(t) for t in times]);vel=np.diff(q,axis=0)/.02;yaw=(q[1:,2]+q[:-1,2])/2
  speed=vel[:,0]*np.cos(yaw)+vel[:,1]*np.sin(yaw);side=-vel[:,0]*np.sin(yaw)+vel[:,1]*np.cos(yaw);acc=np.diff(vel,axis=0)/.02
  width=wheel_spacing['burger' if n.startswith('burger') else 'waffle'];wheel_speed=np.c_[speed-vel[:,2]*width/2,speed+vel[:,2]*width/2];wheel_acceleration=np.diff(wheel_speed,axis=0)/.02
  result['vehicles'][n]={'reverse_distance_m':float(np.maximum(0,-speed).sum()*.02),'path_length_m':float(np.linalg.norm(vel[:,:2],axis=1).sum()*.02),'max_speed_m_s':float(np.linalg.norm(vel[:,:2],axis=1).max()),'max_acceleration_m_s2':float(np.linalg.norm(acc[:,:2],axis=1).max()),'max_angular_speed_rad_s':float(abs(vel[:,2]).max()),'max_angular_acceleration_rad_s2':float(abs(acc[:,2]).max()),'max_lateral_speed_m_s':float(abs(side).max()),'wheel_spacing_m':width,'max_wheel_ground_speed_m_s':float(abs(wheel_speed).max()),'max_wheel_ground_acceleration_m_s2':float(abs(wheel_acceleration).max())}
 return result
old=summarize(before);new=summarize(p)
if not 0<p.duration<=600:failures.append('process must finish within 600 nominal seconds')
for n,row in new['vehicles'].items():
 lim=p.cfg['vehicle_motion']
 for key,bound in [('max_speed_m_s',lim['linear_velocity']),('max_acceleration_m_s2',lim['linear_acceleration']),('max_angular_speed_rad_s',lim['angular_velocity']),('max_angular_acceleration_rad_s2',lim['angular_acceleration'])]:
  if row[key]>bound*1.01:failures.append(n+'/'+key)
 if row['max_lateral_speed_m_s']>.001:failures.append(n+'/kinematic lateral error exceeds 1mm/s')
 if row['max_wheel_ground_acceleration_m_s2']>.18:failures.append(n+'/wheel acceleration exceeds 0.18m/s2')
 # Reverse distance is intentional: every departure backs all the way to the aisle.
for n in p.arms:
 if not np.allclose(p.arms[n].at(p.duration)[:5],p.safe_pose(n)[:5],atol=1e-6):failures.append(n+'/not parked at completion')
for task in p.tasks:
 if task['arm']=='arm_c':
  for t in np.arange(task['start']+.01,task['end'],.1):
   if not np.allclose(p.arms['arm_b'].at(t)[:5],p.safe_pose('arm_b')[:5],atol=1e-5) or any(p.entry(h,t)[1]=='arm_b' for h in p.parts.values()):failures.append(task['id']+'/B must wait parked and empty during assembly');break
 if not np.allclose(p.arms[task['arm']].at(task['end'])[:5],p.safe_pose(task['arm'])[:5],atol=1e-6):failures.append(task['id']+'/not parked after kit')
for arm,row in new['arms'].items():
 for field in ('max_joint_speed_rad_s','max_joint_acceleration_rad_s2'):
  if row[field]>old['arms'][arm][field]*1.01:failures.append(arm+'/'+field+' was not reduced')
if new['arms']['arm_a']['joint_travel_rad']>=old['arms']['arm_a']['joint_travel_rad']:failures.append('A joint travel did not decrease')
if sum(e['arm']=='arm_a' for e in p.park_events)!=1:failures.append('A must explicitly park once after nine parts')
r={'status':'pass' if not failures else 'blocked','plan_sha256':hashlib.sha256(blob).hexdigest(),'before':old,'after':new,'failures':failures,'scope':'sampled motion derivatives, forward heading, travel length, nominal pace and task-safe endpoints; physical motor loads and jerk limits unverified'}
(out/'motion_quality_check.json').write_text(json.dumps(r,indent=2));print(json.dumps(r));raise SystemExit(bool(failures))
