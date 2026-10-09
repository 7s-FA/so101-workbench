#!/usr/bin/env python3
"""Audit the actual return-stage generator without Isaac startup or motor I/O."""
import ast,copy,json,math,sys,textwrap
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT.parent/'Sim_real')]
from arm_sim.demo_kinematics import ArmKinematics
from arm_sim.conveyor_hardware import clamp_boxes
from teach_io.mapping import Mapping
from teach_io.entry_collision import EntryCollisionAudit
from teach_io.motion_timing import VELOCITY_RAD_S,ACCELERATION_RAD_S2
source=(ROOT/'scripts/sim.py').read_text()
plan='stages=[];target=home.copy()'+source.split('        stages=[];target=home.copy()',1)[1].split('        # Validate the whole commanded path',1)[0]
plan='stages=[];target=home.copy()\n'+textwrap.dedent(plan.split('\n',1)[1])
model=json.loads((ROOT.parent/'Sim_real/assets/generated/conveyor_demo/models.json').read_text())['omx']
mapping=Mapping(json.loads((ROOT/'config/follower_visual_checked.json').read_text()),allow_visual_confirmation=True)
results=[]
for real_plan in (False,True):
 for part in ('all','housing','ball','cap'):
  cfg=json.loads((ROOT/'config/scene.json').read_text())
  if real_plan:cfg['rest_point']=[.205,0,.325]
  kin=ArmKinematics(model,dict(position=[0,0],yaw=0,base_plate_thickness=0,demo_pitch_degrees=90),cfg['table']['height'])
  def solve(point,previous):
   q=kin.solve(point,previous);q[4]=q[0];return q
  home=np.radians(cfg['start_angles_deg']);home=np.r_[home,-home[5]]
  work_pose=np.r_[solve(cfg['rest_point'],model['home']),.75,-.75]
  initial={k:np.array([*p['source'],cfg['table']['height']+p['seating_z']+.0005])+[*p['offset_xy'],0] for k,p in cfg['parts'].items()}
  namespace=dict(np=np,math=math,home=home,work_pose=work_pose,cfg=cfg,initial=initial,initial_point=kin.position(home),args=SimpleNamespace(part=part),solve=solve)
  exec(compile(plan,'actual_sim_plan','exec'),namespace)
  q=home.copy();start=kin.position(home);startq=home.copy();segment_results=[]
  fixed={'table_height':cfg['table']['height'],'arms':{'omx':{'model':'omx','position':[0,0]}},'tables':[{'id':'table',**cfg['table']}],
   'arm_fixtures':{'omx':{'table':'table','side':'left','clamp_side':'left','clamp_offsets':[-.065,.065],'clamp_pad_depth':.018}}}
  audit=EntryCollisionAudit(cfg,model,clamp_boxes(fixed,'omx'))
  # Preserve every source/destination jig. Move only transferred part obstacles.
  for kind,p in cfg['parts'].items():
   if part in ('all',kind):
    pose=np.eye(4)
    pose[:3,3]=[p['destination'][0]+p['offset_xy'][0],p['destination'][1]+p['offset_xy'][1],cfg['table']['height']+p.get('destination_seating_z',p['seating_z'])+.0005]
    audit.world.set_transform(kind,pose)
  for st in namespace['stages']:
   returned=st['kind'] is None and st['label'].startswith(('Return','Settle'))
   trajectory=[];checks=0;failures=[]
   for t in np.linspace(0,1,301 if returned else 61):
    s=t*t*t*(10+t*(-15+6*t))
    q=namespace['stage_joints'](st,start,startq,s,q);mapping.to_ticks(q[:6]);trajectory.append(q[:6].copy())
    if returned:
     check=audit.check(q[:6].tolist(),q[:6].tolist(),mapping);checks+=check['samples']
     if not check['collision_checked']:failures.append(check)
   if returned:
    trajectory=np.asarray(trajectory);dt=st['duration']/st['real_time_scale']/300
    peak_v=np.max(np.abs(np.diff(trajectory,axis=0)/dt),axis=0)
    peak_a=np.max(np.abs(np.diff(trajectory,n=2,axis=0)/dt**2),axis=0)
    assert np.all(peak_v<=VELOCITY_RAD_S),peak_v
    assert np.all(peak_a<=ACCELERATION_RAD_S2),peak_a
    assert not failures,failures[:1]
    segment_results.append(dict(label=st['label'],samples=checks,peak_velocity=peak_v.tolist(),peak_acceleration=peak_a.tolist(),duration_real=st['duration']/st['real_time_scale'],collision='clear'))
   start=st['point'];startq=q.copy()
  assert np.allclose(q,home,atol=1e-12)
  results.append(dict(real_plan=real_plan,part=part,final_angles_deg=np.degrees(q[:6]).tolist(),segments=segment_results))
report=dict(status='passed',scope='Actual stage generator; 8 mode/part cases; nominal URDF self/table/clamp/jig/part meshes at final placement, 301 samples per return segment; not physical cable/contact validation.',cases=results)
(ROOT/'reports/return_to_start_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS',len(results),'cases; exact startup targets; return mesh/range/speed/acceleration checks')
