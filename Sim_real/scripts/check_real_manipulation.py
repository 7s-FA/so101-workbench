"""Sample moving arms and carried parts against actual product/jig meshes."""
import hashlib
import pickle,json,itertools
import numpy as np,fcl
from scipy.spatial.transform import Rotation
from arm_sim.common import ROOT
from arm_sim.grasp_geometry import GraspMeshes,stl_triangles,triangle_model,collision_object
plan_bytes=(ROOT/'outputs/sim_real_demo/plan.pkl').read_bytes();plan_sha=hashlib.sha256(plan_bytes).hexdigest();p=pickle.loads(plan_bytes);c=p.cfg;rev=json.loads((ROOT/'config/conveyor_demo.json').read_text());folder=ROOT/'assets/generated/conveyor_demo/parts';kinds=('housing','ball','cap')
vertices={k:stl_triangles(ROOT/'assets/product/02_ball_d30.stl' if k=='ball' else folder/rev['parts'][k])*.001 for k in kinds}
jv={k:stl_triangles(folder/rev['parts'][k+'_jig'])*.001 for k in kinds}
from arm_sim.conveyor_hardware import d435_assembly,d435_vertices
mount,camera=d435_assembly();extras=[(d435_vertices(),camera),(stl_triangles(ROOT/'vendor/so_arm100/Optional/Wrist_Cam_Mount_RealSense_D435/stl/Wrist_cam_mount_RealSense_D435.stl')*.001,mount)]
extras=[(triangle_model(v),v,tf) for v,tf in extras]
models={k:triangle_model(v) for k,v in vertices.items()};jm={k:triangle_model(v) for k,v in jv.items()};arms={n:GraspMeshes(k) for n,k in p.ik.items()};cache={};fails=[];samples=0

def shape(model,v,r,pos):
 world=v@r.T+pos;return (collision_object(model,r,pos),world.min(0),world.max(0))
def hit(a,b):
 if np.any(a[2]<b[1]-1e-8) or np.any(b[2]<a[1]-1e-8):return False
 return bool(fcl.collide(a[0],b[0],fcl.CollisionRequest(),fcl.CollisionResult()))
def fail(t,n,why):
 if not any(f['arm']==n and f['pair']==why for f in fails):fails.append({'time':float(t),'arm':n,'pair':why});print(fails[-1],flush=True)
static=[]
for col,k in enumerate(kinds):
 for row in range(3):
  x,y=c['source_banks'][0];pos=[x+(col-1)*c['source_pitch'],y+(row-1)*c['source_pitch'],c['table_height']];static.append(('source/'+k+str(row),shape(jm[k],jv[k],np.eye(3),pos)))
for s in c['preparation_slots']:static.append(('prep/'+s['kind'],shape(jm[s['kind']],jv[s['kind']],np.eye(3),[*s['position'],c['table_height']])))
for name,xy in [('assembly',c['jig']),('handoff',c['handoff'])]:static.append((name,shape(jm['housing'],jv['housing'],np.eye(3),[*xy,c['table_height']])))
for phase in p.phases:
 n=phase['arm'];g=arms[n]
 for t in np.linspace(phase['start'],phase['end'],max(2,int((phase['end']-phase['start'])/.1)+1)):
  samples+=1;q=p.arms[n].at(t);ar=g.geometry.at(q);obs={(l,i):o for l,i,o in g.at(q)};arm=[(l,(obs[l,i],lo,hi)) for l,i,_,_,lo,hi in ar]
  if p.ik[n].so:
   wrist=g.geometry.transforms(q)['gripper_link']
   for idx,(model,vertices_extra,local) in enumerate(extras):
    tf=wrist@local;arm.append(('wrist_camera_'+str(idx),shape(model,vertices_extra,tf[:3,:3],tf[:3,3])))
  parts={name:shape(models[name.split('_')[0]],vertices[name.split('_')[0]],p.part_rotation(name,t),p.part_at(name,t)) for name in p.parts}
  fixtures=static.copy()
  for vehicle in ('burger_1','burger_2'):
   pose=p.vehicles[vehicle].at(t);rot=Rotation.from_euler('z',pose[2]).as_matrix()
   for col,k in enumerate(kinds):
    for row in range(3):
     from arm_sim.conveyor_scene import cargo_jig_pose
     off,yaw=cargo_jig_pose(c['burger'],col,row);pos=rot@off+[*pose[:2],0.]
     fixtures.append((vehicle+'/'+k+str(row),shape(jm[k],jv[k],rot@Rotation.from_euler('z',yaw).as_matrix(),pos)))
  spec=c['finished_container']
  for tray in range(2):
   pose=p.tray_pose(tray,t);r=Rotation.from_euler('z',pose[3]).as_matrix()
   size=np.array(spec['size']);corners=np.array(list(itertools.product(*[(-s/2,s/2) for s in size])))
   fixtures.append((f'finished/tray{tray}/carrier',shape(fcl.Box(*size),corners,r,pose[:3]+r@[0,0,size[2]/2])))
   for slot in range(3):
    pos=pose[:3]+r@[(slot-1)*spec['pitch'],0,spec['size'][2]]
    fixtures.append((f'finished/tray{tray}/slot{slot}',shape(jm['housing'],jv['housing'],r,pos)))
  for link,obj in arm:
   for part,pobj in parts.items():
    if hit(obj,pobj):fail(t,n,link+'/part/'+part.split('_')[0])
   for f,fobj in fixtures:
    if hit(obj,fobj):fail(t,n,link+'/jig/'+f)
  for name,obj in parts.items():
   if p.entry(p.parts[name],t)[1]!=n:continue
   for f,fobj in fixtures:
    if hit(obj,fobj):fail(t,n,'payload/'+name.split('_')[0]+'/jig/'+f)
   for other,ob in parts.items():
    if other==name or other.rsplit('_',1)[1]==name.rsplit('_',1)[1]:continue
    if hit(obj,ob):fail(t,n,'payloads/'+name+'/'+other)
 print('phase',phase['action'],n,'samples',samples,'failures',len(fails),flush=True)
 if len(fails)>40:break
r={'plan_sha256':plan_sha,'status':'pass' if not fails else 'blocked','samples':samples,'failures':fails,'scope':'arm/camera triangles, carrier/parts/jigs through pick and place phases at <=100ms spacing; same-product mating contacts exempt; park and transit arm/table checks in path report'}
(ROOT/'outputs/sim_real_demo/manipulation_check.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))

raise SystemExit(r['status']!='pass')
