"""Conservative 3D boxes for vehicles, modified lift, map and fixed structures."""
import hashlib
import pickle,json,math,itertools
import numpy as np,fcl
from scipy.spatial.transform import Rotation
from arm_sim.common import ROOT
from arm_sim.conveyor_hardware import forklift_boxes,shipping_boxes,warehouse_boxes
from arm_sim.factory import table_legs
from scripts.check_conveyor_floorplan import vehicle_vertices
plan_bytes=(ROOT/'outputs/sim_real_demo/plan.pkl').read_bytes();plan_sha=hashlib.sha256(plan_bytes).hexdigest();p=pickle.loads(plan_bytes);c=p.cfg;fixed=[]
def obj(pos,size,r=np.eye(3)):return fcl.CollisionObject(fcl.Box(*size),fcl.Transform(r,np.array(pos)))
def box(n,pos,size):fixed.append((n,obj(pos,size)))
for t in c['tables']:
 box('table/'+t['id'],[*t['center'],c['table_height']-.015],[*t['size'],.03])
 for i,(pos,size) in enumerate(table_legs(t,c['table_height'])):box('leg/'+t['id']+str(i),pos,size)
for i,r in enumerate(c['warehouses']):
 if c.get('trident_supports'):
  for leaf,pos,size in warehouse_boxes(r):box(f'rack{i}/'+leaf,pos,size)
  continue
 x,y=r['center'];w,d=r['size']
 for sx,sy in itertools.product((-1,1),repeat=2):box(f'rack{i}/post',[x+sx*(w/2-.012),y+sy*(d/2-.012),r['post_height']/2],[.018,.018,r['post_height']])
 for h in r['levels']:
  for a,b in ((-w/2,-.073),(-.037,.037),(.073,w/2)):box(f'rack{i}/shelf',[x+(a+b)/2,y,h-.008],[b-a,d,.016])
for n,pos,size in shipping_boxes(c):box('shipping/'+n,pos,size)
for panel in c['docking_panels']:box('marker/'+panel['id'],panel['center'],panel['size'])
local={};fixed_lift,moving=forklift_boxes(c)
for name in p.vehicles:
 kind='burger' if name.startswith('burger') else 'waffle';vs=np.array(vehicle_vertices(kind));lo=vs.min(0);hi=vs.max(0);local[name]=[('body',(lo+hi)/2,hi-lo)]
 if kind=='burger':local[name].append(('deck',[-.032,0,c['burger']['deck_top']-.004],[*c['burger']['deck_size'],.008]))
 else:local[name]+=fixed_lift
fails=[];last=None;count=0
for t in sorted(set(np.arange(0,p.duration,.1))|{v for tr in p.vehicles.values() for v in tr.times}|set(p.lift.times)):
 poses={n:tr.at(t) for n,tr in p.vehicles.items()};lift=p.lift.at(t)[0];key=(*np.concatenate(list(poses.values())),lift)
 if key==last:continue
 last=key;count+=1;shapes={}
 for n,pose in poses.items():
  rot=Rotation.from_euler('z',pose[2]).as_matrix();shift=np.r_[pose[:2],0.];ls=local[n]+([(leaf,np.array(pos)+[0,0,lift],s) for leaf,pos,s in moving] if n=='waffle_1' else [])
  near_support=any(np.linalg.norm(pose[:2]-d)<.20 for d in [p.rack_dock(i) for i in range(2)]+[c['shipping_vehicle_pose'][:2]])
  if n=='waffle_1' and not near_support and any(p.entry(h,t)[1]=='lift' for h in p.trays):
   # Include the carried container and all three products during travel/turns.
   # The separate support audit covers docking insertion and seating contact.
   loaded=any(p.entry(h,t)[1]=='lift' for h in p.parts.values());height=.040 if loaded else .012
   size=c['finished_container']['size'];ls=ls+[('carried_container',[c['waffle']['tray_offset'],0,c['waffle']['fork_top']+lift+height/2],[size[1],size[0],height])]
  shapes[n]=[]
  for leaf,pos,size in ls:
   world=rot@pos+shift;shape=obj(world,size,rot);shapes[n].append((leaf,shape))
   corners=np.array(pos)+np.array(size)/2*np.array(list(itertools.product((-1,1),repeat=3)));world_corners=corners@rot.T+shift
   if (abs(world_corners[:,:2])>np.array(c['map']['size'])/2).any():why=n+'/map'
   else:why=None
   for other,ob in fixed:
    if fcl.collide(shape,ob,fcl.CollisionRequest(),fcl.CollisionResult()):why=n+'/'+leaf+'/'+other;break
   if why and not any(f['pair']==why for f in fails):fails.append({'time':float(t),'pair':why})
 for (a,aa),(b,bb) in itertools.combinations(shapes.items(),2):
  if any(fcl.collide(x,y,fcl.CollisionRequest(),fcl.CollisionResult()) for _,x in aa for _,y in bb):
   why=a+'/'+b
   if not any(f['pair']==why for f in fails):fails.append({'time':float(t),'pair':why})
r={'plan_sha256':plan_sha,'status':'pass' if not fails else 'blocked','samples':count,'failures':fails,'scope':'sampled conservative vehicle boxes, lift, carried container/products during travel/turns, table/legs, rack shelves/posts, shipping stand, map and vehicle pairs'}
(ROOT/'outputs/sim_real_demo/traffic_check.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))

raise SystemExit(r['status']!='pass')
