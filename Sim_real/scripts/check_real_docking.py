"""Audit measured planned poses, independent of movement mode labels."""
import hashlib,json,pickle,math,itertools
import numpy as np
from scipy.spatial import ConvexHull
from arm_sim.common import ROOT
from arm_sim.conveyor_hardware import forklift_boxes
from scripts.check_conveyor_floorplan import vehicle_vertices
out=ROOT/'outputs/sim_real_demo';blob=(out/'plan.pkl').read_bytes();p=pickle.loads(blob);c=p.cfg;failures=[];entries=[];exits=[];turn_gaps=[]
lane=c['vehicle_lane_y']
docks={'burger_1':[c['loading_dock'],c['unload_dock']],'burger_2':[c['loading_dock'],c['unload_dock']], 'waffle_1':[p.rack_dock(i) for i in range(2)]+[c['shipping_vehicle_pose'][:2]]}
fixed,moving=forklift_boxes(c);vertices=list(vehicle_vertices('waffle'))
for _,pos,size in fixed+moving:
 vertices.extend(np.array(pos)+np.array(size)/2*np.array(list(itertools.product((-1,1),repeat=3))))
spec=c['finished_container'];vertices.extend([[c['waffle']['tray_offset']+sx*spec['size'][1]/2,sy*spec['size'][0]/2,0] for sx,sy in itertools.product((-1,1),repeat=2)])
radius=float(np.linalg.norm(np.array(vertices)[:,:2],axis=1).max())
vertices=np.array(vertices)[:,:2];vertices=vertices[ConvexHull(vertices).vertices]
for event in p.vehicle_moves:
 n=event['vehicle'];tr=p.vehicles[n];times=np.linspace(event['start'],event['end'],51);q=np.array([tr.at(t) for t in times]);a,b=q[0],q[-1];delta=b[:2]-a[:2]
 translating=np.linalg.norm(delta)>1e-7;rotating=abs(b[2]-a[2])>1e-7
 if rotating:
  if np.max(np.linalg.norm(q[:,:2]-a[:2],axis=1))>1e-7:failures.append(n+'/rotation while translating')
  if abs(b[2]-a[2])>math.pi/2+1e-7:failures.append(n+'/unnecessary half-turn')
  if abs(a[1]-lane)>1e-7:failures.append(n+'/turn before reaching aisle')
 else:
  if np.ptp(q[:,2])>1e-7:failures.append(n+'/yaw changes during straight travel')
  if translating:
   side=q[:,:2]-a[:2];side=side[:,0]*-math.sin(a[2])+side[:,1]*math.cos(a[2])
   if abs(side).max()>1e-7:failures.append(n+'/sideways travel')
   if n=='waffle_1' and abs(delta[0])>1e-7 and abs(math.cos(a[2])+1)>1e-7:failures.append(n+'/forklift must always face left across aisle')
 # Every primitive must come to rest, including immediately before/after turns.
 for t in (event['start'],event['end']):
  velocity=(tr.at(t+1e-4)-tr.at(t-1e-4))/(2e-4)
  if abs(velocity).max()>1e-4:failures.append(n+'/not stopped at movement boundary')
 at_start=any(np.linalg.norm(a[:2]-d)<1e-5 for d in docks[n]);at_end=any(np.linalg.norm(b[:2]-d)<1e-5 for d in docks[n])
 if at_start:
  exits.append(event)
  if rotating or np.dot(delta,[math.cos(a[2]),math.sin(a[2])])>=0 or abs(b[1]-lane)>1e-7:failures.append(n+'/must reverse straight from dock all the way to aisle')
 if at_end:
  entries.append(event)
  initial_approach=n=='waffle_1' and event is next(e for e in p.vehicle_moves if e['vehicle']==n) and np.allclose(a,c['waffle']['pose'],atol=1e-8)
  if rotating or np.dot(delta,[math.cos(b[2]),math.sin(b[2])])<=0 or abs(math.sin(b[2])-1)>1e-6 or (abs(a[1]-lane)>1e-7 and not initial_approach):failures.append(n+'/must drive forward from aisle into dock (or initial aligned approach)')
 if n=='waffle_1' and rotating:
  # Check the requested quarter-turn, not an unrequested full circle.
  angles=np.linspace(a[2],b[2],721)
  sweeps=np.array([vertices@np.array([[math.cos(angle),math.sin(angle)],[-math.sin(angle),math.cos(angle)]])+a[:2] for angle in angles])
  sampling_bound=2*radius*math.sin(abs(angles[1]-angles[0])/4)
  gap=float(np.min(np.array(c['map']['size'])/2-abs(sweeps)))-sampling_bound
  turn_gaps.append(float(gap))
  if gap<.05:failures.append('forklift requested turn wall clearance below 50mm')
for n in docks:
 if not any(e['vehicle']==n for e in entries) or not any(e['vehicle']==n for e in exits):failures.append(n+'/missing docking or departure evidence')
before=pickle.loads((out/'motion_revision/before_docking.pkl').read_bytes())
for name in p.parts:
 if name.endswith(('_1','_2','_3')) and not np.allclose(p.part_at(name,0),before.part_at(name,0),atol=1e-8):failures.append(name+'/initial pickup moved')
r={'status':'pass' if not failures else 'blocked','plan_sha256':hashlib.sha256(blob).hexdigest(),'dock_entries':len(entries),'reverse_departures':len(exits),'forklift_full_turn_radius_m':radius,'requested_turn_wall_clearances_m':turn_gaps,'forklift_cross_aisle_heading':'left (west), forward to warehouses, reverse to worktable','failures':sorted(set(failures)),'scope':'straight reverse dock-to-aisle departure, straight forward aisle-to-dock entry, stopped in-place quarter-turns only, requested swept turn wall clearance including fork and container; real friction/slip unverified'}
(out/'docking_check.json').write_text(json.dumps(r,indent=2));print(json.dumps(r));raise SystemExit(bool(failures))
