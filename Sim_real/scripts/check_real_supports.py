"""Dimensional and swept insertion audit; not structural strength certification."""
import hashlib,json,pickle,itertools
import numpy as np,fcl
from scipy.spatial.transform import Rotation
from arm_sim.common import ROOT
from arm_sim.real_demo import ASSEMBLY
from arm_sim.conveyor_hardware import forklift_boxes,warehouse_boxes,shipping_boxes
from arm_sim.grasp_geometry import stl_triangles
blob=(ROOT/'outputs/sim_real_demo/plan.pkl').read_bytes();p=pickle.loads(blob);c=p.cfg;failures=[];metrics={};rev=json.loads((ROOT/'config/conveyor_demo.json').read_text())
top=max(ASSEMBLY[k]+stl_triangles(ROOT/'assets/product/02_ball_d30.stl' if k=='ball' else ROOT/'assets/generated/conveyor_demo/parts'/rev['parts'][k])[:,2].max()*.001 for k in ASSEMBLY)
spec=c['finished_container'];height=spec['size'][2]+p.seating['contact_height']['housing']+p.seating['render_gap']+top
metrics['empty_container_height_mm']=(spec['size'][2]+stl_triangles(ROOT/'assets/generated/conveyor_demo/parts'/rev['parts'][spec['jig']])[:,2].max()*.001)*1000
metrics.update(loaded_container_height_mm=height*1000,lower_top_mm=c['warehouses'][0]['levels'][0]*1000,upper_top_mm=c['warehouses'][0]['levels'][1]*1000,layer_spacing_mm=np.diff(c['warehouses'][0]['levels']).item()*1000,max_commanded_stroke_mm=float(np.max(p.lift.values))*1000,fork_length_mm=c['waffle']['fork_length']*1000,fork_floor_gap_mm=(c['waffle']['fork_top']-c['waffle']['fork_thickness'])*1000)
metrics['lower_loaded_extraction_roof_gap_mm']=(np.diff(c['warehouses'][0]['levels']).item()-c['warehouses'][0]['shelf_thickness']-height-.005)*1000
if metrics['lower_loaded_extraction_roof_gap_mm']<5:failures.append('less than 5mm loaded extraction roof clearance')
if metrics['max_commanded_stroke_mm']>100:failures.append('actual lift stroke exceeds 100mm')
if metrics['layer_spacing_mm']>=100:failures.append('shelf spacing must stay below 100mm')
if metrics['fork_floor_gap_mm']<.9:failures.append('fork intersects/too close to floor')
metrics['lowest_shelf_insertion_gap_mm']=c['fork_insertion_clearance']*1000
metrics['lower_shelf_underfloor_gap_mm']=(c['warehouses'][0]['levels'][0]-c['warehouses'][0]['shelf_thickness'])*1000
metrics['shelf_thicknesses_mm']=[r['shelf_thickness']*1000 for r in c['warehouses']]
if c['warehouses'][0]['levels'][0]-c['fork_insertion_clearance']<c['waffle']['fork_top']:failures.append('lowest insertion target below physical fork minimum')
metrics['slot_side_gap_mm']=(36-22)/2
# Continuous translation is bounded by the union of each axis-aligned sweep.
# Fork heading is +Y at all docking stations; sample all vertical positions.
def box(pos,size,r=np.eye(3)):return fcl.CollisionObject(fcl.Box(*size),fcl.Transform(r,np.array(pos)))
rot=Rotation.from_euler('z',np.pi/2).as_matrix();samples=0
stations=[('shipping',c['shipping_dock'],[c['shipping_height']],shipping_boxes(c))]+[(f'rack{i}',r['center'],r['levels'],warehouse_boxes(r)) for i,r in enumerate(c['warehouses'])]
for name,center,levels,boards in stations:
 retreat=center[1]-c['waffle']['tray_offset']-c['vehicle_lane_y']
 obs=[(n,box(pos,size)) for n,pos,size in boards]
 for level in levels:
  tray_sweep=box([center[0],center[1]-retreat/2,level+.005+height/2],[spec['size'][0],spec['size'][1]+retreat,height])
  for leaf,ob in obs:
   if fcl.collide(tray_sweep,ob,fcl.CollisionRequest(),fcl.CollisionResult()):failures.append(name+'/loaded_container/'+leaf+' insertion sweep')
  for top_z in (max(c['waffle']['fork_top'],level-c['fork_insertion_clearance']),level+.005):
   if top_z-c['waffle']['fork_top']>c['waffle']['lift_stroke']:failures.append(name+' exceeds stroke')
   _,moving=forklift_boxes(c,top_z)
   for leaf,pos,size in moving:
    # Swept box for the complete insertion, including the heel/carriage.
    world=rot@pos+[*center,0.]-[0,c['waffle']['tray_offset']+retreat/2,0]
    dims=rot@np.diag(size);extent=np.abs(dims).sum(axis=1)+[0,retreat,0]
    swept=box(world,extent);samples+=1
    for n,ob in obs:
     if fcl.collide(swept,ob,fcl.CollisionRequest(),fcl.CollisionResult()):failures.append(name+'/'+leaf+'/'+n+' insertion sweep')
metrics['insertion_sweeps']=samples
r={'status':'pass' if not failures else 'blocked','plan_sha256':hashlib.sha256(blob).hexdigest(),'metrics':metrics,'failures':sorted(set(failures)),'limitations':['5mm insertion clearance is nominal; actual fork minimum height and docking accuracy must be measured','3mm fork plate thickness is provisional; strength and actuator rated load not verified','physical friction, force closure and assembly insertion are not verified']}
(ROOT/'outputs/sim_real_demo/support_check.json').write_text(json.dumps(r,indent=2));print(json.dumps(r));raise SystemExit(r['status']!='pass')
