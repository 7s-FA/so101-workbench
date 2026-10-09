import hashlib
import pickle,json
import numpy as np
import fcl
from arm_sim.common import ROOT
from arm_sim.grasp_geometry import *
from arm_sim.real_demo import KINDS
plan_bytes=(ROOT/'outputs/sim_real_demo/plan.pkl').read_bytes();plan_sha=hashlib.sha256(plan_bytes).hexdigest();p=pickle.loads(plan_bytes);cfg=p.cfg;rev=json.loads((ROOT/'config/conveyor_demo.json').read_text());folder=ROOT/'assets/generated/conveyor_demo/parts'
models={k:triangle_model(stl_triangles(ROOT/'assets/product/02_ball_d30.stl' if k=='ball' else folder/rev['parts'][k])*.001) for k in KINDS}
fixtures=[]
for col,k in enumerate(KINDS):
 for row in range(3):
  x,y=cfg['source_banks'][0];fixtures.append((k,[x+(col-1)*cfg['source_pitch'],y+(row-1)*cfg['source_pitch'],cfg['table_height']]))
fixtures += [(s['kind'],[*s['position'],cfg['table_height']]) for s in cfg['preparation_slots']]
fixtures += [('housing',[*xy,cfg['table_height']]) for xy in (cfg['jig'],cfg['handoff'])]
jigs={k:triangle_model(stl_triangles(folder/rev['parts'][k+'_jig'])*.001) for k in KINDS};static=[(k,collision_object(jigs[k],np.eye(3),pos)) for k,pos in fixtures]
arms={n:GraspMeshes(k) for n,k in p.ik.items()};fails=[]
for e in p.grasp_events:
 t=e['time'];n=e['arm'];q=p.arms[n].at(t);meshes=arms[n].jaws(q)
 for part in e['parts']:
  obj=collision_object(models[part.split('_')[0]],p.part_rotation(part,t),p.part_at(part,t))
  for link,idx,jaw in meshes:
   if fcl.collide(jaw,obj,fcl.CollisionRequest(),fcl.CollisionResult()):fails.append({'time':t,'arm':n,'part':part,'link':link,'reason':'jaw intersects held part'})
 for kind,obj in static:
  for link,idx,jaw in meshes:
   if fcl.collide(jaw,obj,fcl.CollisionRequest(),fcl.CollisionResult()):fails.append({'time':t,'arm':n,'part':kind,'link':link,'reason':'jaw intersects fixed jig'})
 print(n,round(t,2),'failures',len(fails),flush=True)
assembly_pairs=[]
from itertools import combinations
for kit,t in enumerate(p.assembly_times,1):
 for a,b in combinations(KINDS,2):
  na,nb=f'{a}_{kit}',f'{b}_{kit}'
  oa=collision_object(models[a],p.part_rotation(na,t),p.part_at(na,t));ob=collision_object(models[b],p.part_rotation(nb,t),p.part_at(nb,t))
  gap,_=separation(oa,ob);assembly_pairs.append({'kit':kit,'pair':a+'/'+b,'gap_m':gap})
  if gap<=0:fails.append({'time':t,'kit':kit,'pair':a+'/'+b,'reason':'assembled part meshes penetrate'})
r={'plan_sha256':plan_sha,'status':'pass' if not fails else 'blocked','grasp_events':len(p.grasp_events),'assembled_pairs':assembly_pairs,'failures':fails,'scope':'closed-jaw grasp endpoints versus held parts and fixed jigs; assembled mating meshes checked; moving trajectory not covered'}
(ROOT/'outputs/sim_real_demo/payload_check.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))

raise SystemExit(r['status']!='pass')
