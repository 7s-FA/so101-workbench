"""Collision-checked joint transit between raised, exact grasp poses."""
import itertools
import numpy as np
import fcl
from scipy.spatial.transform import Rotation
from arm_sim.common import ROOT
from arm_sim.grasp_geometry import triangle_model,collision_object,stl_triangles
from arm_sim.conveyor_hardware import clamp_boxes

def joint_route(plan,arm,goal):
    k=plan.ik[arm];g=plan.grasp_meshes[arm];start=plan.arms[arm].at(plan.time).copy();limits=np.asarray(k.model['limits'][:5]);fixed=[]
    def mesh(vertices,tf):
        world=vertices@tf[:3,:3].T+tf[:3,3]
        return collision_object(triangle_model(vertices),tf[:3,:3],tf[:3,3]),world.min(0),world.max(0)
    def box(pos,size):
        pos=np.asarray(pos);size=np.asarray(size);fixed.append((collision_object(fcl.Box(*size),np.eye(3),pos),pos-size/2,pos+size/2))
    def hit(a,b):
        return np.all(a[2]>=b[1]) and np.all(b[2]>=a[1]) and bool(fcl.collide(a[0],b[0],fcl.CollisionRequest(),fcl.CollisionResult()))
    for table in plan.cfg['tables']:box([*table['center'],plan.cfg['table_height']-.015],[*table['size'],.03])
    for _,pos,size in clamp_boxes(plan.cfg,arm):box(pos,size)
    import json
    rev=json.loads((ROOT/'config/conveyor_demo.json').read_text());folder=ROOT/'assets/generated/conveyor_demo/parts'
    jigs={kind:stl_triangles(folder/rev['parts'][kind+'_jig'])*.001 for kind in ('housing','ball','cap')}
    def jig(kind,pos,r=np.eye(3)):
        tf=np.eye(4);tf[:3,:3]=r;tf[:3,3]=pos;fixed.append(mesh(jigs[kind],tf))
    for col,kind in enumerate(jigs):
        for row in range(3):
            x,y=plan.cfg['source_banks'][0];jig(kind,[x+(col-1)*plan.cfg['source_pitch'],y+(row-1)*plan.cfg['source_pitch'],plan.cfg['table_height']])
    for vehicle in ('burger_1','burger_2'):
        pose=plan.vehicles[vehicle].at(plan.time);r=Rotation.from_euler('z',pose[2]).as_matrix()
        for col,kind in enumerate(jigs):
            for row in range(3):
                from arm_sim.conveyor_scene import cargo_jig_pose
                off,yaw=cargo_jig_pose(plan.cfg['burger'],col,row)
                jig(kind,r@off+[*pose[:2],0],r@Rotation.from_euler('z',yaw).as_matrix())
    held=[];tool=k.transform(start)
    for name,history in plan.parts.items():
        kind=name.split('_')[0];vertices=stl_triangles(ROOT/'assets/product/02_ball_d30.stl' if kind=='ball' else folder/rev['parts'][kind])*.001
        tf=np.eye(4);tf[:3,:3]=plan.part_rotation(name,plan.time);tf[:3,3]=plan.part_at(name,plan.time)
        if plan.entry(history,plan.time)[1]==arm:held.append((vertices,np.linalg.inv(tool)@tf))
        else:fixed.append(mesh(vertices,tf))
    los=np.array([o[1] for o in fixed]);his=np.array([o[2] for o in fixed]);cache={};reasons={}
    def valid(v):
        key=tuple(np.round(v,7))
        if key in cache:return cache[key]
        q=start.copy();q[:5]=v;shapes=g.geometry.at(q);obs={(l,i):o for l,i,o in g.at(q)}
        rows=[(l,(obs[l,i],lo,hi)) for l,i,_,_,lo,hi in shapes]
        for (a,aa),(b,bb) in itertools.combinations(rows,2):
            if frozenset((a,b)) in g.self_pairs and hit(aa,bb):cache[key]=False;reasons[key]='self '+a+'/'+b;return False
        wrist=g.geometry.transforms(q)['gripper_link']
        for model,local in plan.camera_shapes:
            tf=wrist@local;ob=collision_object(model,tf[:3,:3],tf[:3,3])
            if any(fcl.collide(ob,row[1][0],fcl.CollisionRequest(),fcl.CollisionResult()) for row in rows):cache[key]=False;reasons[key]='camera';return False
        checks=[o for _,o in rows]
        for vertices,local in held:checks.append(mesh(vertices,k.transform(q)@local))
        for obj,lo,hi in checks:
            indices=np.flatnonzero(np.all(hi>=los,axis=1)&np.all(his>=lo,axis=1))
            if any(hit((obj,lo,hi),fixed[i]) for i in indices):cache[key]=False;reasons[key]='fixed '+str([(int(i),fixed[i][1].tolist(),fixed[i][2].tolist()) for i in indices if hit((obj,lo,hi),fixed[i])])+' moving '+str((lo.tolist(),hi.tolist()));return False
        cache[key]=True;return True
    def edge(a,b):
        return all(valid(a+(b-a)*u) for u in np.linspace(0,1,max(2,int(np.max(abs(b-a))/.035)+2)))
    a=start[:5];b=goal[:5]
    if not valid(a) or not valid(b):raise ValueError('Raised transit endpoint intersects geometry '+str(reasons))
    if edge(a,b):return [goal]
    # Try simple lifted elbow/shoulder postures before any broader search.
    midpoint=(a+b)/2
    candidates=[np.r_[midpoint[0],s,e,w,midpoint[4]] for s,e,w in [(-.8,.4,1.3),(-1.,.6,1.2),(-1.2,.9,1.),(-.5,.3,.8)]]
    for v in candidates:
        if np.all(v>=limits[:,0]) and np.all(v<=limits[:,1]) and edge(a,v) and edge(v,b):
            q=start.copy();q[:5]=v;print('Raised joint detour',arm,'states',len(cache),flush=True);return [q,goal]
    rng=np.random.default_rng(17);nodes=[a];parents=[-1]
    for iteration in range(2500):
        target=b if iteration%4==0 else np.clip(midpoint+rng.normal(0,[.8,.8,.8,1.,.7]),limits[:,0],limits[:,1])
        distances=np.linalg.norm(np.asarray(nodes)-target,axis=1);idx=int(np.argmin(distances));v=nodes[idx]+(target-nodes[idx])*min(1,.28/max(distances[idx],1e-9))
        if not edge(nodes[idx],v):continue
        nodes.append(v);parents.append(idx)
        if np.linalg.norm(v-b)<.7 and edge(v,b):
            route=[b];idx=len(nodes)-1
            while idx!=-1:route.append(nodes[idx]);idx=parents[idx]
            route.reverse();i=0
            while i<len(route)-2:
                j=len(route)-1
                while j>i+1 and not edge(route[i],route[j]):j-=1
                route[i+1:j]=[];i+=1
            result=[]
            for v in route[1:]:q=start.copy();q[:5]=v;result.append(q)
            print('Checked joint detour',arm,'segments',len(result),'states',len(cache),flush=True);return result
    raise ValueError('No collision-free raised joint transfer found')
