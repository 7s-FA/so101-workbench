#!/usr/bin/env python3
"""Sample the revised trajectories against actual arm triangles and fixtures."""
import hashlib
import json,pickle,math
import faulthandler
faulthandler.enable()
from itertools import combinations
import numpy as np
import fcl
from scipy.spatial.transform import Rotation
from arm_sim.common import ROOT
from arm_sim.grasp_geometry import GraspMeshes,triangle_model,stl_triangles,collision_object,separation
from arm_sim.conveyor_hardware import clamp_boxes,d435_assembly,d435_vertices,shipping_boxes,warehouse_boxes
from arm_sim.overhead_camera import assembly

def distance(a,b):
    if fcl.collide(a,b,fcl.CollisionRequest(num_max_contacts=1),fcl.CollisionResult()):return 0.
    return separation(a,b)[0]



def check(plan,step=.12,start=0.,end=None):
    end=plan.duration if end is None else end
    cfg=plan.cfg;arms={n:GraspMeshes(k) for n,k in plan.ik.items()};cache={};failures=[];samples=0;minimum=10.;pairmin={};fixed=[];last_checked={};pair_checked={}
    def box(name,p,s):fixed.append((name,collision_object(fcl.Box(*s),np.eye(3),p),np.array(p)-np.array(s)/2,np.array(p)+np.array(s)/2))
    for t in cfg['tables']:box('table/'+t['id'],[*t['center'],cfg['table_height']-.015],[*t['size'],.03])
    for n in cfg['arms']:
        for name,p,s in clamp_boxes(cfg,n):box('clamp/'+n+'/'+name,p,s)
    for name,p,s in shipping_boxes(cfg):box('shipping/'+name,p,s)
    for i,rack in enumerate(cfg['warehouses']):
        for leaf,pos,size in warehouse_boxes(rack):box('warehouse'+str(i)+'/'+leaf,pos,size)
    mount,cam=d435_assembly();mountv=stl_triangles(ROOT/'vendor/so_arm100/Optional/Wrist_Cam_Mount_RealSense_D435/stl/Wrist_cam_mount_RealSense_D435.stl')*.001
    additions=[('camera_mount',mountv@mount[:3,:3].T+mount[:3,3]),('camera',d435_vertices()@cam[:3,:3].T+cam[:3,3])]
    additions=[(name,v,triangle_model(v)) for name,v in additions]
    for name,file,scale,tf in assembly(cfg)[0]:
        v=stl_triangles(file)*scale;v=v@tf[:3,:3].T+tf[:3,3];fixed.append(('overhead/'+name,collision_object(triangle_model(v),np.eye(3),np.zeros(3)),v.min(0),v.max(0)))
    fixed_los=np.array([row[2] for row in fixed]);fixed_his=np.array([row[3] for row in fixed])
    def near(lo,hi,lo2,hi2,margin=0):return np.all(hi+margin>=lo2) and np.all(hi2+margin>=lo)
    def frames(n,q):
        key=tuple(np.round(q,8))
        if cache.get(n,(None,))[0]==key:return cache[n][1]
        g=arms[n];shapes=g.geometry.at(q);obs={(l,i):o for l,i,o in g.at(q)}
        rows=[(l,i,obs[l,i],lo,hi) for l,i,_,_,lo,hi in shapes]
        if plan.ik[n].so:
            tf=g.geometry.transforms(q)['gripper_link']
            for name,v,model in additions:
                world=v@tf[:3,:3].T+tf[:3,3];rows.append((name,0,collision_object(model,tf[:3,:3],tf[:3,3]),world.min(0),world.max(0)))
        cache[n]=(key,rows);return rows
    def fail(t,why):
        if not any(x['pair']==why for x in failures):
            failures.append({'time':float(t),'pair':why});print(failures[-1],flush=True)
    times=set(np.arange(0,plan.duration,step))|{0.,plan.duration}
    for tr in plan.arms.values():times.update(tr.times)
    for t in sorted(t for t in times if start<=t and (t<end or end==plan.duration and t<=end)):
        qs={n:tr.at(t) for n,tr in plan.arms.items()};rows={n:frames(n,q) for n,q in qs.items()};samples+=1
        for n,rs in rows.items():
            pose_key=tuple(qs[n])
            if last_checked.get(n)==pose_key:continue
            last_checked[n]=pose_key
            for a,b in combinations(rs,2):
                if frozenset((a[0],b[0])) not in arms[n].self_pairs and not ((a[0].startswith('camera') != b[0].startswith('camera'))):continue
                if near(a[3],a[4],b[3],b[4]) and distance(a[2],b[2])<1e-7:fail(t,f'self:{n}/{a[0]}/{b[0]}')
            for l,i,obj,lo,hi in rs:
                indices=np.flatnonzero(np.all(hi>=fixed_los,axis=1)&np.all(fixed_his>=lo,axis=1))
                for index in indices:
                    fn,other,fl,fh=fixed[index]
                    if l=='link0' and fn=='table/assembly':continue  # Original OMX base rests on the worktop.
                    if n=='arm_c' and fn=='overhead/ArmBase' and l in ('base_link','shoulder_link'):continue
                    if near(lo,hi,fl,fh) and distance(obj,other)<1e-7:fail(t,f'fixture:{n}/{l}/{fn}')
        for (a,aa),(b,bb) in combinations(rows.items(),2):
            key=(tuple(qs[a]),tuple(qs[b]))
            if pair_checked.get((a,b))==key:continue
            pair_checked[a,b]=key
            alo=np.array([r[3] for r in aa]);ahi=np.array([r[4] for r in aa]);blo=np.array([r[3] for r in bb]);bhi=np.array([r[4] for r in bb])
            near_pairs=np.all(ahi[:,None,:]+.02>=blo[None,:,:],axis=2)&np.all(bhi[None,:,:]+.02>=alo[:,None,:],axis=2)
            for i,j in np.argwhere(near_pairs):
                x,y=aa[i],bb[j];d=distance(x[2],y[2]);pair=a+'/'+b;pairmin[pair]=min(pairmin.get(pair,10),d)
                if d<.02:fail(t,'arms:'+pair+'/'+x[0]+'/'+y[0])
        if samples%100==0:print('checked',samples,'failures',len(failures),flush=True)
        if len(failures)>50:break
    return {'status':'pass' if not failures else 'blocked','samples':samples,'sample_interval_s':step,'failures':failures,'arm_pair_minimum_m':pairmin,'scope':'sampled arm nonadjacent self, tables, clamps, overhead camera, wrist cameras and arm pairs; payload and traffic checks separate'}

def check_chunk(args):
    blob,start,end=args
    return check(pickle.loads(blob),start=start,end=end)

if __name__=='__main__':
    from concurrent.futures import ProcessPoolExecutor
    plan_bytes=(ROOT/'outputs/sim_real_demo/plan.pkl').read_bytes();plan_sha=hashlib.sha256(plan_bytes).hexdigest();p=pickle.loads(plan_bytes)
    edges=np.linspace(0,p.duration,4)
    with ProcessPoolExecutor(max_workers=3) as pool:
        reports=list(pool.map(check_chunk,[(plan_bytes,float(a),float(b)) for a,b in zip(edges[:-1],edges[1:])]))
    failures={};pairmin={}
    for row in reports:
        for failure in row['failures']:
            pair=failure['pair']
            if pair not in failures or failure['time']<failures[pair]['time']:failures[pair]=failure
        for pair,distance_m in row['arm_pair_minimum_m'].items():pairmin[pair]=min(pairmin.get(pair,10),distance_m)
    r={**reports[0],'status':'blocked' if failures else 'pass','samples':sum(row['samples'] for row in reports),'failures':sorted(failures.values(),key=lambda x:x['time']),'arm_pair_minimum_m':pairmin,'plan_sha256':plan_sha}
    (ROOT/'outputs/sim_real_demo/path_check.json').write_text(json.dumps(r,indent=2));print(json.dumps(r));raise SystemExit(r['status']!='pass')
