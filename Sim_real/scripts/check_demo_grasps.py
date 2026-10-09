#!/usr/bin/env python3
"""Offline triangle and trajectory checks; does not start hardware or servers."""
import argparse
import importlib
import json
import hashlib
import pickle
import time
from pathlib import Path
import numpy as np
from arm_sim.factory import load_layout
from arm_sim.demo_geometry import GeometryGuard
from arm_sim.grasp_geometry import GraspMeshes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',action='store_true')
    parser.add_argument('--interval',type=float,default=.04)
    parser.add_argument('--start',type=float,default=0.)
    parser.add_argument('--end',type=float)
    parser.add_argument('--load-plan',type=Path,help='Trusted locally generated pickle only; skip rebuilding this audit plan')
    parser.add_argument('--save-plan',type=Path,help='Save this locally generated plan for independent audits')
    parser.add_argument('--output',type=Path,default=Path('outputs/demo/grasp_check.json'))
    args=parser.parse_args()
    if not .001<=args.interval<=.1:parser.error('Interval must be 0.001 to 0.1 seconds')
    if args.start<0 or (args.end is not None and args.end<=args.start):parser.error('Invalid coverage interval')
    module=importlib.import_module('experiments.grasp_revision.factory_demo_candidate' if args.candidate else 'arm_sim.factory_demo')
    result={'status':'running','candidate':args.candidate,'scope':'URDF triangle geometry and sampled motion; not force/friction or real hardware validation','interval_s':args.interval}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();guard=None;t=0.
    def save():
        result['elapsed_s']=time.monotonic()-started
        args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    try:
        if args.load_plan:
            with args.load_plan.open('rb') as source:plan=pickle.load(source)
        else:plan=module.DemoPlan(module.demo_layout(load_layout()))
        if args.save_plan:
            args.save_plan.parent.mkdir(parents=True,exist_ok=True)
            with args.save_plan.open('wb') as target:pickle.dump(plan,target)
        result['plan']=plan.validate();result['grasp_observations']=[];save()
        fingerprint=hashlib.sha256(json.dumps(plan.cfg,sort_keys=True).encode())
        for n,tr in plan.arms.items():
            fingerprint.update(n.encode());fingerprint.update(np.asarray(tr.times).tobytes());fingerprint.update(np.asarray(tr.values).tobytes())
            fingerprint.update(json.dumps({'smooth':getattr(tr,'smooth',False),'tangents':{str(i):v.tolist() for i,v in getattr(tr,'tangents',{}).items()}},sort_keys=True).encode())
            seen=set()
            for span in getattr(tr,'spans',{}).values():
                if id(span) in seen:continue
                seen.add(id(span))
                fingerprint.update(np.asarray([span.start,span.duration]).tobytes())
                fingerprint.update(span.spline.x.tobytes());fingerprint.update(span.spline.c.tobytes())
        result['trajectory_sha256']=fingerprint.hexdigest()
        end=min(args.end,plan.duration) if args.end is not None else plan.duration
        result.update(coverage_start_s=args.start,coverage_end_s=end)
        print('PLAN READY',plan.duration,flush=True)
        meshes={name:GraspMeshes(k) for name,k in plan.ik.items()}
        for event in getattr(plan,'grasp_events',[]):
            t=event['time'];name=event['parts'][0];arm=event['arm'];q=plan.arms[arm].at(t)
            union={}
            for part in event['parts']:
                observations=meshes[arm].inspect(q,part.split('_')[0],plan.part_at(part,t),plan.part_rotation(part,t))
                for p in observations:
                    if p['link']=='gripper_link' and p['mesh']==0:continue
                    key=(p['link'],p['mesh'])
                    if key not in union or p['distance_m']<union[key]['distance_m']:union[key]=p
            pads=list(union.values())
            observation={**event,'jaw_gaps_m':[p['distance_m'] for p in pads]}
            result['grasp_observations'].append(observation)
            if len(pads)!=2 or any(not .00005<p['distance_m']<.0005 for p in pads):
                raise RuntimeError(f'Grasp does not bracket part with two jaws: {observation}')
        guard=GeometryGuard(plan);progress=time.monotonic()
        for t in np.r_[np.arange(max(0,args.start-args.interval),end,args.interval),end]:
            guard.check({n:tr.at(t) for n,tr in plan.arms.items()},time=float(t))
            if time.monotonic()-progress>10:
                result.update(time_s=float(t),geometry_checks=guard.checked,triangle_queries=guard.parts.checked);save()
                print('CHECK',round(t,2),'/',round(plan.duration,2),flush=True);progress=time.monotonic()
        result['time_s']=float(end)
        result['status']='clear' if args.start==0 and end==plan.duration else 'segment_clear'
    except (RuntimeError,ValueError) as error:
        result.update(status='blocked',error=str(error),time_s=float(t))
        if guard:result['active_tasks']=[x for x in plan.tasks if x['start']<=t<=x['end']]
        print('BLOCKED',error,flush=True)
    except KeyboardInterrupt:
        result.update(status='interrupted',time_s=float(t))
    finally:
        if guard:result.update(geometry_checks=guard.checked,triangle_queries=guard.parts.checked)
        save()
    return 0 if result['status'] in ('clear','segment_clear') else 1

if __name__=='__main__':raise SystemExit(main())
