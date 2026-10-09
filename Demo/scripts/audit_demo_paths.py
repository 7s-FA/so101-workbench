#!/usr/bin/env python3
"""Collect failures throughout a virtual trajectory; never drives robots or a server.

Unlike the runtime fail-stop guard, this audit continues *offline* past invalid
poses to identify later faults. Continuing is not permission to play these poses.
"""
import argparse
import hashlib
import json
import re
import time
from itertools import combinations
from pathlib import Path
import numpy as np
from arm_sim.factory import load_layout
from arm_sim.factory_demo import DemoPlan, demo_layout
from arm_sim.demo_geometry import GeometryGuard, intersections
from arm_sim.demo_validation import route_check


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interval',type=float,default=.04)
    parser.add_argument('--start',type=float,default=0.)
    parser.add_argument('--end',type=float)
    parser.add_argument('--output',type=Path,default=Path('outputs/demo/path_audit.json'))
    args=parser.parse_args()
    if not .001<=args.interval<=.1:parser.error('interval must be 0.001..0.1 seconds')
    if args.start<0 or (args.end is not None and args.end<=args.start):parser.error('Invalid coverage interval')
    started=time.monotonic();plan=DemoPlan(demo_layout(load_layout()));guard=GeometryGuard(plan)
    result={'status':'running','scope':'Default demo at original follower height. Offline continuation past faults; not runtime execution or contact physics validation.',
            'interval_s':args.interval,'plan':plan.validate(),'faults':{},'arm_pairs':{},'samples':0}
    digest=hashlib.sha256(json.dumps(plan.cfg,sort_keys=True).encode())
    for n,tr in plan.arms.items():
        digest.update(n.encode());digest.update(np.asarray(tr.times).tobytes());digest.update(np.asarray(tr.values).tobytes())
        digest.update(json.dumps({'smooth':getattr(tr,'smooth',False),'tangents':{str(i):v.tolist() for i,v in getattr(tr,'tangents',{}).items()}},sort_keys=True).encode())
    result['trajectory_sha256']=digest.hexdigest()
    try:result['vehicle_routes']=route_check(plan)
    except ValueError as e:result['vehicle_routes']={'result':'blocked','error':str(e)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        result['elapsed_s']=time.monotonic()-started
        args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    def record(bucket,key,t,detail):
        if key not in bucket:bucket[key]={'first_s':float(t),'last_s':float(t),'samples':0,'example':detail}
        b=bucket[key];b['last_s']=float(t);b['samples']+=1
    end=min(plan.duration,args.end) if args.end is not None else plan.duration
    result.update(coverage_start_s=args.start,coverage_end_s=end)
    print('PLAN',plan.duration,flush=True);save();progress=time.monotonic();frame_cache={};pair_cache={}
    # Include segment and task boundaries even when not on the regular grid.
    times=set(np.arange(0,plan.duration,args.interval));times.add(plan.duration)
    for tr in plan.arms.values():times.update(tr.times)
    try:
        times.update([max(0,args.start-args.interval),end])
        for t in sorted(t for t in times if max(0,args.start-args.interval)<=t<=end):
            joints={n:tr.at(t) for n,tr in plan.arms.items()}
            # Pair checks run independently: a fixture/grasp fault must not hide
            # a later B-C or C-D problem. These use conservative URDF envelopes.
            frames={}
            for n,g in guard.geometry.items():
                key=tuple(joints[n])
                if frame_cache.get(n,(None,))[0]!=key:frame_cache[n]=(key,g.at(joints[n]))
                frames[n]=frame_cache[n][1]
            for a,b in combinations(frames,2):
                key=(frame_cache[a][0],frame_cache[b][0])
                if pair_cache.get((a,b),(None,))[0]!=key:
                    hit=intersections(frames[a],frames[b],margin=guard.arm_clearance)
                    overlap=intersections(frames[a],frames[b],margin=-.00001) if hit else []
                    pair_cache[a,b]=(key,hit,overlap)
                _,hit,overlap=pair_cache[a,b]
                if hit:
                    record(result['arm_pairs'],a+'/'+b,t,{'clearance_guard_m':guard.arm_clearance,'links':hit,'penetrating_envelopes':overlap})
            try:guard.check(joints,time=t)
            except RuntimeError as e:
                message=re.sub(r' at [0-9.]+s$', '', str(e))
                record(result['faults'],message,t,{'active_tasks':[x['id'] for x in plan.tasks if x['start']<=t<=x['end']],
                       'joints':{n:q.tolist() for n,q in joints.items()}})
                # Reset the failed interval's start only in this offline audit.
                # Production GeometryGuard remains fail-stop and is unchanged.
                guard.last={n:q.copy() for n,q in joints.items()};guard.last_time=t
            result.update(samples=result['samples']+1,checked_through_s=float(t))
            if time.monotonic()-progress>10:
                save();print('AUDIT',round(t,2),'/',round(plan.duration,2),'fault types',len(result['faults']),flush=True);progress=time.monotonic()
        result['status']='blocked' if result['faults'] or result['arm_pairs'] or result['vehicle_routes']['result']!='clear' else 'clear'
    except KeyboardInterrupt:result['status']='interrupted'
    finally:save()
    print(result['status'],flush=True)
    return 0 if result['status']=='clear' else 1

if __name__=='__main__':raise SystemExit(main())
