#!/usr/bin/env python3
"""Offline check of the new scene's REST poses, not a full process clearance pass."""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
import fcl
import numpy as np
from arm_sim.common import ROOT
from arm_sim.conveyor_scene import preview_layout
from arm_sim.demo_kinematics import ArmKinematics
from arm_sim.grasp_geometry import GraspMeshes,separation,collision_object


def check():
    cfg=preview_layout();models=json.loads((ROOT/'assets/generated/conveyor_demo/models.json').read_text())
    objects={};failures=[];poses={};clearances={}
    for name,p in cfg['arms'].items():
        k=ArmKinematics(models[p['model']],p,cfg['table_height']);q=cfg['rest_joints'][name]
        limits=np.asarray(k.model['limits'])
        if not np.isfinite(q).all() or (q<limits[:,0]).any() or (q>limits[:,1]).any():failures.append({'arm':name,'reason':'joint_limits'})
        g=GraspMeshes(k);objs=g.at(q);objects[name]=objs
        for a,b in combinations(objs,2):
            if frozenset([a[0],b[0]]) in g.self_pairs and separation(a[2],b[2])[0]<=1e-7:
                failures.append({'arm':name,'reason':'self_contact','links':[a[:2],b[:2]]})
        for tab in cfg['tables']:
            table=collision_object(fcl.Box(*tab['size'],.03),np.eye(3),[*tab['center'],cfg['table_height']-.015])
            for link,i,obj in objs:
                # Only the original OMX base/table mating surface is intentional.
                own_base=(p['model']=='omx' and link=='link0' and tab['id']==cfg['arm_fixtures'][name]['table'] and p.get('base_plate_thickness')==0. and p.get('base_origin_offset')==0.)
                if own_base:
                    shape=next(v for v in g.geometry.at(q) if v[0]==link and v[1]==i)
                    if shape[4][2]<cfg['table_height']-1e-7:failures.append({'arm':name,'reason':'base_below_table_surface'})
                    continue
                if separation(obj,table)[0]<=1e-7:failures.append({'arm':name,'reason':'table_contact','link':link,'mesh':i})
        poses[name]={'joints':q,'tcp_m':k.position(q).tolist()}
    for a,b in combinations(objects,2):
        distance=min(separation(x[2],y[2])[0] for x in objects[a] for y in objects[b])
        clearances[a+'/'+b]=distance
        if distance<.020:failures.append({'arms':[a,b],'reason':'less_than_20_mm_clearance'})
    return {'status':'rest_pose_clear' if not failures else 'blocked','failures':failures,
            'scope':'URDF triangle meshes at specified rest poses: joint limits, non-neighbour self contact, wooden tabletops, arm-to-arm clearance.',
            'exclusions':['motion paths','cameras and overhead mounts','clamps','source and conveyor jigs','payload and force closure'],
            'full_process_verified':False,'poses':poses,'arm_clearance_m':clearances,
            'layout_sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
            'urdf_sha256':{n:hashlib.sha256((ROOT/m['urdf']).read_bytes()).hexdigest() for n,m in models.items()}}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'outputs/conveyor_preview_latest/rest_check.json')
    args=parser.parse_args();report=check();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':report['status'],'failures':report['failures'],'arm_clearance_m':report['arm_clearance_m']},ensure_ascii=False))
    raise SystemExit(0 if report['status']=='rest_pose_clear' else 1)
