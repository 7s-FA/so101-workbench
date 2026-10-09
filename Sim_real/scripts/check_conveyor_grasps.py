#!/usr/bin/env python3
"""Check isolated vertical approaches and closing with the revised finger STL.

This excludes robot IK, fixtures, cameras, motion dynamics and force closure.
"""
import json
from pathlib import Path
import numpy as np
from arm_sim.common import ROOT
from arm_sim.demo_kinematics import ArmKinematics
from arm_sim.grasp_geometry import GraspMeshes,triangle_model,stl_triangles,collision_object,separation


def check():
    models=json.loads((ROOT/'assets/generated/conveyor_demo/models.json').read_text())
    cfg=json.loads((ROOT/'config/conveyor_demo.json').read_text())
    profiles=json.loads((ROOT/'config/conveyor_grasps.json').read_text())['profiles']
    kin=ArmKinematics(models['so101_xlerobot'],{'position':[0,0],'yaw':0},0)
    meshes=GraspMeshes(kin);q=np.zeros(6);tool=kin.transform(q);rotation=tool[:3,:3]@np.diag([1,-1,-1]);report={};failures=[]
    for kind,p in profiles.items():
        source=ROOT/'assets/product/02_ball_d30.stl' if kind=='ball' else ROOT/'assets/generated/conveyor_demo/parts'/cfg['parts'][kind]
        shape=triangle_model(stl_triangles(source)*.001)
        samples=[('approach',p['open'],lift) for lift in np.linspace(.06,0,61)]
        samples += [('close',angle,0.) for angle in np.linspace(p['open'],p['closed'],41)]
        minimum=float('inf');jaw_distances={};normals={};contacts={}
        for phase,angle,lift in samples:
            q[5]=angle;offset=np.asarray(p['part_base_in_tool'])+[0,0,lift]
            part=collision_object(shape,rotation,tool[:3,3]+tool[:3,:3]@offset)
            for link,index,obj in meshes.jaws(q):
                distance,nearest=separation(obj,part);minimum=min(minimum,distance)
                if distance<=1e-7:
                    failures.append({'kind':kind,'phase':phase,'angle':float(angle),'lift_m':float(lift),'link':link,'mesh':index});break
                if phase=='close' and angle==p['closed'] and (link,index) in [('gripper_link',2),('moving_jaw_so101_v1_link',1)]:
                    jaw_distances[link]=distance
                    normals[link]=((nearest[0]-nearest[1])@tool[:3,:3]/distance).tolist()
                    contacts[link]=((nearest[1]-part.getTranslation())@rotation).tolist()
            if failures and failures[-1]['kind']==kind:break
        report[kind]={'samples':len(samples),'minimum_clearance_m':minimum,'closed_finger_gap_m':jaw_distances,'contact_normals_in_tool':normals,'contact_points_in_part_m':contacts}
        if len(jaw_distances)!=2 or any(not .00005<x<.0005 for x in jaw_distances.values()):
            failures.append({'kind':kind,'reason':'two-sided bracketing was not verified'})
        if len(normals)==2:
            left,right=list(normals.values())
            if left[0]<.7 or right[0]>-.7 or np.dot(left,right)>-.7:
                failures.append({'kind':kind,'reason':'contact directions do not oppose across the part'})
    return {'status':'isolated_approach_clear' if not failures else 'blocked','parts':report,'failures':failures,
            'scope':__doc__,'full_process_verified':False,'physical_grasp_verified':False}

if __name__=='__main__':
    result=check();out=ROOT/'outputs/conveyor_preview_latest/grasp_check.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));raise SystemExit(bool(result['failures']))
