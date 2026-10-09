#!/usr/bin/env python3
"""Check camera assembly and rest-arm clearance using rendered mesh transforms."""
import json
import numpy as np
import fcl
from itertools import combinations
from arm_sim.common import ROOT
from arm_sim.conveyor_scene import preview_layout
from arm_sim.overhead_camera import assembly,support_boxes
from arm_sim.grasp_geometry import stl_triangles,triangle_model,collision_object,separation,GraspMeshes
from arm_sim.demo_kinematics import ArmKinematics


def check():
    cfg=preview_layout();meshes,eye,direction=assembly(cfg);objs={};failures=[];joints={}
    for name,path,scale,tf in meshes:
        objs[name]=collision_object(triangle_model(stl_triangles(path)*scale),tf[:3,:3],tf[:3,3])
    for a,b in combinations(objs,2):
        distance=separation(objs[a],objs[b])[0];joints[a+'/'+b]=distance
        # Adjacent printed parts seat flush; surface contact is intentional.
        if distance < -1e-7:failures.append({'parts':[a,b],'reason':'intersecting_surfaces'})
    for name,pos,size in support_boxes(cfg):objs[name]=collision_object(fcl.Box(*size),np.eye(3),pos)
    models=json.loads((ROOT/'assets/generated/conveyor_demo/models.json').read_text());clearances={}
    for name,pose in cfg['arms'].items():
        arm=GraspMeshes(ArmKinematics(models[pose['model']],pose,cfg['table_height'])).at(cfg['rest_joints'][name])
        d=min(separation(part[2],fixture)[0] for part in arm for fixture in objs.values());clearances[name]=d
        if d<.02:failures.append({'arm':name,'reason':'camera_fixture_clearance_below_20mm','distance_m':d})
    # The optical ray must hit the table, not point above or behind it.
    ground=eye+direction*((cfg['table_height']-eye[2])/direction[2])
    table=next(t for t in cfg['tables'] if t['id']=='assembly')
    if not (np.abs(ground[:2]-table['center'])<np.array(table['size'])/2).all():failures.append({'reason':'optical_axis_misses_table'})
    return {'status':'pass' if not failures else 'blocked','failures':failures,
            'camera_count':cfg['assembly_camera']['count'],'arm_to_camera_clearance_m':clearances,
            'mount_mesh_distance_m':joints,'optical_axis_table_intersection_m':ground.tolist(),
            'scope':'static mesh assembly and camera support versus URDF arm meshes at rest',
            'exclusions':['moving arms','D435 attachments','optical calibration','clamp strength','hardware mounting measurements']}

if __name__=='__main__':
    result=check();(ROOT/'outputs/conveyor_preview_latest/overhead_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));raise SystemExit(result['status']!='pass')
