#!/usr/bin/env python3
"""Check camera assembly and rest-arm clearance using rendered mesh transforms."""
import json
import hashlib
import numpy as np
import fcl
from itertools import combinations
from arm_sim.common import ROOT
from arm_sim.conveyor_scene import preview_layout
from arm_sim.overhead_camera import assembly,support_boxes
from arm_sim.grasp_geometry import stl_triangles,triangle_model,collision_object,separation,GraspMeshes
from arm_sim.demo_kinematics import ArmKinematics
from arm_sim.conveyor_hardware import clamp_boxes,d435_assembly,d435_vertices


def check():
    cfg=preview_layout();meshes,eye,direction=assembly(cfg);objs={};failures=[];joints={}
    table=next(t for t in cfg['tables'] if t['id']=='assembly')
    for name,path,scale,tf in meshes:
        vertices=stl_triangles(path)*scale
        objs[name]=collision_object(triangle_model(vertices),tf[:3,:3],tf[:3,3])
        if name in ('ArmBase','Bottom'):
            world=vertices@tf[:3,:3].T+tf[:3,3];foot=world[world[:,2]<cfg['table_height']+.012]
            if (abs(foot[:,:2]-table['center'])>np.array(table['size'])/2+1e-6).any():failures.append({'part':name,'reason':'foot_outside_table'})
            if abs(world[:,2].min()-cfg['table_height'])>1e-6:failures.append({'part':name,'reason':'foot_not_seated_on_table'})
    for a,b in combinations(objs,2):
        distance=separation(objs[a],objs[b])[0];joints[a+'/'+b]=distance
        # Adjacent printed parts seat flush; surface contact is intentional.
        if distance < -1e-7:failures.append({'parts':[a,b],'reason':'intersecting_surfaces'})
    for name,pos,size in support_boxes(cfg):objs[name]=collision_object(fcl.Box(*size),np.eye(3),pos)
    models=json.loads((ROOT/'assets/generated/conveyor_demo/models.json').read_text());clearances={};pan_clearance=None
    for name,pose in cfg['arms'].items():
        kin=ArmKinematics(models[pose['model']],pose,cfg['table_height']);geometry=GraspMeshes(kin)
        arm=geometry.at(cfg['rest_joints'][name])
        distances=[]
        for part in arm:
            for fixture_name,fixture in objs.items():
                d=separation(part[2],fixture)[0]
                attached=(name==cfg['assembly_camera']['attached_arm'] and part[0]=='base_link' and fixture_name=='ArmBase')
                if attached:
                    if d<0:failures.append({'arm':name,'reason':'arm_penetrates_shared_base'})
                    continue
                distances.append(d)
                # The integrated mount sits beside the fixed base and first
                # bearing; the former independent-stand 20mm rule is inapplicable
                # here. Keep 5mm for these local parts and sweep the pan below.
                local_mount=(name==cfg['assembly_camera']['attached_arm'] and part[0] in ('base_link','shoulder_link') and fixture_name in ('ArmBase','Bottom'))
                limit=.005 if local_mount else .02
                if d<limit:failures.append({'arm':name,'part':part[:2],'fixture':fixture_name,'reason':'camera_fixture_clearance','required_m':limit,'distance_m':d})
        clearances[name]=min(distances)
        if name==cfg['assembly_camera']['attached_arm']:
            pan_clearance=float('inf')
            for angle in np.linspace(*kin.model['limits'][0],81):
                q=list(cfg['rest_joints'][name]);q[0]=float(angle)
                for link,_,part in geometry.at(q):
                    if link!='shoulder_link':continue
                    for fixture in ('ArmBase','Bottom'):pan_clearance=min(pan_clearance,separation(part,objs[fixture])[0])
            if pan_clearance<.005:failures.append({'reason':'first_bearing_pan_clearance_below_5mm','distance_m':pan_clearance})
        if kin.so:
            wrist=geometry.geometry.transforms(cfg['rest_joints'][name])['gripper_link'];mount,camera=d435_assembly()
            attached_meshes=[(stl_triangles(ROOT/'vendor/so_arm100/Optional/Wrist_Cam_Mount_RealSense_D435/stl/Wrist_cam_mount_RealSense_D435.stl')*.001,mount),(d435_vertices(),camera)]
            for vertices,offset in attached_meshes:
                tf=wrist@offset;part=collision_object(triangle_model(vertices),tf[:3,:3],tf[:3,3])
                if min(separation(part,fixture)[0] for fixture in objs.values())<.02:failures.append({'arm':name,'reason':'D435_to_overhead_clearance_below_20mm'})
        for leaf,pos,size in clamp_boxes(cfg,name):
            clamp=collision_object(fcl.Box(*size),np.eye(3),pos)
            for fixture_name,fixture in objs.items():
                if separation(clamp,fixture)[0]<=0:failures.append({'arm':name,'clamp':leaf,'fixture':fixture_name,'reason':'clamp_intersects_camera_fixture'})
    # The optical ray must hit the table, not point above or behind it.
    ground=eye+direction*((cfg['table_height']-eye[2])/direction[2])
    table=next(t for t in cfg['tables'] if t['id']=='assembly')
    if not (np.abs(ground[:2]-table['center'])<np.array(table['size'])/2).all():failures.append({'reason':'optical_axis_misses_table'})
    return {'status':'pass' if not failures else 'blocked','failures':failures,
            'layout_sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
            'camera_count':cfg['assembly_camera']['count'],'arm_to_camera_clearance_m':clearances,
            'first_bearing_pan_samples':81,'first_bearing_to_shared_mount_clearance_m':pan_clearance,
            'mount_mesh_distance_m':joints,'optical_axis_table_intersection_m':ground.tolist(),
            'scope':'static mesh assembly and camera support versus URDF arm meshes at rest',
            'exclusions':['full moving arm envelope (only first bearing pan swept)','optical calibration','clamp strength','hardware mounting measurements']}

if __name__=='__main__':
    result=check();(ROOT/'outputs/conveyor_preview_latest/overhead_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));raise SystemExit(result['status']!='pass')
