#!/usr/bin/env python3
"""Check the static floor plan and reserved forklift turn circles, not driving."""
import json
import math
import hashlib
import xml.etree.ElementTree as ET
import numpy as np
import fcl
from scipy.spatial.transform import Rotation
from arm_sim.demo_geometry import origin
from arm_sim.grasp_geometry import stl_triangles,collision_object,separation
from arm_sim.conveyor_hardware import forklift_boxes
from itertools import combinations
from arm_sim.common import ROOT
from arm_sim.conveyor_scene import preview_layout


def vehicle_vertices(kind):
    folder=ROOT/'assets/factory/turtlebot3_description'
    root=ET.fromstring((folder/'urdf'/f'turtlebot3_{kind}.urdf').read_text().replace('${namespace}',''))
    transforms={'base_footprint':np.eye(4)}
    for _ in range(len(root.findall('joint'))):
        for joint in root.findall('joint'):
            parent=joint.find('parent').get('link');child=joint.find('child').get('link')
            if parent in transforms:transforms[child]=transforms[parent]@origin(joint.find('origin'))
    vertices=[]
    for link in root.findall('link'):
        for visual in link.findall('visual'):
            mesh=visual.find('geometry/mesh')
            if mesh is None:continue
            filename=folder/mesh.get('filename').split('package://turtlebot3_description/')[1]
            if filename.suffix=='.stl':
                v=stl_triangles(filename)*np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
                tf=transforms[link.get('name')]@origin(visual.find('origin'))
            else:
                # Official URDF collision box represents the stock R200 camera.
                collision=link.find('collision');dims=np.fromstring(collision.find('geometry/box').get('size'),sep=' ')
                v=dims/2*np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
                tf=transforms[link.get('name')]@origin(collision.find('origin'))
            vertices.extend(v@tf[:3,:3].T+tf[:3,3])
    return vertices


def check():
    cfg=preview_layout();failures=[]
    blocks=[(t['id'],t['center'],t['size']) for t in cfg['tables']]
    blocks += [(f'warehouse_{i}',r['center'],r['size']) for i,r in enumerate(cfg['warehouses'])]
    blocks += [(f'waiting_{i}',b['center'],b['size']) for i,b in enumerate(cfg['traffic']['waiting_bays'])]
    for name,center,size in blocks:
        if any(abs(c)+s/2>m/2+1e-9 for c,s,m in zip(center,size,cfg['map']['size'])):
            failures.append(name+' outside map')
    for a,b in combinations(blocks,2):
        overlap=[(sa+sb)/2-abs(x-y) for x,y,sa,sb in zip(a[1],b[1],a[2],b[2])]
        if min(overlap)>1e-8:failures.append(a[0]+' overlaps '+b[0])
    tables={t['id']:t for t in cfg['tables']}
    front=[tables[n]['center'][1]-tables[n]['size'][1]/2 for n in ('supply','assembly')]
    if max(front)-min(front)>1e-8:failures.append('workcell front edges not aligned')
    # Actual planned quarter-turns, not a full 360-degree disc at the map edge.
    vertices=vehicle_vertices('waffle')
    fixed,moving=forklift_boxes(cfg)
    for _,pos,size in fixed+moving:
        vertices.extend(np.array(pos)+np.array(size)/2*np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)]))
    lo=np.min(vertices,axis=0)[:2];hi=np.max(vertices,axis=0)[:2]
    # The loaded carrier is also inside this envelope (236mm lateral,95mm fore/aft).
    lo=np.minimum(lo,[.32-.095/2,-.236/2]);hi=np.maximum(hi,[.32+.095/2,.236/2])
    corners=np.array([[x,y,0.] for x in (lo[0],hi[0]) for y in (lo[1],hi[1])])
    centre=np.r_[(lo+hi)/2,0.];size=np.r_[hi-lo,1.]
    radius=float(max(np.linalg.norm(corners[:,:2],axis=1)))
    # Every angle is at most half a sampling step away; bound its displacement.
    sweep_bound=2*radius*math.sin(math.radians(.5)/2)
    rows=[]
    for turn in cfg['traffic']['forklift_turns']:
        point=turn['center'];clear=float('inf');closest='map boundary'
        for yaw in np.linspace(*turn['yaw_range'],91):
            rot=Rotation.from_euler('z',yaw).as_matrix();v=corners@rot.T+[*point,0.]
            gap=float(np.min(np.array(cfg['map']['size'])/2-np.abs(v[:,:2])))-sweep_bound
            if gap<clear:clear=gap;closest='map boundary'
            obj=collision_object(fcl.Box(*size),rot,rot@centre+[*point,0.])
            for name,center,block_size in blocks:
                other=collision_object(fcl.Box(*block_size,1.),np.eye(3),[*center,0.])
                gap=separation(obj,other)[0]-sweep_bound
                if gap<clear:clear=gap;closest=name
        rows.append({'center_m':point,'yaw_range_rad':turn['yaw_range'],'clearance_m':clear,'closest':closest,'samples':91})
        if clear<.05-1e-9:failures.append(f'turn at {point} has less than 50mm conservative clearance')
    burger_vertices=vehicle_vertices('burger')
    burger_vertices.extend([[-.032+x*cfg['burger']['deck_size'][0]/2,y*cfg['burger']['deck_size'][1]/2,0.] for x in (-1,1) for y in (-1,1)])
    blo=np.min(burger_vertices,axis=0)[:2];bhi=np.max(burger_vertices,axis=0)[:2]
    bcentre=np.r_[(blo+bhi)/2,0.];bsize=np.r_[bhi-blo,1.]
    bcorners=np.array([[x,y,0.] for x in (blo[0],bhi[0]) for y in (blo[1],bhi[1])])
    bbound=2*max(np.linalg.norm(bcorners[:,:2],axis=1))*math.sin(math.radians(.5)/2)
    burger_turns=[]
    for point,yaws in [([cfg['loading_dock'][0],-.420],[-math.pi/2,0.]),([cfg['unload_dock'][0],-.420],[0.,math.pi/2])]:
        clear=float('inf')
        for yaw in np.linspace(*yaws,91):
            rot=Rotation.from_euler('z',yaw).as_matrix();v=bcorners@rot.T+[*point,0.]
            clear=min(clear,float(np.min(np.array(cfg['map']['size'])/2-np.abs(v[:,:2])))-bbound)
            obj=collision_object(fcl.Box(*bsize),rot,rot@bcentre+[*point,0.])
            for _,center,block_size in blocks:
                other=collision_object(fcl.Box(*block_size,1.),np.eye(3),[*center,0.])
                clear=min(clear,separation(obj,other)[0]-bbound)
        burger_turns.append({'center_m':point,'clearance_m':clear,'samples':91})
        if clear<.05:failures.append(f'Burger turn at {point} has less than 50mm clearance')
    # Parked spare must clear each straight travel segment, including both rack
    # approaches. Sweep an entire footprint rectangle, not just a centreline.
    parking=[]
    for bay in cfg['traffic']['waiting_bays']:
        parked=collision_object(fcl.Box(*bay['size'],1.),np.eye(3),[*bay['center'],0.])
        for route in cfg['traffic']['routes']:
            vehicle_size=np.array([.30,.30,1.]) if route['name']=='MaterialU' else size.copy()
            vehicle_center=np.zeros(3) if route['name']=='MaterialU' else centre
            gaps=[]
            for start,end in zip(route['points'],route['points'][1:]):
                delta=np.array(end)-start;length=float(np.linalg.norm(delta))
                rot=Rotation.from_euler('z',math.atan2(delta[1],delta[0])).as_matrix()
                swept_size=vehicle_size+[length,0,0]
                swept=collision_object(fcl.Box(*swept_size),rot,rot@vehicle_center+[*((np.array(start)+end)/2),0.])
                gaps.append(separation(swept,parked)[0])
            gap=min(gaps);parking.append({'route':route['name'],'clearance_m':gap})
            if gap<.05:failures.append('spare parking blocks '+route['name']+' (less than 50mm footprint clearance)')
    return {'status':'pass' if not failures else 'blocked','failures':failures,
            'layout_sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
            'spare_parking_route_clearance':parking,
            'burger_turns':burger_turns,'burger_envelope_bounds_m':[blo.tolist(),bhi.tolist()],
            'map_size_m':cfg['map']['size'],'workcell_front_y_m':front[0],
            'envelope_bounds_m':[lo.tolist(),hi.tolist()],'sampling_displacement_bound_m':sweep_bound,'turns':rows,
            'scope':'table/rack/parking rectangles; URDF+modified-lift footprint quarter-turns; spare parking versus continuous straight route sweeps (Burger conservative 300mm square)',
            'exclusions':['full driving trajectories','docking insertion','simultaneous vehicle movement','measured vehicle dimensions','stock R200 visual mesh (official collision box used)','traffic controller'],
            'traffic_control_implemented':False}

if __name__=='__main__':
    report=check();(ROOT/'outputs/conveyor_preview_latest/floorplan_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False));raise SystemExit(report['status']!='pass')
