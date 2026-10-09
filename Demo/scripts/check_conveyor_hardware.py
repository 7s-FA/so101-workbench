#!/usr/bin/env python3
"""Audit previously omitted fixtures with the same geometry as the renderer."""
import json
import hashlib
import math
import xml.etree.ElementTree as ET
import numpy as np
import fcl
from itertools import combinations
from scipy.spatial.transform import Rotation
from arm_sim.common import ROOT
from arm_sim.conveyor_scene import preview_layout
from arm_sim.conveyor_hardware import BASE_FOOT,clamp_boxes,d435_assembly,d435_vertices,d435_adapter_meshes,d435_optical_pose,shipping_boxes,forklift_boxes
from arm_sim.grasp_geometry import GraspMeshes,stl_triangles,triangle_model,collision_object,separation
from arm_sim.demo_kinematics import ArmKinematics
from arm_sim.demo_geometry import origin


def box(pos,size):return collision_object(fcl.Box(*size),np.eye(3),pos)


def side_section(vertices):
    """Intersect triangles with gripper Y=0; use the visible outer side (X<-33mm)."""
    points=[]
    for tri in vertices.reshape(-1,3,3):
        for a,b in zip(tri,np.roll(tri,1,axis=0)):
            if a[1]*b[1]<0:
                p=a+(b-a)*(-a[1]/(b[1]-a[1]))
                if p[0]<-.033:points.append(p)
    return np.asarray(points)


def check():
    cfg=preview_layout();models=json.loads((ROOT/'assets/generated/conveyor_demo/models.json').read_text());failures=[];metrics={}
    table=next(t for t in cfg['tables'] if t['id']=='assembly');lo=np.array(table['center'])-np.array(table['size'])/2;hi=np.array(table['center'])+np.array(table['size'])/2
    mount,camera=d435_assembly()
    mount_model=triangle_model(stl_triangles(ROOT/'vendor/so_arm100/Optional/Wrist_Cam_Mount_RealSense_D435/stl/Wrist_cam_mount_RealSense_D435.stl')*.001)
    camera_model=triangle_model(d435_vertices())
    adapters=[(name,triangle_model(v),np.eye(4)) for name,v in d435_adapter_meshes()]
    # Compare the SAME projected edges visible in the user's side-on close-up.
    # This is an edge spacing, separate from the shortest 3D mesh distance.
    model=models['so101_xlerobot'];root=ET.parse(ROOT/model['urdf']).getroot()
    co=root.find("link[@name='gripper_link']/collision[@name='xlerobot_fixed_finger']")
    mesh=co.find('geometry/mesh');ft=origin(co.find('origin'))
    finger=stl_triangles(mesh.get('filename'))*np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
    finger=finger@ft[:3,:3].T+ft[:3,3]
    mount_vertices=stl_triangles(ROOT/'vendor/so_arm100/Optional/Wrist_Cam_Mount_RealSense_D435/stl/Wrist_cam_mount_RealSense_D435.stl')*.001
    mount_vertices=mount_vertices@mount[:3,:3].T+mount[:3,3]
    side_gap=side_section(mount_vertices)[:,2].min()-side_section(finger)[:,2].max()
    metrics['mount_fixed_finger_side_profile_gap_m']=float(side_gap)
    metrics['mount_fixed_finger_shortest_3d_gap_m']=separation(collision_object(triangle_model(mount_vertices),np.eye(3),np.zeros(3)),collision_object(triangle_model(finger),np.eye(3),np.zeros(3)))[0]
    metrics['photo_side_gap_estimate_mm']=4.4
    metrics['photo_side_gap_uncertainty_mm']=1.0
    if not .0034<=side_gap<=.0054:failures.append('mount/finger side gap outside approximate photo range')
    cameras={};all_arms={}

    for name,pose in cfg['arms'].items():
        k=ArmKinematics(models[pose['model']],pose,cfg['table_height']);g=GraspMeshes(k);q=cfg['rest_joints'][name];arm=g.at(q);all_arms[name]=arm
        clamp_min=1.
        for leaf,pos,size in clamp_boxes(cfg,name):
            obj=box(pos,size)
            for link,i,part in arm:
                d=separation(obj,part)[0];clamp_min=min(clamp_min,d)
                if d<=1e-8:failures.append(f'{name}/{leaf} intersects {link}/{i}')
        metrics[name+'_clamp_to_arm_m']=clamp_min
        fixture=cfg['arm_fixtures'][name]
        support=next(t for t in cfg['tables'] if t['id']==fixture['table'])
        low=np.array(support['center'])-np.array(support['size'])/2
        high=np.array(support['center'])+np.array(support['size'])/2
        bounds=BASE_FOOT if pose['model']=='so101_xlerobot' else (-.060,.060,-.075,.075)
        foot=np.array([[x,y] for x in bounds[:2] for y in bounds[2:]])@Rotation.from_euler('z',pose['yaw']).as_matrix()[:2,:2].T+pose['position']
        gaps=np.r_[foot.min(0)-low,high-foot.max(0)]
        metrics[name+'_support_edge_gaps_mm']=(gaps*1000).tolist()
        if min(gaps)<-.0001:failures.append(name+' base extends beyond tabletop')
        edge_index={'left':0,'front':1,'right':2,'back':3}[fixture['side']]
        if abs(gaps[edge_index])>.0001:failures.append(name+' base is not flush with mounting edge')
        if 'corner' in fixture and abs(gaps[1])>.0001:failures.append(name+' base does not meet front corner')
        if pose['model']!='so101_xlerobot':continue
        tf=g.geometry.transforms(q)['gripper_link']
        cameras[name]=[]
        for label,model,offset in [('mount',mount_model,mount),('camera',camera_model,camera)]+adapters:
            world=tf@offset;obj=collision_object(model,world[:3,:3],world[:3,3]);cameras[name].append(obj)
            distances=[]
            for link,i,part in arm:
                d=separation(obj,part)[0];distances.append(d)
                if d<=1e-8:failures.append(f'{name}/{label} intersects {link}/{i}')
            metrics[name+'_'+label+'_arm_clearance_m']=min(distances)
        metrics[name+'_camera_mount_gap_m']=separation(*cameras[name][:2])[0]
        if metrics[name+'_camera_mount_gap_m']<=1e-8:failures.append(name+' camera intersects its mount')
        for index,obj in enumerate(cameras[name][2:]):
            for fixture in cameras[name][:2]:
                if separation(obj,fixture)[0]<=1e-8:failures.append(f'{name} adapter {index} intersects camera or bracket')
        # Sweep every camera/adapter against the moving jaw through the URDF range.
        sweep_min=1.;sweep_hits=set();closest=None
        for opening in np.linspace(*k.model['limits'][-1],81):
            pose_q=list(q);pose_q[-1]=float(opening)
            for link,i,part in g.at(pose_q):
                if link!='moving_jaw_so101_v1_link':continue
                for index,obj in enumerate(cameras[name]):
                    d=separation(obj,part)[0]
                    if d<sweep_min:
                        sweep_min=d;closest={'attachment':(['mount','camera']+[a[0] for a in adapters])[index],'jaw_mesh':i,'opening_rad':float(opening)}
                    if d<=1e-8:sweep_hits.add(f'{name} camera assembly intersects moving jaw/{i} at {opening:.3f}')
        failures.extend(sorted(sweep_hits));metrics[name+'_opening_sweep_min_m']=sweep_min
        metrics[name+'_opening_sweep_closest']=closest
        metrics[name+'_opening_axis_vertical_component']=abs(float(tf[2,0]))
        if abs(tf[2,0])>.01:failures.append(name+' opening axis is not horizontal')
        eye,forward,up=d435_optical_pose(tf);left=np.cross(up,forward)
        target=tf[:3,3]+tf[:3,:3]@np.array([.0202,0,-.09]);delta=target-eye
        depth=float(delta@forward)
        angles=np.degrees(np.arctan2([delta@left,delta@up],depth))
        metrics[name+'_grasp_roi_camera_angles_deg']=angles.tolist()
        if depth<=0 or abs(angles[0])>40 or abs(angles[1])>28:failures.append(name+' grasp point outside nominal camera view')
        # A thin line of sight to the nominal gap centre must clear both fingers.
        direction=delta/np.linalg.norm(delta)
        ray_x=np.cross([0,1,0] if abs(direction[1])<.9 else [1,0,0],direction);ray_x/=np.linalg.norm(ray_x)
        ray_rotation=np.column_stack((ray_x,np.cross(direction,ray_x),direction))
        ray=collision_object(fcl.Cylinder(.0001,np.linalg.norm(delta)),ray_rotation,(eye+target)/2)
        blockers=[f'{link}/{i}' for link,i,obj in arm if separation(ray,obj)[0]<=0]
        metrics[name+'_gap_view_blockers']=blockers
        if blockers:failures.append(name+' finger-gap view is occluded by '+','.join(blockers))

        for tab in cfg['tables']:
            table_obj=box([*tab['center'],cfg['table_height']-.015],[*tab['size'],.03])
            if min(separation(obj,table_obj)[0] for obj in cameras[name])<=0:failures.append(name+' camera assembly intersects table')

    expanded={n:[a[2] for a in arm]+cameras.get(n,[]) for n,arm in all_arms.items()}
    for a,b in combinations(expanded,2):
        d=min(separation(x,y)[0] for x in expanded[a] for y in expanded[b]);metrics[a+'_'+b+'_including_cameras_m']=d
        if d<.020:failures.append(f'{a}/{b} less than 20mm including cameras')
    # Workpiece fixtures, support, and side marker include the actual supplied meshes.
    rev=json.loads((ROOT/'config/conveyor_demo.json').read_text());parts=ROOT/'assets/generated/conveyor_demo/parts'
    specs=[('assembly','housing',cfg['assembly_jig']),('handoff','housing',cfg['handoff'])]
    specs += [(f"prep_{p['kit']}_{p['kind']}",p['kind'],p['position']) for p in cfg['preparation_slots']]
    specs=[(*row,'assembly') for row in specs]
    for bank,(cx,cy) in enumerate(cfg['source_banks']):
        if bank not in cfg['initial_materials']['source_loaded_banks']:continue
        for col,kind in enumerate(('housing','ball','cap')):
            for row in range(3):specs.append((f'source_{bank}_{kind}_{row}',kind,[cx+(col-1)*.075,cy+(row-1)*.075],cfg['source_table']))
    objects=[]
    for name,kind,pos,table_id in specs:
        support=next(t for t in cfg['tables'] if t['id']==table_id)
        lo=np.array(support['center'])-np.array(support['size'])/2
        hi=np.array(support['center'])+np.array(support['size'])/2
        jig=collision_object(triangle_model(stl_triangles(parts/rev['parts'][kind+'_jig'])*.001),np.eye(3),[*pos,cfg['table_height']])
        objects.append((name,jig))
        if (np.array(pos)-.034<lo).any() or (np.array(pos)+.034>hi).any():failures.append(name+' extends beyond supporting tabletop')
        gap=min(separation(jig,x)[0] for shapes in expanded.values() for x in shapes)
        metrics[name+'_rest_arm_gap_m']=gap
        if gap<.005:failures.append(name+' less than 5mm from a resting arm')
    for (a,x),(b,y) in combinations(objects,2):
        if separation(x,y)[0]<.005:failures.append(a+'/'+b+' fixture clearance below 5mm')
    # Check the visible, seated source parts as well as their jigs.
    seating=rev['jig_seating'];loaded=[]
    product_models={kind:triangle_model(stl_triangles(ROOT/'assets/product/02_ball_d30.stl' if kind=='ball' else parts/rev['parts'][kind])*.001) for kind in ('housing','ball','cap')}
    for name,kind,pos,table_id in specs:
        if not name.startswith('source_'):continue
        xyz=[pos[0]+seating['offset_xy'][0],pos[1]+seating['offset_xy'][1],cfg['table_height']+seating['contact_height'][kind]+seating['render_gap']]
        loaded.append((name,collision_object(product_models[kind],np.eye(3),xyz)))
    for vehicle_index in cfg['initial_materials']['burger_loaded_indices']:
        vx,vy,yaw=cfg['burger']['poses'][vehicle_index];rot=Rotation.from_euler('z',yaw).as_matrix()
        for col,kind in enumerate(('housing','ball','cap')):
            for row in range(3):
                pos=rot@np.array([-.032+(col-1)*.075+seating['offset_xy'][0],(row-1)*.075+seating['offset_xy'][1],cfg['burger']['deck_top']+seating['contact_height'][kind]+seating['render_gap']])+[vx,vy,0.]
                loaded.append((f'burger_{vehicle_index}_{kind}_{row}',collision_object(product_models[kind],rot,pos)))
    for name,obj in loaded:
        gap=min(separation(obj,x)[0] for shapes in expanded.values() for x in shapes)
        metrics[name+'_loaded_part_rest_arm_gap_m']=gap
        if gap<.005:failures.append(name+' loaded part less than 5mm from a resting arm')
    metrics['loaded_parts_checked']=len(loaded)
    for name,point,table_id in [('supply',cfg['loading_dock'],cfg['source_table']),('unload',cfg['unload_dock'],'assembly')]:
        support=next(t for t in cfg['tables'] if t['id']==table_id)
        gap=support['center'][1]-support['size'][1]/2-(point[1]+.032+cfg['burger']['deck_size'][0]/2)
        metrics[name+'_cargo_to_table_plan_gap_m']=gap
        if gap<.020-1e-9:failures.append(name+' cargo has less than 20mm tabletop clearance')
    metrics['assembly_table_size_m']=table['size'];metrics['conveyor_enabled']=cfg['conveyor']['enabled']
    if cfg['conveyor']['enabled']:failures.append('conveyor must be absent')
    if {p['id'] for p in cfg['docking_panels']}!={'supply','unload'}:failures.append('forklift markers remain or Burger markers missing')
    supports=[box(pos,size) for _,pos,size in shipping_boxes(cfg)]
    x,y=cfg['shipping_dock'];h=cfg['shipping_height']
    supports.append(box([x,y,h+.002],[.236,.095,.004]))
    pallet=triangle_model(stl_triangles(parts/rev['parts']['pallet'])*.001)
    supports.extend(collision_object(pallet,np.eye(3),[x+(i-1)*.078,y,h+.004]) for i in range(3))
    # Exercise the complete intended lift stroke against fixed frame members.
    min_clear=1.;samples=0
    allowed={('ScrewNut','LeadScrew'),('NutBridge','LeadScrew')}
    for height in np.linspace(cfg['waffle']['fork_top'],cfg['waffle']['fork_max_top'],57):
        fixed,moving=forklift_boxes(cfg,float(height));samples+=1
        for mn,mp,ms in moving:
            for fn,fp,fs in fixed:
                if (mn,fn) in allowed or (mn.startswith('Slider') and fn.startswith('Mast') and not fn.startswith('MastAnchor')):continue
                d=separation(box(mp,ms),box(fp,fs))[0];min_clear=min(min_clear,d)
                if d< -1e-8:failures.append(f'lift {height:.4f}: {mn} intersects {fn}')
    # The lift must pass through the real stand slots, not through its boards.
    vehicle=cfg['waffle']['pose'];rot=Rotation.from_euler('z',vehicle[2]).as_matrix()
    stand_clear=1.
    for height in np.linspace(cfg['waffle']['fork_top'],cfg['waffle']['fork_max_top'],57):
        _,moving=forklift_boxes(cfg,float(height))
        for leaf,pos,size in moving:
            obj=collision_object(fcl.Box(*size),rot,rot@pos+[*vehicle[:2],.001])
            stand_clear=min(stand_clear,*(separation(obj,s)[0] for s in supports[:len(shipping_boxes(cfg))]))
    metrics['lift_to_shipping_stand_m']=stand_clear
    if stand_clear<=0:failures.append('lift intersects shipping stand')
    metrics['lift_samples']=samples;metrics['lift_unintended_frame_clearance_m']=min_clear
    return {'status':'pass' if not failures else 'blocked','failures':failures,'metrics':metrics,
            'hardware_source_sha256':hashlib.sha256((ROOT/'arm_sim/conveyor_hardware.py').read_bytes()).hexdigest(),
            'real_assembly_verified':False,'prototype_adapter_required':False,'holder_fastener_count':5,'hole_registration':'photo-based; manually drilled holes confirmed; dimensions unmeasured',
            'layout_sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
            'scope':'base edge alignment, rest-arm meshes including D435, clamp bodies, assembly jig, shipping stand, 57 lift heights, 81 gripper opening samples per SO101, nominal camera ROI, horizontal opening axis',
            'exclusions':['full arm motion','measured drilled-hole coordinates and screw bores','fastener strength','lift stability and rated payload','LiDAR visibility','physical camera calibration']}

if __name__=='__main__':
    r=check();(ROOT/'outputs/conveyor_preview_latest/hardware_check.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));raise SystemExit(r['status']!='pass')
