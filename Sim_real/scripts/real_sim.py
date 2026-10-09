#!/usr/bin/env python3
"""Finite Isaac Sim render of the four-arm candidate layout; no web server."""
import argparse
import copy
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=Path('outputs/sim_real_demo'))
parser.add_argument('--verify',action='store_true',help='Finite complete kinematic playback without server')
parser.add_argument('--port',type=int,default=8767)
parser.add_argument('--steps',type=int,default=0)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True,device='cpu',enable_cameras=True)
args=parser.parse_args();args.enable_cameras=True
launcher=AppLauncher(args);app=launcher.app

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from pxr import UsdGeom,UsdShade,UsdPhysics,Gf
import isaaclab.sim as sim_utils
from isaaclab.sim.converters import UrdfConverter,UrdfConverterCfg
from isaaclab.assets import Articulation,ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaacsim.core.utils.extensions import enable_extension
enable_extension('isaacsim.sensors.camera')
from isaacsim.sensors.camera import Camera
from arm_sim.common import ROOT
from arm_sim.conveyor_scene import ConveyorCell,preview_layout
from arm_sim.factory import arm_base_height


def mobile_source(kind):
    folder=ROOT/'assets/factory/turtlebot3_description'
    root=ET.fromstring((folder/'urdf'/f'turtlebot3_{kind}.urdf').read_text().replace('${namespace}',''))
    for node in list(root):
        if node.tag.startswith('{http://ros.org/wiki/xacro}'):root.remove(node)
    for mesh in root.iter('mesh'):
        mesh.set('filename',str(folder/mesh.get('filename').split('package://turtlebot3_description/')[1]))
    for joint in root.findall('joint'):joint.set('type','fixed')
    output=ROOT/'assets/generated/conveyor_demo'/f'{kind}_parked.urdf'
    ET.ElementTree(root).write(output,encoding='utf-8',xml_declaration=True)
    return output


def orient(camera,eye,target,up=(0,0,1)):
    eye=np.array(eye,dtype=float);back=eye-np.array(target);back/=np.linalg.norm(back)
    right=np.cross(up,back);right/=np.linalg.norm(right);vertical=np.cross(back,right)
    q=Rotation.from_matrix(np.column_stack((right,vertical,back))).as_quat()
    camera.set_world_pose(eye,q[[3,0,1,2]],camera_axes='usd')


cameras=[]
sim=None;stream=None;monitor=None
try:
    import pickle,time,signal,hashlib
    plan_bytes=(ROOT/'outputs/sim_real_demo/plan.pkl').read_bytes();plan=pickle.loads(plan_bytes)
    cfg=plan.cfg;out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps({'status':'rendering'}))
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/120,device='cpu'))
    sim_utils.DomeLightCfg(intensity=1100).func('/World/Light',sim_utils.DomeLightCfg(intensity=1100))
    paint_cfg=copy.deepcopy(cfg)
    # Dock approach bars follow current lane centres, never legacy arm positions.
    dock_centres=[cfg['loading_dock'][0],cfg['unload_dock'][0]]
    paint_cfg['traffic']['yield_points']=[[x,.05] for x in dock_centres]
    forklift_dock_bars=[cfg['shipping_vehicle_pose'][:2],*[plan.rack_dock(i) for i in range(len(cfg['warehouses']))]]
    for i,rack in enumerate(cfg['warehouses']):
        route=next(r for r in paint_cfg['traffic']['routes'] if r['name']==f'Warehouse{i}')
        route['points'][-1]=[rack['center'][0],rack['center'][1]-rack['size'][1]/2-.025]
    cell=ConveyorCell(sim.stage,paint_cfg)
    forklift_route=next(r for r in paint_cfg['traffic']['routes'] if r['name']=='ForkliftU')
    forklift_paint=cell.material('route_ForkliftU',tuple(forklift_route['color']))
    for i,(x,y) in enumerate(forklift_dock_bars):
        cell.line(f'/World/Traffic/ForkliftDockBar{i}',(x-.13,y),(x+.13,y),forklift_paint,.020)
    cell.box('/World/Floor',(0,0,-.006),(*cfg['map']['size'],.01),cell.material('floor',tuple(cfg['map']['floor_color'])),False)
    # Paint only: stock Waffle footprint with ~10mm clearance, excluding the forks.
    px,py,heading=cfg['forklift_parking_pose']
    corners=[(-.207,-.163),(.084,-.163),(.084,.163),(-.207,.163)]
    corners=[(px+math.cos(heading)*x-math.sin(heading)*y,py+math.sin(heading)*x+math.cos(heading)*y) for x,y in corners]
    for i,(a,b) in enumerate(zip(corners,corners[1:]+corners[:1])):
        cell.line(f'/World/Traffic/ForkliftParking/Edge{i}',a,b,cell.materials['teal'],.008)
    # Leave a clear gap around parking outlines, including their interiors.
    paint_exclusions=[(np.min(corners,axis=0)-.020,np.max(corners,axis=0)+.020)]
    for bay in cfg['traffic'].get('waiting_bays',[]):
        center=np.array(bay['center']);half=np.array(bay['size'])/2+.020
        paint_exclusions.append((center-half,center+half))
    for x,y in paint_cfg['traffic']['yield_points']+forklift_dock_bars:
        paint_exclusions.append((np.array([x-.15,y-.03]),np.array([x+.15,y+.03])))
    # Rebuild every centre dash on one grid; shared stretches have one mixed stroke.
    for prim in list(sim.stage.Traverse()):
        if str(prim.GetPath()).startswith('/World/Traffic/') and prim.GetName().startswith('Dash'):
            sim.stage.RemovePrim(prim.GetPath())
    from scripts.real_markings import centre_dashes
    mixed=tuple((a+b)/2 for a,b in zip((.91,.47,.10),forklift_route['color']))
    paints={'material':cell.materials['orange'],'forklift':forklift_paint,'shared':cell.material('shared_route',mixed)}
    for i,(a,b,family) in enumerate(centre_dashes(paint_cfg,paint_cfg['traffic']['yield_points']+forklift_dock_bars,paint_exclusions)):
        cell.line(f'/World/Traffic/CentreDashes/Dash{i}',a,b,paints[family],.006)
    models=json.loads((ROOT/'assets/generated/conveyor_demo/models.json').read_text());robots={}
    charcoal=cell.material('robot_charcoal',(.045,.055,.065));blue=cell.material('finger_blue',(.025,.26,.67))
    for name,pose in cfg['arms'].items():
        model=models[pose['model']];kind=pose['model'];yaw=pose['yaw']
        converter=UrdfConverter(UrdfConverterCfg(asset_path=str(ROOT/model['urdf']),usd_dir=str(ROOT/'assets/generated/conveyor_demo'/kind),usd_file_name=kind+'.usd',
            fix_base=True,merge_fixed_joints=False,self_collision=False,force_usd_conversion=True,make_instanceable=False,collision_from_visuals=False,collider_type='convex_hull',
            joint_drive=UrdfConverterCfg.JointDriveCfg(drive_type='force',target_type='position',gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=150.,damping=4.))))
        home=cfg['rest_joints'][name]
        robots[name]=Articulation(ArticulationCfg(prim_path='/World/'+name,spawn=sim_utils.UsdFileCfg(usd_path=converter.usd_path),
            init_state=ArticulationCfg.InitialStateCfg(pos=(*pose['position'],arm_base_height(pose,cfg['table_height'])),rot=(math.cos(yaw/2),0,0,math.sin(yaw/2)),joint_pos=dict(zip(model['joints'],home))),
            actuators={'hold':ImplicitActuatorCfg(joint_names_expr=['.*'],stiffness=150.,damping=4.,effort_limit_sim=10.,velocity_limit_sim=1.)}))
        x,y=pose['position'];p='/World/Clamps/'+name
        # Keep the robot's URDF base, omit added plates beneath B/C.
        from arm_sim.conveyor_hardware import clamp_boxes,mount_plate
        plate=mount_plate(cfg,name)
        if plate:cell.box(p+'/Plate',*plate,cell.materials['metal'])
        for leaf,pos,size in clamp_boxes(cfg,name):
            color=cell.materials['metal'] if 'Rail' in leaf else cell.materials['orange']
            cell.box(p+'/'+leaf,pos,size,color)
    for kind,poses in [('burger',cfg['burger']['poses']),('waffle',[cfg['waffle']['pose']])]:
        converter=UrdfConverter(UrdfConverterCfg(asset_path=str(mobile_source(kind)),usd_dir=str(ROOT/'assets/generated/conveyor_demo'/kind),usd_file_name=kind+'.usd',
            fix_base=True,merge_fixed_joints=True,make_instanceable=False,collision_from_visuals=False,joint_drive=None))
        for i,(x,y,yaw) in enumerate(poses):
            path=f'/World/{kind}_{i+1}';spawn=sim_utils.UsdFileCfg(usd_path=converter.usd_path)
            spawn.func(path,spawn,translation=(x,y,.001),orientation=(math.cos(yaw/2),0,0,math.sin(yaw/2)))
            if kind=='burger':cell.attach_burger_deck(path)
            else:cell.attach_forklift(path)
    camera_converter=UrdfConverter(UrdfConverterCfg(asset_path=str(ROOT/'assets/factory/realsense/d435_body.urdf'),
        usd_dir=str(ROOT/'assets/generated/conveyor_demo/d435'),usd_file_name='d435.usd',
        fix_base=True,merge_fixed_joints=True,make_instanceable=False,joint_drive=None))
    for prim in list(sim.stage.Traverse()):
        path=str(prim.GetPath())
        if prim.IsA(UsdGeom.Mesh) and '/visuals/' in path and path.startswith(('/World/arm_','/World/burger_','/World/waffle_')):
            mat=blue if 'xlerobot' in path and 'finger' in path else charcoal
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat,bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        if path.startswith(tuple('/World/'+n+'/' for n,v in cfg['arms'].items() if v['model']=='so101_xlerobot')) and prim.GetName() in ('wrist_camera_link','wrist_camera_mount_link'):
            for sub in prim.GetChildren():
                if sub.GetName()=='visuals':UsdGeom.Imageable(sub).MakeInvisible()
        if path.startswith(tuple('/World/'+n+'/' for n,v in cfg['arms'].items() if v['model']=='so101_xlerobot')) and prim.GetName()=='gripper_link':
            from arm_sim.conveyor_hardware import d435_assembly,d435_adapter_meshes,d435_optical_pose
            mount_tf,camera_tf=d435_assembly()
            p=path+'/D435'
            spawn=sim_utils.UsdFileCfg(usd_path=camera_converter.usd_path)
            q=Rotation.from_matrix(camera_tf[:3,:3]).as_quat()
            spawn.func(p,spawn,translation=tuple(camera_tf[:3,3]),orientation=tuple(q[[3,0,1,2]]))
            for child in list(sim.stage.Traverse()):
                if not str(child.GetPath()).startswith(p+'/') and str(child.GetPath())!=p:continue
                if child.IsA(UsdPhysics.Joint):child.SetActive(False)
                for api in (UsdPhysics.ArticulationRootAPI,UsdPhysics.RigidBodyAPI,UsdPhysics.CollisionAPI):
                    if child.HasAPI(api):child.RemoveAPI(api)
            mount=ROOT/'vendor/so_arm100/Optional/Wrist_Cam_Mount_RealSense_D435/stl/Wrist_cam_mount_RealSense_D435.stl'
            shape=cell.asset_mesh(path+'/D435Mount',mount,(0,0,0),charcoal)
            shape.ClearXformOpOrder();shape.AddTransformOp().Set(Gf.Matrix4d(mount_tf.T.tolist()))
            for leaf,vertices in d435_adapter_meshes():
                mesh=UsdGeom.Mesh.Define(sim.stage,path+'/D435Adapter/'+leaf)
                mesh.CreatePointsAttr(vertices.tolist());mesh.CreateFaceVertexCountsAttr([3]*(len(vertices)//3))
                mesh.CreateFaceVertexIndicesAttr(list(range(len(vertices))));mesh.CreateSubdivisionSchemeAttr('none')
                cell.bind(mesh.GetPrim(),cell.materials['metal'])

    for name,pose in cfg['arms'].items():
        if pose['model']=='omx':
            UsdShade.MaterialBindingAPI.Apply(sim.stage.GetPrimAtPath('/World/'+name)).Bind(charcoal,bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    # Independent world roots: one unique mesh per physical part/container.
    remove_parts=[str(prim.GetPath()) for prim in sim.stage.Traverse() if prim.GetName()=='Part' and str(prim.GetPath()).startswith('/World/Source/')]
    for p in remove_parts:sim.stage.RemovePrim(p)
    for name in ('burger_1','burger_2'):sim.stage.RemovePrim('/World/'+name+'/Cargo/Parts')
    for i in range(2):sim.stage.RemovePrim(f'/World/Warehouse{i}/EmptyContainer')
    part_ops={};tray_ops={}
    for name in plan.parts:
        shape=cell.product_asset('/World/DemoParts/'+name,name.split('_')[0],(0,0,0));shape.ClearXformOpOrder();part_ops[name]=shape.AddTransformOp()
    for i in range(2):
        root=f'/World/DemoTrays/Tray{i}';xf=UsdGeom.Xform.Define(sim.stage,root);tray_ops[i]=xf.AddTransformOp();cell.shipping_container(root,0,0,0)
    mobile_ops={}
    for name in plan.vehicles:
        xf=UsdGeom.Xformable(sim.stage.GetPrimAtPath('/World/'+name));xf.ClearXformOpOrder();mobile_ops[name]=xf.AddTransformOp()
    lift_xf=UsdGeom.Xformable(sim.stage.GetPrimAtPath('/World/waffle_1/Lift'));lift_xf.ClearXformOpOrder();lift_op=lift_xf.AddTranslateOp()
    for prim in list(sim.stage.Traverse()):
        path=str(prim.GetPath())
        if path.startswith(('/World/burger_','/World/waffle_')):
            if prim.IsA(UsdPhysics.Joint):prim.SetActive(False)
            for api in (UsdPhysics.ArticulationRootAPI,UsdPhysics.RigidBodyAPI):
                if prim.HasAPI(api):prim.RemoveAPI(api)
        if prim.HasAPI(UsdPhysics.CollisionAPI):UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    overhead_eye,overhead_direction=cell.build_assembly_camera()
    cams={}
    for key,eye,target,up,res in [('top',(0,0,4),(0,0,0),(0,1,0),(1200,800)),('perspective',(3,-3.8,3.4),(0,.05,.1),(0,0,1),(1200,800)),('top_b',overhead_eye,overhead_eye+overhead_direction,(0,0,1),(640,480))]:
        cam=Camera('/World/Views/'+key,resolution=res);cam.initialize();orient(cam,eye,target,up);cam.set_clipping_range(.003,10);cams[key]=cam;cameras.append((key,cam))
        if key=='top':
            sensor=UsdGeom.Camera(cam.prim);sensor.CreateProjectionAttr(UsdGeom.Tokens.orthographic);sensor.CreateHorizontalApertureAttr(31.5);sensor.CreateVerticalApertureAttr(21.)
        else:UsdGeom.Camera(cam.prim).CreateFocalLengthAttr(24. if key=='perspective' else 14.)
    if args.verify:
        for key,eye,target in [('assembly_detail',(1.72,-.18,1.12),(.91,.58,.25)),('warehouse_detail',(-.61,.29,.40),(-.20,.79,.065))]:
            cam=Camera('/World/Views/'+key,resolution=(960,640));cam.initialize();orient(cam,eye,target);cam.set_clipping_range(.003,10.);UsdGeom.Camera(cam.prim).CreateFocalLengthAttr(24.);cams[key]=cam;cameras.append((key,cam))
    for arm in cfg['arms']:
        key=('depth_' if plan.ik[arm].so else 'wrist_')+arm[-1]
        cam=Camera('/World/Views/'+key,resolution=(480,360));cam.initialize();cam.set_clipping_range(.003,10.);cams[key]=cam;cameras.append((key,cam))
        UsdGeom.Camera(cam.prim).CreateFocalLengthAttr(12.)
        if plan.ik[arm].so:cam.add_distance_to_image_plane_to_frame()
    sim.reset()
    ids={n:r.find_joints(plan.ik[n].model['joints'],preserve_order=True)[0] for n,r in robots.items()}
    from arm_sim.demo_coordination import WorkcellGate
    from arm_sim.traffic import TrafficGate
    from arm_sim.demo_geometry import ArmGeometry
    from arm_sim.conveyor_hardware import d435_optical_pose
    from arm_sim.camera_stream import CameraStream
    from arm_sim.real_monitor import RealMonitor
    from scripts.real_workflow import Workflow
    workflow=Workflow(plan)
    gate=WorkcellGate(plan);traffic_gate=TrafficGate(plan);geometry={n:ArmGeometry(k) for n,k in plan.ik.items()}
    if not args.verify:
        overview_options={key:{'quality':92,'subsampling':0} for key in ('top','perspective')}
        stream=CameraStream([*cams,'top_d','depth_a_depth','depth_c_depth'],port=args.port+1,demo_control=True,frame_options=overview_options);monitor=RealMonitor(stream,args.port)
    stopped=False
    def stop(*_):
        global stopped
        stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    paused=not args.verify;started=args.verify;done=False;t=0.;speed=1.;last_wall=time.monotonic();last_render=0.;elapsed=0.;steps=0;capture_index=0;next_capture=0.;max_joint_error=0.;max_tool_error=0.;max_scene_error=0.;fault=None;speed_wall=time.monotonic();speed_time=0.;effective_speed=0.
    report={'plan_sha256':hashlib.sha256(plan_bytes).hexdigest(),'status':'running','runtime':'assisted kinematic Isaac Sim','physical_grasp_verified':False}
    # Capture actual departure/turn/docking poses as well as periodic assembly views.
    captures=[];vehicle_capture_times=[]
    if args.verify:
        selected=[e for n in plan.vehicles for e in [next(e for e in plan.vehicle_moves if e['vehicle']==n and e['mode']=='turn')]]
        selected += [e for e in plan.vehicle_moves if e['vehicle']=='waffle_1' and e['mode']=='turn' and plan.vehicles['waffle_1'].at(e['start'])[0]>1.]
        vehicle_capture_times=sorted({float(t) for e in selected for t in (e['start'],(e['start']+e['end'])/2,e['end'])})
    from PIL import Image
    print(f'SIM REAL READY http://127.0.0.1:{args.port}/ — press Start',flush=True)
    while app.is_running() and not stopped and (not args.steps or steps<args.steps):
        now=time.monotonic();dt=min(.1,now-last_wall);last_wall=now
        if stream:
            for command in stream.pop_commands():
                if command=='resume':paused=False;started=True
                elif command=='pause':paused=True
                elif command=='restart':t=0.;paused=True;started=False;done=False;elapsed=0.;gate.reset();traffic_gate.reset();fault=None
                elif isinstance(command,dict) and command['command']=='speed':speed=command['speed']
        if steps and not paused and not done and not fault:
            measured={n:r.data.joint_pos[0,ids[n]].cpu().numpy().copy() for n,r in robots.items()}
            end=plan.advance(t,1/30 if args.verify else dt*speed,measured)
            vehicle_feedback={}
            for n,op in mobile_ops.items():
                tf=np.array(op.Get()).T
                vehicle_feedback[n]=np.array([tf[0,3],tf[1,3],math.atan2(tf[1,0],tf[0,0])])
            end=gate.advance(t,end,measured);end=traffic_gate.advance(t,end,vehicle_feedback)
            if end==t and (gate.hold_reason or traffic_gate.hold_reason):raise RuntimeError(gate.hold_reason or traffic_gate.hold_reason)
            t=end;elapsed+=dt
        state=plan.sample(t)
        for n,r in robots.items():
            q=r.data.default_joint_pos.clone();q[0,ids[n]]=torch.tensor(state['joints'][n],dtype=q.dtype);r.write_joint_state_to_sim(q,torch.zeros_like(q));r.set_joint_position_target(q);r.write_data_to_sim()
        def matrix(pos,rotation):
            tf=np.eye(4);tf[:3,:3]=rotation;tf[:3,3]=pos;return Gf.Matrix4d(tf.T.tolist())
        for n,op in mobile_ops.items():
            p=state['vehicles'][n];op.Set(matrix([*p[:2],0.],Rotation.from_euler('z',p[2]).as_matrix()))
        lift_op.Set(Gf.Vec3d(0,0,state['lift']))
        for n,op in part_ops.items():op.Set(matrix(state['parts'][n],state['part_rotations'][n]))
        for i,op in tray_ops.items():
            p=plan.tray_pose(i,t);op.Set(matrix(p[:3],Rotation.from_euler('z',p[3]).as_matrix()))
        for arm in cfg['arms']:
            q=state['joints'][arm];k=plan.ik[arm];tf=k.transform(q)
            if k.so:
                mount=geometry[arm].transforms(q)['gripper_link'];eye,direction,up=d435_optical_pose(mount);orient(cams['depth_'+arm[-1]],eye,eye+direction,up)
            else:
                eye=tf[:3,3]+tf[:3,:3]@[.01,0,.09];target=tf[:3,3]+tf[:3,:3]@[.01,0,0];orient(cams['wrist_'+arm[-1]],eye,target,tf[:3,0])
        sim.forward()
        for n,r in robots.items():
            r.update(sim.get_physics_dt());error=float(np.max(abs(r.data.joint_pos[0,ids[n]].cpu().numpy()-state['joints'][n])));max_joint_error=max(max_joint_error,error)
            if error>.001:raise RuntimeError(f'Joint readback mismatch {n}: {error}')
            link='gripper_frame_link' if plan.ik[n].so else 'end_effector_link';bi=r.find_bodies(link)[0][0]
            error=float(np.linalg.norm(r.data.body_pos_w[0,bi].cpu().numpy()-plan.ik[n].position(state['joints'][n])));max_tool_error=max(max_tool_error,error)
            if error>.001:raise RuntimeError(f'Tool frame readback mismatch {n}: {error}')
        for n,op in part_ops.items():
            error=float(np.linalg.norm(np.array(op.Get()).T[:3,3]-state['parts'][n]));max_scene_error=max(max_scene_error,error)
            if error>1e-6:raise RuntimeError(f'Payload transform mismatch {n}')
        for n,op in mobile_ops.items():
            error=float(np.linalg.norm(np.array(op.Get()).T[:2,3]-state['vehicles'][n][:2]));max_scene_error=max(max_scene_error,error)
            if error>1e-6:raise RuntimeError(f'Vehicle transform mismatch {n}')
            rotation=np.array(UsdGeom.Xformable(sim.stage.GetPrimAtPath('/World/'+n)).ComputeLocalToWorldTransform(0)).T[:3,:3]
            if np.max(abs(rotation-Rotation.from_euler('z',state['vehicles'][n][2]).as_matrix()))>1e-6:raise RuntimeError(f'Vehicle heading mismatch {n}')
        steps+=1
        keyframe_due=bool(vehicle_capture_times and t>=vehicle_capture_times[0])
        render_due=(t>=next_capture or keyframe_due or state['done'] or steps<5) if args.verify else now-last_render>=1/15
        if render_due:
            sim.render();last_render=now
            if args.verify:
                for _ in range(3):sim.render()
                for key in ('top','perspective','assembly_detail','warehouse_detail'):
                    rgba=np.asarray(cams[key].get_rgba(),dtype=np.uint8)
                    if rgba.size:Image.fromarray(rgba).save(out/f'{key}_{capture_index:03d}.png')
                captures.append({'index':capture_index,'time_s':t})
                while vehicle_capture_times and vehicle_capture_times[0]<=t:vehicle_capture_times.pop(0)
                capture_index+=1;next_capture=t+30.
            else:
                for key,cam in cams.items():
                    stream.update(key,cam.get_rgba())
                    if key.startswith('depth_'):
                        depth=cam.get_depth()
                        if depth is not None and depth.size:
                            v=np.where(np.isfinite(depth)&(depth>0),255*(1-np.clip(depth,0,1)),0).astype(np.uint8);stream.update(key+'_depth',np.repeat(v[...,None],3,axis=2))
                stream.update('top_d',cams['top_b'].get_rgba())
                if now-speed_wall>=1.:
                    effective_speed=max(0.,t-speed_time)/(now-speed_wall);speed_wall=now;speed_time=t
                stream.set_metadata(scene='sim_real',mode='demo',camera_fps=15,hardware_control=False,demo={
                    **{k:state[k] for k in ('label','progress','assembled','stored','goal','cargo','time','coordination','traffic')},
                    'paused':paused,'started':started,'done':done,'speed':speed,'effective_speed':effective_speed,'start_mode':'preloaded','start_modes':['preloaded'],'cycles':int(done),'assisted':True,'active_elapsed_s':elapsed,'nominal_duration_s':plan.duration,'completed_tasks':sorted(gate.completed),'hold_reason':gate.hold_reason or traffic_gate.hold_reason,'geometry_enabled':False,'fault':fault,
                    'workflow':workflow.snapshot(t,state,started=started,paused=paused,done=done,arm_acks=gate.completed,traffic_acks=traffic_gate.completed,fault=fault)})
        if state['done']:
            done=True;paused=True
            if args.verify:break
        if not args.verify:time.sleep(.002)
    if args.verify and not done:raise RuntimeError('Finite verification stopped before full process completion')
    report.update(status='complete' if done else 'stopped',completed=done,steps=steps,process_time_s=t,max_joint_error_rad=max_joint_error,max_tool_error_m=max_tool_error,max_scene_error_m=max_scene_error,task_acknowledgements=gate.acknowledgements,traffic_acknowledgements=len(traffic_gate.completed),captures=captures,**plan.validate())
    (out/'runtime_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n');sim.stage.GetRootLayer().Export(str(out/'scene.usda'))
    print('SIM REAL COMPLETE '+json.dumps(report,ensure_ascii=False),flush=True)
    import os
    if os.environ.get('REAL_STATUS_FILE'):Path(os.environ['REAL_STATUS_FILE']).write_text('ok')
except BaseException as exc:
    import traceback
    traceback.print_exc()
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'runtime_report.json').write_text(json.dumps({'status':'failed','error':str(exc),'completed':False},ensure_ascii=False))
    raise
finally:
    if monitor:monitor.close()
    if stream:stream.close()
    if sim:
        sim._disable_app_control_on_stop_handle=True
        for _,cam in cameras:cam.pause()
        for _,cam in cameras:
            for annotator in list(cam._custom_annotators):cam.detach_annotator(annotator)
        for _,cam in cameras:cam.destroy()
        sim.stop()
    app.close(wait_for_replicator=False)
