#!/usr/bin/env python3
"""Finite Isaac Sim render of the four-arm candidate layout; no web server."""
import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=Path('outputs/conveyor_preview'))
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
sim=None
try:
    cfg=preview_layout();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps({'status':'rendering'}))
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/120,device='cpu'))
    sim_utils.DomeLightCfg(intensity=1100).func('/World/Light',sim_utils.DomeLightCfg(intensity=1100))
    cell=ConveyorCell(sim.stage,cfg)
    cell.box('/World/Floor',(0,0,-.006),(*cfg['map']['size'],.01),cell.material('floor',tuple(cfg['map']['floor_color'])),False)
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
    overhead_eye,overhead_direction=cell.build_assembly_camera()
    camera=Camera('/World/Views/assembly_overhead',resolution=(1280,960));camera.initialize()
    orient(camera,overhead_eye,overhead_eye+overhead_direction)
    camera.set_clipping_range(.005,10.)
    UsdGeom.Camera(camera.prim).CreateFocalLengthAttr(14.)
    cameras.append(('assembly_overhead',camera))
    view_shift=np.array([cfg['shipping_dock'][0]-1.14,cfg['shipping_dock'][1]-.250,0.])
    for name,eye,target,up,res in [('top',(0,0,4.0),(0,0,0),(0,1,0),(1600,1067)),
                                   ('perspective',(3.0,-3.8,3.4),(0,.05,.10),(0,0,1),(1600,1000)),
                                   ('assembly_detail',(1.9,-.5,1.4),(1.02,.62,.43),(0,0,1),(1600,1200)),
                                   ('forklift_detail',(1.85,-.65,.62),(1.14,.22,.12),(0,0,1),(1400,1000)),
                                   ('corner_b',(.46,.26,.56),(.71,.51,.27),(0,0,1),(1200,1000)),
                                   ('corner_c',(1.61,.27,.56),(1.29,.51,.27),(0,0,1),(1200,1000))]:
        if name not in ('top','perspective'):
            eye=np.array(eye)+view_shift;target=np.array(target)+view_shift
        if name in ('corner_b','corner_c'):
            key='arm_b' if name=='corner_b' else 'arm_d'
            bx,by=cfg['arms'][key]['position'];side=-1 if key=='arm_b' else 1
            target=(bx,by,cfg['table_height']+.045);eye=(bx+side*.22,by-.26,cfg['table_height']+.22)
        camera=Camera('/World/Views/'+name,resolution=res);camera.initialize();orient(camera,eye,target,up)
        camera.set_clipping_range(.01,10.)
        if name=='top':
            UsdGeom.Camera(camera.prim).CreateProjectionAttr(UsdGeom.Tokens.orthographic)
            aperture=10*max(cfg['map']['size'][0]+.2,(cfg['map']['size'][1]+.2)*res[0]/res[1])
            UsdGeom.Camera(camera.prim).CreateHorizontalApertureAttr(aperture)
            UsdGeom.Camera(camera.prim).CreateVerticalApertureAttr(aperture*res[1]/res[0])
        else:UsdGeom.Camera(camera.prim).CreateFocalLengthAttr(24.)
        cameras.append((name,camera))
    from arm_sim.demo_kinematics import ArmKinematics
    for arm in cfg['arms']:
        k=ArmKinematics(models[cfg['arms'][arm]['model']],cfg['arms'][arm],.23)
        target=k.position(cfg['rest_joints'][arm])+np.array([0.,0.,.035]);eye=target+np.array([.22,-.27,.16])
        cam=Camera('/World/Views/gripper_'+arm,resolution=(1200,900));cam.initialize();orient(cam,eye,target)
        cam.set_clipping_range(.003,10.);UsdGeom.Camera(cam.prim).CreateFocalLengthAttr(35.)
        cameras.append(('gripper_'+arm,cam))
        if cfg['arms'][arm]['model']!='so101_xlerobot':continue
        if arm=='arm_a':
            from arm_sim.demo_geometry import ArmGeometry
            tf=ArmGeometry(k).transforms(cfg['rest_joints'][arm])['gripper_link']
            eye=tf[:3,3]+tf[:3,:3]@np.array([-.22,.10,-.14])
            target=tf[:3,3]+tf[:3,:3]@np.array([-.025,.02,-.025])
            cam=Camera('/World/Views/d435_fastening',resolution=(1200,1000));cam.initialize()
            orient(cam,eye,target,tf[:3,1]);cam.set_clipping_range(.003,10.)
            UsdGeom.Camera(cam.prim).CreateFocalLengthAttr(45.)
            cameras.append(('d435_fastening',cam))
            # A normal view of the attachment face exposes spacing without
            # interpreting perspective tilt as a change in mounting angle.
            centre=np.array([0.,.02,-.03]);eye=tf[:3,3]+tf[:3,:3]@(centre+[-.30,0,0])
            target=tf[:3,3]+tf[:3,:3]@centre
            cam=Camera('/World/Views/d435_spacing_side',resolution=(1400,1000));cam.initialize()
            orient(cam,eye,target,tf[:3,1]);cam.set_clipping_range(.003,10.)
            sensor=UsdGeom.Camera(cam.prim);sensor.CreateProjectionAttr(UsdGeom.Tokens.orthographic)
            sensor.CreateHorizontalApertureAttr(1.6);sensor.CreateVerticalApertureAttr(1.6*1000/1400)
            cameras.append(('d435_spacing_side',cam))


        from arm_sim.demo_geometry import ArmGeometry
        tf=ArmGeometry(k).transforms(cfg['rest_joints'][arm])['gripper_link']
        eye,forward,up=d435_optical_pose(tf)
        cam=Camera('/World/Views/d435_'+arm,resolution=(1280,720));cam.initialize()
        orient(cam,eye,eye+forward,up);cam.set_clipping_range(.003,10.)
        sensor=UsdGeom.Camera(cam.prim)
        sensor.CreateHorizontalApertureAttr(20.955);sensor.CreateVerticalApertureAttr(20.955*720/1280)
        sensor.CreateFocalLengthAttr(20.955/(2*math.tan(math.radians(43))))
        cameras.append(('d435_'+arm,cam))

    # Isolated grasp specimens are outside the map. These show the exact
    # revised mesh/profile pair, not assembled products or process progress.
    from arm_sim.demo_geometry import ArmGeometry,origin
    profiles=json.loads((ROOT/'config/conveyor_grasps.json').read_text())['profiles']
    root=ET.parse(models['so101_xlerobot']['urdf']).getroot()
    k=ArmKinematics(models['so101_xlerobot'],{'position':[0,0],'yaw':0},0)
    geometry=ArmGeometry(k)
    for i,(kind,profile) in enumerate(profiles.items()):
        q=np.array([0.,0.,0.,0.,0.,profile['closed']]);frames=geometry.transforms(q)
        specimen=np.eye(4);specimen[:3,:3]=np.diag([1.,-1.,-1.]);specimen[:3,3]=[4.,i*.3,.5]
        tool=k.transform(q)
        for link in ('gripper_link','moving_jaw_so101_v1_link'):
            for j,visual in enumerate(root.find(f"link[@name='{link}']").findall('visual')):
                mesh=visual.find('geometry/mesh');file=Path(mesh.get('filename'))
                scale=float(mesh.get('scale','1 1 1').split()[0])
                tf=specimen@np.linalg.inv(tool)@frames[link]@origin(visual.find('origin'))
                shape=cell.asset_mesh(f'/World/GraspReview/{kind}/{link}_{j}',file,(0,0,0),blue if 'finger' in file.name else charcoal,scale=scale)
                shape.ClearXformOpOrder();shape.AddTransformOp().Set(Gf.Matrix4d(tf.T.tolist()))
        pos=specimen[:3,3]+specimen[:3,:3]@profile['part_base_in_tool']
        cell.product_asset('/World/GraspReview/'+kind+'/Part',kind,pos)
        center=specimen[:3,3]+[0,0,.025]
        camera=Camera('/World/Views/grasp_'+kind,resolution=(1200,1000));camera.initialize()
        orient(camera,center+np.array([.20,-.23,.14]),center);camera.set_clipping_range(.003,10.)
        UsdGeom.Camera(camera.prim).CreateFocalLengthAttr(40.);cameras.append(('grasp_'+kind,camera))
    sim.reset()
    for robot in robots.values():
        robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(),torch.zeros_like(robot.data.default_joint_pos));robot.reset()
    for _ in range(40):sim.forward();sim.render()
    from PIL import Image
    for name,camera in cameras:
        rgba=np.asarray(camera.get_rgba(),dtype=np.uint8)
        if not rgba.size or not rgba[...,:3].any():raise RuntimeError('Empty capture: '+name)
        Image.fromarray(rgba).convert('RGB').save(out/(name+'.png'))
    material_inventory={'worktable':sum(1 for prim in sim.stage.Traverse() if str(prim.GetPath()).startswith('/World/Source/') and prim.GetName()=='Part')}
    for number in (1,2):
        stock=sim.stage.GetPrimAtPath(f'/World/burger_{number}/Cargo/Parts')
        material_inventory[f'burger_{number}']=len(stock.GetChildren()) if stock.IsValid() else 0
    if material_inventory!={'worktable':9,'burger_1':9,'burger_2':0}:
        raise RuntimeError('Incorrect production material inventory: '+str(material_inventory))
    if sim.stage.GetPrimAtPath('/World/Conveyor').IsValid():raise RuntimeError('Conveyor geometry must be absent')
    for name in ('arm_b','arm_d'):
        if sim.stage.GetPrimAtPath('/World/Clamps/'+name+'/Plate').IsValid():raise RuntimeError('Unexpected OMX supplementary plate')
    empty_containers=sum(sim.stage.GetPrimAtPath(f'/World/Warehouse{i}/EmptyContainer').IsValid() for i in range(2))
    if empty_containers!=2 or sim.stage.GetPrimAtPath('/World/ShippingDock/Container').IsValid():
        raise RuntimeError('Both empty containers must start in warehouses')
    sim.stage.GetRootLayer().Export(str(out/'scene.usda'))
    import hashlib
    audit_path=ROOT/'outputs/conveyor_preview_latest/hardware_check.json'
    audit=json.loads(audit_path.read_text()) if audit_path.exists() else {'layout_sha256':'','hardware_source_sha256':'','status':'missing','failures':['mechanical audit missing'],'scope':'unverified'}
    same_hardware=audit.get('hardware_source_sha256')==hashlib.sha256((ROOT/'arm_sim/conveyor_hardware.py').read_bytes()).hexdigest()
    same_layout=audit['layout_sha256']==hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    (out/'report.json').write_text(json.dumps({'status':'rendered_four_arm_layout_preview_no_conveyor','source_config':cfg,
        'motion_verified':False,'physical_grasp_verified':False,
        'arm_count':len(robots),'conveyor_present':False,'material_inventory':material_inventory,'initial_empty_containers_in_warehouses':empty_containers,
        'preview_geometry_verified':same_hardware and same_layout and audit['status']=='pass','real_assembly_verified':False,'hardware_failures':audit['failures'],
        'hardware_scope':audit['scope'],'prototype_d435_adapter_mm':0,'holder_fastener_count':5,'prototype_camera_washer_mm':.5,
        'camera_mounts':'one independent assembly overhead camera; shaft sections aligned to CAD mating centres; PCB behind lens plate; custom table clamp; D435 directly on left opening-motor holder with five user-confirmed drilled fastening points; approximate hole coordinates; nominal depth-imager views; actual adapter fit and optical calibration pending',
        'parts_source':'new.zip, latest two_side_cradle ball jig; unchanged original 30 mm ball',
        'images':[name+'.png' for name,_ in cameras]},indent=2,ensure_ascii=False)+'\n')
    import os
    if os.environ.get('PREVIEW_STATUS_FILE'):Path(os.environ['PREVIEW_STATUS_FILE']).write_text('ok')
    print('PREVIEW COMPLETE '+str(out),flush=True)
except BaseException:
    import traceback,sys
    details=traceback.format_exc();print(details,file=sys.stderr,flush=True)
    if 'out' in globals():(out/'report.json').write_text(json.dumps({'status':'failed','error':details}))
    raise
finally:
    if sim is not None:sim._disable_app_control_on_stop_handle=True
    for _,camera in cameras:camera.pause()
    for _,camera in cameras:
        for annotator in list(camera._custom_annotators):camera.detach_annotator(annotator)
    for _,camera in cameras:camera.destroy()
    if sim is not None:sim.stop()
    app.close(wait_for_replicator=False)
