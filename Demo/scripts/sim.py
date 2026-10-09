#!/usr/bin/env python3
"""Two URDF articulations, independent ROS 2 commands, optional wrist rendering."""
import argparse
import json
import math
import time
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--models', choices=['both','so101','omx'], default='both')
parser.add_argument('--demo', action='store_true', help='Local generated input instead of ROS command input')
parser.add_argument('--steps', type=int, default=0, help='0: run until closed')
parser.add_argument('--live-cameras', action='store_true', help='Three live camera views at http://127.0.0.1:8766')
parser.add_argument('--capture', action='store_true', help='Save overview and SO-101 wrist images')
parser.add_argument('--rebuild', action='store_true')
parser.add_argument('--scene', choices=['robots','assembly'], default='robots')
parser.add_argument('--mode', choices=['manual','replay'], default='manual')
parser.add_argument('--record', type=Path, help='New JSONL episode; manual mode only, never overwrites')
parser.add_argument('--episode', type=Path, help='Completed episode to replay in simulation')
parser.add_argument('--allow-synthetic-replay', action='store_true', help='Testing only')
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")
args = parser.parse_args()
if args.mode=='replay' and (not args.episode or args.record or args.demo):
    parser.error('Replay requires --episode; --record and --demo are manual-only')
if args.mode=='manual' and args.episode:
    parser.error('--episode requires --mode replay')
if (args.record or args.mode=='replay') and (args.scene!='assembly' or args.models!='both'):
    parser.error('Record/replay requires --scene assembly --models both')
if args.record and args.record.exists():
    parser.error('Recording already exists; choose a new name')
# The minimal headless experience omits OmniGraph, required by the ROS bridge.
# Use the rendering experience in every mode; render products remain opt-in.
args.enable_cameras = True
launcher = AppLauncher(args)
app = launcher.app

import numpy as np
import torch
import omni.graph.core as og
from pxr import UsdGeom, Gf
import isaaclab.sim as sim_utils
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaacsim.core.utils.extensions import enable_extension
from arm_sim.common import ROOT, load_models, CommandGate, demo_positions


def graph_for(kind, prim):
    path = f'/World/ROS_{kind}'
    keys = og.Controller.Keys
    og.Controller.edit({'graph_path':path,'evaluator_name':'execution'}, {
        keys.CREATE_NODES:[('Tick','omni.graph.action.OnPlaybackTick'),
                           ('Sub','isaacsim.ros2.bridge.ROS2SubscribeJointState'),
                           ('Hold','isaacsim.ros2.bridge.ROS2Subscriber'),
                           ('Pub','isaacsim.ros2.bridge.ROS2PublishJointState'),
                           ('Time','isaacsim.core.nodes.IsaacReadSimulationTime')],
        keys.CONNECT:[('Tick.outputs:tick','Hold.inputs:execIn'),('Tick.outputs:tick','Sub.inputs:execIn'),('Tick.outputs:tick','Pub.inputs:execIn'),
                      ('Time.outputs:simulationTime','Pub.inputs:timeStamp')],
        keys.SET_VALUES:[('Hold.inputs:messagePackage','std_msgs'),('Hold.inputs:messageSubfolder','msg'),
                         ('Hold.inputs:messageName','Float64'),('Hold.inputs:topicName',f'/{kind}/input_hold'),('Hold.inputs:queueSize',1),
                         ('Sub.inputs:topicName',f'/{kind}/joint_command'),('Sub.inputs:queueSize',1),
                         ('Pub.inputs:topicName',f'/{kind}/joint_states'),('Pub.inputs:targetPrim',[prim])],
    })
    app.update()  # Generic subscriber creates its typed output dynamically.
    attrs={name:og.Controller.attribute(f'{path}/Sub.outputs:{name}') for name in ['timeStamp','jointNames','positionCommand']}
    attrs['hold']=og.Controller.attribute(f'{path}/Hold.outputs:data')
    return attrs


recorder = None
stream = None
stop_requested = False

def request_stop(*_):
    global stop_requested
    stop_requested = True


def main():
    global recorder, stream
    import signal
    signal.signal(signal.SIGINT,request_stop)
    models = load_models()
    cell = None
    replay = None
    if args.scene=='assembly':
        from arm_sim.workcell import Workcell, config, signature
        from arm_sim.episodes import Recorder, Replay
        layout=config()
        for kind,pose in layout['robots'].items():
            models[kind]['position']=pose['position']
            models[kind]['rotation']=[math.cos(pose['yaw']/2),0.,0.,math.sin(pose['yaw']/2)]
    selected = list(models) if args.models=='both' else [args.models]
    if not enable_extension('isaacsim.ros2.bridge'):
        raise RuntimeError('Cannot enable Isaac Sim ROS 2 bridge')
    app.update()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/120,render_interval=6 if args.live_cameras else 2,device=args.device))
    eye = (1.25,-1.65,1.70) if args.scene=='assembly' else (1.5,-1.8,1.3)
    target = (0.,0.,.64) if args.scene=='assembly' else (0.,0.,.2)
    sim.set_camera_view(eye=eye, target=target)
    sim_utils.GroundPlaneCfg().func('/World/Ground',sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=1800).func('/World/Light',sim_utils.DomeLightCfg(intensity=1800))
    if args.scene=='assembly':
        cell=Workcell(sim.stage)
        if args.mode=='replay':
            replay=Replay(args.episode,models,signature(models),sim.get_physics_dt(),cell.initial,args.allow_synthetic_replay)
        if args.record:
            recorder=Recorder(args.record,models,signature(models),sim.get_physics_dt(),cell.initial,args.demo)
        print(f'WORKCELL: 3 trays x 4 parts; mode={args.mode}; source={"synthetic_test" if args.demo else "leader_ros"}',flush=True)
    robots, gates, graphs, ids = {}, {}, {}, {}
    for kind in selected:
        model = models[kind]
        converter = UrdfConverter(UrdfConverterCfg(
            asset_path=str(ROOT/model['urdf']),usd_dir=str(ROOT/'assets/generated'/kind),usd_file_name=f'{kind}.usd',
            fix_base=True,merge_fixed_joints=False,self_collision=False,make_instanceable=False,
            force_usd_conversion=args.rebuild,collision_from_visuals=False,collider_type='convex_hull',
            joint_drive=UrdfConverterCfg.JointDriveCfg(drive_type='force',target_type='position',
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=150.,damping=4.))))
        cfg = ArticulationCfg(prim_path=f'/World/{kind}',
            spawn=sim_utils.UsdFileCfg(usd_path=converter.usd_path,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False,
                    solver_position_iteration_count=16,solver_velocity_iteration_count=4)),
            init_state=ArticulationCfg.InitialStateCfg(pos=tuple(model['position']),rot=tuple(model.get('rotation',[1.,0.,0.,0.])),joint_pos=dict(zip(model['joints'],model['home']))),
            actuators=({'arm':ImplicitActuatorCfg(joint_names_expr=['joint[1-5]'],stiffness=150.,damping=4.,
                                                effort_limit_sim=10.,velocity_limit_sim=4.),
                        'gripper':ImplicitActuatorCfg(joint_names_expr=['gripper_joint_[12]'],**model['gripper_drive'])}
                       if kind=='omx' else
                       {'arm':ImplicitActuatorCfg(joint_names_expr=['.*'],stiffness=150.,damping=4.,
                                                effort_limit_sim=10.,velocity_limit_sim=4.)}))
        robots[kind] = Articulation(cfg)
        gates[kind] = CommandGate(model)
        # Pub uses discovered articulation root; initialized after reset below.
    if 'omx' in selected:
        # A single hull bridges the finger recesses and wedges small parts outward.
        from pxr import UsdPhysics,PhysxSchema
        for name in ('link6','link7'):
            sim.stage.GetPrimAtPath('/World/omx/'+name+'/collisions').SetInstanceable(False)
        for prim in sim.stage.Traverse():
            if any(str(prim.GetPath()).startswith('/World/omx/'+name+'/collisions/') for name in ('link6','link7')) and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexDecomposition')
                decomp=PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
                decomp.CreateMaxConvexHullsAttr(32)
                decomp.CreateHullVertexLimitAttr(64)
                decomp.CreateVoxelResolutionAttr(200000)
    contact_view = None
    contact_attr = None
    if 'omx' in selected:
        # Isaac Lab defaults this off; Core RigidPrim does not enable CPU contact reports.
        sim.carb_settings.set_bool('/physics/disableContactProcessing',False)
        from isaacsim.core.prims import RigidPrim
        finger_paths=[str(p.GetPath()) for p in sim.stage.Traverse()
                      if str(p.GetPath()).startswith('/World/omx/') and p.GetName() in ('link6','link7')]
        if len(finger_paths)!=2:raise RuntimeError(f'Expected two OMX fingers: {finger_paths}')
        contact_view=RigidPrim(prim_paths_expr=finger_paths,name='omx_finger_contacts',
                              reset_xform_properties=False,track_contact_forces=True,
                              contact_filter_prim_paths_expr=[[],[]],disable_stablization=False)
        keys=og.Controller.Keys
        og.Controller.edit({'graph_path':'/World/OMXContact','evaluator_name':'execution'}, {
            keys.CREATE_NODES:[('Tick','omni.graph.action.OnPlaybackTick'),('Pub','isaacsim.ros2.bridge.ROS2Publisher')],
            keys.CONNECT:[('Tick.outputs:tick','Pub.inputs:execIn')],
            keys.SET_VALUES:[('Pub.inputs:messagePackage','std_msgs'),('Pub.inputs:messageSubfolder','msg'),
                             ('Pub.inputs:messageName','Float64MultiArray'),('Pub.inputs:topicName','/omx/gripper_contact'),
                             ('Pub.inputs:queueSize',1)]})
        app.update()
        contact_attr=og.Controller.attribute('/World/OMXContact/Pub.inputs:data')
        og.Controller.set(contact_attr,[0.,0.,0.,0.])
    sim.reset()
    if contact_view:contact_view.initialize()
    if cell:cell.initialize()
    for kind, robot in robots.items():
        robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(),robot.data.default_joint_vel.clone())
        robot.reset()
        robot.update(sim.get_physics_dt())
        ids[kind] = [robot.joint_names.index(n) for n in models[kind]['joints']]
        graphs[kind] = graph_for(kind, robot.root_physx_view.prim_paths[0])
        print(f'READY {kind}: {robot.joint_names}',flush=True)
    cameras = {}
    if args.capture or args.live_cameras:
        if not enable_extension('isaacsim.sensors.camera'):
            raise RuntimeError('Camera extension unavailable')
        from isaacsim.sensors.camera import Camera
        from isaacsim.core.utils.rotations import euler_angles_to_quat
        overview = Camera('/World/Overview',resolution=(960,720))
        overview.initialize()
        if cell:UsdGeom.Camera(overview.prim).CreateFocalLengthAttr(28.)
        view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0.,0.,1.)).GetInverse().ExtractRotationQuat()
        overview.set_world_pose(np.array(eye),np.array([view.GetReal(),*view.GetImaginary()]),camera_axes='usd')
        cameras['overview'] = overview
        if 'so101' in selected:
            links = [p for p in sim.stage.Traverse() if p.GetPath().pathString.startswith('/World/so101/') and p.GetName()=='wrist_camera_link']
            if len(links)!=1:
                raise RuntimeError(f'Expected one wrist_camera_link, got {len(links)}')
            wrist = Camera(str(links[0].GetPath())+'/Camera',resolution=(640,480))
            wrist.initialize()
            # Optical approximation: physical camera intrinsics/extrinsics require measurement.
            wrist.set_local_pose(np.array([0.,0.,0.028]),np.array([0.,2**-0.5,2**-0.5,0.]),camera_axes='usd')
            wrist.set_clipping_range(0.002,10.)
            UsdGeom.Camera(wrist.prim).CreateFocalLengthAttr(13.5)
            cameras['wrist'] = wrist
    if args.live_cameras:
        # Reuse the existing wrist view; replace the capture overview with a vertical view.
        overview.set_resolution((640,480))
        top_eye=(0.,0.,2.25)
        top_target=(0.,0.,.64)
        q=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*top_eye),Gf.Vec3d(*top_target),Gf.Vec3d(0.,1.,0.)).GetInverse().ExtractRotationQuat()
        overview.set_world_pose(np.array(top_eye),np.array([q.GetReal(),*q.GetImaginary()]),camera_axes='usd')
        UsdGeom.Camera(overview.prim).CreateFocalLengthAttr(22.)
        cameras['top']=cameras.pop('overview')
        if 'wrist' in cameras:cameras['so101']=cameras.pop('wrist')
        if 'omx' in selected:
            wrist=Camera('/World/omx/link5/Camera',resolution=(640,480))
            wrist.initialize()
            pos=(.045,0.,.09)
            q=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*pos),Gf.Vec3d(.105,-.0016,0.),Gf.Vec3d(0.,0.,1.)).GetInverse().ExtractRotationQuat()
            wrist.set_local_pose(np.array(pos),np.array([q.GetReal(),*q.GetImaginary()]),camera_axes='usd')
            wrist.set_clipping_range(.002,10.)
            UsdGeom.Camera(wrist.prim).CreateFocalLengthAttr(13.5)
            cameras['omx']=wrist
        if cell:
            metal=cell.materials['metal']
            cell.box('/World/CameraMount/Post',(.68,.43,1.15),(.025,.025,2.3),metal,False)
            cell.box('/World/CameraMount/ArmY',(.68,.215,2.3),(.025,.455,.025),metal,False)
            cell.box('/World/CameraMount/ArmX',(.34,0.,2.3),(.705,.025,.025),metal,False)
            cell.box('/World/CameraMount/Head',(0.,0.,2.28),(.065,.045,.035),metal,False)
            if 'omx' in selected:
                cell.box('/World/omx/link5/CameraBracket',(.025,0.,.078),(.02,.026,.025),metal,False)
        from arm_sim.camera_stream import CameraStream
        stream=CameraStream(cameras)
        print('CAMERAS http://127.0.0.1:8766/',flush=True)
    last_camera_update=0.
    input_held={k:False for k in selected}
    last_stamp = {k:None for k in selected}
    rejected = {k:0 for k in selected}
    accepted = {k:0 for k in selected}
    max_error = {k:0. for k in selected}
    motion_min = {k:np.full(len(models[k]['joints']),np.inf) for k in selected}
    motion_max = {k:np.full(len(models[k]['joints']),-np.inf) for k in selected}
    last_stale = {k:True for k in selected}
    replay_error = {k:0. for k in selected}
    replay_object_error = 0.
    count = 0
    output = ROOT/'outputs'
    output.mkdir(exist_ok=True)
    while app.is_running() and not stop_requested and (not args.steps or count<args.steps):
        now = time.monotonic()
        frame_commands,frame_measured,frame_tracking = {},{},{}
        for kind,robot in robots.items():
            gate = gates[kind]
            hold_stamp=og.Controller.get(graphs[kind]['hold'])
            hold_active=bool(not args.demo and replay is None and hold_stamp and -0.2<=time.time()-hold_stamp<=0.5)
            if replay is not None:
                pass  # ROS input is ignored throughout replay.
            elif hold_active:
                pass
            elif args.demo:
                gate.receive(models[kind]['joints'],demo_positions(models[kind],count/120),now)
                accepted[kind] += 1
            else:
                stamp = og.Controller.get(graphs[kind]['timeStamp'])
                if stamp and stamp != last_stamp[kind]:
                    last_stamp[kind] = stamp
                    try:
                        names = list(og.Controller.get(graphs[kind]['jointNames']))
                        values = list(og.Controller.get(graphs[kind]['positionCommand']))
                        # Input process uses wall time even when simulator is paused.
                        age = time.time()-stamp
                        if not -0.2 <= age <= 0.5:
                            raise ValueError(f'Expired or future command timestamp (age={age:.3f}s)')
                        if kind=='omx' and len(values)==7:
                            named = dict(zip(names,values))
                            if abs(named.get('gripper_joint_1',0)+named.get('gripper_joint_2',0))>1e-6:
                                raise ValueError('OMX fingers must have opposite angles')
                        gate.receive(names,values,now)
                        accepted[kind] += 1
                    except (ValueError,TypeError) as exc:
                        rejected[kind] += 1
                        if rejected[kind] <= 5:
                            print(f'REJECT {kind}: {exc}',flush=True)
            measured = robot.data.joint_pos[0,ids[kind]].cpu().numpy()
            if not np.isfinite(measured).all():
                raise RuntimeError(f'Non-finite physics state: {kind}')
            if hold_active and not input_held[kind]:
                gate.hold(measured)
                print(f'INPUT HOLD {kind}: sensor fault',flush=True)
            elif input_held[kind] and not hold_active:
                print(f'INPUT RELEASE {kind}: normal input restored or watchdog active',flush=True)
            input_held[kind]=hold_active
            if replay is not None:
                command = replay.frame(count)['commands'][kind]
            else:
                command = gate.step(now,sim.get_physics_dt(),measured)
            frame_commands[kind]=list(command)
            frame_measured[kind]=measured.tolist()
            if replay is not None and count<len(replay.frames):
                replay_error[kind]=max(replay_error[kind],float(np.max(np.abs(measured-np.array(replay.frame(count)['measured'][kind])))))
            frame_tracking[kind]=not gate.stale
            if gate.stale != last_stale[kind]:
                print(f'STATE {kind}: {"HOLD (input timeout)" if gate.stale else "TRACKING"}',flush=True)
                last_stale[kind] = gate.stale
            robot.set_joint_position_target(torch.tensor([command],device=sim.device),joint_ids=ids[kind])
            robot.write_data_to_sim()
            motion_min[kind] = np.minimum(motion_min[kind],measured)
            motion_max[kind] = np.maximum(motion_max[kind],measured)
            if count>120:
                max_error[kind] = max(max_error[kind],float(np.max(np.abs(measured-command))))
        if replay is not None and count<len(replay.frames):
            poses=cell.snapshot()
            replay_object_error=max(replay_object_error,max(math.dist(pose[:3],replay.frame(count)['objects'][path][:3]) for path,pose in poses.items()))
        if recorder is not None:
            recorder.append(frame_commands,frame_measured,frame_tracking,cell.snapshot())
        if contact_view:
            forces=contact_view.get_net_contact_forces(dt=sim.get_physics_dt())
            magnitude=float(np.linalg.norm(np.asarray(forces),axis=-1).sum())
            angle=float(robots['omx'].data.joint_pos[0,ids['omx'][5]])
            enabled=float(not args.demo and replay is None and not gates['omx'].stale and not input_held['omx'])
            og.Controller.set(contact_attr,[time.time(),max(0.,min(1.1,angle)),magnitude,enabled])
        sim.step()
        for robot in robots.values():
            robot.update(sim.get_physics_dt())
        count += 1
        if stream and time.monotonic()-last_camera_update>=.1:
            last_camera_update=time.monotonic()
            for name,camera in cameras.items():stream.update(name,camera.get_rgba())
        if not args.demo:
            time.sleep(max(0.,sim.get_physics_dt()-(time.monotonic()-now)))
        if args.capture and count==240:
            from PIL import Image
            for _ in range(20):
                sim.render()
            for name,camera in cameras.items():
                print('CAMERA_POSE',name,camera.get_world_pose(camera_axes='usd'),flush=True)
                rgba = camera.get_rgba()
                if rgba is None or np.asarray(rgba).size==0:
                    raise RuntimeError(f'No camera image: {name}')
                Image.fromarray(np.asarray(rgba,dtype=np.uint8)).save(output/f'{"assembly_" if cell else ""}{name}.png')
            print('CAPTURED overview/wrist',flush=True)
        if replay is not None and count==len(replay.frames):
            print('REPLAY COMPLETE: final target held; no hardware commands published',flush=True)
            if args.headless and not args.steps:break
    report = {k:dict(accepted=accepted[k],rejected=rejected[k],stale=gates[k].stale,
               max_tracking_error_rad=max_error[k],motion_range_rad=(motion_max[k]-motion_min[k]).tolist(),
               joints=models[k]['joints']) for k in selected}
    (output/('demo_report.json' if args.demo else 'ros_report.json')).write_text(json.dumps(report,indent=2)+'\n')
    if cell:
        poses=cell.snapshot()
        if any(not math.isfinite(v) for pose in poses.values() for v in pose):
            raise RuntimeError('Non-finite product state')
        cell_report={'parts':len(poses),'trays':3,'capacity_per_tray':4,'mode':args.mode,'frames':count,'initial_objects':cell.initial,'final_objects':poses}
        if replay:
            reference=replay.frame(min(count-1,len(replay.frames)-1))
            cell_report['replay_max_joint_error_rad']=replay_error
            cell_report['replay_max_object_position_error_m']=replay_object_error
        (output/'assembly_report.json').write_text(json.dumps(cell_report,indent=2)+'\n')
    if recorder is not None:recorder.close(complete=True)
    sim.stage.GetRootLayer().Export(str(output/('assembly_scene.usda' if cell else 'dual_scene.usda')))
    print('RESULT '+json.dumps(report),flush=True)
    # This installed Isaac Lab keeps rendering on STOP unless teardown disables its handler.
    sim._disable_app_control_on_stop_handle = True
    for camera in cameras.values():
        camera.pause()
    for camera in cameras.values():
        camera.destroy()
    sim.stop()
    print('CLEANUP cameras stopped',flush=True)

try:
    main()
finally:
    if stream is not None:stream.close()
    if recorder is not None:recorder.close(complete=False)
    context = sim_utils.SimulationContext.instance()
    if context is not None:
        context._disable_app_control_on_stop_handle = True
    app.close(wait_for_replicator=False)
