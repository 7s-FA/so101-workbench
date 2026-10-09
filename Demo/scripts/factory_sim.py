#!/usr/bin/env python3
"""Hardware-free four-arm factory preview; no serial devices or ROS commands."""
import argparse
import json
import math
import signal
import os
import time
from pathlib import Path
import xml.etree.ElementTree as ET
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--demo',action='store_true',help='Assisted full factory process, no hardware')
parser.add_argument('--demo-start',choices=['load','preloaded'],default='preloaded',help='Start with A loading, or with nine parts already on Burger 1')
parser.add_argument('--demo-autostart',action='store_true',help='Skip the Start button for unattended verification')
parser.add_argument('--demo-speed',type=float,default=1.,help='Wall-clock choreography playback speed, 0.25 to 16; default 1')
parser.add_argument('--runtime-geometry',action=argparse.BooleanOptionalAction,default=True,
                    help='Check swept geometry every demo frame; demo.sh defaults to disabled')
parser.add_argument('--demo-cycles',type=int,default=1,help='0 repeats indefinitely; otherwise hold after N cycles')
parser.add_argument('--exit-on-complete',action='store_true',help='Exit after all requested demo cycles; useful for verification')
parser.add_argument('--layout',type=Path)
parser.add_argument('--steps',type=int,default=0,help='0 runs until the viewer is closed')
parser.add_argument('--capture',action='store_true',help='Save perspective, top, wrist RGB and metric depth')
parser.add_argument('--camera-fps',type=float,default=15.,help='Maximum live camera/render frame rate in wall-clock seconds')
parser.add_argument('--capture-interval',type=float,default=.6,help='Seconds of process time between verification images')
parser.add_argument('--demo-render-interval',type=int,default=1,help='Minimum kinematic frames per render, also capped by camera-fps')
parser.add_argument('--verify-from',type=float,default=0.,help='Verification only: start at this process time; report partial coverage')
parser.add_argument('--view',choices=['top','perspective'],default='top')
parser.add_argument('--verify-views',action='store_true',help='Exercise all GUI camera selectors during a finite test run')
parser.add_argument('--no-monitor',action='store_true',help='Disable the integrated web monitor and streaming')
parser.add_argument('--output',type=Path,default=Path('outputs/factory'))
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device='cpu')
args=parser.parse_args(); args.enable_cameras=True
if not .25<=args.demo_speed<=16 or args.demo_cycles<0:parser.error('Invalid demo speed/cycles')
if not math.isfinite(args.capture_interval) or args.capture_interval<=0 or args.demo_render_interval<1:parser.error('Invalid capture/render interval')
if not math.isfinite(args.camera_fps) or not 1<=args.camera_fps<=60:parser.error('Camera FPS must be between 1 and 60')
if not math.isfinite(args.verify_from) or args.verify_from<0:parser.error('Invalid verification start time')
if args.verify_from and not (args.demo and args.headless and args.no_monitor and args.exit_on_complete and args.demo_cycles==1):parser.error('--verify-from requires one finite headless verification cycle without the monitor')
if args.demo and args.layout:parser.error('--demo currently uses the verified default map; --layout is preview-only')
if args.demo and args.no_monitor and args.headless and not args.demo_autostart:
    parser.error('--no-monitor --headless requires --demo-autostart because no Start button is available')
if args.exit_on_complete and (not args.demo or args.demo_cycles==0):parser.error('--exit-on-complete requires a finite --demo')
args.output=(Path('outputs/demo') if args.demo and args.output==Path('outputs/factory') else args.output).resolve()
args.output.mkdir(parents=True,exist_ok=True)
run_id=str(time.time_ns())
(args.output/'report.json').write_text(json.dumps({'run_id':run_id,'status':'starting','demo':{'completed':False,'completed_cycles':0}}))
launcher=AppLauncher(args); app=launcher.app

import numpy as np
import torch
from threadpoolctl import threadpool_limits
# SimulationApp replaces OPENBLAS_NUM_THREADS during startup. Tiny per-frame
# FK/geometry and tensor operations run faster without a large BLAS thread pool.
os.environ['OPENBLAS_NUM_THREADS']='1'
_blas_thread_limit=threadpool_limits(limits=1,user_api='blas')
torch.set_num_threads(1)
from pxr import UsdGeom, UsdShade, Gf, Sdf
import isaaclab.sim as sim_utils
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaacsim.core.utils.extensions import enable_extension
if not enable_extension('isaacsim.sensors.camera'):
    raise RuntimeError('Isaac Sim camera extension could not be enabled')
from isaacsim.sensors.camera import Camera
from arm_sim.common import ROOT, load_models
from arm_sim.factory import FactoryCell, load_layout

stop=False
stream=None
monitor=None
def request_stop(*_):
    global stop
    stop=True
signal.signal(signal.SIGINT,request_stop)
signal.signal(signal.SIGTERM,request_stop)


def mobile_urdf(kind):
    source=ROOT/'assets/factory/turtlebot3_description/urdf'/f'turtlebot3_{kind}.urdf'
    root=ET.fromstring(source.read_text().replace('${namespace}',''))
    for node in list(root):
        if node.tag.startswith('{http://ros.org/wiki/xacro}'):
            root.remove(node)
    for mesh in root.iter('mesh'):
        rel=mesh.get('filename').split('package://turtlebot3_description/')[1]
        mesh.set('filename',str(ROOT/'assets/factory/turtlebot3_description'/rel))
    # Parked preview, not a wheel controller. Source URDFs are kept untouched.
    for joint in root.findall('joint'):
        joint.set('type','fixed')
    if kind=='waffle':
        camera=root.find("link[@name='camera_link']")
        for child in list(camera):camera.remove(child)
    dest=ROOT/'assets/generated/factory'/f'{kind}_parked.urdf'
    dest.parent.mkdir(parents=True,exist_ok=True)
    ET.ElementTree(root).write(dest,encoding='utf-8',xml_declaration=True)
    return dest


def look_at(camera,eye,target,up=(0,0,1),local=False):
    q=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float,eye)),Gf.Vec3d(*map(float,target)),Gf.Vec3d(*map(float,up))).GetInverse().ExtractRotationQuat()
    quat=np.array([q.GetReal(),*q.GetImaginary()])
    if local:camera.set_local_pose(np.array(eye),quat,camera_axes='usd')
    else:camera.set_world_pose(np.array(eye),quat,camera_axes='usd')


def capture(cameras,output,report):
    from PIL import Image
    metrics={}
    for name,camera in cameras.items():
        rgba=np.asarray(camera.get_rgba())
        if rgba.size==0 or rgba.ndim!=3:raise RuntimeError(f'Empty camera {name}')
        Image.fromarray(rgba.astype('uint8')).save(output/(name+'.png'))
        if name.startswith('depth_'):
            depth=np.asarray(camera.get_depth())
            valid=np.isfinite(depth)&(depth>0)
            if not valid.any():raise RuntimeError(f'No valid metric depth from {name}')
            np.save(output/(name+'_metres.npy'),depth)
            # Debug grayscale is separate from the raw float depth used by learning.
            gray=np.zeros(depth.shape,dtype=np.uint8)
            gray[valid]=(255*(1-np.clip(depth[valid],0,1.0))).astype(np.uint8)
            Image.fromarray(gray).save(output/(name+'_visual.png'))
            pos,rot=camera.get_world_pose(camera_axes='usd')
            metrics[name]={'valid_fraction':float(valid.mean()),'range_m':[float(depth[valid].min()),float(depth[valid].max())],
                           'resolution':list(camera.get_resolution()),'world_position':pos.tolist(),'world_quaternion':rot.tolist(),
                           'intrinsics':np.asarray(camera.get_intrinsics_matrix()).tolist(),
                           'depth_type':'distance_to_image_plane_metres','calibration':'provisional simulation optics, not measured hardware'}
    report['cameras']=metrics
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


def main():
    global stream,monitor
    cfg=load_layout(args.layout)
    plan=None
    if args.demo:
        from arm_sim.factory_demo import demo_layout,DemoPlan
        cfg=demo_layout(cfg)
        print('DEMO: checking all IK waypoints before playback',flush=True)
        plan=DemoPlan(cfg,start_mode=args.demo_start)
        if args.verify_from>=plan.duration:raise ValueError('Verification start is beyond the process end')
        from arm_sim.demo_validation import route_check
        route_report=route_check(plan)
        demo_plans={args.demo_start:(plan,route_report)}
        if not args.no_monitor:
            # Build both paths before opening the monitor so a button switch
            # does not freeze the live stream while IK is being solved.
            other='load' if args.demo_start=='preloaded' else 'preloaded'
            other_plan=DemoPlan(cfg,start_mode=other)
            demo_plans[other]=(other_plan,route_check(other_plan))
        print('DEMO PLAN '+json.dumps({**plan.validate(),'routes':route_report}),flush=True)
    output=(Path('outputs/demo') if args.demo and args.output==Path('outputs/factory') else args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/120,render_interval=6,device='cpu'))
    sim_utils.GroundPlaneCfg(color=(.77,.80,.82)).func('/World/Ground',sim_utils.GroundPlaneCfg(color=(.77,.80,.82)))
    sim_utils.DomeLightCfg(intensity=650).func('/World/Light',sim_utils.DomeLightCfg(intensity=650))
    sim.set_camera_view(eye=(2.4,-3.,2.6),target=(.05,-.12,.2))
    UsdGeom.Imageable(sim.stage.GetPrimAtPath('/World/Ground')).MakeInvisible()
    cell=FactoryCell(sim.stage,cfg)
    cell.box('/World/Factory/Floor',(0,0,-.006),(*cfg['map']['size'],.01),cell.material('floor',tuple(cfg['map']['floor_color'])),False)
    base_models=load_models();robots={};targets={}
    for name,pose in cfg['arms'].items():
        kind=pose['model'];model=base_models[kind];yaw=pose['yaw']
        source=ROOT/model['urdf']
        converter=UrdfConverter(UrdfConverterCfg(asset_path=str(source),usd_dir=str(ROOT/'assets/generated'/kind),usd_file_name=kind+'.usd',
              fix_base=True,merge_fixed_joints=False,self_collision=False,make_instanceable=False,
              collision_from_visuals=False,collider_type='convex_hull',
              joint_drive=UrdfConverterCfg.JointDriveCfg(drive_type='force',target_type='position',gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=150.,damping=4.))))
        actuators={'hold':ImplicitActuatorCfg(joint_names_expr=['.*'],stiffness=150.,damping=4.,effort_limit_sim=10.,velocity_limit_sim=1.)}
        from arm_sim.factory import arm_base_height
        robots[name]=Articulation(ArticulationCfg(prim_path='/World/'+name,
            spawn=sim_utils.UsdFileCfg(usd_path=converter.usd_path,articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(*pose['position'],arm_base_height(pose,cfg['table_height'])),rot=(math.cos(yaw/2),0,0,math.sin(yaw/2)),joint_pos=dict(zip(model['joints'],model['home']))),actuators=actuators))
        x,y=pose['position']
        cell.label('/World/Labels/'+name,pose['label'],(x,y+.095 if name=='arm_c' else y-.085),.30,.035,z=cfg['table_height']+.009)
        if pose.get('mount_height_status')=='provisional_simulation_only':
            cell.label('/World/Labels/'+name+'_riser',f"SIM RISER {pose['mount_height']*1000:.0f} mm",(x,y+.13),.24,.026,z=cfg['table_height']+.01)
        if not cfg.get('clamps'):
            support_y=y-.0215 if kind=='so101' else y
            support_size=(.12,.11) if kind=='so101' else (.14,.17)
            cell.box('/World/Factory/Clamps/'+name+'/Plate',(x,support_y,cfg['table_height']+.002),(*support_size,.004),cell.materials['metal'])
            rise=pose.get('mount_height',0.)
            if rise:
                cell.box('/World/Factory/Clamps/'+name+'/Riser',(x,support_y,cfg['table_height']+.004+rise/2),(*support_size,rise),cell.materials['metal'])
    if cfg.get('clamps'):
        from arm_sim.demo_geometry import clamp_shapes
        for shapes in clamp_shapes(cfg).values():
            for name,_,_,_,low,high in shapes:
                material=cell.materials['orange'] if any(k in name for k in ('Jaw','Handle')) else cell.materials['metal']
                cell.box('/World/Factory/Clamps/'+name,(low+high)/2,high-low,material)
    for kind,poses in [('burger',cfg['burger']['poses']),('waffle',[cfg['waffle']['pose']])]:
        converter=UrdfConverter(UrdfConverterCfg(asset_path=str(mobile_urdf(kind)),usd_dir=str(ROOT/'assets/generated/factory'/kind),usd_file_name=kind+'.usd',fix_base=True,merge_fixed_joints=True,make_instanceable=False,collision_from_visuals=False,collider_type='convex_hull',joint_drive=None))
        for i,(x,y,yaw) in enumerate(poses):
            path=f'/World/{kind}_{i+1}'
            spawn=sim_utils.UsdFileCfg(usd_path=converter.usd_path)
            spawn.func(path,spawn,translation=(x,y,.001),orientation=(math.cos(yaw/2),0,0,math.sin(yaw/2)))
            # Added pieces use the vehicle root's local coordinates.
            if kind=='burger':cell.attach_burger_deck(path)
            else:cell.attach_forklift(path)
    # Keep source geometry; apply readable preview colours on this stage only.
    charcoal=cell.material('robot_charcoal',(.035,.045,.055))
    blue=cell.material('robot_finger_blue',(.025,.26,.65))
    for prim in sim.stage.Traverse():
        path=str(prim.GetPath())
        if prim.GetName()=='visuals' and path.startswith(('/World/arm_', '/World/burger_', '/World/waffle_')):
            mat=blue if path.startswith(('/World/arm_a/', '/World/arm_c/')) and 'gripper' in path else charcoal
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat,bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    cameras={}
    top=Camera('/World/Views/Top',resolution=(960,int(960*cfg['map']['size'][1]/cfg['map']['size'][0])));top.initialize()
    look_at(top,(0,0,3.0),(0,0,0),(0,1,0))
    UsdGeom.Camera(top.prim).CreateProjectionAttr(UsdGeom.Tokens.orthographic)
    UsdGeom.Camera(top.prim).CreateHorizontalApertureAttr(32.)
    UsdGeom.Camera(top.prim).CreateVerticalApertureAttr(32.*cfg['map']['size'][1]/cfg['map']['size'][0])
    top.set_clipping_range(.01,10.);cameras['top']=top
    iso=Camera('/World/Views/Perspective',resolution=(1024,640));iso.initialize()
    look_at(iso,(3.2,-3.8,3.5),(0,0,.10));UsdGeom.Camera(iso.prim).CreateFocalLengthAttr(22.)
    cameras['perspective']=iso
    for name in ('arm_a','arm_c'):
        links=[p for p in sim.stage.Traverse() if str(p.GetPath()).startswith('/World/'+name+'/') and p.GetName()=='wrist_camera_link']
        if len(links)!=1:raise RuntimeError(f'Camera mounting link missing: {name}')
        parent=str(links[0].GetPath());dcfg=cfg['depth_camera']
        # Existing optical link is a provisional reference; photograph extrinsics are unknown.
        for prim in sim.stage.Traverse():
            if str(prim.GetPath()).startswith(parent+'/visuals'):
                UsdGeom.Imageable(prim).MakeInvisible()
        cam=Camera(parent+'/DepthOptical',resolution=tuple(dcfg['resolution']));cam.initialize()
        cam.set_local_pose(np.array(dcfg['mount_xyz']),np.array([1.,0.,0.,0.]),camera_axes='usd')
        optical=str(cam.prim.GetPath())
        cell.box(optical+'/Housing',(0,0,.017),dcfg['body_size'],cell.materials['white'],False)
        for i,offset in enumerate((-.032,-.010,.016,.033)):
            cell.box(optical+f'/Lens{i}',(offset,0,.003),(.010,.014,.003),cell.materials['metal'],False)
        cam.set_clipping_range(.01,5.);UsdGeom.Camera(cam.prim).CreateFocalLengthAttr(dcfg['focal_length_mm'])
        cam.add_distance_to_image_plane_to_frame()
        cameras['depth_'+name[-1]]=cam
    for name in ('arm_b','arm_d'):
        links=[p for p in sim.stage.Traverse() if str(p.GetPath()).startswith('/World/'+name+'/') and p.GetName()=='link5']
        if len(links)!=1:raise RuntimeError(f'OMX wrist mounting link missing: {name}')
        cam=Camera(str(links[0].GetPath())+'/WristRGB',resolution=(320,240));cam.initialize()
        cam.set_clipping_range(.01,5.);UsdGeom.Camera(cam.prim).CreateFocalLengthAttr(12.)
        cameras['wrist_'+name[-1]]=cam
        overhead=Camera('/World/Views/Top_'+name[-1],resolution=(480,360));overhead.initialize();overhead.set_clipping_range(.01,5.)
        x,y=cfg['arms'][name]['position'];x+=.12;y+=.10
        look_at(overhead,(x,y,1.15),(x,y,.21),(0,1,0))
        UsdGeom.Camera(overhead.prim).CreateFocalLengthAttr(23.)
        cameras['top_'+name[-1]]=overhead
    if plan:
        from arm_sim.demo_runtime import prepare_demo_stage,DemoRuntime
        prepare_demo_stage(sim.stage,cell)
    sim.reset();cell.initialize()
    for name,robot in robots.items():
        robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(),robot.data.default_joint_vel.clone())
        robot.reset();robot.update(sim.get_physics_dt());targets[name]=robot.data.default_joint_pos.clone()
    runtime=DemoRuntime(sim.stage,cell,robots,plan,runtime_geometry=args.runtime_geometry) if plan else None
    demo_time=args.verify_from;demo_paused=not args.demo_autostart;demo_started=args.demo_autostart;demo_cycles=0;demo_done=False;demo_cycle_pending=False
    demo_reset_pending=True
    demo_fault=None
    def demo_command(command):
        nonlocal demo_time,demo_paused,demo_started,demo_cycles,demo_done,demo_cycle_pending,demo_reset_pending,demo_fault,plan,route_report,speed_wall,speed_time,effective_speed
        if isinstance(command,dict):
            if command['command']=='speed':
                args.demo_speed=command['speed'];return
            if command['command']=='start_mode':
                plan,route_report=demo_plans[command['mode']]
                runtime.select_plan(plan)
                args.demo_start=command['mode']
                command='restart'
        if command=='pause':demo_paused=True
        elif command=='resume' and not demo_fault:demo_paused=False;demo_started=True
        elif command=='restart':
            demo_time=0.;demo_cycles=0;demo_done=False;demo_paused=True;demo_started=False;demo_cycle_pending=False
            demo_reset_pending=True;demo_fault=None
            cycle_reports.clear();capture_stages.clear()
            speed_wall=time.monotonic();speed_time=0.;effective_speed=0.
            playback_clock.reset()
    window=None
    if not args.headless:
        from omni.kit.viewport.utility import get_active_viewport
        import omni.ui as ui
        def select(path):
            viewport=get_active_viewport()
            viewport.camera_path=path
            if str(viewport.camera_path)!=path:raise RuntimeError('Viewport camera selection failed')
        select('/World/Views/Top' if args.view=='top' else '/World/Views/Perspective')
        window=ui.Window('Factory Demo' if plan else 'Factory Preview',width=340,height=330 if plan else 220)
        with window.frame:
            with ui.VStack(spacing=5):
                ui.Label('4 arms / 2 cargo Burgers / 1 Waffle lift')
                ui.Label('18 parts / cargo 9 x 2 / output 6' if plan else '3 part types x 3 = 3 finished units')
                ui.Label('Assisted demo / no hardware I/O' if plan else 'Layout preview: robots parked; no hardware I/O')
                if plan:
                    ui.Button('Pause demo',clicked_fn=lambda:demo_command('pause'))
                    ui.Button('Start / Resume demo',clicked_fn=lambda:demo_command('resume'))
                    ui.Button('Restart demo',clicked_fn=lambda:demo_command('restart'))
                for label,key in [('Top view','top'),('Perspective','perspective'),('SO-101 A RGB-D','depth_a'),('SO-101 C RGB-D','depth_c')]:
                    ui.Button(label,clicked_fn=lambda k=key:select(cameras[k].prim_path))
                ui.Label('Depth: ideal simulation; mount is provisional')
    if not args.no_monitor:
        from arm_sim.camera_stream import CameraStream
        from arm_sim.monitor import ensure_monitor
        stream=CameraStream([*cameras,'depth_a_depth','depth_c_depth'],demo_control=bool(plan))
        monitor=ensure_monitor()
        print('통합 모니터 http://127.0.0.1:8767/',flush=True)
    report=cell.report_layout();report.update(run_id=run_id,status='running');report['source_config']=cfg;report['viewport_checks']=[]
    if plan:
        report['so101_tracking']={**plan.ik['arm_a'].model['relative_target'],
            'basis':'fresh simulated follower positions', 'loop_hz':1/sim.get_physics_dt(),
            'hardware_calibration_applied':False, 'scene_resets_exempt':True}
    captured=False;steps=0;start=time.monotonic()
    last_event='';last_motion=-1.;motion_index=0;demo_state=None;cycle_reports=[];capture_stages=set()
    from arm_sim.frame_clock import FrameClock,PlaybackClock
    camera_clock=FrameClock(args.camera_fps);step_clock=FrameClock(1/sim.get_physics_dt())
    playback_clock=PlaybackClock()
    speed_wall=time.monotonic();speed_time=demo_time;effective_speed=0.
    try:
        print('FACTORY READY: no serial devices; no physical robot commands',flush=True)
        while app.is_running() and not stop and (not args.steps or steps<args.steps):
            if runtime:step_clock.wait()
            camera_rendered=False
            if plan:
                if stream:
                    for command in stream.pop_commands():demo_command(command)
                if demo_cycle_pending and not demo_paused:
                    demo_time=0.;demo_cycle_pending=False
                    demo_reset_pending=True
                    playback_clock.reset()
                reset_frame=demo_reset_pending
                if reset_frame:last_motion=-1.
                running=steps>30 and not demo_paused and not demo_done and not reset_frame
                requested=playback_clock.request(demo_time,args.demo_speed,running)
                if running:
                    demo_time=runtime.advance(demo_time,requested)
                demo_state=plan.sample(demo_time)
                # Scene resets restore every object together, with no hardware I/O.
                try:
                    runtime.apply(demo_state,reset=reset_frame)
                except RuntimeError as error:
                    if args.exit_on_complete or args.no_monitor:raise
                    demo_fault=str(error);demo_paused=True
                    print('DEMO SAFETY HOLD '+demo_fault,flush=True)
                    if runtime.last is None:raise
                    demo_time=runtime.last['time'];demo_state=runtime.last
                demo_reset_pending=False
                if demo_state['label']!=last_event:
                    last_event=demo_state['label'];print('DEMO STEP '+last_event,flush=True)
            for name,robot in robots.items():
                if not plan:robot.set_joint_position_target(targets[name])
                robot.write_data_to_sim()
            if runtime:
                # Pure kinematic playback: do not integrate motor/rigid dynamics
                # after applying the approved assisted demonstration pose.
                sim.forward()
                if steps%args.demo_render_interval==0 and camera_clock.due():
                    sim.render();camera_rendered=True
            else:
                sim.step()
                camera_rendered=camera_clock.due()
            for robot in robots.values():robot.update(sim.get_physics_dt())
            steps+=1
            if runtime:
                # Read back the simulator state after every kinematic frame, including resets.
                runtime.verify()
                if demo_state['done'] and not demo_done and not demo_cycle_pending:
                    demo_cycles+=1
                    cycle_reports.append({'cycle':demo_cycles,'coverage_start_s':args.verify_from,'full_cycle':not bool(args.verify_from),**plan.validate(),
                        'max_part_error_m':runtime.max_part_error,'max_joint_error_rad':runtime.max_joint_error,'max_tool_error_m':runtime.max_tool_error,'max_tray_error_m':runtime.max_tray_error,
                        'max_relative_target_ratio':runtime.max_relative_ratio,'retimed_steps':runtime.retimed_steps,
                        'active_elapsed_s':playback_clock.elapsed_s,'requested_speed':args.demo_speed,
                        **runtime.geometry_status(),'safe_return_acknowledgements':runtime.gate.acknowledgements,'traffic_acknowledgements':len(runtime.traffic_gate.completed)})
                    print(('DEMO SEGMENT COMPLETE ' if args.verify_from else 'DEMO CYCLE COMPLETE ')+str(demo_cycles),flush=True)
                    demo_done=bool(args.demo_cycles and demo_cycles>=args.demo_cycles)
                    demo_cycle_pending=not demo_done
                    report.update(status=('segment_complete' if args.verify_from else 'complete') if demo_done else 'running',final_parts=cell.snapshot())
                    report['demo']={'routes':route_report,'completed':demo_done and not bool(args.verify_from),'completed_cycles':0 if args.verify_from else demo_cycles,'coverage_start_s':args.verify_from,
                        'cycles':cycle_reports,'stage':demo_state['label'],'assist_mode':'kinematic choreography; no physical success claim'}
                    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
                if args.capture and steps>120 and (demo_time-last_motion>=args.capture_interval or demo_state['done']):
                    from PIL import Image,ImageDraw
                    last_motion=demo_time;sim.render()
                    canvas=Image.new('RGB',(1280,760),(14,17,22))
                    for key,box in [('perspective',(0,40,640,440)),('top_b',(640,40,1280,440)),('depth_a',(0,440,320,680)),('wrist_b',(320,440,640,680)),('depth_c',(640,440,960,680)),('wrist_d',(960,440,1280,680))]:
                        frame=np.asarray(cameras[key].get_rgba(),dtype=np.uint8)
                        if frame.size:canvas.paste(Image.fromarray(frame).convert('RGB').resize((box[2]-box[0],box[3]-box[1])),box[:2])
                    draw=ImageDraw.Draw(canvas);draw.text((15,12),f'Cycle {max(1,demo_cycles) if demo_state["done"] else demo_cycles+1} / process {demo_time:.2f}s / assisted geometry demo',fill='white')
                    for x,label in [(15,'A / SO-101'),(335,'B / OMX'),(655,'C / SO-101'),(975,'D / OMX')]:draw.text((x,690),label,fill='white')
                    folder=output/'motion';folder.mkdir(exist_ok=True)
                    canvas.save(folder/f'{motion_index:05d}.jpg',quality=88);motion_index+=1
                if demo_done and args.exit_on_complete:break
            if args.verify_views and not args.headless and steps in (35,55,75,95,115):
                key=('top','perspective','depth_a','depth_c','top')[(steps-35)//20]
                select(cameras[key].prim_path);report['viewport_checks'].append(key)
            if steps==30:
                # Aim once, then retain a wrist-relative mounting transform.
                # These are provisional views, not measured hand-eye calibration.
                for name,robot in robots.items():
                    key=('depth_' if name in ('arm_a','arm_c') else 'wrist_')+name[-1]
                    # Set wrist-relative poses directly. USD world transforms may
                    # still contain the spawn pose while Fabric has moved the arm.
                    from arm_sim.demo_geometry import origin
                    model=base_models[cfg['arms'][name]['model']]
                    tree=ET.parse(ROOT/model['urdf']).getroot()
                    joints={j.get('name'):j for j in tree.findall('joint')}
                    if name in ('arm_a','arm_c'):
                        mount=origin(joints['wrist_camera_mount_joint'].find('origin'))@origin(joints['wrist_camera_joint'].find('origin'))
                        tool_in_parent=np.linalg.inv(mount)@origin(joints['gripper_frame_joint'].find('origin'))
                        eye_tool=(np.linalg.inv(tool_in_parent)@np.array([0.,0.,.05,1.])).tolist();target_tool=[0,0,.015,1.]
                    else:
                        tool_in_parent=origin(joints['end_effector_joint'].find('origin'))
                        eye_tool=[.01,0,.10,1.];target_tool=[0,0,0,1.]
                    eye=(tool_in_parent@eye_tool)[:3];point=(tool_in_parent@target_tool)[:3]
                    up=tool_in_parent[:3,:3]@np.array([0,1,0] if name in ('arm_a','arm_c') else [1,0,0])
                    look_at(cameras[key],eye,point,up,local=True)
            if stream and steps>30 and camera_rendered:
                for name,camera in cameras.items():
                    stream.update(name,camera.get_rgba())
                    if name.startswith('depth_'):
                        depth=camera.get_depth()
                        if depth is not None and np.asarray(depth).size:
                            depth=np.asarray(depth);valid=np.isfinite(depth)&(depth>0)
                            gray=np.zeros(depth.shape,dtype=np.uint8)
                            gray[valid]=(255*(1-np.clip(depth[valid],0,1.))).astype(np.uint8)
                            stream.update(name+'_depth',np.repeat(gray[...,None],3,axis=-1))
                now=time.monotonic()
                if now-speed_wall>=1.:
                    effective_speed=max(0.,demo_time-speed_time)/(now-speed_wall)
                    speed_wall=now;speed_time=demo_time
                stream.set_metadata(scene='factory',mode='demo' if plan else 'preview',camera_fps=args.camera_fps,sim_time_s=steps*sim.get_physics_dt(),part_count=len(cell.initial),hardware_control=False,
                    demo=({'label':demo_state['label'],'progress':demo_state['progress'],'assembled':demo_state['assembled'],
                           'stored':demo_state['stored'],'goal':demo_state['goal'],'paused':demo_paused,'started':demo_started,'done':demo_done,'cycles':demo_cycles,
                           'speed':args.demo_speed,'effective_speed':effective_speed,'start_mode':args.demo_start,'start_modes':list(demo_plans),'assisted':True,'cargo':demo_state['cargo'],
                           'time':demo_time,'active_elapsed_s':playback_clock.elapsed_s,'nominal_duration_s':plan.duration,'coordination':demo_state['coordination'],
                           'completed_tasks':sorted(runtime.gate.completed),'hold_reason':runtime.gate.hold_reason or runtime.traffic_gate.hold_reason,
                           'traffic':demo_state['traffic'],**runtime.geometry_status(),
                           'fault':demo_fault} if plan else None))
            if args.capture and steps==120:
                for _ in range(16):sim.render()
                capture(cameras,output,report);captured=True
                sim.stage.GetRootLayer().Export(str(output/'factory_scene.usda'))
                print('FACTORY CAPTURED '+str(output),flush=True)
        if args.capture and not captured:raise RuntimeError('Need at least 120 steps to capture settled scene')
        poses=cell.snapshot()
        if len(poses)!=len(cell.initial) or any(not math.isfinite(v) for p in poses.values() for v in p):raise RuntimeError('Invalid part state')
        if not plan and any(p[2]<cfg['table_height']-.01 for p in poses.values()):raise RuntimeError('A source part fell below its tray')
        if not plan and any(math.dist(p[:2],cell.initial[name][:2])>.015 for name,p in poses.items()):
            raise RuntimeError('A source part moved outside its cradle')
        if plan:
            report['demo']={'routes':route_report,'completed':demo_done and not bool(args.verify_from),'completed_cycles':0 if args.verify_from else demo_cycles,'coverage_start_s':args.verify_from,'cycles':cycle_reports,
                            'stage':demo_state['label'],**runtime.geometry_status(),'assist_mode':'kinematic choreography; no physical success claim'}
            if args.exit_on_complete and not demo_done:raise RuntimeError('Demo verification ended before completing its requested cycles')
            if args.capture:
                for _ in range(4):sim.render()
                capture(cameras,output,report)
        if plan:
            for path,pose in poses.items():
                xf=UsdGeom.Xformable(sim.stage.GetPrimAtPath(path))
                for op in xf.GetOrderedXformOps():
                    if op.GetOpType()==UsdGeom.XformOp.TypeTranslate:op.Set(Gf.Vec3d(*map(float,pose[:3])))
        report['status']='complete' if not plan or demo_done else 'stopped';report['final_parts']=poses;report['steps']=steps;report['elapsed_s']=time.monotonic()-start;report['camera_fps']=args.camera_fps
        if args.verify_from and demo_done:report['status']='segment_complete'
        report['arm_joint_positions']={name:robot.data.joint_pos[0].tolist() for name,robot in robots.items()}
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        sim.stage.GetRootLayer().Export(str(output/'factory_scene.usda'))
        print('FACTORY RESULT: '+('demo cycles='+str(demo_cycles) if plan else '9 parts retained')+'; scene saved',flush=True)
    finally:
        sim._disable_app_control_on_stop_handle=True
        for cam in cameras.values():cam.pause()
        # Shared annotator graph must be detached before any render product dies.
        for cam in cameras.values():
            for annotator in list(cam._custom_annotators):
                cam.detach_annotator(annotator)
        for cam in cameras.values():cam.destroy()
        sim.stop()
        print('FACTORY CLEANUP complete',flush=True)

try:
    main()
    if os.environ.get('FACTORY_STATUS_FILE'):Path(os.environ['FACTORY_STATUS_FILE']).write_text('ok')
except Exception as error:
    (args.output/'report.json').write_text(json.dumps({'run_id':run_id,'status':'failed','error':str(error),'demo':{'completed':False}},ensure_ascii=False,indent=2))
    import traceback
    traceback.print_exc()
    raise
finally:
    if monitor is not None:monitor.close()
    if stream is not None:stream.close()
    ctx=sim_utils.SimulationContext.instance()
    if ctx is not None:ctx._disable_app_control_on_stop_handle=True
    app.close(wait_for_replicator=False)
