#!/usr/bin/env python3
"""Single OMX tabletop experiment: physics grasp trials with optional calibrated OMX-F mirroring."""
import argparse, copy, json, math, os, signal, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent/'Sim_real'
sys.path.insert(0,str(SOURCE))
sys.path.insert(0,str(ROOT))
from teach_io.lifecycle import notify as notify_launcher
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--verify',action='store_true',help='Run a finite trial, save observations and exit')
parser.add_argument('--preview',action='store_true',help='Save settled layout views and exit')
parser.add_argument('--observed-pose',type=Path,help='Preview a saved six-angle observation; never controls hardware')
parser.add_argument('--observe-smoke',action='store_true',help='Finite GUI/read-only USB observation test; never sends motor commands')
parser.add_argument('--observation-map',type=Path,default=ROOT/'config/follower_visual_checked.json',help='Motor-to-simulation mapping; defaults to the same operator-checked profile as real control')
parser.add_argument('--ui-smoke',action='store_true',help='Finite GUI construction and start/pause/reset callback test')
parser.add_argument('--part',choices=['all','housing','ball','cap'],default='all',help='Transfer all three or test one part')
parser.add_argument('--friction',type=float,default=None,help='Override provisional pad static friction')
parser.add_argument('--follower',action='store_true',help='Opt-in real OMX-F; starts read-only, needs explicit GUI enable')
parser.add_argument('--visual-confirmed',action='store_true',help='Allow the separate operator-checked low-speed profile; not precision calibration')
parser.add_argument('--follower-read-smoke',action='store_true',help='Finite real-mode GUI/read test; motor enable is disabled')
parser.add_argument('--mode-switch-smoke',action='store_true',help='Finite GUI mode switching with actual USB reads and motor writes disabled')
parser.add_argument('--show-warnings',action='store_true',help='Show Isaac warnings in the terminal (always retained in the Kit log)')
parser.add_argument('--follower-plan',action='store_true',help='Test physical-range path without USB motor control')
parser.add_argument('--port',help='Explicit follower /dev/serial/by-id/...')
parser.add_argument('--calibration',type=Path,default=ROOT/'config/follower.json')
parser.add_argument('--camera',help='Real gripper UVC device, shown in a separate 15 FPS window')
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device='cpu',enable_cameras=True)
args=parser.parse_args()
if args.mode_switch_smoke and (args.headless or args.follower or args.verify or args.preview or args.ui_smoke or args.observe_smoke):parser.error('--mode-switch-smoke requires the default GUI')
if args.visual_confirmed and not args.follower:parser.error('--visual-confirmed requires --follower')
if args.follower_read_smoke and not args.follower:parser.error('--follower-read-smoke requires --follower')
args.follower_plan=args.follower_plan or args.follower
if args.follower and (ROOT/'config/devices.json').exists():
    devices=json.loads((ROOT/'config/devices.json').read_text())
    args.port=args.port or devices.get('follower_port')
    args.camera=args.camera or devices.get('gripper_camera')
if args.observe_smoke and (args.headless or args.follower or args.verify or args.preview or args.ui_smoke):
    parser.error('--observe-smoke needs GUI and cannot run a motion trial or follower')
if args.verify and args.preview:parser.error('Choose --verify or --preview, not both')
if args.observed_pose and (not args.preview or args.follower or args.verify or args.ui_smoke):
    parser.error('--observed-pose is only allowed with --preview and no follower')
if args.ui_smoke and args.headless:parser.error('--ui-smoke needs the GUI')
if args.headless and not (args.verify or args.preview):parser.error('--headless needs --verify or --preview (no hidden persistent server)')
if args.friction is not None and not .05<=args.friction<=2:parser.error('friction must be between 0.05 and 2')
if args.follower:
    if not args.port:parser.error('--follower requires --port')
    if args.verify or args.preview or args.ui_smoke or args.headless:
        parser.error('Real follower requires interactive GUI; automatic verification cannot move hardware')
    from teach_io.mapping import Mapping
    try:follower_mapping=Mapping(json.loads(args.calibration.read_text()),allow_visual_confirmation=args.visual_confirmed)
    except Exception as e:parser.error(f'Follower calibration is not ready: {e}')
if args.camera and (args.headless or not args.camera.startswith('/dev/')):
    parser.error('--camera requires GUI and an explicit /dev/... device')
if not args.show_warnings:
    args.kit_args=(args.kit_args or '')+' --/log/outputStreamLevel=error --/log/fileLogLevel=info'
launcher=AppLauncher(args);app=launcher.app
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from pxr import UsdGeom,UsdShade,UsdPhysics,PhysxSchema,Gf
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation,ArticulationCfg,RigidObject,RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.converters import UrdfConverter,UrdfConverterCfg
from isaacsim.core.utils.extensions import enable_extension
enable_extension('isaacsim.sensors.camera')
from isaacsim.sensors.camera import Camera
from arm_sim.conveyor_scene import ConveyorCell
from arm_sim.demo_kinematics import ArmKinematics
from arm_sim.factory import table_legs
from arm_sim.conveyor_hardware import clamp_boxes
from PIL import Image

cfg=json.loads((ROOT/'config/scene.json').read_text())
if args.follower_plan:cfg['rest_point']=[.205,0,.325]
materials=json.loads((ROOT/'config/materials.json').read_text())
out=ROOT/'outputs'/('preview' if args.preview else 'ui_smoke' if args.ui_smoke else 'follower_plan' if args.follower_plan else 'trial')
if args.observed_pose:out=ROOT/'outputs/pose_comparison'
if args.observe_smoke:out=ROOT/'outputs/observe_smoke'
if args.follower_read_smoke:out=ROOT/'outputs/follower_read_smoke'
if args.mode_switch_smoke:out=ROOT/'outputs/mode_switch_smoke'
out.mkdir(parents=True,exist_ok=True)
report={'run_id':os.environ.get('TEST_TEACH_RUN_ID'),'status':'initializing','physical_grasp_verified':False,'hardware_verified':False,
        'assisted_attachment':False,'configuration':cfg,'materials':materials,'events':[]}
import carb
report['kit_log_file']=carb.settings.get_settings().get('/log/file')
print('ISAAC_LOG',report['kit_log_file'],flush=True)
def save(): (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
save()

class Cell(ConveyorCell):
    def build(self):
        z=cfg['table']['height'];x,y=cfg['table']['center'];w,d=cfg['table']['size']
        self.box('/World/Table/Top',(x,y,z-.015),(w,d,.03),self.wood[3])
        for i,(p,s) in enumerate(table_legs(cfg['table'],z)):self.box('/World/Table/Leg'+str(i),p,s,self.wood[1])
        self.box('/World/Floor',(0,0,-.015),(1.2,1.2,.03),self.material('floor',(.08,.09,.10)))
        fixed={'table_height':z,'arms':{'omx':{'model':'omx','position':[0,0]}},
               'tables':[{'id':'table',**cfg['table']}],
               'arm_fixtures':{'omx':{'table':'table','side':'left','clamp_side':'left','clamp_offsets':[-.065,.065],'clamp_pad_depth':.018}}}
        for leaf,p,s in clamp_boxes(fixed,'omx'):self.box('/World/Clamps/'+leaf,p,s,self.materials['metal'])
        for kind,p in cfg['parts'].items():
            for area in ('source','destination'):
                if kind=='cap':
                    self.box(f'/World/Jigs/{area}_cap/Base',(*p[area],z+.00175),(.068,.068,.0035),self.materials['finished'])
                    continue
                root=self.asset_mesh(f'/World/Jigs/{area}_{kind}',p['jig'],(*p[area],z),self.materials['prep' if area=='source' else 'finished'])
                mesh=self.stage.GetPrimAtPath(str(root.GetPath())+'/Visual')
                self.collision(mesh);UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr('none')


def orient(camera,eye,target,up=(0,0,1)):
    eye=np.asarray(eye,dtype=float);back=eye-np.asarray(target);back/=np.linalg.norm(back)
    right=np.cross(up,back);right/=np.linalg.norm(right)
    q=Rotation.from_matrix(np.column_stack((right,np.cross(back,right),back))).as_quat()
    camera.set_world_pose(eye,q[[3,0,1,2]],camera_axes='usd')

def launch_camera(device):
    import subprocess
    from teach_io.client import IO_PYTHON
    camera_log=ROOT/'reports/camera_stderr.log'
    camera_log.parent.mkdir(parents=True,exist_ok=True)
    (ROOT/'outputs/camera').mkdir(parents=True,exist_ok=True)
    (ROOT/'outputs/camera/status.json').unlink(missing_ok=True)
    with camera_log.open('a') as stream:
        return subprocess.Popen([str(IO_PYTHON),str(ROOT/'scripts/camera_view.py'),
            '--device',device,'--parent',str(os.getpid()),'--status-file',str(ROOT/'outputs/camera/status.json')],stderr=stream)


sim=None;exitcode=0;follower=None;camera_process=None;observe_panel=None
exit_release_port=None;intentional_exit=False
def user_interrupt(*_):
    global intentional_exit
    intentional_exit=True
    notify_launcher('exit',reason='Ctrl+C')
    raise KeyboardInterrupt
signal.signal(signal.SIGINT,user_interrupt)
signal.signal(signal.SIGTERM,user_interrupt)
close_subscription=None
if not args.headless:
    import omni.appwindow
    def window_close_requested(_event):
        global intentional_exit
        intentional_exit=True
        notify_launcher('exit',reason='Isaac window close')
    # Notify on push, before Kit's queued close/STOP handlers can block.
    close_subscription=omni.appwindow.get_default_app_window().get_window_close_event_stream().create_subscription_to_push(window_close_requested)
try:
    if args.camera:
        camera_process=launch_camera(args.camera)
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=cfg['dt'],device='cpu'))
    # IsaacLab's default STOP callback renders until the timeline plays again.
    # A closing window never resumes: our main loop owns pause/exit handling.
    sim._disable_app_control_on_stop_handle=True
    sim_utils.DomeLightCfg(intensity=1000).func('/World/Light',sim_utils.DomeLightCfg(intensity=1000))
    sim.set_camera_view(eye=[.9,-.9,1.05],target=[.12,0,.29])
    cell=Cell(sim.stage,{'physics':{'friction':.5,'contact_offset':.0005}})
    model=json.loads((SOURCE/'assets/generated/conveyor_demo/models.json').read_text())['omx']
    kin=ArmKinematics(model,{'position':[0,0],'yaw':0,'base_plate_thickness':0,'demo_pitch_degrees':90},cfg['table']['height'])
    def solve(point,previous):
        q=kin.solve(point,previous);q[4]=q[0] # keep tool/finger orientation fixed in world
        return q
    work_pose=np.r_[solve(cfg['rest_point'],model['home']),.75,-.75]
    home=work_pose.copy()
    if 'start_angles_deg' in cfg:
        from teach_io.mapping import check_physical_limits,vector
        start_angles=np.radians(vector(cfg['start_angles_deg']))
        check_physical_limits(start_angles.tolist())
        home=np.r_[start_angles,-start_angles[5]]
    initial_point=kin.position(home)
    if args.observed_pose:
        observation=json.loads(args.observed_pose.read_text())
        from teach_io.mapping import vector
        values=vector(observation['angles_rad'])
        if any(not lo<=v<=hi for v,(lo,hi) in zip(values,model['limits'][:6])):
            raise ValueError('Observation exceeds the visual model joint ranges')
        home=np.r_[values,-values[5]]
        report['pose_observation']=observation
        report['calibration_verified']=False
    conv=UrdfConverter(UrdfConverterCfg(asset_path=cfg['model_source'],usd_dir=str(ROOT/'assets/omx'),usd_file_name='omx.usd',
        fix_base=True,merge_fixed_joints=True,self_collision=True,convert_mimic_joints_to_normal_joints=True,force_usd_conversion=True,make_instanceable=False,
        collision_from_visuals=False,collider_type='convex_decomposition',
        joint_drive=UrdfConverterCfg.JointDriveCfg(drive_type='force',target_type='position',gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=80.,damping=4.))))
    robot=Articulation(ArticulationCfg(prim_path='/World/OMX',spawn=sim_utils.UsdFileCfg(usd_path=conv.usd_path),
        init_state=ArticulationCfg.InitialStateCfg(pos=tuple(cfg['base']),joint_pos=dict(zip(model['joints'],home.tolist()))),
        actuators={'arm':ImplicitActuatorCfg(joint_names_expr=['joint[1-5]'],stiffness=80.,damping=4.,effort_limit_sim=cfg['arm_effort_nm'],velocity_limit_sim=.8),
                   'passive_gear':ImplicitActuatorCfg(joint_names_expr=['gripper_joint_2'],stiffness=0.,damping=0.,effort_limit_sim=0.,velocity_limit_sim=.5),
                   'grip':ImplicitActuatorCfg(joint_names_expr=['gripper_joint_1'],stiffness=2.,damping=.08,effort_limit_sim=cfg['jaw_effort_nm'],velocity_limit_sim=.5)}))
    mimic_prims=[p for p in sim.stage.Traverse() if p.HasAPI(PhysxSchema.PhysxMimicJointAPI)]
    if len(mimic_prims)!=1:raise RuntimeError('OMX gear constraint was not imported')
    gear=PhysxSchema.PhysxMimicJointAPI(mimic_prims[0],'rotZ')
    gear.GetReferenceJointAxisAttr().Set('rotZ')
    # A rigid gear competes with hard finger/object contacts. Finite stiffness
    # and damping regularize the contact solve without removing gear coupling.
    # Numerical parameters, not measured physical gearbox compliance.
    mimic_prims[0].GetAttribute('physxMimicJoint:rotZ:naturalFrequency').Set(4000.)
    mimic_prims[0].GetAttribute('physxMimicJoint:rotZ:dampingRatio').Set(1.2)
    # This installed importer defaults the reference axis to X although both joints rotate Z.
    report['gripper_gear_constraint']=str(mimic_prims[0].GetPath())
    report['mimic_attributes']={a.GetName():str(a.Get()) for a in mimic_prims[0].GetAttributes()}
    report['mimic_relations']={a.GetName():str(a.GetTargets()) for a in mimic_prims[0].GetRelationships()}
    charcoal=cell.material('robot_black',(.025,.03,.035))
    for prim in sim.stage.Traverse():
        if str(prim.GetPath()).startswith('/World/OMX/') and prim.GetName()=='visuals':
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(charcoal,bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    import xml.etree.ElementTree as ET
    links={p.GetName():p for p in sim.stage.Traverse() if str(p.GetPath()).startswith('/World/OMX/') and p.HasAPI(UsdPhysics.RigidBodyAPI)}
    pairs=[(j.find('parent').get('link'),j.find('child').get('link')) for j in ET.parse(cfg['model_source']).findall('joint')]
    pairs.append(('link6','link7')) # driven opposing gear fingers, gear contact is not simulated
    for a,b in pairs:
        if a in links and b in links:UsdPhysics.FilteredPairsAPI.Apply(links[a]).CreateFilteredPairsRel().AddTarget(links[b].GetPath())
    report['self_collision_enabled']=True
    report['excluded_link_pairs']=[(a,b) for a,b in pairs if a in links and b in links]
    for prim in sim.stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            art=PhysxSchema.PhysxArticulationAPI.Apply(prim);art.CreateSolverPositionIterationCountAttr(32);art.CreateSolverVelocityIterationCountAttr(4)
    rubber=cell.material('rubber',(.025,.03,.035));pm=UsdPhysics.MaterialAPI(rubber.GetPrim())
    mu=args.friction if args.friction is not None else materials['contact']['static_friction']
    pm.CreateStaticFrictionAttr(mu);pm.CreateDynamicFrictionAttr(mu*.7142857)
    # Thin octagonal contact patches on the distal inner sides, provisional placement.
    for name,sign in [('link6',-1),('link7',1)]:
        link=next(p for p in sim.stage.Traverse() if p.GetName()==name and str(p.GetPath()).startswith('/World/OMX/'))
        pts=[]
        for y in (sign*.00915,sign*(.00915+materials['contact']['pad_thickness_m'])):
            pts.extend([[.060+.005*math.cos(a),y,.005*math.sin(a)] for a in np.arange(8)*math.tau/8])
        from scipy.spatial import ConvexHull
        faces=ConvexHull(pts).simplices
        mesh=UsdGeom.Mesh.Define(sim.stage,str(link.GetPath())+'/RubberPad')
        mesh.CreatePointsAttr(pts);mesh.CreateFaceVertexCountsAttr([3]*len(faces));mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
        mesh.CreateSubdivisionSchemeAttr('none');cell.bind(mesh.GetPrim(),rubber);cell.collision(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
        bound,_=UsdShade.MaterialBindingAPI(mesh.GetPrim()).ComputeBoundMaterial('physics')
        if bound.GetPath()!=rubber.GetPath():raise RuntimeError('Pad physics material binding was overridden')
    for prim in sim.stage.Traverse():
        if str(prim.GetPath()).startswith('/World/OMX/') and prim.HasAPI(UsdPhysics.CollisionAPI):
            phys=PhysxSchema.PhysxCollisionAPI.Apply(prim);phys.CreateContactOffsetAttr(.0005);phys.CreateRestOffsetAttr(0.)
    bodies={};initial={}
    for kind,p in cfg['parts'].items():
        pos=np.array([*p['source'],cfg['table']['height']+p['seating_z']+.0005]);pos[:2]+=p['offset_xy']
        path='/World/Parts/'+kind;root=cell.asset_mesh(path,p['mesh'],pos,cell.materials[kind])
        mesh=sim.stage.GetPrimAtPath(path+'/Visual');cell.collision(mesh)
        UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr('convexDecomposition')
        dc=PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(mesh);dc.CreateMaxConvexHullsAttr(64);dc.CreateHullVertexLimitAttr(64);dc.CreateVoxelResolutionAttr(500000)
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim());UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(materials['parts'][kind]['mass_kg'])
        rb=PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim());rb.CreateSolverPositionIterationCountAttr(32);rb.CreateSolverVelocityIterationCountAttr(8);rb.CreateEnableCCDAttr(True);rb.CreateMaxDepenetrationVelocityAttr(.5)
        bodies[kind]=RigidObject(RigidObjectCfg(prim_path=path,spawn=None));initial[kind]=pos
    cams={}
    views=[('perspective',[.85,-.9,.95],[.12,0,.27],[0,0,1]),('top',[.14,0,1.35],[.14,0,.23],[0,1,0])]
    if args.observed_pose:views.append(('side',[.15,-1.1,.45],[.15,0,.45],[0,0,1]))
    for name,eye,target,up in views:
        cam=Camera('/World/Views/'+name,resolution=(1000,850));cam.initialize();orient(cam,eye,target,up)
        cam.set_clipping_range(.005,10);UsdGeom.Camera(cam.prim).CreateFocalLengthAttr(28.)
        cams[name]=cam
    def capture(prefix):
        for _ in range(3):sim.render()
        for name,cam in cams.items():
            rgba=np.asarray(cam.get_rgba(),dtype=np.uint8)
            if rgba.size and rgba[...,:3].any():Image.fromarray(rgba).convert('RGB').save(out/f'{prefix}_{name}.png')
    sim.reset();ids,_=robot.find_joints(model['joints'],preserve_order=True)
    robot.write_joint_state_to_sim(torch.tensor([home.tolist()],dtype=torch.float32),torch.zeros((1,7)),joint_ids=ids);robot.reset()
    def native_tool():
        i=robot.body_names.index('link5');r=robot.data.body_quat_w[0,i].cpu().numpy();xyz=robot.data.body_pos_w[0,i].cpu().numpy()
        return xyz+Rotation.from_quat(r[[1,2,3,0]]).as_matrix()@np.array([.09193,-.0016,0.])
    def command(q):robot.set_joint_position_target(torch.as_tensor(np.asarray(q)[None,:],dtype=torch.float32),joint_ids=ids);robot.write_data_to_sim()
    from collections import deque
    physics_history=deque(maxlen=32)
    def step(q,render=False):
        command(q);sim.step(render=render);robot.update(cfg['dt'])
        for body in bodies.values():body.update(cfg['dt'])
        measured=robot.data.joint_pos[0].cpu().numpy()
        metrics=report.setdefault('physics_metrics',{'max_mimic_error_deg':0.,'max_joint_speed_rad_s':0.})
        ordered=robot.data.joint_pos[0,ids].cpu().numpy()
        metrics['max_mimic_error_deg']=max(metrics['max_mimic_error_deg'],abs(float(ordered[5]+ordered[6]))*180/math.pi)
        metrics['max_joint_speed_rad_s']=max(metrics['max_joint_speed_rad_s'],float(torch.max(torch.abs(robot.data.joint_vel[0]))))
        physics_history.append({'target':np.asarray(q).tolist(),'position':measured.tolist(),
            'velocity':robot.data.joint_vel[0].cpu().numpy().tolist(),
            'effort':robot.data.applied_torque[0].cpu().numpy().tolist()})
        if not np.isfinite(measured).all() or np.max(np.abs(measured))>7:
            bad=[name for name,value in zip(robot.joint_names,measured) if not np.isfinite(value) or abs(value)>7]
            report['physics_failure']={'joints':bad,'history':list(physics_history),
                'objects':{name:body.data.root_state_w[0].cpu().numpy().tolist() for name,body in bodies.items()}}
            save()
            raise RuntimeError('Physics became unstable in '+', '.join(bad)+'; trial aborted without completing the motion')
    for i in range(240):step(home,i%16==0)
    capture('initial')
    report['body_masses']=robot.root_physx_view.get_masses().cpu().numpy().tolist();report['applied_torque']=robot.data.applied_torque.cpu().numpy().tolist();report['joint_names']=robot.joint_names;report['initial_joint_actual']=robot.data.joint_pos[0,ids].cpu().numpy().tolist();report['initial_joint_target']=home.tolist();report['initial_object_positions']={k:b.data.root_pos_w[0].cpu().numpy().tolist() for k,b in bodies.items()};report['status']='ready';report['friction_used']=mu;report['ball_surface_model']='nominal STL; fuzzy skin not reconstructed, friction uncalibrated'
    report['pad_model']='rigid 10mm octagonal contact patch, 0.5mm thickness; rubber deformation uncalibrated'
    save()
    if args.preview:
        report['status']='layout_preview_only';save()
    else:
        # Cartesian segments are solved continuously; transported object is NEVER position-written.
        state={'playing':args.verify,'done':False,'failed':False,'label':'Ready','reset':False,'speed':1.}
        from teach_io.compensation import stage_context
        target_context,target_phase='empty','free'
        stages=[];target=home.copy()
        def add(label,end_point,jaw,duration,kind=None,event=None,real_time_scale=1.):
            stages.append({'label':label,'point':np.asarray(end_point),'jaw':jaw,'duration':duration,'kind':kind,'event':event,'real_time_scale':real_time_scale})
        if np.max(np.abs(home[:5]-work_pose[:5]))>1e-6:
            # Move from the upright start to the checked working waypoint before opening.
            duration=max(2.,1.875*float(np.max(np.abs(home[:5]-work_pose[:5])))/.35)
            add('Move to working height',cfg['rest_point'],float(home[5]),duration,real_time_scale=.75)
            stages[-1]['joint_target']=work_pose[:5].copy()
        if abs(home[5]-.75)>1e-6:
            add('Open gripper at working height',cfg['rest_point'],.75,3.,real_time_scale=.75)
        for kind,p in cfg['parts'].items():
            if args.part!='all' and kind!=args.part:continue
            src=initial[kind]+[0,0,p['grasp_height']]
            dst=np.array([*p['destination'],cfg['table']['height']+p.get('destination_seating_z',p['seating_z'])+p['grasp_height']+.0005]);dst[:2]+=p['offset_xy']
            lift=max(.325,src[2]+.06)
            add(f'{kind}: approach', [src[0],src[1],lift],.75,2.,kind)
            add(f'{kind}: descend',src,.75,1.5,kind)
            add(f'{kind}: close',src,0.,1.5,kind)
            add(f'{kind}: lift',[src[0],src[1],lift],0.,1.5,kind,'lift')
            add(f'{kind}: transfer',[dst[0],dst[1],lift],0.,4.,kind)
            add(f'{kind}: lower',dst,0.,1.5,kind)
            add(f'{kind}: release',dst,.75,1.,kind)
            add(f'{kind}: retreat',[dst[0],dst[1],lift],.75,1.5,kind,'placed')
        add('Return via working height',cfg['rest_point'],.75,2.)
        # Retract clear of the placed part before returning every joint,
        # including wrist roll and gripper, to the same saved startup pose.
        duration=max(2.,1.875*float(np.max(np.abs(home[:6]-work_pose[:6])))/.35)
        add('Return to start',initial_point,float(home[5]),duration,real_time_scale=.75)
        stages[-1]['joint_target']=home[:5].copy()
        add('Settle at start',initial_point,float(home[5]),1.)
        stages[-1]['joint_target']=home[:5].copy()
        def stage_point(st,start,u):
            if 'transfer' not in st['label']:return start+(st['point']-start)*u
            end=st['point'];a=math.atan2(start[1],start[0]);b=math.atan2(end[1],end[0])
            radius=np.linalg.norm(start[:2])*(1-u)+np.linalg.norm(end[:2])*u
            angle=a+(b-a)*u
            return np.array([radius*math.cos(angle),radius*math.sin(angle),start[2]+(end[2]-start[2])*u])
        def stage_joints(st,start,startq,u,previous):
            q=np.asarray(previous).copy()
            q[:5]=(startq[:5]+(st['joint_target']-startq[:5])*u if 'joint_target' in st
                   else solve(stage_point(st,start,u),previous))
            jaw=startq[5]+(st['jaw']-startq[5])*u
            q[5:]=[jaw,-jaw]
            return q
        # Validate the whole commanded path before enabling the trial.
        q=home.copy();last=initial_point.copy();startq=home.copy();hardware_path=[]
        for st in stages:
            for u in np.linspace(0,1,61):
                q=stage_joints(st,last,startq,u,q)
                hardware_path.append(q[:6].tolist())
            last=st['point'];startq=q.copy()
        if args.follower_plan:
            from teach_io.mapping import check_physical_limits
            for q in hardware_path:check_physical_limits(q)
            report['physical_joint_range_checked']=True
        if args.follower:
            follower_mapping.to_ticks(home[:6])
            for q in hardware_path:follower_mapping.to_ticks(q)
            report['hardware_encoder_range_checked']=True
        report['path_ik_valid']=True;report['full_path_collision_audit']=False;report['requested_part']=args.part;save()
        from teach_io.client import Follower
        from teach_io.modes import control_profile,release_supported
        devices=json.loads((ROOT/'config/devices.json').read_text()) if (ROOT/'config/devices.json').exists() else {}
        port=args.port or devices.get('follower_port','')
        control_mode={'real':False,'name':'simulation'}
        control_config={};entry_audit=None
        writes_allowed=not any((args.verify,args.preview,args.ui_smoke,args.observe_smoke,args.follower_read_smoke,args.mode_switch_smoke))
        hardware_state={'text':'Simulation only','fault':False,'pose_dirty':False}
        def stop_follower():
            global follower
            state['playing']=False
            if not args.headless:
                compensation_checkbox.enabled=True;reuse_map_checkbox.enabled=True;evaluation_checkbox.enabled=True
            was_armed=follower is not None and follower.armed
            if follower:
                follower.close();follower=None
            if control_mode['real']:
                hardware_state.update(text='Stopped. Support torque may remain ON. Use Release motors before manual movement.' if was_armed else 'Disconnected. Select Sim -> Real to reconnect.',fault=True)
        def enable_follower():
            global follower,exit_release_port,entry_audit
            if not writes_allowed:
                hardware_state['text']='Read-only test: motor enable is disabled.';return
            if (observe_panel is not None and observe_panel.active) or hardware_state['pose_dirty'] or state['reset']:
                hardware_state['text']='Select Sim -> Real and wait for the virtual scene reset.';return
            if not control_mode['real']:
                hardware_state['text']='Select Sim -> Real first.';return
            if hardware_state['fault'] or state['done'] or state['playing']:return
            if follower is not None and follower.armed:return
            try:
                if follower is None:raise RuntimeError('Select Sim -> Real to connect first')
                from teach_io.entry_collision import EntryCollisionAudit
                from teach_io.entry import EntryPoseChanged
                _,_,mapping=control_profile(args.calibration,args.visual_confirmed)
                if entry_audit is None:
                    fixed={'table_height':cfg['table']['height'],'arms':{'omx':{'model':'omx','position':[0,0]}},
                        'tables':[{'id':'table',**cfg['table']}],
                        'arm_fixtures':{'omx':{'table':'table','side':'left','clamp_side':'left','clamp_offsets':[-.065,.065],'clamp_pad_depth':.018}}}
                    entry_audit=EntryCollisionAudit(cfg,model,clamp_boxes(fixed,'omx'))
                for attempt in range(3):
                    feedback=follower.request('read')
                    if feedback.get('release_required'):
                        hardware_state['text']='Support the arm, check Arm supported, then press Release motors.';return
                    measured=feedback.get('angles')
                    if measured is None:raise RuntimeError('No current joint feedback; reconnect first')
                    entry=entry_audit.check(measured,home[:6].tolist(),mapping)
                    report['current_pose_entry']=entry;save()
                    if not entry['collision_checked']:
                        hardware_state['text']='BLOCKED: entry path intersects '+str(entry['pairs'])+'. Torque is still OFF.';return
                    exit_release_port=port  # user requested torque release on intentional application exit
                    notify_launcher('motor_port',port=port)
                    try:
                        follower.arm(home,hardware_path,compensation_model.as_bool,reuse_map_model.as_bool,evaluation_model.as_bool,entry=entry)
                        break
                    except EntryPoseChanged:
                        if attempt==2:
                            hardware_state['text']='Pose is still moving. Support it and press Enable again. Torque is OFF.'
                            return
                compensation_checkbox.enabled=False;reuse_map_checkbox.enabled=False;evaluation_checkbox.enabled=False
                support_model.set_value(False)
                hardware_state['text']='Enabled: moving slowly from the current pose. Wait for Ready, then Start.'
            except Exception as e:
                stop_follower();hardware_state['text']='BLOCKED: '+str(e)
                print('FOLLOWER BLOCKED:',e,file=sys.stderr,flush=True)
        def start_action():
            if state['reset']:return
            if (observe_panel is not None and observe_panel.active) or hardware_state['pose_dirty']:
                hardware_state['text']='Stop observation and Reset before starting the trial.';return
            if control_mode['real'] and (follower is None or not follower.armed or hardware_state['fault']):
                hardware_state['text']='Enable follower before Start.';return
            if control_mode['real'] and not follower.latest.get('aligned'):
                hardware_state['text']='Wait for start alignment to finish before Start.';return
            state.update(playing=not state['done'])
        def pause_action():
            state.update(playing=False)
            if observe_panel is not None and observe_panel.active:observe_panel.stop()
            if control_mode['real']:stop_follower()
        def hold_action():
            state['playing']=False
            if follower is not None and follower.armed and not follower.latest.get('aligned'):
                stop_follower()  # Hold must also stop the longer automatic entry.
        def reset_action():
            if observe_panel is not None and observe_panel.active:observe_panel.stop()
            if control_mode['real']:stop_follower()
            state.update(reset=True,playing=False)
        def begin_observation():
            state['playing']=False
            stop_follower()
            control_mode.update(real=False,name='observation')
            support_model.set_value(False)
            hardware_state['text']='Reading real angles; no motor commands.'
            setup_window.visible=False
            state['label']='Observing real pose | Physics trial paused'
            window.visible=False
            return True
        def end_observation():
            state['playing']=False
            state['label']='Observation stopped | Reset before trial'
            window.visible=True
        def switch_mode(name):
            global follower
            if name=='observation':
                observe_panel.start();return
            if name=='real' and control_mode['real'] and follower is not None:
                setup_window.visible=True;window.visible=False;return
            state['playing']=False
            if observe_panel.active:observe_panel.stop()
            observe_panel.window.visible=False
            stop_follower()
            control_mode.update(real=name=='real',name=name)
            support_model.set_value(False)
            hardware_state.update(fault=False,pose_dirty=False)
            # Always discard a partial trial before selecting a different direction.
            state.update(reset=True,playing=False)
            setup_window.visible=name=='real';window.visible=name!='real'
            if name!='real':
                hardware_state['text']='Simulation only. Physical motor torque is not changed by this mode.';return
            try:
                path,visual,mapping=control_profile(args.calibration,args.visual_confirmed)
                mapping.to_ticks(home[:6])
                for q in hardware_path:mapping.to_ticks(q)
                control_config.update(path=path,visual=visual)
                report['hardware_encoder_range_checked']=True
                report['mapping_assurance']='measured_calibration' if mapping.cfg.get('verified') else 'user_visual_confirmation'
                setup_assurance.text=('User visually checked mapping; precision and physical grasp are unverified.' if not mapping.cfg.get('verified') else 'Measured calibration loaded; physical grasp remains unverified.')
                follower=Follower(port,path,visual)
                hardware_state['text']='Read-only. Enable checks a slow path from the current pose.'
                report['hardware_start_angles_rad']=home[:6].tolist()
                report['real_mode']='wall-clock joint target mirroring: startup 0.75x, transfer 1x, adaptive pacing; not real grasp perception'
                save()
                open_gripper_camera()
            except Exception as e:
                stop_follower();hardware_state.update(text='Connection failed: '+str(e),fault=True)
        def release_action():
            if not writes_allowed or not support_model.as_bool:
                hardware_state['text']='Support the arm and check Arm supported before releasing motors.'
                return False
            selected=control_mode['name']
            state['playing']=False
            if observe_panel.active:observe_panel.stop()
            stop_follower()
            try:
                release_supported(port,support_model.as_bool,writes_allowed)
                support_model.set_value(False)
                hardware_state['text']='Torque OFF; watchdog cleared.'
                switch_mode(selected)
                return True
            except Exception as e:
                support_model.set_value(False)
                hardware_state.update(text='RELEASE FAILED: '+str(e),fault=True)
                print(hardware_state['text'],file=sys.stderr,flush=True)
                return False
        def apply_observed_pose(values):
            hardware_state['pose_dirty']=True
            q=np.r_[values,-values[5]]
            robot.write_joint_state_to_sim(torch.tensor([q.tolist()],dtype=torch.float32),torch.zeros((1,7)),joint_ids=ids)
            command(q);sim.forward();robot.update(0.)
            report['observation_native_fk_error_m']=float(np.linalg.norm(native_tool()-kin.position(q)))
            if report['observation_native_fk_error_m']>.001:
                raise RuntimeError('Native rendered articulation did not match the observed pose')
            return robot.data.joint_pos[0,ids].cpu().numpy()[:6].tolist()
        def open_gripper_camera():
            global camera_process
            if camera_process is not None and camera_process.poll() is None:
                return
            devices=json.loads((ROOT/'config/devices.json').read_text())
            device=args.camera or devices.get('gripper_camera')
            if not device:
                cameralabel.text='Camera: no device configured in config/devices.json';return
            if not Path(device).exists():
                cameralabel.text='Camera USB not found. Reconnect USB, then press Open camera.';return
            camera_process=launch_camera(device)
            cameralabel.text='Camera: connecting...'
        def restart_camera():
            global camera_process
            if camera_process is not None:
                if camera_process.poll() is None:
                    camera_process.terminate()
                    try:camera_process.wait(timeout=1)
                    except Exception:camera_process.kill();camera_process.wait()
                camera_process=None
            open_gripper_camera()
        def update_camera_status():
            global camera_process
            path=ROOT/'outputs/camera/status.json'
            if camera_process is None:return
            try:
                info=json.loads(path.read_text())
                if info.get('pid')!=camera_process.pid:return
                if info['state']=='live':
                    age=time.time()-info['updated_at']
                    cameralabel.text=(f'Camera: LIVE | {info["width"]} x {info["height"]} | {info["fps"]:.1f} fps | separate window' if age<2 else 'Camera: no recent frames. Use Restart camera.')
                elif info['state']=='error':cameralabel.text='Camera error: '+info['message']
                elif info['state']=='closed':cameralabel.text='Camera closed. Press Open camera.'
            except (OSError,ValueError,KeyError):pass
            if camera_process.poll() is not None:
                if camera_process.returncode:
                    if not cameralabel.text.startswith('Camera error:'):
                        cameralabel.text='Camera failed to open. Use Restart camera; details: reports/camera_stderr.log'
                    print('CAMERA ERROR: preview exited; details in reports/camera_stderr.log',file=sys.stderr,flush=True)
                else:cameralabel.text='Camera closed. Press Open camera.'
                camera_process=None
        labelmodel=None
        if not args.headless:
            import omni.ui as ui
            from teach_io.observe_panel import ObservePanel,TEXT_STYLE
            window=ui.Window('OMX transfer test',width=650,height=660)
            window.frame.style=TEXT_STYLE
            with window.frame:
                trial_scroll=ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF)
                with trial_scroll, ui.VStack(spacing=6,height=0,style=TEXT_STYLE):
                    ui.Label('Simulation trial',height=30,style={'font_size':22})
                    statuslabel=ui.Label('Ready',height=34)
                    with ui.HStack(height=38,spacing=6):
                        ui.Button('Start / Resume',clicked_fn=start_action)
                        ui.Button('Pause',clicked_fn=pause_action)
                        ui.Button('Reset',clicked_fn=reset_action)
                    ui.Label('Housing / Ball / Cap | Physics grasp test',height=28)
                    hardwarelabel=ui.Label(hardware_state['text'],word_wrap=True,height=52)
                    with ui.CollapsableFrame('Trial details',collapsed=True,height=0):
                        with ui.VStack(spacing=6,height=0):
                            ui.Label('A failed grasp stops the trial. No assisted attachment.',word_wrap=True,height=44)
                            ui.Label(f'Pad friction: {mu:.2f} (assumed, not measured).',word_wrap=True,height=40)
                            ui.Label('Mass, friction and pad thickness need real calibration.',word_wrap=True,height=44)
                            ui.Label('Real: startup 0.75x / transfer 1.0x. Evaluation tolerates small tracking errors.',word_wrap=True,height=44)
            devices=json.loads((ROOT/'config/devices.json').read_text()) if (ROOT/'config/devices.json').exists() else {}
            observe_panel=ObservePanel(args.port or devices.get('follower_port',''),args.observation_map,model['limits'][:6],
                apply_observed_pose,begin_observation,end_observation,open_gripper_camera,ROOT/'outputs/observation/latest.json')
            setup_window=ui.Window('OMX real control',width=860,height=650,visible=False)
            import omni.appwindow
            setup_window.set_visibility_changed_fn(lambda visible:stop_follower() if not visible and control_mode['real'] else None)
            setup_window.frame.style=TEXT_STYLE
            setup_rows=[]
            compensation_model=ui.SimpleBoolModel(True)
            reuse_map_model=ui.SimpleBoolModel(False)
            evaluation_model=ui.SimpleBoolModel(True)
            with setup_window.frame:
                real_scroll=ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF)
                with real_scroll, ui.VStack(spacing=6,height=0):
                    ui.Label('Sim -> Real | Synchronized motion',height=30,style={'font_size':22})
                    setup_status=ui.Label(hardware_state['text'],height=48,word_wrap=True)
                    with ui.HStack(height=28,spacing=6):
                        evaluation_checkbox=ui.CheckBox(evaluation_model,width=24)
                        ui.Label('Evaluation: keep speed below 6 deg lag',word_wrap=True)
                    with ui.HStack(height=38,spacing=6):
                        ui.Button('Enable follower',clicked_fn=enable_follower,enabled=writes_allowed)
                        ui.Button('Start / Resume',clicked_fn=start_action)
                        ui.Button('Hold / learn',clicked_fn=hold_action)
                        ui.Button('Stop + hold',clicked_fn=stop_follower)
                    widths=[ui.Percent(v) for v in (28,24,24,24)]
                    with ui.HStack(height=30):
                        ui.Label('Joint',width=widths[0]);ui.Label('Real deg',width=widths[1])
                        goal_header=ui.Label('Start deg',width=widths[2]);ui.Label('Delta deg',width=widths[3])
                    for name,value in zip(['11 Base','12 Shoulder','13 Elbow','14 Wrist pitch','15 Wrist roll','16 Gripper'],home[:6]):
                        with ui.HStack(height=28):
                            ui.Label(name,width=widths[0])
                            actual_label=ui.Label('--',width=widths[1])
                            goal_label=ui.Label(f'{math.degrees(value):.2f}',width=widths[2])
                            delta_label=ui.Label('--',width=widths[3])
                            setup_rows.append((actual_label,goal_label,delta_label))
                    ui.Label('Enable checks a slow path from your current pose. Keep the gripper empty and workspace clear.',height=42,word_wrap=True)
                    tracking_label=ui.Label('Position compensation ON | Waiting for feedback',height=48,word_wrap=True)
                    ui.Label('Stop holds torque. Ctrl+C / app exit releases motors: support the arm.',height=44,word_wrap=True)
                    with ui.CollapsableFrame('Control details',collapsed=True,height=0) as comp_details:
                        with ui.VStack(spacing=6,height=0):
                            with ui.HStack(height=30,spacing=6):
                                compensation_checkbox=ui.CheckBox(compensation_model,width=24)
                                ui.Label('Position compensation (up to 3 deg)')
                            with ui.HStack(height=30,spacing=6):
                                reuse_map_checkbox=ui.CheckBox(reuse_map_model,width=24)
                                ui.Label('Reuse saved poses: same hardware / payload',word_wrap=True)
                            ui.Label('Options apply on Enable. Learn only at steady free-space poses.',height=40,word_wrap=True)
                            comp_detail_label=ui.Label('No compensation telemetry yet.',height=44,word_wrap=True)
                            setup_assurance=ui.Label('Select Sim -> Real to load the control profile.',height=44,word_wrap=True)
                            ui.Label('No manual start matching. Checked entry: smooth motion at up to 0.1 rad/s.',height=44,word_wrap=True)
                            ui.Button('Show trial controls',clicked_fn=lambda:setattr(window,'visible',True),height=36)
            mode_window=ui.Window('OMX modes',width=900,height=180)
            mode_window.frame.style=TEXT_STYLE
            support_model=ui.SimpleBoolModel(False)
            with mode_window.frame:
                with ui.VStack(spacing=5):
                    with ui.HStack(height=38,spacing=6):
                        ui.Button('Simulation only',clicked_fn=lambda:switch_mode('simulation'))
                        ui.Button('Real -> Sim',clicked_fn=lambda:switch_mode('observation'))
                        ui.Button('Sim -> Real',clicked_fn=lambda:switch_mode('real'))
                    with ui.HStack(height=32,spacing=6):
                        ui.CheckBox(support_model,width=24)
                        ui.Label('Arm supported (torque release)',width=410)
                        ui.Button('Release motors',clicked_fn=release_action,enabled=writes_allowed)
                    mode_status=ui.Label('Simulation only | Motors are not enabled automatically.',height=52,word_wrap=True)
                    with ui.HStack(height=34,spacing=8):
                        ui.Button('Open camera',clicked_fn=open_gripper_camera,width=180)
                        ui.Button('Restart camera',clicked_fn=restart_camera,width=180)
                        cameralabel=ui.Label('Camera: closed. Press Open camera.',word_wrap=True)
            from teach_io.layout import TeachLayout
            layout=TeachLayout(mode_window,[window,setup_window,observe_panel.window])
        stage_template=copy.deepcopy(stages)
        idx=0;elapsed=0.;startpoint=initial_point.copy();startjoint=home.copy();tick=0
        from teach_io.motion_timing import MotionClock
        real_clock=MotionClock(cfg['dt'])
        if args.follower:switch_mode('real')
        while not intentional_exit and app.is_running():
            if observe_panel is not None:observe_panel.poll()
            real_ready=True;pace=1.
            if follower is not None and not hardware_state['fault']:
                try:
                    feedback=follower.update(target,context=target_context,phase=target_phase,trial_active=state['playing'])
                    if 'angles' in feedback:
                        error=max(abs(a-b) for a,b in zip(target[:5],feedback['angles'][:5]))
                        tracking=feedback.get('tracking',{})
                        pace=tracking.get('pace',0.)
                        real_ready=follower.armed and feedback.get('aligned',False) and pace>0
                        if follower.armed:
                            if not feedback.get('aligned') or hardware_state.get('entry_preview'):
                                displayed=home[:6] if feedback.get('aligned') else tracking.get('desired_angles',feedback['angles'])
                                entry_q=np.r_[displayed,-displayed[5]]
                                robot.write_joint_state_to_sim(torch.tensor([entry_q.tolist()],dtype=torch.float32),torch.zeros((1,7)),joint_ids=ids)
                                command(entry_q);sim.forward();robot.update(0.)
                                hardware_state['entry_preview']=not feedback.get('aligned')
                            if not feedback.get('aligned'):
                                hardware_state['text']='Moving slowly from current pose. Wait for Ready before Start.'
                            else:
                                recovery_limit=5 if tracking.get('profile')=='evaluation' else 3
                                action={'following':'Following','slowing':'Slowing for load / lag','catching_up':f'Waiting for arm (resume <= {recovery_limit} deg)'}
                                motion_label='Ready' if not state['playing'] and pace>0 else action.get(tracking.get('state'),'Ready')
                                hardware_state['text']=f'{motion_label} | ID {tracking.get("worst_id","--")} lag {tracking.get("lag_deg",0):.1f} deg | pace {pace*100:.0f}%'
                        else:
                            difference=max(abs(a-b) for a,b in zip(home[:6],feedback['angles']))
                            hardware_state['text']=('Torque/watchdog active. Support the arm, check Arm supported, then press Release motors.' if feedback.get('release_required') else f'Read-only / torque OFF | Start difference {math.degrees(difference):.1f} deg | Enable checks current-pose entry')
                        if not args.headless:
                            goals=tracking.get('desired_angles',target[:6]) if follower.armed else home[:6]
                            goal_header.text='Target deg' if follower.armed else 'Start deg'
                            for (actual_label,goal_label,delta_label),actual,goal in zip(setup_rows,feedback['angles'],goals):
                                actual_label.text=f'{math.degrees(actual):.2f}'
                                goal_label.text=f'{math.degrees(goal):.2f}'
                                delta_label.text=f'{math.degrees(goal-actual):+.2f}'
                            biases=tracking.get('bias_deg',[0]*6)
                            comp=tracking.get('compensation',{})
                            comp_detail_label.text=f'{comp.get("map_status","New session")} | support {comp.get("map_confidence",0)*100:.0f}% | {comp.get("context","unknown")} / {comp.get("phase","unknown")}'
                            comp_state='Waiting for Enable' if not follower.armed else 'Learning' if comp.get('learning') else 'Contact: no learning' if comp.get('phase')=='contact' else 'Ready'
                            tracking_label.text=(f'Comp {"ON" if comp.get("enabled",compensation_model.as_bool) else "OFF"} | {comp_state} | {comp.get("learned_poses",0)} poses | PWM {tracking.get("pwm_percent",0):.0f}%\n'
                                f'Shoulder / elbow / wrist: {biases[1]:+.1f} / {biases[2]:+.1f} / {biases[3]:+.1f} deg')
                        if feedback.get('telemetry_file'):report['hardware_telemetry_file']=feedback['telemetry_file']
                        if feedback.get('motion_error'):report['hardware_motion_error']=feedback['motion_error']
                except Exception as e:
                    stop_follower();hardware_state['text']='STOPPED: '+str(e)
                    print('FOLLOWER STOPPED:',e,file=sys.stderr,flush=True)
            if not args.headless:
                setup_status.text=hardware_state['text']
                mode_status.text=control_mode['name'].upper()+' | '+hardware_state['text']
            if not args.headless:
                update_camera_status()
                if tick>=5:layout.update()
            if state['reset']:
                robot.write_joint_state_to_sim(torch.as_tensor(home[None,:],dtype=torch.float32),torch.zeros((1,7)),joint_ids=ids);robot.reset()
                for kind,body in bodies.items():
                    pose=torch.tensor([[*initial[kind],1.,0,0,0]],dtype=torch.float32)
                    body.write_root_pose_to_sim(pose);body.write_root_velocity_to_sim(torch.zeros((1,6)));body.reset()
                stages=copy.deepcopy(stage_template)
                target_context,target_phase='empty','free'
                idx=0;elapsed=0.;target=home.copy();startpoint=initial_point.copy();startjoint=home.copy()
                state.update(reset=False,done=False,failed=False,label='Ready');hardware_state['pose_dirty']=False;report['events']=[];report['stages']=[];report['status']='ready';save()
                command(home);sim.forward();robot.update(0.)
            active=state['playing'] and not state['done'] and real_ready
            real_scale=stages[idx]['real_time_scale'] if idx<len(stages) else 1.
            substeps=real_clock.steps(time.monotonic(),active and control_mode['real'],real_scale,pace)
            if active:
                for substep in range(substeps if control_mode['real'] else 1):
                    st=stages[idx];elapsed+=cfg['dt'];u=min(1,elapsed/st['duration']);s=u*u*u*(10+u*(-15+6*u))
                    target=stage_joints(st,startpoint,startjoint,s,target)
                    target_context,target_phase=stage_context(st)
                    state['label']=st['label']
                    step(target,not control_mode['real'] and tick%16==0)
                    if elapsed>=st['duration']:break
                if control_mode['real']:
                    sim.render()
                    # Worker IO is ~20-30 ms; physics substeps keep the virtual path in time.
                    # The clock discards stalls instead of issuing a catch-up burst.
                    if not substeps:time.sleep(.002)
            else:sim.render();time.sleep(.01)
            tick+=1
            if state['playing'] and not state['done'] and elapsed>=stages[idx]['duration']:
                st=stages[idx];kind=st['kind'];event=st['event']
                report.setdefault('stages',[]).append({'native_tool':native_tool().tolist(),'label':st['label'],'real_time_scale':st['real_time_scale'],'object_position':bodies[kind].data.root_pos_w[0].cpu().numpy().tolist() if kind else None,'object_quaternion':bodies[kind].data.root_quat_w[0].cpu().numpy().tolist() if kind else None,'target':target.tolist(),'actual':robot.data.joint_pos[0,ids].cpu().numpy().tolist(),'torques':robot.data.applied_torque[0,ids].cpu().numpy().tolist(),'tool_actual':kin.position(robot.data.joint_pos[0,ids].cpu().numpy()).tolist()});save()
                if not control_mode['real'] and st['label'].endswith(('descend','close','lower','release')):capture(st['label'].replace(': ','_'))
                if event:
                    actual=bodies[kind].data.root_pos_w[0].cpu().numpy().copy()
                    if event=='lift':ok=actual[2]>initial[kind][2]+.035
                    else:
                        destination=np.array(cfg['parts'][kind]['destination'])+cfg['parts'][kind]['offset_xy']
                        expected_z=cfg['table']['height']+cfg['parts'][kind].get('destination_seating_z',cfg['parts'][kind]['seating_z'])
                        quat=bodies[kind].data.root_quat_w[0].cpu().numpy()
                        upright=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()[2,2]
                        ok=np.linalg.norm(actual[:2]-destination)<.008 and abs(actual[2]-expected_z)<.004 and (kind=='ball' or upright>math.cos(math.radians(8)))
                    report['events'].append({'event':event,'part':kind,'position':actual.tolist(),'passed':bool(ok)})
                    if event=='placed':report['events'][-1].update(xy_error_m=float(np.linalg.norm(actual[:2]-destination)),z_error_m=float(abs(actual[2]-expected_z)),tilt_degrees=float(math.degrees(math.acos(np.clip(upright,-1,1)))))
                    if event=='lift' and ok and not (args.follower_plan or control_mode['real']):
                        # Observe grasp offset once; do not write or attach the object.
                        tcp=kin.position(robot.data.joint_pos[0,ids].cpu().numpy())
                        measured_offset=actual-tcp
                        nominal_offset=np.array([0.,0.,-cfg['parts'][kind]['grasp_height']])
                        correction=nominal_offset-measured_offset
                        for future in stages[idx+1:]:
                            if future['kind']!=kind:continue
                            future['point'][:2]+=correction[:2]
                            if future['label'].endswith(('lower','release')):future['point'][2]+=correction[2]+.003
                        report['events'][-1]['measured_grasp_offset']=measured_offset.tolist()
                        report['placement_feedback']='simulator pose observation after lift; no perception model'
                    if not control_mode['real']:capture(f'{kind}_{event}')
                    if not ok:
                        state.update(done=True,playing=False,failed=True,label=f'FAILED: {kind} {event}')
                        report['status']='grasp_or_placement_failed';save()
                        if control_mode['real']:stop_follower()
                startpoint=st['point'];startjoint=target.copy();elapsed=0.;idx+=1
                if idx==len(stages) and not state['failed']:
                    state.update(done=True,playing=False,label='Completed (real grasp not verified)' if control_mode['real'] else 'Completed (simulation trial only)')
                    report['status']='mirroring_sequence_complete_unverified' if control_mode['real'] else 'simulation_trial_passed';save()
                    if control_mode['real']:stop_follower()
                    else:capture('final')
            if not args.headless:
                statuslabel.text=state['label'];hardwarelabel.text=hardware_state['text']
            if args.ui_smoke:
                if tick==30:
                    assert not state['playing'] and idx==0
                    report['ui_initially_paused']=True;start_action()
                if tick==150:
                    pause_action();report['ui_pause_test']=not state['playing'];reset_action()
                if tick==180:
                    assert idx==0 and not state['playing'] and not state['reset']
                    report['ui_reset_test']=True
                    assert evaluation_model.as_bool and evaluation_checkbox.enabled
                    evaluation_model.set_value(False);assert not evaluation_model.as_bool
                    evaluation_model.set_value(True)
                    setup_window.visible=True
                    report['ui_evaluation_toggle_test']=True
                if tick==190:
                    import omni.renderer_capture
                    omni.renderer_capture.acquire_renderer_capture_interface().capture_next_frame_swapchain(str(out/'ui_evaluation.png'))
                if tick==200:
                    report['status']='ui_smoke_passed';save();capture('ui_ready');break
            if args.observe_smoke:
                if tick==30:
                    assert not state['playing']
                    report['observation_button_started']=observe_panel.start()
                if tick==70:
                    observe_panel.camera()
                if tick==100:
                    if observe_panel.error:raise RuntimeError(observe_panel.error)
                    assert observe_panel.active and observe_panel.frames>=5
                    assert not state['playing']
                    report['observation_last']=observe_panel.last
                    report['observation_frames']=observe_panel.frames
                    assert camera_process is not None and camera_process.poll() is None
                    report['observation_camera_started']=True
                    capture('observed_pose')
                    try:
                        import omni.renderer_capture
                        ui_capture=omni.renderer_capture.acquire_renderer_capture_interface()
                        (out/'ui_screen.png').unlink(missing_ok=True)
                        ui_capture.capture_next_frame_swapchain(str(out/'ui_screen.png'))
                        report['ui_screen_capture_requested']=True
                    except Exception as e:report['ui_screen_capture_error']=str(e)
                if tick==140:
                    observe_panel.stop();start_action()
                    assert not state['playing']
                    report['observation_blocks_stale_demo_resume']=True
                    reset_action()
                if tick==170:
                    assert not observe_panel.active and not hardware_state['pose_dirty'] and not state['playing']
                    report['observation_reset_ok']=True
                    report['ui_screen_captured']=(out/'ui_screen.png').exists()
                    report['status']='observation_ui_smoke_passed';save();break
            if args.follower_read_smoke:
                if tick==80:
                    assert follower is not None and not follower.armed and 'angles' in follower.latest
                    start_action();enable_follower()
                    assert not state['playing'] and not follower.armed
                    report['read_only_motor_feedback']=follower.latest
                    report['read_only_enable_blocked']=True
                    assert compensation_model.as_bool and not reuse_map_model.as_bool
                    compensation_model.set_value(False);reuse_map_model.set_value(True)
                    assert not compensation_model.as_bool and reuse_map_model.as_bool
                    compensation_model.set_value(True);reuse_map_model.set_value(False)
                    assert compensation_checkbox.enabled and reuse_map_checkbox.enabled
                    assert real_scroll.scroll_y_max<=1
                    report['compensation_default_on_reuse_off']=True
                    report['compensation_controls_toggle_read_only']=True
                    report['real_scroll_max']=real_scroll.scroll_y_max
                    import omni.renderer_capture
                    (out/'ui_screen.png').unlink(missing_ok=True)
                    omni.renderer_capture.acquire_renderer_capture_interface().capture_next_frame_swapchain(str(out/'ui_screen.png'))
                if tick==90:
                    comp_details.collapsed=False
                if tick==100:
                    import omni.renderer_capture
                    omni.renderer_capture.acquire_renderer_capture_interface().capture_next_frame_swapchain(str(out/'compensation_details.png'))
                if tick==110:
                    assert not follower.armed and not state['playing']
                    report['status']='follower_read_smoke_passed';save();break
            if args.mode_switch_smoke:
                if tick==15:
                    report['initial_layout']=layout.snapshot()
                    assert trial_scroll.scroll_y_max<=1
                    report['simulation_scroll_max']=trial_scroll.scroll_y_max
                    assert window.docked and mode_window.docked and mode_window.is_selected_in_dock()
                    import omni.renderer_capture
                    omni.renderer_capture.acquire_renderer_capture_interface().capture_next_frame_swapchain(str(out/'layout_simulation.png'))
                    open_gripper_camera()
                if tick==20:
                    switch_mode('observation')
                if tick==60:
                    assert observe_panel.active and observe_panel.frames>5 and follower is None
                    report['first_observation_frames']=observe_panel.frames
                    old_observer=observe_panel.reader.process
                    switch_mode('real')
                    assert old_observer.poll() is not None and not observe_panel.active
                if tick==100:
                    assert follower is not None and not follower.armed and 'angles' in follower.latest
                    start_action();enable_follower()
                    assert not state['playing'] and not follower.armed
                    assert not release_action()
                    report['real_mode_read_only']=True
                    report['real_layout']=layout.snapshot()
                    assert real_scroll.scroll_y_max<=1
                    report['real_scroll_max']=real_scroll.scroll_y_max
                    assert setup_window.docked and mode_window.docked
                    camera_info=json.loads((ROOT/'outputs/camera/status.json').read_text())
                    assert camera_info['state']=='live' and time.time()-camera_info['updated_at']<2
                    report['camera_live_before_restart']=camera_info
                    old_camera=camera_process
                    restart_camera()
                    assert old_camera.poll() is not None
                    import omni.renderer_capture
                    (out/'ui_screen.png').unlink(missing_ok=True)
                    omni.renderer_capture.acquire_renderer_capture_interface().capture_next_frame_swapchain(str(out/'ui_screen.png'))
                    old_follower=follower.process
                if tick==115:
                    switch_mode('observation')
                    assert old_follower.poll() is not None and follower is None
                if tick==145:
                    assert observe_panel.window.docked
                    assert observe_panel.scroll.scroll_y_max<=1
                    report['observation_scroll_max']=observe_panel.scroll.scroll_y_max
                    info=json.loads((ROOT/'outputs/camera/status.json').read_text())
                    assert info['state']=='live' and time.time()-info['updated_at']<2
                    report['camera_live_after_restart']=info
                    omni.renderer_capture.acquire_renderer_capture_interface().capture_next_frame_swapchain(str(out/'layout_observation.png'))
                if tick==155:
                    assert observe_panel.active and observe_panel.frames>5
                    report['second_observation_frames']=observe_panel.frames
                    old_observer=observe_panel.reader.process
                    switch_mode('simulation')
                    assert not observe_panel.active and old_observer.poll() is not None
                if tick==165:
                    assert follower is None and not hardware_state['pose_dirty'] and not state['reset']
                    start_action()
                if tick==180:
                    assert state['playing'] and not control_mode['real']
                    pause_action();report['simulation_resume_ok']=True
                    switch_mode('real')
                if tick==200:
                    assert follower is not None and not follower.armed
                    old_follower=follower.process
                    setup_window.visible=False
                if tick==205:
                    assert follower is None and old_follower.poll() is not None and not state['playing']
                    report['control_window_close_stops_connection']=True
                    report['status']='mode_switch_smoke_passed';save();break
            if args.verify and state['done']:break
        if not app.is_running():intentional_exit=True
        report['physics_trial_passed']=report['status']=='simulation_trial_passed'
        report['physical_grasp_verified']=False
        report['hardware_verified']=False;save()
        if args.verify and report['status']!='simulation_trial_passed':exitcode=2
    print('TEST_TEACH_RESULT',report['status'],flush=True)
except KeyboardInterrupt:
    intentional_exit=True;exitcode=130
    report.update(status='interrupted');save()
except Exception as error:
    report.update(status='error',error=str(error));save();traceback.print_exc();exitcode=1
finally:
    notify_launcher('cleanup',intentional=intentional_exit)
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    signal.signal(signal.SIGTERM,signal.SIG_IGN)
    if observe_panel is not None:
        try:observe_panel.shutdown()
        except Exception as close_error:print('OBSERVER CLOSE:',close_error,file=sys.stderr,flush=True)
    if follower is not None:
        try:follower.close()
        except Exception as close_error:print('FOLLOWER CLOSE:',close_error,file=sys.stderr,flush=True)
    if intentional_exit and exit_release_port:
        try:
            from teach_io.modes import release_supported
            release_supported(exit_release_port,True)
            notify_launcher('released',port=exit_release_port)
            print('EXIT: all six motor torques OFF',flush=True)
            report['exit_torque_released']=True;save()
        except Exception as release_error:
            print('EXIT TORQUE RELEASE FAILED:',release_error,file=sys.stderr,flush=True)
            report['exit_torque_released']=False;report['exit_release_error']=str(release_error);save()
    if camera_process is not None and camera_process.poll() is None:
        camera_process.terminate()
        try:camera_process.wait(timeout=2)
        except Exception:camera_process.kill();camera_process.wait()
    if args.headless or args.verify or args.preview or args.ui_smoke or args.observe_smoke or args.follower_read_smoke or args.mode_switch_smoke:
        import os
        sys.stdout.flush();sys.stderr.flush();os._exit(exitcode)
    import threading
    shutdown_timer=threading.Timer(5.,lambda:os._exit(exitcode));shutdown_timer.daemon=True;shutdown_timer.start()
    if sim is not None and app.is_running():sim.stop()
    app.close(wait_for_replicator=False,skip_cleanup=True)
# Isaac shutdown can alter process exit handling; the launcher also checks report status.
raise SystemExit(exitcode)
