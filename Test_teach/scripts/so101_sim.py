#!/usr/bin/env python3
"""Interactive SO-101 tabletop jig teaching and physics-only lift trials."""
import argparse,json,math,os,signal,sys,time,traceback,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT.parent/'Sim_real'
sys.path[:0]=[str(ROOT),str(SOURCE)]
from teach_io.lifecycle import notify
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--preview',action='store_true')
parser.add_argument('--verify',action='store_true')
parser.add_argument('--ui-smoke',action='store_true')
parser.add_argument('--part',choices=['all','cap','ball','housing'],default='ball')
parser.add_argument('--friction',type=float)
parser.add_argument('--config',type=Path,default=ROOT/'config/so101_scene.json')
parser.add_argument('--show-warnings',action='store_true')
AppLauncher.add_app_launcher_args(parser);parser.set_defaults(device='cpu',enable_cameras=True)
args=parser.parse_args()
if args.headless and not(args.preview or args.verify):parser.error('--headless requires --preview or --verify')
if sum((args.preview,args.verify,args.ui_smoke))>1:parser.error('Select one finite verification mode')
if args.ui_smoke and args.headless:parser.error('--ui-smoke requires a GUI')
if args.friction is not None and not .05<=args.friction<=2:parser.error('friction must be 0.05..2')
if not args.show_warnings:args.kit_args=(args.kit_args or '')+' --/log/outputStreamLevel=error --/log/fileLogLevel=info'
if not args.headless:
    # Kit's default atlas excludes Hangul even when a Korean font is styled.
    args.kit_args=(args.kit_args or '')+' --/app/font/customFontPath=/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf --/app/font/customRegionFiles=['+str(ROOT/'assets/korean_ui.txt')+'] --/app/font/overresolutionSize=66'
if args.ui_smoke:
    import faulthandler
    faulthandler.dump_traceback_later(30,repeat=False)
launcher=AppLauncher(args);app=launcher.app
import numpy as np,torch
from scipy.spatial.transform import Rotation
from pxr import UsdGeom,UsdShade,UsdPhysics,PhysxSchema,Gf
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation,ArticulationCfg,RigidObject,RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.converters import UrdfConverter,UrdfConverterCfg
from isaacsim.core.utils.extensions import enable_extension
enable_extension('isaacsim.sensors.camera')
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.viewports import set_camera_view
from arm_sim.conveyor_scene import ConveyorCell
from teach_io import so101_model as model
from PIL import Image

cfg=model.load(args.config)
if args.friction is not None:cfg['contact'].update(static_friction=args.friction,dynamic_friction=args.friction*.8)
out=ROOT/'outputs'/('so101_preview' if args.preview else 'so101_ui_smoke' if args.ui_smoke else 'so101_trial')
out.mkdir(parents=True,exist_ok=True)
report={'run_id':os.environ.get('TEST_TEACH_RUN_ID'),'robot':'so101','hardware_commands':False,'assisted_attachment':False,
        'mass_applied':True,'physical_grasp_verified':False,'configuration':copy.deepcopy(cfg),'events':[],'status':'initializing'}

def save_report():model.atomic_json(out/'report.json',report)

class Cell(ConveyorCell):
    def build(self):
        table=cfg['table'];h=table['height_mm']/1000;w,d=np.array(table['size_mm'])/1000;x,y=np.array(table['center_mm'])/1000
        self.box('/World/Table/Top',(x,y,h-.015),(w,d,.03),self.wood[3])
        self.box('/World/Floor',(0,0,-.025),(1.5,1.5,.04),self.material('floor',(.08,.09,.1)))
        for i,(sx,sy) in enumerate(((-1,-1),(-1,1),(1,-1),(1,1))):self.box('/World/Table/Leg'+str(i),(x+sx*(w/2-.03),y+sy*(d/2-.03),(h-.03)/2),(.035,.035,h-.03),self.wood[1])
    def fixed_mesh(self,path,filename,position,yaw):
        root=self.asset_mesh(path,filename,position,self.materials['white'],rotation=(0,0,yaw))
        visual=self.stage.GetPrimAtPath(path+'/Visual');self.collision(visual)
        UsdPhysics.MeshCollisionAPI.Apply(visual).CreateApproximationAttr('none')
        return root

intentional_exit=False;close_subscription=None;sim=None;exitcode=0

def user_interrupt(*_):
    global intentional_exit
    intentional_exit=True;notify('exit',reason='SO101 simulation close')
signal.signal(signal.SIGINT,user_interrupt);signal.signal(signal.SIGTERM,user_interrupt)
if not args.headless:
    import omni.appwindow
    close_subscription=omni.appwindow.get_default_app_window().get_window_close_event_stream().create_subscription_to_push(user_interrupt)
try:
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=cfg['dt'],device='cpu'))
    sim._disable_app_control_on_stop_handle=True
    sim_utils.DomeLightCfg(intensity=1100.).func('/World/Light',sim_utils.DomeLightCfg(intensity=1100.))
    cell=Cell(sim.stage,{'physics':{'friction':.5,'contact_offset':.00015}})
    cell.materials['white']=cell.material('white_jigs',(.82,.84,.83))
    cell.materials['cap']=cell.material('part_cap',(.40,.43,.45));cell.materials['housing']=cell.material('part_housing',(.78,.80,.79));cell.materials['ball']=cell.material('part_ball',(.45,.47,.49))
    base=np.array(cfg['base_mm'])/1000
    from mount_scene import parts as mount_parts,camera_pose as mount_camera_pose
    mount_material=cell.material('camera_mount_black',(.045,.05,.055))
    for name,path,position,rotation in mount_parts():
        # Mount display shares the exact CAD transforms used for camera calibration.
        cell.asset_mesh('/World/OverheadMount/'+name,str(path),base+position,mount_material,rotation=(90,0,0))
    pcb_pos,pcb_rotation=mount_camera_pose()
    pcb=UsdGeom.Cube.Define(sim.stage,'/World/OverheadMount/CameraPCB');pcb.CreateSizeAttr(1.)
    pcb.AddTranslateOp().Set(Gf.Vec3d(*(base+pcb_pos)))
    q=Rotation.from_matrix(pcb_rotation).as_quat();pcb.AddOrientOp().Set(Gf.Quatf(float(q[3]),Gf.Vec3f(*q[:3])))
    pcb.AddScaleOp().Set(Gf.Vec3f(.032,.032,.0016));cell.bind(pcb.GetPrim(),cell.material('camera_pcb',(.1,.35,.22)))
    report['mount']={'base_above_table_mm':float(cfg['base_mm'][2]-cfg['table']['height_mm']),
                     'pcb_center_world_m':(base+pcb_pos).tolist(),'optical_offset_mm':0.,'collision_model':False}
    carrier_tf=model.frames(cfg,'cap')[0];carrier_pos=base+carrier_tf[:3,3]/1000
    carrier=cell.fixed_mesh('/World/Carrier',str(ROOT/'assets/so101_workcell/carrier.stl'),carrier_pos,cfg['carrier']['yaw_deg'])
    jigs={};bodies={};part_centers={}
    for kind in model.ORDER:
        _,jig,part=model.frames(cfg,kind);p=cfg['parts'][kind];yaw=cfg['carrier']['yaw_deg']+p['jig_yaw_deg']
        jigs[kind]=cell.fixed_mesh('/World/Jigs/'+kind,p['jig_mesh'],base+jig[:3,3]/1000,yaw)
        path='/World/Parts/'+kind;root=cell.asset_mesh(path,p['mesh'],base+part[:3,3]/1000,cell.materials[kind],rotation=(0,0,yaw+p['part_yaw_deg']))
        visual=sim.stage.GetPrimAtPath(path+'/Visual')
        if kind=='ball':
            sphere=UsdGeom.Sphere.Define(sim.stage,path+'/Collider');sphere.CreateRadiusAttr(.015);sphere.AddTranslateOp().Set(Gf.Vec3d(0,0,.015));sphere.CreateVisibilityAttr('invisible');cell.collision(sphere.GetPrim())
        else:
            cell.collision(visual);UsdPhysics.MeshCollisionAPI.Apply(visual).CreateApproximationAttr('convexDecomposition')
            dc=PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(visual);dc.CreateMaxConvexHullsAttr(64);dc.CreateHullVertexLimitAttr(64);dc.CreateVoxelResolutionAttr(500000)
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim());UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(p['mass_kg'])
        rb=PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim());rb.CreateSolverPositionIterationCountAttr(32);rb.CreateSolverVelocityIterationCountAttr(8);rb.CreateEnableCCDAttr(True);rb.CreateMaxDepenetrationVelocityAttr(.3)
        bodies[kind]=RigidObject(RigidObjectCfg(prim_path=path,spawn=None))
    conv=UrdfConverter(UrdfConverterCfg(asset_path=cfg['model_source'],usd_dir=str(ROOT/'assets/so101/usd'),usd_file_name='so101.usd',fix_base=True,
        merge_fixed_joints=True,self_collision=True,force_usd_conversion=True,make_instanceable=False,collision_from_visuals=False,collider_type='convex_decomposition',
        joint_drive=UrdfConverterCfg.JointDriveCfg(drive_type='force',target_type='position',gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=80.,damping=4.))))
    names=[*model.JOINTS,'gripper'];home=np.r_[np.radians(cfg['home_deg'][:5]),cfg['home_deg'][5]/100*1.74533];act=cfg['actuators']
    safe_degrees=cfg.get('safe_deg',cfg['home_deg'])
    safe=np.r_[np.radians(safe_degrees[:5]),safe_degrees[5]/100*1.74533]
    robot=Articulation(ArticulationCfg(prim_path='/World/SO101',spawn=sim_utils.UsdFileCfg(usd_path=conv.usd_path),
        init_state=ArticulationCfg.InitialStateCfg(pos=tuple(base),joint_pos=dict(zip(names,home.tolist()))),
        actuators={'arm':ImplicitActuatorCfg(joint_names_expr=list(model.JOINTS),stiffness=act['arm_stiffness'],damping=act['arm_damping'],effort_limit_sim={n:(act.get('wrist_effort_nm',act['arm_effort_nm']) if n.startswith('wrist') else act['arm_effort_nm']) for n in model.JOINTS},velocity_limit_sim=.8),
                   'grip':ImplicitActuatorCfg(joint_names_expr=['gripper'],stiffness=act['gripper_stiffness'],damping=act['gripper_damping'],effort_limit_sim=act['gripper_effort_nm'],velocity_limit_sim=.8)}))
    # The importer may instance per-link meshes even when the robot root is not
    # instanceable. Material and contact edits require ordinary descendants.
    for _ in range(4):
        instances=[p.GetPath() for p in sim.stage.Traverse() if str(p.GetPath()).startswith('/World/SO101') and p.IsInstance()]
        if not instances:break
        for path in instances:sim.stage.GetPrimAtPath(path).SetInstanceable(False)
    rubber=cell.material('razer_grip_estimate',(.025,.025,.025));material_api=UsdPhysics.MaterialAPI(rubber.GetPrim())
    def apply_friction():
        material_api.CreateStaticFrictionAttr(cfg['contact']['static_friction']);material_api.CreateDynamicFrictionAttr(cfg['contact']['dynamic_friction']);material_api.CreateRestitutionAttr(0.)
        px=PhysxSchema.PhysxMaterialAPI.Apply(rubber.GetPrim());px.CreateFrictionCombineModeAttr('max');px.CreateRestitutionCombineModeAttr('min')
    apply_friction()
    body_material=cell.material('so101_black',(.045,.05,.055))
    blue_material=cell.material('so101_tpu_blue',(.025,.30,.72))
    for prim in sim.stage.Traverse():
        path=str(prim.GetPath())
        if path.startswith('/World/SO101') and '/visuals/' in path and prim.IsA(UsdGeom.Mesh):
            material=rubber if 'razer_' in path else blue_material if 'xlerobot' in path and 'finger' in path else body_material
            cell.bind(prim,material)
    links={p.GetName():p for p in sim.stage.Traverse() if str(p.GetPath()).startswith('/World/SO101') and p.HasAPI(UsdPhysics.RigidBodyAPI)}
    import xml.etree.ElementTree as ET
    urdf=ET.parse(cfg['model_source']).getroot()
    for joint in urdf.findall('joint'):
        a=joint.find('parent').get('link');b=joint.find('child').get('link')
        if a in links and b in links:UsdPhysics.FilteredPairsAPI.Apply(links[a]).CreateFilteredPairsRel().AddTarget(links[b].GetPath())
    pad_paths=[]
    for prim in sim.stage.Traverse():
        if not str(prim.GetPath()).startswith('/World/SO101'):continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            px=PhysxSchema.PhysxCollisionAPI.Apply(prim);px.CreateContactOffsetAttr(.00005 if 'razer_' in str(prim.GetPath()) else .00015);px.CreateRestOffsetAttr(0.)
            if 'razer_' in str(prim.GetPath()):
                cell.bind(prim,rubber);pad_paths.append(str(prim.GetPath()))
    if len(pad_paths)!=2:
        report['robot_collision_paths']=[str(p.GetPath()) for p in sim.stage.Traverse() if str(p.GetPath()).startswith('/World/SO101') and p.HasAPI(UsdPhysics.CollisionAPI)]
        report['robot_prims']=[(str(p.GetPath()),p.GetAppliedSchemas()) for p in sim.stage.Traverse() if str(p.GetPath()).startswith('/World/SO101')][:50]
        sim.stage.Export('/tmp/so101-stage.usda')
        raise RuntimeError('Expected two physical rubber pad colliders: '+str(pad_paths))
    report['pad_colliders']=pad_paths
    # Visible markers have no mass or collision.
    marks={}
    def marker(name,color,radius=.003):
        item=UsdGeom.Sphere.Define(sim.stage,'/World/Teaching/'+name);item.CreateRadiusAttr(radius);item.AddTranslateOp();cell.bind(item.GetPrim(),cell.material('mark_'+name,color));marks[name]=item;return item
    marker('carrier',(.95,.7,.1));marker('tcp',(.95,.2,.1));marker('goal',(.1,.9,.7))
    for kind in model.ORDER:marker('jig_'+kind,(.2,.6,1.));marker('part_'+kind,(.8,.3,.9),.002)
    def set_marker(name,p):marks[name].GetOrderedXformOps()[0].Set(Gf.Vec3d(*np.asarray(p,dtype=float)))
    cameras={}
    for name,eye,target,up in [('perspective',[.69,-.62,.70],[.18,0,.29],[0,0,1]),('top',[.22,0,1.0],[.22,0,.23],[1,0,0])]:
        cam=Camera(prim_path='/World/Camera_'+name,position=np.array(eye),frequency=20,resolution=(1280,800))
        cam.initialize();back=np.array(eye)-target;back/=np.linalg.norm(back);right=np.cross(up,back);right/=np.linalg.norm(right);actual_up=np.cross(back,right)
        rot=np.column_stack((right,actual_up,back));q=Rotation.from_matrix(rot).as_quat();cam.set_world_pose(position=np.array(eye),orientation=np.r_[q[3],q[:3]],camera_axes='usd')
        cam.set_clipping_range(.005,10.);UsdGeom.Camera(cam.prim).CreateFocalLengthAttr(18.);cameras[name]=cam
    set_camera_view(eye=[.69,-.62,.70],target=[.18,0,.29])
    sim.reset();ids,_=robot.find_joints(names,preserve_order=True)
    robot.write_joint_state_to_sim(torch.tensor([home.tolist()],dtype=torch.float32),torch.zeros((1,6)),joint_ids=ids);robot.reset()
    kin=model.Kinematics(cfg['tcp']);selected='ball' if args.part=='all' else args.part
    state={'playing':False,'plan':[],'time':0.,'status':'부품을 선택하고 집기점을 조정하세요.','phase':'대기','target':home.copy(),'command':home.copy(),'selected':selected}
    trial=None;widgets={};joint_models=[];refresh_ui=False;state['manual']=False
    current_grip_limit=act['gripper_effort_nm']
    arm_limits=np.array([act.get('wrist_effort_nm',act['arm_effort_nm']) if n.startswith('wrist') else act['arm_effort_nm'] for n in model.JOINTS])
    def set_grip_limit(value):
        # Keep the physics solver limit and the implicit actuator's diagnostic
        # estimate in agreement. The estimate is not a force-sensor reading.
        robot.write_joint_effort_limit_to_sim(value,joint_ids=[ids[5]])
        robot.actuators['grip'].effort_limit[:]=value
    def q_actual():return robot.data.joint_pos[0,ids].cpu().numpy().astype(float)
    def tcp_base():return kin.fk(model.pose_from_angles(q_actual()))
    def part_frame(kind):
        body=bodies[kind];tf=np.eye(4);tf[:3,3]=(body.data.root_pos_w[0].cpu().numpy()-base)*1000
        q=body.data.root_quat_w[0].cpu().numpy();tf[:3,:3]=Rotation.from_quat(np.r_[q[1:],q[0]]).as_matrix();return tf
    def snapshot():
        candidates={}
        for kind in model.ORDER:
            try:candidates[kind]={'reachable':True,'steps':model.teaching_steps(cfg,kind,q_actual())}
            except ValueError as error:candidates[kind]={'reachable':False,'error':str(error)}
        return {'schema':1,'robot':'so101','units':'mm / degrees / gripper_percent','configuration':copy.deepcopy(cfg),
            'nominal_measurements':model.measurements(cfg),'actual_tcp_base_mm':tcp_base()[:3,3].tolist(),
            'nominal_step_candidates':candidates,
            'joint_pose':model.pose_from_angles(q_actual()),'parts_actual_base':{k:part_frame(k).tolist() for k in model.ORDER},
            'selected':state['selected'],'hardware_verified':False,'friction_status':'assumed, not measured','saved_at':time.time()}
    def check_imported_tcp():
        index=robot.body_names.index('gripper_frame_link')
        pos=robot.data.body_pos_w[0,index].cpu().numpy();q=robot.data.body_quat_w[0,index].cpu().numpy()
        actual=(pos-base)*1000+Rotation.from_quat(np.r_[q[1:],q[0]]).apply(cfg['tcp']['xyz_mm'])
        error=float(np.linalg.norm(actual-tcp_base()[:3,3]));report['imported_tcp_error_mm']=error
        if error>.1:raise RuntimeError(f'Imported robot TCP differs from kinematics by {error:.3f} mm')
    def update_markers():
        for kind in model.ORDER:
            carrier,jig,_=model.frames(cfg,kind);set_marker('jig_'+kind,base+jig[:3,3]/1000)
            p=part_frame(kind);center=p[:3,3]+p[:3,:3]@[0,0,cfg['parts'][kind]['mesh_height_mm']/2];set_marker('part_'+kind,base+center/1000)
        set_marker('carrier',base+carrier[:3,3]/1000+[0,0,.007]);set_marker('tcp',base+tcp_base()[:3,3]/1000)
        point,_=model.goal(cfg,state['selected'],part_frame(state['selected']));set_marker('goal',base+point/1000)
    def capture(prefix):
        for _ in range(10):sim.render()
        for name,cam in cameras.items():
            rgba=cam.get_rgba()
            if rgba is not None and rgba.size:Image.fromarray(rgba[:,:,:3].astype(np.uint8)).save(out/(prefix+'_'+name+'.png'))
    def set_transform(item,tf):
        ops=item.GetOrderedXformOps();ops[0].Set(Gf.Vec3d(*(base+tf[:3,3]/1000)))
        if len(ops)>1:ops[1].Set(Gf.Vec3f(0,0,float(Rotation.from_matrix(tf[:3,:3]).as_euler('xyz',degrees=True)[2])))
    def reset_layout():
        pause()
        state.update(playing=False,plan=[],time=0.,phase='대기',status='배치를 적용했습니다.')
        set_transform(carrier,model.frames(cfg,'cap')[0])
        for kind in model.ORDER:
            _,jig,p=model.frames(cfg,kind);set_transform(jigs[kind],jig)
            q=Rotation.from_matrix(p[:3,:3]).as_quat();pose=np.r_[base+p[:3,3]/1000,q[3],q[:3]]
            bodies[kind].write_root_pose_to_sim(torch.as_tensor(pose[None],dtype=torch.float32));bodies[kind].write_root_velocity_to_sim(torch.zeros((1,6)));bodies[kind].reset()
        apply_friction()
    def guard(fn):
        try:fn()
        except Exception as exc:state.update(playing=False,status=str(exc));report['events'].append({'error':str(exc),'at':time.time()})
    def selected_part(index):
        pause();state['selected']=model.ORDER[index];load_fields();state['status']='집기 위치를 조정한 뒤 상단 대기로 이동하세요.'
    def load_fields():
        p=cfg['parts'][state['selected']]
        for key,variable in widgets.items():
            section,index=key
            if section in ('carrier','carrier_yaw','mu_static','mu_dynamic'):continue
            defaults={'hold_effort_nm':act.get('gripper_hold_effort_nm',act['gripper_effort_nm']),
                'lift_tilt_deg':0.,'approach_clearance_mm':0.}
            values=p.get(section,defaults.get(section))
            variable.set_value(float(values[index] if isinstance(values,list) else values))
    def apply_fields(reset=False):
        candidate=copy.deepcopy(cfg);p=candidate['parts'][state['selected']]
        for (section,index),variable in widgets.items():
            if not reset and section in ('carrier','carrier_yaw','jig_center_carrier_mm','jig_yaw_deg','part_offset_jig_mm','part_yaw_deg'):continue
            value=variable.as_float
            if section=='carrier':candidate['carrier']['center_base_mm'][index]=value
            elif section=='carrier_yaw':candidate['carrier']['yaw_deg']=value
            elif section=='mu_static':candidate['contact']['static_friction']=value
            elif section=='mu_dynamic':candidate['contact']['dynamic_friction']=value
            elif isinstance(p.get(section),list):p[section][index]=value
            else:p[section]=value
        candidate=model.checked(candidate);cfg.clear();cfg.update(candidate)
        if reset:reset_layout()
        else:apply_friction();state['status']='집기점 설정을 적용했습니다.'
    def vertical_plan(kind,seed,start_height,end_height,pf,label,approach=False):
        points=model.vertical_waypoints(cfg,kind,seed,start_height,end_height,pf,approach)
        return waypoint_plan(points,seed,abs(end_height-start_height),label)
    def waypoint_plan(points,seed,distance,label):
        interval=max(2.,distance/25.)/len(points)
        plan=[];previous=seed
        for q in points:
            duration=max(interval,float(np.max(np.abs(q-previous)))/.45)
            plan.append({'phase':label,'q':q,'duration':duration,'linear':True});previous=q
        return plan
    def move(phase):
        pause();apply_fields();kind=state['selected'];pf=part_frame(kind)
        seed=q_actual();q=model.solve(cfg,kind,seed,'hover',pf)
        plan=[{'phase':'상단 대기','q':q,'duration':model.joint_move_duration(seed,q)}]
        if phase=='grasp':
            plan+=vertical_plan(kind,q,cfg['parts'][kind]['hover_mm'],0.,pf,'수직 하강',approach=True)
            contact=model.solve(cfg,kind,plan[-1]['q'],'grasp',pf)
            plan.append({'phase':'고정 집게 접근','q':contact,'duration':model.joint_move_duration(plan[-1]['q'],contact,1.5)})
        state.update(playing=True,plan=plan,time=0.,start=q_actual(),phase=phase)
    def joint_goal(index,value):
        if refresh_ui:return
        pause();state['manual']=True;state['target'][index]=model.angle_from_motor(cfg,index,value)
    def start_trial(kind=None):
        global trial
        pause();apply_fields();kind=kind or state['selected'];state['selected']=kind;p=cfg['parts'][kind];seed=q_actual();pf=part_frame(kind)
        hover=model.solve(cfg,kind,seed,'hover',pf);grasp=model.solve(cfg,kind,hover,'grasp',pf);closed=grasp.copy();closed[5]=p['close_percent']/100*1.74533
        descent=vertical_plan(kind,hover,p['hover_mm'],0.,pf,'수직 하강',approach=True)
        descent.append({'phase':'고정 집게 접근','q':grasp,'duration':model.joint_move_duration(descent[-1]['q'],grasp,1.5)})
        ascent=waypoint_plan(model.lift_waypoints(cfg,kind,closed,pf),closed,p['lift_mm'],'상승 · 지그 위에서 기울임');lifted=ascent[-1]['q']
        trial={'part':kind,'initial_z_m':float(bodies[kind].data.root_pos_w[0,2]),'max_lift_mm':0.,'status':'running','settings':copy.deepcopy(p)}
        report['events'].append(trial)
        plan=[{'phase':'상단 대기','q':hover,'duration':model.joint_move_duration(seed,hover)}]+descent+[{'phase':'집기','q':closed,'duration':model.joint_move_duration(grasp,closed,1.5)}]+ascent+[{'phase':'유지','q':lifted,'duration':3.}]
        state.update(playing=True,plan=plan,time=0.,start=seed,phase='상단 대기',status='접촉·마찰을 이용해 들어 올리기를 시험합니다.')
    def return_safe():
        pause();seed=q_actual();kind=state['selected'];pf=part_frame(kind)
        if any(part_frame(k)[2,3]-model.frames(cfg,k)[2][2,3]>15 for k in model.ORDER):
            state['status']='들어 올린 부품을 내려놓거나 배치 초기화 후 안전 복귀를 누르세요.';return
        goal,_=model.goal(cfg,kind,pf);tcp=tcp_base()[:3,3];height=tcp[2]-goal[2]
        plan=[];previous=seed
        if np.linalg.norm(tcp[:2]-goal[:2])<5 and height<cfg['parts'][kind]['hover_mm']:
            plan=vertical_plan(kind,seed,height,cfg['parts'][kind]['hover_mm'],pf,'지그 위로 후퇴')
            previous=plan[-1]['q']
        plan.append({'phase':'안전 자세 복귀','q':safe.copy(),'duration':model.joint_move_duration(previous,safe)})
        state.update(playing=True,plan=plan,time=0.,start=seed,
            phase=plan[0]['phase'],status='마지막 사진을 참고한 준비·안전 자세로 돌아갑니다.')
    def pause():
        if trial and trial['status']=='running':trial['status']='cancelled';save_report()
        q=q_actual();state.update(playing=False,plan=[],target=q,command=q.copy(),manual=False,phase='정지',status='현재 관절 위치를 유지합니다.')
    def save_values():
        apply_fields();saved=snapshot();model.save(cfg,args.config);model.atomic_json(ROOT/'teaching/so101_layout.json',saved)
        failed=[cfg['parts'][k]['label'] for k,v in saved['nominal_step_candidates'].items() if not v['reachable']]
        state['status']='배치·집기점·스텝 관절값을 teaching/so101_layout.json에 저장했습니다.'
        if failed:state['status']+=' 도달 불가: '+', '.join(failed)
    def step(render=False):
        global trial,current_grip_limit
        finished_phase=None
        if state['playing'] and state['plan']:
            node=state['plan'][0];state['phase']=node['phase'];state['time']+=cfg['dt'];u=min(1.,state['time']/node['duration']);a=u if node.get('linear') else u*u*(3-2*u)
            holding=node['phase'] in ('상승 · 지그 위에서 기울임','유지')
            hold_limit=cfg['parts'][trial['part']].get('hold_effort_nm',act.get('gripper_hold_effort_nm',act['gripper_effort_nm'])) if trial else act.get('gripper_hold_effort_nm',act['gripper_effort_nm'])
            requested_limit=hold_limit if holding else act['gripper_effort_nm']
            if current_grip_limit!=requested_limit:
                set_grip_limit(requested_limit);current_grip_limit=requested_limit
            state['target']=state['start']*(1-a)+node['q']*a
            if u>=1:
                state['plan'].pop(0);state['start']=state['target'].copy();state['time']=0.
                if not state['plan'] or state['plan'][0]['phase']!=node['phase']:finished_phase=node['phase']
                if not state['plan']:
                    state['playing']=False;state['phase']='완료'
                    if trial and trial['status']=='running':
                        lift=(float(bodies[trial['part']].data.root_pos_w[0,2])-trial['initial_z_m'])*1000
                        requested=trial['settings']['lift_mm'];held=(lift>=max(15,requested-10) and trial['max_lift_mm']-lift<5)
                        trial.update(final_lift_mm=lift,drop_from_peak_mm=trial['max_lift_mm']-lift,hold_seconds=3.,status='lifted' if held else 'not_lifted');state['status']=f"{cfg['parts'][trial['part']]['label']}: 최종 상승 {lift:.1f} mm · "+('3초 유지 확인' if held else '파지 안 됨 / 위치 조정 필요');save_report()
        state['command']+=np.clip(state['target']-state['command'],-.8*cfg['dt'],.8*cfg['dt'])
        robot.set_joint_position_target(torch.tensor([state['command'].tolist()],dtype=torch.float32),joint_ids=ids);robot.write_data_to_sim();sim.step(render=render);robot.update(cfg['dt'])
        for body in bodies.values():body.update(cfg['dt'])
        if args.verify and trial and state['phase']:
            effort=np.abs(robot.data.applied_torque[0,ids].cpu().numpy())
            metrics=trial.setdefault('simulated_load',dict(source='Implicit actuator PD torque estimate; not measured contact force',arm_limits_nm=arm_limits.tolist(),closing_limit_nm=act['gripper_effort_nm'],hold_limit_nm=cfg['parts'][trial['part']].get('hold_effort_nm',act.get('gripper_hold_effort_nm')),samples=0,peak_abs_nm=[0.]*6,sum_abs_nm=[0.]*6,arm_limit_samples=0))
            metrics['samples']+=1
            metrics['peak_abs_nm']=np.maximum(metrics['peak_abs_nm'],effort).tolist()
            metrics['sum_abs_nm']=(np.asarray(metrics['sum_abs_nm'])+effort).tolist()
            metrics['arm_limit_samples']+=int(np.any(effort[:5]>=arm_limits*.95))
            by_phase=metrics.setdefault('phases',{}).setdefault(state['phase'],dict(samples=0,peak_abs_nm=[0.]*6,arm_limit_samples=0))
            by_phase['samples']+=1;by_phase['peak_abs_nm']=np.maximum(by_phase['peak_abs_nm'],effort).tolist()
            by_phase['arm_limit_samples']+=int(np.any(effort[:5]>=arm_limits*.95))
        if not np.isfinite(q_actual()).all() or np.max(np.abs(q_actual()))>7:raise RuntimeError('Simulation joint state became invalid')
        if trial and trial['status']=='running':trial['max_lift_mm']=max(trial['max_lift_mm'],(float(bodies[trial['part']].data.root_pos_w[0,2])-trial['initial_z_m'])*1000)
        if args.verify and finished_phase and trial:
            key={'상단 대기':'hover','수직 하강':'approach','고정 집게 접근':'grasp','집기':'closed','상승 · 지그 위에서 기울임':'lift','유지':'hold'}.get(finished_phase,'motion')
            report.setdefault('checkpoints',[]).append(dict(part=trial['part'],phase=key,
                joint_actual=q_actual().tolist(),joint_target=state['target'].tolist(),
                tcp_base=tcp_base().tolist(),part_frame=part_frame(trial['part']).tolist()))
            capture('check_'+trial['part']+'_'+key);save_report()
    for i in range(240):step(i%16==0)
    check_imported_tcp()
    update_markers();capture('initial')
    report.update(status='ready',measurements=model.measurements(cfg),body_names=robot.body_names,body_masses=robot.root_physx_view.get_masses().cpu().numpy().tolist(),initial_state=snapshot())
    save_report()
    if args.ui_smoke:faulthandler.cancel_dump_traceback_later()
    if args.preview:
        report['status']='so101_layout_preview_passed';save_report()
    else:
        panel=None;info_label=status_label=joint_readout=None
        if not args.headless:
            import omni.ui as ui
            # Use Kit's merged default font. A per-widget font bypasses the
            # custom Hangul region configured before AppLauncher starts.
            style={'font_size':16,'Label':{'color':0xffeeeeee},'Button':{'height':32}}
            panel=ui.Window('SO-101 티칭 테스트',width=580,height=940);panel.frame.style=style
            def field(label,key,value,lo=-1000,hi=1000,step=.5):
                with ui.HStack(height=28,spacing=6):
                    ui.Label(label,width=ui.Percent(58));entry=ui.FloatDrag(min=lo,max=hi,step=step);entry.model.set_value(float(value));widgets[key]=entry.model
            def row_buttons(items):
                with ui.HStack(height=34,spacing=6):
                    for label,callback in items:ui.Button(label,clicked_fn=lambda f=callback:guard(f))
            def view_camera(name):
                from omni.kit.viewport.utility import get_active_viewport
                get_active_viewport().camera_path='/World/Camera_'+name
            with panel.frame:
                with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF):
                    with ui.VStack(spacing=7,height=0):
                        ui.Label('SO-101 · 테이블 위 부품 A',height=30,style={'font_size':22})
                        ui.Label('파랑: 지그 중심 / 보라: 부품 중심 / 민트: 집기 목표 / 빨강: TCP',word_wrap=True,height=38)
                        row_buttons([('전체 보기',lambda:view_camera('perspective')),('위에서 보기',lambda:view_camera('top'))])
                        selector=ui.ComboBox(model.ORDER.index(selected),*[cfg['parts'][k]['label'] for k in model.ORDER],height=32)
                        selector.model.add_item_changed_fn(lambda m,_:guard(lambda:selected_part(m.get_item_value_model().as_int)))
                        status_label=ui.Label(state['status'],word_wrap=True,height=58)
                        row_buttons([('상단 대기로 이동',lambda:move('hover')),('집기점으로 이동',lambda:move('grasp')),('정지',pause)])
                        row_buttons([('집고 올리기 시험',start_trial),('배치 초기화',reset_layout),('값 저장',save_values)])
                        row_buttons([('안전 자세로 복귀',return_safe)])
                        ui.Label('1. 집기점 조정 · 선택 부품 기준',height=28,style={'font_size':18})
                        p=cfg['parts'][selected]
                        for i,axis in enumerate('XYZ'):field('TCP '+axis+' 오프셋 (mm)',('grasp_tcp_part_mm',i),p['grasp_tcp_part_mm'][i],-100,150)
                        field('집게 방향 (°)',('grasp_yaw_deg',0),p['grasp_yaw_deg'],-360,360)
                        field('집게 기울기 (180°=아래)',('grasp_pitch_deg',0),p['grasp_pitch_deg'],70,210)
                        with ui.CollapsableFrame('대기 높이 · 그리퍼',collapsed=True,height=0):
                            with ui.VStack(spacing=5,height=0):
                                for label,key,lo,hi in [('상단 여유 높이 (mm)','hover_mm',5,150),('들어 올릴 높이 (mm)','lift_mm',5,150),('열림 (%)','open_percent',0,100),('닫힘 (%)','close_percent',0,100)]:field(label,(key,0),p[key],lo,hi)
                                field('상승 시 최대 기울임 (°)',('lift_tilt_deg',0),p.get('lift_tilt_deg',0),0,30)
                                field('하강할 때 옆 여유 (mm)',('approach_clearance_mm',0),p.get('approach_clearance_mm',0),0,10)
                                ui.Label('40 mm까지 수직 · 이후 관절 한계에서만 기울임',height=28)
                        row_buttons([('집기점 적용',lambda:apply_fields()),('현재 TCP 위치 사용',lambda:teach_current())])
                        ui.Label('2. 거리와 좌표 · 로봇 베이스 기준 mm',height=28,style={'font_size':18})
                        info_label=ui.Label('',word_wrap=True,height=156)
                        with ui.CollapsableFrame('3. 하위 지그와 부품 지그 배치',collapsed=True,height=0):
                            with ui.VStack(spacing=5,height=0):
                                for i,axis in enumerate('XYZ'):field('하위 지그 중심 '+axis,('carrier',i),cfg['carrier']['center_base_mm'][i],-600,600)
                                field('하위 지그 회전 (°)',('carrier_yaw',0),cfg['carrier']['yaw_deg'],-360,360)
                                for i,axis in enumerate(('U 긴 변','V 짧은 변','높이')):field('부품 지그 '+axis,('jig_center_carrier_mm',i),p['jig_center_carrier_mm'][i],-76 if i==0 else -40 if i==1 else 0,76 if i==0 else 40 if i==1 else 30)
                                field('부품 지그 회전 (°)',('jig_yaw_deg',0),p['jig_yaw_deg'],-360,360)
                                for i,axis in enumerate('XYZ'):field('부품 안착 오프셋 '+axis,('part_offset_jig_mm',i),p['part_offset_jig_mm'][i],-100,100)
                                field('안착 부품 회전 (°)',('part_yaw_deg',0),p['part_yaw_deg'],-360,360)
                                row_buttons([('배치 적용 · 부품 재배치',lambda:apply_fields(True))])
                        with ui.CollapsableFrame('4. 관절 직접 조정',collapsed=True,height=0):
                            with ui.VStack(spacing=5,height=0):
                                for i,name in enumerate(('어깨 회전','어깨 들기','팔꿈치','손목 굽힘','손목 회전','그리퍼 %')):
                                    with ui.HStack(height=30):
                                        raw_mode=i<5 and bool(cfg.get('motor_reference'))
                                        ui.Label(name+(' 틱' if raw_mode else ''),width=ui.Percent(40))
                                        lo,hi=(0.,100.) if i==5 else (kin.low[i],kin.high[i])
                                        if raw_mode:
                                            entry=cfg['motor_reference']['calibration'][model.JOINTS[i]];mid=(entry['range_min']+entry['range_max'])/2
                                            lo,hi=math.ceil(lo*4095/360+mid-1e-8),math.floor(hi*4095/360+mid+1e-8)
                                        slider=ui.FloatSlider(min=float(lo),max=float(hi),step=1 if raw_mode else .2)
                                        slider.model.set_value(float(model.motor_value(cfg,i,home[i])))
                                        slider.model.add_value_changed_fn(lambda m,i=i:joint_goal(i,m.as_float))
                                        joint_models.append(slider.model)
                                joint_readout=ui.Label('',word_wrap=True,height=52)
                        with ui.CollapsableFrame('5. 접촉 조건 · 추정값',collapsed=True,height=0):
                            with ui.VStack(spacing=5,height=0):
                                ui.Label('Razer Universal Grip Tape / 두께 0.2 mm\n마찰은 측정값이 아닙니다. TPU·패드 변형은 강체로 근사합니다.',word_wrap=True,height=58)
                                ui.Label(f"손목 상한 {act.get('wrist_effort_nm',act['arm_effort_nm']):.2f} N·m / 집게 닫기 {act['gripper_effort_nm']:.2f} N·m",word_wrap=True,height=36)
                                field('선택 부품 유지 상한 (N·m)',('hold_effort_nm',0),p.get('hold_effort_nm',act.get('gripper_hold_effort_nm',act['gripper_effort_nm'])),.01,act['gripper_effort_nm'],step=.005)
                                field('정지 마찰',('mu_static',0),cfg['contact']['static_friction'],.05,2.)
                                field('운동 마찰',('mu_dynamic',0),cfg['contact']['dynamic_friction'],.05,2.)
                                row_buttons([('마찰 적용',lambda:apply_fields())])
                        ui.Label('저장: Test_teach/teaching/so101_layout.json\n시뮬레이션의 위치 후보입니다. 실물 보정은 별도로 확인하세요.',word_wrap=True,height=52)
            def teach_current():
                kind=state['selected'];part=part_frame(kind);tcp=tcp_base()
                local=part[:3,:3].T@(tcp[:3,3]-part[:3,3])
                for i in range(3):widgets[('grasp_tcp_part_mm',i)].set_value(float(local[i]))
                apply_fields();state['status']='현재 TCP 위치를 집기점에 반영했습니다. 값 저장을 누르세요.'
            # Reuse the existing Isaac dock area; hide unrelated panels, retain the viewport.
            for _ in range(3):app.update()
            # Clear only this simulator's layout, then split the full-height viewport.
            for window in ui.Workspace.get_windows():
                if window.title not in ('Viewport',panel.title):window.visible=False
            viewport=ui.Workspace.get_window('Viewport')
            if viewport:
                viewport.visible=True
                panel.dock_in(viewport,ui.DockPosition.RIGHT,.4);panel.focus()
                ui.Workspace.set_dock_id_width(panel.dock_id,580)
        def refresh_text():
            global refresh_ui
            if status_label:status_label.text=state['phase']+' · '+state['status']
            if info_label:
                kind=state['selected'];m=model.measurements(cfg)[kind];actual=part_frame(kind);center=actual[:3,3]+actual[:3,:3]@[0,0,cfg['parts'][kind]['mesh_height_mm']/2]
                fmt=lambda a:' / '.join(f'{v:+.1f}' for v in a)
                info_label.text='하위 지그 → 지그 U/V/Z: '+fmt(m['jig_center_carrier_mm'])+'\n수평 거리: %.1f mm\n'%m['jig_distance_from_carrier_center_xy_mm']+'지그 → 부품 안착: '+fmt(m['part_offset_jig_mm'])+'\n부품 중심 XYZ: '+fmt(center)+'\n현재 TCP XYZ: '+fmt(tcp_base()[:3,3])+'\n저장된 집기점 XYZ: '+fmt(m['tcp_goal_base_mm'])
            if joint_readout:joint_readout.text='시뮬 모터값: '+', '.join(f'{model.motor_value(cfg,i,q_actual()[i]):g}' for i in range(5))+f" / 집게 {q_actual()[5]/1.74533*100:.1f}%"
            if not state['manual']:
                refresh_ui=True
                try:
                    for i,variable in enumerate(joint_models):variable.set_value(float(model.motor_value(cfg,i,q_actual()[i])))
                finally:refresh_ui=False
        if args.verify:
            kinds=model.ORDER if args.part=='all' else [args.part]
            for kind in kinds:
                reset_layout()
                for i in range(120):step(i%16==0)
                start_trial(kind)
                timeout=sum(node['duration'] for node in state['plan'])+2.
                for i in range(int(timeout/cfg['dt'])):
                    step(i%16==0)
                    if i%32==0:update_markers()
                    if not state['playing']:break
                capture('trial_'+kind)
            passed=all(row.get('status')=='lifted' for row in report['events'] if 'part' in row)
            report.update(status='so101_physics_trial_passed' if passed else 'so101_physics_trial_not_lifted',physics_trial_passed=passed)
            exitcode=0 if passed else 2;save_report()
        else:
            tick=0;last_ui=0.
            while app.is_running() and not intentional_exit:
                step(tick%8==0)
                if tick%32==0:update_markers();refresh_text()
                if args.ui_smoke:
                    if tick==80:
                        # Exercise edit/persistence in temporary paths; never overwrite user settings.
                        original=cfg['parts'][state['selected']]['grasp_tcp_part_mm'][0]
                        widgets[('grasp_tcp_part_mm',0)].set_value(original+1);apply_fields()
                        temp=out/'edited_layout.json';model.save(cfg,temp);assert model.load(temp)['parts'][state['selected']]['grasp_tcp_part_mm'][0]==original+1
                        widgets[('grasp_tcp_part_mm',0)].set_value(original);apply_fields();move('hover')
                    if tick==150:
                        pause();selector.model.get_item_value_model().set_value(0)
                        assert state['selected']=='cap'
                        view_camera('top')
                    if tick==170:
                        joint_models[0].set_value(float(model.motor_value(cfg,0,q_actual()[0])+(10 if cfg.get('motor_reference') else 1)))
                        assert state['manual'] and not state['playing']
                    if tick==180:pause();view_camera('perspective')
                    if tick==200:
                        capture('ui');model.atomic_json(out/'values.json',snapshot())
                        # Capture this Kit window, not the XWayland desktop root
                        # (the latter can reject XGetImage with BadMatch).
                        import omni.renderer_capture
                        (out/'ui_window.png').unlink(missing_ok=True)
                        omni.renderer_capture.acquire_renderer_capture_interface().capture_next_frame_swapchain(str(out/'ui_window.png'))
                        reset_layout();return_safe()
                    if tick>200 and not state['playing']:
                        for _ in range(240):step(False)
                        if np.max(np.abs(q_actual()-safe))>math.radians(2):raise RuntimeError('Safe pose did not settle')
                        capture('safe')
                        if not (out/'ui_window.png').exists():raise RuntimeError('GUI screenshot was not captured')
                        report.update(status='so101_ui_smoke_passed',gui_controls_checked=True,safe_pose_actual=q_actual().tolist(),safe_pose_target=safe.tolist());save_report();break
                tick+=1
    report['final_state']=snapshot();report['configuration']=copy.deepcopy(cfg);save_report()
    print('TEST_TEACH_RESULT',report['status'],flush=True)
except KeyboardInterrupt:
    intentional_exit=True;exitcode=130;report.update(status='interrupted');save_report()
except Exception as error:
    exitcode=1;report.update(status='error',error=str(error));save_report();traceback.print_exc()
finally:
    notify('cleanup',intentional=intentional_exit)
    signal.signal(signal.SIGINT,signal.SIG_IGN);signal.signal(signal.SIGTERM,signal.SIG_IGN)
    if args.headless or args.preview or args.verify or args.ui_smoke:
        sys.stdout.flush();sys.stderr.flush();os._exit(exitcode)
    import threading
    timer=threading.Timer(5.,lambda:os._exit(exitcode));timer.daemon=True;timer.start()
    if sim is not None and app.is_running():sim.stop()
    app.close(wait_for_replicator=False,skip_cleanup=True)
raise SystemExit(exitcode)
