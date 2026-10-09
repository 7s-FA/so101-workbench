"""Single-world Isaac Sim contact environment; import only after AppLauncher."""
from collections import deque
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
import gymnasium as gym
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from arm_sim.conveyor_scene import ConveyorCell
from core import JOINTS, bounded_target, task_success, atomic_json, read_json
from teaching_executor import TeachingExecutor, estimated_load_raw
from appearance import finish_material, indoor_lights

class Cell(ConveyorCell):
    def build(self):
        pass

def root_pose(matrix_mm):
    tf = np.array(matrix_mm)
    q = Rotation.from_matrix(tf[:3,:3]).as_quat()[[3,0,1,2]]
    return np.r_[tf[:3,3]/1000,q, np.zeros(6)]

class BEnvironment(gym.Env):
    metadata = {'render_modes': []}
    def __init__(self, reference, scene, output):
        super().__init__()
        self.ref, self.cfg, self.output = reference, scene, Path(output).resolve()
        self._magnet_near=False;self.last_magnets={'active_pairs':0,'total_attraction_n':0.}
        self.assembly_status={}
        self.dt, self.decimation = scene['dt'], 16
        self.control_dt = self.dt*self.decimation
        self.randomized = True
        self.observation_space = gym.spaces.Box(-np.inf,np.inf,(26,),np.float32)
        self.action_space = gym.spaces.Box(-1,1,(6,),np.float32)
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=self.dt,device='cuda:0'))
        self.sim._disable_app_control_on_stop_handle = True
        self.lighting=indoor_lights(self.sim.stage)
        cell = Cell(self.sim.stage, {'physics':{'friction':scene['background_friction'],
                                              'contact_offset':scene['contact_offset_m']}})
        self.carrier=UsdGeom.Xform.Define(self.sim.stage,'/World/Carrier')
        self.carrier.GetPrim().SetDisplayName('운반 지그 · 터틀봇')
        self._write_carrier_pose(scene['carrier']['pose'])
        self.receiver=UsdGeom.Xform.Define(self.sim.stage,'/World/Receiver')
        self.receiver.GetPrim().SetDisplayName('리니어 팔레트')
        self.receiver.AddTranslateOp().Set(Gf.Vec3d(*[float(v)/1000 for v in scene.get('assembly',{}).get('receiver_offset_mm',[0,0,0])]))
        for i,row in enumerate(scene['static']):
            path = ('/World/Carrier/g' if row.get('group')=='carrier' else '/World/Receiver/g' if row.get('group')=='receiver' else '/World/Environment/g')+str(i)
            material=cell.material('static'+str(i),tuple(row['color']))
            cell.asset_mesh(path,row['mesh'],(0,0,0),material,scale=1.)
            finish_material(cell,material,row['color'],row.get('surface'),self.sim.stage.GetPrimAtPath(path+'/Visual'))
            if row['collision']:
                visual = self.sim.stage.GetPrimAtPath(path+'/Visual')
                cell.collision(visual)
                UsdPhysics.MeshCollisionAPI.Apply(visual).CreateApproximationAttr('none')
        self.parts = {}
        self.initial_states = {}
        self.contact_materials=[]
        for i,(kind,row) in enumerate(scene['parts'].items()):
            path = '/World/Parts/'+kind
            state = root_pose(row['initial_pose_mm'])
            material=finish_material(cell,cell.material(kind,tuple(row['color_rgb'])),row['color_rgb'])
            self.contact_materials.append(material)
            tpu_contact=scene.get('assembly',{}).get('tpu_contact',{})
            if kind=='tpu' and tpu_contact.get('stiffness_n_m',0)>0:
                soft=PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
                soft.CreateCompliantContactStiffnessAttr(tpu_contact['stiffness_n_m'])
                soft.CreateCompliantContactDampingAttr(tpu_contact.get('damping_ns_m',5.))
            root = cell.asset_mesh(path,row['mesh'],state[:3],material,scale=1.)
            # Use one explicit quaternion rather than an Euler convention change.
            root.AddOrientOp().Set(Gf.Quatf(float(state[3]),Gf.Vec3f(*map(float,state[4:7]))))
            visual = self.sim.stage.GetPrimAtPath(path+'/Visual')
            if kind in scene.get('assembly',{}).get('magnet_seats',{}):
                magnet_material=finish_material(cell,cell.material('magnet_'+kind,(.65,.67,.69)),(.65,.67,.69),{'roughness':.25,'specular':.8})
                for j,seat in enumerate(scene['assembly']['magnet_seats'][kind]):
                    cylinder=UsdGeom.Cylinder.Define(self.sim.stage,path+'/Visual/magnet_'+str(j))
                    cylinder.CreateRadiusAttr(seat['diameter_mm']/2000);cylinder.CreateHeightAttr(seat['depth_mm']/1000)
                    cylinder.CreateAxisAttr('Z');center=np.asarray(seat['face_mm'],float);center[2]-=seat['depth_mm']/2
                    cylinder.AddTranslateOp().Set(Gf.Vec3d(*map(float,center/1000)));cell.bind(cylinder.GetPrim(),magnet_material)
            cell.collision(visual)
            if kind in ('cap','housing'):
                # Preserve the cap recess and the housing cavity for the TPU.
                UsdPhysics.MeshCollisionAPI.Apply(visual).CreateApproximationAttr('sdf')
                sdf=PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(visual)
                sdf.CreateSdfResolutionAttr(512)
                sdf.CreateSdfSubgridResolutionAttr(6)
            else:
                UsdPhysics.MeshCollisionAPI.Apply(visual).CreateApproximationAttr('convexDecomposition')
                dc = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(visual)
                dc.CreateMaxConvexHullsAttr(64);dc.CreateHullVertexLimitAttr(64);dc.CreateVoxelResolutionAttr(500000)
            UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
            UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(row['mass_kg'])
            rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
            rb.CreateSolverPositionIterationCountAttr(32);rb.CreateSolverVelocityIterationCountAttr(8)
            rb.CreateEnableCCDAttr(True);rb.CreateMaxDepenetrationVelocityAttr(.3)
            self.parts[kind] = RigidObject(RigidObjectCfg(prim_path=path,spawn=None))
            self.initial_states[kind] = torch.tensor(state[None],dtype=torch.float32)
        converter = UrdfConverter(UrdfConverterCfg(asset_path=scene['robot'],
            usd_dir=str(self.output/'assets/usd'),usd_file_name='so101.usd',fix_base=True,
            merge_fixed_joints=True,self_collision=True,force_usd_conversion=True,
            make_instanceable=False,collision_from_visuals=False,collider_type='convex_decomposition',
            joint_drive=UrdfConverterCfg.JointDriveCfg(drive_type='force',target_type='position',
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=80.,damping=4.))))
        act = scene['actuators']
        self.home = reference.sample(0)[0]
        self.robot = Articulation(ArticulationCfg(prim_path='/World/SO101',
            spawn=sim_utils.UsdFileCfg(usd_path=converter.usd_path),
            init_state=ArticulationCfg.InitialStateCfg(joint_pos=dict(zip(JOINTS,self.home.tolist()))),
            actuators={'arm':ImplicitActuatorCfg(joint_names_expr=list(JOINTS[:5]),
                stiffness=act['arm_stiffness'],damping=act['arm_damping'],velocity_limit_sim=.8,
                effort_limit_sim={n:act['wrist_effort_nm'] if n.startswith('wrist') else act['arm_effort_nm'] for n in JOINTS[:5]}),
                'grip':ImplicitActuatorCfg(joint_names_expr=['gripper'],stiffness=act['gripper_stiffness'],
                    damping=act['gripper_damping'],effort_limit_sim=act['gripper_effort_nm'],velocity_limit_sim=.8)}))
        for _ in range(4):
            instances=[p.GetPath() for p in self.sim.stage.Traverse() if str(p.GetPath()).startswith('/World/SO101') and p.IsInstance()]
            if not instances:break
            for path in instances:self.sim.stage.GetPrimAtPath(path).SetInstanceable(False)
        rubber=finish_material(cell,cell.material('grip_pad',(.025,.025,.025)),(.025,.025,.025))
        plastic=finish_material(cell,cell.material('robot_plastic',tuple(scene['robot_color_rgb'])),scene['robot_color_rgb'])
        tpu=finish_material(cell,cell.material('robot_tpu',tuple(scene['finger_color_rgb'])),scene['finger_color_rgb'])
        finger_compliance=scene['pad_contact'].get('finger_contact_stiffness',0)
        if finger_compliance>0:
            finger_material=PhysxSchema.PhysxMaterialAPI.Apply(tpu.GetPrim())
            finger_material.CreateCompliantContactStiffnessAttr(finger_compliance)
            finger_material.CreateCompliantContactDampingAttr(scene['pad_contact'].get('finger_contact_damping',5.))
        m=UsdPhysics.MaterialAPI(rubber.GetPrim());m.CreateStaticFrictionAttr(scene['pad_contact']['static_friction'])
        m.CreateDynamicFrictionAttr(scene['pad_contact']['dynamic_friction'])
        pad_material=PhysxSchema.PhysxMaterialAPI.Apply(rubber.GetPrim())
        pad_material.CreateFrictionCombineModeAttr('max')
        if scene['pad_contact'].get('compliant_contact_stiffness',0)>0:
            pad_material.CreateCompliantContactStiffnessAttr(scene['pad_contact']['compliant_contact_stiffness'])
            pad_material.CreateCompliantContactDampingAttr(scene['pad_contact'].get('compliant_contact_damping',0))
        pad_count=0
        self.contact_materials.extend([rubber,tpu])
        self.contact_materials=[(UsdPhysics.MaterialAPI(mat.GetPrim()),
                                 UsdPhysics.MaterialAPI(mat.GetPrim()).GetStaticFrictionAttr().Get(),
                                 UsdPhysics.MaterialAPI(mat.GetPrim()).GetDynamicFrictionAttr().Get()) for mat in self.contact_materials]
        self.has_compliance=(scene['pad_contact'].get('compliant_contact_stiffness',0)>0 or finger_compliance>0
                             or scene.get('assembly',{}).get('tpu_contact',{}).get('stiffness_n_m',0)>0)
        for prim in self.sim.stage.Traverse():
            if not str(prim.GetPath()).startswith('/World/SO101'):continue
            if '/visuals/' in str(prim.GetPath()) and prim.IsA(UsdGeom.Mesh):
                cell.bind(prim,rubber if 'razer_' in str(prim.GetPath()) else
                          tpu if 'xlerobot' in str(prim.GetPath()) and 'finger' in str(prim.GetPath()) else plastic)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                is_pad='razer_' in str(prim.GetPath())
                px=PhysxSchema.PhysxCollisionAPI.Apply(prim)
                px.CreateContactOffsetAttr(.00005 if is_pad else .00015);px.CreateRestOffsetAttr(0.)
                if is_pad:cell.bind(prim,rubber);pad_count+=1
                elif finger_compliance>0 and 'xlerobot' in str(prim.GetPath()) and 'finger' in str(prim.GetPath()):cell.bind(prim,tpu)
        if pad_count != 2:raise RuntimeError('두 물리 집게 패드가 필요합니다.')
        links={p.GetName():p for p in self.sim.stage.Traverse() if str(p.GetPath()).startswith('/World/SO101') and p.HasAPI(UsdPhysics.RigidBodyAPI)}
        for joint in ET.parse(scene['robot']).getroot().findall('joint'):
            a,b=joint.find('parent').get('link'),joint.find('child').get('link')
            if a in links and b in links:UsdPhysics.FilteredPairsAPI.Apply(links[a]).CreateFilteredPairsRel().AddTarget(links[b].GetPath())
        self.sim.reset()
        self.ids,_=self.robot.find_joints(list(JOINTS),preserve_order=True)
        model_limits=self.robot.data.joint_pos_limits[0,self.ids].cpu().numpy()
        self.limits=np.column_stack((np.maximum(model_limits[:,0],reference.limits[:,0]),
                                    np.minimum(model_limits[:,1],reference.limits[:,1])))
        if np.any(self.limits[:,0]>=self.limits[:,1]):raise ValueError('모터/CAD 관절 범위가 겹치지 않습니다.')
        for q,_ in (reference.sample(t) for t in reference.ends):
            if np.any(q<self.limits[:,0]-.005) or np.any(q>self.limits[:,1]+.005):
                raise ValueError('티칭 경로가 CAD 관절 한계 밖입니다. 원본을 검토하세요.')
        self.defaults={}
        for key,asset in [('robot',self.robot),*self.parts.items()]:
            view=asset.root_physx_view
            self.defaults[key]=(view.get_masses().clone(),view.get_inertias().clone(),view.get_material_properties().clone())
        self.episode_number=0
        self.load_scale=np.asarray(scene['load_model']['full_scale_effort_nm'])

    def _write_carrier_pose(self,pose):
        x,y,yaw=map(float,pose)
        offset=self.cfg['carrier'].get('cad_offset_mm',[0,0,0])
        common=UsdGeom.XformCommonAPI(self.carrier)
        common.SetTranslate(Gf.Vec3d((x+offset[0])/1000,(y+offset[1])/1000,(self.cfg['carrier']['bottom_z_mm']+offset[2])/1000))
        tilt=self.cfg['carrier'].get('cad_tilt_deg',[0.,0.])
        common.SetRotate(Gf.Vec3f(*map(float,[*tilt,yaw+self.cfg['carrier'].get('cad_yaw_deg',0.)])))

    def carrier_pose(self):
        """Read the authored scene, including viewport/gizmo changes, in arm2 XY."""
        tf=np.asarray(self.carrier.ComputeLocalToWorldTransform(Usd.TimeCode.Default()),dtype=float).T
        tf[:3,3]-=np.asarray(self.cfg['carrier'].get('cad_offset_mm',[0,0,0]))/1000
        correction=Rotation.from_euler('xyz',[*self.cfg['carrier'].get('cad_tilt_deg',[0.,0.]),self.cfg['carrier'].get('cad_yaw_deg',0.)],degrees=True).as_matrix()
        rotation=Rotation.from_matrix(tf[:3,:3]@correction.T)
        if not np.allclose(tf[:3,:3].T@tf[:3,:3],np.eye(3),atol=1e-5):
            raise ValueError('운반 지그의 크기는 변경할 수 없습니다. X/Y/방향만 조정하세요.')
        roll,pitch,yaw=rotation.as_euler('xyz',degrees=True)
        if abs(roll)>1e-3 or abs(pitch)>1e-3 or abs(tf[2,3]*1000-self.cfg['carrier']['bottom_z_mm'])>.01:
            raise ValueError('현재 티칭은 지그 높이·기울기를 유지합니다. X/Y/방향만 조정하세요.')
        return [float(tf[0,3]*1000),float(tf[1,3]*1000),float(yaw)]

    def carrier_matches_plan(self):
        pose=np.asarray(self.carrier_pose());old=np.asarray(self.ref.jig_pose)
        return bool(np.linalg.norm(pose[:2]-old[:2])<.01 and abs((pose[2]-old[2]+180)%360-180)<.001)

    def apply_jig_pose(self,pose=None):
        """Validate an entire new plan before moving the carrier and resetting B."""
        from scene_assets import transform
        pose=self.carrier_pose() if pose is None else self.ref.checked_jig_pose(pose)
        candidate=self.ref.retarget(pose)
        for q,_ in (candidate.sample(t) for t in candidate.ends):
            if np.any(q<self.limits[:,0]-.005) or np.any(q>self.limits[:,1]+.005):
                raise ValueError('새 지그 위치의 경로가 CAD 관절 범위를 벗어납니다.')
        old_pose=self.cfg['carrier']['pose'];z=self.cfg['carrier']['bottom_z_mm']
        offset=np.asarray(self.cfg['carrier'].get('cad_offset_mm',[0,0,0]))
        yaw=self.cfg['carrier'].get('cad_yaw_deg',0.)
        delta=transform(np.r_[pose[:2],z]+offset,pose[2]+yaw)@np.linalg.inv(transform(np.r_[old_pose[:2],z]+offset,old_pose[2]+yaw))
        parts=deepcopy(self.cfg['parts'])
        for row in parts.values():
            initial=delta@np.array(row['initial_pose_mm'])
            goal=candidate.kin.fk(candidate.steps[row['release_step']-1]['ticks'])@np.linalg.inv(
                 candidate.kin.fk(candidate.steps[row['grasp_step']]['ticks']))@initial
            row['initial_pose_mm']=initial.tolist();row['target_pose_mm']=goal.tolist()
        if 'assembly' in self.cfg:
            for kind,row in parts.items():row['target_pose_mm']=deepcopy(self.cfg['assembly']['target_poses_mm'][kind])
        self.sim.pause()
        old_ref,old_cfg,old_states=self.ref,deepcopy(self.cfg),self.initial_states
        try:
            self._write_carrier_pose(pose)
            self.ref=candidate;self.cfg['parts']=parts;self.cfg['carrier']['pose']=list(pose)
            self.initial_states={k:torch.tensor(root_pose(row['initial_pose_mm'])[None],dtype=torch.float32) for k,row in parts.items()}
            self.sim.play();self.reset(seed=42,options={'randomized':False})
        except Exception:
            self.ref=old_ref;self.cfg=old_cfg;self.initial_states=old_states
            self._write_carrier_pose(old_pose)
            self.sim.play();self.reset(seed=42,options={'randomized':False})
            raise
        finally:
            self.sim.forward();self.sim.pause()
        actual=self.carrier_pose()
        if not self.carrier_matches_plan():raise RuntimeError('장면의 지그 위치와 계산 입력이 일치하지 않습니다.')
        record=dict(source='Isaac scene carrier pose',pose=actual,jig_id=self.ref.jig_id,
                    coordinate_frame='arm2_base',camera_detection=False,hardware_commands=False,
                    max_position_error_mm=max(p.get('position_error_mm',0) for p in self.ref.corrections),
                    reset_to_start=True)
        atomic_json(self.output/'jig_application.json',record)
        atomic_json(self.output/'scene.json',self.cfg)
        atomic_json(self.output/'resolved_plan.json',dict(jig_references=self.ref.jig_references,
                    steps=self.ref.steps,corrections=self.ref.corrections))
        return record

    def set_cad_offset(self,offset_mm,yaw_deg=None,tilt_deg=None):
        from alignment import aligned_scene
        candidate=aligned_scene(self.cfg,self.ref,offset_mm,yaw_deg,tilt_deg)
        self.sim.pause();self.cfg=candidate
        self.initial_states={k:torch.tensor(root_pose(row['initial_pose_mm'])[None],dtype=torch.float32) for k,row in candidate['parts'].items()}
        self._write_carrier_pose(self.ref.jig_pose)
        self.sim.reset();self.reset(seed=42,options={'randomized':False})

    def q(self):
        return self.robot.data.joint_pos[0,self.ids].cpu().numpy().copy()

    def _physics_step(self, target):
        self.applied_target=np.asarray(target).copy()
        self.robot.set_joint_position_target(torch.tensor(target[None],dtype=torch.float32,device=self.robot.device),joint_ids=self.ids)
        if self._magnet_near:self._update_magnets()
        self.robot.write_data_to_sim()
        for body in self.parts.values():body.write_data_to_sim()
        self.sim.step(render=False);self.robot.update(self.dt)
        for body in self.parts.values():body.update(self.dt)

    def _update_magnets(self):
        from assembly import magnetic_wrenches
        transforms=[];centers=[]
        for name in ('housing','cap'):
            body=self.parts[name];state=body.data.root_state_w[0].cpu().numpy()
            tf=np.eye(4);tf[:3,3]=state[:3];tf[:3,:3]=Rotation.from_quat(state[[4,5,6,3]]).as_matrix()
            transforms.append(tf);centers.append(body.data.root_com_pos_w[0].cpu().numpy())
        fh,th,fc,tc,self.last_magnets=magnetic_wrenches(*transforms,*centers,self.cfg['assembly'])
        for name,force,torque in [('housing',fh,th),('cap',fc,tc)]:
            body=self.parts[name]
            body.set_external_force_and_torque(torch.tensor(force[None,None],dtype=torch.float32,device=body.device),
                                              torch.tensor(torque[None,None],dtype=torch.float32,device=body.device),is_global=True)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._magnet_near=False;self.last_magnets={'active_pairs':0,'total_attraction_n':0.};self.assembly_status={}
        self.assembly_seen={'housing':False,'tpu':False};self.assembly_timers=dict(housing=0.,tpu=0.,complete=0.)
        self.episode_number+=1
        randomize=(options or {}).get('randomized',self.randomized)
        u=lambda lo,hi:float(self.np_random.uniform(lo,hi)) if randomize else 1.
        self.domain={'mass_scale':u(.8,1.2),'friction_scale':u(.8,1.2),'gain_scale':u(.85,1.15),
                     'delay_steps':int(self.np_random.integers(0,3)) if randomize else 0}
        if self.has_compliance:
            # This Isaac version's three-channel tensor setter has no damping
            # channel. Keep authored compliant materials and edit their friction
            # in place, instead of replacing them with friction/restitution data.
            for material,static,dynamic in self.contact_materials:
                material.GetStaticFrictionAttr().Set(static*self.domain['friction_scale'])
                material.GetDynamicFrictionAttr().Set(dynamic*self.domain['friction_scale'])
            self.domain['friction_scope']='explicit part, pad and TPU finger materials'
        indices=torch.tensor([0],dtype=torch.int32)
        for key,asset in [('robot',self.robot),*self.parts.items()]:
            mass,inertia,materials=self.defaults[key]
            scale=self.domain['mass_scale'];view=asset.root_physx_view
            view.set_masses(mass*scale,indices);view.set_inertias(inertia*scale,indices)
            materials=materials.clone();materials[...,:2]*=self.domain['friction_scale']
            if not self.has_compliance:view.set_material_properties(materials,indices)
        act=self.cfg['actuators'];gain=self.domain['gain_scale']
        for group,names,kp,kd in [('arm',self.ids[:5],act['arm_stiffness'],act['arm_damping']),
                                 ('grip',[self.ids[5]],act['gripper_stiffness'],act['gripper_damping'])]:
            self.robot.write_joint_stiffness_to_sim(kp*gain,joint_ids=names)
            self.robot.write_joint_damping_to_sim(kd*gain,joint_ids=names)
            self.robot.actuators[group].stiffness[:]=kp*gain;self.robot.actuators[group].damping[:]=kd*gain
        self.robot.write_joint_state_to_sim(torch.tensor(self.home[None],dtype=torch.float32,device=self.robot.device),
                                            torch.zeros((1,6),device=self.robot.device),joint_ids=self.ids)
        self.robot.reset()
        for key,body in self.parts.items():
            state=self.initial_states[key].to(body.device).clone()
            if randomize:state[0,:2]+=torch.tensor(self.np_random.uniform(-.001,.001,2),dtype=torch.float32,device=body.device)
            body.write_root_state_to_sim(state);body.reset()
        self.command=self.home.copy();self.previous_action=np.zeros(6)
        self.queue=deque([self.home.copy() for _ in range(max(1,self.domain['delay_steps']))])
        self.robot.write_joint_effort_limit_to_sim(act['gripper_effort_nm'],joint_ids=[self.ids[5]])
        self.robot.actuators['grip'].effort_limit[:]=act['gripper_effort_nm']
        for _ in range(120):self._physics_step(self.home)
        self.time=0.;self.step_index=0;self.noise=randomize
        self.playback=TeachingExecutor(self.ref,self.q())
        self.load_out_of_range_steps=0
        self.arm_load_out_of_range_steps=dict.fromkeys(JOINTS[:5],0)
        self.initial_positions={k:b.data.root_pos_w[0].cpu().numpy().copy() for k,b in self.parts.items()}
        self.settling={}
        for kind,body in self.parts.items():
            initial=self.initial_states[kind][0].numpy()
            q=body.data.root_quat_w[0].cpu().numpy()
            self.settling[kind]=dict(displacement_mm=float(np.linalg.norm(self.initial_positions[kind]-initial[:3])*1000),
                 rotation_error_deg=float(np.degrees(2*np.arccos(np.clip(abs(q@initial[3:7]),0,1)))))
        self.lifted={k:False for k in self.parts};self.peak={k:0. for k in self.parts};self.hold={k:0. for k in self.parts}
        self.peak_tracking_error=0.;self.peak_error_by_joint=np.zeros(6);self.failed=False
        return self._observation(),{'domain':self.domain}

    def _observation(self):
        q=self.q();vel=self.robot.data.joint_vel[0,self.ids].cpu().numpy().copy()
        if self.noise:
            q+=self.np_random.normal(0,np.radians(.1),6);vel+=self.np_random.normal(0,.01,6)
        nominal=self.playback.nominal
        phase=2*np.pi*self.playback.phase
        result=np.r_[q,vel,nominal,self.previous_action,np.sin(phase),np.cos(phase)].astype(np.float32)
        if not np.isfinite(result).all():raise RuntimeError('비유한 시뮬 관측')
        return result

    def step(self, action):
        if not self.carrier_matches_plan():
            raise ValueError('지그 위치가 경로 계산 후 바뀌었습니다. 현재 장면 위치로 다시 계산하세요.')
        action=np.asarray(action,float)
        if action.shape!=(6,) or not np.isfinite(action).all():raise ValueError('정책 행동 오류')
        previous=self.previous_action.copy()
        if 'assembly' in self.cfg:
            distance=float(torch.linalg.vector_norm(self.parts['housing'].data.root_pos_w-self.parts['cap'].data.root_pos_w))
            was_near=self._magnet_near;self._magnet_near=distance<.05
            if was_near and not self._magnet_near:
                for name in ('housing','cap'):
                    body=self.parts[name];zero=torch.zeros((1,1,3),device=body.device)
                    body.set_external_force_and_torque(zero,zero,is_global=True)
                self.last_magnets={'active_pairs':0,'total_attraction_n':0.}
        for _ in range(self.decimation):
            velocity=self.robot.data.joint_vel[0,self.ids].cpu().numpy()
            effort=self.robot.data.applied_torque[0,self.ids].cpu().numpy()
            actual_ticks=self.playback.ticks(self.q());applied_ticks=self.playback.ticks(self.applied_target)
            errors_ticks=[actual_ticks[n]-applied_ticks[n] for n in JOINTS]
            speeds_ticks=self.playback.velocity_ticks(self.q(),velocity)
            velocity_ticks=[speeds_ticks[n] for n in JOINTS]
            error_ticks=errors_ticks[5]
            grip_model=self.cfg['load_model']['gripper']
            arm_model=self.cfg['load_model'].get('arm')
            if arm_model:
                for i,n in enumerate(JOINTS[:5]):
                    if abs(errors_ticks[i])>arm_model['joints'][n]['error_ticks'][-1]:self.arm_load_out_of_range_steps[n]+=1
            if self.playback.gripper_closing and error_ticks>grip_model['error_ticks'][-1]:self.load_out_of_range_steps+=1
            raw=estimated_load_raw(effort,self.load_scale,gripper_error_ticks=error_ticks,gripper_model=grip_model,
                                   gripper_closing=self.playback.gripper_closing,
                                   joint_error_ticks=errors_ticks,arm_model=arm_model,joint_velocity_ticks=velocity_ticks)
            nominal,self.step_index=self.playback.advance(self.dt,self.q(),velocity,raw)
            self.time=self.playback.time
            # The actual teaching worker controls contact/arrival. Residuals
            # cannot bypass its stopped state or its retained grip position.
            residual=np.zeros(6) if self.playback.waiting or self.playback.failure else action
            self.command=bounded_target(nominal,residual,self.command,self.limits,self.dt)
            if self.playback.hold_tick is not None or self.playback.pending_contact:
                self.command[5]=nominal[5]
            # Delay is sampled at the control rate, so the same delayed command
            # persists over each 16-step physics interval.
            target=self.command if not self.domain['delay_steps'] else self.queue[0]
            self._physics_step(target)
            if self.playback.failure:
                self.failed=True
                break
        self.queue.append(self.command.copy());self.queue.popleft()
        self.previous_action=np.clip(action,-1,1)
        error=float(np.max(np.abs(self.q()-nominal)))
        self.peak_error_by_joint=np.maximum(self.peak_error_by_joint,np.abs(self.q()-nominal))
        self.peak_tracking_error=max(self.peak_tracking_error,error)
        reward=-.1*float(np.mean((self.q()-nominal)**2))-.002*float(np.mean(action**2))-.005*float(np.mean((action-previous)**2))
        errors,speeds=[],[]
        for kind,body in self.parts.items():
            row=self.cfg['parts'][kind];pos=body.data.root_pos_w[0].cpu().numpy()
            lift=float(pos[2]-self.initial_positions[kind][2]);self.peak[kind]=max(self.peak[kind],lift)
            active=row['grasp_step']<=self.step_index<row['release_step']
            if active and lift>.03:
                self.hold[kind]+=self.control_dt
                if self.hold[kind]>=.25 and not self.lifted[kind]:reward+=5.;self.lifted[kind]=True
            else:self.hold[kind]=0.
            goal=np.array(row['target_pose_mm'])[:3,3]/1000
            distance=float(np.linalg.norm(pos-goal));errors.append(distance*1000)
            speeds.append(float(torch.linalg.vector_norm(body.data.root_lin_vel_w[0])))
            if active:reward+=self.control_dt*(min(max(lift,0)/.06,1.)+float(np.exp(-distance/.04)))
            if self.step_index>=row['release_step'] and self.lifted[kind]:reward+=self.control_dt*float(np.exp(-distance/.008))
            if pos[2]<self.cfg['table_z_mm']/1000-.08:self.failed=True
        self.failed=self.failed or error>1.2
        finished=self.playback.finished and self.playback.elapsed>=self.playback.finished_at+2.
        success=False
        if 'assembly' in self.cfg:
            from assembly import assembly_state
            poses={}
            for name,body in self.parts.items():
                state=body.data.root_state_w[0].cpu().numpy();tf=np.eye(4)
                tf[:3,3]=state[:3]*1000;tf[:3,:3]=Rotation.from_quat(state[[4,5,6,3]]).as_matrix();poses[name]=tf
            self.assembly_status=assembly_state(poses,dict(zip(self.parts,speeds)),self.cfg['assembly'])
            if self.assembly_status['housing_on_pallet'] and self.lifted['housing']:
                self.assembly_timers['housing']+=self.control_dt
                if self.assembly_timers['housing']>=self.cfg['assembly']['criteria']['stage_dwell_s']:self.assembly_seen['housing']=True
            else:self.assembly_timers['housing']=0.
            if self.assembly_status['tpu_in_housing'] and self.assembly_seen['housing'] and self.lifted['tpu']:
                self.assembly_timers['tpu']+=self.control_dt
                if self.assembly_timers['tpu']>=self.cfg['assembly']['criteria']['stage_dwell_s']:self.assembly_seen['tpu']=True
            else:self.assembly_timers['tpu']=0.
            assembled=(all(self.lifted.values()) and self.assembly_seen['tpu'] and self.assembly_status['housing_on_pallet']
                       and self.assembly_status['tpu_in_housing'] and self.assembly_status['cap_aligned']
                       and self.last_magnets['active_pairs']==4 and self.playback.hold_tick is None)
            self.assembly_timers['complete']=self.assembly_timers['complete']+self.control_dt if assembled else 0.
            success=bool(finished and not self.failed and self.assembly_timers['complete']>=self.cfg['assembly']['criteria']['assembly_dwell_s'])
        if self.failed:reward-=10.
        if success:reward+=30.
        info={'is_success':success,'step_index':self.step_index,'time_s':self.time,
              'elapsed_physics_s':self.playback.elapsed,'grasp_waiting':self.playback.waiting,
              'grasp_wait_s':self.playback.wait_s,'gripper_contacts':deepcopy(self.playback.contact_events),
              'step_arrivals':deepcopy(self.playback.arrivals),'gripper_hold_tick':self.playback.hold_tick,
              'gripper_load_raw_estimate':self.playback.last_load_raw['gripper']&1023,
              'gripper_contact_load':self.playback.session.grip_contact.threshold,
              'gripper_closing_command':self.playback.gripper_closing,
              'load_model':self.cfg['load_model'],
              'gripper_load_out_of_recorded_range_steps':self.load_out_of_range_steps,
              'arm_load_out_of_recorded_range_steps':self.arm_load_out_of_range_steps.copy(),
              'failure_reason':self.playback.failure,
              'assembly':deepcopy(self.assembly_status),'assembly_stage_seen':self.assembly_seen.copy(),
              'magnets':self.last_magnets.copy(),
              'lifted':self.lifted.copy(),'peak_lift_mm':{k:v*1000 for k,v in self.peak.items()},
              'placement_errors_mm':dict(zip(self.parts,errors)),
              'peak_joint_error_deg':float(np.degrees(self.peak_tracking_error)),
              'peak_error_by_joint_deg':dict(zip(JOINTS,np.degrees(self.peak_error_by_joint).tolist())),
              'failure':self.failed,'domain':self.domain.copy()}
        info['initial_settling']=self.settling
        return self._observation(),float(reward),bool(self.failed or success),bool(finished and not success),info

    def diagnostic_state(self):
        nominal=self.playback.nominal;index=self.step_index
        return dict(time_s=self.time,step_index=index,step_name=self.ref.steps[index]['name'],
                    elapsed_physics_s=self.playback.elapsed,grasp_waiting=self.playback.waiting,
                    gripper_hold_tick=self.playback.hold_tick,
                    load_raw_estimate=self.playback.last_load_raw.copy(),
                    actual_joint_deg=np.degrees(self.q()).tolist(),reference_joint_deg=np.degrees(nominal).tolist(),
                    command_joint_deg=np.degrees(self.command).tolist(),
                    actual_tcp_mm=self.ref.kin.fk_angles(self.q())[:3,3].tolist(),
                    reference_tcp_mm=self.ref.kin.fk_angles(nominal)[:3,3].tolist(),
                    simulated_actuator_torque_nm=self.robot.data.applied_torque[0,self.ids].cpu().numpy().tolist(),
                    parts={k:dict(position_mm=(b.data.root_pos_w[0].cpu().numpy()*1000).tolist(),
                                  quaternion_wxyz=b.data.root_quat_w[0].cpu().numpy().tolist(),
                                  velocity_m_s=b.data.root_lin_vel_w[0].cpu().numpy().tolist()) for k,b in self.parts.items()})

    def close(self):
        # App lifetime belongs to the CLI; no hardware, windows or services.
        pass

    def render(self):
        from render_pose import RenderPoseBridge
        if not hasattr(self,'render_pose_bridge'):self.render_pose_bridge=RenderPoseBridge(self)
        self.render_pose_bridge.update()
        self.sim.render()
