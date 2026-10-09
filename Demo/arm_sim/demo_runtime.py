"""Isaac adapter for the explicitly assisted factory choreography."""
import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema


def transform(pose):
    x,y,yaw=pose;c=math.cos(yaw);s=math.sin(yaw)
    matrix=np.eye(4);matrix[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]];matrix[:3,3]=[x,y,.001]
    return matrix


def prepare_demo_stage(stage, cell):
    # This is a kinematic demonstration. Collision/rigid dynamics are not used
    # as evidence of grasping, driving, magnet retention or insertion success.
    remove_joints=[]
    for prim in stage.Traverse():
        path=str(prim.GetPath())
        if path.startswith(('/World/burger_','/World/waffle_')):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if prim.IsA(UsdPhysics.Joint):remove_joints.append(path)
        if path.startswith(('/World/burger_','/World/waffle_','/World/Parts/','/World/arm_')):
            if prim.HasAPI(UsdPhysics.CollisionAPI):UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    for path in remove_joints:stage.GetPrimAtPath(path).SetActive(False)
    for path in cell.bodies:
        UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(path)).CreateKinematicEnabledAttr(True)
        PhysxSchema.PhysxRigidBodyAPI(stage.GetPrimAtPath(path)).CreateEnableCCDAttr(False)
    # Independent shipping pallets replace the shelf preview containers.
    for i in (0,1) if cell.cfg['warehouse'].get('moving_trays') else (1,):
        UsdGeom.Imageable(stage.GetPrimAtPath(f'/World/Factory/Warehouse/Tray{i}')).MakeInvisible()


class DemoRuntime:
    def __init__(self, stage, cell, robots, plan,*,runtime_geometry=True):
        self.stage=stage;self.cell=cell;self.robots=robots;self.plan=plan
        self.ids={name:robot.find_joints(plan.ik[name].model['joints'],preserve_order=True)[0] for name,robot in robots.items()}
        self.mobile_ops={}
        for name in plan.vehicles:
            xf=UsdGeom.Xformable(stage.GetPrimAtPath('/World/'+name))
            xf.ClearXformOpOrder()
            self.mobile_ops[name]=xf.AddTransformOp(opSuffix='demo')
        self.lift_ops={}
        for name in ('Fork0','Fork1','Carriage','Pallet','Tray'):
            xf=UsdGeom.Xformable(stage.GetPrimAtPath('/World/waffle_1/Lift/'+name))
            ops=xf.GetOrderedXformOps()
            op=xf.AddTransformOp(opSuffix='demo')
            xf.SetXformOpOrder([op,*ops]);self.lift_ops[name]=op
        self.tray_paths=['/World/waffle_1/Lift']
        self.tray_ops=[{n:self.lift_ops[n] for n in ('Pallet','Tray')}]
        for i in range(1,len(plan.trays)):
            root=f'/World/waffle_1/Lift/Shipping{i}'
            cell.attach_shipping_pallet(root)
            for prim in stage.Traverse():
                if str(prim.GetPath()).startswith(root+'/') and prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            pair={}
            for n in ('Pallet','Tray'):
                xf=UsdGeom.Xformable(stage.GetPrimAtPath(root+'/'+n));ops=xf.GetOrderedXformOps()
                op=xf.AddTransformOp(opSuffix='demo');xf.SetXformOpOrder([op,*ops]);pair[n]=op
            self.tray_ops.append(pair);self.tray_paths.append(root)
        self.part_names=[path.rsplit('/',1)[-1] for path in cell.view.prim_paths]
        self.quaternions=torch.tensor(np.tile([1.,0.,0.,0.],(len(self.part_names),1)),dtype=torch.float32,device=cell.view._device)
        self.select_plan(plan,runtime_geometry=runtime_geometry)

    def select_plan(self,plan,*,runtime_geometry=None):
        """Switch choreography on the same stage, then require a frame-0 reset."""
        if plan.cfg!=self.plan.cfg:raise ValueError('Start modes must share the same scene')
        if runtime_geometry is None:runtime_geometry=self.geometry is not None
        from arm_sim.demo_geometry import GeometryGuard
        from arm_sim.demo_coordination import WorkcellGate
        from arm_sim.traffic import TrafficGate
        self.geometry=GeometryGuard(plan) if runtime_geometry else None
        self.gate=WorkcellGate(plan)
        self.traffic_gate=TrafficGate(plan)
        self.plan=plan
        self.last=None;self.max_part_error=0.;self.max_joint_error=0.;self.max_tool_error=0.;self.max_tray_error=0.
        self.max_relative_ratio=0.;self.retimed_steps=0

    def geometry_status(self):
        return {'geometry_enabled':self.geometry is not None,
                'geometry_checks':self.geometry.checked if self.geometry is not None else 0,
                'traffic_checks':self.geometry.traffic.checked if self.geometry is not None else 0}

    def measured(self):
        return {name:robot.data.joint_pos[0,self.ids[name]].cpu().numpy().copy()
                for name,robot in self.robots.items()}

    def advance(self,t,dt):
        measured=self.measured()
        result=self.plan.advance(t,dt,measured)
        result=self.gate.advance(t,result,measured)
        mobile={}
        for name,op in self.mobile_ops.items():
            value=op.Get()
            if value is not None:
                matrix=np.asarray(value).T
                mobile[name]=np.array([matrix[0,3],matrix[1,3],math.atan2(matrix[1,0],matrix[0,0])])
        result=self.traffic_gate.advance(t,result,mobile)
        if result < min(self.plan.duration,t+dt)-1e-7:self.retimed_steps+=1
        return result

    def apply(self,state,reset=False):
        if reset:self.gate.reset();self.traffic_gate.reset()
        if self.geometry is not None:
            self.geometry.check(state['joints'],reset=reset,time=state['time'])
        if not reset:
            measured=self.measured()
            for name,limits in self.plan.relative_limits.items():
                ratio=float(np.max(np.abs(state['joints'][name]-measured[name])/limits))
                if not math.isfinite(ratio) or ratio>1.+1e-5:
                    raise RuntimeError(f'Demo relative target exceeded: {name} {ratio}')
                self.max_relative_ratio=max(self.max_relative_ratio,ratio)
        for name,robot in self.robots.items():
            q=robot.data.default_joint_pos.clone()
            q[0,self.ids[name]]=torch.tensor(state['joints'][name],dtype=q.dtype,device=q.device)
            robot.write_joint_state_to_sim(q,torch.zeros_like(q));robot.set_joint_position_target(q)
        for name,op in self.mobile_ops.items():
            op.Set(Gf.Matrix4d(transform(state['vehicles'][name]).T.tolist()))
        lift=np.eye(4);lift[2,3]=state['lift']
        for name in ('Fork0','Fork1','Carriage'):
            self.lift_ops[name].Set(Gf.Matrix4d(lift.T.tolist()))
        for entry,ops in zip(state['trays'],self.tray_ops):
            tray=lift
            if entry[1]=='world':
                value=entry[2]
                frozen=transform(value[:3]);frozen[2,3]+=value[3]
                tray=np.linalg.inv(transform(state['vehicles']['waffle_1']))@frozen
            for op in ops.values():op.Set(Gf.Matrix4d(tray.T.tolist()))
        poses=torch.tensor(np.array([state['parts'][name] for name in self.part_names]),dtype=torch.float32,device=self.cell.view._device)
        xyzw=Rotation.from_matrix(np.array([state.get('part_rotations',{}).get(name,np.eye(3)) for name in self.part_names])).as_quat()
        quaternions=torch.tensor(xyzw[:,[3,0,1,2]],dtype=torch.float32,device=self.cell.view._device)
        self.cell.view.set_world_poses(poses,quaternions)
        self.last=state

    def verify(self):
        cache=UsdGeom.XformCache()
        for root,entry in zip(self.tray_paths,self.last['trays']):
            pose=self.last['vehicles']['waffle_1'] if entry[1]=='lift' else entry[2][:3]
            height=self.last['lift'] if entry[1]=='lift' else entry[2][3]
            world=transform(pose);world[2,3]+=height
            c=self.plan.cfg['waffle']
            for leaf,z in [('Pallet',c['fork_top']+.004),('Tray/Carrier',c['fork_top']+.009)]:
                expected=(world@np.array([.15+c['fork_length']/2,0,z,1.]))[:3]
                actual=np.asarray(cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(root+'/'+leaf)).ExtractTranslation())
                error=float(np.linalg.norm(actual-expected));self.max_tray_error=max(self.max_tray_error,error)
                if not np.isfinite(error) or error>.00001:raise RuntimeError(f'Demo tray transform mismatch: {root}/{leaf} {error} m')
        actual=self.cell.snapshot()
        error=max(np.linalg.norm(np.array(p[:3])-self.last['parts'][path.rsplit('/',1)[-1]]) for path,p in actual.items())
        if not np.isfinite(error) or error>.004:raise RuntimeError(f'Demo payload tracking error: {error} m')
        self.max_part_error=max(self.max_part_error,float(error))
        for name,robot in self.robots.items():
            q=robot.data.joint_pos[0,self.ids[name]].cpu().numpy()
            error=float(np.max(np.abs(q-self.last['joints'][name])))
            if not np.isfinite(error) or error>.05:raise RuntimeError(f'Demo joint tracking error: {name} {error}, actual={q.tolist()}, target={self.last["joints"][name].tolist()}, limits={robot.data.joint_pos_limits.tolist()}')
            self.max_joint_error=max(self.max_joint_error,error)

            link='gripper_frame_link' if self.plan.ik[name].so else 'end_effector_link'
            actual=robot.data.body_pos_w[0,robot.body_names.index(link)].cpu().numpy()
            tool_error=float(np.linalg.norm(actual-self.plan.ik[name].position(q)))
            if not np.isfinite(tool_error) or tool_error>.005:raise RuntimeError(f'Demo FK mismatch: {name} {tool_error} m')
            self.max_tool_error=max(self.max_tool_error,tool_error)
