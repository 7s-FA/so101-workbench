#!/usr/bin/env python3
"""Isaac Sim measured-pose viewer. No hardware writes and no invented object tracking."""
import argparse
import io
import json
import os
import signal
import sys
import time
import urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
TEACH = ROOT.parent/'Test_teach'
sys.path[:0] = [str(ROOT),str(TEACH),str(ROOT.parent/'Sim_real')]
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument('--server',required=True)
parser.add_argument('--smoke-seconds',type=float,default=0)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device='cpu',enable_cameras=True)
args = parser.parse_args()
args.kit_args = '--/log/outputStreamLevel=error --/log/fileLogLevel=info'
launcher = AppLauncher(args)
app = launcher.app
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from pxr import UsdGeom, UsdShade, UsdPhysics, PhysxSchema
import isaaclab.sim as su
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaacsim.core.utils.extensions import enable_extension
enable_extension('isaacsim.sensors.camera')
from isaacsim.sensors.camera import Camera
from arm_sim.conveyor_scene import ConveyorCell
from arm_sim.factory import table_legs
from arm_sim.conveyor_hardware import clamp_boxes
from studio.store import atomic_json
from studio.robot import NAMES
from studio.safe_pose import load_safe_pose

cfg = json.loads((ROOT/'config/scene.json').read_text())
out = ROOT/'outputs'
out.mkdir(exist_ok=True)
stopping = False
def stop(*_):
    global stopping
    stopping = True
signal.signal(signal.SIGINT,stop)
signal.signal(signal.SIGTERM,stop)


class Cell(ConveyorCell):
    def build(self):
        z = cfg['table']['height']
        x,y = cfg['table']['center']
        w,d = cfg['table']['size']
        self.box('/World/Table/Top',(x,y,z-.015),(w,d,.03),self.wood[3],False)
        for i,(p,s) in enumerate(table_legs(cfg['table'],z)):
            self.box('/World/Table/Leg'+str(i),p,s,self.wood[1],False)
        self.box('/World/Floor',(0,0,-.015),(1.2,1.2,.03),self.material('floor',(.08,.09,.10)),False)




try:
    atomic_json(out/'sim_status.json',dict(state='loading',time=time.time()))
    sim = su.SimulationContext(su.SimulationCfg(dt=1/60,device='cpu'))
    sim._disable_app_control_on_stop_handle = True
    su.DomeLightCfg(intensity=1100).func('/World/Light',su.DomeLightCfg(intensity=1100))
    cell = Cell(sim.stage,{'physics':{'friction':.5,'contact_offset':.0005}})
    def make_robot(kind,scene):
        conv=UrdfConverter(UrdfConverterCfg(asset_path=scene['model_source'],usd_dir=str(ROOT/'outputs'/('viewer_'+kind)),usd_file_name=kind+'.usd',
            fix_base=True,merge_fixed_joints=True,self_collision=False,convert_mimic_joints_to_normal_joints=True,
            make_instanceable=False,collision_from_visuals=False,
            joint_drive=UrdfConverterCfg.JointDriveCfg(drive_type='force',target_type='position',gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.,damping=0.))))
        names=NAMES if kind=='so101' else ['joint1','joint2','joint3','joint4','joint5','gripper_joint_1','gripper_joint_2']
        q=load_safe_pose()['model_angles'] if kind=='so101' else np.radians(scene['start_angles_deg']).tolist()
        if kind=='omx':q+=[-q[5]]
        path='/World/'+kind.upper()
        robot=Articulation(ArticulationCfg(prim_path=path,spawn=su.UsdFileCfg(usd_path=conv.usd_path),
            init_state=ArticulationCfg.InitialStateCfg(pos=tuple(scene['base']),joint_pos=dict(zip(names,q))),
            actuators={'display':ImplicitActuatorCfg(joint_names_expr=['.*'],stiffness=0.,damping=0.)}))
        for _ in range(4):
            instances=[p for p in sim.stage.Traverse() if str(p.GetPath()).startswith(path) and p.IsInstance()]
            if not instances:break
            for p in instances:p.SetInstanceable(False)
        black=cell.material('robot_black',(.025,.03,.035));blue=cell.material('tpu_blue',(.025,.30,.72));silver=cell.material('camera_silver',(.55,.57,.59))
        for prim in sim.stage.Traverse():
            p=str(prim.GetPath())
            if p.startswith(path+'/'):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)
                if prim.HasAPI(UsdPhysics.CollisionAPI):UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
                if '/visuals/' in p and prim.IsA(UsdGeom.Mesh):
                    mat=blue if 'xlerobot' in p and 'finger' in p else silver if any(v in p for v in ('overhead_d435','ScrewHead','Washer')) else black
                    cell.bind(prim,mat)
        return robot,names,path,q
    robots={kind:make_robot(kind,scene) for kind,scene in [('so101',cfg),('omx',json.loads((ROOT/'config/legacy_omx/scene.json').read_text()))]}
    camera = Camera('/World/View',resolution=(960,640))
    camera.initialize()
    eye = np.array([1.1,-1.2,1.05]);target = np.array([.14,-.06,.53])
    back = eye-target;back /= np.linalg.norm(back)
    right = np.cross([0,0,1],back);right /= np.linalg.norm(right)
    quat = Rotation.from_matrix(np.column_stack((right,np.cross(back,right),back))).as_quat()
    camera.set_world_pose(eye,quat[[3,0,1,2]],camera_axes='usd')
    camera.set_clipping_range(.005,10)
    UsdGeom.Camera(camera.prim).CreateFocalLengthAttr(22.)
    sim.set_camera_view(eye=eye.tolist(),target=target.tolist())
    sim.reset()
    joint_ids={kind:robot.find_joints(names,preserve_order=True)[0] for kind,(robot,names,_,_) in robots.items()}
    began = time.monotonic();frames = 0
    while app.is_running() and not stopping:
        tick = time.monotonic()
        if args.smoke_seconds and tick-began>args.smoke_seconds:break
        with urllib.request.urlopen(args.server+'/api/sim-state',timeout=1) as response:
            state = json.load(response)
        q=state['display_angles'];kind=state.get('display_robot','so101')
        for key,(robot,names,path,home) in robots.items():
            visible=key==kind
            imageable=UsdGeom.Imageable(sim.stage.GetPrimAtPath(path))
            imageable.MakeVisible() if visible else imageable.MakeInvisible()
            values=(list(q)+([-q[5]] if key=='omx' else [])) if visible else home
            tensor=torch.tensor([values],dtype=torch.float32)
            robot.write_joint_state_to_sim(tensor,torch.zeros_like(tensor),joint_ids=joint_ids[key])
        # Display measured poses directly, with no physics drift/contact authority.
        sim.render()
        rgba = np.asarray(camera.get_rgba(),dtype=np.uint8)
        if rgba.size and rgba[...,:3].any():
            buffer = io.BytesIO()
            Image.fromarray(rgba).convert('RGB').save(buffer,format='JPEG',quality=85)
            temp = out/'isaac.jpg.tmp';temp.write_bytes(buffer.getvalue());temp.replace(out/'isaac.jpg')
            frames += 1
            atomic_json(out/'sim_status.json',dict(state='ready',time=time.time(),frames=frames,
                source=state['display_source'],robot_type=kind,joint_angles=q,object_tracking=False))
        time.sleep(max(0,1/15-(time.monotonic()-tick)))
    atomic_json(out/'sim_status.json',dict(state='stopped',time=time.time(),frames=frames))
except Exception as error:
    atomic_json(out/'sim_status.json',dict(state='error',error=str(error),time=time.time()))
    raise
finally:
    app.close()
