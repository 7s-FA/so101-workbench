"""Finite Isaac Sim renders of the camera assembly; no server or hardware IO."""
import sys,json,math
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT.parent/'Sim_real')]
from isaaclab.app import AppLauncher
import argparse
p=argparse.ArgumentParser();p.add_argument('--only-view',default='all');AppLauncher.add_app_launcher_args(p);p.set_defaults(headless=True,device='cpu',enable_cameras=True)
a=p.parse_args();a.kit_args='--/log/outputStreamLevel=error --/log/fileLogLevel=info'
launcher=AppLauncher(a);app=launcher.app
import numpy as np,torch
from PIL import Image
from scipy.spatial.transform import Rotation
from pxr import UsdGeom,UsdPhysics,PhysxSchema
import isaaclab.sim as su
from isaaclab.assets import Articulation,ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.converters import UrdfConverter,UrdfConverterCfg
from isaacsim.core.utils.extensions import enable_extension
enable_extension('isaacsim.sensors.camera')
from isaacsim.sensors.camera import Camera
from arm_sim.conveyor_scene import ConveyorCell
from studio.robot import NAMES
cfg=json.loads((ROOT/'config/scene.json').read_text());assembly=json.loads(Path(cfg['camera_assembly']).read_text())
out=ROOT/'outputs/camera_mount_preview';out.mkdir(parents=True,exist_ok=True)
class Cell(ConveyorCell):
 def build(self):
  x,y=cfg['table']['center'];w,d=cfg['table']['size'];z=cfg['table']['height']
  self.box('/World/Table',(x,y,z-.015),(w,d,.03),self.wood[3],False)
  for i,(a,b) in enumerate(((-1,-1),(-1,1),(1,-1),(1,1))):self.box('/World/Leg'+str(i),(x+a*(w/2-.03),y+b*(d/2-.03),(z-.03)/2),(.035,.035,z-.03),self.wood[1],False)
  self.box('/World/Floor',(0,0,-.015),(1.5,1.5,.03),self.material('floor',(.08,.09,.10)),False)
def orient(camera,eye,target,up=(0,0,1)):
 back=np.asarray(eye)-target;back/=np.linalg.norm(back);right=np.cross(up,back);right/=np.linalg.norm(right)
 quat=Rotation.from_matrix(np.column_stack([right,np.cross(back,right),back])).as_quat()
 camera.set_world_pose(eye,quat[[3,0,1,2]],camera_axes='usd')
 camera.set_clipping_range(.002,10);UsdGeom.Camera(camera.prim).CreateFocalLengthAttr(30.)
try:
 sim=su.SimulationContext(su.SimulationCfg(dt=1/60,device='cpu'));sim._disable_app_control_on_stop_handle=True
 light=su.DomeLightCfg(intensity=1300);light.func('/World/Light',light)
 cell=Cell(sim.stage,{'physics':{'friction':.5,'contact_offset':.0005}})
 conv=UrdfConverter(UrdfConverterCfg(asset_path=cfg['model_source'],usd_dir=str(out/'usd'),usd_file_name='so101.usd',fix_base=True,
  merge_fixed_joints=True,self_collision=False,make_instanceable=False,collision_from_visuals=False,
  joint_drive=UrdfConverterCfg.JointDriveCfg(drive_type='force',target_type='position',gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.,damping=0.))))
 home=np.radians(cfg['start_angles_deg']).tolist();robot=Articulation(ArticulationCfg(prim_path='/World/SO101',spawn=su.UsdFileCfg(usd_path=conv.usd_path),
  init_state=ArticulationCfg.InitialStateCfg(pos=tuple(cfg['base']),joint_pos=dict(zip(NAMES,home))),
  actuators={'display':ImplicitActuatorCfg(joint_names_expr=['.*'],stiffness=0.,damping=0.)}))
 for _ in range(4):
  instances=[v for v in sim.stage.Traverse() if v.IsInstance()]
  if not instances:break
  for v in instances:v.SetInstanceable(False)
 black=cell.material('robot_black',(.035,.04,.045));blue=cell.material('tpu',(.025,.3,.72));silver=cell.material('silver',(.6,.63,.66));adapter=cell.material('new_adapters',(.055,.28,.29))
 for prim in sim.stage.Traverse():
  path=str(prim.GetPath())
  if not path.startswith('/World/SO101/'):continue
  if prim.HasAPI(UsdPhysics.RigidBodyAPI):PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)
  if prim.HasAPI(UsdPhysics.CollisionAPI):UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
  if '/visuals/' in path and prim.IsA(UsdGeom.Mesh):
   mat=adapter if any(v in path for v in ('overhead_top_d435','wrist_xlerobot_mount')) else silver if any(v in path for v in ('overhead_d435','overhead_screw','overhead_washer')) else blue if 'xlerobot' in path and 'finger' in path else black
   cell.bind(prim,mat)
 # FK from exact URDF origins; all positions in metres here.
 root=ET.parse(cfg['model_source']).getroot();links={'base_link':np.eye(4)};links['base_link'][:3,3]=cfg['base'];joints=list(root.findall('joint'));q=dict(zip(NAMES,home))
 while joints:
  advanced=False
  for j in joints[:]:
   parent=j.find('parent').get('link')
   if parent not in links:continue
   e=j.find('origin');tf=np.eye(4)
   if e is not None:tf[:3,3]=list(map(float,e.get('xyz','0 0 0').split()));tf[:3,:3]=Rotation.from_euler('xyz',list(map(float,e.get('rpy','0 0 0').split()))).as_matrix()
   if j.get('type')=='revolute':tf[:3,:3]=tf[:3,:3]@Rotation.from_rotvec(np.array(list(map(float,j.find('axis').get('xyz').split())))*q[j.get('name')]).as_matrix()
   links[j.find('child').get('link')]=links[parent]@tf;joints.remove(j);advanced=True
  if not advanced:raise ValueError('URDF tree cycle')
 top=np.array(assembly['new_rgb_center_base_mm'])/1000+cfg['base'];forward=np.array(assembly['optical_axis_base'])
 wrist=np.array(assembly['wrist_camera_transform_mm']);wrist[:3,3]/=1000;wrist=links['gripper_link']@wrist
 target=wrist[:3,3];normal=wrist[:3,2]
 views=[('overview',np.array([1.,-1.35,1.04]),np.array([.10,-.065,.51])),
  ('overhead_mount',top+np.array([.20,-.25,.11]),top+np.array([-.025,-.025,.00])),
  ('overhead_front',top+forward*.23+np.array([0,-.13,.025]),top+np.array([-.01,-.025,.01])),
  ('wrist_mount',(links['gripper_link']@np.array([-.26,.20,.04,1]))[:3],(links['gripper_link']@np.array([-.01,.025,-.04,1]))[:3])]
 cameras=[]
 for name,eye,target in views:
  if a.only_view not in ('all',name):continue
  c=Camera('/World/Views/'+name,resolution=(1400,1100));c.initialize();orient(c,eye,target);cameras.append((name,c))
 sim.reset();ids,_=robot.find_joints(NAMES,preserve_order=True);tensor=torch.tensor([home],dtype=torch.float32)
 if a.only_view=='wrist_mount':
  for prim in sim.stage.Traverse():
   path=str(prim.GetPath())
   if '/base_link/visuals/' in path and any(n in path for n in ('cam_mount','overhead_','arm_base')):
    if prim.IsA(UsdGeom.Mesh):UsdGeom.Imageable(prim).MakeInvisible()
 for _ in range(18):
  robot.write_joint_state_to_sim(tensor,torch.zeros_like(tensor),joint_ids=ids);sim.render()
 for name,c in cameras:
  pixels=np.asarray(c.get_rgba(),dtype=np.uint8)
  if not pixels.size or not pixels[...,:3].any():raise RuntimeError(name+' rendering failed')
  Image.fromarray(pixels).convert('RGB').save(out/(name+'.png'))
 (out/'error.log').unlink(missing_ok=True)
 print('CAMERA_PREVIEWS',out,flush=True)
except BaseException:
 import traceback
 text=traceback.format_exc();(out/'error.log').write_text(text);print(text,file=sys.stderr,flush=True)
 raise
finally:
 # Finite renderer has no hardware IO; bound Kit cleanup after artifact files close.
 import threading,os
 exitcode=1 if (out/'error.log').exists() else 0
 deadline=threading.Timer(8.,lambda:os._exit(exitcode));deadline.daemon=True;deadline.start()
 app.close(wait_for_replicator=False)
