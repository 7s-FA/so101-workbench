"""Make a derived cap mesh with the user-confirmed mating face down."""
import json,struct
import numpy as np,fcl
from arm_sim.common import ROOT
from arm_sim.grasp_geometry import stl_triangles,triangle_model,collision_object
folder=ROOT/'assets/generated/conveyor_demo/parts';original=folder/'03_cap_10mm_original_ball_exposure.stl';target=folder/'03_cap_10mm_mating_face_down.stl'
data=bytearray(original.read_bytes());count=struct.unpack_from('<I',data,80)[0];v=stl_triangles(original);height=float(v[:,2].max()+v[:,2].min());rot=np.diag([1.,-1.,-1.])
for i in range(count):
 values=np.array(struct.unpack_from('<12f',data,84+50*i));values[:3]=rot@values[:3]
 vv=values[3:].reshape(3,3)@rot.T;vv[:,2]+=height;values[3:]=vv.ravel();struct.pack_into('<12f',data,84+50*i,*values)
target.write_bytes(data)
cfg=json.loads((ROOT/'config/conveyor_demo.json').read_text());cfg['parts']['cap']=target.name
models={kind:triangle_model(stl_triangles(ROOT/'assets/product/02_ball_d30.stl' if kind=='ball' else folder/cfg['parts'][kind])*.001) for kind in ('housing','ball','cap','cap_jig')}
def first_contact(moving,xy,obstacles):
 def collides(z):
  obj=collision_object(models[moving],np.eye(3),[*xy,z])
  return any(fcl.collide(obj,other,fcl.CollisionRequest(),fcl.CollisionResult()) for other in obstacles)
 clear=.08
 for z in np.arange(.0799,-.030,-.0001):
  if collides(z):
   lo,hi=z,clear
   for _ in range(24):
    mid=(lo+hi)/2
    if collides(mid):lo=mid
    else:hi=mid
   return float(hi)
  clear=z
 raise ValueError('No mating contact found')
jig=collision_object(models['cap_jig'],np.eye(3),np.zeros(3));seating=first_contact('cap',cfg['jig_seating']['offset_xy'],[jig])
housing=collision_object(models['housing'],np.eye(3),np.zeros(3));ball=collision_object(models['ball'],np.eye(3),[0,0,.003053]);assembly=first_contact('cap',[0,0],[housing,ball])
cfg['jig_seating']['contact_height']['cap']=seating
cfg['cap_orientation']={'source':original.name,'derived':target.name,'rotation_xyz_degrees':[180,0,0],'z_translation_mm':height,'supply':'mating_face_down','four_holes':'future_magnets','magnetic_force_calibrated':False,'assembly_base_z_m':assembly+.00005}
(ROOT/'config/conveyor_demo.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'cap_bounds_mm':[v.min(0).tolist(),v.max(0).tolist()],'jig_seating_m':seating,'assembly_z_m':assembly+.00005}))
