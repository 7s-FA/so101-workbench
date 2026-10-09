from mesh_io import read
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
import xml.etree.ElementTree as ET
import json
r=Path(__file__).resolve().parent
source=r/'source'
model=r.parents[1]/'assets'/'so101'
# Express original STL assembly in robot-aligned forward/left/up coordinates.
A=np.array([[1.,0,0],[0,0,-1],[0,1,0]])
arm_t=np.array([93.92092,5.,-26.08911])
middle_t=np.array([18.7,188.1,36.512])
top_t=middle_t.copy() # upper and middle already share their exported longitudinal datum
tri,n=read(source/'cam_mount_top.stl');v=np.unique(tri.reshape(-1,3),axis=0)
a=np.radians(25);up=np.array([-np.sin(a),np.cos(a),0]);along=np.array([np.cos(a),np.sin(a),0])
sel=(v[:,1]>350)&(abs(v@up-318.8325)<.006);p=v[sel]
center=p.mean(0);_,_,vh=np.linalg.svd(p-center);plane_normal=vh[-1];plane_normal*=np.sign(plane_normal@up)
uv=np.column_stack((p@along,p[:,2]));hole_centers=[];circle_residual=[]
for guess in [(166.4,13.5),(166.4,-13.5),(193.4,13.5),(193.4,-13.5)]:
 q=uv[np.linalg.norm(uv-guess,axis=1)<1.6]
 fit=least_squares(lambda c:np.linalg.norm(q-c[:2],axis=1)-c[2],[*guess,1.])
 hole_centers.append(fit.x[:2]);circle_residual.append(float(np.max(abs(fit.fun))))
uvc=np.mean(hole_centers,axis=0)
board_center_raw=uvc[0]*along+uvc[1]*np.array([0,0,1])
board_center_raw+=up*((plane_normal@center-plane_normal@board_center_raw)/(plane_normal@up))
board_center=A@(board_center_raw+top_t)
optical_axis=A@(-plane_normal)
# Four mounting-hole centers of SO101's STL and their matching locating holes.
urdf=ET.parse(model/'so101_modified.urdf')
base_visual=next(v for v in urdf.find("link[@name='base_link']").findall('visual') if v.find('geometry/mesh').get('filename').endswith('base_so101_v2.stl'))
origin=base_visual.find('origin')
rotation=Rotation.from_euler('xyz',[float(x) for x in origin.get('rpy').split()]).as_matrix()
position=np.array([float(x) for x in origin.get('xyz').split()])
robot_base,_=read(model/base_visual.find('geometry/mesh').get('filename'))
robot_base=(robot_base@rotation.T+position)*1000
v=robot_base.reshape(-1,3)
flat=np.unique(v[abs(v[:,2]+2.4)<.001,:2],axis=0)
# Rear through-holes have a teardrop relief; fit only the actual circular flank.
robot_holes=[]
for guess in [(-13.844,31.75),(-13.844,-31.75),(55.9103,27.7765),(55.9103,-27.7765)]:
 q=flat[np.linalg.norm(flat-guess,axis=1)<3]
 if guess[0]<0:q=q[q[:,0]<-15.0]
 fit=least_squares(lambda c:np.linalg.norm(q-c[:2],axis=1)-c[2],[*guess,2.5])
 robot_holes.append(fit.x[:2])
arm_tri,_=read(source/'arm_base.stl');v=arm_tri.reshape(-1,3);flat=np.unique(v[abs(v[:,1]+5)<.001][:,[0,2]],axis=0)
arm_holes=[]
for guess in [(-86.8,-85.18),(-86.8,-21.68),(-17.046,-81.207),(-17.046,-25.654)]:
 q=flat[np.linalg.norm(flat-guess,axis=1)<3.5]
 fit=least_squares(lambda c:np.linalg.norm(q-c[:2],axis=1)-c[2],[*guess,2.5])
 arm_holes.append([fit.x[0]+arm_t[0],-(fit.x[1]+arm_t[2])])
robot_holes=np.array(robot_holes);arm_holes=np.array(arm_holes)
translation_xy=np.mean(arm_holes-robot_holes,axis=0)
residual=np.linalg.norm(arm_holes-(robot_holes+translation_xy),axis=1)
base=np.r_[translation_xy,7.4] # 5 mm support plate + base_link 2.4 mm above sole

j1=np.array([float(v) for v in urdf.find("joint[@name='shoulder_pan']/origin").get('xyz').split()])*1000
j1_world=base+j1
result={'units':'mm','frame':'X forward (toward work area), Y left, Z up; origin = raw bottom STL origin on support table',
 'left_robot':True,'assembly_translations_original_stl':{'arm_base':arm_t.tolist(),'cam_mount_bottom':[0.,0.,0.],'cam_mount_middle':middle_t.tolist(),'cam_mount_top':top_t.tolist()},
 'assembly_to_forward_left_up':A.tolist(),'robot_base_link_in_assembly':base.tolist(),'shoulder_pan_joint_origin_in_assembly':j1_world.tolist(),
 'camera_board_seating_center_in_assembly':board_center.tolist(),'camera_board_center_in_base_link':(board_center-base).tolist(),
 'camera_board_center_relative_to_j1_origin':(board_center-j1_world).tolist(),'nominal_optical_axis_in_robot_axes':optical_axis.tolist(),
 'camera_tilt_from_vertical_deg':float(np.degrees(np.arccos(-optical_axis[2]))),'board_hole_centers_in_plane':np.array(hole_centers).tolist(),
 'board_hole_fit_max_residual_mm':max(circle_residual),'robot_base_hole_registration_residual_mm':residual.tolist(),
 'mount_top_height_above_table_mm':float((tri.reshape(-1,3)[:,1]+top_t[1]).max()),
 'verified_for_robot_motion':False,'camera_optical_transform':None,
 'notes':['Camera optical/projection center offset from PCB seating plane is unknown.', 'Optical axis assumes module lens axis is perpendicular to PCB.', 'Camera image rotation around the optical axis is unknown.', 'Robot assembly coordinates are nominal CAD alignment, not measured calibration.', 'Frame origin and robot base_link differ from the shoulder_pan rotation-axis origin.']}
assert np.max(residual)<.05, 'Base-hole alignment does not match current robot model'
assert max(circle_residual)<.01, 'Camera mounting-hole fit is inconsistent'
assert np.linalg.det(A)>0.999
result['camera_board_center_from_j1_ground']=(board_center-np.r_[j1_world[:2],0.]).tolist()
result['camera_lens_axis_table_intersection_from_j1_ground']=(board_center-optical_axis*board_center[2]/optical_axis[2]-np.r_[j1_world[:2],0.]).tolist()
(r/'nominal_pose.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
