"""Render the real STL assembly and URDF reference pose without opening hardware."""
from pathlib import Path
import xml.etree.ElementTree as ET
import json
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PolyCollection
from mesh_io import read

r=Path(__file__).resolve().parent
model=r.parents[1]/'assets'/'so101'
pose=json.loads((r/'nominal_pose.json').read_text())
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':11,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
A=np.array(pose['assembly_to_forward_left_up'])
base=np.array(pose['robot_base_link_in_assembly']);j1=np.array(pose['shoulder_pan_joint_origin_in_assembly'])
reference=np.r_[j1[:2],0.]
# Drawing axes: right, forward, up. The reference is J1's projection onto the table.
D=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
def drawing(p):return (p-reference)@D.T
parts=[]
colors=['#527787','#426271','#4d7181','#008c9a']
for (name,t),color in zip(pose['assembly_translations_original_stl'].items(),colors):
 tri,_=read(r/'source'/(name+'.stl'))
 parts.append((name,drawing((tri+np.array(t))@A.T),color))
root=ET.parse(model/'so101_modified.urdf').getroot()
links={link.get('name'):link for link in root.findall('link')}
transforms={'base_link':np.eye(4)}
def transform(origin):
 m=np.eye(4)
 if origin is not None:
  m[:3,:3]=Rotation.from_euler('xyz',[float(v) for v in origin.get('rpy','0 0 0').split()]).as_matrix()
  m[:3,3]=[float(v) for v in origin.get('xyz','0 0 0').split()]
 return m
remaining=list(root.findall('joint'))
while remaining:
 for joint in remaining[:]:
  parent=joint.find('parent').get('link');child=joint.find('child').get('link')
  if parent not in transforms:continue
  transforms[child]=transforms[parent]@transform(joint.find('origin'))
  remaining.remove(joint)
for name,link in links.items():
 for visual in link.findall('visual'):
  mesh=visual.find('geometry/mesh')
  if mesh is None:continue
  tri,_=read(model/mesh.get('filename'))
  tri*=np.array([float(v) for v in mesh.get('scale','1 1 1').split()])
  tf=transforms[name]@transform(visual.find('origin'))
  tri=(tri@tf[:3,:3].T+tf[:3,3])*1000+base
  color='#118fd0' if 'finger' in mesh.get('filename') else '#88949b'
  parts.append((name,drawing(tri),color))
assert all(np.max(np.abs(tri))<1000 for _,tri,_ in parts), 'Check URDF/STL mesh units and scale'
cam=drawing(np.array(pose['camera_board_seating_center_in_assembly']))
optic=np.array(pose['nominal_optical_axis_in_robot_axes'])@D.T
end=cam-optic*cam[2]/optic[2]
fig=plt.figure(figsize=(16,10.5),layout='constrained',facecolor='#f9fafb')
gs=fig.add_gridspec(2,3,width_ratios=[1.35,1,1])
ax3=fig.add_subplot(gs[:,0],projection='3d')
front=fig.add_subplot(gs[:,1]);side=fig.add_subplot(gs[0,2]);top=fig.add_subplot(gs[1,2])
for name,tri,color in parts:
 # Every triangle is used; the arm is a kinematic reference pose, not a measured pose.
 ax3.add_collection3d(Poly3DCollection(tri,facecolors=color,linewidth=0,alpha=1,shade=True))
 for ax,dims in [(front,[0,2]),(side,[1,2]),(top,[0,1])]:
  ax.add_collection(PolyCollection(tri[:,:,dims],facecolors=color,edgecolors='none',alpha=.58 if name.endswith('_link') else .82))
ax3.plot([cam[0],end[0]],[cam[1],end[1]],[cam[2],end[2]],'--',color='#bc6513',lw=2)
ax3.scatter(*cam,s=45,color='#d66214');ax3.scatter(0,0,0,s=35,color='#ad3040')
ax3.set(xlim=(-100,170),ylim=(-100,400),zlim=(0,585),xlabel='오른쪽 R (mm)',ylabel='앞쪽 F (mm)',zlabel='높이 H (mm)',title='STL 조립 배치 · 왼쪽 팔 / 오른쪽 카메라')
ax3.set_box_aspect((270,500,585));ax3.view_init(21,-70)
front.set(xlim=(-105,165),ylim=(-12,592),xlabel='오른쪽 R (mm)',ylabel='받침 바닥에서 높이 H (mm)',title='정면 · 카메라 장착면 높이')
side.set(xlim=(-85,400),ylim=(-10,585),xlabel='앞쪽 F (mm)',ylabel='H (mm)',title='측면 · 광축은 수직에서 약 25°')
top.set(xlim=(-105,165),ylim=(-65,400),xlabel='오른쪽 R (mm)',ylabel='앞쪽 F (mm)',title='평면 · 1번 회전축 기준')
for ax,dims in [(front,[0,2]),(side,[1,2]),(top,[0,1])]:
 ax.grid(alpha=.18);ax.set_aspect('equal',adjustable='box')
 ax.plot([cam[dims[0]],end[dims[0]]],[cam[dims[1]],end[dims[1]]],'--',color='#bc6513',lw=1.5)
 ax.scatter(cam[dims[0]],cam[dims[1]],color='#d66214',s=30,zorder=10)
 ax.scatter(0,0,color='#ad3040',s=24,zorder=10)
front.annotate('',xy=(0,575),xytext=(cam[0],575),arrowprops={'arrowstyle':'<->','color':'#1b3948'})
front.text(cam[0]/2,580,f'오른쪽 {cam[0]:.1f} mm',ha='center',fontsize=11)
front.annotate(f'장착면 중심\nH ≈ {cam[2]:.1f} mm',xy=(cam[0],cam[2]),xytext=(-95,505),arrowprops={'arrowstyle':'-','color':'#a65013'},color='#95410f',fontsize=11)
front.text(-98,118,'SO-101\n회색 팔: URDF 기준 자세\n실물 관절값은 미측정',fontsize=10)
front.text(-98,19,'● 원점: J1 회전축의 바닥 투영점',color='#9b2535',fontsize=9)
side.annotate('장착면 기울기 25°\n광축: 앞쪽·아래쪽',xy=(cam[1]+60,cam[2]-130),xytext=(75,490),arrowprops={'arrowstyle':'-','color':'#a65013'},fontsize=10)
top.annotate(f'카메라\nR {cam[0]:.1f}, F {cam[1]:.1f}',xy=cam[:2],xytext=(5,105),arrowprops={'arrowstyle':'-','color':'#a65013'},color='#95410f',fontsize=10)
fig.suptitle('SO-101 + 32×32 상단 카메라 마운트 · CAD 위치 추정',fontsize=19,fontweight='bold')
fig.supxlabel('기준: 원본 STL, 축척 100%, 결합부 완전 삽입 가정. 표시점은 PCB 장착면 중심이며 렌즈의 광학 중심은 별도 확인이 필요합니다.',fontsize=11)
fig.savefig(r/'배치_검토.png',dpi=155)
print('Saved',r/'배치_검토.png','triangles',sum(len(p[1]) for p in parts))
