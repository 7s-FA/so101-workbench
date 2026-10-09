"""Render actual assembled jaw meshes and the derived TCP without devices."""
from pathlib import Path
import sys,json
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from model_tcp import MODEL,read_stl,origin_mm,fixed_finger_tip

reference=fixed_finger_tip()
(HERE/'fixed_finger_tip.json').write_text(json.dumps(reference,ensure_ascii=False,indent=2)+'\n')
root=ET.parse(MODEL/'so101_modified.urdf').getroot()
frame=origin_mm(root.find("joint[@name='gripper_frame_joint']/origin"))
moving=origin_mm(root.find("joint[@name='gripper']/origin"))
opening=np.eye(4);opening[:3,:3]=Rotation.from_euler('z',35,degrees=True).as_matrix()
parts=[]
for name,link,link_tf in [('fixed','gripper_link',np.eye(4)),('moving','moving_jaw_so101_v1_link',moving@opening)]:
 for role in ('holder','finger'):
  visual=root.find(f"link[@name='{link}']/visual[@name='xlerobot_{name}_{role}']")
  mesh=visual.find('geometry/mesh')
  triangles=read_stl(MODEL/mesh.get('filename'))*np.array([float(v) for v in mesh.get('scale','1 1 1').split()])*1000
  transform=np.linalg.inv(frame)@link_tf@origin_mm(visual.find('origin'))
  triangles=triangles@transform[:3,:3].T+transform[:3,3]
  color=('#138eb8' if name=='fixed' else '#7cabc4') if role=='finger' else '#596674'
  parts.append((name,role,triangles,color))
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':11})
fig=plt.figure(figsize=(15,8.2),facecolor='#f9fafb',layout='constrained')
ax=fig.add_subplot(1,3,1,projection='3d');side=fig.add_subplot(1,3,2);end=fig.add_subplot(1,3,3)
point=np.array(reference['xyz_mm']);ends=np.array(reference['edge_endpoints_mm'])
for name,role,triangles,color in parts:
 ax.add_collection3d(Poly3DCollection(triangles,facecolors=color,linewidth=0,shade=True))
 side.add_collection(PolyCollection(triangles[:,:,[0,2]],facecolors=color,edgecolors='none',alpha=.75))
 if name=='fixed' and role=='finger':
  end.add_collection(PolyCollection(triangles[:,:,[1,2]],facecolors=color,edgecolors='none',alpha=.75))
ax.scatter(*point,color='#d43828',s=55,depthshade=False)
ax.scatter(0,0,0,color='#222e36',s=28,depthshade=False)
ax.set(xlim=(-95,35),ylim=(-26,26),zlim=(-105,15),xlabel='X (mm)',ylabel='Y (mm)',zlabel='',title='조립된 고정·가동 집게')
ax.set_box_aspect((130,52,120));ax.view_init(17,110)
side.set(xlim=(40,-100),ylim=(-105,16),xlabel='기존 집게 기준 X (mm)',ylabel='기존 집게 기준 Z (mm)',title='측면 · 빨간 점이 새 TCP')
side.set_aspect('equal');side.scatter(point[0],point[2],color='#d43828',s=65,zorder=5)
side.scatter(0,0,color='#222e36',s=30,zorder=5)
side.annotate('고정 집게 끝단 중심',point[[0,2]],xytext=(.36,.94),textcoords='axes fraction',arrowprops={'arrowstyle':'->','color':'#b72d23'},color='#b72d23')
side.annotate('기존 gripper_frame_link', (0,0),xytext=(.3,.72),textcoords='axes fraction',arrowprops={'arrowstyle':'->','color':'#222e36'},fontsize=9)
side.text(.04,.055,'가동 집게: 설명용 35° 열림\n고정 집게의 TCP는 열림 각도와 무관',transform=side.transAxes,fontsize=10,bbox={'facecolor':'white','edgecolor':'none','alpha':.9})
end.set(xlim=(-8,9),ylim=(.5,8),xlabel='기존 집게 기준 Y (mm)',ylabel='Z (mm)',title='고정 집게 끝단 확대 · 폭의 중심')
end.set_aspect('equal')
end.plot(ends[:,1],ends[:,2],color='#d43828',lw=3,zorder=5)
end.scatter(point[1],point[2],color='#d43828',s=75,zorder=6)
end.annotate('끝선 양 끝의 중점',point[[1,2]],xytext=(-7,7.4),arrowprops={'arrowstyle':'->','color':'#b72d23'},color='#b72d23',fontsize=10)
end.text(-7,1.4,f"끝선 폭 {reference['edge_width_mm']:.3f} mm\nX {point[0]:+.3f} mm\nY {point[1]:+.3f} mm\nZ {point[2]:+.3f} mm",fontsize=11,bbox={'facecolor':'white','edgecolor':'#c4ced4','alpha':.96})
for a in (side,end):a.grid(alpha=.18);a.spines['top'].set_visible(False);a.spines['right'].set_visible(False)
fig.suptitle('왼쪽 고정 집게 끝단 중심을 TCP로 지정',fontsize=20,fontweight='bold')
fig.supxlabel('실제 조립 STL + URDF 변환 기준 · XYZ 위치만 변경, 회전축은 유지 · TPU 변형과 실물 조립 오차는 미반영',fontsize=11)
fig.savefig(HERE/'고정집게_TCP.png',dpi=140)
print(json.dumps(reference,ensure_ascii=False,indent=2))
