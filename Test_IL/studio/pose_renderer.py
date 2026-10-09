"""MuJoCo/OpenGL pose display: no time integration, contacts or robot IO."""
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from .config import ROOT
from .robot import NAMES, vector


def binary_mesh(path, cache):
    """MuJoCo accepts binary STL; keep ASCII source meshes unchanged."""
    content=path.read_bytes()
    count=struct.unpack_from('<I',content,80)[0] if len(content)>=84 else 0
    if len(content)>=84 and len(content)==84+50*count:
        if count<=200000:return path
        # The native D435 CAD exceeds MuJoCo's STL decoder limit. OBJ keeps
        # every vertex/triangle without changing the user's mesh or its scale.
        output=cache/(hashlib.sha256(content).hexdigest()[:20]+'.obj')
        if not output.exists():
            faces=np.frombuffer(content,dtype=[('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attr','<u2')],offset=84)
            vertices,indices=np.unique(faces['vertices'].reshape(-1,3),axis=0,return_inverse=True)
            with output.open('w') as file:
                np.savetxt(file,vertices,fmt='v %.9g %.9g %.9g')
                np.savetxt(file,indices.reshape(-1,3)+1,fmt='f %d %d %d')
        return output
    vertices=[]
    for line in content.decode('ascii').splitlines():
        words=line.split()
        if words and words[0]=='vertex':vertices.append([float(v) for v in words[1:]])
    triangles=np.asarray(vertices,dtype=np.float32).reshape(-1,3,3)
    if not len(triangles):raise ValueError('STL triangles missing: '+str(path))
    output=cache/(hashlib.sha256(content).hexdigest()[:20]+'.stl')
    if not output.exists():
        normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        lengths=np.linalg.norm(normals,axis=1);normals/=np.maximum(lengths[:,None],1e-20)
        faces=np.zeros(len(triangles),dtype=[('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attr','<u2')])
        faces['normal']=normals;faces['vertices']=triangles
        output.write_bytes(b'Test_IL visual mesh'.ljust(80,b' ')+struct.pack('<I',len(faces))+faces.tobytes())
    return output


def scene_xml(kind, cache):
    """Use MuJoCo's URDF importer, preserving the existing visual transforms."""
    import mujoco
    scene=ROOT/('config/scene.json' if kind=='so101' else 'config/legacy_omx/scene.json')
    cfg=json.loads(scene.read_text());path=Path(cfg['model_source']);urdf=ET.parse(path).getroot()
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    for link in urdf.findall('link'):
        for item in link.findall('collision'):link.remove(item)
        for index,visual in enumerate(link.findall('visual')):
            name=visual.get('name','')
            color='.08 .40 .75 1' if 'xlerobot' in name and 'finger' in name else '.60 .63 .66 1' if any(k in name for k in ('overhead_d435','ScrewHead','Washer')) else '.16 .18 .20 1'
            for material in visual.findall('material'):visual.remove(material)
            E=ET.SubElement(visual,'material',name='display_'+link.get('name')+'_'+str(index))
            ET.SubElement(E,'color',rgba=color)
            mesh=visual.find('geometry/mesh')
            if mesh is not None:
                source=Path(mesh.get('filename'));source=source if source.is_absolute() else path.parent/source
                mesh.set('filename',str(binary_mesh(source,cache)))
    for old in urdf.findall('mujoco'):urdf.remove(old)
    ext=ET.SubElement(urdf,'mujoco')
    ET.SubElement(ext,'compiler',discardvisual='false',fusestatic='false',strippath='false',balanceinertia='true')
    model=mujoco.MjModel.from_xml_string(ET.tostring(urdf,encoding='unicode'))
    with tempfile.TemporaryDirectory(dir=cache) as folder:
        file=Path(folder)/'import.xml';mujoco.mj_saveLastXML(str(file),model);root=ET.parse(file).getroot()
    world=root.find('worldbody')
    world.find('body').set('pos',' '.join(map(str,cfg['base'])))
    for geom in root.iter('geom'):geom.set('contype','0');geom.set('conaffinity','0')
    for joint in world.iter('joint'):joint.set('limited','false')
    option=ET.SubElement(root,'option',gravity='0 0 0')
    ET.SubElement(option,'flag',contact='disable',constraint='disable',actuation='disable')
    visual=ET.SubElement(root,'visual')
    ET.SubElement(visual,'global',offwidth='960',offheight='640')
    ET.SubElement(visual,'quality',shadowsize='1024',offsamples='2')
    ET.SubElement(visual,'headlight',ambient='.4 .4 .4',diffuse='.6 .6 .6',specular='.15 .15 .15')
    ET.SubElement(root.find('asset'),'texture',type='skybox',builtin='flat',rgb1='.12 .15 .18',width='128',height='768')
    ET.SubElement(world,'light',pos='.5 -.5 1.5',dir='-.3 .3 -1',diffuse='.7 .7 .7')
    def box(name,xyz,size,color):
        ET.SubElement(world,'geom',name=name,type='box',pos=' '.join(map(str,xyz)),
            size=' '.join(str(s/2) for s in size),rgba=color,contype='0',conaffinity='0')
    x,y=cfg['table']['center'];w,d=cfg['table']['size'];z=cfg['table']['height']
    box('table',(x,y,z-.015),(w,d,.03),'.65 .49 .29 1')
    for i,(a,b) in enumerate(((-1,-1),(-1,1),(1,-1),(1,1))):
        box('table_leg_'+str(i),(x+a*(w/2-.03),y+b*(d/2-.03),(z-.03)/2),(.035,.035,z-.03),'.5 .35 .19 1')
    box('floor',(0,0,-.015),(1.2,1.2,.03),'.16 .19 .21 1')
    xml=ET.tostring(root,encoding='unicode');(cache/(kind+'.xml')).write_text(xml)
    return xml


class PoseRenderer:
    def __init__(self,width=960,height=640,cache=None):
        import mujoco
        self.mj=mujoco;self.width=width;self.height=height
        self.cache=Path(cache or ROOT/'outputs/pose_models')
        self.models={};self.kind=None;self.renderer=None;self.last_key=None;self.frames=0

    def render(self,angles,kind='so101'):
        q=vector(angles)
        if kind not in ('so101','omx'):raise ValueError('Unknown display robot: '+kind)
        key=(kind,tuple(q))
        if key==self.last_key:return None
        if kind not in self.models:
            model=self.mj.MjModel.from_xml_string(scene_xml(kind,self.cache))
            names=NAMES if kind=='so101' else ['joint1','joint2','joint3','joint4','joint5','gripper_joint_1','gripper_joint_2']
            self.models[kind]=(model,self.mj.MjData(model),[model.joint(n).qposadr[0] for n in names])
        model,data,addresses=self.models[kind]
        if self.kind!=kind:
            if self.renderer:self.renderer.close()
            self.renderer=self.mj.Renderer(model,width=self.width,height=self.height);self.kind=kind
        data.qpos[addresses]=q+([-q[5]] if kind=='omx' else [])
        # Exactly update the received pose; never clamp it to CAD joint limits.
        self.mj.mj_kinematics(model,data);self.mj.mj_camlight(model,data)
        camera=self.mj.MjvCamera();camera.lookat[:]=(.12,0.,.34)
        camera.azimuth=135;camera.elevation=-28;camera.distance=1.45
        self.renderer.update_scene(data,camera=camera)
        self.renderer.scene.flags[self.mj.mjtRndFlag.mjRND_SHADOW]=False
        image=self.renderer.render();self.last_key=key;self.frames+=1
        return image

    def close(self):
        if self.renderer:self.renderer.close();self.renderer=None
