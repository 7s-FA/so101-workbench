"""Build the SO-101 teaching assets from user geometry and mass evidence."""
from pathlib import Path
import sys,copy,hashlib,json,shutil,xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull

ROOT=Path(__file__).resolve().parents[1];OCV=ROOT.parent/'OpenCV_teach'
sys.path.insert(0,str(OCV));sys.path.insert(0,str(ROOT.parent/'Sim_real'))
from model_tcp import read_stl,origin_mm
from arm_sim.grasp_geometry import triangle_model,collision_object
import fcl


def mesh_properties(tri,mass):
    a,b,c=tri[:,0],tri[:,1],tri[:,2]
    volume=np.einsum('ij,ij->i',a,np.cross(b,c))/6
    total=volume.sum()
    if abs(total)<1e-12:raise ValueError('Zero signed volume')
    s=a+b+c;center=(volume[:,None]*s/4).sum(0)/total
    second=(np.einsum('ni,nj->nij',s,s)+np.einsum('ni,nj->nij',a,a)+np.einsum('ni,nj->nij',b,b)+np.einsum('ni,nj->nij',c,c))
    cov=(second*volume[:,None,None]/20).sum(0)/total-np.outer(center,center)
    inertia=mass*(np.trace(cov)*np.eye(3)-cov)
    if np.linalg.eigvalsh(inertia).min()<=0:raise ValueError('Non-positive mesh inertia')
    return center,inertia


def first_contact(moving,fixed,xy):
    model=triangle_model(moving.reshape(-1,3));other=collision_object(triangle_model(fixed.reshape(-1,3)),np.eye(3),np.zeros(3))
    def hit(z):return fcl.collide(collision_object(model,np.eye(3),[*xy,z]),other,fcl.CollisionRequest(),fcl.CollisionResult())>0
    clear=.06
    for z in np.arange(.0598,-.020,-.0002):
        if hit(z):
            lo,hi=z,clear
            for _ in range(24):
                mid=(lo+hi)/2
                if hit(mid):lo=mid
                else:hi=mid
            return float(hi+.00005)
        clear=z
    raise ValueError('No mesh seating contact')


def add_pads(root,out):
    """Nominal octagonal 8 mm patch on each inner finger surface, 0.2 mm thick."""
    import struct
    fixed=read_stl(out/'meshes/xlerobot_fixed_finger.stl')*.001
    # The distal edge in the known source mesh has the greatest extension +Z.
    v=np.unique(fixed.reshape(-1,3),axis=0);end=v[v[:,2]>v[:,2].max()-.00001]
    adjacent=[np.linalg.norm(fixed-p,axis=2).min(1)<1e-8 for p in end]
    candidates=np.where(adjacent[0]&adjacent[1])[0]
    areas=np.linalg.norm(np.cross(fixed[:,1]-fixed[:,0],fixed[:,2]-fixed[:,0]),axis=1)
    index=int(max(candidates,key=lambda i:areas[i]))
    indices=[int(np.argmin(np.linalg.norm(fixed[index]-p,axis=1))) for p in end]
    rows=[]
    for kind,link_name in [('fixed','gripper_link'),('moving','moving_jaw_so101_v1_link')]:
        link=root.find(f"link[@name='{link_name}']")
        visual=link.find(f"visual[@name='xlerobot_{kind}_finger']")
        tri=read_stl(out/f'meshes/xlerobot_{kind}_finger.stl')*.001
        face=tri[index];tip=face[indices].mean(0);width=face[indices[1]]-face[indices[0]];width/=np.linalg.norm(width)
        normal=np.cross(face[1]-face[0],face[2]-face[0]);normal/=np.linalg.norm(normal)
        along=np.cross(width,normal)
        if np.dot(along,tri.mean((0,1))-tip)<0:along=-along
        center=tip+along*.0045
        vertices=np.array([center+normal*t+.004*(width*np.cos(a)+along*np.sin(a)) for t in (0.,.0002) for a in np.arange(8)*np.pi/4])
        faces=ConvexHull(vertices).simplices
        # Orient faces outwards for a closed convex mesh.
        triangles=vertices[faces].copy()
        for f in triangles:
            if np.dot(np.cross(f[1]-f[0],f[2]-f[0]),f.mean(0)-vertices.mean(0))<0:f[[1,2]]=f[[2,1]]
        path=out/f'meshes/razer_{kind}_pad.stl'
        with path.open('wb') as stream:
            stream.write(b'Nominal Razer pad; m; thickness .0002; width .008'.ljust(80,b'\0'));stream.write(struct.pack('<I',len(triangles)))
            for f in triangles:
                n=np.cross(f[1]-f[0],f[2]-f[0]);n/=np.linalg.norm(n)
                stream.write(struct.pack('<12fH',*n,*f.ravel(),0))
        for tag in ('visual','collision'):
            node=ET.SubElement(link,tag,name=f'razer_{kind}_pad');node.append(copy.deepcopy(visual.find('origin')))
            ET.SubElement(ET.SubElement(node,'geometry'),'mesh',filename=str(path.resolve()))
            if tag=='visual':ET.SubElement(ET.SubElement(node,'material',name='razer_black'),'color',rgba='0.025 0.025 0.025 1')
        tf=origin_mm(visual.find('origin'));center_link=tf[:3,:3]@center+tf[:3,3]/1000
        rows.append({'link':link_name,'center_link_m':center_link.tolist(),'normal_link':(tf[:3,:3]@normal).tolist(),'thickness_mm':.2,'span_mm':8.})
    return rows


def main():
    source=OCV/'assets/so101';out=ROOT/'assets/so101';out.mkdir(exist_ok=True)
    shutil.copytree(source/'assets',out/'meshes',dirs_exist_ok=True)
    root=ET.parse(source/'so101_modified.urdf').getroot();root.set('name','so101_teaching_physics')
    estimates=json.loads((OCV/'calibration/link_mass/link_mass_estimate.json').read_text())
    by_link={r['link']:r for r in estimates['links']};rows=[]
    for link in root.findall('link'):
        if link.get('name') not in by_link:continue
        row=by_link[link.get('name')];parts={p['mesh']:p['mass_g']/1000 for p in row['printed_components']};pieces=[]
        for visual in link.findall('visual'):
            mesh=visual.find('geometry/mesh')
            if mesh is None:continue
            old=Path(mesh.get('filename')).name;file=out/'meshes'/old
            tri=read_stl(file)*np.array([float(v) for v in mesh.get('scale','1 1 1').split()])
            tf=origin_mm(visual.find('origin'));tri=tri@tf[:3,:3].T+tf[:3,3]/1000
            mass=.055 if old.startswith('sts3215') else parts[old]
            center,inertia=mesh_properties(tri,mass);pieces.append((mass,center,inertia))
            mesh.set('filename',str(file.resolve()))
            collision=copy.deepcopy(visual);collision.tag='collision'
            for material in list(collision.findall('material')):collision.remove(material)
            link.append(collision)
        total=sum(p[0] for p in pieces);center=sum(m*c for m,c,_ in pieces)/total
        inertia=sum(i+m*((d@d)*np.eye(3)-np.outer(d,d)) for m,c,i in pieces for d in [c-center])
        node=link.find('inertial');node.find('origin').set('xyz',' '.join(f'{x:.12g}' for x in center));node.find('origin').set('rpy','0 0 0')
        node.find('mass').set('value',f'{total:.12g}')
        node.find('inertia').attrib={key:f'{inertia[i,j]:.12g}' for key,i,j in [('ixx',0,0),('ixy',0,1),('ixz',0,2),('iyy',1,1),('iyz',1,2),('izz',2,2)]}
        assert abs(total-row['mass_kg_estimate'])<1e-9
        rows.append({'link':link.get('name'),'mass_kg':total,'com_m':center.tolist(),'inertia_kg_m2':inertia.tolist()})
    pads=add_pads(root,out)
    ET.indent(root);ET.ElementTree(root).write(out/'so101_teach.urdf',encoding='utf-8',xml_declaration=True)
    assets=ROOT/'assets/so101_workcell';assets.mkdir(exist_ok=True)
    downloads=Path('/home/fresh/Downloads/모델링')
    mapping={'carrier.stl':downloads/'버거위에지그/버거위에지그.stl',
             'cap_jig.stl':downloads/'이전 부품 지그/03_cap_jig_no_grip_hollow_post.stl',
             'ball_jig.stl':downloads/'이전 부품 지그/02_ball_jig_no_grip_two_side.stl',
             'housing_jig.stl':downloads/'이전 부품 지그/01_housing_jig_no_grip_long_slope.stl',
             'cap.stl':downloads/'A제품/03_cap_10mm_original_ball_exposure.stl',
             'ball.stl':downloads/'A제품/02_ball_d30.stl',
             'housing.stl':downloads/'A제품/01_housing_matched_10mm_cap.stl'}
    for name,path in mapping.items():shutil.copyfile(path,assets/name)
    from teach_left_arm import Hardware
    from vision_teaching import digest
    backend=Hardware(OCV/'follower.json',follower_only=True)
    current_station=OCV/'teaching/vision'/(digest(backend.identity)[:16]+'.json')
    cfg={'schema':1,'robot':'so101','model_source':str(out/'so101_teach.urdf'),
         'table':{'height_mm':230.,'size_mm':[650.,800.],'center_mm':[220.,0.]},
         'base_mm':[0.,0.,237.4],'carrier':{'center_base_mm':[210.,0.,-7.4],'yaw_deg':-90.,'size_mm':[220.,148.,6.]},
         'tcp':json.loads(current_station.read_text())['tool'],
         'part_order':['cap','ball','housing'],'parts':{},'dt':1/480,
         'home_deg':[0.,-30.,50.,60.,0.,50.],
         'contact':{'product':'Razer Universal Grip Tape','static_friction':1.,'dynamic_friction':.8,'restitution':0.,
                    'combine_mode':'max','thickness_mm':.2,'span_mm':8.,'shape':'octagon','pad_count':2,
                    'status':'friction and 8 mm patch width are engineering estimates; thickness 0.2 mm specified by user; rigid contact approximation'},
         'actuators':{'arm_effort_nm':2.,'gripper_effort_nm':.4,'arm_stiffness':80.,'arm_damping':4.,'gripper_stiffness':2.,'gripper_damping':.08},
         'mass_source':str(OCV/'calibration/link_mass/link_mass_estimate.json'),
         'coordinate_note':'Distances in millimetres. Carrier local U is long side, V short side. Part seating offsets are retained.'}
    carrier=read_stl(assets/'carrier.stl')*.001
    for kind,xy in [('cap',[-72.,36.]),('ball',[0.,-36.]),('housing',[72.,36.])]:
        jig=read_stl(assets/(kind+'_jig.stl'))*.001;part=read_stl(assets/(kind+'.stl'))*.001
        jig_z=first_contact(jig,carrier,np.array(xy)*.001)*1000
        # Preserve the previous jig's off-centre pocket, not the STL/plate centre.
        seat=first_contact(part,jig,[0.,.004])*1000
        height=float(part[:,:,2].max()*1000)
        cfg['parts'][kind]={'label':{'cap':'상단 · 뚜껑','ball':'구','housing':'하단 · 하우징'}[kind],
            'jig_center_carrier_mm':[*xy,jig_z],'jig_yaw_deg':0.,'part_offset_jig_mm':[0.,4.,seat],
            'part_yaw_deg':0.,'mesh_height_mm':height,
            'grasp_tcp_part_mm':[15. if kind=='ball' else 25.,0.,max(2.,height/2-4.)],
            'grasp_yaw_deg':0.,'grasp_pitch_deg':180.,'hover_mm':40.,'lift_mm':40.,'open_percent':45. if kind=='ball' else 55.,'close_percent':0.,
            'mesh':str(assets/(kind+'.stl')),'jig_mesh':str(assets/(kind+'_jig.stl')),
            'mass_kg':json.loads((ROOT/'config/materials.json').read_text())['parts'][kind]['mass_kg']}
    # Existing user layout files are edited through the app, never reset on normal launch.
    (ROOT/'config/so101_scene.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
    provenance={'source_urdf_sha256':hashlib.sha256((source/'so101_modified.urdf').read_bytes()).hexdigest(),
        'link_inertias':rows,'inertia_method':'Uniform mass within each closed component mesh, weighted by 3MF extrusion/55g motor mass; parallel-axis combination. Approximation, not measured infill or internal motor distribution.',
        'source_assets':{name:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for name,p in mapping.items()},
        'total_mass_kg':sum(row['mass_kg'] for row in rows),'pads':pads}
    (out/'physics_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'mass_kg':provenance['total_mass_kg'],'seating':{k:v['part_offset_jig_mm'] for k,v in cfg['parts'].items()},'jigs':{k:v['jig_center_carrier_mm'] for k,v in cfg['parts'].items()}},ensure_ascii=False))

if __name__=='__main__':main()
