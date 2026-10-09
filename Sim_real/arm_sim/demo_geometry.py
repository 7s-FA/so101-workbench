"""Offline URDF collision-mesh audit of assisted arm motion (metres).

Convex envelopes are conservative, not contact-physics success evidence.
"""
import math
import xml.etree.ElementTree as ET
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from scipy.optimize import linprog
from arm_sim.common import ROOT
from arm_sim.factory import warehouse_shelf_sections,warehouse_world_size,table_legs


def box_shape(name,center,size):
    c=np.asarray(center);h=np.asarray(size)/2
    points=np.array([[c[0]+x*h[0],c[1]+y*h[1],c[2]+z*h[2]] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    normals=np.r_[np.eye(3),-np.eye(3)]
    return (name,0,points,np.c_[normals,-np.r_[c+h,-c+h]],c-h,c+h)


def tray_shapes(prefix,center,kinds,pitch,z,vertical=True,carrier_width=.08):
    shapes=[]
    x,y=center;size=[carrier_width,(len(kinds)-1)*pitch+carrier_width]
    if not vertical:size.reverse()
    shapes.append(box_shape(prefix+'/Carrier',[x,y,z+.001],[*size,.002]))
    for i,kind in enumerate(kinds):
        offset=(i-(len(kinds)-1)/2)*pitch
        px,py=(x,y+offset) if vertical else (x+offset,y)
        width=.04 if kind=='ball' else .062
        shapes.append(box_shape(prefix+f'/Slot{i}',[px,py,z+.002],[width,width,.004]))
        if kind=='ball':
            for dx in (-1,1):
                for dy in (-1,1):shapes.append(box_shape(prefix+f'/Stop{i}_{dx}_{dy}',[px+dx*.014,py+dy*.014,z+.006],[.008,.008,.008]))
        else:
            for side in (-1,1):shapes.append(box_shape(prefix+f'/Stop{i}_{side}',[px,py+side*.029,z+.005],[.038,.004,.006]))
    return shapes

def fixture_shapes(cfg):
    """Match FactoryCell's carrier plates, cradle bases and raised stops."""
    z=cfg['table_height'];shapes=[]
    for table in cfg['tables']:
        for i,(center,size) in enumerate(table_legs(table,z)):
            shapes.append(box_shape(f'table_leg/{table["id"]}/{i}',center,size))
    for panel in cfg.get('docking_panels',[]):
        shapes.append(box_shape('docking_panel/'+panel['id'],panel['center'],panel['size']))
    for area in ('source','preparation'):
        batches=cfg.get('source_batches',[cfg[area]]) if area=='source' else [cfg[area]]
        for i,c in enumerate(batches):
            for kind,x in c['columns'].items():shapes.extend(tray_shapes(area+f'/Batch{i}/'+kind,[x,c['center_y']],[kind]*c['count'],c['pitch'],z,carrier_width=c.get('carrier_width',.08)))
    c=cfg['finished']
    if c.get('enabled',True):shapes.extend(tray_shapes('finished',c['center'],['housing']*c['count'],c['pitch'],z,False))
    shapes.extend(tray_shapes('handoff',cfg['handoff'],['housing'],.09,z))
    rack=cfg['warehouse'];x,y=rack['center'];w,d=warehouse_world_size(cfg)
    post_height=rack.get('post_height',max(rack['levels'])+.12)
    for i,level in enumerate(rack['levels']):
        for center,size in warehouse_shelf_sections(cfg,i):shapes.append(box_shape('warehouse/shelf',center,size))
        # In the demo the upper tray is the original moving pallet, not a
        # second fixed tray occupying the same storage slot.
        if not rack.get('moving_trays') and not (i==1 and rack.get('fork_slots')):
            vertical=bool(round(rack.get('yaw',0.)/(math.pi/2))%2)
            shapes.extend(tray_shapes('warehouse/tray',[x,y],['housing']*3,.09,level,vertical))
    for dx in (-1,1):
        for dy in (-1,1):shapes.append(box_shape('warehouse/post',[x+dx*(w/2-.012),y+dy*(d/2-.012),post_height/2],[.018,.018,post_height]))
    return shapes


def moved_shapes(shapes,pose,lift=0.):
    x,y,yaw=pose;rot=Rotation.from_euler('z',yaw).as_matrix();shift=np.array([x,y,.001+lift])
    result=[]
    for name,index,points,planes,_,_ in shapes:
        world=points@rot.T+shift;normals=planes[:,:3]@rot.T
        result.append((name,index,world,np.c_[normals,planes[:,3]-normals@shift],world.min(axis=0),world.max(axis=0)))
    return result


def cargo_shapes(cfg):
    """Local container geometry matching both mobile modifications."""
    b=cfg['burger'];w=cfg['waffle'];z=b['deck_top']
    burger=[box_shape('cargo/deck',[-.032,0,z-.004],[*b['deck_size'],.008])]
    for i,kind in enumerate(('housing','ball','cap')):
        burger.extend(tray_shapes('cargo/'+kind,[-.032+(i-1)*b['cargo_pitch'],0],[kind]*3,b['cargo_pitch'],z,carrier_width=b.get('cargo_carrier_width',.08)))
    x=.15+w['fork_length']/2;z=w['fork_top']
    fork=[box_shape('fork/pallet',[x,0,z+.004],[.18,.29,.008])]
    fork.extend(tray_shapes('fork/tray',[x,0],['housing']*3,.09,z+.008))
    return burger,fork


def forklift_shapes(cfg):
    w=cfg['waffle'];z=w['fork_top']
    # Source Waffle URDF collision box including its base_footprint offset.
    fixed=[box_shape('waffle/body',[-.064,0,.057],[.266,.266,.094])]
    moving=[box_shape('fork/carriage',[.14,0,z+.02],[.018,.15,.07])]
    for side in (-1,1):
        fixed.append(box_shape('fork/mast',[.125,side*.055,w['mast_start']+w['mast_height']/2],[.018,.018,w['mast_height']]))
        moving.append(box_shape('fork/tine',[.15+w['fork_length']/2,side*.055,z-.009],[w['fork_length'],.025,.018]))
    return fixed,moving


def clamp_shapes(cfg):
    result={};z=cfg['table_height']
    for name,c in cfg.get('clamps',{}).items():
        x,y=cfg['arms'][name]['position'];so=cfg['arms'][name]['model']=='so101'
        # Plate footprint follows the base, not the gripper's home direction.
        center=[x,y-.0215 if so else y,z+.002]
        shapes=[box_shape(name+'/Plate',center,[.12,.11,.004] if so else [.14,.17,.004])]
        rise=cfg['arms'][name].get('mount_height',0.)
        if rise>0:
            shapes.append(box_shape(name+'/Riser',[center[0],center[1],z+.004+rise/2],[.12,.11,rise] if so else [.14,.17,rise]))
        side=c['edge']
        edge=(y+x-c['edge_value']) if side=='left' else (y+c['edge_value']-x) if side=='right' else c['edge_value']
        sign=1 if side in ('back','left','right') else -1
        for i,dx in enumerate((-.04,.04)):
            shapes.extend([
                box_shape(name+f'/UpperJaw{i}',[x+dx,edge-sign*.005,z+.010],[.023,.065,.012]),
                box_shape(name+f'/LowerJaw{i}',[x+dx,edge-sign*.005,z-.048],[.023,.065,.012]),
                box_shape(name+f'/Rail{i}',[x+dx,edge+sign*.023,z-.028],[.009,.009,.094]),
                box_shape(name+f'/Pad{i}',[x+dx,edge-sign*.024,z-.032],[.020,.017,.004]),
                box_shape(name+f'/Handle{i}',[x+dx,edge+sign*.023,z-.079],[.033,.017,.016])])
        if side in ('left','right'):
            # Rotate the back-edge clamp about its arm base onto a side edge.
            s=1 if side=='left' else -1
            r=np.array([[0.,-s,0.],[s,0.,0.],[0.,0.,1.]])
            pivot=np.array([x,y,0.]);rotated=shapes[:1]
            for shape in shapes[1:]:
                pts=(shape[2]-pivot)@r.T+pivot
                rotated.append(box_shape(shape[0],(pts.min(0)+pts.max(0))/2,pts.max(0)-pts.min(0)))
            shapes=rotated
        result[name]=shapes
    return result


def origin(node):
    tf=np.eye(4)
    if node is not None:
        tf[:3,3]=np.fromstring(node.get('xyz','0 0 0'),sep=' ')
        tf[:3,:3]=Rotation.from_euler('xyz',np.fromstring(node.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return tf


@lru_cache(maxsize=128)
def mesh_vertices(path):
    raw=Path(path).read_bytes()
    count=int.from_bytes(raw[80:84],'little')
    if len(raw)==84+count*50:
        dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attr','<u2')])
        points=np.frombuffer(raw,offset=84,count=count,dtype=dtype)['vertices'].reshape(-1,3)
    else:
        points=np.array([[float(v) for v in line.split()[1:]] for line in raw.decode().splitlines() if line.strip().startswith('vertex ')])
    points=np.unique(points,axis=0).astype(float)
    return points[ConvexHull(points).vertices]


class ArmGeometry:
    def __init__(self,kinematics):
        self.kin=kinematics;self.joints=[];self.meshes=[]
        root=ET.parse(ROOT/kinematics.model['urdf']).getroot()
        for j in root.findall('joint'):
            axis=None if j.get('type')=='fixed' else np.fromstring(j.find('axis').get('xyz'),sep=' ')
            index=None if axis is None else kinematics.model['joints'].index(j.get('name'))
            self.joints.append((j.find('parent').get('link'),j.find('child').get('link'),origin(j.find('origin')),axis,index))
        children={c for _,c,*_ in self.joints}
        self.root=next(l.get('name') for l in root.findall('link') if l.get('name') not in children)
        for link in root.findall('link'):
            for i,c in enumerate(link.findall('collision')):
                mesh=c.find('geometry/mesh')
                if mesh is None:raise ValueError('Arm audit expects URDF collision meshes')
                points=mesh_vertices(mesh.get('filename'))*np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
                tf=origin(c.find('origin'));points=points@tf[:3,:3].T+tf[:3,3]
                hull=ConvexHull(points)
                self.meshes.append((link.get('name'),i,points[hull.vertices],hull.equations))

    def transforms(self,q):
        transforms={self.root:self.kin.base};pending=list(self.joints)
        while pending:
            for item in pending[:]:
                parent,child,offset,axis,index=item
                if parent not in transforms:continue
                tf=transforms[parent]@offset
                if axis is not None:
                    rot=np.eye(4);rot[:3,:3]=Rotation.from_rotvec(axis*q[index]).as_matrix();tf=tf@rot
                transforms[child]=tf;pending.remove(item)
        return transforms

    def at(self,q):
        transforms=self.transforms(q)
        result=[]
        for link,index,points,eq in self.meshes:
            tf=transforms[link];world=points@tf[:3,:3].T+tf[:3,3]
            normals=eq[:,:3]@tf[:3,:3].T
            planes=np.c_[normals,eq[:,3]-normals@tf[:3,3]]
            result.append((link,index,world,planes,world.min(axis=0),world.max(axis=0)))
        return result


def intersections(a,b,margin=0.):
    hits=[]
    if np.any(np.max([s[5] for s in a],axis=0)+margin < np.min([s[4] for s in b],axis=0)) or np.any(np.max([s[5] for s in b],axis=0)+margin < np.min([s[4] for s in a],axis=0)):
        return hits
    alo=np.array([s[4] for s in a]);ahi=np.array([s[5] for s in a])
    blo=np.array([s[4] for s in b]);bhi=np.array([s[5] for s in b])
    candidates=np.all(ahi[:,None,:]+margin>=blo[None,:,:],axis=2)&np.all(bhi[None,:,:]+margin>=alo[:,None,:],axis=2)
    for i,j in zip(*np.where(candidates)):
        left,right=a[i],b[j];eq=np.vstack([left[3],right[3]])
        solution=linprog(np.zeros(3),A_ub=eq[:,:3],b_ub=-eq[:,3]+margin/2,
                         bounds=[(None,None)]*3,method='highs')
        if solution.success:hits.append((left[0],right[0]))
    return hits


def audit(plan,interval=.5):
    geometries={n:ArmGeometry(k) for n,k in plan.ik.items()}
    table_hits=[];arm_hits=[];lowest={n:float('inf') for n in geometries}
    for t in np.arange(0,plan.duration+interval,interval):
        frames={n:g.at(plan.arms[n].at(t)) for n,g in geometries.items()}
        for name,shapes in frames.items():
            for link,idx,pts,_,lo,hi in shapes:
                lowest[name]=min(lowest[name],float(lo[2]))
                if lo[2]>=plan.cfg['table_height']-.001:continue
                for table in plan.cfg['tables']:
                    center=np.array(table['center']);half=np.array(table['size'])/2
                    mask=np.all(np.abs(pts[:,:2]-center)<half-.001,axis=1)&(pts[:,2]<plan.cfg['table_height']-.001)
                    if mask.any():table_hits.append({'t':round(float(t),3),'arm':name,'link':link,'table':table['id'],'z':float(pts[mask,2].min())})
        for a,b in combinations(frames,2):
            hits=intersections(frames[a],frames[b])
            if hits:arm_hits.append({'t':round(float(t),3),'arms':[a,b],'links':hits})
    return {'interval_s':interval,'lowest_mesh_z_m':lowest,'table_hits':table_hits,'arm_hits':arm_hits,
            'scope':'Sampled URDF convex collision meshes; actual new finger/camera CAD and contact forces are unverified'}


class GeometryGuard:
    """Reject a candidate before it is written, including intermediate joint poses.

    This supplements disabled PhysX contacts in the assisted demo. It is a
    sampled geometric guard, not a certificate for unmodelled hardware.
    """
    def __init__(self, plan):
        self.plan=plan
        self.geometry={n:ArmGeometry(k) for n,k in plan.ik.items()}
        self.fixtures=fixture_shapes(plan.cfg);self.fixture_cache={}
        self.clamps=clamp_shapes(plan.cfg)
        legs=[s for s in self.fixtures if s[0].startswith('table_leg/')]
        for shapes in self.clamps.values():
            hits=intersections(shapes,legs,margin=-.00001)
            if hits:raise ValueError(f'Mounting clamp intersects a table leg: {hits}')
        self.payload_fixtures=self.fixtures+[s for v in self.clamps.values() for s in v]
        for table in plan.cfg['tables']:
            self.payload_fixtures.append(box_shape('table/'+table['id'],[*table['center'],plan.cfg['table_height']-.015],[*table['size'],.03]))
        self.arm_clearance=float(plan.cfg.get('arm_clearance',.02))
        if not math.isfinite(self.arm_clearance) or self.arm_clearance<.002:
            raise ValueError('Invalid inter-arm clearance')
        self.cache={};self.pair_cache={};self.last=None;self.checked=0
        self.mobile_local=cargo_shapes(plan.cfg);self.mobile_cache={};self.mobile_frames={};self.last_time=None
        self.forklift_local=forklift_shapes(plan.cfg);self.warehouse=[s for s in self.fixtures if s[0].startswith('warehouse/')];self.last_rack_check=None;self.forklift_frames=[]
        from arm_sim.grasp_geometry import PartClearanceGuard
        self.parts=PartClearanceGuard(plan)
        from arm_sim.traffic import TrafficGuard
        self.traffic=TrafficGuard(plan)

    def check(self, joints, reset=False, time=None):
        if any(not np.isfinite(q).all() for q in joints.values()):
            raise RuntimeError('Non-finite arm geometry')
        distance=0 if self.last is None or reset else max(float(np.max(np.abs(joints[n]-self.last[n]))) for n in joints)
        count=max(1,math.ceil(distance/.01))
        if time is not None and self.last_time is not None and not reset:
            for track in self.plan.vehicles.values():
                delta=np.abs(track.at(time)-track.at(self.last_time))
                count=max(count,math.ceil(max(delta[:2])/.005),math.ceil(delta[2]/.02))
            count=max(count,math.ceil(abs(float(self.plan.lift.at(time)[0]-self.plan.lift.at(self.last_time)[0]))/.005))
        for u in np.linspace(1/count,1,count):
            poses=joints if self.last is None or reset else {n:self.last[n]*(1-u)+q*u for n,q in joints.items()}
            frames={}
            for name,q in poses.items():
                key=tuple(q)
                if self.cache.get(name,(None,))[0]!=key:
                    self.cache[name]=(key,self.geometry[name].at(q))
                frames[name]=self.cache[name][1]
            for a,b in combinations(frames,2):
                key=(self.cache[a][0],self.cache[b][0])
                if self.pair_cache.get((a,b))==key:continue
                hits=intersections(frames[a],frames[b],margin=self.arm_clearance)
                if hits:raise RuntimeError(f'Arm clearance below {self.arm_clearance*1000:g} mm: {a}/{b} {hits}')
                self.pair_cache[a,b]=key
            for name,shapes in frames.items():
                key=self.cache[name][0]
                if self.fixture_cache.get(name)!=key:
                    hits=intersections(shapes,self.fixtures,margin=-.00001)
                    if hits:raise RuntimeError(f'Arm intersects a parts container: {name} {hits}')
                    for owner,clamp in self.clamps.items():
                        moving=[s for s in shapes if owner!=name or s[0] not in ('base_link','link0')]
                        hits=intersections(moving,clamp,margin=-.00001)
                        if hits:raise RuntimeError(f'Arm intersects a mounting clamp: {name} {hits}')
                    self.fixture_cache[name]=key
                for link,_,pts,_,lo,_ in shapes:
                    if lo[2]>=self.plan.cfg['table_height']-.001:continue
                    for table in self.plan.cfg['tables']:
                        center=np.array(table['center']);half=np.array(table['size'])/2
                        inside=np.all(np.abs(pts[:,:2]-center)<half-.001,axis=1)
                        if np.any(inside & (pts[:,2]<self.plan.cfg['table_height']-.001)):
                            raise RuntimeError(f'Arm below tabletop: {name}/{link}/{table["id"]}')
            if time is not None:
                t=time if self.last_time is None or reset else self.last_time*(1-u)+time*u
                vehicles={n:tr.at(t) for n,tr in self.plan.vehicles.items()}
                self.traffic.check(t,vehicles)
                lift=float(self.plan.lift.at(t)[0])
                histories=getattr(self.plan,'trays',[self.plan.tray])
                entries=[self.plan.entry(h,t) for h in histories]
                rack_key=(*vehicles['waffle_1'],lift,*(e[1] for e in entries))
                if rack_key!=self.last_rack_check:
                    forklift=moved_shapes(self.forklift_local[0],vehicles['waffle_1'])+moved_shapes(self.forklift_local[1],vehicles['waffle_1'],lift)
                    self.forklift_frames=forklift.copy()
                    if any(e[1]=='lift' for e in entries):forklift+=moved_shapes(self.mobile_local[1],vehicles['waffle_1'],lift)
                    stored_trays=[shape for e in entries if e[1]=='world' for shape in moved_shapes(self.mobile_local[1],e[2][:3],e[2][3])]
                    hits=intersections(forklift,self.warehouse+stored_trays,margin=-.00001)
                    if hits:raise RuntimeError(f'Forklift intersects warehouse: {hits}')
                    self.last_rack_check=rack_key
                mobile=[(n,p,0.,self.mobile_local[0]) for n,p in vehicles.items() if n!='waffle_1']
                for i,entry in enumerate(entries):
                    pose=vehicles['waffle_1'] if entry[1]=='lift' else entry[2][:3]
                    height=lift if entry[1]=='lift' else entry[2][3]
                    mobile.append((f'shipping_{i}',pose,height,self.mobile_local[1]))
                for vehicle,pose,height,local in mobile:
                    mobile_key=(*pose,height)
                    if self.mobile_frames.get(vehicle,(None,))[0]!=mobile_key:
                        self.mobile_frames[vehicle]=(mobile_key,moved_shapes(local,pose,height))
                    shapes=self.mobile_frames[vehicle][1]
                    for arm,frame in frames.items():
                        key=(self.cache[arm][0],mobile_key)
                        if self.mobile_cache.get((arm,vehicle))==key:continue
                        hits=intersections(frame,shapes,margin=-.00001)
                        if hits:raise RuntimeError(f'Arm intersects mobile container: {arm}/{vehicle} {hits}')
                        self.mobile_cache[arm,vehicle]=key
                for arm,frame in frames.items():
                    key=(self.cache[arm][0],rack_key)
                    if self.mobile_cache.get((arm,'forklift'))!=key:
                        hits=intersections(frame,self.forklift_frames,margin=-.00001)
                        if hits:raise RuntimeError(f'Arm intersects forklift: {arm} {hits}')
                        self.mobile_cache[arm,'forklift']=key
                # Includes untouched parts during approach and the carried part during retreat.
                self.parts.check(t,poses,frames,self.payload_fixtures+self.forklift_frames+[s for _,shapes in self.mobile_frames.values() for s in shapes])
            self.checked+=1
        self.last={n:q.copy() for n,q in joints.items()}
        self.last_time=time
