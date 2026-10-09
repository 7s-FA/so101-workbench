"""Triangle-mesh proximity for grasp feasibility; no force-closure claim."""
from functools import lru_cache
from pathlib import Path
from itertools import combinations
from scipy.spatial import ConvexHull
import xml.etree.ElementTree as ET
import numpy as np
import fcl
from arm_sim.common import ROOT
from arm_sim.demo_geometry import ArmGeometry, origin
from arm_sim.workcell import vertices


def stl_triangles(path):
    raw=Path(path).read_bytes()
    n=int.from_bytes(raw[80:84],'little')
    if len(raw)==84+50*n:
        dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attr','<u2')])
        return np.frombuffer(raw,offset=84,count=n,dtype=dtype)['vertices'].reshape(-1,3).astype(float)
    return np.array([[float(v) for v in line.split()[1:]] for line in raw.decode().splitlines() if line.strip().startswith('vertex ')])


def triangle_model(v):
    model=fcl.BVHModel()
    model.beginModel(len(v),len(v)//3)
    model.addSubModel(v,np.arange(len(v)).reshape(-1,3))
    model.endModel()
    return model


@lru_cache(maxsize=4)
def part_model(kind):
    return triangle_model(np.asarray(vertices(kind)))


def collision_object(model,rotation,position):
    return fcl.CollisionObject(model,fcl.Transform(np.asarray(rotation),np.asarray(position)))


def separation(a,b):
    result=fcl.DistanceResult()
    d=fcl.distance(a,b,fcl.DistanceRequest(enable_nearest_points=True),result)
    return float(d),np.asarray(result.nearest_points)


class GraspMeshes:
    def __init__(self,kinematics):
        self.kin=kinematics;self.geometry=ArmGeometry(kinematics);self.meshes=[]
        root=ET.parse(ROOT/kinematics.model['urdf']).getroot()
        parents={j.find('child').get('link'):j.find('parent').get('link') for j in root.findall('joint')}
        def ancestors(link):
            result=[link]
            while link in parents:link=parents[link];result.append(link)
            return result
        ancestry={link.get('name'):ancestors(link.get('name')) for link in root.findall('link')}
        self.self_pairs=set()
        for a,b in combinations(ancestry,2):
            left,right=ancestry[a],ancestry[b]
            distance=min(left.index(link)+right.index(link) for link in set(left)&set(right))
            # Local bearing/shaft assemblies share surfaces; check separated
            # link assemblies for folding through one another.
            if distance>2:self.self_pairs.add(frozenset((a,b)))
        for link in root.findall('link'):
            for index,c in enumerate(link.findall('collision')):
                m=c.find('geometry/mesh')
                v=stl_triangles(m.get('filename'))*np.fromstring(m.get('scale','1 1 1'),sep=' ')
                tf=origin(c.find('origin'));v=v@tf[:3,:3].T+tf[:3,3]
                self.meshes.append((link.get('name'),index,triangle_model(v)))

    def at(self,q):
        transforms=self.geometry.transforms(q)
        return [(link,index,collision_object(model,transforms[link][:3,:3],transforms[link][:3,3])) for link,index,model in self.meshes]

    def jaws(self,q):
        names={'gripper_link','moving_jaw_so101_v1_link'} if self.kin.so else {'link6','link7'}
        return [(link,index,obj) for link,index,obj in self.at(q) if link in names]

    def inspect(self,q,kind,position,rotation=None):
        obj=collision_object(part_model(kind),np.eye(3) if rotation is None else rotation,position)
        distances=[]
        for link,index,jaw in self.jaws(q):
            d,p=separation(jaw,obj)
            distances.append({'link':link,'mesh':index,'distance_m':d,'nearest_points':p.tolist()})
        return distances


class PartClearanceGuard:
    """Check real part surfaces against arms and held parts against fixtures."""
    def __init__(self,plan):
        self.plan=plan
        self.arms={name:GraspMeshes(k) for name,k in plan.ik.items()}
        self.local={kind:np.asarray(vertices(kind)) for kind in ('housing','ball','cap')}
        self.fixture_objects={};self.cache={};self.part_cache={};self.pair_cache={};self.self_cache={};self.checked=0

    def check(self,t,poses,frames,fixtures):
        objects={};boxes={};owners={};part_keys={};positions={};rotations={}
        for name,history in self.plan.parts.items():
            _,owner,offset=self.plan.entry(history,t);owners[name]=owner
            if owner in poses and hasattr(self.plan,'rotations'):
                tf=self.plan.ik[owner].transform(poses[owner]);position=tf[:3,3]+tf[:3,:3]@offset
                rotation=tf[:3,:3]@self.plan.entry(self.plan.rotations[name],t)[2]
            else:
                position=self.plan.part_at(name,t)
                rotation=self.plan.part_rotation(name,t) if hasattr(self.plan,'part_rotation') else np.eye(3)
            key=(owner,*position,*rotation.ravel());part_keys[name]=key
            positions[name]=position;rotations[name]=rotation
            if self.part_cache.get(name,(None,))[0]!=key:
                kind=name.split('_')[0];v=self.local[kind]@rotation.T+position
                self.part_cache[name]=(key,(v.min(0),v.max(0)),collision_object(part_model(kind),rotation,position))
            _,boxes[name],objects[name]=self.part_cache[name]
        def near(a,b,margin=.00001):return np.all(a[1]+margin>=b[0]) and np.all(b[1]+margin>=a[0])
        for arm,frame in frames.items():
            key=tuple(poses[arm])
            if self.cache.get(arm,(None,))[0]!=key:self.cache[arm]=(key,{(l,i):o for l,i,o in self.arms[arm].at(poses[arm])})
            mesh_objects=self.cache[arm][1]
            if self.self_cache.get(arm)!=key:
                for a,b in combinations(frame,2):
                    if frozenset((a[0],b[0])) not in self.arms[arm].self_pairs:continue
                    if not near((a[4],a[5]),(b[4],b[5])):continue
                    d,_=separation(mesh_objects[a[0],a[1]],mesh_objects[b[0],b[1]]);self.checked+=1
                    if d<=.000001:raise RuntimeError(f'Arm self intersection: {arm}/{a[0]}/{b[0]} at {t:.3f}s')
                self.self_cache[arm]=key
            bounds=(np.min([s[4] for s in frame],axis=0),np.max([s[5] for s in frame],axis=0))
            for part,box in boxes.items():
                pair_key=(key,part_keys[part])
                if self.pair_cache.get((arm,part))==pair_key:continue
                foreign_payload=owners[part] in poses and owners[part]!=arm
                margin=float(self.plan.cfg.get('arm_clearance',.02)) if foreign_payload else .000001
                if not near(bounds,box,max(.00001,margin)):
                    self.pair_cache[arm,part]=pair_key;continue
                for shape in frame:
                    if not near((shape[4],shape[5]),box,max(.00001,margin)):continue
                    d,_=separation(mesh_objects[shape[0],shape[1]],objects[part]);self.checked+=1
                    if d<=margin:
                        reason='Arm too close to another arm payload' if foreign_payload else 'Gripper/arm intersects part'
                        raise RuntimeError(f'{reason}: {arm}/{shape[0]} / {part} at {t:.3f}s')
                self.pair_cache[arm,part]=pair_key
        def mating_contact(part,other):
            kind,kit=part.rsplit('_',1);other_kind,other_kit=other.rsplit('_',1)
            if kit!=other_kit or kind not in ('ball','cap') or other_kind not in ('housing','ball'):return False
            if owners[part] not in ('arm_c','fall') or owners[other]!='world':return False
            if np.linalg.norm(positions[other][:2]-np.asarray(self.plan.cfg['jig']))>.00005:return False
            if rotations[part][2,2]<.999999 or rotations[other][2,2]<.999999:return False
            offsets={'housing':0.,'ball':.003,'cap':.0203}
            delta=positions[part]-positions[other];height=offsets[kind]-offsets[other_kind]
            # Only the documented CAD seating surface, never a lower pose or
            # an arm/payload collision. Touching is necessary for assembly.
            return np.linalg.norm(delta[:2])<.00005 and height-1e-7<=delta[2]<=height+.00005
        for part,owner in owners.items():
            if owner not in poses and owner!='fall':continue
            for shape in fixtures:
                if not near(boxes[part],(shape[4],shape[5])):continue
                key=(shape[0],shape[1]);points=shape[2];stamp=points.tobytes()
                if self.fixture_objects.get(key,(None,))[0]!=stamp:
                    hull=ConvexHull(points)
                    obj=collision_object(triangle_model(points[hull.simplices].reshape(-1,3)),np.eye(3),np.zeros(3))
                    self.fixture_objects[key]=(stamp,obj)
                d,_=separation(objects[part],self.fixture_objects[key][1]);self.checked+=1
                if d<=.000001:raise RuntimeError(f'Held part intersects fixture: {part}/{shape[0]} at {t:.3f}s')
            for other in objects:
                margin=float(self.plan.cfg.get('arm_clearance',.02)) if owners[other] in poses and owners[other]!=owner else .000001
                if other==part or owners[other]==owner or not near(boxes[part],boxes[other],max(.00001,margin)):continue
                d,_=separation(objects[part],objects[other]);self.checked+=1
                if d<=margin and not mating_contact(part,other):raise RuntimeError(f'Held part clearance violation: {part}/{other} at {t:.3f}s')
