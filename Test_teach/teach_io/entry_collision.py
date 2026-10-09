"""Nominal URDF mesh check of current-pose entry, using the startup audit geometry.

Runs in Isaac Python before motor enable. Real cables, payloads and unmodeled
camera geometry are not certified by this discrete mesh check.
"""
import xml.etree.ElementTree as ET
from itertools import combinations
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from .entry import entry_samples


def origin(element):
    transform=np.eye(4)
    if element is not None:
        transform[:3,3]=np.fromstring(element.get('xyz','0 0 0'),sep=' ')
        transform[:3,:3]=Rotation.from_euler('xyz',np.fromstring(element.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return transform


class EntryCollisionAudit:
    def __init__(self,cfg,model,clamps):
        self.cfg=cfg;self.names=model['joints']
        tree=ET.parse(cfg['model_source']).getroot();self.joints=tree.findall('joint')
        self.moving=trimesh.collision.CollisionManager();self.world=trimesh.collision.CollisionManager()
        self.link_managers={}
        self.links=[]
        for link in tree.findall('link'):
            pieces=[]
            for collision in link.findall('collision'):
                element=collision.find('geometry/mesh')
                if element is None:raise ValueError('Unsupported entry collision geometry')
                mesh=trimesh.load(element.get('filename'),force='mesh')
                mesh.apply_scale(np.fromstring(element.get('scale','1 1 1'),sep=' '))
                mesh.apply_transform(origin(collision.find('origin')));pieces.append(mesh)
            if pieces:
                name=link.get('name');self.links.append(name)
                mesh=trimesh.util.concatenate(pieces)
                manager=trimesh.collision.CollisionManager();manager.add_object(name,mesh)
                self.link_managers[name]=manager
                if name!='link0':self.moving.add_object(name,mesh)
        self.excluded={frozenset((j.find('parent').get('link'),j.find('child').get('link'))) for j in self.joints}
        self.excluded.add(frozenset(('link6','link7')))
        self.self_pairs=[(a,b) for a,b in combinations(self.links,2) if frozenset((a,b)) not in self.excluded]
        self.joint_transforms={j.get('name'):origin(j.find('origin')) for j in self.joints}
        self.joint_axes={j.get('name'):np.fromstring(j.find('axis').get('xyz'),sep=' ')
                         for j in self.joints if j.get('type')!='fixed'}
        def obstacle(name,mesh,xyz):
            transform=np.eye(4);transform[:3,3]=xyz;self.world.add_object(name,mesh,transform)
        z=cfg['table']['height']
        obstacle('table',trimesh.creation.box([*cfg['table']['size'],.03]),[*cfg['table']['center'],z-.015])
        obstacle('floor',trimesh.creation.box([1.2,1.2,.03]),[0,0,-.015])
        for name,position,size in clamps:obstacle('clamp_'+name,trimesh.creation.box(size),position)
        for name,part in cfg['parts'].items():
            for area in ('source','destination'):
                mesh=trimesh.creation.box([.068,.068,.0035]) if name=='cap' else trimesh.load(part['jig'],force='mesh')
                obstacle(area+'_'+name,mesh,[*part[area],z+(.00175 if name=='cap' else 0)])
            obstacle(name,trimesh.load(part['mesh'],force='mesh'),
                     [part['source'][0]+part['offset_xy'][0],part['source'][1]+part['offset_xy'][1],z+part['seating_z']+.0005])

    def check(self,start,goal,mapping,allow_start_residual=False):
        samples=entry_samples(start,goal)
        if allow_start_residual:
            mapping.validate_observation(start);mapping.to_ticks(goal)
            # Audit both the actual observed geometry and the projected commands.
            samples=[pose for q in samples for pose in (q,mapping.clamp_target(q))]
        result=dict(collision_checked=False,start=list(start),goal=list(goal),samples=len(samples),
                    scope='Nominal URDF meshes; <=0.5 deg samples; no real cable, payload or external obstacle guarantee')
        for index,q in enumerate(samples):
            if allow_start_residual:mapping.validate_observation(q)
            else:mapping.to_ticks(q)
            values=dict(zip(self.names,[*q,-q[5]]))
            transforms={'link0':np.eye(4)};transforms['link0'][:3,3]=self.cfg['base']
            pending=list(self.joints)
            while pending:
                progressed=False
                for joint in pending[:]:
                    parent=joint.find('parent').get('link');child=joint.find('child').get('link')
                    if parent not in transforms:continue
                    transform=self.joint_transforms[joint.get('name')]
                    if joint.get('type')!='fixed':
                        rotation=np.eye(4)
                        rotation[:3,:3]=Rotation.from_rotvec(self.joint_axes[joint.get('name')]*values[joint.get('name')]).as_matrix()
                        transform=transform@rotation
                    transforms[child]=transforms[parent]@transform;pending.remove(joint);progressed=True
                if not progressed:raise ValueError('Disconnected entry URDF')
            for name in self.links:
                self.link_managers[name].set_transform(name,transforms[name])
                if name!='link0':self.moving.set_transform(name,transforms[name])
            # Query only non-excluded pairs and stop at the first contact. The old
            # name-returning broadphase collected up to 100,000 contacts, mostly
            # from deliberately adjoining links and the base touching the table.
            bad=[]
            for a,b in self.self_pairs:
                if self.link_managers[a].in_collision_other(self.link_managers[b]):
                    bad.append([a,b]);break
            if not bad and self.moving.in_collision_other(self.world):
                _,pairs=self.moving.in_collision_other(self.world,return_names=True)
                bad.extend(list(pair) for pair in pairs)
            if bad:
                result.update(blocked_sample=index,pairs=bad)
                return result
        result['collision_checked']=True
        return result
