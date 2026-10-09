"""Quasistatic gravity loads from URDF link masses, excluding unmeasured add-ons."""
import xml.etree.ElementTree as ET
import numpy as np
from arm_sim.common import ROOT
from arm_sim.demo_geometry import ArmGeometry


class StaticLoad:
    def __init__(self,kinematics):
        self.geometry=ArmGeometry(kinematics);self.inertials=[]
        for link in ET.parse(ROOT/kinematics.model['urdf']).getroot().findall('link'):
            inertia=link.find('inertial')
            if inertia is None:continue
            origin=inertia.find('origin')
            com=np.fromstring(origin.get('xyz','0 0 0') if origin is not None else '0 0 0',sep=' ')
            self.inertials.append((link.get('name'),float(inertia.find('mass').get('value')),com))

    def potential(self,q):
        transforms=self.geometry.transforms(q)
        return sum(9.81*mass*(transforms[link]@np.r_[com,1.])[2] for link,mass,com in self.inertials)

    def torques(self,q):
        q=np.asarray(q,dtype=float);step=1e-5
        return np.asarray([(self.potential(q+np.eye(len(q))[i]*step)-self.potential(q-np.eye(len(q))[i]*step))/(2*step) for i in range(5)])
