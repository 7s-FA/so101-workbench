import json
import unittest
import numpy as np
from arm_sim.common import ROOT,load_models
from arm_sim.demo_kinematics import ArmKinematics
from arm_sim.grasp_geometry import GraspMeshes


class GraspGeometryTests(unittest.TestCase):
    def test_profiles_fit_between_both_real_meshes_without_intersection(self):
        profiles=json.loads((ROOT/'config/demo_grasps.json').read_text())['profiles']
        for kind in ('so101','omx'):
            k=ArmKinematics(load_models()[kind],{'yaw':0,'position':[0,0]},.21)
            meshes=GraspMeshes(k)
            canonical=np.diag([1.,-1.,-1.]) if k.so else np.array([[0,1.,0],[0,0,-1.],[-1.,0,0]])
            for part,profile in profiles[kind].items():
                q=np.asarray(k.model['home']).copy();q[5:]=[profile['closed']] if k.so else [profile['closed'],-profile['closed']]
                tf=k.transform(q);position=tf[:3,3]+tf[:3,:3]@profile['part_base_in_tool']
                rotation=tf[:3,:3]@canonical.T
                components=[('housing',0),('ball',.003),('cap',.0203)] if part=='product' else [(part,0)]
                by_pad={}
                for component,z in components:
                    for pad in meshes.inspect(q,component,position+rotation@[0,0,z],rotation):
                        key=(pad['link'],pad['mesh'])
                        if key not in by_pad or pad['distance_m']<by_pad[key]['distance_m']:by_pad[key]=pad
                result=list(by_pad.values())
                pads=[d for d in result if not (d['link']=='gripper_link' and d['mesh']==0)]
                with self.subTest(model=kind,part=part):
                    self.assertEqual(len(pads),2)
                    for pad in pads:
                        self.assertGreater(pad['distance_m'],.0001)
                        self.assertLess(pad['distance_m'],.00025)

    def test_old_so101_housing_command_is_not_a_valid_grasp(self):
        k=ArmKinematics(load_models()['so101'],{'yaw':-np.pi/2,'position':[-.95,.78]},.21)
        q=np.array([1.04912448,1.33227403,-1.28843544,.603051881,-.000001,.25])
        result=GraspMeshes(k).inspect(q,'housing',[-1.3,.54,.215])
        self.assertTrue(any(d['distance_m']<=0 for d in result))
        moving=next(d for d in result if d['link']=='moving_jaw_so101_v1_link')
        self.assertGreater(moving['distance_m'],.01)
