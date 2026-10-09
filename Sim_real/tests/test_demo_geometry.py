import pickle
import unittest
import numpy as np
from arm_sim.demo_geometry import intersections


def box(center,half):
    center=np.array(center);half=np.array(half);normals=np.r_[np.eye(3),-np.eye(3)]
    equations=np.c_[normals,-np.r_[center+half,-center+half]]
    return [('box',0,None,equations,center-half,center+half)]


class GeometryTests(unittest.TestCase):
    def test_detects_penetration_and_enforces_clearance(self):
        a=box([0,0,0],[.05]*3)
        self.assertTrue(intersections(a,box([.09,0,0],[.05]*3)))
        self.assertFalse(intersections(a,box([.101,0,0],[.05]*3)))
        self.assertTrue(intersections(a,box([.101,0,0],[.05]*3),margin=.002))
        self.assertFalse(intersections(a,box([.11,0,0],[.05]*3),margin=.002))
