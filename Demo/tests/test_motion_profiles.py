import unittest
from types import SimpleNamespace
import numpy as np
from arm_sim.factory_demo import Track, DemoPlan


class MotionProfiles(unittest.TestCase):
    def test_short_moves_obey_velocity_and_acceleration_bounds(self):
        plan=SimpleNamespace(time=0.,arms={'arm':Track([0.,0.],smooth=True)},
                             motion_limits=lambda *args:('empty',np.full(2,.8),np.full(2,2.)))
        DemoPlan.joints(plan,'arm',[.04,-.4],.08)
        t=np.linspace(0,plan.time,10001);q=np.array([plan.arms['arm'].at(x) for x in t])
        velocity=np.gradient(q,t,axis=0);acceleration=np.gradient(velocity,t,axis=0)
        self.assertLessEqual(float(abs(velocity).max()),.8001)
        self.assertLessEqual(float(abs(acceleration).max()),2.001)
        self.assertLess(float(abs(velocity[[0,-1]]).max()),.00001)
        self.assertLess(float(abs(acceleration[[0,-1]]).max()),.002)

    def test_vehicle_interpolation_is_preserved(self):
        track=Track([0.]);track.add(1.,[1.])
        self.assertAlmostEqual(track.at(.25)[0],.15625)

    def test_transit_knots_do_not_stop_but_action_endpoints_do(self):
        track=Track([0.,0.],smooth=True)
        for t in range(1,5):track.add(float(t),[.1*t,-.05*t])
        track.blend(0,4)
        eps=1e-4
        for t in (1.,2.,3.):
            v=(track.at(t+eps)-track.at(t-eps))/(2*eps)
            np.testing.assert_allclose(v,[.1,-.05],atol=1e-6)
        for t,sign in ((0.,1),(4.,-1)):
            self.assertLess(np.linalg.norm((track.at(t+sign*eps)-track.at(t))/eps),1e-6)

    def test_blended_reversal_stays_in_bounds_and_obeys_limits(self):
        plan=SimpleNamespace(time=0.,arms={'arm':Track([0.,0.],smooth=True)},
                             motion_limits=lambda *args:('empty',np.full(2,.8),np.full(2,2.)))
        for q in ([.02,.03],[.06,.07],[.10,.03],[.12,0.]):DemoPlan.joints(plan,'arm',q,.08)
        track=plan.arms['arm'];track.blend(0,len(track.times)-1)
        ts=np.linspace(0,plan.time,20001);qs=np.array([track.at(t) for t in ts])
        vs=np.gradient(qs,ts,axis=0);acc=np.gradient(vs,ts,axis=0)
        self.assertLessEqual(abs(vs).max(),.8001);self.assertLessEqual(abs(acc).max(),2.001)
        self.assertGreaterEqual(qs[:,0].min(),-1e-9);self.assertLessEqual(qs[:,0].max(),.12+1e-9)
        self.assertGreaterEqual(qs[:,1].min(),-1e-9);self.assertLessEqual(qs[:,1].max(),.07+1e-9)

    def test_left_edge_clamp_keeps_rails_outboard_and_jaws_over_edge(self):
        from arm_sim.factory import load_layout
        from arm_sim.demo_geometry import clamp_shapes
        cfg=load_layout();cfg['arms']['arm_b']['position']=[.16,.51]
        cfg['clamps']={'arm_b':{'edge':'left','edge_value':.09,'table':'assembly'}}
        shapes=clamp_shapes(cfg)['arm_b']
        for shape in shapes:
            center=(shape[4]+shape[5])/2
            if '/Rail' in shape[0]:self.assertLess(center[0],.09)
            if '/UpperJaw' in shape[0]:
                self.assertLess(shape[4][0],.09)
                self.assertGreater(shape[5][0],.09)

    def test_right_edge_clamp_keeps_rails_outboard_and_jaws_over_edge(self):
        from arm_sim.factory import load_layout
        from arm_sim.demo_geometry import clamp_shapes
        cfg=load_layout();cfg['arms']['arm_d']['position']=[1.30,.51]
        cfg['clamps']={'arm_d':{'edge':'right','edge_value':1.37,'table':'assembly'}}
        for shape in clamp_shapes(cfg)['arm_d']:
            center=(shape[4]+shape[5])/2
            if '/Rail' in shape[0]:self.assertGreater(center[0],1.37)
            if '/UpperJaw' in shape[0]:
                self.assertLess(shape[4][0],1.37)
                self.assertGreater(shape[5][0],1.37)
