import unittest
from types import SimpleNamespace
import numpy as np
from arm_sim.factory import load_layout
from arm_sim.factory_demo import demo_layout,Track
from arm_sim.traffic import TrafficGuard,TrafficGate


class TrafficTests(unittest.TestCase):
    def plan(self):
        cfg=demo_layout(load_layout())
        return SimpleNamespace(cfg=cfg,vehicles={
            'burger_1':Track([-.95,.51,-np.pi/2]),
            'burger_2':Track([-.60,.10,np.pi/2]),
            'waffle_1':Track(cfg['waffle']['pose'])})

    def test_empty_feedback_fails_closed(self):
        p=self.plan();g=TrafficGuard(p)
        with self.assertRaisesRegex(RuntimeError,'feedback'):g.check(0,{})
        with self.assertRaisesRegex(RuntimeError,'feedback'):g.check(0,{'burger_1':[np.nan,0,0]})

    def test_marker_wall_is_checked_before_and_during_motion(self):
        from arm_sim.demo_validation import route_check
        p=self.plan();p.duration=.1
        p.cfg['docking_panels'][0]['center'][:2]=p.vehicles['burger_1'].at(0)[:2].tolist()
        with self.assertRaisesRegex(ValueError,'docking_panel'):route_check(p)
        with self.assertRaisesRegex(RuntimeError,'docking_panel'):TrafficGuard(p).check(0)

    def test_close_but_not_touching_vehicles_are_rejected(self):
        p=self.plan();g=TrafficGuard(p)
        poses={n:t.at(0) for n,t in p.vehicles.items()}
        poses['burger_1']=np.array([-.5,-.4,0.]);poses['burger_2']=np.array([-.285,-.4,0.])
        with self.assertRaisesRegex(RuntimeError,'clearance'):g.check(0,poses)
        poses['burger_2'][0]=-.25;g.check(0,poses)

    def test_gate_waits_for_readback_before_next_reservation(self):
        p=self.plan();p.traffic_jobs=[{'id':'one','vehicle':'burger_1','start':1.,'end':2.,'end_pose':[0,0,0]},
                                    {'id':'two','vehicle':'waffle_1','start':2.,'end':4.,'end_pose':[1,0,0]}]
        g=TrafficGate(p)
        self.assertEqual(g.advance(0.,3.,{}),1.)
        self.assertEqual(g.advance(1.,3.,{}),2.)
        self.assertEqual(g.advance(2.,3.,{}),2.)
        self.assertIsNotNone(g.hold_reason)
        self.assertEqual(g.advance(2.,3.,{'burger_1':np.array([0.,0.,0.])}),3.)
        self.assertEqual(g.completed,{'one'})
        g.reset();self.assertFalse(g.completed)

    def test_gate_rejects_overlapping_reservations(self):
        p=self.plan();p.traffic_jobs=[{'id':'one','vehicle':'burger_1','start':1.,'end':3.},
                                    {'id':'two','vehicle':'burger_2','start':2.,'end':4.}]
        with self.assertRaisesRegex(ValueError,'overlap'):TrafficGate(p)

if __name__=='__main__':unittest.main()
