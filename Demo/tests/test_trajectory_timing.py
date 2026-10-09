import unittest
import numpy as np
from arm_sim.factory_demo import Track
from arm_sim.trajectory_timing import retime_tail


class ContinuousTimingTests(unittest.TestCase):
    def test_many_waypoints_do_not_introduce_repeated_stops(self):
        tr=Track([0.,0.],smooth=True)
        for i,x in enumerate(np.linspace(0,1,41)[1:],1):tr.add(i,[x,.5*x])
        retime_tail(tr,0,len(tr.times)-1,max_velocity=[1.,.5],max_acceleration=[2.,1.])
        self.assertLess(tr.times[-1],2.)
        span=tr.spans[0]
        t=np.linspace(0,span.duration,10001);q=span.at(t)
        dt=t[1]-t[0];v=np.gradient(q,dt,axis=0);a=np.gradient(v,dt,axis=0)
        self.assertLessEqual(float(np.max(abs(v)/[1.,.5])),1.001)
        self.assertLessEqual(float(np.max(abs(a)/[2.,1.])),1.002)
        np.testing.assert_allclose(span.spline([0,span.duration],1),0,atol=1e-7)
        for time,value in zip(tr.times,tr.values):np.testing.assert_allclose(tr.at(time),value,atol=1e-7)
        self.assertGreater(np.linalg.norm(span.spline(span.duration/2,1)),.8)

    def test_tiny_ik_correction_keeps_exact_endpoint_without_solver_failure(self):
        tr=Track([-.2,.05,.29],smooth=True)
        target=np.array([-.2,.05,.29])+[0.,5e-8,-7e-8]
        tr.add(1,target)
        retime_tail(tr,0,1,max_velocity=[2.4]*3,max_acceleration=[6.]*3)
        self.assertLess(tr.times[-1],.02)
        self.assertTrue(np.isfinite(tr.at(tr.times[-1]/2)).all())
        np.testing.assert_allclose(tr.at(tr.times[-1]),target,atol=1e-12)

    def test_carrying_profile_is_slower_and_append_does_not_change_prior_motion(self):
        fast=Track([0.,0.]);slow=Track([0.,0.])
        for tr in (fast,slow):
            tr.add(1,[.6,.2]);tr.add(2,[1.,.3])
        retime_tail(fast,0,2,1.8,4.,phase='empty')
        retime_tail(slow,0,2,1.2,2.5,phase='loaded')
        self.assertGreater(slow.times[-1],fast.times[-1])
        before=[fast.at(t) for t in np.linspace(0,fast.times[-1],20)]
        boundary=fast.times[-1];first=len(fast.times)-1
        fast.add(boundary+1,[.5,.1]);retime_tail(fast,first,len(fast.times)-1,.65,1.5,phase='precision')
        np.testing.assert_allclose(before,[fast.at(t) for t in np.linspace(0,boundary,20)],atol=1e-8)
        self.assertEqual(fast.spans[first].phase,'precision')
