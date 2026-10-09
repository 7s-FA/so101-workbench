import math
import unittest
from teach_io.motion_timing import MotionClock,motor_profile,VELOCITY_UNIT,ACCELERATION_UNIT,VELOCITY_RAD_S,ACCELERATION_RAD_S2
from teach_io.tracking import Tracking
from teach_io.mapping import Mapping
from test_follower import calibration


class TimingTests(unittest.TestCase):
    def test_usb_cycle_length_does_not_slow_requested_wall_clock_rate(self):
        for interval in (.02,.03,.04,.05):
            for scale in (.75,1.):
                clock=MotionClock(1/480);clock.steps(0,True,scale)
                steps=sum(clock.steps(i*interval,True,scale) for i in range(1,101))
                self.assertAlmostEqual(steps/480,100*interval*scale,delta=1/480)

    def test_pause_and_stall_are_not_replayed(self):
        clock=MotionClock(1/480);clock.steps(0,True)
        self.assertEqual(clock.steps(1,True),0)
        self.assertLessEqual(clock.steps(1.02,True),10)
        clock.steps(5,False)
        self.assertLessEqual(clock.steps(5.02,True),10)
        self.assertLessEqual(clock.steps(5.12,True),24)

    def test_load_pacing_and_zero_pace(self):
        clock=MotionClock(1/480);clock.steps(0,True)
        self.assertEqual(clock.steps(.04,True,1,.5),9)
        before=clock.remainder
        self.assertEqual(clock.steps(.08,True,1,0),0)
        self.assertEqual(clock.remainder,before)

    def test_motor_profiles_cover_audited_path_with_finite_limits(self):
        peak_v=[2.232,1.151,1.233,.622,2.232,1.407]
        peak_a=[4.124,5.774,4.870,1.359,4.124,4.331]
        for i,(v,a) in enumerate(zip(peak_v,peak_a)):
            pa,pv=motor_profile(i)
            self.assertGreaterEqual(pv*VELOCITY_UNIT,v)
            self.assertGreaterEqual(pa*ACCELERATION_UNIT,a)
            self.assertLessEqual(pv*VELOCITY_UNIT,VELOCITY_RAD_S[i])
            self.assertLessEqual(pa*ACCELERATION_UNIT,ACCELERATION_RAD_S2[i])

    def test_tracking_accepts_path_speed_but_still_limits_jumps(self):
        mapping=Mapping(calibration());q=[0,.1,.1,.1,0,.5]
        tracker=Tracking(mapping,q,0)
        for i in range(1,21):
            requested=q.copy();requested[0]=i*.02*2.232;requested[4]=requested[0]
            command,status=tracker.update(requested,requested,[{}]*6,i*.02)
            self.assertAlmostEqual(command[0],requested[0])
        target=command.copy();target[0]+=.15
        bounded,_=tracker.update(target,command,[{}]*6,.42)
        self.assertLessEqual(bounded[0]-command[0],VELOCITY_RAD_S[0]*.02+1e-9)
