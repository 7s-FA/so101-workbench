import unittest
from arm_sim.frame_clock import FrameClock,PlaybackClock


class FrameClockTests(unittest.TestCase):
    def test_camera_is_limited_in_wall_time_without_backlog(self):
        now=[0.];clock=FrameClock(15,clock=lambda:now[0])
        self.assertTrue(clock.due())
        for i in range(1,8):
            now[0]=i/120;self.assertFalse(clock.due())
        now[0]=8/120;self.assertTrue(clock.due())
        now[0]=5.;self.assertTrue(clock.due())
        self.assertFalse(clock.due())

    def test_fast_steps_are_paced_but_slow_steps_do_not_catch_up(self):
        now=[0.];sleeps=[]
        def sleep(delay):sleeps.append(delay);now[0]+=delay
        clock=FrameClock(120,clock=lambda:now[0],sleep=sleep)
        clock.wait();clock.wait();self.assertAlmostEqual(now[0],1/120)
        now[0]=1.;clock.wait();self.assertEqual(len(sleeps),1)
        clock.wait();self.assertAlmostEqual(now[0],1+1/120)

    def test_playback_tracks_wall_time_with_slow_render_frames(self):
        now=[0.];clock=PlaybackClock(clock=lambda:now[0]);position=0.
        # Variable frame time, including 80ms render stalls, below 120Hz.
        for i in range(1000):
            now[0]+=.08 if i%8==0 else .012
            position+=clock.request(position,8.,True)
        self.assertLess(abs(position-now[0]*8),.3)
        self.assertAlmostEqual(clock.elapsed_s,now[0])

    def test_one_times_playback_keeps_up_with_twenty_updates_per_second(self):
        now=[0.];clock=PlaybackClock(clock=lambda:now[0]);position=0.
        for _ in range(1200):
            now[0]+=.05
            position+=clock.request(position,1.,True)
        self.assertAlmostEqual(position,60.,places=6)

    def test_pause_reset_and_long_stall_do_not_create_unbounded_jumps(self):
        now=[0.];clock=PlaybackClock(clock=lambda:now[0])
        now[0]=10.;self.assertEqual(clock.request(0.,8.,False),0.)
        self.assertEqual(clock.elapsed_s,0.)
        now[0]+=.02;self.assertAlmostEqual(clock.request(0.,8.,True),.16)
        now[0]+=30.;self.assertLessEqual(clock.request(.16,8.,True),8/15)
        clock.reset();self.assertEqual(clock.elapsed_s,0.)
        now[0]+=.01;self.assertAlmostEqual(clock.request(0.,4.,True),.04)
