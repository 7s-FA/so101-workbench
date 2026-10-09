import unittest
from arm_sim.haptics import ContactCue

class HapticsTests(unittest.TestCase):
    def run_frames(self,c,raw,angle=.8,force=2,start=10,n=40):
        out=None
        for i in range(n):
            now=start+i/60;out=c.step(raw,[now,angle,force,1],now)
        return out
    def test_contact_stop_does_not_follow_hand(self):
        c=ContactCue(1900,2400);first=self.run_frames(c,2100)
        second=self.run_frames(c,2000,angle=.7,start=11)
        self.assertEqual(first[0],second[0]);self.assertGreater(first[0],2140)
        self.assertEqual(second[1],100)
    def test_free_space_opening_spring(self):
        c=ContactCue(1900,2400)
        self.assertEqual(self.run_frames(c,2100,force=0),(2400,40));self.assertEqual(c.state,'opening_spring')
    def test_opening_releases_contact_stop(self):
        c=ContactCue(1900,2400);self.run_frames(c,2100)
        out=c.step(2120,[11,.8,2,1],11)
        self.assertIsNone(c.wall);self.assertEqual(out,(2400,40))
    def test_full_open_releases_motor(self):
        c=ContactCue(1900,2400);self.run_frames(c,2100)
        self.assertIsNone(c.step(2400,[11,1.1,0,1],11))
    def test_stale_invalid_disabled_release_immediately(self):
        for sample in (None,[10,.8,2,1],[11,.8,2,0],[11,float('nan'),2,1],[11],[11,.8,-1,1]):
            c=ContactCue(1900,2400);self.run_frames(c,2100)
            self.assertIsNone(c.step(2100,sample,11));self.assertEqual(c.current,0);self.assertIsNone(c.wall)
    def test_reverse_calibration(self):
        c=ContactCue(2400,1900);out=self.run_frames(c,2200)
        self.assertLess(out[0],2200)
    def test_current_ramps_and_recovery_starts_low(self):
        c=ContactCue(1900,2400);out=self.run_frames(c,2100,n=2)
        self.assertLessEqual(out[1],4)
        self.run_frames(c,2100,start=11)
        c.step(2100,None,12)
        out=c.step(2100,[12.01,.8,2,1],12.01)
        self.assertLessEqual(out[1],3)
