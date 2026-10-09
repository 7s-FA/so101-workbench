import unittest
from arm_sim.recovery import RecoveryGate
from arm_sim.common import CommandGate,load_models
class RecoveryTest(unittest.TestCase):
    def test_three_consecutive_good_samples_required_after_each_fault(self):
        g=RecoveryGate()
        self.assertFalse(g.update(True));self.assertFalse(g.update(True));self.assertTrue(g.update(True))
        self.assertFalse(g.update(False));self.assertFalse(g.update(True));self.assertFalse(g.update(False))
        self.assertFalse(g.update(True));self.assertFalse(g.update(True));self.assertTrue(g.update(True))
        self.assertEqual(g.faults,1);self.assertEqual(g.recoveries,2)
    def test_hold_cancels_pending_goal_and_recovery_is_rate_limited(self):
        m=load_models()['so101'];g=CommandGate(m);target=list(m['home']);target[0]=1.5
        g.receive(m['joints'],target,0);g.step(0,.01,m['home'])
        measured=list(m['home']);measured[0]=.02;g.hold(measured)
        self.assertEqual(g.step(.1,.01,measured),measured);self.assertTrue(g.stale)
        g.receive(m['joints'],target,.2);q=g.step(.2,.01,measured)
        self.assertAlmostEqual(q[0],.02+m['max_command_speed']*.01)
if __name__=='__main__':unittest.main()
