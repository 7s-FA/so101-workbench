import math
import unittest
from arm_sim.common import CommandGate, load_models

class Commands(unittest.TestCase):
    def setUp(self):
        self.m=load_models()['so101']
        self.m['max_command_speed']=0.8  # Explicit fixture; independent of tuning defaults.
        self.g=CommandGate(self.m)
    def test_names_not_array_order(self):
        vals=list(self.m['home']); vals[0]=0.2
        self.g.receive(self.m['joints'][::-1],vals[::-1],0)
        self.assertEqual(self.g.target,vals)
    def test_invalid_commands_do_not_refresh_watchdog(self):
        self.g.receive(self.m['joints'],self.m['home'],0)
        for v in [math.nan,math.inf,10.]:
            q=list(self.m['home']); q[0]=v
            with self.assertRaises(ValueError): self.g.receive(self.m['joints'],q,0.4)
        self.assertEqual(self.g.received,0)
    def test_timeout_holds_measured_position(self):
        q=list(self.m['home']); q[0]=1.
        self.g.receive(self.m['joints'],q,0)
        before=self.g.step(0.1,0.1,self.m['home'])
        self.assertAlmostEqual(before[0],0.08)
        measured=list(self.m['home']); measured[0]=0.03
        self.assertEqual(self.g.step(0.6,0.1,measured),measured)
        self.assertTrue(self.g.stale)
    def test_fast_target_ramp_reaches_90_degrees_without_overshoot(self):
        model=load_models()['so101']; gate=CommandGate(model)
        target=list(model['home']);target[0]=math.pi/2
        measured=list(model['home'])
        for i in range(72):
            gate.receive(model['joints'],target,i/120)
            q=gate.step(i/120,1/120,measured)
            measured=q
            self.assertLessEqual(q[0],target[0])
        self.assertAlmostEqual(q[0],target[0])

    def test_duplicate_missing_joint_rejected(self):
        with self.assertRaises(ValueError): self.g.receive(['shoulder_pan']*6,self.m['home'],0)
        with self.assertRaises(ValueError): self.g.receive(self.m['joints'][:5],self.m['home'][:5],0)

if __name__=='__main__': unittest.main()
