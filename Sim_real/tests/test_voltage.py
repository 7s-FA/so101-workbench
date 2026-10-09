import unittest
from arm_sim.voltage import VoltageState

class VoltageTest(unittest.TestCase):
    def setUp(self):
        self.t=0.;self.s=VoltageState(lambda:self.t);self.s.limits(6,4.,12.)
    def sample(self,raw,status=0):
        data=[0]*15;data[6]=raw;data[7]=28;data[9]=status
        self.s.update(6,data,0,status)
    def test_single_sample_dip_survives_display_binning(self):
        self.sample(53);self.t=.016;self.sample(39,1);self.t=.032;self.sample(53)
        m=self.s.snapshot()['motors'][5]
        self.assertEqual(m['history'],[[0.,3.9,5.3]])
        self.assertEqual(m['minimum'],3.9);self.assertEqual(m['voltage'],5.3)
        self.assertEqual(m['error_samples'],1);self.assertEqual(m['fault_events'],1)
    def test_fault_events_are_separate_from_bad_sample_count(self):
        self.sample(39,1);self.t=.02;self.sample(39,1);self.t=.04;self.sample(53)
        self.t=.06;self.sample(39,1)
        m=self.s.snapshot()['motors'][5]
        self.assertEqual(m['error_samples'],3);self.assertEqual(m['fault_events'],2)
    def test_timeout_is_unknown_not_zero_volts(self):
        self.sample(53);self.t=.1;self.s.update(6,[],-6,0)
        m=self.s.snapshot()['motors'][5]
        self.assertIsNone(m['voltage']);self.assertTrue(m['stale'])
        self.assertEqual(m['minimum'],5.3);self.assertEqual(m['comm_errors'],1)
    def test_old_sample_is_not_live(self):
        self.sample(53);self.t=.6
        self.assertIsNone(self.s.snapshot()['motors'][5]['voltage'])
    def test_threshold_violation_without_status_bit_is_recorded(self):
        self.sample(39)
        self.assertEqual(self.s.snapshot()['motors'][5]['error_samples'],1)
if __name__=='__main__':unittest.main()
