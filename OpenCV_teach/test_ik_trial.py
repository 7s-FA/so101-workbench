"""Opt-in IK trial limits; no hardware I/O."""
import unittest
from copy import deepcopy
import numpy as np
from vision_teaching import Kinematics,JOINTS

class TrialLimitsTests(unittest.TestCase):
    def test_default_and_trial_are_isolated(self):
        trial=Kinematics(trial_tolerance={'position_mm':4,'angle_deg':6})
        normal=Kinematics()
        self.assertEqual((normal.position_tolerance,normal.angle_tolerance),(.5,1.))
        self.assertEqual((trial.position_tolerance,trial.angle_tolerance),(4.,6.))
        np.testing.assert_equal(trial.low,normal.low)
        np.testing.assert_equal(trial.high,normal.high)
    def test_unbounded_or_nonfinite_override_is_rejected(self):
        for pos,ang in [(4.1,6),(4,6.1),(float('nan'),6),(.5,float('inf'))]:
            with self.subTest(pos=pos,ang=ang),self.assertRaises(ValueError):
                Kinematics(trial_tolerance={'position_mm':pos,'angle_deg':ang})
    def test_exact_goal_still_matches_and_records_error(self):
        kin=Kinematics(trial_tolerance={'position_mm':4,'angle_deg':6})
        pose={**{n+'.pos':0. for n in JOINTS},'gripper.pos':50.}
        tf=kin.fk(pose);result=kin.ik(tf[:3,3],tf[:3,:3],pose,50.)
        self.assertLess(np.linalg.norm(kin.fk(result)-tf),1e-6)
        self.assertEqual(len(kin.ik_errors),1)
