import math
import unittest
from arm_sim.common import CommandGate, load_models, relative_target_limits, limit_relative_target


class RelativeTargetTests(unittest.TestCase):
    def test_units_and_omx_unchanged(self):
        models=load_models();limits=relative_target_limits(models['so101'])
        self.assertAlmostEqual(limits[0],math.radians(10))
        self.assertAlmostEqual(limits[5],1.74533*.07)
        self.assertIsNone(relative_target_limits(models['omx']))

    def test_blocked_joint_does_not_accumulate_or_stop_other_joints(self):
        model=load_models()['so101'];gate=CommandGate(model)
        measured=list(model['home']);target=list(measured)
        target[0]=1.;target[1]=.7;target[5]=1.5
        for i in range(240):
            gate.receive(model['joints'],target,i/120)
            command=gate.step(i/120,1/120,measured)
            self.assertLessEqual(abs(command[0]-measured[0]),math.radians(10)+1e-12)
            self.assertLessEqual(abs(command[5]-measured[5]),1.74533*.07+1e-12)
            measured[1]=command[1]  # Only joint 2 follows; shoulder and grip stuck.
        self.assertAlmostEqual(command[0],math.radians(10))
        self.assertAlmostEqual(command[1],.7)
        # Fresh displacement changes the anchor, not the last sent command.
        measured[0]=-.05
        command=gate.step(239/120,1/120,measured)
        self.assertAlmostEqual(command[0],-.05+math.radians(10))

    def test_grip_points_not_taken_from_urdf_negative_lower_bound(self):
        model=load_models()['so101'];limits=relative_target_limits(model)
        self.assertNotAlmostEqual(limits[-1],(model['limits'][-1][1]-model['limits'][-1][0])*.07)

    def test_invalid_feedback_cannot_issue_command(self):
        with self.assertRaises(ValueError):limit_relative_target([1.],[math.nan],[.1])
        with self.assertRaises(ValueError):limit_relative_target([1.],[],[.1])

    def test_both_directions(self):
        self.assertEqual(limit_relative_target([-2.,2.],[.5,-.5],[.1,.2]),[.4,-.3])
