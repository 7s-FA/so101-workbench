import json
import math
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from teach_io.observation import PoseObservation
from teach_io.observe_worker import ReadOnlyBus
from teach_io.mapping import Mapping


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((ROOT/'config/follower_candidate.json').read_text())
        self.limits=[[-math.tau,math.tau]]*5+[[0,1.1]]

    def test_candidate_can_be_observed_but_not_used_for_actuation(self):
        view=PoseObservation(self.cfg)
        decoded=view.decode([2048]*5+[2039],self.limits)
        self.assertEqual(decoded['motor_angles_rad'],[0]*6)
        self.assertFalse(decoded['mapping_verified'])
        with self.assertRaises(ValueError):Mapping(self.cfg)

    def test_open_gripper_reports_model_clipping_without_rescaling(self):
        decoded=PoseObservation(self.cfg).decode([2048]*5+[3209],self.limits)
        self.assertAlmostEqual(decoded['motor_angles_rad'][5],1170*math.tau/4096)
        self.assertEqual(decoded['display_angles_rad'][5],1.1)
        self.assertTrue(decoded['limited'][5])
        self.assertLess(decoded['display_error_rad'][5],-.6)
        self.assertFalse(any(decoded['limited'][:5]))

    def test_positive_and_negative_quarter_turn(self):
        decoded=PoseObservation(self.cfg).decode([1024,3072,1024,3072,3072,2039],self.limits)
        self.assertEqual(decoded['motor_angles_rad'][:5],[-math.pi/2,math.pi/2,-math.pi/2,math.pi/2,math.pi/2])

    def test_malformed_telemetry_or_scale_rejected(self):
        view=PoseObservation(self.cfg)
        for values in ([1]*5,[1]*5+[float('nan')]):
            with self.assertRaises(ValueError):view.decode(values,self.limits)
        self.cfg['ticks_per_rad'][0]=0
        with self.assertRaises(ValueError):PoseObservation(self.cfg)

    def test_read_only_adapter_refuses_all_motor_writes(self):
        bus=ReadOnlyBus.__new__(ReadOnlyBus)
        for method,args in [(bus.write,(11,64,1,1)),(bus.positions,([2048]*6,)),(bus.enable,([],[]))]:
            with self.assertRaisesRegex(RuntimeError,'prohibits'):method(*args)
        bus.hold()  # explicitly a no-op; unlike the active follower close path


if __name__=='__main__':unittest.main()
