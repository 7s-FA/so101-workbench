import json
from pathlib import Path
from copy import deepcopy
import unittest
import numpy as np
from calibration_rebase import convert_value,convert_pose,raw_position,calibration_digest
from teach_left_arm import Hardware,validate_program
from vision_teaching import Kinematics,digest,calibration_signature

ROOT=Path(__file__).parent

class RebaseTests(unittest.TestCase):
    def setUp(self):
        self.report=json.loads((ROOT/'verification/20260922_zero_rebase/migration.json').read_text())
        self.old,self.new=self.report['old_calibration'],self.report['new_calibration']
    def test_fixed_safe_pose_preserves_physical_encoder_angle_through_wrap(self):
        before=self.report['safe_before'];after=convert_pose(before,self.old,self.new)
        for key,value in before.items():
            name=key.removesuffix('.pos');grip=name=='gripper'
            a=raw_position(self.old[name],value,grip)+self.old[name]['homing_offset']
            b=raw_position(self.new[name],after[key],grip)+self.new[name]['homing_offset']
            self.assertAlmostEqual((b-a+2048)%4096-2048,0.,places=9)
        self.assertAlmostEqual(after['wrist_roll.pos'],1.8241758241758241)
        self.assertAlmostEqual(after['shoulder_pan.pos'],2.989010989010989)
    def test_new_gripper_endpoint_is_explicit_and_out_of_range_poses_are_rejected(self):
        with self.assertRaises(ValueError):convert_value('gripper',0,self.old,self.new)
        self.assertEqual(convert_value('gripper',0,self.old,self.new,endpoints=True),0)
        with self.assertRaises(ValueError):convert_value('shoulder_pan',150,self.old,self.new)
        changed=deepcopy(self.new);changed['wrist_roll']['id']=99
        with self.assertRaises(ValueError):convert_value('wrist_roll',90,self.old,changed)
    def test_scene_rebase_preserves_geometry_and_converts_registered_safe_pose(self):
        # Regression inputs are historical snapshots, never the operator's live
        # calibration file: the operator can recalibrate while tests are running.
        from calibration_rebase import rebase_scene,motor_limits
        before=self.report['before_documents']
        scene=deepcopy(next(v for k,v in before.items() if k.endswith('so101_scene.json')))
        station=deepcopy(next(v for k,v in before.items() if '/vision/' in k))
        safe=convert_pose(self.report['safe_before'],self.old,self.new)
        rebased=rebase_scene(scene,station,self.old,self.new,safe,Path('/fixture/follower.json'))
        self.assertEqual(rebased['safe_deg'],list(safe.values()))
        self.assertEqual(rebased['home_deg'],rebased['safe_deg'])
        self.assertEqual(rebased['motor_reference']['calibration_sha256'],calibration_digest(self.new))
        self.assertEqual(rebased['tcp']['motor_limits_deg'],motor_limits(self.new))
        for key in ('xyz_mm','rpy_deg','joint_offsets_deg'):
            self.assertEqual(rebased['tcp'][key],station['tool'][key])
        for key in ('table','base_mm','carrier'):
            self.assertEqual(rebased[key],scene[key])
        self.assertNotEqual(rebased['motor_reference'].get('source_calibration_sha256'),rebased['motor_reference']['calibration_sha256'])

    def test_kinematics_intersects_motor_and_model_limits_instead_of_relaxing_them(self):
        plain=Kinematics();tool=deepcopy(plain.tool);tool['motor_limits_deg']=[[-10,10]]*5
        kin=Kinematics(tool)
        self.assertTrue(np.all(kin.low>=-10));self.assertTrue(np.all(kin.high<=10))
        tool['motor_limits_deg']=[[20,-20]]*5
        with self.assertRaises(ValueError):Kinematics(tool)

if __name__=='__main__':unittest.main()
