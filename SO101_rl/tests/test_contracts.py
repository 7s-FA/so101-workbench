import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import Reference, JOINTS, bounded_target, task_success

class TeachingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref=Reference()

    def test_uses_episode_arm_instead_of_active_arm3(self):
        self.assertEqual(self.ref.profile.get('robot_id','arm2'),'arm2')
        self.assertEqual(self.ref.document['calibration_sha256'],self.ref.calibration.sha256)
        self.assertIsNotNone(self.ref.calibration.angle_mapping)

    def test_fixed_steps_and_gripper_preserved_by_jig_correction(self):
        for original,resolved in zip(self.ref.document['steps'],self.ref.steps):
            self.assertEqual(original['ticks']['gripper'],resolved['ticks']['gripper'])
            if original.get('jig_id') is None:
                self.assertEqual(original['ticks'],resolved['ticks'])

    def test_path_hits_each_saved_or_corrected_knot(self):
        for t,step in zip(self.ref.ends,self.ref.steps):
            actual,_=self.ref.sample(float(t))
            np.testing.assert_allclose(actual,self.ref.mapping.angles(step['ticks']),atol=1e-12)

    def test_model_mapping_round_trip(self):
        for step in self.ref.steps:
            self.assertEqual(self.ref.mapping.ticks_from_angles(self.ref.mapping.angles(step['ticks'])),step['ticks'])

    def test_bound_and_speed_limits(self):
        bounds=np.tile([-.1,.1],(6,1));start=np.zeros(6)
        target=bounded_target(np.ones(6),np.ones(6)*100,start,bounds,1/30)
        self.assertLessEqual(abs(target).max(),.65/30+1e-12)
        np.testing.assert_array_less(target,bounds[:,1]+1e-12)

    def test_nonfinite_action_rejected(self):
        with self.assertRaises(ValueError):
            bounded_target(np.zeros(6),[0,0,0,0,0,np.nan],np.zeros(6),np.tile([-1,1],(6,1)),1/30)

    def test_untouched_or_falling_parts_cannot_succeed(self):
        self.assertFalse(task_success([False]*3,[0]*3,[0]*3,True))
        self.assertFalse(task_success([True]*3,[0]*3,[0,.2,0],True))
        self.assertFalse(task_success([True]*3,[0]*3,[0]*3,False))
        self.assertTrue(task_success([True]*3,[3,4,5],[.001]*3,True))

    def test_upside_down_cap_seats_inside_registered_fixture(self):
        from scene_assets import mesh,product_orientation,seating_height,PRODUCTS_DIR
        from core import TEACH
        cap=mesh(PRODUCTS_DIR/'B 상단.stl')
        flipped=cap.copy();tf=product_orientation('cap',cap);flipped.apply_transform(tf)
        fixture=mesh(TEACH/'assets/carrier_assembly/source/current/b_cap_jig.stl')
        z=seating_height(flipped,fixture)
        self.assertAlmostEqual(tf[2,2],-1.)
        self.assertAlmostEqual(flipped.bounds[0,2],0.,places=6)
        # In this jig the flipped cap is almost flush with the top supports;
        # the unflipped cap sits more than 3 mm too high.
        self.assertLessEqual(abs(flipped.bounds[1,2]+z-fixture.bounds[1,2]),.1)
        self.assertGreater(seating_height(cap,fixture)-z,3.)

    def test_shifted_jig_tracks_transformed_taught_tcp(self):
        from so101_teach.geometry import transform_jig_pose
        pose=np.array(self.ref.jig_pose)+[2,1,.5]
        changed=self.ref.retarget(pose)
        count=0
        for original,resolved in zip(changed.document['steps'],changed.steps):
            if not original.get('jig_id'):
                self.assertEqual(original['ticks'],resolved['ticks'])
                continue
            target=transform_jig_pose(changed.kin.fk(original['ticks']),original['jig_reference']['pose'],pose)
            actual=changed.kin.fk(resolved['ticks'])
            self.assertLessEqual(np.linalg.norm(target[:3,3]-actual[:3,3]),3.)
            self.assertEqual(original['ticks']['gripper'],resolved['ticks']['gripper'])
            count+=1
        self.assertEqual(count,12)
        self.assertNotEqual(changed.steps,self.ref.steps)

    def test_repeated_retarget_does_not_accumulate(self):
        pose=np.array(self.ref.jig_pose)
        moved=self.ref.retarget(pose+[2,1,.5])
        moved=moved.retarget(pose+[-1,2,-.5])
        self.assertEqual(moved.retarget(pose).steps,self.ref.steps)

    def test_initial_pose_override_preserves_taught_reference(self):
        pose=np.asarray(self.ref.jig_pose)+[2,1,.5]
        initial=Reference(jig_pose=pose)
        self.assertEqual(initial.document,self.ref.document)
        self.assertEqual(initial.steps,self.ref.retarget(pose).steps)

    def test_bad_jig_pose_keeps_current_plan(self):
        from copy import deepcopy
        before=deepcopy(self.ref.steps)
        for pose in ([1,2,np.nan],[1,2], [2000,0,0], [900,900,0]):
            with self.assertRaises(ValueError):self.ref.retarget(pose)
        self.assertEqual(before,self.ref.steps)

if __name__=='__main__':unittest.main()
