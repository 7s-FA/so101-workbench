import sys
from copy import deepcopy
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import Reference
from alignment import aligned_scene,carrier_transform


class CADAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.ref=Reference()

    def scene(self):
        pose=np.eye(4);pose[:3,3]=[146,-195,1]
        return {'carrier':{'pose':self.ref.jig_pose,'bottom_z_mm':-7.4},
                'parts':{'housing':{'initial_pose_mm':pose.tolist(),'grasp_step':4,'release_step':8}}}

    def test_registration_moves_geometry_without_retargeting_teaching(self):
        source=self.scene();before=deepcopy(source);steps=deepcopy(self.ref.steps)
        changed=aligned_scene(source,self.ref,[8,4,2],-8,[1,-2])
        self.assertEqual(source,before);self.assertEqual(self.ref.steps,steps)
        self.assertEqual(changed['carrier']['pose'],self.ref.jig_pose)
        a=np.linalg.inv(carrier_transform(source,[0,0,0],0))@np.array(source['parts']['housing']['initial_pose_mm'])
        b=np.linalg.inv(carrier_transform(changed,[8,4,2],-8,[1,-2]))@np.array(changed['parts']['housing']['initial_pose_mm'])
        np.testing.assert_allclose(a,b,atol=1e-10)

    def test_offset_changes_are_absolute_and_reversible(self):
        source=self.scene()
        moved=aligned_scene(aligned_scene(source,self.ref,[8,4,2],-8),self.ref,[-5,6,-1],3)
        restored=aligned_scene(moved,self.ref,[0,0,0],0)
        np.testing.assert_allclose(restored['parts']['housing']['initial_pose_mm'],source['parts']['housing']['initial_pose_mm'],atol=1e-10)

    def test_invalid_alignment_is_rejected(self):
        for offset,yaw in [([31,0,0],0),([0,0,float('nan')],0),([0,0,0],13)]:
            with self.assertRaises(ValueError):aligned_scene(self.scene(),self.ref,offset,yaw)
        with self.assertRaises(ValueError):aligned_scene(self.scene(),self.ref,[0,0,0],0,[0,5])

    def test_carrier_registration_preserves_assembly_destination(self):
        scene=self.scene();goal=np.eye(4).tolist()
        scene['assembly']={'target_poses_mm':{'housing':goal}}
        changed=aligned_scene(scene,self.ref,[8,4,2],-8,[1,-2])
        self.assertEqual(changed['parts']['housing']['target_pose_mm'],goal)

    def test_linear_pallet_is_not_filtered_as_an_ear(self):
        from scene_assets import is_contact_geometry
        for name in ['linear_a_fixture_0_visual','linear_carriage_plate_visual','linear_rail_-1_0_visual']:
            self.assertTrue(is_contact_geometry(name),name)
        for name in ['linear_ear_positive_visual','linear_front_pin_shaft','linear_l12_rod_visual','linear_rail_screw_0_head']:
            self.assertFalse(is_contact_geometry(name),name)


if __name__=='__main__':unittest.main()
