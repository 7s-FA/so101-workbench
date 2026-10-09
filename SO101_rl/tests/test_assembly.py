import sys
from pathlib import Path
from copy import deepcopy
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import Reference,ROOT,read_json
from assembly import build_assembly,assembly_state,magnetic_wrenches
from scene_assets import PRODUCTS_DIR


class BAssemblyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        target=np.eye(4);target[:3,3]=[140,0,10]
        scene={'parts':{k:{'source':str(PRODUCTS_DIR/f'B {label}.stl'),'target_pose_mm':target.tolist()}
                         for k,label in [('housing','하단'),('tpu','중단'),('cap','상단')]},
               'workcell':{'linear_stage':{'position_mm':[0,0,0],'yaw_deg':0,'stroke_mm':100,
                    'carriage_home_x_mm':0,'a_fixture_asset':'assets/carrier_assembly/source/current/b_housing_jig.stl'}}}
        cls.assembly=build_assembly(scene,Reference(),read_json(ROOT/'config/assembly.json'))

    def poses(self):return {k:np.array(v) for k,v in self.assembly['target_poses_mm'].items()}

    def test_holes_and_nonzero_gap_come_from_cad(self):
        for seats in self.assembly['magnet_seats'].values():
            self.assertEqual(len(seats),4)
            for seat in seats:
                self.assertAlmostEqual(seat['diameter_mm'],6.3,places=3)
                self.assertAlmostEqual(seat['depth_mm'],2.2,places=3)
        self.assertAlmostEqual(self.assembly['cap_face_gap_mm'],1.6,places=3)

    def test_nested_state_accepts_gap_and_rejects_flush_cap(self):
        poses=self.poses();speeds=dict.fromkeys(poses,0.)
        state=assembly_state(poses,speeds,self.assembly)
        self.assertTrue(all(state[k] for k in ['housing_on_pallet','tpu_in_housing','cap_aligned']))
        poses['cap'][2,3]-=self.assembly['cap_face_gap_mm']
        self.assertFalse(assembly_state(poses,speeds,self.assembly)['cap_aligned'])

    def test_magnets_do_not_replace_pallet_or_insert_conditions(self):
        poses=self.poses();speeds=dict.fromkeys(poses,0.)
        poses['tpu'][0,3]+=10
        self.assertFalse(assembly_state(poses,speeds,self.assembly)['tpu_in_housing'])
        poses=self.poses()
        for pose in poses.values():pose[0,3]+=20
        self.assertFalse(assembly_state(poses,speeds,self.assembly)['housing_on_pallet'])

    def test_magnets_are_bounded_short_range_and_conserve_wrench(self):
        poses=self.poses();h,c=poses['housing'],poses['cap']
        h[:3,3]/=1000;c[:3,3]/=1000
        hc=h[:3,3]+[0,0,.01];cc=c[:3,3]+[0,0,-.005]
        fh,th,fc,tc,stats=magnetic_wrenches(h,c,hc,cc,self.assembly)
        self.assertEqual(stats['active_pairs'],4)
        self.assertGreater(stats['total_attraction_n'],.12)
        self.assertLessEqual(stats['total_attraction_n'],4*self.assembly['magnets']['max_pair_force_n'])
        np.testing.assert_allclose(fh+fc,0,atol=1e-12)
        np.testing.assert_allclose(th+tc+np.cross(hc,fh)+np.cross(cc,fc),0,atol=1e-12)
        c[0,3]+=.1
        self.assertEqual(magnetic_wrenches(h,c,hc,cc,self.assembly)[4]['active_pairs'],0)

    def test_wrong_facing_cap_is_not_attracted(self):
        poses=self.poses();h,c=poses['housing'],poses['cap'];h[:3,3]/=1000;c[:3,3]/=1000
        c[:3,:3]=h[:3,:3]
        self.assertEqual(magnetic_wrenches(h,c,h[:3,3],c[:3,3],self.assembly)[4]['active_pairs'],0)


if __name__=='__main__':unittest.main()
