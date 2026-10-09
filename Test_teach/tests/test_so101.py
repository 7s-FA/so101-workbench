import json,tempfile,unittest,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from teach_io.so101_model import ROOT,load,checked,frames,goal,measurements,save,solve,vertical_waypoints,lift_waypoints,joint_move_duration,teaching_steps,Kinematics,pose_from_angles,ORDER
from scipy.spatial.transform import Rotation

class SO101Tests(unittest.TestCase):
    def test_motor_ui_values_use_the_accepted_calibration_and_gripper_percent(self):
        from teach_io.so101_model import motor_value,angle_from_motor
        c=load();self.assertTrue(c['motor_reference']['accepted_by_user'])
        self.assertEqual(c['tcp']['joint_offsets_deg'],[0.]*5)
        for i,name in enumerate(('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll')):
            entry=c['motor_reference']['calibration'][name]
            for raw in (entry['range_min'],2047,entry['range_max']):
                self.assertEqual(motor_value(c,i,angle_from_motor(c,i,raw)),raw)
        self.assertAlmostEqual(motor_value(c,5,angle_from_motor(c,5,33.3)),33.3)

    def test_approach_clearance_moves_outward_without_rotating_gripper(self):
        c=load()
        for kind in ORDER:
            contact,r=goal(c,kind);approach,ar=goal(c,kind,approach=True)
            delta=frames(c,kind)[2][:3,:3].T@(approach-contact)
            np.testing.assert_allclose(delta,[c['parts'][kind]['approach_clearance_mm'],0,0],atol=1e-10)
            np.testing.assert_allclose(ar,r,atol=1e-10)

    def test_lift_clears_jig_before_tilting_and_reaches_requested_height(self):
        c=load();kin=Kinematics(c['tcp']);seed=np.r_[np.radians(c['home_deg'][:5]),.0]
        for kind in ORDER:
            grasp=solve(c,kind,seed);grasp[5]=0.
            xyz,orientation=goal(c,kind);path=lift_waypoints(c,kind,grasp)
            for i,q in enumerate(path):
                tf=kin.fk(pose_from_angles(q));height=(i+1)*c['parts'][kind]['lift_mm']/len(path)
                np.testing.assert_allclose(tf[:2,3],xyz[:2],atol=.5)
                self.assertAlmostEqual(tf[2,3]-xyz[2],height,delta=.5)
                if height<=c['parts'][kind]['lift_clearance_mm']:
                    self.assertLess(np.linalg.norm(Rotation.from_matrix(orientation.T@tf[:3,:3]).as_rotvec()),np.radians(1))
                self.assertEqual(q[5],grasp[5])
            self.assertLess(abs((kin.fk(pose_from_angles(path[-1]))[2,3]-xyz[2])-100),.5)

    def test_large_joint_moves_finish_below_streaming_speed_limit(self):
        start=np.radians([0,-55,15,-50,0,15]);end=np.radians([-20,15,0,95,80,55])
        duration=joint_move_duration(start,end)
        self.assertLessEqual(1.5*np.max(np.abs(end-start))/duration,.45000001)

    def test_exact_user_assets_and_noncentral_parts(self):
        c=load();self.assertEqual(c['part_order'],['cap','ball','housing'])
        self.assertEqual(c['contact']['thickness_mm'],.2)
        self.assertEqual(c['contact']['static_friction'],1.)
        for k in ORDER:
            _,jig,part=frames(c,k)
            offset=jig[:3,:3].T@(part[:3,3]-jig[:3,3])
            np.testing.assert_allclose(offset,c['parts'][k]['part_offset_jig_mm'],atol=1e-10)
            self.assertEqual(offset[1],4.)
            self.assertTrue(Path(c['parts'][k]['mesh']).is_file())
            self.assertTrue(Path(c['parts'][k]['jig_mesh']).is_file())
        # Top camera's right direction is -Y: cap, sphere, housing in that order.
        centers=[frames(c,k)[1][1,3] for k in ORDER]
        self.assertGreater(centers[0],centers[1]);self.assertGreater(centers[1],centers[2])

    def test_rigid_layout_rotation_preserves_offsets_and_distances(self):
        c=load();before=measurements(c)
        c['carrier']['yaw_deg']+=37.;c['carrier']['center_base_mm'][0]+=20
        after=measurements(c)
        for k in ORDER:
            self.assertAlmostEqual(before[k]['jig_distance_from_carrier_center_xy_mm'],after[k]['jig_distance_from_carrier_center_xy_mm'],places=8)
            np.testing.assert_allclose(before[k]['tcp_goal_carrier_mm'],after[k]['tcp_goal_carrier_mm'],atol=1e-9)

    def test_save_roundtrip_and_invalid_values_do_not_overwrite(self):
        c=load()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'layout.json';save(c,path);before=path.read_bytes()
            c['parts']['ball']['grasp_tcp_part_mm'][2]=float('nan')
            with self.assertRaises(ValueError):save(c,path)
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(load(path)['robot'],'so101')

    def test_mass_and_inertia_applied_with_two_pad_colliders(self):
        c=load();root=ET.parse(c['model_source']).getroot();total=0
        for link in root.findall('link'):
            inertial=link.find('inertial')
            if inertial is None:continue
            total+=float(inertial.find('mass').get('value'))
            i=inertial.find('inertia');matrix=np.array([[float(i.get(k)) for k in row] for row in [('ixx','ixy','ixz'),('ixy','iyy','iyz'),('ixz','iyz','izz')]])
            if link.get('name')!='gripper_frame_link':self.assertGreater(np.linalg.eigvalsh(matrix).min(),0)
        self.assertAlmostEqual(total,.636259356464,places=8)
        pads=[x for x in root.findall('.//collision') if x.get('name','').startswith('razer_')]
        self.assertEqual(len(pads),2)
        self.assertEqual(len(root.findall("joint[@type='revolute']")),6)

    def test_initial_cartesian_targets_reachable_for_all_parts(self):
        c=load();q=np.r_[np.radians(c['home_deg'][:5]),c['home_deg'][5]/100*1.74533]
        for k in ORDER:
            for phase in ('hover','grasp','lift'):
                result=solve(c,k,q,phase)
                self.assertTrue(np.isfinite(result).all())

    def test_descent_keeps_xy_and_gripper_open(self):
        c=load();kin=Kinematics(c['tcp']);seed=np.r_[np.radians(c['home_deg'][:5]),.8]
        for k in ORDER:
            hover=solve(c,k,seed,'hover');target,_=goal(c,k,approach=True)
            path=vertical_waypoints(c,k,hover,c['parts'][k]['hover_mm'],0.,approach=True)
            previous=kin.fk(pose_from_angles(hover))[:3,3]
            for q in path:
                pos=kin.fk(pose_from_angles(q))[:3,3]
                np.testing.assert_allclose(pos[:2],target[:2],atol=.5)
                self.assertLess(pos[2],previous[2]+.1)
                self.assertAlmostEqual(q[5],hover[5],places=6);previous=pos
            np.testing.assert_allclose(previous,target,atol=.5)

    def test_exported_steps_separate_height_from_gripper_action(self):
        c=load();seed=np.r_[np.radians(c['home_deg'][:5]),.8]
        steps=teaching_steps(c,'ball',seed)
        self.assertEqual([s['step'] for s in steps],['hover','descend','close','lift'])
        np.testing.assert_allclose(steps[1]['tcp_base_mm'],steps[2]['tcp_base_mm'],atol=1e-8)
        self.assertEqual(steps[1]['joint_pose']['gripper.pos'],c['parts']['ball']['open_percent'])
        self.assertEqual(steps[2]['joint_pose']['gripper.pos'],c['parts']['ball']['close_percent'])

if __name__=='__main__':unittest.main()
