from copy import deepcopy
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import model_zero as z
import vision_teaching as v
from calibration_rebase import convert_pose


def calibration():
    return {n:dict(id=i,drive_mode=0,homing_offset=10,range_min=800+i*25,range_max=3000+i*20)
            for i,n in enumerate((*z.JOINTS,'gripper'),1)}

def pose_at_ticks(cal,ticks):
    return {n+'.pos':(ticks[n]-(e['range_min']+e['range_max'])/2)*360/4095 if n!='gripper'
            else (ticks[n]-e['range_min'])/(e['range_max']-e['range_min'])*100 for n,e in cal.items()}

class ModelZeroTests(unittest.TestCase):
    def test_asymmetric_ranges_do_not_move_recorded_model_middle(self):
        mid=dict.fromkeys((*z.JOINTS,'gripper'),2047)
        for cal in [calibration(),{n:{**e,'range_min':e['range_min']+120} for n,e in calibration().items()}]:
            original=deepcopy(cal);tool=z.aligned_tool(v.default_station({})['tool'],cal,mid)
            p=pose_at_ticks(cal,mid)
            for i,n in enumerate(z.JOINTS):self.assertAlmostEqual(p[n+'.pos']+z.model_offsets(tool)[i],z.official_middle()['model_degrees'][n])
            self.assertEqual(cal,original)

    def test_fk_and_preview_share_middle_with_vertical_upper_and_forward_forearm(self):
        import mujoco
        from robot_preview import SCENE,pose_to_radians
        cal=calibration();mid=dict.fromkeys(cal,2047);tool=z.aligned_tool(v.default_station({})['tool'],cal,mid)
        pose=pose_at_ticks(cal,mid);m=mujoco.MjModel.from_xml_path(str(SCENE));d=mujoco.MjData(m)
        for i,(name,q) in enumerate(zip((*z.JOINTS,'gripper'),pose_to_radians(pose))):
            d.qpos[m.joint(name).qposadr[0]]=q+(np.deg2rad(z.model_offsets(tool)[i]) if i<5 else 0)
        mujoco.mj_forward(m,d)
        for a,b,axis in [('upper_arm_link','lower_arm_link',[0,0,1]),('lower_arm_link','wrist_link',[1,0,0])]:
            direction=-d.body(a).xmat.reshape(3,3)[:,0]
            np.testing.assert_allclose(direction,axis,atol=1e-5)
        np.testing.assert_allclose(d.xaxis[m.joint('gripper').id],[0,-1,0],atol=1e-5)
        np.testing.assert_allclose(v.Kinematics(tool).fk(pose)[:3,3],d.body('gripper_frame_link').xpos*1000,atol=.002)

    def test_import_preserves_model_pose_without_motor_rehoming(self):
        old=calibration();new={n:{**e,'homing_offset':30,'range_min':e['range_min']-10,'range_max':e['range_max']+50} for n,e in old.items()}
        tool=z.aligned_tool(v.default_station({})['tool'],old,dict.fromkeys(old,2047))
        rebased=z.carry_alignment(tool,old,new);pose={n+'.pos':15. for n in z.JOINTS};pose['gripper.pos']=40.
        converted=convert_pose(pose,old,new)
        np.testing.assert_allclose(v.Kinematics(tool).fk(pose),v.Kinematics(rebased).fk(converted),atol=1e-8)

    def test_preview_loads_current_station_without_camera_workspace(self):
        from robot_preview import preview_tool
        with tempfile.TemporaryDirectory() as tmp:
            app=SimpleNamespace(output=Path(tmp),backend=SimpleNamespace(identity={'demo':False}),vision=None,preview_only=False,vision_plan={})
            path=Path(tmp)/'vision'/(v.digest(app.backend.identity)[:16]+'.json');path.parent.mkdir()
            tool=z.aligned_tool(v.default_station({})['tool'],calibration(),dict.fromkeys(calibration(),2047))
            path.write_text(json.dumps({'identity':app.backend.identity,'tool':tool}))
            self.assertEqual(preview_tool(app),tool)
            app.preview_only=True;app.episode_model_tool={'joint_offsets_deg':[1]*5}
            self.assertEqual(preview_tool(app),app.episode_model_tool)

    def test_no_side_or_midpoint_guess_from_partial_reference(self):
        with self.assertRaises(ValueError):z.aligned_tool(v.default_station({})['tool'],calibration(),{'elbow_flex':2047})
        self.assertIsNone(z.carry_alignment(v.default_station({})['tool'],calibration(),calibration()))

    def test_fixed_episode_keeps_model_reference_without_camera_page(self):
        import tkinter as tk
        import teach_left_arm as t
        from robot_preview import preview_tool
        from tk_test_support import cancel_jobs
        with tempfile.TemporaryDirectory() as tmp:
            root=tk.Tk();app=t.App(root,t.Demo(),tmp)
            try:
                tool=z.aligned_tool(v.default_station({})['tool'],calibration(),dict.fromkeys(calibration(),2047))
                app.episode_model_tool=tool
                app.steps=[{'name':'reference','pose':dict.fromkeys(t.KEYS,0.)}]
                document=app.document();self.assertEqual(document['schema'],1)
                self.assertEqual(document['model_tool'],tool)
                file=Path(tmp)/'reference.json';t.atomic_json(file,document)
                app.episode_model_tool=None;app.load_path(file)
                self.assertEqual(preview_tool(app),tool)
            finally:app.close();cancel_jobs(root);root.destroy()

    def test_gripper_midpoint_is_right_angle_and_tick_delta_is_not_percent_angle(self):
        from robot_preview import pose_to_radians
        cal=calibration();mid=dict.fromkeys(cal,2047)
        tool=z.aligned_tool(v.default_station({})['tool'],cal,mid)
        p=pose_at_ticks(cal,mid)
        self.assertAlmostEqual(np.rad2deg(pose_to_radians(p,tool)[5]),90.)
        moved=dict(mid);moved['gripper']+=100
        p2=pose_at_ticks(cal,moved)
        self.assertAlmostEqual(np.rad2deg(pose_to_radians(p2,tool)[5])-90.,100*360/4095)
        new={n:{**e,'homing_offset':e['homing_offset']+20} for n,e in cal.items()}
        updated=z.carry_alignment(tool,cal,new)
        converted=convert_pose(p2,cal,new)
        np.testing.assert_allclose(pose_to_radians(p2,tool),pose_to_radians(converted,updated),atol=1e-10)

    def test_model_link_axes_not_joint_center_line_define_the_reference(self):
        reference=z.official_middle()['model_degrees']
        for name in ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex'):
            self.assertAlmostEqual(reference[name],0.,delta=.001)

    def test_zero_manual_corrections_preserve_recorded_zero_and_gripper_scale(self):
        from robot_preview import pose_to_radians
        cal=calibration();mid=dict.fromkeys(cal,2047)
        tool=z.aligned_tool(v.default_station({})['tool'],cal,mid)
        self.assertEqual(tool['joint_offsets_deg'],[0.]*5)
        p=pose_at_ticks(cal,mid)
        q=np.rad2deg(pose_to_radians(p,tool))
        np.testing.assert_allclose(q[:5],[z.official_middle()['model_degrees'][n] for n in z.JOINTS])
        self.assertAlmostEqual(q[5],90.)
        altered=deepcopy(tool);altered['joint_offsets_deg'][1]=1.
        delta=np.rad2deg(np.array(pose_to_radians(p,altered))-pose_to_radians(p,tool))
        np.testing.assert_allclose(delta,[0,1,0,0,0,0],atol=1e-10)
        altered['joint_offsets_deg']=[0.]*5
        np.testing.assert_allclose(pose_to_radians(p,altered),pose_to_radians(p,tool))

    def test_legacy_reference_and_invalid_new_reference(self):
        self.assertEqual(z.model_offsets({'joint_offsets_deg':[1,2,3,4,5]}),[1,2,3,4,5])
        for bad in [[0]*4,[0,0,0,0,float('nan')]]:
            with self.assertRaises(ValueError):z.model_offsets({'joint_reference_offsets_deg':bad})

    def test_safe_pose_compensates_both_reference_and_manual_correction(self):
        from safe_episode import model_safe_pose
        source={'joint_offsets_deg':[1,2,3,4,5]}
        tool={'joint_reference_offsets_deg':[3,4,5,6,7],'joint_offsets_deg':[0,1,0,0,0]}
        q=[10,20,30,40,50,60]
        result=model_safe_pose({'home_deg':q,'tcp':source},tool)
        np.testing.assert_allclose([result[n+'.pos']+z.model_offsets(tool)[i] for i,n in enumerate(z.JOINTS)],np.array(q[:5])+z.model_offsets(source))
