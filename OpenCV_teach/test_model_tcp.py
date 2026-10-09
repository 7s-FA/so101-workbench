"""Model TCP alignment, independent MuJoCo FK, and persistent GUI settings."""
from copy import deepcopy
import json
import tempfile
import unittest

import numpy as np

import model_tcp as m
import vision_teaching as v


class ModelTipTests(unittest.TestCase):
    def test_tip_is_centre_of_the_fixed_distal_edge(self):
        reference=m.fixed_finger_tip()
        np.testing.assert_allclose(reference['xyz_mm'],[1.773409676,.602656734,6.549653900],atol=1e-6)
        np.testing.assert_allclose(np.mean(reference['edge_endpoints_mm'],axis=0),reference['xyz_mm'],atol=1e-9)
        self.assertAlmostEqual(reference['edge_width_mm'],11.172699233,places=6)
        self.assertFalse(reference['physically_verified'])

    def test_tcp_matches_assembled_mesh_in_mujoco_for_arm_and_gripper_poses(self):
        import mujoco
        model=mujoco.MjModel.from_xml_path(str(m.MODEL/'so101_modified.xml'))
        data=mujoco.MjData(model)
        kin=v.Kinematics(m.model_tip_tool(v.default_station({})['tool']))
        # The two STL end vertices, independently read from the pinned assembled mesh.
        mesh_point=np.array([-9.673392295837402,-.6024694442749023,105.62676620483398])
        import xml.etree.ElementTree as ET
        root=ET.parse(m.MODEL/'so101_modified.urdf').getroot()
        visual=root.find("link[@name='gripper_link']/visual[@name='xlerobot_fixed_finger']/origin")
        transform=m.origin_mm(visual)
        local=transform[:3,:3]@mesh_point+transform[:3,3]
        body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'gripper_link')
        for joints in ([0,0,0,0,0],[25,-30,35,15,-20],[-40,15,-20,-25,60]):
            fixed=[]
            for grip in (0,80):
                pose={name+'.pos':float(q) for name,q in zip(v.JOINTS,joints)}
                pose['gripper.pos']=grip
                for name,q in zip((*v.JOINTS,'gripper'),(*joints,grip)):
                    index=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
                    data.qpos[model.jnt_qposadr[index]]=q/100*1.74533 if name=='gripper' else np.radians(q)
                mujoco.mj_forward(model,data)
                point=data.xpos[body]*1000+data.xmat[body].reshape(3,3)@local
                np.testing.assert_allclose(kin.fk(pose)[:3,3],point,atol=.02)
                fixed.append(point)
            np.testing.assert_allclose(*fixed,atol=1e-9)

    def test_model_reference_preserves_axes_and_joint_corrections_without_verifying(self):
        old=v.default_station({})['tool'];old.update(rpy_deg=[1,2,3],joint_offsets_deg=[1,2,3,4,5],verified=True)
        original=deepcopy(old);new=m.model_tip_tool(old)
        self.assertEqual(old,original)
        self.assertEqual(new['rpy_deg'],old['rpy_deg'])
        self.assertEqual(new['joint_offsets_deg'],old['joint_offsets_deg'])
        self.assertFalse(new['verified'])
        self.assertTrue(m.reference_matches(new,new['reference']))

    def test_gui_saves_reference_and_drops_it_when_point_is_manually_changed(self):
        import tkinter as tk
        import teach_left_arm as teaching
        with tempfile.TemporaryDirectory() as folder:
            root=tk.Tk();app=teaching.App(root,teaching.Demo(),folder)
            try:
                view=app.open_vision();view.station['tool']['verified']=True;view.load_model_tip()
                view.save_tool();view.save_tool()
                saved=json.loads(view.profile_path.read_text())['tool']
                self.assertTrue(m.reference_matches(saved,saved['reference']))
                self.assertIn('고정 집게',view.tool_reference.get())
                view.vars['tool_xyz_mm0'].set('10');view.save_tool()
                self.assertNotIn('reference',view.station['tool'])
                self.assertFalse(view.station['tool']['verified'])
            finally:
                if app.vision:app.vision.close()
                for job in root.tk.call('after','info'):root.after_cancel(job)
                root.destroy()


class ModelExecutionReferenceTests(unittest.TestCase):
    def setUp(self):
        import model_tcp
        self.station=v.default_station({'follower_id':'current'})
        self.station['tool']=model_tcp.model_tip_tool(self.station['tool'])
        self.signature=v.calibration_signature(self.station)
        self.station['tcp_execution_reference']={'mode':'model_estimate',
            'tool_signature':v.digest(self.station['tool']),
            'identity_signature':v.digest(self.station['identity'])}
        self.plan={'targets':{},'calibration_signature':self.signature}

    def test_explicit_model_use_passes_without_claiming_physical_verification(self):
        v.validate_execution_setup(self.plan,self.station,live=True)
        self.assertFalse(self.station['tool']['verified'])
        self.assertFalse(self.station['tool']['reference']['physically_verified'])
        self.assertEqual(v.calibration_signature(self.station),self.signature)

    def test_changed_tcp_zero_or_robot_requires_current_plan(self):
        from copy import deepcopy
        for field in ('xyz_mm','joint_offsets_deg'):
            station=deepcopy(self.station);station['tool'][field][0]+=.1
            plan={**self.plan,'calibration_signature':v.calibration_signature(station)}
            with self.assertRaisesRegex(ValueError,'보정'):v.validate_execution_setup(self.plan,station,live=True)
            v.validate_execution_setup(plan,station,live=True)
        station=deepcopy(self.station);station['identity']['follower_id']='different'
        with self.assertRaisesRegex(ValueError,'보정'):v.validate_execution_setup(self.plan,station,live=True)

    def test_saved_geometry_does_not_require_verification_flags(self):
        from test_vision_teaching import target
        plan={**self.plan,'targets':{'slot':target()}}
        self.station.pop('tcp_execution_reference')
        v.validate_execution_setup(plan,self.station,live=True)
        self.assertFalse(self.station['tool']['verified'])
        self.assertFalse(plan['targets']['slot']['geometry_verified'])
        plan['targets']['slot']['geometry_verified']=True
        v.validate_execution_setup(plan,self.station,live=True)


if __name__=='__main__':unittest.main()
