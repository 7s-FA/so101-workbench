from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import tkinter as tk
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import teach_left_arm as app
import vision_teaching as v
from step_reference import change_reference,needs_measurement,used_plan
from safe_episode import with_safe_endpoints
from test_vision_teaching import target


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.station=v.default_station(app.Demo.identity)
        self.target=target();self.plan={'targets':{'t':self.target},'calibration_signature':v.calibration_signature(self.station)}
        self.pose=dict.fromkeys(app.KEYS,0.)
        self.steps=with_safe_endpoints(v.make_steps('t',self.target,self.pose),self.pose)

    def test_disable_restore_preserves_taught_joints_and_unused_targets_need_no_measurement(self):
        fixed,plan=change_reference(self.steps,self.plan,range(1,5),'고정 위치',self.station)
        self.assertFalse(needs_measurement(fixed,plan));self.assertEqual(used_plan(fixed,plan)['targets'],{})
        self.assertTrue(all('motion' not in s for s in fixed))
        restored,plan=change_reference(fixed,plan,range(1,5),'지그 따라가기',self.station)
        self.assertTrue(needs_measurement(restored,plan))
        self.assertEqual([s.get('motion') for s in restored],[s.get('motion') for s in self.steps])
        self.assertEqual([s['pose'] for s in restored],[s['pose'] for s in self.steps])
        doc={'schema':2,'identity':app.Demo.identity,'units':app.UNITS,'steps':fixed,'vision_plan':plan}
        loaded=app.validate_program(doc,app.Demo.identity)
        self.assertIn('follow_motion',loaded[1]);self.assertIn('fixed_pose',loaded[1])
        for mode,grip in [('지그 따라가기',False),('고정 위치',True)]:
            with self.assertRaisesRegex(ValueError,'안전 자세'):
                change_reference(self.steps,self.plan,[0],mode,self.station,gripper_only=grip)

    def test_arbitrary_taught_pose_binds_relative_tcp_and_fixed_pose_survives_recompile(self):
        kin=v.Kinematics(self.station['tool']);ob={'x_mm':200.,'y_mm':0.,'yaw_deg':0.}
        p=kin.ik(np.array([200,0,50.]),Rotation.from_euler('xyz',[180,0,180],degrees=True).as_matrix(),self.pose,45)
        source=[{'name':'측정한 자세','pose':p}]
        with self.assertRaisesRegex(ValueError,'지그'):
            change_reference(source,self.plan,[0],'지그 따라가기',self.station)
        bound,plan=change_reference(source,self.plan,[0],'지그 따라가기',self.station,profile=self.target,observation=ob)
        self.assertEqual(bound[0]['motion']['phase'],'waypoint')
        moved={**ob,'x_mm':203.,'y_mm':2.,'yaw_deg':3.}
        key=v.reference_key(self.target)
        compiled=v.compile_steps(bound,plan,self.station,{key:moved},p,single=True)
        t=plan['targets'][bound[0]['motion']['target']];xyz,rot=v.target_pose(t,{key:moved})
        actual=kin.fk(compiled[0]['pose'])
        np.testing.assert_allclose(actual[:3,3],xyz,atol=.5)
        np.testing.assert_allclose(actual[:3,:3],rot,atol=.02)
        fixed,_=change_reference(compiled,plan,[0],'고정 위치',self.station)
        self.assertEqual(fixed[0]['pose'],p)
        self.assertEqual(compiled[0]['pose']['gripper.pos'],45)

    def test_gripper_only_uses_actual_initial_arm_pose_in_validation_and_playback(self):
        from episode_preview import EpisodePlayer
        source=[{'name':'집게만','pose':{**self.pose,'gripper.pos':80}}]
        steps,plan=change_reference(source,self.plan,[0],'고정 위치',self.station,gripper_only=True)
        initial={**self.pose,'elbow_flex.pos':42,'wrist_flex.pos':20}
        goal=app.validate_sequence(app.Demo(),steps,initial)[0]
        self.assertEqual(goal,{**initial,'gripper.pos':80})
        document={'schema':2,'identity':app.Demo.identity,'units':app.UNITS,'steps':[{'name':'현재위치','pose':initial},*steps],'vision_plan':plan}
        player=EpisodePlayer();player.load(document)
        self.assertEqual(player.steps[1]['pose'],goal)

    def test_generated_fixed_pose_is_computed_pose_and_gripper_only_keeps_vision_sequence(self):
        key=v.reference_key(self.target);obs={key:{'x_mm':200.,'y_mm':0.,'yaw_deg':0.}}
        generated=v.compile_steps(self.steps,self.plan,self.station,obs,self.pose)
        for s in generated[1:-1]:self.assertEqual(s['fixed_pose'],s['pose'])
        self.assertNotEqual(generated[1]['fixed_pose'],self.pose)
        converted,plan=change_reference(generated,self.plan,[3],'고정 위치',self.station,gripper_only=True)
        self.assertNotIn('motion',converted[3]);self.assertTrue(converted[3]['gripper_only'])
        moved={key:{'x_mm':203.,'y_mm':2.,'yaw_deg':3.}}
        result=v.compile_steps(converted,plan,self.station,moved,self.pose)
        for joint in v.JOINTS:self.assertEqual(result[2]['pose'][joint+'.pos'],result[3]['pose'][joint+'.pos'])
        self.assertEqual(result[3]['pose']['gripper.pos'],self.target['closed'])
        self.assertTrue(result[4].get('path'))


class WorkflowUITests(unittest.TestCase):
    def setUp(self):
        self.root=tk.Tk();self.folder=tempfile.TemporaryDirectory()
        self.ui=app.App(self.root,app.Demo(),self.folder.name);self.view=self.ui.open_vision()
        self.view.add_steps();self.ui.steps=with_safe_endpoints(self.ui.steps,dict.fromkeys(app.KEYS,0.));self.ui.refresh_steps(1)
        self.root.update()

    def tearDown(self):
        if self.ui.reference_editor:self.ui.reference_editor.close()
        self.ui.close()
        owners=[]
        def collect(w):
            owners.append(w)
            for c in w.winfo_children():collect(c)
        collect(self.root)
        for job in self.root.tk.call('after','info'):
            script=self.root.tk.call('after','info',job)[0]
            next((w for w in owners if script in (w._tclCommands or [])),self.root).after_cancel(job)
        self.root.destroy();self.folder.cleanup()

    def test_settings_are_separate_and_group_editor_updates_auto_measurement_row(self):
        self.assertEqual(len(self.view.notebook.tabs()),2)
        self.assertEqual(len(self.view.settings_notebook.tabs()),4)
        self.assertTrue(self.ui.table.exists('measure'))
        self.view.select_tab(0);self.root.update()
        self.assertEqual(self.view.pages[0].winfo_toplevel(),self.root)
        self.assertIs(self.ui.navigation.current,self.view.settings_window)
        with patch.object(self.ui.session,'command') as send:
            self.ui.edit_step_reference();editor=self.ui.preview.step_settings
            editor.mode.set('고정 위치');editor.apply()
            self.assertIsNone(self.ui.reference_editor)
            self.assertFalse(self.ui.table.exists('measure'))
            self.assertTrue(all(not s.get('motion') for s in self.ui.steps))
            self.ui.edit_step_reference();self.ui.preview.step_settings.mode.set('지그 따라가기');self.ui.preview.step_settings.apply()
            self.assertTrue(self.ui.table.exists('measure'))
            self.ui.table.selection_set('0');self.ui.select();self.ui.edit_step_reference()
            self.assertEqual(str(self.ui.preview.step_settings.choice.cget('state')),'disabled')
            send.assert_not_called()

    def test_failed_editor_save_keeps_original_plan_and_steps(self):
        before=deepcopy((self.ui.steps,self.ui.vision_plan));self.ui.edit_step_reference()
        self.ui.preview.step_settings.mode.set('고정 위치')
        with patch.object(app,'atomic_json',side_effect=OSError('save failed')):self.ui.preview.step_settings.apply()
        self.assertEqual((self.ui.steps,self.ui.vision_plan),before)
        self.assertIn('save failed',self.ui.preview.step_settings.message.get())

    def test_fixed_episode_executes_without_camera_or_verified_tcp(self):
        self.ui.steps,self.ui.vision_plan=change_reference(self.ui.steps,self.ui.vision_plan,range(1,5),'고정 위치',self.view.station)
        self.ui.refresh_steps(0);self.ui.state='HOLD'
        self.ui.latest={'follower':dict.fromkeys(app.KEYS,0.),'sample_time':time.monotonic()}
        path=Path(self.folder.name)/'fixed.json';path.write_text(json.dumps(self.ui.document()));self.ui.saved_path=path
        with patch('tkinter.messagebox.askokcancel',return_value=True),patch.object(self.ui,'send') as send:
            self.ui.play();self.assertEqual(send.call_args.args[0],'play')
        self.assertFalse(self.view.frame_live);self.assertFalse(self.view.station['tool']['verified'])

    def test_detection_method_persists_without_marking_geometry_verified(self):
        self.view.vars['t_detection_method'].set('윤곽 중심');self.view.save_detection_method()
        self.assertTrue(all(t['detection_method']=='edge' for t in self.ui.vision_plan['targets'].values()))
        stored=json.loads(self.view.profile_path.read_text())
        self.assertEqual(next(iter(stored['jigs'].values()))['detection_method'],'edge')
        self.assertFalse(next(iter(stored['jigs'].values()))['geometry_verified'])
