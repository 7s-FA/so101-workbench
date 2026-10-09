"""Slider gestures, raw goal editing and gripper values without motor writes."""
from copy import deepcopy
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch
import teach_left_arm as t
from motor_slider import MotorSlider
from motor_positions import from_ticks


class SliderTests(unittest.TestCase):
    def setUp(self):
        self.root=tk.Tk();self.temp=tempfile.TemporaryDirectory();self.app=t.App(self.root,t.Demo(),self.temp.name)
        self.app.steps=[{'name':'저장 자세','pose':dict.fromkeys(t.KEYS,0.)}];self.app.refresh_steps(0)
    def tearDown(self):
        self.app.close()
        for job in self.root.tk.call('after','info'):self.root.after_cancel(job)
        self.root.destroy();self.temp.cleanup()
    def test_live_slider_only_sends_once_on_release_and_disabled_gesture_is_discarded(self):
        sent=[];slider=MotorSlider(self.root,50,'틱',sent.append);slider.frame.grid();slider.enable(True)
        self.root.update();slider.scale.set(17.3);self.root.update();self.assertEqual(sent,[])
        slider.scale.event_generate('<ButtonRelease-1>');self.root.update();self.assertEqual(sent,[17])
        slider.scale.event_generate('<ButtonRelease-1>');self.assertEqual(sent,[17])
        slider.scale.set(20);slider.enable(False);slider.release();self.assertEqual(sent,[17])
        fractional=MotorSlider(self.root,2,'mm',sent.append,step=.5);fractional.frame.grid()
        fractional.enable(True);fractional.scale.set(.6);fractional.release();self.assertEqual(sent[-1],.5)
    def test_step_sliders_edit_encoder_counts_but_preserve_untouched_angles_and_grip_units(self):
        self.app.steps[0]['pose']['elbow_flex.pos']=1.23456789
        self.app.edit_step_pose();editor=self.app.pose_editor
        with patch.object(self.app.backend,'send',side_effect=AssertionError('unexpected motion')):
            editor.sliders['shoulder_pan.pos'].scale.set(2100)
            editor.sliders['gripper.pos'].scale.set(42.7)
            self.root.update()
            self.assertEqual(editor.motor_values['shoulder_pan.pos'].get(),'2100')
            self.assertAlmostEqual(float(editor.values['shoulder_pan.pos'].get()),from_ticks(self.app.backend,'shoulder_pan.pos',2100))
            editor.apply()
        self.assertEqual(self.app.steps[0]['pose']['gripper.pos'],42.7)
        self.assertEqual(self.app.steps[0]['pose']['elbow_flex.pos'],1.23456789)
    def test_encoder_text_rejects_fractional_ticks_and_grip_target_sliders_never_move_arm(self):
        self.app.edit_step_pose();editor=self.app.pose_editor
        editor.motor_values['shoulder_pan.pos'].set('2100.5');editor.apply()
        self.assertIs(self.app.pose_editor,editor);self.assertTrue(editor.message.get());editor.close()
        view=self.app.open_vision(show=False)
        with patch.object(self.app,'send',side_effect=AssertionError('unexpected motion')):
            view.servo_sliders['t_open'].scale.set(76.5)
            view.servo_sliders['t_closed'].scale.set(12.3)
            self.root.update()
        self.assertEqual(float(view.vars['t_open'].get()),76.5)
        self.assertEqual(float(view.vars['t_closed'].get()),12.3)


if __name__=='__main__':unittest.main()
