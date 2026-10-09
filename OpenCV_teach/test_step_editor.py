"""Step editing regression checks using real Tk widgets and the demo backend.

Run with a desktop display: python3 -m unittest -v test_step_editor
"""

import json
import tempfile
import time
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch

import teach_left_arm as app


class StepEditorTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk display required: {exc}")
        self.directory = tempfile.TemporaryDirectory()
        self.callback_errors = []
        self.root.report_callback_exception = lambda kind, error, trace: self.callback_errors.append(error)
        self.backend = app.Demo()
        self.ui = app.App(self.root, self.backend, self.directory.name)
        self.step = {"name": "step1", "pose": dict.fromkeys(app.KEYS, 0.), "captured_at": app.stamp()}
        self.step["pose"]["elbow_flex.pos"] = 1.234567891234
        self.ui.steps = [self.step]
        self.ui.refresh_steps(0)
        self.root.update()

    def tearDown(self):
        if not hasattr(self, "ui"):
            return
        self.ui.session.stop.set()
        self.ui.session.wakeup.set()
        if self.ui.session.thread:
            self.ui.session.thread.join(3)
        from tk_test_support import cancel_jobs
        cancel_jobs(self.root)
        self.root.destroy()
        self.directory.cleanup()
        self.assertEqual(self.callback_errors, [], "Tk callback failed")

    def open_editor(self):
        self.ui.buttons["edit_pose"].invoke()
        self.root.update()
        self.assertIsNotNone(self.ui.pose_editor)
        return self.ui.pose_editor

    def wait_for(self, condition, seconds=5):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.root.update()
            if condition():
                return
            time.sleep(.005)
        self.fail(f"Timed out in {self.ui.state}")

    def test_decimal_edit_persists_without_motion_or_rounding_other_joints(self):
        editor = self.open_editor()
        old_elbow = self.step["pose"]["elbow_flex.pos"]
        editor.values["shoulder_pan.pos"].set("-12.3456")
        editor.values["gripper.pos"].set("42.75")
        with patch.object(self.backend, "send", side_effect=AssertionError("Unexpected motion")):
            editor.apply_button.invoke()
        self.assertIsNone(self.ui.pose_editor)
        saved = json.loads(self.ui.program_path.read_text())["steps"][0]
        self.assertEqual(saved["pose"]["shoulder_pan.pos"], -12.3456)
        self.assertEqual(saved["pose"]["gripper.pos"], 42.75)
        self.assertEqual(saved["pose"]["elbow_flex.pos"], old_elbow)
        self.assertEqual(saved["captured_at"], self.step["captured_at"])
        self.assertEqual(self.backend.target, dict.fromkeys(app.KEYS, 0.))

    def test_motor_editor_matches_table_at_half_tick_without_rounding_degrees_first(self):
        from motor_positions import from_ticks, to_ticks
        key='shoulder_pan.pos'
        self.step['pose'][key]=from_ticks(self.backend,key,2080.5)
        exact=self.step['pose'][key]
        editor=self.open_editor()
        self.assertEqual(int(editor.motor_values[key].get()),to_ticks(self.backend,key,exact))
        editor.apply_button.invoke()
        self.assertEqual(self.step['pose'][key],exact)

    def test_calculated_joints_are_disabled_and_gripper_edit_updates_target(self):
        import vision_teaching as v
        from test_vision_teaching import target
        profile=target();station=v.default_station(self.backend.identity)
        self.ui.vision_plan={'targets':{'part':profile},'calibration_signature':v.calibration_signature(station)}
        self.ui.steps=v.make_steps('part',profile,self.step['pose']);self.ui.refresh_steps(2)
        editor=self.open_editor();original=self.ui.steps[2]['pose'].copy()
        for key in app.KEYS[:-1]:
            self.assertTrue(editor.fields[key].instate(['disabled']))
            self.assertTrue(editor.sliders[key].scale.instate(['disabled']))
            self.assertTrue(all(b.instate(['disabled']) for b in editor.adjust_buttons[key]))
        self.assertFalse(editor.fields['gripper.pos'].instate(['disabled']))
        editor.values['shoulder_pan.pos'].set('20');editor.apply_button.invoke()
        self.assertIs(self.ui.pose_editor,editor)
        self.assertEqual(self.ui.steps[2]['pose'],original)
        editor.reset();editor.values['gripper.pos'].set('22.5')
        with patch.object(self.backend,'send',side_effect=AssertionError('Edit moved robot')):
            editor.apply_button.invoke()
        self.assertIsNone(self.ui.pose_editor)
        self.assertEqual(self.ui.vision_plan['targets']['part']['closed'],22.5)
        self.assertEqual(self.ui.steps[2]['pose']['gripper.pos'],22.5)
        self.assertEqual(self.ui.steps[3]['pose']['gripper.pos'],22.5)
        self.assertEqual(self.ui.steps[2]['pose']['shoulder_pan.pos'],original['shoulder_pan.pos'])

    def test_invalid_input_keeps_original_step_and_editor_open(self):
        editor = self.open_editor()
        original = self.step["pose"].copy()
        for key, text in (("shoulder_pan.pos", ""), ("shoulder_pan.pos", "abc"),
                          ("shoulder_pan.pos", "nan"), ("shoulder_pan.pos", "inf"),
                          ("shoulder_pan.pos", "181"), ("gripper.pos", "-0.1"),
                          ("gripper.pos", "100.1")):
            with self.subTest(key=key, text=text):
                editor.reset()
                editor.values[key].set(text)
                editor.apply_button.invoke()
                self.assertIs(self.ui.pose_editor, editor)
                self.assertTrue(editor.message.get())
                self.assertEqual(self.step["pose"], original)
                self.assertFalse(self.ui.program_path.exists())

    def test_actual_calibration_validator_is_used(self):
        hardware = app.Hardware.__new__(app.Hardware)
        hardware.identity = self.backend.identity.copy()
        hardware.bounds = {key: (0, 100) if key == "gripper.pos" else (-20, 20) for key in app.KEYS}
        self.ui.backend = hardware
        editor = self.open_editor()
        editor.values["shoulder_pan.pos"].set("20.1")
        editor.apply_button.invoke()
        self.assertIn("보정 범위", editor.message.get())
        self.assertEqual(self.step["pose"]["shoulder_pan.pos"], 0.)
        editor.values["shoulder_pan.pos"].set("20")
        editor.apply_button.invoke()
        self.assertEqual(self.step["pose"]["shoulder_pan.pos"], 20.)

    def test_cancel_reset_noop_and_arrow_increment(self):
        editor = self.open_editor()
        self.root.update()
        field = editor.fields["shoulder_pan.pos"]
        field.event_generate("<<Increment>>")
        self.assertAlmostEqual(float(editor.values["shoulder_pan.pos"].get()), 360/4095)
        editor.reset()
        self.assertEqual(float(editor.values["shoulder_pan.pos"].get()), 0.)
        editor.values["gripper.pos"].set("70")
        editor.close()
        self.assertEqual(self.step["pose"]["gripper.pos"], 0.)
        editor = self.open_editor()
        editor.apply_button.invoke()
        self.assertFalse(self.ui.program_path.exists())

    def test_failed_save_preserves_pose_and_saved_teaching(self):
        self.ui.persist()
        before = self.ui.program_path.read_bytes()
        saved_path = self.ui.saved_path = Path(self.directory.name) / "existing.json"
        original = self.step["pose"].copy()
        editor = self.open_editor()
        editor.values["shoulder_pan.pos"].set("8.5")
        with patch.object(app, "atomic_json", side_effect=OSError("Disk full")):
            editor.apply_button.invoke()
        self.assertIn("저장 공간", editor.message.get())
        self.assertEqual(self.step["pose"], original)
        self.assertEqual(self.ui.saved_path, saved_path)
        self.assertEqual(self.ui.program_path.read_bytes(), before)

    def test_state_guards_and_stop_remain_available(self):
        for state in ("TEACH", "PLAY", "MOVE", "SYNCING", "STOPPING"):
            self.ui.state = state
            self.ui.refresh_buttons()
            self.assertTrue(self.ui.buttons["edit_pose"].instate(["disabled"]))
            with self.assertRaises(ValueError):
                self.ui.update_step_pose(self.step, self.step["pose"])
        self.ui.state = "HOLD"
        self.ui.refresh_buttons()
        editor = self.open_editor()
        self.assertTrue(self.ui.buttons["teach"].instate(["disabled"]))
        self.assertFalse(self.ui.buttons["stop"].instate(["disabled"]))
        self.assertIsNone(self.root.grab_current())
        editor.stop_and_close()
        self.assertTrue(self.ui.session.stop.is_set())
        self.assertIsNone(self.ui.pose_editor)

    def test_selection_change_does_not_edit_another_step(self):
        other = {"name": "step2", "pose": dict.fromkeys(app.KEYS, 2.), "captured_at": app.stamp()}
        self.ui.steps.append(other)
        self.ui.refresh_steps(0)
        self.root.update()
        editor = self.open_editor()
        self.ui.table.selection_set("1")
        self.root.update()
        editor.values["shoulder_pan.pos"].set("3")
        editor.apply_button.invoke()
        self.assertEqual(self.step["pose"]["shoulder_pan.pos"], 3.)
        self.assertEqual(other["pose"]["shoulder_pan.pos"], 2.)
        self.assertEqual(self.ui.selected(), 0)

    def test_record_edit_save_reload_and_play_in_demo(self):
        self.ui.steps = []
        self.ui.refresh_steps()
        self.ui.buttons["connect"].invoke()
        self.wait_for(lambda: self.ui.state == "HOLD")
        self.ui.buttons["teach"].invoke()
        self.wait_for(lambda: self.ui.state == "TEACH")
        self.ui.send("quick_capture")
        self.wait_for(lambda: len(self.ui.steps) == 1)
        self.ui.buttons["pause"].invoke()
        self.wait_for(lambda: self.ui.state == "HOLD")
        editor = self.open_editor()
        editor.values["shoulder_pan.pos"].set("3.25")
        editor.values["gripper.pos"].set("2.5")
        editor.apply_button.invoke()
        self.assertEqual(self.backend.target["shoulder_pan.pos"], 0.)
        with patch.object(app, "ask_teaching_title", return_value="편집한 데모"):
            self.ui.save_as()
        path = self.ui.saved_path
        self.assertIsNotNone(path)
        self.ui.load_path(path)
        self.assertEqual(self.ui.steps[0]["pose"]["shoulder_pan.pos"], 3.25)
        with patch("tkinter.messagebox.askokcancel", return_value=True):
            self.ui.buttons["play"].invoke()
        self.wait_for(lambda: self.ui.state == "PLAY")
        self.wait_for(lambda: self.ui.state == "HOLD")
        self.assertAlmostEqual(self.backend.current["shoulder_pan.pos"], 3.25, places=3)
        self.assertAlmostEqual(self.backend.current["gripper.pos"], 2.5, places=3)
        editor = self.open_editor()
        editor.values["shoulder_pan.pos"].set("3.5")
        editor.apply_button.invoke()
        self.assertIsNone(self.ui.saved_path)
        self.assertTrue(self.ui.buttons["play"].instate(["disabled"]))
        self.assertEqual(json.loads(path.read_text())["steps"][0]["pose"]["shoulder_pan.pos"], 3.25)


if __name__ == "__main__":
    unittest.main()
