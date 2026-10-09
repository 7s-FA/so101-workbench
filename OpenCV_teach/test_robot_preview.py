"""Pose mapping, stale-data handling and actual demo-to-MuJoCo integration.

Run in a desktop session with the optional preview environment:
  .preview-venv/bin/python -m unittest -v test_robot_preview
"""

import importlib.util
import json
import math
from pathlib import Path
import queue
import tempfile
import time
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import robot_preview as preview
import teach_left_arm as teaching


def zero_pose():
    return dict.fromkeys(teaching.KEYS, 0.)


class PoseMappingTests(unittest.TestCase):
    def test_midrange_degrees_and_gripper_closed_open(self):
        pose = zero_pose()
        pose.update({"shoulder_pan.pos": 90, "wrist_roll.pos": -180})
        radians = preview.pose_to_radians(pose)
        self.assertAlmostEqual(radians[0], math.pi / 2)
        self.assertAlmostEqual(radians[4], -math.pi)
        self.assertEqual(radians[5], 0.)  # Not the CAD minimum, -10 degrees.
        pose["gripper.pos"] = 100
        self.assertAlmostEqual(preview.pose_to_radians(pose)[5], 1.74533)
        # Outside the CAD body range remains visible, not silently clamped.
        pose["shoulder_pan.pos"] = 170
        self.assertAlmostEqual(preview.pose_to_radians(pose)[0], math.radians(170))

    def test_incomplete_nonfinite_and_out_of_range_input_rejected(self):
        for key, value in (("elbow_flex.pos", ""), ("elbow_flex.pos", "nan"),
                           ("wrist_roll.pos", float("inf")), ("gripper.pos", -1),
                           ("gripper.pos", 101), ("shoulder_pan.pos", 181)):
            with self.subTest(key=key, value=value):
                pose = zero_pose()
                pose[key] = value
                with self.assertRaises(ValueError):
                    preview.pose_to_radians(pose)
        with self.assertRaises(ValueError):
            preview.pose_to_radians({})

    def test_live_measurement_not_command_and_stale_data_hidden(self):
        sample = zero_pose()
        sample["shoulder_pan.pos"] = 7
        app = SimpleNamespace(state="PLAY", latest={"sample_time": 10., "follower": sample},
                              backend=SimpleNamespace(identity={"demo": True}))
        pose, _, text = preview.choose_pose(app, preview.CURRENT, now=10.01)
        self.assertEqual(pose, sample)
        self.assertIsNot(pose, sample)
        self.assertIn("데모", text)
        self.assertIn("실행", text)
        self.assertIsNone(preview.choose_pose(app, preview.CURRENT, now=12)[0])
        app.state = "DISCONNECTED"
        self.assertIsNone(preview.choose_pose(app, preview.CURRENT, now=10.01)[0])

    def test_selected_step_and_unapplied_editor_take_precedence_offline(self):
        step = {"name": "준비", "pose": zero_pose()}
        app = SimpleNamespace(pose_editor=None, selected=lambda: 0, steps=[step])
        pose, token, _ = preview.choose_pose(app, preview.TARGET)
        self.assertEqual(pose, step["pose"])
        edited = dict(step["pose"], **{"elbow_flex.pos": "12.3"})
        app.pose_editor = SimpleNamespace(step=step, values={key: SimpleNamespace(get=lambda v=value: v)
                                                            for key, value in edited.items()})
        draft, draft_token, text = preview.choose_pose(app, preview.TARGET)
        self.assertNotEqual(token, draft_token)
        self.assertEqual(draft["elbow_flex.pos"], "12.3")
        self.assertEqual(step["pose"]["elbow_flex.pos"], 0)
        self.assertIn("적용 전", text)


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "Optional MuJoCo environment required")
class RenderProcessTests(unittest.TestCase):
    def test_modified_fingers_move_without_wrist_camera_or_mount(self):
        import mujoco
        import numpy as np
        model = mujoco.MjModel.from_xml_path(str(preview.SCENE))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for name in ("xlerobot_fixed_holder", "xlerobot_fixed_finger", "xlerobot_moving_holder",
                     "xlerobot_moving_finger"):
            self.assertGreaterEqual(model.geom(name).id, 0)
        for i in range(model.ngeom):
            self.assertFalse(any(word in model.geom(i).name.lower()
                                 for word in ('d435', 'camera', 'holderscrewhead')))
        fixed = data.geom("xlerobot_fixed_finger").xpos.copy()
        moving = data.geom("xlerobot_moving_finger").xpos.copy()
        data.qpos[model.joint("gripper").qposadr[0]] = 1.74533
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.geom("xlerobot_fixed_finger").xpos, fixed, atol=1e-9)
        self.assertGreater(np.linalg.norm(data.geom("xlerobot_moving_finger").xpos-moving), .02)
        data.qpos[model.joint("wrist_roll").qposadr[0]] = -.9
        mujoco.mj_forward(model, data)
        self.assertGreater(np.linalg.norm(data.geom("xlerobot_fixed_finger").xpos-fixed), .01)

    def wait_item(self, client, seconds=10):
        return client.inbox.get(timeout=seconds)

    def finish_client(self, client):
        client.close()
        self.assertTrue(client.finished.wait(5))
        self.assertFalse(client.process.is_alive())

    def test_real_meshes_render_and_movement_changes_pixels(self):
        client = preview.RenderClient()
        try:
            pose = zero_pose()
            client.submit(("first", preview.pose_to_radians(pose), preview.DEFAULT_CAMERA))
            first = self.wait_item(client)
            self.assertEqual(first[0], "frame", first[:2])
            self.assertTrue(first[4].startswith(f"P6\n{preview.WIDTH} {preview.HEIGHT}\n255\n".encode()))
            pose.update({"shoulder_pan.pos": 150, "gripper.pos": 100})
            client.submit(("second", preview.pose_to_radians(pose), preview.DEFAULT_CAMERA))
            second = self.wait_item(client)
            self.assertEqual(second[0], "frame", second[:2])
            self.assertNotEqual(first[4], second[4])
            self.assertIn("shoulder_pan", second[3])
            self.assertAlmostEqual(second[2][0], math.radians(150))
        finally:
            self.finish_client(client)

    def test_model_load_failure_is_reported_without_hanging(self):
        client = preview.RenderClient(scene_path="/nonexistent/so101.xml")
        try:
            item = self.wait_item(client)
            self.assertEqual(item[0], "error")
            self.assertIn("so101.xml", item[1])
        finally:
            self.finish_client(client)

    def test_render_size_grows_past_scene_defaults_and_shrinks(self):
        client = preview.RenderClient()
        try:
            angles = preview.pose_to_radians(zero_pose())
            for size in ((961, 541), (321, 239), (1000, 600)):
                with self.subTest(size=size):
                    client.submit(("resize", angles, preview.DEFAULT_CAMERA, size))
                    item = self.wait_item(client)
                    self.assertEqual(item[0], "frame", item[:2])
                    self.assertEqual(item[5], size)
                    header, dimensions, maximum, pixels = item[4].split(b"\n", 3)
                    self.assertEqual((header, dimensions, maximum),
                                     (b"P6", f"{size[0]} {size[1]}".encode(), b"255"))
                    self.assertEqual(len(pixels), size[0]*size[1]*3)
        finally:
            self.finish_client(client)


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "Optional MuJoCo environment required")
class PreviewIntegrationTests(unittest.TestCase):
    def setUp(self):
        import gc
        gc.collect()  # Keep old Tcl interpreters out of renderer/session workers.
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        self.errors = []
        self.root.report_callback_exception = lambda kind, error, trace: self.errors.append(error)
        self.root.tk.call("tk", "scaling", 1.3333)
        self.temp = tempfile.TemporaryDirectory()
        self.app = teaching.App(self.root, teaching.Demo(), self.temp.name)
        self.app.steps = [{"name": "준비 자세", "pose": zero_pose(), "captured_at": teaching.stamp()}]
        self.app.refresh_steps(0)
        self.app.show_preview()
        self.view = self.app.preview
        self.client = self.view.client

    def wait_for(self, condition, seconds=10):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.root.update()
            if condition():
                return
            if self.view.render_error:
                self.fail(self.view.render_error)
            time.sleep(.005)
        self.fail(f"Timeout: {self.view.status.get()}, {self.app.state}, rendered={self.view.last_frame_angles}, requested={self.view.last_request[:4] if self.view.last_request else None}, viewport={self.view.viewport}, actual_canvas={(self.view.canvas.winfo_width(), self.view.canvas.winfo_height())}, editor={ {k:v.get() for k,v in self.app.pose_editor.values.items()} if self.app.pose_editor else None}")

    def tearDown(self):
        if not hasattr(self, "app"):
            return
        self.app.close()
        self.assertTrue(self.client.finished.wait(5))
        self.assertFalse(self.client.process.is_alive())
        if self.app.session.thread:
            self.app.session.thread.join(4)
        from tk_test_support import cancel_jobs
        cancel_jobs(self.root)
        self.root.destroy()
        self.temp.cleanup()
        self.assertEqual(self.errors, [])

    def click_widget(self, widget, x=5, y=5):
        # Separate synthetic clicks so Tk does not interpret them as double-clicks.
        self.click_time = getattr(self, "click_time", 0) + 1000
        widget.event_generate("<Enter>", x=x, y=y)
        widget.event_generate("<ButtonPress-1>", x=x, y=y, time=self.click_time)
        self.root.update()
        widget.event_generate("<ButtonRelease-1>", x=x, y=y, time=self.click_time+50)
        self.root.update()

    def test_clicking_outside_selected_step_returns_to_live_pose_and_clears_selection(self):
        self.app.connect()
        self.wait_for(lambda: self.app.state == "HOLD")
        self.app.steps[0]["pose"]["shoulder_pan.pos"] = 25.
        self.root.update()
        x, y, width, height = self.app.table.bbox("0")
        with patch.object(self.app.session, "command", side_effect=AssertionError("Click queued control")), \
             patch.object(self.app.backend, "connect", side_effect=AssertionError("Click opened ports")):
            for widget, outside_x, outside_y in (
                    (self.app.joints, 5, 5),
                    (self.app.table, 20, self.app.table.winfo_height()-10),
                    (self.app.shortcut_hint, 5, 5)):
                with self.subTest(outside=widget):
                    self.click_widget(self.app.table, x+width//2, y+height//2)
                    self.assertEqual(self.view.mode.get(), preview.TARGET)
                    self.wait_for(lambda: self.view.last_frame_angles == preview.pose_to_radians(self.app.steps[0]["pose"]))
                    self.click_widget(widget, outside_x, outside_y)
                    self.assertEqual(self.view.mode.get(), preview.CURRENT)
                    self.assertIsNone(self.app.selected())
                    self.assertEqual(self.app.table.selection(), ())
                    self.assertEqual(self.app.table.focus(), "")
                    self.assertEqual(self.app.detail.get(), "스텝을 선택하면 저장된 관절값을 표시합니다.")
                    for joint in teaching.JOINTS:
                        self.assertEqual(self.app.joints.set(joint, "step"), "—")
                    for action in ("edit_pose", "move", "delete"):
                        self.assertTrue(self.app.buttons[action].instate(["disabled"]), action)
                    self.wait_for(lambda: self.view.last_frame_angles == preview.pose_to_radians(self.app.latest["follower"]))
        self.assertEqual(self.app.state, "HOLD")

    def test_edit_immediately_after_loading_episode_shows_draft_pose(self):
        self.app.connect()
        self.wait_for(lambda: self.app.state == "HOLD")
        self.view.mode.set(preview.CURRENT)
        goal = dict(zero_pose(), **{"shoulder_pan.pos": 25.})
        episode = {**self.app.document(), "title": "열기 후 편집",
                   "steps": [{"name": "검사 자세", "pose": goal}]}
        path = Path(self.temp.name) / "load_then_edit.json"
        path.write_text(json.dumps(episode), encoding="utf-8")
        original_file = path.read_bytes()
        with patch.object(self.app.session, "command", side_effect=AssertionError("Edit queued control")), \
             patch.object(self.app.backend, "connect", side_effect=AssertionError("Edit opened ports")):
            self.app.load_path(path)
            self.root.update()
            self.assertEqual(self.app.selected(), 0)
            self.assertEqual(self.view.mode.get(), preview.CURRENT)
            self.click_widget(self.app.buttons["edit_pose"])
            editor = self.app.pose_editor
            self.assertIsNotNone(editor)
            self.root.update()
            nav=self.app.navigation
            self.assertEqual(nav.host.winfo_manager(),'place')
            self.assertLessEqual(nav.host.winfo_rootx()+nav.host.winfo_width(),self.view.frame.winfo_rootx())
            self.assertTrue(self.view.canvas.winfo_ismapped())
            self.assertIn('검사 자세',nav.heading.get())
            self.assertEqual(nav.active_section,'episode')
            editor.values["shoulder_pan.pos"].set("45")
            self.assertEqual(self.view.mode.get(), preview.TARGET)
            self.wait_for(lambda: self.view.last_frame_angles ==
                          preview.pose_to_radians(dict(goal, **{"shoulder_pan.pos": 45.})))
            self.assertEqual(self.app.steps[0]["pose"], goal)
            editor.close()
            self.wait_for(lambda: self.view.last_frame_angles == preview.pose_to_radians(goal))
        self.assertEqual(path.read_bytes(), original_file)
        self.assertEqual(self.app.state, "HOLD")

    def test_camera_interaction_preserves_selected_step(self):
        self.app.steps[0]["pose"]["shoulder_pan.pos"] = 25.
        self.wait_for(lambda: self.view.photo is not None)
        x, y, width, height = self.app.table.bbox("0")
        self.click_widget(self.app.table, x+width//2, y+height//2)
        target = preview.pose_to_radians(self.app.steps[0]["pose"])
        self.wait_for(lambda: self.view.last_frame_angles == target)
        original_frame = self.root.tk.call(str(self.view.photo), "data", "-format", "ppm")
        with patch.object(self.app.session, "command", side_effect=AssertionError("Camera queued control")), \
             patch.object(self.app.backend, "send", side_effect=AssertionError("Camera sent motion")):
            for timestamp in (20000, 22000):
                before = self.view.camera
                self.view.canvas.event_generate("<ButtonPress-1>", x=100, y=100, time=timestamp)
                self.view.canvas.event_generate("<B1-Motion>", x=160, y=130, time=timestamp+40)
                self.view.canvas.event_generate("<ButtonRelease-1>", x=160, y=130, time=timestamp+80)
                self.root.update()
                self.assertNotEqual(self.view.camera, before)
                self.assertEqual(self.view.mode.get(), preview.TARGET)
                self.assertEqual(self.app.selected(), 0)
                self.assertEqual(self.app.table.selection(), ("0",))
            self.wait_for(lambda: self.root.tk.call(str(self.view.photo), "data", "-format", "ppm") != original_frame)
            self.view.canvas.event_generate("<Button-4>", x=160, y=130)
            self.root.update()
            self.assertNotEqual(self.view.camera[2], preview.DEFAULT_CAMERA[2])
            self.click_widget(self.view.camera_reset)
            self.assertEqual(self.view.camera, preview.DEFAULT_CAMERA)
            self.assertEqual(self.view.mode.get(), preview.TARGET)
            self.assertEqual(self.app.selected(), 0)
            self.assertEqual(self.view.last_frame_angles, target)
            self.assertEqual(self.app.joints.set("shoulder_pan", "step"), "2331")

    def test_repeated_reordering_preserves_step_selection_and_pose(self):
        self.app.steps = [{"name": name, "pose": dict(zero_pose(), **{"shoulder_pan.pos": angle})}
                          for name, angle in (("첫째", 5.), ("둘째", 15.), ("셋째", 25.))]
        selected_step = self.app.steps[2]
        self.app.refresh_steps(2)
        self.wait_for(lambda: self.view.photo is not None)
        x, y, width, height = self.app.table.bbox("2")
        self.click_widget(self.app.table, x+width//2, y+height//2)
        target = preview.pose_to_radians(selected_step["pose"])
        self.wait_for(lambda: self.view.last_frame_angles == target)
        with patch.object(self.app.session, "command", side_effect=AssertionError("Reorder queued control")), \
             patch.object(self.app.backend, "send", side_effect=AssertionError("Reorder sent motion")):
            for action, index, names in (
                    ("up", 1, ["첫째", "셋째", "둘째"]),
                    ("up", 0, ["셋째", "첫째", "둘째"]),
                    ("down", 1, ["첫째", "셋째", "둘째"]),
                    ("down", 2, ["첫째", "둘째", "셋째"])):
                self.assertTrue(self.app.can_reorder())
                self.app.reorder(-1 if action=="up" else 1)
                self.root.update()
                self.assertEqual(self.app.selected(), index)
                self.assertIs(self.app.steps[index], selected_step)
                self.assertEqual([step["name"] for step in self.app.steps], names)
                self.assertEqual(self.app.table.selection(), (str(index),))
                self.assertEqual(self.view.mode.get(), preview.TARGET)
                self.assertEqual(self.view.last_frame_angles, target)
                self.assertEqual(self.app.joints.set("shoulder_pan", "step"), "2331")
                draft = json.loads(self.app.program_path.read_text(encoding="utf-8"))
                self.assertEqual([step["name"] for step in draft["steps"]], names)

    def test_outside_click_preserves_editing_and_simulation(self):
        self.wait_for(lambda: self.view.photo is not None)
        x, y, width, height = self.app.table.bbox("0")
        self.click_widget(self.app.table, x+width//2, y+height//2)
        self.click_widget(self.app.buttons["edit_pose"])
        self.assertIsNotNone(self.app.pose_editor)
        self.assertEqual(self.view.mode.get(), preview.TARGET)
        self.assertEqual(self.app.selected(), 0)
        editor = self.app.pose_editor
        self.click_widget(editor.fields["shoulder_pan.pos"])
        editor.values["shoulder_pan.pos"].set("23.5")
        self.wait_for(lambda: self.view.last_frame_angles and abs(self.view.last_frame_angles[0]-math.radians(23.5)) < 1e-8)
        self.assertEqual(self.view.mode.get(), preview.TARGET)
        editor.close()
        bounds = self.app.table.bbox("0", "name")
        self.app.begin_step_name_edit(SimpleNamespace(x=bounds[0]+5, y=bounds[1]+5))
        self.root.update()
        self.click_widget(self.app.name_editor)
        self.assertEqual(self.view.mode.get(), preview.TARGET)
        self.assertEqual(self.app.selected(), 0)
        self.app.cancel_step_name_edit()
        self.wait_for(lambda: not self.app.buttons["simulate"].instate(["disabled"]))
        self.click_widget(self.app.buttons["simulate"])
        self.assertEqual(self.view.mode.get(), preview.SIMULATION)
        self.click_widget(self.app.shortcut_hint)
        self.assertEqual(self.view.mode.get(), preview.SIMULATION)

    def test_edit_cancel_camera_playback_stale_and_viewer_close(self):
        self.wait_for(lambda: self.view.last_frame_angles is not None)
        self.assertIsNone(self.root.grab_current())
        original = self.root.tk.call(str(self.view.photo), "data", "-format", "ppm")
        with patch.object(self.app.backend, "send", side_effect=AssertionError("Preview sent motion")):
            self.app.edit_step_pose()
            self.app.pose_editor.values["shoulder_pan.pos"].set("23.5")
            self.app.pose_editor.values["gripper.pos"].set("45")
            self.wait_for(lambda: self.view.last_frame_angles and abs(self.view.last_frame_angles[0]-math.radians(23.5)) < 1e-8)
            self.assertEqual(self.app.steps[0]["pose"]["shoulder_pan.pos"], 0)
            self.assertNotEqual(original, self.root.tk.call(str(self.view.photo), "data", "-format", "ppm"))
            self.app.pose_editor.values["gripper.pos"].set("nan")
            self.wait_for(lambda: self.view.canvas.itemcget(self.view.image_item, "state") == "hidden")
            self.app.pose_editor.close()
            self.wait_for(lambda: self.view.last_frame_angles == (0.,)*6)
            frame = self.root.tk.call(str(self.view.photo), "data", "-format", "ppm")
            self.view.rotate(30, 0)
            self.wait_for(lambda: self.root.tk.call(str(self.view.photo), "data", "-format", "ppm") != frame)
        self.app.buttons["connect"].invoke()
        self.wait_for(lambda: self.app.state == "HOLD")
        self.view.mode.set(preview.CURRENT)
        self.app.update_step_pose(self.app.steps[0], dict(zero_pose(), **{"shoulder_pan.pos": 7., "gripper.pos": 10.}))
        with patch.object(teaching, "ask_teaching_title", return_value="3D 재생 검증"):
            self.app.save_as()
        with patch("tkinter.messagebox.askokcancel", return_value=True):
            self.app.play()
        self.wait_for(lambda: self.app.state == "PLAY")
        self.wait_for(lambda: self.view.last_frame_angles is not None and self.view.last_frame_angles[0] > .01)
        self.assertLess(self.view.last_frame_angles[0], math.radians(7.))
        self.wait_for(lambda: self.app.state == "HOLD")
        self.wait_for(lambda: abs(self.view.last_frame_angles[0]-math.radians(7.)) < 1e-6)
        self.assertFalse(self.app.session.stop.is_set(), "Preview starved the GUI heartbeat")
        self.app.stop()
        self.wait_for(lambda: self.app.state == "DISCONNECTED")
        self.wait_for(lambda: self.view.canvas.itemcget(self.view.image_item, "state") == "hidden")
        self.app.buttons["connect"].invoke()
        self.wait_for(lambda: self.app.state == "HOLD")
        self.view.close()
        self.assertIsNone(self.app.preview)
        self.assertFalse(self.app.session.stop.is_set(), "Closing only the viewer stopped the robot")

    def test_renderer_crash_keeps_main_controls_and_stop_available(self):
        self.wait_for(lambda: self.view.last_frame_angles is not None)
        self.app.connect()
        self.wait_for(lambda: self.app.state == "HOLD")
        self.client.process.terminate()
        self.wait_for(lambda: self.view.render_error is not None)
        self.assertEqual(self.view.canvas.itemcget(self.view.image_item, "state"), "hidden")
        self.assertFalse(self.app.session.stop.is_set())
        self.assertEqual(self.app.state, "HOLD")
        self.app.buttons["stop"].invoke()
        self.assertTrue(self.app.session.stop.is_set())
        self.assertEqual(self.app.state, "STOPPING")

    def test_one_window_keeps_preview_on_right_and_one_global_stop(self):
        def widgets(parent):
            for child in parent.winfo_children():
                yield child
                yield from widgets(child)
        self.wait_for(lambda: self.view.last_frame_angles is not None)
        self.assertIs(self.view.frame.winfo_toplevel(), self.root)
        self.assertFalse(any(isinstance(w, tk.Toplevel) for w in self.root.winfo_children()))
        self.assertEqual([w for w in widgets(self.view.frame) if w.winfo_class() == "TCombobox"],
                         [self.view.sim_speed])
        stops = [w for w in widgets(self.root) if "text" in w.keys() and w.cget("text") == "정지 / 토크 해제"]
        self.assertEqual(stops, [self.app.buttons["stop"]])
        for geometry in ("1680x900", "1366x768", "1080x720"):
            self.root.geometry(geometry)
            self.root.update()
            self.assertGreater(self.view.frame.winfo_rootx(), self.app.table.winfo_rootx())
            self.assertTrue(self.view.canvas.winfo_ismapped())
            self.assertTrue(self.app.buttons["stop"].winfo_ismapped())
            self.assertTrue(self.app.joints.winfo_ismapped())
            self.assertTrue(self.app.table.winfo_ismapped())
            row = self.app.table.bbox("0")
            self.assertTrue(row, geometry)
            self.assertLessEqual(row[1]+row[3], self.app.table.winfo_height(), geometry)
            self.assertGreater(self.view.canvas.winfo_height(), 240)
            self.assertLess(self.view.frame.winfo_rooty(), self.app.joints_panel.winfo_rooty())

    def test_mouse_drag_rotates_in_reverse_direction(self):
        from types import SimpleNamespace
        self.view.camera = preview.DEFAULT_CAMERA
        self.view.begin_drag(SimpleNamespace(x=100, y=100))
        self.view.drag(SimpleNamespace(x=120, y=110))
        azimuth, elevation, distance = self.view.camera
        self.assertAlmostEqual(azimuth, preview.DEFAULT_CAMERA[0]-10)
        self.assertAlmostEqual(elevation, preview.DEFAULT_CAMERA[1]-5)
        self.assertEqual(distance, preview.DEFAULT_CAMERA[2])

    def test_preview_fills_viewport_after_resize_and_has_no_collapse_controls(self):
        def buttons(parent):
            for widget in parent.winfo_children():
                if "text" in widget.keys():
                    yield widget.cget("text")
                yield from buttons(widget)
        self.assertNotIn("3D 보기", list(buttons(self.root)))
        self.assertNotIn("접기", list(buttons(self.root)))
        for geometry in ("1720x1000", "1080x1000", "1080x800", "1720x900"):
            self.root.geometry(geometry)
            self.root.update()
            self.wait_for(lambda: self.view.photo is not None
                          and self.view.photo.width() == self.view.canvas.winfo_width()-2
                          and self.view.photo.height() == self.view.canvas.winfo_height()-2
                          and self.view.canvas.itemcget(self.view.image_item, "state") == "normal")
            self.assertEqual(self.view.canvas.coords(self.view.image_item), [1., 1.])
            self.assertTrue(self.app.buttons["stop"].winfo_ismapped())
        # Drag out and back before the debounce: the image must reappear even
        # though the final dimensions match the last render request.
        self.root.geometry("1710x910")
        self.root.update()
        self.root.geometry("1720x900")
        self.root.update()
        self.wait_for(lambda: self.view.canvas.itemcget(self.view.image_item, "state") == "normal")

    def test_current_episode_simulates_all_steps_and_uses_new_edits(self):
        steps = [{"name": "시작", "pose": zero_pose()},
                 {"name": "도착", "pose": dict(zero_pose(), **{"shoulder_pan.pos": 12, "gripper.pos": 25})}]
        self.app.steps = steps
        self.app.teaching_title = "현재 에피소드"
        self.app.refresh_steps(0)
        self.wait_for(lambda: not self.app.buttons["simulate"].instate(["disabled"]))
        with patch.object(self.app, "send", side_effect=AssertionError("Simulation called control")), \
             patch.object(self.app.backend, "send", side_effect=AssertionError("Simulation wrote a servo")), \
             patch.object(self.app.backend, "connect", side_effect=AssertionError("Simulation opened ports")), \
             patch.object(teaching, "atomic_json", side_effect=AssertionError("Simulation saved a file")):
            self.app.buttons["simulate"].invoke()
            self.assertEqual(self.view.player.title, "현재 에피소드")
            self.assertEqual(len(self.view.player.steps), 2)
            self.wait_for(lambda: self.view.last_frame_angles and self.view.last_frame_angles[0] > .02)
            self.view.sim_pause.invoke()
            self.assertFalse(self.view.player.running)
            self.view.seek_simulation(str(self.view.player.duration))
            self.wait_for(lambda: self.view.last_frame_angles and abs(self.view.last_frame_angles[0]-math.radians(12)) < 1e-8)
            self.view.sim_reset.invoke()
            self.view.sim_pause.invoke()
            self.wait_for(lambda: self.view.player.position == self.view.player.duration)
            self.assertFalse(self.view.player.running)
        self.app.update_step_pose(self.app.steps[1], dict(zero_pose(), **{"shoulder_pan.pos": 24, "gripper.pos": 30}))
        self.assertFalse(self.view.player.steps, "Old simulation must be discarded on edits")
        self.app.simulate()
        self.view.seek_simulation(str(self.view.player.duration))
        self.wait_for(lambda: self.view.last_frame_angles and abs(self.view.last_frame_angles[0]-math.radians(24)) < 1e-8)
        self.assertEqual(self.app.state, "DISCONNECTED")
        self.assertIsNone(self.app.saved_path)

    def test_foreign_episode_uses_common_loader_and_cannot_drive_or_overwrite(self):
        import json
        identity = {"demo": False, "side": "left", "serials": {"leader": "another", "follower": "arm"},
                    "calibration_sha256": {"follower": "different"}}
        steps = [{"name": "시작", "pose": zero_pose()},
                 {"name": "도착", "pose": dict(zero_pose(), **{"shoulder_pan.pos": 12, "gripper.pos": 25})}]
        document = {"schema": 1, "identity": identity, "units": teaching.UNITS, "title": "실물 저장본", "steps": steps}
        path = Path(self.temp.name) / "hardware_episode.json"
        path.write_text(json.dumps(document))
        original = path.read_bytes()
        with patch.object(self.app.backend, "send", side_effect=AssertionError("Simulation wrote a servo")), \
             patch.object(self.app.backend, "connect", side_effect=AssertionError("Simulation opened ports")), \
             patch.object(self.app.session, "command", side_effect=AssertionError("Foreign episode queued motion")), \
             patch.object(teaching, "atomic_json", side_effect=AssertionError("Simulation rewrote a file")):
            self.app.load_path(path)
            self.assertTrue(self.app.preview_only)
            self.assertEqual(self.app.steps[1]["pose"], steps[1]["pose"])
            self.assertEqual(self.app.teaching_title, "실물 저장본")
            self.assertEqual(self.app.document()["identity"], identity)
            self.wait_for(lambda: not self.app.buttons["simulate"].instate(["disabled"]))
            self.app.buttons["simulate"].invoke()
            self.wait_for(lambda: self.view.player.position > .5)
            for key in ("connect", "move", "play", "save", "edit_pose", "capture"):
                self.assertTrue(self.app.buttons[key].instate(["disabled"]), key)
            self.app.connect()
            self.app.move()
            self.app.play()
            self.app.send("move", zero_pose())
            self.app.save_as()
            self.assertEqual(self.app.state, "DISCONNECTED")
        self.assertEqual(path.read_bytes(), original)
        with self.assertRaises(ValueError):
            teaching.validate_program(document, self.app.backend.identity)
        with patch("tkinter.messagebox.askokcancel", return_value=True):
            self.app.new_program()
        self.assertFalse(self.app.preview_only)
        self.assertEqual(self.app.document()["identity"], self.app.backend.identity)
        self.assertFalse(self.view.player.steps)

    def test_simulation_returns_to_live_pose_three_seconds_after_completion(self):
        self.app.connect()
        self.wait_for(lambda: self.app.state == "HOLD")
        self.app.steps[0]["pose"]["shoulder_pan.pos"] = 25.
        self.wait_for(lambda: self.view.photo is not None)
        with patch.object(self.app, "send", side_effect=AssertionError("Preview queued motion")), \
             patch.object(self.app.session, "command", side_effect=AssertionError("Preview queued control")), \
             patch.object(self.app.backend, "connect", side_effect=AssertionError("Preview opened ports")):
            self.app.simulate()
            self.assertEqual(self.app.buttons["simulate"].cget("text"), "실행 중지")
            self.wait_for(lambda: self.view.auto_return_at is not None)
            completed = time.monotonic()
            self.assertEqual(self.view.mode.get(), preview.SIMULATION)
            self.assertIn("3초 후 현재 자세", self.view.status.get())
            self.app.refresh_buttons()
            self.assertEqual(self.app.buttons["simulate"].cget("text"), "시뮬레이션 실행")
            self.wait_for(lambda: time.monotonic() - completed >= 2.5, seconds=3)
            self.assertEqual(self.view.mode.get(), preview.SIMULATION)
            self.assertAlmostEqual(self.view.last_frame_angles[0], math.radians(25.))
            self.wait_for(lambda: self.view.mode.get() == preview.CURRENT, seconds=2)
            elapsed = time.monotonic() - completed
            self.assertGreaterEqual(elapsed, 2.9)
            self.assertLess(elapsed, 3.5)
            self.wait_for(lambda: self.view.last_frame_angles == preview.pose_to_radians(self.app.latest["follower"]))
        self.assertEqual(self.app.state, "HOLD")
        self.assertFalse(self.app.session.stop.is_set())

    def test_simulation_button_stops_running_and_paused_playback_without_robot_commands(self):
        self.app.clear_step_selection()
        self.app.connect()
        self.wait_for(lambda: self.app.state == "HOLD" and self.view.mode.get() == preview.CURRENT)
        self.app.steps = [{"name": "시작", "pose": zero_pose()},
                          {"name": "끝", "pose": dict(zero_pose(), **{"shoulder_pan.pos": 150.})}]
        self.app.refresh_steps(0)
        with patch.object(self.app.session, "command", side_effect=AssertionError("Simulation queued control")), \
             patch.object(self.app.backend, "connect", side_effect=AssertionError("Simulation opened ports")), \
             patch.object(teaching, "atomic_json", side_effect=AssertionError("Simulation saved a file")):
            for paused in (False, True):
                with self.subTest(paused=paused):
                    self.wait_for(lambda: not self.app.buttons["simulate"].instate(["disabled"]))
                    self.click_widget(self.app.buttons["simulate"])
                    self.assertEqual(self.app.buttons["simulate"].cget("text"), "실행 중지")
                    self.assertTrue(self.view.simulation_active)
                    self.assertIsNone(self.app.selected())
                    self.wait_for(lambda: self.view.player.position > .1)
                    if paused:
                        self.click_widget(self.view.sim_pause)
                        self.assertFalse(self.view.player.running)
                        self.assertEqual(self.app.buttons["simulate"].cget("text"), "실행 중지")
                    self.click_widget(self.app.buttons["simulate"])
                    self.assertFalse(self.view.simulation_active)
                    self.assertFalse(self.view.player.running)
                    self.assertEqual(self.view.player.position, 0.)
                    self.assertEqual(self.view.mode.get(), preview.CURRENT)
                    self.assertEqual(self.app.buttons["simulate"].cget("text"), "시뮬레이션 실행")
                    self.assertFalse(self.view.transport.winfo_ismapped())
                    self.assertEqual(len(self.app.steps), 2)
                    self.assertEqual(self.app.state, "HOLD")
                    self.assertFalse(self.app.session.stop.is_set())

    def test_manual_actions_cancel_pending_return_and_pause_does_not_arm_it(self):
        def tick_at(now):
            self.view.frame.after_cancel(self.view.job)
            with patch.object(preview.time, "monotonic", return_value=now):
                self.view.tick()

        actions = {
            "display choice": lambda: self.view.mode.set(preview.TARGET),
            "reselect simulation": lambda: self.view.mode.set(preview.SIMULATION),
            "seek": lambda: self.view.seek_simulation(str(self.view.player.duration)),
            "reset": self.view.reset_simulation,
            "stop": self.view.pause_simulation,
            "restart": self.view.play_simulation,
            "resume": self.view.toggle_pause,
            "episode changed": self.view.invalidate_episode,
        }
        for name, action in actions.items():
            with self.subTest(action=name):
                with patch.object(preview.time, "monotonic", return_value=0.):
                    self.view.play_simulation()
                tick_at(1.)
                self.assertEqual(self.view.auto_return_at, 4.)
                with patch.object(preview.time, "monotonic", return_value=2.):
                    action()
                self.assertIsNone(self.view.auto_return_at)
                tick_at(4.1)
                self.assertNotEqual(self.view.mode.get(), preview.CURRENT)
        with patch.object(preview.time, "monotonic", return_value=10.):
            self.view.play_simulation()
        with patch.object(preview.time, "monotonic", return_value=10.1):
            self.view.pause_simulation()
        tick_at(20.)
        self.assertIsNone(self.view.auto_return_at)
        self.assertEqual(self.view.mode.get(), preview.SIMULATION)



if __name__ == "__main__":
    unittest.main()
