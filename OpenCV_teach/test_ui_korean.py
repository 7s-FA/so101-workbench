"""Localized UI behavior, native dialogs and teaching/diagnostics integration."""

import tempfile
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox, ttk
import time
import unittest
from unittest.mock import patch

import teach_left_arm as app
from motor_dashboard import Dashboard
from teaching_ui import ask_teaching_title, message_ko


class KoreanUITests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk display required: {exc}")
        self.errors = []
        self.root.report_callback_exception = lambda kind, error, trace: self.errors.append(error)
        self.directory = tempfile.TemporaryDirectory()
        if self._testMethodName == "test_scaled_compact_layout_keeps_all_joints_visible":
            self.root.tk.call("tk", "scaling", 1.5)
        self.ui = app.App(self.root, app.Demo(), self.directory.name)
        self.root.update()

    def tearDown(self):
        if not hasattr(self, "ui"):
            return
        self.ui.session.stop.set()
        self.ui.session.wakeup.set()
        if self.ui.session.thread:
            self.ui.session.thread.join(3)
        owners = [self.root, *self.widgets(self.root)]
        for job in self.root.tk.splitlist(self.root.tk.call("after", "info")):
            script = self.root.tk.call("after", "info", job)[0]
            owner = next((widget for widget in owners if script in (widget._tclCommands or [])), self.root)
            owner.after_cancel(job)
        self.root.destroy()
        self.directory.cleanup()
        self.assertEqual(self.errors, [])

    def widgets(self, parent):
        for child in parent.winfo_children():
            yield child
            yield from self.widgets(child)

    def seed(self):
        self.ui.steps = [{"name": "준비 자세", "pose": dict.fromkeys(app.KEYS, 1.234567), "captured_at": app.stamp()},
                         {"name": "물체 잡기", "pose": dict.fromkeys(app.KEYS, 2.), "captured_at": app.stamp()}]
        self.ui.refresh_steps(0)
        self.root.update()

    def test_empty_state_selection_actions_and_reorder(self):
        self.assertIn("연결 안 됨", self.ui.status.get())
        self.assertEqual(self.ui.empty_steps.winfo_manager(), "place")
        for key in ("edit_pose", "delete", "safe"):
            self.assertTrue(self.ui.buttons[key].instate(["disabled"]))
        self.seed()
        self.assertFalse(self.ui.empty_steps.winfo_manager())
        self.assertNotIn("up",self.ui.buttons)
        self.assertNotIn("down",self.ui.buttons)
        self.ui.reorder(1)
        self.root.update()
        self.assertEqual(self.ui.steps[1]["name"], "준비 자세")
        self.assertEqual(self.ui.selected(), 1)
        self.assertFalse(self.ui.buttons["safe"].instate(["disabled"]))

    def test_drag_reorders_and_run_safe_delete_layout_is_visible(self):
        self.seed()
        self.ui.steps.append({'name':'세 번째','pose':dict.fromkeys(app.KEYS,3.)})
        self.ui.refresh_steps(0);self.root.update()
        def drag(source,target):
            x,y,w,h=self.ui.table.bbox(str(source));tx,ty,tw,th=self.ui.table.bbox(str(target))
            self.ui.table.event_generate('<ButtonPress-1>',x=x+50,y=y+h//2)
            self.ui.table.event_generate('<B1-Motion>',x=tx+50,y=ty+th//2)
            self.ui.table.event_generate('<ButtonRelease-1>',x=tx+50,y=ty+th//2)
            self.root.update()
        with patch.object(self.ui,'send',side_effect=AssertionError('Reorder moved robot')):
            drag(0,2)
        self.assertEqual([s['name'] for s in self.ui.steps],['물체 잡기','세 번째','준비 자세'])
        before=self.ui.steps.copy();self.ui.state='MOVE';drag(0,1);self.assertEqual(self.ui.steps,before)
        self.assertEqual(self.ui.buttons['safe'].winfo_y(),self.ui.buttons['delete'].winfo_y())
        self.assertGreater(self.ui.buttons['move'].winfo_width(),self.ui.buttons['safe'].winfo_width())
        self.assertLess(self.ui.buttons['move'].winfo_rooty(),self.ui.buttons['reference'].winfo_rooty())

    def test_live_updates_preserve_selected_target_and_show_korean_state(self):
        self.seed()
        self.ui.session.events.put(("status", {"state": "TEACH", "leader": dict.fromkeys(app.KEYS, 12.),
                                              "follower": dict.fromkeys(app.KEYS, 10.),
                                              "sample_time": time.monotonic(), "limited_joints": [app.KEYS[0]]}))
        self.ui.poll()
        self.assertEqual(self.ui.joints.set("shoulder_pan", "step"), "2061")
        self.assertEqual(self.ui.joints.set("shoulder_pan", "leader"), "2184")
        self.assertIn("티칭 중", self.ui.status.get())
        self.assertIn("범위 제한", self.ui.status.get())
        self.assertIn("스페이스", self.ui.hint.get())
        self.assertTrue(self.ui.buttons["edit_pose"].instate(["disabled"]))

    def test_korean_save_dialog_keyboard_and_cancel(self):
        def enter_title():
            entry = next(widget for widget in self.widgets(self.root._title_panel)
                         if isinstance(widget, ttk.Entry))
            entry.delete(0, "end")
            entry.insert(0, "조립 동작 01")
            dialog = self.root._title_panel
            save = next(widget for widget in self.widgets(dialog)
                        if isinstance(widget, ttk.Button) and widget.cget("text") == "저장")
            save.invoke()
        self.root.after(50, enter_title)
        self.assertEqual(ask_teaching_title(self.root, "이전 이름"), "조립 동작 01")
        def cancel_title():
            cancel = next(widget for widget in self.widgets(self.root._title_panel)
                          if isinstance(widget, ttk.Button) and widget.cget("text") == "취소")
            cancel.invoke()
        self.root.after(50, cancel_title)
        self.assertIsNone(ask_teaching_title(self.root))

    def test_native_confirmation_buttons_are_korean(self):
        observed = []
        def cancel_message():
            dialog = ".__tk__messagebox"
            children = self.root.tk.splitlist(self.root.tk.call("winfo", "children", dialog))
            for child in children:
                if self.root.tk.call("winfo", "class", child) == "TButton":
                    observed.append(self.root.tk.call(child, "cget", "-text"))
            self.root.tk.call(dialog + ".cancel", "invoke")
        self.root.after(50, cancel_message)
        self.assertFalse(messagebox.askokcancel("이동 확인", "선택한 자세로 이동할까요?", parent=self.root))
        self.assertIn("확인", observed)
        self.assertIn("취소", observed)

    def test_diagnostic_window_shares_style_without_changing_main_table(self):
        rowheight = ttk.Style(self.root).lookup("Treeview", "rowheight")
        window = tk.Toplevel(self.root)
        dashboard = Dashboard(window, demo=True)
        self.root.update()
        self.assertEqual(ttk.Style(self.root).lookup("Treeview", "rowheight"), rowheight)
        self.assertEqual(len(dashboard.table.get_children()), 6)
        self.assertEqual(dashboard.table.set("6", "joint"), "그리퍼")
        for metric in ("온도", "전류", "부하", "위치 오차"):
            dashboard.metric.set(metric)
            dashboard.draw()
            self.assertTrue(dashboard.canvas.find_all())
        self.assertIn("정지 조건", dashboard.status.get())

    def test_stop_failure_and_controller_errors_keep_meaning(self):
        self.ui.session.events.put(("closed", ["torque OFF not confirmed"]))
        self.ui.poll()
        self.assertIn("토크 해제 미확인", self.ui.status.get())
        self.assertIn("모터 전원을 차단", self.ui.hint.get())
        self.assertTrue(self.ui.buttons["connect"].instate(["disabled"]))
        self.assertIn("화면 응답", message_ko("RuntimeError: GUI heartbeat lost; stopping"))
        self.assertIn("정지", message_ko("shoulder_pan.pos: no measurable forward progress for 3s; no automatic retry"))
        self.assertIn("보정 범위", message_ko("Outside follower calibration range: elbow_flex.pos=21.00, allowed [-20.00, 20.00]"))

    def test_scaled_compact_layout_keeps_all_joints_visible(self):
        self.seed()
        self.ui.steps[0]["name"] = "긴 자세 이름도 조작부를 가리지 않습니다. " * 4
        self.ui.select()
        self.root.geometry("1080x720")
        self.root.update()
        row = self.ui.joints.bbox("gripper")
        self.assertTrue(row)
        self.assertLessEqual(row[1] + row[3], self.ui.joints.winfo_height())
        # Enlarged Korean text must not hide units or any operation button.
        unit_font = tkfont.Font(self.root, font=ttk.Style(self.root).lookup("Treeview", "font"))
        self.assertGreaterEqual(self.ui.joints.column("unit", "width"), unit_font.measure("0–100") + 10)
        for key, button in self.ui.buttons.items():
            if key in ('hold_current','release'):continue  # follower-only controls hidden in pair mode
            with self.subTest(button=key):
                self.assertTrue(button.winfo_ismapped())
                right = button.winfo_rootx() - self.root.winfo_rootx() + button.winfo_width()
                bottom = button.winfo_rooty() - self.root.winfo_rooty() + button.winfo_height()
                self.assertLessEqual(right, self.root.winfo_width())
                self.assertLessEqual(bottom, self.root.winfo_height())
        self.assertFalse(self.ui.log.winfo_ismapped())
        self.ui.log_expanded.set(True)
        self.ui.toggle_log()
        self.root.update()
        expanded_geometry = self.root.geometry()
        row = self.ui.joints.bbox("gripper")
        self.assertTrue(self.ui.log.winfo_ismapped())
        self.assertTrue(self.ui.buttons["stop"].winfo_ismapped())
        self.ui.log_expanded.set(False)
        self.ui.toggle_log()
        self.assertEqual(self.root.geometry(), expanded_geometry)


if __name__ == "__main__":
    unittest.main()
