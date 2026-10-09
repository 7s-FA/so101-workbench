"""No hardware access: python -m unittest -v test_optimization.

Ubuntu uses the real helper imports; Windows stubs only Linux OS helpers.
"""
import json
import queue
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import teach_left_arm as app
import teaching_settings as settings

if sys.platform == "win32":
    shim = types.ModuleType("diagnose_shoulder_lift")
    shim.ensure_ports_available = shim.protect_port = lambda *a: None
    shim.now_iso = app.stamp
    with patch.dict(sys.modules, {"diagnose_shoulder_lift": shim}):
        import monitor_shoulder_temperature as temperature
        import motor_diagnostics as motors
    sys.modules["monitor_shoulder_temperature"] = temperature
    sys.modules["motor_diagnostics"] = motors
else:
    import monitor_shoulder_temperature as temperature
    import motor_diagnostics as motors


class SettingsTests(unittest.TestCase):
    def changed(self, old, new):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "settings.toml"
            source = (settings.ROOT / "settings.toml").read_text(encoding="utf-8")
            self.assertIn(old, source)
            path.write_text(source.replace(old, new), encoding="utf-8")
            return settings.load_settings(path)

    def test_valid_edit(self):
        self.assertEqual(self.changed("CONTROL_HZ = 100", "CONTROL_HZ = 60")["CONTROL_HZ"], 60)

    def test_invalid_settings(self):
        for old, new in (("CONTROL_HZ = 100", "CONTROL_HZ = 0"),
                         ("CONTROL_HZ = 100", "CONTROL_HZ = true"),
                         ("CONTROL_HZ = 100", "CONTROL_HZ = 100.5"),
                         ("COMMAND_RATE = 30.0", "COMMAND_RATE = nan"),
                         ("GUI_POLL_MS = 16", "GUI_POL_MS = 16"),
                         ("TEMPERATURE_LIMIT = 65", "TEMPERATURE_LIMIT = 90"),
                         ("CAPTURE_STABLE_SECONDS = 0.5", "CAPTURE_STABLE_SECONDS = 2.0")):
            with self.subTest(new=new), self.assertRaises(ValueError):
                self.changed(old, new)


class ControlTests(unittest.TestCase):
    def setUp(self):
        # Earlier GUI tests leave Tk cycles. Finalize them on the main thread
        # before starting a timing-sensitive Session worker.
        import gc
        gc.collect()

    def test_event_coalescing_preserves_order(self):
        events = app.EventMailbox()
        for i in range(10000):
            events.put(("status", i))
        events.put(("captured", "step1"))
        events.put(("status", 10001))
        events.put(("closed", []))
        self.assertEqual([events.get_nowait() for _ in range(4)],
                         [("status", 9999), ("captured", "step1"), ("status", 10001), ("closed", [])])
        with self.assertRaises(queue.Empty):
            events.get_nowait()

    def test_slew_and_stall_guards(self):
        zero = dict.fromkeys(app.KEYS, 0.)
        target = dict.fromkeys(app.KEYS, 90.)
        step = app.approach(zero, target, 10, gripper_rate=settings.GRIPPER_COMMAND_RATE)
        self.assertLessEqual(step["shoulder_pan.pos"], settings.COMMAND_RATE * settings.MAX_COMMAND_DT)
        self.assertLessEqual(step["gripper.pos"], settings.GRIPPER_COMMAND_RATE * settings.MAX_COMMAND_DT)
        guard = app.TrackingProgress()
        self.assertFalse(guard.update(0, 0, 15, 10)["blocked"])
        self.assertTrue(guard.update(settings.STALL_SECONDS, 0, 15, 10)["blocked"])

    def test_quick_capture_keeps_teaching_and_saves_actual_pose(self):
        with tempfile.TemporaryDirectory() as d:
            backend = app.Demo()
            session = app.Session(backend, d, auto_sync=True)
            session.start()
            events = []
            def wait_for(predicate, seconds=3):
                deadline = time.monotonic() + seconds
                while time.monotonic() < deadline:
                    session.heartbeat = time.monotonic()
                    try:
                        while True:
                            events.append(session.events.get_nowait())
                    except queue.Empty:
                        pass
                    if predicate():
                        return
                    time.sleep(.005)
                self.fail(f"Timed out in {session.state}: {events[-5:]}")
            try:
                wait_for(lambda: session.state == "HOLD")
                session.command("teach")
                wait_for(lambda: session.state == "TEACH")
                backend.lead["shoulder_pan.pos"] = 15.
                wait_for(lambda: backend.current["shoulder_pan.pos"] > .1)
                for i in range(3):
                    session.command("quick_capture", f"step{i+1}")
                    wait_for(lambda: len([e for e in events if e[0] == "captured"]) == i+1)
                    self.assertEqual(session.state, "TEACH")
                saved = [e[1] for e in events if e[0] == "captured"]
                self.assertLess(saved[0]["pose"]["shoulder_pan.pos"], 15.)
                document = {"schema": 1, "identity": backend.identity, "units": app.UNITS,
                            "title": "space test", "steps": saved}
                path = app.save_named_teaching(d, document)
                self.assertEqual(len(app.validate_program(json.loads(path.read_text()), backend.identity)), 3)
                session.command("pause")
                wait_for(lambda: session.state == "HOLD")
                session.command("play", {"steps": saved, "follower": backend.current.copy()})
                wait_for(lambda: session.state == "PLAY")
                wait_for(lambda: session.state == "HOLD", 5)
            finally:
                session.stop.set()
                session.wakeup.set()
                session.thread.join(5)
            self.assertFalse(session.thread.is_alive())
            report = json.loads(next(Path(d).glob("session_*.json")).read_text())
            self.assertEqual(report["outcome"], "stopped")
            self.assertGreater(report["loop_metrics"]["cycles"], 10)

    def test_heartbeat_stops_worker(self):
        with tempfile.TemporaryDirectory() as d, patch.object(app, "GUI_HEARTBEAT_TIMEOUT", .04):
            session = app.Session(app.Demo(), d)
            session.start()
            session.thread.join(2)
            self.assertFalse(session.thread.is_alive())
            report = json.loads(next(Path(d).glob("session_*.json")).read_text())
            self.assertIn("heartbeat lost", report["outcome"])


class HealthTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.hot = False
        self.alarm = False
        self.bad_packet = False
        self.bad_checksum = False
        self.torque = 1
        def txrx(port, request):
            self.requests.append(request)
            motor_id, address, length = request[2], request[5], request[6]
            block = bytearray(31)
            block[0], block[23], block[25] = self.torque, 65 if self.hot else 30, int(self.alarm)
            payload = list(block if address == 40 else block[23:24])
            response = [255,255,motor_id,length+2,4 if self.bad_packet else 0] + payload
            response.append((~sum(response[2:]) & 255) ^ int(self.bad_checksum))
            return response, 0, response[4]
        handler = types.SimpleNamespace(txRxPacket=txrx)
        def read(port, motor_id, address, length):
            response, comm, error = handler.txRxPacket(port, [255,255,motor_id,4,2,address,length,0])
            return response[5:-1], comm, error
        handler.readTxRx = read
        bus = types.SimpleNamespace(packet_handler=handler, port_handler=object(),
                                    motors={j: types.SimpleNamespace(id=i) for i,j in enumerate(app.JOINTS,1)})
        self.hardware = app.Hardware.__new__(app.Hardware)
        self.hardware.follower = types.SimpleNamespace(bus=bus)

    def test_health_12_packets_and_no_validator_stacking(self):
        sample = self.hardware.health(True)
        self.assertTrue(sample["complete"])
        self.assertEqual(len(sample["temperatures_raw"]), 6)
        self.assertEqual(len(self.requests), 12)
        original = self.hardware.follower.bus.packet_handler.txRxPacket
        self.hardware.health(True)
        self.assertIs(original, self.hardware.follower.bus.packet_handler.txRxPacket)

    def test_faults_stop_immediately(self):
        for attribute in ("alarm", "bad_packet", "bad_checksum"):
            with self.subTest(attribute=attribute):
                setattr(self, attribute, True)
                with self.assertRaises(RuntimeError):
                    self.hardware.health(True)
                self.assertFalse(self.hardware.temperature_sample["complete"])
                setattr(self, attribute, False)

    def test_hot_preactivation_stops_immediately(self):
        self.hot = True
        self.torque = 0
        with self.assertRaises(RuntimeError):
            self.hardware.health(False)

    def test_torque_mismatch(self):
        self.torque = 0
        with self.assertRaises(RuntimeError):
            self.hardware.health(True)
        self.assertTrue(self.hardware.health(False)["complete"])

    def test_dashboard_reuses_health_sample(self):
        rows = [{"id": i, "name": j, "raw": {"Status":0,"Torque_Enable":1,"Present_Temperature":30},
                 "temperature_single":30, "issues":[]} for i,j in enumerate(app.JOINTS,1)]
        self.hardware.motor_dashboard = types.SimpleNamespace(capture=lambda *a: {"rows":rows}, fault=lambda *a: None)
        self.assertTrue(self.hardware.health()["complete"])
        self.assertEqual(self.requests, [])


class ShortcutTests(unittest.TestCase):
    def setUp(self):
        self.commands = []
        self.jobs = {}
        self.root = types.SimpleNamespace(grab_current=lambda: None,
                                         after_cancel=lambda j: self.jobs.pop(j, None))
        def idle(callback):
            self.jobs[1] = callback
            return 1
        self.root.after_idle = idle
        self.widget = types.SimpleNamespace(winfo_toplevel=lambda: self.root, winfo_class=lambda: "TButton")
        self.ui = app.App.__new__(app.App)
        self.ui.root, self.ui.state = self.root, "TEACH"
        self.ui.closing = self.ui.finish_pending = self.ui.space_down = False
        self.ui.space_release_job = self.ui.name_editor = None
        self.ui.send = self.commands.append
        self.event = types.SimpleNamespace(widget=self.widget)

    def test_one_capture_per_press_with_x11_repeat(self):
        self.assertEqual(self.ui.space_press(self.event), "break")
        self.ui.space_press(self.event)
        self.ui.space_release(self.event)
        self.ui.space_press(self.event)
        self.assertEqual(self.commands, ["quick_capture"])
        self.ui.space_release(self.event)
        self.jobs[1]()
        self.ui.space_press(self.event)
        self.assertEqual(len(self.commands), 2)

    def test_input_and_modal_exclusions(self):
        for widget_class in ("TEntry", "Entry", "Text", "TCombobox"):
            self.widget.winfo_class = lambda: widget_class
            self.assertIsNone(self.ui.space_press(self.event))
        self.widget.winfo_class = lambda: "TButton"
        self.root.grab_current = lambda: object()
        self.assertIsNone(self.ui.space_press(self.event))
        self.assertEqual(self.commands, [])

    def test_no_capture_outside_teaching(self):
        for state in ("HOLD", "PLAY", "SYNCING", "STOPPING", "DISCONNECTED"):
            self.ui.state = state
            self.ui.space_press(self.event)
        self.assertEqual(self.commands, [])


if __name__ == "__main__":
    unittest.main()
