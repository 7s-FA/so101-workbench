"""Shared read-only telemetry for the motor dashboard and teaching worker."""

from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path
import queue
import time

import monitor_shoulder_temperature as temperature
from temperature_guard import TemperatureFault, check_readings, pending_label
from diagnose_shoulder_lift import ensure_ports_available, protect_port, now_iso

from teaching_settings import (TEMPERATURE_LIMIT, TEMPERATURE_DIFFERENCE_WARNING,
                               DASHBOARD_QUEUE_SIZE, READONLY_INTERVAL)

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
PORT = "/dev/serial/by-id/usb-YOUR_ROBOT-if00"
SETTINGS = ("Model_Number", "ID", "Operating_Mode", "Max_Temperature_Limit", "Min_Voltage_Limit",
            "Max_Voltage_Limit", "Max_Torque_Limit", "Protection_Current", "Protection_Time",
            "Overload_Torque", "Over_Current_Protection_Time", "Protective_Torque", "Unloading_Condition",
            "P_Coefficient", "I_Coefficient", "D_Coefficient", "Min_Position_Limit", "Max_Position_Limit")
LIVE = ("Goal_Position", "Present_Position", "Present_Velocity", "Present_Load", "Present_Current",
        "Torque_Limit", "Present_Voltage", "Present_Temperature", "Status", "Torque_Enable")


def readonly_packets(bus):
    original = bus.port_handler.writePort

    def guarded(packet):
        p = list(packet)
        if (len(p) != 8 or p[:2] != [255, 255] or p[2] not in range(1, 7)
                or p[3:5] != [4, 2] or tuple(p[5:7]) not in {(0, 40), (40, 31), (63, 1)}
                or p[-1] != (~sum(p[2:-1]) & 255)):
            raise RuntimeError("Dashboard permits whitelisted READ packets only; motor writes blocked")
        return original(packet)

    bus.port_handler.writePort = guarded


def issues(row, expected_active=None):
    result = []
    raw, settings = row.get("raw", {}), row.get("settings", {})
    if row.get("read_error") or row.get("settings_error"):
        result.append(("error", "Read/packet error: " + (row.get("read_error") or row["settings_error"])))
    if settings.get("Model_Number") not in (None, 777):
        result.append(("error", "Unexpected model"))
    temps = [raw.get("Present_Temperature"), row.get("temperature_single")]
    if row.get('temperature_error'):
        result.append(('error', row['temperature_error']))
    elif row.get('temperature_checked'):
        if row.get('temperature_pending'):
            result.append(('warning', pending_label(row['temperature_pending'])))
    elif any(type(v) is not int or not 0 <= v <= 150 for v in temps):
        result.append(('error', 'Invalid temperature data'))
    elif any(v >= TEMPERATURE_LIMIT for v in temps):
        result.append(("error", f"Temperature >= {TEMPERATURE_LIMIT} raw"))
    if all(isinstance(v, int) for v in temps) and abs(temps[0] - temps[1]) > TEMPERATURE_DIFFERENCE_WARNING:
        result.append(("warning", "Temperature difference (warning only)"))
    if raw.get("Status", 0):
        result.append(("error", f"Status alarm: {raw['Status']}"))
    torque = raw.get("Torque_Enable")
    if expected_active is not None and torque is not None and torque != int(expected_active):
        result.append(("error", f"Torque mismatch: expected {int(expected_active)}, got {torque}"))
    elif expected_active is None and torque:
        result.append(("warning", "Torque ON; monitor cannot switch it OFF"))
    voltage = raw.get("Present_Voltage")
    if voltage is not None and all(k in settings for k in ("Min_Voltage_Limit", "Max_Voltage_Limit")):
        if not settings["Min_Voltage_Limit"] <= voltage <= settings["Max_Voltage_Limit"]:
            result.append(("error", "Voltage outside device limits"))
    return result


def stop_reason(row):
    return next((message for severity, message in row["issues"]
                 if severity == "error" and message != "Voltage outside device limits"), None)


class Collector:
    def __init__(self, bus, output, *, demo=False):
        self.bus = bus
        self.demo = demo
        self.events = queue.Queue(maxsize=DASHBOARD_QUEUE_SIZE)
        self.settings = {}
        self.latest = None
        self.temperature_guard = None  # assigned by the teaching worker, never by read-only monitoring
        self.index = 0
        self.started = False
        self.path = Path(output) / ("motors_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jsonl")
        if bus is not None:
            temperature.validate_responses(bus)

    def emit(self, frame):
        self.latest = frame
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a" if self.started else "x", encoding="utf-8") as stream:
            stream.write(json.dumps(frame) + "\n")
        self.started = True
        if self.events.full():
            try:
                self.events.get_nowait()
            except queue.Empty:
                pass
        self.events.put_nowait(frame)

    def capture(self, active=None, context=None):
        began = time.monotonic()
        rows = []
        for motor_id, name in enumerate(JOINTS, 1):
            motor = self.bus.motors[name]
            table = self.bus.model_ctrl_table[motor.model]
            row = {"id": motor_id, "name": name, "raw": {}, "unit": "0-100" if motor_id == 6 else "deg"}
            if name not in self.settings:
                packet = temperature.read_packet(self.bus, 0, 40, motor_id)
                if packet.get("error"):
                    row["settings_error"] = packet["error"]
                elif "data" in packet:
                    data = bytes(packet["data"])
                    self.settings[name] = {key: int.from_bytes(data[table[key][0]:sum(table[key])], "little")
                                           for key in SETTINGS}
                row["settings_packet"] = packet
            row["settings"] = self.settings.get(name, {})
            pair = temperature.read_pair(self.bus, self.index, motor_id)
            row["packets"] = pair
            row["read_error"] = pair.get("read_error", "")
            row["temperature_single"] = pair.get("temperature_single_raw")
            data = bytes.fromhex(pair.get("block_hex", ""))
            if len(data) == 31 and pair.get("block_comm") == 0:
                for key in LIVE:
                    address, length = table[key]
                    row["raw"][key] = int.from_bytes(data[address - 40:address - 40 + length], "little")
                try:
                    for key, dest in (("Goal_Position", "goal"), ("Present_Position", "position")):
                        signed = self.bus._decode_sign(key, {motor_id: row["raw"][key]})
                        row[dest] = self.bus._normalize(signed)[motor_id]
                    row["error"] = row["goal"] - row["position"]
                    row["load_signed"] = self.bus._decode_sign("Present_Load", {motor_id: row["raw"]["Present_Load"]})[motor_id]
                except Exception as exc:
                    row["read_error"] += f" Conversion: {exc}"
            if self.temperature_guard is not None and active is not None:
                try:
                    check_readings(row, self.temperature_guard, name,
                                   {'block': row['raw'].get('Present_Temperature'),
                                    'single': row.get('temperature_single')}, active=active, source='health')
                except TemperatureFault:
                    pass  # preserve the evidence in the frame before validate_live raises
                row['temperature_checked'] = True
            row["issues"] = issues(row, active)
            rows.append(row)
            if active is not None and stop_reason(row):
                break
        complete = len(rows) == len(JOINTS)
        for motor_id in range(len(rows) + 1, len(JOINTS) + 1):
            name = JOINTS[motor_id - 1]
            rows.append({"id": motor_id, "name": name, "raw": {}, "settings": self.settings.get(name, {}),
                         "unit": "0-100" if motor_id == 6 else "deg",
                         "issues": [("warning", "Not read after stop-condition reading")]})
        frame = {"time": now_iso(), "mode": "teaching" if active is not None else "read_only",
                 "connected": True, "complete": complete, "demo": self.demo, "rows": rows, "context": context or {},
                 "cycle_ms": (time.monotonic() - began) * 1000, "log_path": str(self.path)}
        self.index += 1
        self.emit(frame)
        return frame

    def finish(self, outcome, cleanup_errors=(), *, torque_off_verified=False):
        frame = {**(self.latest or {"rows": [], "demo": self.demo}), "connected": False,
                 "event_time": now_iso(), "outcome": outcome, "cleanup_errors": list(cleanup_errors),
                 "torque_off_verified": torque_off_verified, "log_path": str(self.path)}
        self.emit(frame)

    def fault(self, outcome, sample):
        frame = self.latest or {"rows": [], "demo": self.demo}
        check = sample.get("last_check", {})
        rows = [{**row, "issues": [("error", f"Health check: {check.get('register')}={check.get('value')}"),
                                   *row["issues"]]} if row["name"] == check.get("joint") else row
                for row in frame["rows"]]
        self.emit({**frame, "rows": rows,
                   "health_failure": sample, "outcome": outcome, "log_path": str(self.path)})


def validate_live(frame):
    # Display device voltage limits, but retain the application's existing stop rules.
    for row in frame["rows"]:
        message = stop_reason(row)
        if message:
            if row.get('temperature_fault'):
                raise TemperatureFault(message, row['temperature_fault'])
            raise RuntimeError(f"ID{row['id']} {row['name']}: {message}")


def make_bus():
    if version("lerobot") != "0.6.1":
        raise RuntimeError("Requires verified LeRobot 0.6.1")
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus
    path = Path.home() / "so101-training/calibration/follower_left.json"
    calibration = json.loads(path.read_text())
    for index, name in enumerate(JOINTS, 1):
        if calibration[name]["id"] != index:
            raise ValueError("Unexpected follower_left calibration IDs")
    bus = FeetechMotorsBus(PORT, {
        name: Motor(i, "sts3215", MotorNormMode.RANGE_0_100 if i == 6 else MotorNormMode.DEGREES)
        for i, name in enumerate(JOINTS, 1)}, calibration={name: MotorCalibration(**calibration[name]) for name in JOINTS})
    protect_port(bus)
    readonly_packets(bus)
    return bus


def readonly_worker(collector, stop):
    errors = []
    outcome = "Monitoring stopped; no motor writes"
    try:
        ensure_ports_available([PORT])
        collector.bus.connect(handshake=False)
        while not stop.is_set():
            began = time.monotonic()
            collector.capture()
            stop.wait(max(0, READONLY_INTERVAL - (time.monotonic() - began)))
    except Exception as exc:
        outcome = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            collector.bus.port_handler.closePort()
        except Exception as exc:
            errors.append(str(exc))
        collector.finish(outcome, errors)


def demo_frame(index=0):
    rows = []
    for i, name in enumerate(JOINTS, 1):
        row = {"id": i, "name": name, "unit": "0-100" if i == 6 else "deg",
               "raw": {"Present_Temperature": 30 + i, "Present_Current": i * 2,
                       "Present_Load": i * 10, "Present_Voltage": 123, "Torque_Enable": 1,
                       "Status": 0, "Torque_Limit": 500},
               "temperature_single": 30 + i, "goal": 40., "position": 39.2,
               "error": .8, "load_signed": i * 10, "packets": {"block_packet_error_raw": 0},
               "settings": {"Max_Temperature_Limit": 70, "Min_Voltage_Limit": 40,
                            "Max_Voltage_Limit": 140, "Overload_Torque": 25, "Protection_Current": 250,
                            "Protection_Time": 200, "Max_Torque_Limit": 500}}
        if i == 2:
            row["raw"]["Present_Temperature"] = 150 if index % 12 >= 9 else 32
        if i == 6:
            row["temperature_single"] = 31
        row["issues"] = issues(row, True)
        rows.append(row)
    return {"time": now_iso(), "mode": "demo", "demo": True, "connected": True,
            "rows": rows, "context": {"state": "DEMO"}, "cycle_ms": 15., "log_path": ""}
