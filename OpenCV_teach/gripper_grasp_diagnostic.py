"""Read-only ID 6 grasp telemetry on the teaching worker's existing connection.

No force thresholds are inferred from unloaded readings. An object-present label
is operator context, not confirmation that the object is securely grasped.
"""

from teaching_settings import TEMPERATURE_LIMIT, COOL_TEMPERATURE_LIMIT
from temperature_guard import TemperatureGuard, check_readings

from datetime import datetime
import json
from pathlib import Path

from diagnose_shoulder_lift import now_iso, read_telemetry
from monitor_shoulder_temperature import annotate_temperature_difference, validate_responses

SETTINGS = ("Torque_Limit", "Max_Torque_Limit", "Overload_Torque", "Protection_Time",
            "Protection_Current", "Over_Current_Protection_Time")


class GripperGraspDiagnostic:
    def __init__(self, bus, output, *, sample_kind, identity):
        if sample_kind not in ("empty", "object_present"):
            raise ValueError("Expected empty or object_present sample kind")
        self.bus = bus
        self.sample_kind = sample_kind
        self.identity = identity
        self.index = 0
        self.temperature_guard = TemperatureGuard()
        self.last_active = None
        self.path = Path(output) / (
            "id6_grasp_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jsonl")
        validate_responses(bus)

    def __call__(self, active):
        row = read_telemetry(self.bus, verify_temperature=True, motor_name="gripper")
        row.update(time=now_iso(), motor_id=6, active=active,
                   sample_kind=self.sample_kind, grasp_state="unclassified",
                   units="positions: normalized 0-100; current/load/temperature/voltage: raw")
        annotate_temperature_difference(row, "present_temperature", "temperature_single_raw", 6)
        if "goal_register" in row and "present_position" in row:
            row["position_error"] = row["goal_register"] - row["present_position"]
        if self.index == 0:
            row["identity"] = self.identity
        if active != self.last_active and not row.get("telemetry_read_error"):
            settings = row["settings_raw"] = {}
            try:
                for name in SETTINGS:
                    settings[name] = self.bus.read(name, "gripper", normalize=False, num_retry=0)
            except Exception as exc:
                row["settings_error"] = f"{type(exc).__name__}: {exc}"
        failure = None
        try:
            validate_sample(row, active, self.temperature_guard)
        except RuntimeError as exc:
            failure = exc
            row['diagnostic_error'] = str(exc)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("x" if self.index == 0 else "a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        if self.index == 0:
            print(f"ID6 grasp diagnostic: {self.path}", flush=True)
        if "settings_raw" in row:
            print(f"ID6 settings raw (active={active}): {row['settings_raw']}", flush=True)
        self.last_active = active
        self.index += 1
        if row.get("temperature_warning"):
            print("WARNING: " + row["temperature_warning"], flush=True)
        print(f"{row['time']} ID6 (goal/pos/error: 0-100 units; others: raw) | " + " ".join(
            f"{name}={row.get(key, '--')}" for name, key in (
                ("goal", "goal_register"), ("pos", "present_position"),
                ("error", "position_error"), ("current", "present_current_raw"),
                ("load", "present_load_signed"), ("voltage", "present_voltage_raw"),
                ("temperature", "present_temperature"), ("temperature_single", "temperature_single_raw"),
                ("status", "status_raw"), ("packet_error", "telemetry_packet_error_raw"),
                ("torque", "torque_enable"))), flush=True)
        if failure is not None:
            raise failure
        return row


def validate_sample(row, active, guard=None):
    for key in ("telemetry_read_error", "temperature_single_read_error", "settings_error"):
        if row.get(key):
            raise RuntimeError(f"ID6 grasp diagnostic: {row[key]}")
    for key in ("telemetry_comm_result", "telemetry_packet_error_raw", "temperature_single_comm",
                "temperature_single_packet_error_raw", "status_raw"):
        if row.get(key) != 0:
            raise RuntimeError(f"ID6 grasp diagnostic: {key}={row.get(key)}")
    if row.get("torque_enable") != int(active):
        raise RuntimeError("ID6 grasp diagnostic: unexpected torque state")
    readings = {'block': row.get('present_temperature'), 'single': row.get('temperature_single_raw')}
    if guard is not None:
        check_readings(row, guard, 'gripper', readings, active=active,
                       source='gripper_diagnostic', inactive_limit=COOL_TEMPERATURE_LIMIT)
    else:
        # Stateless callers retain immediate validation; never create a fresh timer per sample.
        limit = TEMPERATURE_LIMIT if active else COOL_TEMPERATURE_LIMIT
        if any(type(value) is not int or not 0 <= value < limit for value in readings.values()):
            raise RuntimeError(f"ID6 temperature {list(readings.values())}: required below {limit} raw")
