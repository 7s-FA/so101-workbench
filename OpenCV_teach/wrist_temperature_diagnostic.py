"""ID 4 telemetry on the teaching worker's existing serial connection."""

from teaching_settings import TEMPERATURE_LIMIT, COOL_TEMPERATURE_LIMIT
from temperature_guard import TemperatureGuard, check_readings

from datetime import datetime
import json
from pathlib import Path

import monitor_shoulder_temperature as temperature


class WristTemperatureDiagnostic:
    def __init__(self, bus, output):
        self.bus = bus
        self.index = 0
        self.temperature_guard = TemperatureGuard()
        self.path = Path(output) / (
            "id4_teaching_temperature_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jsonl")
        temperature.validate_responses(bus)

    def __call__(self, active):
        row = temperature.read_pair(self.bus, self.index, motor_id=4)
        row.update(time=temperature.now_iso(), motor_id=4, active=active,
                   units="raw; physical scale unverified")
        temperature.annotate_temperature_difference(
            row, "temperature_block_raw", "temperature_single_raw", 4)
        failure = None
        try:
            if row.get("read_error"):
                raise RuntimeError(f"ID4 diagnostic read failure: {row['read_error']}")
            if row.get("status_raw") != 0 or row.get("torque_enable_raw") != int(active):
                raise RuntimeError("ID4 diagnostic status alarm or unexpected torque state")
            check_readings(row, self.temperature_guard, 'wrist_flex',
                           {'block': row.get('temperature_block_raw'), 'single': row.get('temperature_single_raw')},
                           active=active, source='wrist_diagnostic', inactive_limit=COOL_TEMPERATURE_LIMIT)
        except RuntimeError as exc:
            failure = exc
            row['diagnostic_error'] = str(exc)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("x" if self.index == 0 else "a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        if self.index == 0:
            print(f"ID4 diagnostic: {self.path}", flush=True)
        self.index += 1
        if row.get("temperature_warning"):
            print("WARNING: " + row["temperature_warning"], flush=True)
        print(f"{row['time']} ID4 wrist_flex raw | "
              f"temperature={row.get('temperature_block_raw', '--')}/{row.get('temperature_single_raw', '--')} | "
              f"current={row.get('current_raw', '--')} voltage={row.get('voltage_raw', '--')} | "
              f"torque={row.get('torque_enable_raw', '--')}", flush=True)
        if failure is not None:
            raise failure
        return row
