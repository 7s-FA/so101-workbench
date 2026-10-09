"""Post-stop evidence collection; never writes motor registers or resumes motion."""

from datetime import datetime
import time

from monitor_shoulder_temperature import read_pair, validate_responses

from teaching_settings import (CAPTURE_SECONDS, CAPTURE_ROUNDS, CAPTURE_INTERVAL,
                               TEMPERATURE_LIMIT, TEMPERATURE_DIFFERENCE_WARNING)


def assess(rows):
    complete = [row for row in rows if not row.get("read_error")
                and all(isinstance(row.get(key), int) for key in
                        ("temperature_block_raw", "temperature_single_raw", "status_raw",
                         "torque_enable_raw"))]
    high = [row for row in rows if any(isinstance(row.get(key), int) and row[key] >= TEMPERATURE_LIMIT
            for key in ("temperature_block_raw", "temperature_single_raw"))]
    return {
        "read_error_count": sum(bool(row.get("read_error")) for row in rows),
        "complete_pairs": len(complete),
        "high_temperature_pairs": len(high),
        "disagreeing_pairs": sum(abs(row["temperature_block_raw"] - row["temperature_single_raw"]) > TEMPERATURE_DIFFERENCE_WARNING
                                  for row in complete),
        "status_alarm_observed": any(row.get("status_raw", 0) for row in rows),
        "result": ("high_temperature_reobserved" if high else
                   "normal_after_stop" if len(complete) >= 3 and len(complete) == len(rows)
                   and all(row["status_raw"] == 0 and row["torque_enable_raw"] == 0
                           for row in complete) else "inconclusive"),
        "interpretation": "Post-torque-off evidence only; does not prove sensor fault or exclude real heating.",
    }


def capture(bus, joints, emit):
    original = bus.packet_handler.txRxPacket
    rows = {name: [] for name in joints}
    started = time.monotonic()
    validate_responses(bus)
    try:
        for index in range(CAPTURE_ROUNDS):
            for name in joints:
                if time.monotonic() - started >= CAPTURE_SECONDS:
                    return {name: assess(values) for name, values in rows.items()}
                motor_id = bus.motors[name].id
                row = {"event": "fault_probe", "time": datetime.now().astimezone().isoformat(),
                       "elapsed_s": time.monotonic() - started, "round": index,
                       "joint": name, "motor_id": motor_id, "phase": "after_torque_off_attempt"}
                row.update(read_pair(bus, index, motor_id))
                rows[name].append(row)
                emit(row)
            if index + 1 < CAPTURE_ROUNDS:
                time.sleep(min(CAPTURE_INTERVAL, max(0, CAPTURE_SECONDS - (time.monotonic() - started))))
        return {name: assess(values) for name, values in rows.items()}
    finally:
        bus.packet_handler.txRxPacket = original
