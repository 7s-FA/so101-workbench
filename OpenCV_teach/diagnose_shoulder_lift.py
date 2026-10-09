"""Single-process selected-arm teleoperation with read-only ID 2/optional ID 3 telemetry.

LeRobot 0.6.1 only. Normal connect/configure and send_action are the only
operational writes; no diagnostic setting writes or calibration updates.
See SHOULDER_DIAGNOSTIC.ko.md for units, limitations, and operating procedure.
"""

import argparse
import csv
from datetime import datetime
import fcntl
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import termios
import time


from teaching_settings import (DIAGNOSTIC_SECONDS, DIAGNOSTIC_FPS, DIAGNOSTIC_TELEMETRY_HZ,
                               DIAGNOSTIC_MAX_ERRORS, DIAGNOSTIC_RELATIVE_TARGET, DIAGNOSTIC_LOG_DIRECTORY)

MOTOR = "shoulder_lift"
KEY = MOTOR + ".pos"
SETTINGS = (
    "P_Coefficient", "I_Coefficient", "D_Coefficient", "Max_Torque_Limit",
    "Torque_Limit", "Protection_Current", "Protective_Torque", "Protection_Time",
    "Overload_Torque", "Over_Current_Protection_Time", "Max_Temperature_Limit",
    "Unloading_Condition", "Min_Voltage_Limit", "Max_Voltage_Limit",
    "Firmware_Major_Version", "Firmware_Minor_Version", "Model_Number",
)
REGISTERS = {
    "Goal_Position": "goal_register_raw",
    "Present_Position": "present_position_raw",
    "Present_Velocity": "present_velocity_raw",
    "Present_Load": "present_load_raw",
    "Present_Current": "present_current_raw",
    "Present_Voltage": "present_voltage_raw",
    "Present_Temperature": "present_temperature",
    "Status": "status_raw",
    "Moving": "moving",
    "Torque_Enable": "torque_enable",
}
COLUMNS = (
    "timestamp_iso", "elapsed_s", "event", "leader_requested_goal",
    "actual_sent_goal", "goal_register", "goal_register_raw", "present_position",
    "present_position_raw", "position_error", "leader_position_error",
    "present_velocity", "present_velocity_raw", "present_load_raw",
    "present_load_converted", "present_load_signed", "present_current_raw",
    "present_current_converted", "present_voltage_raw", "present_voltage_converted",
    "present_temperature", "status_raw", "status_decoded_if_verified", "moving",
    "torque_enable", "control_loop_ms", "control_loop_fps", "control_work_ms",
    "control_window_fps", "telemetry_read_ms", "telemetry_comm_result",
    "telemetry_packet_error_raw", "telemetry_raw_hex", "telemetry_read_error",
    "control_error", "consecutive_control_errors",
    "temperature_single_raw", "temperature_raw_difference", "temperature_single_hex",
    "temperature_single_comm", "temperature_single_packet_error_raw",
    "temperature_single_read_ms", "temperature_single_offset_ms",
    "temperature_single_read_error", "telemetry_total_ms",
    "side", "motor", "motor_id", "telemetry_offset_ms", "position_unit",
)


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def read_settings(bus, motor_name=MOTOR):
    """Public read() is sufficient for unsigned configuration registers."""
    result = {}
    print(f"{motor_name} ID {bus.motors[motor_name].id} configuration (after normal configure; raw)")
    for name in SETTINGS:
        try:
            value = bus.read(name, motor_name, normalize=False, num_retry=0)
            result[name] = {"raw": value}
        except Exception as exc:
            result[name] = {"raw": None, "error": str(exc)}
        print(f"{name:32}: {result[name]}", flush=True)
    return result


def read_telemetry(bus, verify_temperature=False, motor_name=MOTOR):
    """One unicast READ through the existing bus; keep payload on motor alarms.

    Public read() decodes sign even with normalize=False and raises on a
    nonzero packet error. The installed SDK's block read preserves both.
    No extra PortHandler, serial.Serial, thread, or register write is used.
    """
    row = {}
    start = time.monotonic()
    try:
        motor = bus.motors[motor_name]
        table = bus.model_ctrl_table[motor.model]
        first = min(table[name][0] for name in REGISTERS)
        end = max(sum(table[name]) for name in REGISTERS)
        data, comm, packet_error = bus.packet_handler.readTxRx(
            bus.port_handler, motor.id, first, end - first
        )
        row.update(telemetry_comm_result=comm, telemetry_packet_error_raw=packet_error,
                   telemetry_raw_hex=bytes(data).hex())
        if comm != 0:
            raise ConnectionError(bus.packet_handler.getTxRxResult(comm))
        if len(data) != end - first:
            raise ConnectionError(f"Short telemetry payload: {len(data)}/{end - first}")
        raw = {}
        for name, column in REGISTERS.items():
            address, size = table[name]
            raw[name] = int.from_bytes(bytes(data[address - first:address - first + size]), "little")
            row[column] = raw[name]
        # These installed LeRobot helpers are pure conversions, with no I/O.
        for register, column in (("Goal_Position", "goal_register"),
                                 ("Present_Position", "present_position")):
            signed = bus._decode_sign(register, {motor.id: raw[register]})
            row[column] = bus._normalize(signed)[motor.id]
        for register, column in (("Present_Load", "present_load_signed"),
                                 ("Present_Velocity", "present_velocity")):
            row[column] = bus._decode_sign(register, {motor.id: raw[register]})[motor.id]
        # No verified physical-unit scales or Status-register bit definitions.
        # Converted columns remain empty, rather than silently guessing units.
        if packet_error:
            row["telemetry_read_error"] = f"Motor response packet error={packet_error}; payload retained"
    except Exception as exc:
        row["telemetry_read_error"] = f"{type(exc).__name__}: {exc}"
    row["telemetry_read_ms"] = (time.monotonic() - start) * 1000
    if verify_temperature:
        single_start = time.monotonic()
        row["temperature_single_offset_ms"] = (single_start - start) * 1000
        try:
            motor = bus.motors[motor_name]
            address, length = bus.model_ctrl_table[motor.model]["Present_Temperature"]
            data, comm, error = bus.packet_handler.readTxRx(
                bus.port_handler, motor.id, address, length
            )
            row.update(temperature_single_hex=bytes(data).hex(), temperature_single_comm=comm,
                       temperature_single_packet_error_raw=error)
            if comm != 0 or len(data) != length:
                raise ConnectionError(f"Temperature single read: comm={comm}, payload={len(data)}/{length}")
            row["temperature_single_raw"] = int.from_bytes(bytes(data), "little")
            if "present_temperature" in row:
                row["temperature_raw_difference"] = row["present_temperature"] - row["temperature_single_raw"]
            if error:
                row["temperature_single_read_error"] = f"Motor packet error={error}; payload retained"
        except Exception as exc:
            row["temperature_single_read_error"] = f"{type(exc).__name__}: {exc}"
        row["temperature_single_read_ms"] = (time.monotonic() - single_start) * 1000
    row["telemetry_total_ms"] = (time.monotonic() - start) * 1000
    return row


def display(row):
    def fmt(key):
        value = row.get(key)
        if value is None or value == "":
            return "-"
        return f"{value:.2f}" if isinstance(value, float) else str(value)

    fields = ("side", "motor_id", "position_unit", "elapsed_s", "actual_sent_goal", "present_position", "position_error",
              "present_velocity", "present_load_raw", "present_current_raw",
              "present_voltage_raw", "present_temperature", "temperature_single_raw", "status_raw",
              "torque_enable", "control_window_fps")
    print(" ".join(f"{fmt(key):>9}" for key in fields), flush=True)
    for key in ("telemetry_read_error", "temperature_single_read_error", "control_error"):
        if row.get(key):
            print(f"  {key}: {row[key]}", flush=True)


def telemetry_motors(args):
    if getattr(args, "wrist_only", False):
        return ("wrist_roll",)
    return (MOTOR, "elbow_flex") if getattr(args, "include_elbow", False) else (MOTOR,)


class WristTrackingGuard:
    """Conservative stop for this diagnostic, not a motor-fault classifier."""

    def __init__(self):
        self.previous_position = None
        self.still_since = None

    def check(self, sample, now):
        position = sample.get("present_position")
        error = sample.get("position_error")
        requested_error = sample.get("leader_position_error", error)
        if position is None or error is None:
            return None
        if not math.isfinite(position) or not math.isfinite(error):
            return "Invalid wrist position/error; stopping."
        if self.previous_position is not None and abs(position - self.previous_position) > 20:
            return "Wrist position changed over 20 degrees between telemetry samples; stopping."
        self.previous_position = position
        if abs(error) <= 2 or abs(requested_error) <= 8:
            self.still_since = None
        elif self.still_since is None or abs(position - self.still_since[1]) > .5:
            self.still_since = (now, position)
        elif now - self.still_since[0] >= 1:
            return "Wrist stationary for 1 second with sent-target error over 2 and leader error over 8 degrees; stopping."
        return None


def control_loop(leader, follower, writer, stream, args, *, targets=None):
    start = time.monotonic()
    previous = None
    telemetry_due = start
    window_start = start
    window_cycles = 0
    failures = 0
    wrist_guards = {}
    motors = telemetry_motors(args)
    if targets is None:
        targets = [(getattr(args, "side", "right"), follower.bus, name, name + ".pos")
                   for name in motors]
    print("     side  motor_id position_unit    time_s      sent       pos       err  vel_sraw  load_raw   cur_raw  volt_raw  temp_raw temp_sraw status_raw    torque   avg_fps")
    while time.monotonic() - start < args.seconds:
        cycle_start = time.monotonic()
        row = {"timestamp_iso": now_iso(), "elapsed_s": cycle_start - start,
               "event": "telemetry"}
        if previous is not None:
            period = cycle_start - previous
            row.update(control_loop_ms=period * 1000, control_loop_fps=1 / period)
        previous = cycle_start
        control_error = None
        action, sent = None, None
        try:
            if not leader.is_connected or not follower.is_connected:
                raise ConnectionError("Leader or follower is disconnected")
            action = leader.get_action()
            sent = follower.send_action(action)
            failures = 0
        except Exception as exc:
            failures += 1
            control_error = exc
            row.update(event="control_error", control_error=f"{type(exc).__name__}: {exc}",
                       consecutive_control_errors=failures)
        row["control_work_ms"] = (time.monotonic() - cycle_start) * 1000
        window_cycles += 1
        fatal = control_error is not None and (
            failures >= args.max_control_errors
            or not leader.is_connected or not follower.is_connected
            or not isinstance(control_error, (OSError, RuntimeError))
        )
        if cycle_start >= telemetry_due or control_error is not None:
            # A command error forces a snapshot before disconnect, if still possible.
            samples = []
            for side, bus, motor_name, key in targets:
                sample = dict(row, side=side, motor=motor_name,
                              motor_id=bus.motors[motor_name].id,
                              position_unit="normalized_0_100" if motor_name == "gripper" else "degrees",
                              telemetry_offset_ms=(time.monotonic() - cycle_start) * 1000)
                if action is not None and key in action:
                    sample["leader_requested_goal"] = action[key]
                if sent is not None and key in sent:
                    sample["actual_sent_goal"] = sent[key]
                sample.update(read_telemetry(bus, verify_temperature=args.verify_temperature,
                                             motor_name=motor_name))
                if getattr(args, "wrist_only", False) and (
                    sample.get("telemetry_read_error") or sample.get("status_raw")
                    or sample.get("present_temperature", 0) >= args.wrist_temperature_limit_raw
                ):
                    fatal = True
                    row["control_error"] = "Wrist telemetry failure, status alarm, or temperature limit; stopping."
                    sample.update(event="telemetry_fault", control_error=row["control_error"])
                if "present_position" in sample:
                    if "actual_sent_goal" in sample:
                        sample["position_error"] = sample["actual_sent_goal"] - sample["present_position"]
                    if "leader_requested_goal" in sample:
                        sample["leader_position_error"] = sample["leader_requested_goal"] - sample["present_position"]
                if sample.get("torque_enable") == 0:
                    fatal = True
                    row["control_error"] = f"Follower {side} ID {sample['motor_id']} torque is disabled; stopping without re-enabling."
                    sample.update(event="torque_disabled", control_error=row["control_error"])
                if getattr(args, "wrist_only", False) and not sample.get("control_error"):
                    guard = wrist_guards.setdefault((side, motor_name), WristTrackingGuard())
                    guard_error = guard.check(sample, time.monotonic())
                    if guard_error:
                        fatal = True
                        row["control_error"] = guard_error
                        sample.update(event="tracking_guard_stop", control_error=guard_error)
                samples.append(sample)
            sample_end = time.monotonic()
            for sample in samples:
                sample["control_window_fps"] = (window_cycles / (sample_end - window_start)
                                                 if window_start != start else "")
                writer.writerow(sample)
                display(sample)
            stream.flush()
            window_start, window_cycles = sample_end, 0
            # Do not catch up with bursts after a slow read or timeout.
            telemetry_due = cycle_start + 1 / args.telemetry_hz
        if fatal:
            raise RuntimeError(row.get("control_error", "Control communication failure")) from control_error
        remaining = 1 / args.fps - (time.monotonic() - cycle_start)
        if remaining > 0:
            time.sleep(remaining)


def ensure_ports_available(ports):
    if shutil.which("fuser") is None:
        raise RuntimeError("fuser is required to check port owners (Ubuntu package: psmisc)")
    resolved = [Path(port).expanduser().resolve(strict=True) for port in ports]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Leader and follower resolve to the same serial device")
    for path in resolved:
        if not path.is_char_device() or not os.access(path, os.R_OK | os.W_OK):
            raise ValueError(f"Not a readable/writable serial device: {path}")
        check = subprocess.run(["fuser", "-v", str(path)], capture_output=True, text=True, timeout=5)
        if check.returncode == 0:
            raise RuntimeError(f"Port already in use; stop its owner first: {path}\n{check.stdout}{check.stderr}")
        if check.returncode != 1:
            raise RuntimeError(f"Cannot check port owners: {path}\n{check.stderr}")
    return [str(path) for path in resolved]


def protect_port(bus):
    """Lock the bus's own descriptor before its handshake/configuration writes."""
    original_open = bus.port_handler.openPort

    def open_exclusively():
        opened = original_open()
        if opened:
            try:
                fd = bus.port_handler.ser.fileno()
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.ioctl(fd, termios.TIOCEXCL)
            except BaseException:
                bus.port_handler.closePort()
                raise
        return opened

    bus.port_handler.openPort = open_exclusively


def compare_calibration(arm):
    fields = {"homing_offset": "Homing_Offset", "range_min": "Min_Position_Limit",
              "range_max": "Max_Position_Limit"}
    report = {"arm": arm.id, "file": str(arm.calibration_fpath),
              "matches": True, "motors": {}}
    print(f"Calibration comparison: {arm.id}\nFile: {arm.calibration_fpath}", flush=True)
    for name, expected in arm.calibration.items():
        entries = {}
        for field, register in fields.items():
            entry = {"file": getattr(expected, field)}
            try:
                entry["motor"] = arm.bus.read(register, name, normalize=False, num_retry=0)
                entry["matches"] = entry["file"] == entry["motor"]
            except Exception as exc:
                entry.update(motor=None, matches=False, error=f"{type(exc).__name__}: {exc}")
            if not entry["matches"]:
                report["matches"] = False
                print(f"  {name}.{field}: file={entry['file']} motor={entry['motor']}"
                      f" {entry.get('error', 'MISMATCH')}", flush=True)
            entries[field] = entry
        report["motors"][name] = entries
    print("MATCH" if report["matches"] else "MISMATCH OR READ ERROR", flush=True)
    return report


def check_motor_calibration(arms, log_dir):
    """Read both buses without arm.connect/configure or torque writes on exit."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / ("calibration_comparison_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
    reports = []
    try:
        for arm in arms:
            try:
                # Bus connect performs model/firmware reads and host serial setup only.
                arm.bus.connect()
                reports.append(compare_calibration(arm))
            except Exception as exc:
                reports.append({"arm": arm.id, "matches": False,
                                "error": f"{type(exc).__name__}: {exc}"})
                print(f"{arm.id}: {exc}", file=sys.stderr)
            finally:
                # bus.disconnect() would disable torque; close only the host descriptor.
                arm.bus.port_handler.closePort()
    finally:
        path.write_text(json.dumps({"timestamp_iso": now_iso(), "mode": "read_only_calibration",
                                    "arms": reports}, indent=2) + "\n", encoding="utf-8")
        print(f"Comparison saved: {path}", flush=True)
    return 0 if all(report["matches"] for report in reports) else 1


def make_follower(args, *, side="left"):
    """Construct one SO-101 follower without importing/creating a leader or opening USB."""
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    class DiagnosticFollower(SO101Follower):
        def calibrate(self):
            raise RuntimeError("Follower calibration mismatch; normal GUI never writes calibration")
    arm = DiagnosticFollower(SO101FollowerConfig(port=args.follower_port, id=getattr(args,"follower_id",f"follower_{side}"),
        calibration_dir=args.calibration_dir, use_degrees=True, max_relative_target=5., cameras={}))
    expected = {name: motor.id for name,motor in arm.bus.motors.items()}
    if set(arm.calibration) != set(expected):
        raise ValueError(f"SO-101 팔로워 보정이 필요합니다: {arm.calibration_fpath}")
    for name,entry in arm.calibration.items():
        if entry.id != expected[name] or not 0 <= entry.range_min < entry.range_max <= 4095:
            raise ValueError(f"Invalid follower calibration: {arm.id}/{name}")
    protect_port(arm.bus)
    return arm


def make_arms(args, *, side="right"):
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    class DiagnosticLeader(SO101Leader):
        def calibrate(self):
            raise RuntimeError("Leader calibration mismatch; diagnostic mode never writes calibration")

    class DiagnosticFollower(SO101Follower):
        def calibrate(self):
            raise RuntimeError("Follower calibration mismatch; diagnostic mode never writes calibration")

    leader = DiagnosticLeader(SO101LeaderConfig(
        port=args.leader_port, id=f"leader_{side}", calibration_dir=args.calibration_dir,
        use_degrees=True,
    ))
    follower = DiagnosticFollower(SO101FollowerConfig(
        port=args.follower_port, id=f"follower_{side}", calibration_dir=args.calibration_dir,
        use_degrees=True, max_relative_target=5.0, cameras={},
    ))
    if args.shoulder_max_relative_target != 5.0:
        limits = dict.fromkeys(follower.bus.motors, 5.0)
        limits[MOTOR] = args.shoulder_max_relative_target
        follower.config.max_relative_target = limits
    for arm in (leader, follower):
        expected = {name: motor.id for name, motor in arm.bus.motors.items()}
        if set(arm.calibration) != set(expected):
            raise ValueError(f"Missing/invalid calibration: {arm.calibration_fpath}")
        for name, entry in arm.calibration.items():
            if entry.id != expected[name] or entry.range_min >= entry.range_max:
                raise ValueError(f"Invalid calibration entry: {arm.id}/{name}")
        protect_port(arm.bus)
    return leader, follower


def disconnect_arms(leader, follower):
    errors = []
    for label, arm in (("leader", leader), ("follower", follower)):
        if arm is None:
            continue
        try:
            if arm.bus.is_connected:
                arm.disconnect()
        except BaseException as exc:
            errors.append(f"{label} disconnect: {exc}; support arm and check DC power")
        finally:
            # SDK disconnect can raise during torque disable before closing the port.
            try:
                if arm.bus.is_connected:
                    arm.bus.port_handler.closePort()
            except BaseException as exc:
                errors.append(f"{label} serial close: {exc}")
    return errors


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("left", "right"), default="right",
                        help="Select port environment variables and calibration IDs; default: right")
    parser.add_argument("--leader-port")
    parser.add_argument("--follower-port")
    parser.add_argument("--calibration-dir", default=os.environ.get("SO101_CAL_DIR"))
    parser.add_argument("--log-dir", type=Path, default=Path(DIAGNOSTIC_LOG_DIRECTORY).expanduser())
    parser.add_argument("--seconds", type=float, default=DIAGNOSTIC_SECONDS)
    parser.add_argument("--fps", type=float, default=DIAGNOSTIC_FPS)
    parser.add_argument("--telemetry-hz", type=float, default=DIAGNOSTIC_TELEMETRY_HZ)
    parser.add_argument("--max-control-errors", type=int, default=DIAGNOSTIC_MAX_ERRORS)
    parser.add_argument("--shoulder-max-relative-target", type=float, choices=(5.0, 7.0), default=DIAGNOSTIC_RELATIVE_TARGET,
                        help="Shoulder lift target offset limit in degrees; other joints stay at 5.0")
    parser.add_argument("--verify-temperature", action="store_true",
                        help="Compare block temperature with an extra single-register READ on the same bus")
    parser.add_argument("--include-elbow", action="store_true",
                        help="Also log elbow_flex (ID 3) on the same bus; does not change its target limit")
    parser.add_argument("--wrist-only", action="store_true",
                        help="Log wrist_roll (ID 5) only during normal all-joint teleoperation")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Validate paths/API without opening serial ports")
    mode.add_argument("--compare-calibration", action="store_true",
                      help="Read and compare motor calibration only; no configure, torque or setting writes")
    args = parser.parse_args(argv)
    if args.wrist_only and args.include_elbow:
        parser.error("Choose --wrist-only or --include-elbow, not both")
    if args.wrist_only and args.seconds > 30:
        parser.error("Wrist diagnostic requires --seconds <= 30")
    for role in ("leader", "follower"):
        field = f"{role}_port"
        if getattr(args, field) is None:
            setattr(args, field, os.environ.get(f"SO101_{role.upper()}_{args.side.upper()}_PORT"))
    for name in ("leader_port", "follower_port", "calibration_dir"):
        if not getattr(args, name) or not str(getattr(args, name)).strip():
            env_file = "bimanual_env.sh" if args.side == "left" else "right_arm_env.sh"
            parser.error(f"{name} is empty; source {env_file} first")
    for name in ("seconds", "fps", "telemetry_hz"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} must be finite and positive")
    if args.fps > 30 or args.telemetry_hz > args.fps or args.max_control_errors < 1:
        parser.error("Require telemetry_hz <= fps <= 30 and max_control_errors >= 1")
    if args.shoulder_max_relative_target > 5.0 and args.seconds > 30:
        parser.error("The 7-degree comparison requires --seconds <= 30")
    args.calibration_dir = Path(args.calibration_dir).expanduser().resolve(strict=True)
    args.log_dir = args.log_dir.expanduser()
    return args


def main(argv=None):
    args = parse_args(argv)
    if version("lerobot") != "0.6.1":
        raise RuntimeError("This diagnostic was verified against installed LeRobot 0.6.1 only")
    resolved = ensure_ports_available([args.leader_port, args.follower_port])
    leader, follower = make_arms(args, side=args.side)
    print(f"Target arm: {args.side}")
    print(f"leader: {args.leader_port} -> {resolved[0]}")
    print(f"follower: {args.follower_port} -> {resolved[1]}")
    print(f"Relative target limits: {follower.config.max_relative_target}")
    print(f"Telemetry motors: {', '.join(telemetry_motors(args))}")
    if args.check_only:
        print("Preflight OK. No serial ports opened; no motor writes.")
        return 0
    if args.compare_calibration:
        return check_motor_calibration((leader, follower), args.log_dir)

    args.log_dir.mkdir(parents=True, exist_ok=True)
    prefix = "wrist_roll_diagnostic_" if args.wrist_only else "shoulder_lift_diagnostic_"
    stem = prefix + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    csv_path = args.log_dir / (stem + ".csv")
    metadata_path = args.log_dir / (stem + ".json")
    metadata = {"started_at": now_iso(), "lerobot": version("lerobot"),
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "resolved_ports": resolved, "motor": telemetry_motors(args)[0],
                "motor_id": 5 if args.wrist_only else 2,
                "max_relative_target": follower.config.max_relative_target,
                "telemetry_motors": list(telemetry_motors(args)),
                "units": {"positions": "LeRobot calibrated degrees",
                          "velocity": "sign-decoded register units, physical unit unverified",
                          "load_signed": "sign-decoded register units, not torque or percent",
                          "current_voltage_temperature_status": "raw; conversions/Status decoding unverified"}}
    if args.wrist_only:
        metadata["wrist_tracking_guard"] = {
            "max_sample_position_change_deg": 20,
            "stationary_duration_s": 1, "stationary_displacement_deg": .5,
            "stationary_target_error_deg": 2,
            "stationary_leader_error_deg": 8,
            "note": "Conservative diagnostic stop, not proof of hardware failure; starts with control-loop telemetry.",
        }
    stream = None
    writer = None
    exit_code = 0
    cleanup_errors = []

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    old_sigterm = signal.signal(signal.SIGTERM, interrupted)
    print(f"CSV: {csv_path}\nSupport both arms. Normal configure can enable follower torque.", flush=True)
    try:
        stream = csv_path.open("x", newline="", encoding="utf-8")
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        stream.flush()
        leader.connect()
        follower.connect()
        if args.wrist_only:
            metadata["wrist_configuration_after_connect"] = read_settings(follower.bus, "wrist_roll")
            limit = metadata["wrist_configuration_after_connect"]["Max_Temperature_Limit"]["raw"]
            if not isinstance(limit, int) or not 0 < limit < 256:
                raise RuntimeError("Cannot verify wrist temperature limit; stopping.")
            args.wrist_temperature_limit_raw = limit
            metadata["wrist_temperature_limit_raw"] = limit
        else:
            metadata["shoulder_configuration_after_connect"] = read_settings(follower.bus)
        if args.include_elbow:
            metadata["elbow_configuration_after_connect"] = read_settings(follower.bus, "elbow_flex")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        control_loop(leader, follower, writer, stream, args)
        metadata["outcome"] = "time_limit"
    except KeyboardInterrupt:
        metadata["outcome"] = "interrupted"
        exit_code = 130
    except Exception as exc:
        metadata["outcome"] = f"{type(exc).__name__}: {exc}"
        print(metadata["outcome"], file=sys.stderr)
        exit_code = 1
    finally:
        # CSV failures and repeated interrupts must not skip either disconnect.
        try:
            if stream is not None:
                try:
                    if writer is not None:
                        writer.writerow({"timestamp_iso": now_iso(), "event": "stop",
                                         "control_error": metadata.get("outcome", "unexpected_exit")})
                    stream.flush()
                finally:
                    stream.close()
        except BaseException as exc:
            cleanup_errors.append(f"CSV close: {exc}")
        finally:
            cleanup_errors.extend(disconnect_arms(leader, follower))
            signal.signal(signal.SIGTERM, old_sigterm)
        metadata.update(finished_at=now_iso(), cleanup_errors=cleanup_errors)
        try:
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            cleanup_errors.append(f"Metadata save: {exc}")
        for error in cleanup_errors:
            print(error, file=sys.stderr)
        if cleanup_errors:
            exit_code = 1
    print(f"Saved: {csv_path}\nSettings/outcome: {metadata_path}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
