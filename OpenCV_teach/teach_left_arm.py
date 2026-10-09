"""SO-101 waypoint and overhead-rectangle teaching. No hardware opens on launch."""
from app_choices import AppChoice
from temperature_guard import TemperatureGuard, TemperatureFault, check_readings, pending_label


import argparse
import csv
from collections import deque
from contextlib import contextmanager
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import queue
import signal
import tempfile
import threading
import time
import unicodedata

from teaching_ui import (configure_ui, joint_label, unit_label, state_label, message_ko,
                         ask_teaching_title, step_correction, INK, MUTED, ACCENT, DANGER, SURFACE, STRIPE)

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
KEYS = tuple(name + ".pos" for name in JOINTS)
UNITS = {key: ("normalized_0_100" if key == "gripper.pos" else "degrees") for key in KEYS}
ROOT = Path(__file__).resolve().parent
SERIALS = {"leader": "5B61035944", "follower": "5B91028218"}
from teaching_settings import (
    ACTIVATION_MOVEMENT_LIMIT, DEMO_RATE, HEALTH_HISTORY_SIZE,
    CONTROL_HZ,
    GUI_POLL_MS,
    HEALTH_INTERVAL,
    GRIPPER_HEALTH_INTERVAL,
    GUI_HEARTBEAT_TIMEOUT,
    DISPLAY_STALE_SECONDS,
    COMMAND_QUEUE_SIZE,
    COMMAND_RATE,
    GRIPPER_COMMAND_RATE,
    SYNC_RATE,
    MAX_COMMAND_DT,
    RELATIVE_TARGET_LIMIT,
    SYNC_TIMEOUT,
    MOVE_TIMEOUT,
    ARRIVAL_SETTLE_SECONDS,
    SYNC_ARRIVAL_TOLERANCE,
    PLAY_GRIPPER_ARRIVAL_TOLERANCE,
    PREVIEW_POSE_TOLERANCE,
    WRIST_MOVE_LIMIT,
    TEACH_START_GAP,
    SYNC_LEADER_DRIFT,
    ENTRY_MARGIN_DEG,
    GRIPPER_ENTRY_MAX_RAW,
    GRIPPER_ENTRY_MARGIN_UNITS,
    ACTIVATION_READ_RETRIES,
    STALL_SECONDS,
    TRACKING_ERROR,
    TRACKING_SENT_ERROR,
    TRACKING_PROGRESS_TICKS,
    TEMPERATURE_LIMIT, TEMPERATURE_DURATION_S, TEMPERATURE_MAX_GAP_S,
    COOL_TEMPERATURE_LIMIT,
    TEMPERATURE_DIFFERENCE_WARNING,
    CAPTURE_HISTORY_SECONDS,
    CAPTURE_STABLE_SECONDS,
    CAPTURE_POSITION_TOLERANCE,
    CAPTURE_TARGET_TOLERANCE,
    MAX_STEPS,
    MAX_NAME_LENGTH,
    TEMPERATURE_LOG_INTERVAL,
    TELEMETRY_LOG_INTERVAL,
    LOG_FLUSH_INTERVAL,
    PLAYBACK_LOG_INTERVAL,
    MOTOR_LOG_INTERVAL,
    CAPTURE_SECONDS,
    CAPTURE_ROUNDS,
    CAPTURE_INTERVAL,
    DASHBOARD_POLL_MS,
    DASHBOARD_HISTORY_SIZE,
    DASHBOARD_EVENT_SIZE,
    DASHBOARD_QUEUE_SIZE,
    DASHBOARD_STALE_SECONDS,
    READONLY_INTERVAL,
    DIAGNOSTIC_SECONDS,
    DIAGNOSTIC_FPS,
    DIAGNOSTIC_TELEMETRY_HZ,
    DIAGNOSTIC_MAX_ERRORS,
    DIAGNOSTIC_RELATIVE_TARGET,
    MONITOR_SECONDS,
    MONITOR_HZ,
    MONITOR_MOTOR_ID,
    REPLAY_ARRIVAL_TOLERANCES,
    OUTPUT_DIRECTORY,
    GUI_GEOMETRY
)


class TrackingProgress:
    """Directional high-water mark: slow progress counts, oscillatory jitter does not."""

    def __init__(self, tick=360 / 4095):
        self.minimum_progress = TRACKING_PROGRESS_TICKS * tick
        self.reset()

    def reset(self):
        self.direction = 0
        self.anchor = None
        self.since = None

    def update(self, now, position, requested, sent):
        error = requested - position
        sent_error = None if sent is None else sent - position
        direction = 1 if error > 0 else -1
        active = abs(error) > TRACKING_ERROR and sent_error is not None and direction * sent_error > TRACKING_SENT_ERROR
        if not active:
            self.reset()
            return {"error": error, "sent_error": sent_error, "progress": 0,
                    "no_progress_s": 0, "blocked": False, "lagging": False}
        if self.direction != direction or self.anchor is None:
            self.direction, self.anchor, self.since = direction, position, now
        progress = direction * (position - self.anchor)
        if progress >= self.minimum_progress - 1e-9:
            self.anchor, self.since = position, now
        elapsed = now - self.since
        return {"error": error, "sent_error": sent_error, "progress": progress,
                "no_progress_s": elapsed, "blocked": elapsed >= STALL_SECONDS, "lagging": True}


class EntryEnvelope:
    """Allow a small initial overshoot only on a path back into calibrated bounds."""

    def __init__(self, bounds, initial):
        self.bounds = bounds
        self.last = initial.copy()
        self.allowed = {}
        for key, (low, high) in bounds.items():
            margin = 0 if key == "gripper.pos" else ENTRY_MARGIN_DEG
            if not low - margin <= initial[key] <= high + margin:
                raise ValueError(f"Initial position too far outside calibration: {key}={initial[key]:.2f}")
            self.allowed[key] = (min(low, initial[key]), max(high, initial[key]))

    def advance(self, target):
        target = pose(target)
        for key, value in target.items():
            low, high = self.allowed[key]
            if not low - 1e-8 <= value <= high + 1e-8:
                raise ValueError(f"Outward motion outside calibrated range blocked: {key}")
        self.last = target.copy()
        return target

    def observe(self, current):
        for key, (low, high) in self.bounds.items():
            if low <= current[key] <= high and low <= self.last[key] <= high:
                self.allowed[key] = (low, high)


def check_cancelled(cancelled):
    if cancelled is not None and cancelled.is_set():
        raise RuntimeError("Connection cancelled")


def sync_goal(backend, approved, leader, follower):
    reference = pose(approved["leader"])
    goal = backend.project_target(reference)
    start = pose(approved["follower"])
    if difference(reference, leader) > PREVIEW_POSE_TOLERANCE or difference(start, follower) > PREVIEW_POSE_TOLERANCE:
        raise ValueError("Poses changed since preview. Keep both arms still and approve a new preview.")
    if abs(goal["wrist_roll.pos"] - follower["wrist_roll.pos"]) > WRIST_MOVE_LIMIT:
        raise ValueError(f"Wrist sync >{WRIST_MOVE_LIMIT:g} degrees blocked. Reposition with motor power OFF; check cable routing.")
    return goal


def stamp():
    return datetime.now().astimezone().isoformat()


def atomic_json(path, value, *, overwrite=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".teaching-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Publish the complete file without replacing an existing file or symlink.
            os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_named_teaching(directory, data):
    title = data["title"].strip()
    if not 1 <= len(title) <= MAX_NAME_LENGTH:
        raise ValueError(f"Use a 1-{MAX_NAME_LENGTH} character teaching title.")
    stem = "".join("_" if char in '<>:"/\\|?*' or unicodedata.category(char).startswith("C")
                   else char for char in unicodedata.normalize("NFC", title))
    stem = stem.strip(" .")
    if stem.lower().endswith(".json"):
        stem = stem[:-5].rstrip(" .")
    stem = stem.encode("utf-8")[:200].decode("utf-8", errors="ignore").rstrip(" .") or "teaching"
    for index in range(1, 10001):
        suffix = "" if index == 1 else f"_{index}"
        path = Path(directory) / f"{stem}{suffix}.json"
        try:
            atomic_json(path, {**data, "kind": "teaching"}, overwrite=False)
        except FileExistsError:
            continue
        return path
    raise FileExistsError("Too many teachings with the same filename; choose another title.")


def pose(values):
    if not isinstance(values, dict) or set(values) != set(KEYS):
        raise ValueError("Expected exactly six joint positions")
    result = {}
    for key in KEYS:
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"Invalid position: {key}")
        low, high = (0, 100) if key == "gripper.pos" else (-180, 180)
        if not low <= value <= high:
            raise ValueError(f"Out of range: {key}={value}")
        result[key] = float(value)
    return result


def difference(a, b):
    return max(abs(a[k] - b[k]) for k in KEYS)


def within_arrival_tolerance(goal, current, state):
    return all(abs(goal[key] - current[key]) <=
               arrival_tolerance(key, state)
               for key in KEYS)


def arrival_tolerance(key, state):
    if state == "PLAY" and key == "gripper.pos":
        return PLAY_GRIPPER_ARRIVAL_TOLERANCE
    return REPLAY_ARRIVAL_TOLERANCES[key] if state in ("MOVE", "PLAY") else SYNC_ARRIVAL_TOLERANCE


def playback_snapshot(name, index, goal, current, sent, started, now):
    joints = {}
    for key in KEYS:
        error = goal[key] - current[key]
        tolerance = arrival_tolerance(key, "PLAY")
        joints[key] = {"goal": goal[key], "actual": current[key],
                       "last_sent": None if sent is None else sent[key],
                       "error": error, "tolerance": tolerance,
                       "outside_tolerance": abs(error) > tolerance}
    return {"time": stamp(), "step": index + 1, "step_name": name,
            "waiting_s": round(now - started, 3), "joints": joints,
            "blocking_joints": [key for key, data in joints.items() if data["outside_tolerance"]]}


def playback_temperatures(sample):
    values = sample.get("temperatures_raw", {}) if isinstance(sample, dict) else {}
    result = {"temperature_sample_time": sample.get("time") if isinstance(sample, dict) else None,
              "temperatures_raw": {name: values.get(name) for name in JOINTS},
              "temperature_complete": bool(sample and sample.get("complete"))}
    if isinstance(sample, dict):
        for key in ('temperature_pending', 'temperature_transition', 'temperature_fault'):
            if key in sample:
                result[key] = sample[key]
        if sample.get("error"):
            result["temperature_error"] = sample["error"]
    return result


def approach(previous, target, dt, rate=COMMAND_RATE, gripper_rate=None):
    # Bound command slew, independently of LeRobot's present-to-goal clipping.
    interval = min(max(dt, 0), MAX_COMMAND_DT)
    result = {}
    for key in KEYS:
        joint_rate = gripper_rate if key == "gripper.pos" and gripper_rate is not None else rate
        step = joint_rate * interval
        result[key] = previous[key] + max(-step, min(step, target[key] - previous[key]))
    return result


def validate_program(data, identity):
    if not isinstance(data, dict) or data.get("schema") not in (1, 2) or data.get("identity") != identity:
        raise ValueError("Wrong arm, calibration, schema, or demo/live program")
    if data.get("units") != UNITS:
        raise ValueError("Wrong position units")
    steps = data.get("steps")
    if not isinstance(steps, list) or len(steps) > MAX_STEPS:
        raise ValueError(f"Invalid steps (maximum {MAX_STEPS})")
    plan = {}
    if data.get("schema") == 2:
        from vision_teaching import validate_plan
        plan = validate_plan(data.get("vision_plan"))
    from safe_episode import validate_boundaries
    validate_boundaries(steps)
    clean = []
    for step in steps:
        if not isinstance(step, dict) or not isinstance(step.get("name"), str) or not step["name"].strip():
            raise ValueError("Each step needs a name")
        if len(step["name"]) > MAX_NAME_LENGTH:
            raise ValueError("Step name too long")
        item = {"name": step["name"], "pose": pose(step.get("pose")),
                "captured_at": str(step.get("captured_at", ""))}
        if step.get("motion"):
            if not plan:
                raise ValueError("카메라 스텝에는 버전 2 목표 정보가 필요합니다.")
            from copy import deepcopy
            from vision_teaching import validate_motion, number
            validate_motion(step, plan["targets"])
            item["motion"] = deepcopy(step["motion"])
            if "path" in step:
                if not isinstance(step["path"], list) or not 1 <= len(step["path"]) <= 1000:
                    raise ValueError("계산된 경로가 올바르지 않습니다.")
                item["path"] = [{"pose": pose(n["pose"]), "hold_s": number(n["hold_s"],0,30),
                                 "duration_s": number(n["duration_s"],0,60)} for n in step["path"]]
        if "fixed_pose" in step:item["fixed_pose"]=pose(step["fixed_pose"])
        if "follow_motion" in step:
            from vision_teaching import validate_motion
            validate_motion({"motion":step["follow_motion"]},plan.get("targets",{}))
            item["follow_motion"]=dict(step["follow_motion"])
        if step.get("gripper_only"):
            if step.get("motion") or step.get("safe_boundary"):raise ValueError("집게 전용 스텝에 이동 목표를 연결할 수 없습니다.")
            item["gripper_only"]=True
        if "teaching_reference" in step:
            from vision_teaching import number
            item["teaching_reference"]={k:number(step["teaching_reference"][k]) for k in ('x_mm','y_mm','yaw_deg')}
        if step.get("safe_boundary"):
            item["safe_boundary"] = step["safe_boundary"]
        clean.append(item)
    return clean


def validate_sequence(backend, steps, current):
    if not isinstance(steps, list) or not 1 <= len(steps) <= MAX_STEPS:
        raise ValueError(f"Select a teaching with 1-{MAX_STEPS} steps")
    targets = []
    previous = pose(current)
    from vision_teaching import expand_steps
    steps = expand_steps(steps,current)
    for index, step in enumerate(steps):
        target = backend.validate_target(step["pose"])
        if abs(target["wrist_roll.pos"] - previous["wrist_roll.pos"]) > WRIST_MOVE_LIMIT:
            raise ValueError(f"step{index + 1}: wrist move >{WRIST_MOVE_LIMIT:g} degrees blocked")
        targets.append(target.copy())
        previous = target
    return targets


def next_step_name(steps):
    numbers = [int(step["name"][4:]) for step in steps
               if step["name"].startswith("step") and step["name"][4:].isascii()
               and step["name"][4:].isdecimal()]
    return f"step{max(numbers, default=0) + 1}"


class Hardware:
    def __init__(self, profile=None, *, follower_only=False):
        self.follower_only = follower_only
        self.profile_path = profile
        import diagnose_shoulder_lift as diagnostic
        if version("lerobot") not in ("0.4.4", "0.6.1"):
            raise RuntimeError("Supported LeRobot versions: 0.4.4 / 0.6.1")
        if follower_only and profile is None:
            raise ValueError("팔로워 단독 모드에는 팔로워 프로필이 필요합니다.")
        if profile is None:
            self.args = diagnostic.parse_args(["--side", "left"])
            for role, serial in SERIALS.items():
                expected = f"/dev/serial/by-id/usb-YOUR_ROBOT-if00"
                if getattr(self.args, role + "_port") != expected:
                    raise ValueError(f"Unexpected LEFT {role} identity; source left_arm_env.sh")
            self.identity = {"side": "left", "serials": SERIALS, "demo": False,
                             "calibration_sha256": {}}
        else:
            from teaching_profile import load_profile
            self.args, self.identity = load_profile(profile, follower_only=follower_only)
            self.follower_only = self.identity.get("control_mode") == "follower_only"
        if self.follower_only:
            self.leader = None
            self.follower = diagnostic.make_follower(self.args, side=self.identity["side"])
        else:
            self.leader, self.follower = diagnostic.make_arms(self.args, side="left")
        self.follower.config.max_relative_target = RELATIVE_TARGET_LIMIT
        for arm in self.arms():
            calibration = json.dumps(json.loads(arm.calibration_fpath.read_text()), sort_keys=True)
            self.identity["calibration_sha256"][arm.id] = hashlib.sha256(calibration.encode()).hexdigest()
        self.limits = self.follower.bus._normalize(
            {entry.id: entry.range_min for entry in self.follower.calibration.values()})
        maxima = self.follower.bus._normalize(
            {entry.id: entry.range_max for entry in self.follower.calibration.values()})
        self.bounds = {name + ".pos": (self.limits[entry.id], maxima[entry.id])
                       for name, entry in self.follower.calibration.items()}
        self.position_ticks = {name + ".pos": (100 / (entry.range_max - entry.range_min)
                                                if name == "gripper" else 360 / 4095)
                               for name, entry in self.follower.calibration.items()}
        from motor_positions import preserve_integer_roundtrip
        preserve_integer_roundtrip(self.follower.bus)
        self.raw_positions = {}
        self.motion_attempted = False
        self.entry_envelope = None
        self.activation_report = []
        self.temperature_sample = None
        self.temperature_guard = TemperatureGuard()

    def arms(self):
        return tuple(arm for arm in (getattr(self, 'leader', None), self.follower) if arm is not None)

    def validate_target(self, target):
        target = pose(target)
        for key, (low, high) in self.bounds.items():
            if not low <= target[key] <= high:
                raise ValueError(f"Outside follower calibration range: {key}={target[key]:.2f}, "
                                 f"allowed [{low:.2f}, {high:.2f}]")
        return target

    def project_target(self, target):
        target = pose(target)
        return {key: max(low, min(high, target[key])) for key, (low, high) in self.bounds.items()}

    def preflight(self):
        import diagnose_shoulder_lift as diagnostic
        return diagnostic.ensure_ports_available([self.args.follower_port] if self.follower_only else [self.args.leader_port, self.args.follower_port])

    def connect(self, cancelled=None):
        import diagnose_shoulder_lift as diagnostic
        self.health_history = deque(maxlen=HEALTH_HISTORY_SIZE)
        self.fault_capture = None
        self.motion_attempted = False
        self.entry_envelope = None
        self.activation_report = []
        self.temperature_sample = None
        self.temperature_guard = TemperatureGuard()
        self.preflight()
        dashboard = getattr(self, "motor_dashboard", None)
        if dashboard is not None:
            dashboard.settings.clear()
        # Keep this stage read-only, regardless of the leader/follower pose gap.
        for arm in self.arms():
            check_cancelled(cancelled)
            arm.bus.connect()
            if not diagnostic.compare_calibration(arm)["matches"]:
                raise RuntimeError(f"Calibration mismatch: {arm.id}; no calibration was written")
            for name in JOINTS:
                if arm.bus.read("Torque_Enable", name, normalize=False) != 0:
                    raise RuntimeError(f"{arm.id}/{name}: torque already on; support arm and turn motor power off")
        initial = self.read()
        return initial[1]

    def activate(self, cancelled=None):
        check_cancelled(cancelled)
        initial = self.read()[1]
        EntryEnvelope(self.bounds, initial)
        for arm in self.arms():
            for name in JOINTS:
                if arm.bus.read("Torque_Enable", name, normalize=False) != 0:
                    raise RuntimeError(f"{arm.id}/{name}: torque changed before activation")
        # Old goals may be zero after power-up; only current positions define the start.
        for name in JOINTS:
            self.current_raw(name)
        self.health(active=False)
        check_cancelled(cancelled)
        # Goal writes themselves may enable torque on this hardware. Cleanup is armed now.
        self.motion_attempted = True
        if self.leader is not None:
            self.leader.configure()
        check_cancelled(cancelled)
        self.configure_follower_off(cancelled)
        self.health(active=False)
        self.prime_and_enable(initial, cancelled)
        check_cancelled(cancelled)
        current = self.read()[1]
        if difference(current, initial) > ACTIVATION_MOVEMENT_LIMIT:
            raise RuntimeError(f"Follower moved >{ACTIVATION_MOVEMENT_LIMIT:g} units during activation; inspect before retrying")
        # Preserve the original hold goal. Re-reading must not adopt drift as a new goal.
        held = self.project_target(initial)
        from hold_position import check_hold, running_speed_raw
        check_hold(held, current)
        for name, entry in self.follower.calibration.items():
            speed = running_speed_raw(name, entry, COMMAND_RATE, GRIPPER_COMMAND_RATE)
            self.follower.bus.write("Goal_Velocity", name, speed, normalize=False, num_retry=0)
            if self.follower.bus.read("Goal_Velocity", name, normalize=False, num_retry=2) != speed:
                raise RuntimeError(f"{name}: 실행 속도 제한 확인 실패")
        self.entry_envelope = EntryEnvelope(self.bounds, current)
        self.send(held)
        return held

    def release(self):
        self.follower.bus.disable_torque()
        for name in JOINTS:
            if self.follower.bus.read('Torque_Enable',name,normalize=False)!=0:
                raise RuntimeError('토크 해제를 확인하지 못했습니다: '+name)
        self.entry_envelope=None

    def current_raw(self, name):
        entry = self.follower.calibration[name]
        # Match other activation reads: retry communication failures only.
        # The SDK still raises motor alarms; raw-range and movement checks remain below.
        value = self.follower.bus.read("Present_Position", name, normalize=False,
                                       num_retry=ACTIVATION_READ_RETRIES)
        margin = (min(GRIPPER_ENTRY_MAX_RAW,
                      (entry.range_max - entry.range_min) * GRIPPER_ENTRY_MARGIN_UNITS / 100)
                  if name == "gripper" else ENTRY_MARGIN_DEG * 4095 / 360)
        if not (0 <= value <= 4095 and entry.range_min - margin <= value <= entry.range_max + margin):
            raise RuntimeError(f"{name}: current raw position outside calibration allowance "
                               f"(Present_Position={value}, allowed "
                               f"[{entry.range_min - margin:g}, {entry.range_max + margin:g}]); activation blocked")
        return value

    def configure_follower_off(self, cancelled):
        bus = self.follower.bus
        original = bus.torque_disabled
        entered = False

        @contextmanager
        def stay_off(motors=None):
            nonlocal entered
            check_cancelled(cancelled)
            bus.disable_torque(motors)
            entered = True
            # Unlike the library context, neither success nor failure re-enables torque.
            yield

        bus.torque_disabled = stay_off
        try:
            self.follower.configure()
            if not entered:
                raise RuntimeError("Unsupported configure path: torque guard was not used")
        finally:
            bus.torque_disabled = original

    def prime_and_enable(self, initial, cancelled):
        bus = self.follower.bus
        self.activation_report = []
        from hold_position import configure_preload, settle
        active_targets = {}
        for name in JOINTS:
            check_cancelled(cancelled)
            configure_preload(bus, name)
            entry = self.follower.calibration[name]
            before = self.current_raw(name)
            measured = bus._normalize({entry.id: before})[entry.id]
            if abs(measured - initial[name + ".pos"]) > ACTIVATION_MOVEMENT_LIMIT:
                raise RuntimeError(f"{name}: moved during activation; support arm and retry")
            for register, expected in (("Torque_Enable", 0), ("Status", 0), ("Operating_Mode", 0)):
                if bus.read(register, name, normalize=False, num_retry=ACTIVATION_READ_RETRIES) != expected:
                    raise RuntimeError(f"{name}: unexpected {register} before goal initialization")
            temperature = bus.read("Present_Temperature", name, normalize=False, num_retry=ACTIVATION_READ_RETRIES)
            if type(temperature) is not int or not 0 <= temperature < TEMPERATURE_LIMIT:
                message = f"{name}: temperature alarm before goal initialization ({temperature} raw)"
                self.temperature_sample = {"time": stamp(), "temperatures_raw": {name: temperature},
                    "complete": False, "error": message,
                    "last_check": {"joint": name, "register": "Present_Temperature", "value": temperature}}
                raise RuntimeError(message)
            target = max(entry.range_min, min(entry.range_max, before))
            record = {"motor": name, "current_raw": before, "initial_goal_raw": target,
                      "previous_goal_raw": bus.read("Goal_Position", name, normalize=False,
                                                    num_retry=ACTIVATION_READ_RETRIES)}
            self.activation_report.append(record)
            check_cancelled(cancelled)
            bus.write("Goal_Position", name, target, normalize=False, num_retry=0)
            if bus.read("Goal_Position", name, normalize=False, num_retry=ACTIVATION_READ_RETRIES) != target:
                raise RuntimeError(f"{name}: initial goal readback mismatch")
            torque = bus.read("Torque_Enable", name, normalize=False, num_retry=ACTIVATION_READ_RETRIES)
            record["torque_after_goal"] = torque
            if torque != 0 or bus.read("Status", name, normalize=False,
                                      num_retry=ACTIVATION_READ_RETRIES) != 0:
                raise RuntimeError(f"{name}: 목표 초기화가 토크를 켰습니다; 시작을 중단합니다.")
            after = self.current_raw(name)
            after_deg = bus._normalize({entry.id: after})[entry.id]
            if abs(after_deg - measured) > ACTIVATION_MOVEMENT_LIMIT:
                raise RuntimeError(f"{name}: moved >{ACTIVATION_MOVEMENT_LIMIT:g} units during goal initialization")
            check_cancelled(cancelled)
            bus.enable_torque(name)
            if (bus.read("Torque_Enable", name, normalize=False, num_retry=ACTIVATION_READ_RETRIES) != 1
                    or bus.read("Status", name, normalize=False, num_retry=ACTIVATION_READ_RETRIES) != 0
                    or bus.read("Lock", name, normalize=False, num_retry=ACTIVATION_READ_RETRIES) != 1):
                raise RuntimeError(f"{name}: activation verification failed")
            active_targets[name] = target
            record["settled_raw_ranges"] = settle(bus, active_targets, cancelled)
            record["verified"] = True
        self.health(active=True)

    def read(self):
        positions = {}
        for role in ('leader', 'follower'):
            arm = getattr(self, role)
            if arm is None:
                positions[role] = None
                continue
            raw = arm.bus.sync_read('Present_Position', normalize=False)
            normalized = arm.bus._normalize({arm.bus.motors[n].id: v for n, v in raw.items()})
            positions[role] = pose({n+'.pos': normalized[arm.bus.motors[n].id] for n in raw})
            self.raw_positions[role] = {n+'.pos': int(v) for n, v in raw.items()}
        leader, follower = positions['leader'], positions['follower']
        envelope = getattr(self, "entry_envelope", None)
        if envelope:
            envelope.observe(follower)
        return leader, follower

    def send(self, target):
        envelope = getattr(self, "entry_envelope", None)
        command = envelope.advance(target) if envelope else self.validate_target(target)
        return self.follower.send_action(self.project_target(command))

    def check_temperature_watchdog(self):
        guard = getattr(self, "temperature_guard", None)
        if guard is None:
            return
        try:
            guard.check_timeouts()
        except TemperatureFault as exc:
            sample = dict(getattr(self, "temperature_sample", None) or {"temperatures_raw": {}})
            sample.update(time=stamp(), complete=False, error=str(exc), temperature_fault=exc.detail,
                          temperature_pending=guard.pending(), temperature_transition=guard.take_transitions(),
                          last_check={"joint": exc.detail.get("joint"), "register": exc.detail.get("channel"),
                                      "value": exc.detail.get("raw")})
            self.temperature_sample = sample
            dashboard = getattr(self, "motor_dashboard", None)
            if dashboard is not None:
                dashboard.fault(str(exc), sample)
            raise

    def health(self, active=True):
        if not hasattr(self, "temperature_guard"):
            self.temperature_guard = TemperatureGuard()
        guard = self.temperature_guard
        if not hasattr(self, "health_history"):
            self.health_history = deque(maxlen=HEALTH_HISTORY_SIZE)
        sample = {"time": stamp(), "temperatures_raw": {}, "complete": False}
        self.temperature_sample = sample
        self.health_history.append(sample)
        try:
            dashboard = getattr(self, "motor_dashboard", None)
            if dashboard is not None:
                from motor_diagnostics import validate_live
                dashboard.temperature_guard = guard
                frame = dashboard.capture(active, getattr(self, "dashboard_context", {}))
                validate_live(frame)
            diagnostic = getattr(self, "temperature_diagnostic", None)
            if diagnostic is not None:
                diagnostic.temperature_guard = guard
                sample["wrist_diagnostic"] = diagnostic(active)
            gripper_diagnostic = getattr(self, "gripper_diagnostic", None)
            if gripper_diagnostic is not None:
                gripper_diagnostic.temperature_guard = guard
                sample["gripper_diagnostic"] = gripper_diagnostic(active)
            # Reuse the dashboard's validated block/single readings when available.
            # Otherwise use two packets per motor instead of three separate registers.
            from monitor_shoulder_temperature import read_pair, validate_responses
            validate_responses(self.follower.bus)
            for index, name in enumerate(JOINTS):
                sample["last_check"] = {"joint": name, "register": "health_packet", "value": None}
                if dashboard is not None:
                    row = frame["rows"][index]
                    values = {"Status": row["raw"].get("Status"),
                              "Torque_Enable": row["raw"].get("Torque_Enable"),
                              "Present_Temperature": row["raw"].get("Present_Temperature"),
                              "Temperature_Single": row.get("temperature_single")}
                else:
                    row = read_pair(self.follower.bus, 0, self.follower.bus.motors[name].id)
                    if row.get("read_error"):
                        raise RuntimeError(f"{name}: {row['read_error']}")
                    values = {"Status": row.get("status_raw"),
                              "Torque_Enable": row.get("torque_enable_raw"),
                              "Present_Temperature": row.get("temperature_block_raw"),
                              "Temperature_Single": row.get("temperature_single_raw")}
                sample["temperatures_raw"][name] = values['Present_Temperature']
                for register in ('Status', 'Torque_Enable'):
                    value = values[register]
                    sample["last_check"] = {"joint": name, "register": register, "value": value}
                    if value != (0 if register == 'Status' else int(active)):
                        raise RuntimeError(f"{name}: {register}={value}; stop and inspect motor power")
                # A real collector already checked the same block/single pair.
                if not (dashboard is not None and row.get('temperature_checked')):
                    check_readings(sample, guard, name,
                                   {'block': values['Present_Temperature'], 'single': values['Temperature_Single']},
                                   active=active, source='health')
        except Exception as exc:
            sample["error"] = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, TemperatureFault):
                sample['temperature_fault'] = exc.detail
                sample['last_check'] = {'joint': exc.detail.get('joint'), 'register': exc.detail.get('channel'),
                                        'value': exc.detail.get('raw')}
            if getattr(self, "motor_dashboard", None) is not None:
                self.motor_dashboard.fault(sample["error"], sample.copy())
            raise
        finally:
            sample['temperature_pending'] = guard.pending()
            sample['temperature_transition'] = guard.take_transitions()
        sample["complete"] = True
        return sample

    def close(self):
        errors = []
        for arm in reversed(self.arms()):
            if not arm.bus.is_connected:
                continue
            if self.motion_attempted:
                # One failed motor must not prevent OFF attempts on the other motors.
                for name in JOINTS:
                    try:
                        arm.bus.write("Torque_Enable", name, 0, normalize=False, num_retry=1)
                    except Exception as exc:
                        errors.append(f"{arm.id}/{name} OFF write: {exc}")
                    try:
                        if arm.bus.read("Torque_Enable", name, normalize=False, num_retry=1) != 0:
                            raise RuntimeError("torque OFF not confirmed")
                    except Exception as exc:
                        errors.append(f"{arm.id}/{name} OFF verification: {exc}")
        try:
            capture = getattr(self, "fault_capture", None)
            if capture is not None and self.follower.bus.is_connected:
                capture(errors.copy())
        finally:
            self.fault_capture = None
            for arm in reversed(self.arms()):
                if not arm.bus.is_connected:
                    continue
                try:
                    arm.bus.port_handler.closePort()
                except Exception as exc:
                    errors.append(f"{arm.id} serial close: {exc}")
        return errors


class Demo:
    simulated_encoders = True
    follower_only = False
    identity = {"side": "left", "serials": SERIALS, "demo": True, "calibration_sha256": {}}

    def __init__(self):
        self.target = dict.fromkeys(KEYS, 0.0)
        self.current = self.target.copy()
        self.lead = self.target.copy()
        self.last_read = time.monotonic()

    def preflight(self):
        return ["DEMO: no hardware"]

    def connect(self, cancelled=None):
        return self.current.copy()

    def activate(self, cancelled=None):
        check_cancelled(cancelled)
        return self.current.copy()

    def release(self):
        self.target=self.current.copy()

    def read(self):
        now = time.monotonic()
        self.current = approach(self.current, self.target, now - self.last_read, rate=DEMO_RATE)
        self.last_read = now
        return (None if self.follower_only else self.lead.copy()), self.current.copy()

    def send(self, target):
        self.target = pose(target)
        return self.target.copy()

    validate_target = staticmethod(pose)
    project_target = staticmethod(pose)

    def health(self, active=True):
        self.temperature_sample = {"time": stamp(), "temperatures_raw": dict.fromkeys(JOINTS, 25),
                                   "complete": True}
        return self.temperature_sample

    def close(self):
        return []


class EventMailbox:
    """Coalesce consecutive display samples without dropping ordered user/fault events."""
    def __init__(self):
        self.items = deque()
        self.lock = threading.Lock()

    def put(self, event):
        with self.lock:
            if event[0] == "status" and self.items and self.items[-1][0] == "status":
                self.items[-1] = event
            else:
                self.items.append(event)

    def get_nowait(self):
        with self.lock:
            if not self.items:
                raise queue.Empty
            return self.items.popleft()


class Session:
    def __init__(self, backend, output, *, auto_sync=False):
        self.backend, self.output = backend, Path(output)
        self.auto_sync = auto_sync and not getattr(backend,"follower_only",False)
        self.commands, self.events = queue.Queue(maxsize=COMMAND_QUEUE_SIZE), EventMailbox()
        self.stop = threading.Event()
        self.wakeup = threading.Event()
        self.heartbeat = time.monotonic()
        self.state = "DISCONNECTED"
        self.thread = None

    def emit(self, kind, value):
        self.events.put((kind, value))

    def start(self):
        if self.thread and self.thread.is_alive():
            raise RuntimeError("Already connected")
        self.stop.clear()
        self.heartbeat = time.monotonic()
        self.thread = threading.Thread(target=self.run, name="left-teaching", daemon=False)
        self.thread.start()

    def command(self, name, value=None):
        self.commands.put_nowait((name, value))
        self.wakeup.set()

    def log_temperatures(self, report, sample, now, *, force=False):
        if not isinstance(sample, dict):
            return
        if (not force and not sample.get('temperature_transition') and not sample.get('temperature_pending')
                and now < getattr(self, "temperature_log_at", 0)):
            return
        record = {**sample, "temperatures_raw": dict(sample["temperatures_raw"]),
                  "state": self.state, "fault": force, "demo": self.backend.identity["demo"]}
        report["temperature_samples"].append(record)
        self.temperature_log_at = now + TEMPERATURE_LOG_INTERVAL
        values = " | ".join(f"ID{index} {name}={record['temperatures_raw'].get(name, '--')}"
                            for index, name in enumerate(JOINTS, 1))
        label = "TEMPERATURE FAULT" if force else "TEMPERATURE"
        if record["demo"]:
            label += " DEMO"
        completeness = "" if record["complete"] else " (partial; unread=--)"
        pending = pending_label(record.get('temperature_pending', []))
        recovered = any(t.get('event') == 'recovered' for t in record.get('temperature_transition', []))
        suffix = (' | '+pending) if pending else ' | 온도 정상 복귀' if recovered else ''
        self.emit("notice", f"{record['time']} {label} follower raw{completeness} | {values}{suffix}")

    def finish_jog(self, report, request, goal, current, arrived, now):
        from manual_teaching import jog_outcome
        result = jog_outcome(request, goal, current, arrived, now)
        report['events'].append({'at': stamp(), 'jog_result': result})
        self.emit('jog_result', result)
        return result['reference']

    def run(self):
        self.temperature_log_at = 0
        health_interval = GRIPPER_HEALTH_INTERVAL if getattr(self.backend, "gripper_diagnostic", None) is not None else HEALTH_INTERVAL
        report = {"started_at": stamp(), "identity": self.backend.identity,
                  "outcome": "starting", "cleanup_errors": [], "events": [],
                  "health_interval_s": health_interval,
                  "temperature_log_interval_s": TEMPERATURE_LOG_INTERVAL, "temperature_samples": [],
                  "temperature_policy": {"limit_raw": TEMPERATURE_LIMIT, "duration_s": TEMPERATURE_DURATION_S,
                                         "max_gap_s": TEMPERATURE_MAX_GAP_S},
                  "fps_target": CONTROL_HZ, "command_rate": COMMAND_RATE,
                  "gripper_command_rate": GRIPPER_COMMAND_RATE,
                  "relative_target_limit": RELATIVE_TARGET_LIMIT,
                  "sync_rate": SYNC_RATE, "sync_timeout_s": SYNC_TIMEOUT,
                  "entry_margin_deg": ENTRY_MARGIN_DEG,
                  "auto_sync_on_connect": self.auto_sync,
                  "gripper_entry_max_raw": GRIPPER_ENTRY_MAX_RAW,
                  "gripper_entry_margin_units": GRIPPER_ENTRY_MARGIN_UNITS,
                  "activation_read_retries": ACTIVATION_READ_RETRIES,
                  "replay_arrival_tolerances": REPLAY_ARRIVAL_TOLERANCES,
                  "play_gripper_arrival_tolerance": PLAY_GRIPPER_ARRIVAL_TOLERANCE,
                  "sync_arrival_tolerance": SYNC_ARRIVAL_TOLERANCE,
                  "tracking_guard": {"no_progress_s": STALL_SECONDS, "progress_ticks": TRACKING_PROGRESS_TICKS,
                                     "requested_error_units": TRACKING_ERROR, "previous_sent_error_units": TRACKING_SENT_ERROR}}
        path = self.output / ("session_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
        csv_path = path.with_suffix(".csv")
        playback_path = path.with_name(path.stem + "_playback.jsonl")
        telemetry = None
        playback_log = None
        last_playback_sample = None
        report["telemetry_csv"] = str(csv_path)
        report["playback_diagnostics_jsonl"] = str(playback_path)
        try:
            from teaching_settings import load_settings
            report["settings"] = load_settings()
            atomic_json(path, report)
            telemetry = csv_path.open("w", newline="", encoding="utf-8")
            playback_log = playback_path.open("w", encoding="utf-8")
            writer = csv.DictWriter(telemetry, fieldnames=("time", "elapsed_s", "state", "joint", "unit",
                "leader", "follower", "planned_target", "previous_sent_target", "planned_error",
                "previous_sent_error", "directional_progress", "no_progress_s", "blocked", "loop_dt_s"))
            writer.writeheader()
            telemetry.flush()
            target = self.backend.connect(self.stop)
            self.state = "READY"
            previous = target.copy()
            final = target.copy()
            sync_reference = target.copy()
            last = time.monotonic()
            health_at = last
            log_start, log_at = last, last
            flush_at = last + LOG_FLUSH_INTERVAL
            loop_count, loop_total, loop_max, overruns = 0, 0.0, 0.0, 0
            ticks = getattr(self.backend, "position_ticks", {})
            guards = {key: TrackingProgress(ticks.get(key, 360 / 4095)) for key in KEYS}
            guard_mode = None
            last_sent = None
            reached_at = None
            history = deque()
            move_start = last
            sequence = []
            sequence_names = []
            sequence_meta = []
            segment_start = previous.copy()
            sequence_index = 0
            playback_log_at = last
            motor_log_at = last
            auto_sync_pending = self.auto_sync and not getattr(self.backend,"follower_only",False)
            jog_active = False
            jog_request = None
            jog_reference = None
            active_play_id = None
            while not self.stop.is_set():
                self.wakeup.clear()
                finish_requested = False
                now = time.monotonic()
                if now - self.heartbeat > GUI_HEARTBEAT_TIMEOUT:
                    raise RuntimeError("GUI heartbeat lost; stopping")
                dt, last = now - last, now
                loop_count += 1
                loop_total += dt
                loop_max = max(loop_max, dt)
                overruns += int(dt > 1.5 / CONTROL_HZ)
                leader, follower = self.backend.read()
                observed_at = time.monotonic()
                if self.state == "HOLD":
                    from hold_position import check_hold
                    check_hold(target, follower)
                if hasattr(self.backend, 'check_temperature_watchdog'):
                    self.backend.check_temperature_watchdog()
                history.append((now, follower.copy()))
                while history and now - history[0][0] > CAPTURE_HISTORY_SECONDS:
                    history.popleft()
                if auto_sync_pending:
                    auto_sync_pending = False
                    name, value = ("sync", {"leader": leader.copy(), "follower": follower.copy()})
                    report["events"].append({"at": stamp(), "command": "auto_sync_on_connect"})
                else:
                    try:
                        name, value = self.commands.get_nowait()
                    except queue.Empty:
                        name = None
                if name not in (None, 'jog', 'capture', 'measure_target'):
                    jog_reference = None
                if getattr(self.backend,"follower_only",False) and name in ("sync","teach","quick_capture"):
                    self.emit("notice", "팔로워 단독 모드는 리더 동작을 사용하지 않습니다.")
                    name = None
                if name == "hold_current" and self.state == "READY" and getattr(self.backend,"follower_only",False):
                    check_cancelled(self.stop)
                    self.state = "ACTIVATING"
                    self.emit("status", {"state":self.state,"leader":None,"follower":follower,"sample_time":observed_at})
                    follower = self.backend.activate(self.stop)
                    previous = target = final = follower.copy()
                    history.clear(); reached_at = None
                    now = last = time.monotonic(); dt = 0
                    self.state = "HOLD"
                    report["events"].append({"at":stamp(),"command":"activate_at_current_pose","follower":follower.copy()})
                    self.emit("notice", "팔로워 현재 자세 유지 중 · 스텝 추가 후 GUI에서 관절값을 편집하세요.")
                elif name == 'release' and self.state=='HOLD' and getattr(self.backend,'follower_only',False):
                    self.backend.release()
                    previous=target=final=follower.copy();history.clear();sequence=[];jog_active=False;last_sent=None
                    self.state='READY'
                    self.emit('notice','토크 꺼짐 · 팔을 받치고 손으로 자세를 맞춘 뒤 스텝을 저장하세요.')
                elif name == "sync" and self.state in ("READY", "HOLD"):
                    try:
                        final = sync_goal(self.backend, value, leader, follower)
                    except (ValueError, KeyError, TypeError) as exc:
                        self.emit("notice", str(exc))
                    else:
                        check_cancelled(self.stop)
                        if self.state == "READY":
                            self.state = "ACTIVATING"
                            self.emit("status", {"state": self.state, "leader": leader,
                                                 "follower": follower, "sample_time": observed_at})
                            report["events"].append({"at": stamp(), "command": "activate_for_sync",
                                                      "approved_goal": final.copy()})
                            follower = self.backend.activate(self.stop)
                        sync_reference = value["leader"].copy()
                        previous = target = follower.copy()
                        history.clear()
                        reached_at = None
                        move_start = now = last = time.monotonic()
                        dt = 0
                        self.state = "SYNCING"
                        report["events"].append({"at": stamp(), "sync_start": follower,
                                                  "sync_goal": final.copy(), "leader_reference": sync_reference.copy()})
                        if difference(sync_reference, final) > .001:
                            self.emit("notice", "Sync target limited to follower calibration range.")
                        self.emit("notice", "Synchronizing to approved leader pose. Keep the leader still.")
                elif name == "teach" and self.state == "HOLD":
                    if difference(leader, follower) > TEACH_START_GAP:
                        self.emit("notice", f"Leader/follower gap >{TEACH_START_GAP:g}. Use Sync to leader before teaching.")
                    else:
                        self.state = "TEACH"
                        previous = follower.copy()
                elif name == "finish" and self.state in ("TEACH", "HOLD"):
                    target = previous = follower.copy()
                    sequence = []
                    self.state = "HOLD"
                    finish_requested = True
                elif name == "pause" and self.state in ("TEACH", "MOVE", "SYNCING", "PLAY"):
                    if active_play_id is not None:
                        self.emit("play_result", {"token":active_play_id,"ok":False,"error":"사용자 중지"})
                        active_play_id = None
                    target = previous = follower.copy()
                    sequence = []
                    self.state = "HOLD"
                elif name == "quick_capture" and self.state == "TEACH":
                    # Record the actual follower pose from this control cycle; keep tracking.
                    try:
                        captured = self.backend.validate_target(follower)
                    except ValueError as exc:
                        self.emit("notice", f"Cannot save this pose: {exc}")
                    else:
                        self.emit("captured", {"name": value.get("name") if isinstance(value,dict) else value, "pose": captured, "captured_at": stamp(), **({"_teaching_context":value["teaching_context"]} if isinstance(value,dict) else {})})
                elif name == "capture" and (self.state == "HOLD" or (self.state == "READY" and getattr(self.backend,"follower_only",False))):
                    stable = (len(history) >= 2 and now - history[0][0] >= CAPTURE_STABLE_SECONDS
                              and all(difference(p, follower) <= CAPTURE_POSITION_TOLERANCE for _, p in history)
                              and (self.state=="READY" or difference(target, follower) <= CAPTURE_TARGET_TOLERANCE))
                    if stable:
                        try:
                            captured = self.backend.validate_target(follower)
                        except ValueError as exc:
                            self.emit("notice", f"Finish synchronization before saving: {exc}")
                        else:
                            self.emit("captured", {"name": value.get("name") if isinstance(value,dict) else value, "pose": captured, "captured_at": stamp(), **({"_teaching_context":value["teaching_context"]} if isinstance(value,dict) else {})})
                    else:
                        self.emit("notice", "Wait for the follower to settle before saving.")
                elif name == "jog":
                    from manual_teaching import checked_jog
                    response = {"token": value.get("token"), "kind": "jog"}
                    try:
                        if self.state != "HOLD":raise ValueError("팔이 멈춘 뒤 조정하세요.")
                        goal = checked_jog(value, follower, time.monotonic(), self.backend.validate_target)
                        if value.get('axis') in KEYS:
                            from manual_teaching import jog_goal
                            reference = jog_reference
                            held = self.backend.project_target(target)
                            if value.get('from_measured'):
                                held[value['axis']] = value['start'][value['axis']]
                                reference = None
                            if value.get('unit') == 'ticks':
                                from motor_positions import tick_goal
                                expected = tick_goal(self.backend, value['start'], value['axis'], value['increment'], reference,
                                                     hold_target=held)
                            else:
                                expected = jog_goal(value['start'], None, value['axis'], value['increment'], reference,
                                                    hold_target=held)
                            if difference(expected, goal) > 1e-6:
                                raise ValueError('이전 조정 기준이 바뀌었습니다. 다시 눌러 주세요.')
                        check_cancelled(self.stop)
                    except (ValueError, KeyError, TypeError) as exc:
                        self.emit("manual_result", {**response, "ok":False, "error":str(exc)})
                    else:
                        final = goal; previous = follower.copy(); jog_active = True
                        jog_request = {**value, "start": follower.copy()}
                        self.state = "MOVE"; move_start = now; reached_at = None
                        self.emit("manual_result", {**response, "ok":True})
                elif name == "measure_target":
                    from manual_teaching import checked_jog
                    response = {"token": value.get("token"), "kind": "measure"}
                    try:
                        if self.state != "HOLD":raise ValueError("팔이 멈춘 뒤 저장하세요.")
                        checked_jog({**value,"goal":follower},follower,time.monotonic(),self.backend.validate_target)
                        stable = (len(history)>=2 and now-history[0][0]>=CAPTURE_STABLE_SECONDS
                                  and all(difference(p,follower)<=.35 for _,p in history)
                                  and difference(target,follower)<=.5)
                        if not stable:raise ValueError("팔이 안정될 때까지 잠시 기다린 뒤 저장하세요.")
                        captured = self.backend.validate_target(follower)
                    except (ValueError, KeyError, TypeError) as exc:
                        self.emit("manual_result", {**response,"ok":False,"error":str(exc)})
                    else:
                        self.emit("manual_result", {**response,"ok":True,"pose":captured,
                                                     "captured_at":stamp(),"sample_time":observed_at})
                elif name == "move" and self.state == "HOLD":
                    jog_active = False
                    final = self.backend.validate_target(value)
                    if abs(final["wrist_roll.pos"] - follower["wrist_roll.pos"]) > WRIST_MOVE_LIMIT:
                        self.emit("notice", f"Wrist move >{WRIST_MOVE_LIMIT:g} degrees blocked. Teach intermediate steps.")
                    else:
                        self.state = "MOVE"
                        previous = follower.copy()
                        move_start = now
                        reached_at = None
                elif name == "play" and self.state == "HOLD":
                    try:
                        if difference(pose(value["follower"]), follower) > PREVIEW_POSE_TOLERANCE:
                            raise ValueError("Position changed since playback preview; approve again")
                        if "vision_checked_at" in value and not 0 <= now-value["vision_checked_at"] <= .8:
                            raise ValueError("지그 실행 확인이 만료됐습니다. 다시 실행하세요.")
                        if any(s.get("motion") for s in value["steps"]) and "vision_checked_at" not in value:
                            raise ValueError("지그 목표를 최신 영상으로 확인한 뒤 실행하세요.")
                        sequence = validate_sequence(self.backend, value["steps"], follower)
                        from vision_teaching import expand_steps
                        sequence_meta = expand_steps(value["steps"],follower)
                    except (ValueError, TypeError, KeyError) as exc:
                        sequence = []
                        if value.get("request_id"):
                            self.emit("play_result", {"token":value["request_id"],"ok":False,"error":str(exc)})
                        self.emit("notice", str(exc))
                    else:
                        active_play_id = value.get("request_id")
                        sequence_index = 0
                        sequence_names = [step.get("name", f"step{i + 1}") for i, step in enumerate(sequence_meta)]
                        final = sequence[0].copy()
                        previous = follower.copy()
                        segment_start = previous.copy()
                        move_start, reached_at = now, None
                        self.state = "PLAY"
                        playback_log_at = motor_log_at = now
                        report["events"].append({"at": stamp(), "play_title": value.get("title", ""),
                                                  "play_targets": sequence, "play_names": sequence_names,
                                                  "vision": value.get("vision_evidence")})
                        settings = {}
                        if not self.backend.identity["demo"]:
                            for register in ("Max_Torque_Limit", "Torque_Limit", "Overload_Torque",
                                             "Protection_Current", "Protection_Time"):
                                try:
                                    settings[register] = self.backend.follower.bus.read(
                                        register, "gripper", normalize=False, num_retry=0)
                                except Exception as exc:
                                    settings[register] = {"error": f"{type(exc).__name__}: {exc}"}
                        playback_log.write(json.dumps({"event": "start", "time": stamp(),
                            "title": value.get("title", ""), "step_count": len(sequence),
                            "gripper_settings_raw": settings}) + "\n")
                        playback_log.flush()
                        self.emit("notice", f"Playing {sequence_names[0]} (1 / {len(sequence)})")
                if name:
                    report["events"].append({"at": stamp(), "command": name, "state": self.state})
                if self.state == "SYNCING" and difference(leader, sync_reference) > SYNC_LEADER_DRIFT:
                    self.state = "HOLD"
                    target = previous = follower.copy()
                    self.emit("notice", "Leader moved during sync. Holding current follower pose; approve sync again.")
                if self.state in ("TEACH", "MOVE", "SYNCING", "PLAY"):
                    requested = (self.backend.project_target(leader) if self.state == "TEACH"
                                 else self.backend.validate_target(final))
                    from manual_teaching import JOG_RATE
                    jogging = self.state == "MOVE" and jog_active
                    target = approach(previous, requested, dt,
                                      rate=min(JOG_RATE, COMMAND_RATE) if jogging else SYNC_RATE if self.state == "SYNCING" else COMMAND_RATE,
                                      gripper_rate=min(JOG_RATE, GRIPPER_COMMAND_RATE) if jogging else None if self.state == "SYNCING" else GRIPPER_COMMAND_RATE)
                    if self.state == "PLAY" and sequence_meta[sequence_index].get("cartesian"):
                        from vision_teaching import synchronized_approach
                        target = synchronized_approach(previous, requested, segment_start, dt,
                                                      sequence_meta[sequence_index].get("duration_s", 0.))
                    previous = target.copy()
                from manual_teaching import JOG_TIMEOUT_SECONDS
                if self.state == 'MOVE' and jog_active and now-move_start >= JOG_TIMEOUT_SECONDS:
                    jog_reference = self.finish_jog(report, jog_request, final, follower, False, now)
                    # Stop pursuing the unmet goal and hold the freshly measured pose.
                    # Never turn a manual adjustment deadline into a power-off timeout.
                    from manual_teaching import jog_hold_pose
                    target = previous = final = jog_hold_pose(jog_request, final, follower, self.backend.project_target)
                    self.state = 'HOLD'; jog_active = False
                    history.clear(); reached_at = None
                if self.state in ("MOVE", "SYNCING", "PLAY"):
                    timeout = SYNC_TIMEOUT if self.state == "SYNCING" else MOVE_TIMEOUT
                    if now - move_start > timeout:
                        if self.state == "PLAY":
                            playback_log.write(json.dumps({"event": "timeout", **playback_snapshot(
                                sequence_names[sequence_index], sequence_index, final, follower,
                                last_sent, move_start, now)}) + "\n")
                            playback_log.flush()
                        raise RuntimeError(f"{self.state} timeout ({timeout} s)")
                    arrived = within_arrival_tolerance(final, follower, self.state)
                    if self.state == "MOVE" and jog_active:
                        from manual_teaching import jog_arrived
                        arrived = jog_arrived(jog_request, final, follower)
                    if self.state == "PLAY" and sequence_meta[sequence_index].get("precision"):
                        arrived = all(abs(final[k]-follower[k]) <= (.35 if k != "gripper.pos" else PLAY_GRIPPER_ARRIVAL_TOLERANCE) for k in KEYS)
                    if self.state == "PLAY" and sequence_meta[sequence_index].get("safe_boundary"):
                        arrived = within_arrival_tolerance(final, follower, "MOVE")
                    if arrived and difference(target, final) < .01:
                        reached_at = reached_at or now
                        settle = (sequence_meta[sequence_index].get("hold_s", ARRIVAL_SETTLE_SECONDS)
                                  if self.state == "PLAY" else ARRIVAL_SETTLE_SECONDS)
                        if now - reached_at >= settle:
                            report["events"].append({"at": stamp(), "reached_state": self.state,
                                                      "step": sequence_index + 1 if self.state == "PLAY" else None,
                                                      "step_name": sequence_names[sequence_index] if self.state == "PLAY" else None})
                            if self.state == "PLAY" and sequence_index + 1 < len(sequence):
                                playback_log.write(json.dumps({"event": "reached", **playback_snapshot(
                                    sequence_names[sequence_index], sequence_index, final, follower,
                                    last_sent, move_start, now)}) + "\n")
                                playback_log.flush()
                                sequence_index += 1
                                final = sequence[sequence_index].copy()
                                segment_start = previous.copy()
                                move_start, reached_at = now, None
                                playback_log_at = motor_log_at = now
                                for guard in guards.values():
                                    guard.reset()
                                self.emit("notice", f"Playing {sequence_names[sequence_index]} "
                                          f"({sequence_index + 1} / {len(sequence)})")
                            else:
                                if self.state == "PLAY":
                                    playback_log.write(json.dumps({"event": "reached", **playback_snapshot(
                                        sequence_names[sequence_index], sequence_index, final, follower,
                                        last_sent, move_start, now)}) + "\n")
                                    playback_log.flush()
                                message = ("Sync complete; ready for Start teaching" if self.state == "SYNCING"
                                           else "Playback complete" if self.state == "PLAY" else "Step reached")
                                if self.state == "PLAY" and active_play_id is not None:
                                    message = "시작 안전 자세 도착 · 지그 위치 읽기 준비"
                                    self.emit("play_result", {"token":active_play_id,"ok":True,"pose":follower.copy(),"sample_time":observed_at})
                                    active_play_id = None
                                if self.state == 'MOVE' and jog_active:
                                    jog_reference = self.finish_jog(report, jog_request, final, follower, True, now)
                                    jog_active = False
                                    message = None
                                self.state = "HOLD"
                                target = final.copy()
                                sequence = []
                                if message:self.emit("reached", message)
                    else:
                        reached_at = None
                if self.state == "PLAY" and now >= playback_log_at:
                    sample = playback_snapshot(sequence_names[sequence_index], sequence_index,
                                               final, follower, last_sent, move_start, now)
                    sample["event"] = "waiting"
                    sample["command_target"] = target.copy()
                    sample.update(playback_temperatures(getattr(self.backend, "temperature_sample", None)))
                    last_playback_sample = sample
                    if now - move_start >= MOTOR_LOG_INTERVAL and now >= motor_log_at and not self.backend.identity["demo"]:
                        from diagnose_shoulder_lift import read_telemetry
                        sample["motor_telemetry"] = {
                            key: read_telemetry(self.backend.follower.bus, motor_name=key.removesuffix(".pos"))
                            for key in sample["blocking_joints"]}
                        motor_log_at = now + MOTOR_LOG_INTERVAL
                    playback_log.write(json.dumps(sample) + "\n")
                    playback_log.flush()
                    playback_log_at = now + PLAYBACK_LOG_INTERVAL
                if self.state != guard_mode:
                    for guard in guards.values():
                        guard.reset()
                    guard_mode = self.state
                tracking = {key: guards[key].update(now, follower[key], target[key],
                            last_sent[key] if last_sent is not None and self.state != "READY" else None)
                            for key in KEYS}
                if getattr(self.backend, "motor_dashboard", None) is not None:
                    self.backend.dashboard_context = {"state": self.state, "tracking": tracking,
                                                      "loop_dt_s": dt, "leader": leader, "follower": follower}
                blocked = [key for key, value in tracking.items() if value["blocked"]]
                if now >= log_at or blocked:
                    for key, value in tracking.items():
                        writer.writerow({"time": stamp(), "elapsed_s": now - log_start,
                            "state": self.state, "joint": key, "unit": UNITS[key],
                            "leader": leader[key] if leader is not None else "", "follower": follower[key], "planned_target": target[key],
                            "previous_sent_target": last_sent[key] if last_sent is not None else "",
                            "planned_error": value["error"], "previous_sent_error": value["sent_error"],
                            "directional_progress": value["progress"], "no_progress_s": value["no_progress_s"],
                            "blocked": value["blocked"], "loop_dt_s": dt})
                    if blocked or now >= flush_at:
                        telemetry.flush()
                        flush_at = now + LOG_FLUSH_INTERVAL
                    log_at = now + TELEMETRY_LOG_INTERVAL
                if blocked:
                    report["tracking_stop"] = {key: tracking[key] for key in blocked}
                    raise RuntimeError(f"{', '.join(blocked)}: no measurable forward progress for "
                                       f"{STALL_SECONDS:g}s; no automatic retry")
                if self.stop.is_set():
                    break
                if hasattr(self.backend, 'check_temperature_watchdog'):
                    self.backend.check_temperature_watchdog()
                if self.state != "READY":
                    last_sent = pose(self.backend.send(target))
                if now >= health_at or finish_requested:
                    temperature_sample = self.backend.health(active=self.state != "READY")
                    self.log_temperatures(report, temperature_sample, time.monotonic())
                    if temperature_sample and temperature_sample.get('temperature_transition'):
                        playback_log.write(json.dumps({'event': 'temperature_transition',
                            'time': temperature_sample['time'], 'state': self.state,
                            'transitions': temperature_sample['temperature_transition']}, ensure_ascii=False)+'\n')
                        playback_log.flush()
                    health_at = now + health_interval
                limited_target = self.backend.project_target(leader) if self.state == "TEACH" else leader
                limited = [key for key in KEYS if leader is not None and abs(leader[key] - limited_target[key]) > .001]
                self.emit("status", {"state": self.state, "leader": leader, "follower": follower,
                                     "limited_joints": limited,
                                     "lagging_joints": [key for key, value in tracking.items() if value["lagging"]],
                                     "sample_time": observed_at,
                                     "motor_raw": {r: dict(p) for r, p in getattr(self.backend, "raw_positions", {}).items()},
                                     "hold_target": self.backend.project_target(target) if self.state == "HOLD" else None,
                                     "jog_reference": jog_reference if self.state == "HOLD" else None,
                                     "temperature_pending": (getattr(self.backend, 'temperature_sample', None) or {}).get('temperature_pending', [])})
                if finish_requested and not self.stop.is_set():
                    self.emit("finish_ready", None)
                self.wakeup.wait(max(0, 1 / CONTROL_HZ - (time.monotonic() - now)))
            report["outcome"] = "stopped"
        except Exception as exc:
            report["outcome"] = f"{type(exc).__name__}: {exc}"
            temperature_sample = getattr(self.backend, "temperature_sample", None)
            if isinstance(temperature_sample, dict) and temperature_sample.get("error"):
                self.log_temperatures(report, temperature_sample, time.monotonic(), force=True)
                if isinstance(self.backend, Hardware):
                    def capture_fault(cleanup_errors):
                        from temperature_fault_capture import capture
                        fault_path = path.with_name(path.stem + "_fault.jsonl")
                        report["fault_diagnostics_jsonl"] = str(fault_path)
                        try:
                            with fault_path.open("x", encoding="utf-8") as stream:
                                def emit_record(record):
                                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                                    stream.flush()
                                emit_record({"event": "fault_context", "time": stamp(),
                                    "outcome": report["outcome"], "state": self.state,
                                    "trigger": temperature_sample,
                                    "last_playback_sample": last_playback_sample,
                                    "health_history": list(getattr(self.backend, "health_history", [])),
                                    "cleanup_errors_before_probe": cleanup_errors,
                                    "identity": self.backend.identity})
                                self.emit("notice", "Torque OFF attempted; collecting post-stop motor evidence (about 5s).")
                                check = temperature_sample.get("last_check", {})
                                first = check.get("joint")
                                joints = ([first] if first in JOINTS else []) + [n for n in JOINTS if n != first]
                                report["fault_assessment"] = capture(self.backend.follower.bus, joints, emit_record)
                                emit_record({"event": "fault_assessment", "motors": report["fault_assessment"]})
                                if playback_log is not None:
                                    playback_log.write(json.dumps({"event": "fault_assessment", "time": stamp(),
                                        "fault_diagnostics_jsonl": str(fault_path),
                                        "motors": report["fault_assessment"]}) + "\n")
                                    playback_log.flush()
                        except Exception as capture_error:
                            report["fault_capture_error"] = f"{type(capture_error).__name__}: {capture_error}"
                            self.emit("error", "Fault capture failed: " + report["fault_capture_error"])
                        self.emit("log", str(fault_path))
                    self.backend.fault_capture = capture_fault
            self.emit("error", report["outcome"])
        finally:
            errors = self.backend.close()
            if telemetry is not None:
                try:
                    telemetry.close()
                    self.emit("log", str(csv_path))
                except Exception as exc:
                    report["telemetry_close_error"] = str(exc)
                    self.emit("error", f"Telemetry close failed: {exc}")
            if playback_log is not None:
                try:
                    playback_log.close()
                    self.emit("log", str(playback_path))
                except Exception as exc:
                    report["playback_log_close_error"] = str(exc)
            report["loop_metrics"] = {"cycles": locals().get("loop_count", 0),
                "mean_hz": loop_count / loop_total if locals().get("loop_total", 0) > 0 else 0,
                "max_interval_ms": locals().get("loop_max", 0) * 1000,
                "overrun_cycles": locals().get("overruns", 0)}
            report["cleanup_errors"] = errors
            report["activation"] = getattr(self.backend, "activation_report", [])
            report["finished_at"] = stamp()
            dashboard = getattr(self.backend, "motor_dashboard", None)
            if dashboard is not None:
                report["motor_dashboard_log"] = str(dashboard.path)
                try:
                    dashboard.finish(report["outcome"], errors,
                        torque_off_verified=bool(getattr(self.backend, "motion_attempted", False) and not errors))
                except Exception as exc:
                    report["dashboard_log_error"] = str(exc)
                    self.emit("error", f"Dashboard log save failed: {exc}")
            self.state = "DISCONNECTED"
            try:
                atomic_json(path, report)
                self.emit("log", str(path))
            except Exception as exc:
                self.emit("error", f"Log save failed: {exc}")
            self.emit("closed", errors)


class StepPoseEditor:
    """Edit a saved pose without sending motion commands."""

    def __init__(self, app, step):
        import tkinter as tk
        from tkinter import ttk
        self.app, self.step = app, step
        from app_navigation import AppPage
        self.window = AppPage(app,section="episode",dock_preview=True)
        self.window.title("스텝 편집 · " + step["name"])
        self.window.columnconfigure(0,weight=1)
        self.window.transient(app.root)
        self.window.resizable(False, False)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.bind("<Escape>", self.stop_and_close)
        self.window.bind("<Return>", self.apply)
        configure_ui(self.window)
        body = ttk.Frame(self.window, padding=8)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(1,weight=1)
        ttk.Label(body, text="슬라이더·숫자를 바꾸면 오른쪽 시뮬에 표시합니다.\n수정값 적용은 저장만 합니다. 실제 이동은 스텝 실행을 누르세요.",
                  wraplength=620).grid(row=1, column=0, columnspan=4, sticky="w", pady=(6, 12))
        for column, label in enumerate(("관절", "목표값", "단위", "허용 범위")):
            ttk.Label(body, text=label).grid(row=2, column=column, sticky="w", padx=(0, 16))
        self.values, self.fields, self.original_text, self.sliders = {}, {}, {}, {}
        self.motor_values={};self.motor_adapters={};self.adjust_buttons={}
        from step_reference import editable_servos
        self.editable_keys=editable_servos(step)
        for row, (joint, key) in enumerate(zip(JOINTS, KEYS), start=3):
            low, high = (0, 100) if key == "gripper.pos" else (-180, 180)
            lower, upper = getattr(app.backend, "bounds", {}).get(key, (low, high))
            low, high = max(low, lower), min(high, upper)
            from motor_positions import to_ticks
            if key!='gripper.pos':
                from motor_positions import TICK_DEGREES, midpoint
                center=midpoint(app.backend,key)
                low=max(0,math.ceil(low/TICK_DEGREES+center-1e-8));high=min(4095,math.floor(high/TICK_DEGREES+center+1e-8))
            text = (f"{step['pose'][key]:.4f}".rstrip("0").rstrip(".") if key == "gripper.pos"
                    else repr(step["pose"][key]))
            self.original_text[key] = text
            self.values[key] = tk.StringVar(master=self.window,value=text)
            if key=='gripper.pos':self.motor_values[key]=self.values[key]
            else:
                from motor_slider import MotorValue
                self.motor_adapters[key]=MotorValue(app.backend,key,self.values[key])
                self.motor_values[key]=self.motor_adapters[key].raw
            ttk.Label(body, text=f"J{row - 2}  {joint_label(joint)}").grid(row=row, column=0, sticky="w", padx=(0, 16), pady=3)
            cell=ttk.Frame(body);cell.grid(row=row,column=1,sticky='ew',padx=(0,16),pady=3)
            field = ttk.Spinbox(cell, textvariable=self.motor_values[key], from_=low, to=high,
                                increment=.1 if key=='gripper.pos' else 1, format="%.4f" if key=='gripper.pos' else '%.0f', width=12, style="Editor.TSpinbox")
            field.pack(fill='x')
            from motor_slider import ValueSlider
            self.sliders[key]=ValueSlider(cell,self.motor_values[key],low,high,.1 if key=='gripper.pos' else 1)
            self.sliders[key].scale.pack(fill='x',pady=(3,0))
            self.fields[key] = field
            if key not in self.editable_keys:
                field.configure(state="disabled");self.sliders[key].scale.configure(state="disabled")
            ttk.Label(body, text='0~100' if key=='gripper.pos' else '모터 틱').grid(
                row=row, column=2, sticky="w", padx=(0, 16))
            ttk.Label(body, text=f"{low:g} ~ {high:g}" if key in self.editable_keys else "자동 계산").grid(row=row, column=3, sticky="w")
        self.adjustment = tk.StringVar(master=self.window,value="1")
        adjust = ttk.Frame(body)
        adjust.grid(row=9, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        ttk.Label(adjust, text="버튼 조정 간격").pack(side="left", padx=(0, 8))
        AppChoice(adjust, textvariable=self.adjustment, values=("1", "5", "10", "20", "50"),
                  state="readonly", width=6).pack(side="left")
        ttk.Label(adjust, text="몸체: 모터 틱 / 집게: 0~100").pack(side="left", padx=12)
        for row, key in enumerate(KEYS, start=3):
            controls = ttk.Frame(body)
            controls.grid(row=row, column=4, sticky="e", padx=(8, 0))
            for direction, label in ((-1, "−"), (1, "+")):
                button=ttk.Button(controls, text=label, width=3,
                           command=lambda k=key, d=direction: self.adjust(k, d),state='normal' if key in self.editable_keys else 'disabled')
                button.pack(side="left", padx=2);self.adjust_buttons.setdefault(key,[]).append(button)
        self.message = tk.StringVar(master=self.window,value="몸체 관절은 자동 계산됩니다. 집게 값은 같은 목표의 열림/닫힘 단계에 함께 반영합니다." if step.get("motion") else "")
        ttk.Label(body, textvariable=self.message, foreground=DANGER, wraplength=620).grid(
            row=10, column=0, columnspan=5, sticky="w", pady=(8, 4))
        actions = ttk.Frame(self.window.footer)
        actions.pack(fill='x')
        ttk.Button(actions, text="수정 전 값 복원", command=self.reset).pack(side="left")
        self.apply_button = ttk.Button(actions, text="수정값 적용", command=self.apply, style="Primary.TButton")
        self.apply_button.pack(side="right")
        ttk.Button(actions, text="취소", command=self.close).pack(side="right", padx=8)
        # No modal grab: the main window's STOP remains available while editing.
        self.window.update_idletasks()
        self.window.minsize(self.window.winfo_reqwidth(), self.window.winfo_reqheight())
        self.fields[KEYS[0]].focus_set()
        self.fields[KEYS[0]].selection_range(0, "end")

    def adjust(self, key, direction):
        if key not in self.editable_keys:return
        try:
            value = float(self.motor_values[key].get()) + direction * float(self.adjustment.get())
            low, high = float(self.fields[key].cget("from")), float(self.fields[key].cget("to"))
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{joint_label(key)}: {low:.2f}~{high:.2f} 범위에서 조정하세요.")
            self.motor_values[key].set(f"{value:.4f}".rstrip("0").rstrip("."))
            self.message.set("")
        except ValueError as exc:
            self.message.set(str(exc))

    def reset(self):
        for key, text in self.original_text.items():
            self.values[key].set(text)
        self.message.set("")

    def apply(self, event=None):
        values = {}
        for key, variable in self.values.items():
            text = variable.get().strip()
            try:
                # Opening/applying the editor must not round untouched measurements.
                values[key] = self.step["pose"][key] if text == self.original_text[key] else float(text)
                if not math.isfinite(values[key]):raise ValueError
            except ValueError:
                self.message.set(f"{joint_label(key)}: 유효한 숫자를 입력하세요.")
                self.fields[key].focus_set()
                return "break"
        try:
            self.app.update_step_pose(self.step, values)
        except (ValueError, OSError) as exc:
            self.message.set(message_ko(exc))
            return "break"
        self.close()
        return "break"

    def stop_and_close(self, event=None):
        self.app.stop()
        self.close()
        return "break"

    def close(self):
        if self.app.pose_editor is self:
            for adapter in self.motor_adapters.values():adapter.close()
            self.app.pose_editor = None
            self.window.destroy()
            self.app.refresh_buttons()


class App:
    def __init__(self, root, backend, output):
        import tkinter as tk
        from tkinter import ttk
        self.root, self.backend = root, backend
        self.session = Session(backend, output)
        self.steps = []
        self.name_editor = None
        self.pose_editor = None
        self.reference_editor = None
        self.preview = None
        self.vision = None
        self.vision_busy = False
        self.capture_follow = tk.BooleanVar(master=root,value=False)
        self.teaching_hint = tk.StringVar(master=root,value="지그 보정을 사용할 에피소드를 선택하세요.")
        self.vision_plan = {"targets": {}, "calibration_signature": ""}
        self.name_edit_step = None
        self.teaching_title = ""
        self.saved_path = None
        self.library_paths = []
        self.finish_pending = False
        self.output = Path(output)
        self.motor_dashboard = None
        # Set up telemetry before the worker starts. This constructs no new bus,
        # opens no port and reads nothing on the UI thread. The existing health
        # loop supplies both protection and the optional monitoring window.
        if isinstance(backend, Hardware) and getattr(backend, 'motor_dashboard', None) is None:
            from motor_diagnostics import Collector
            backend.motor_dashboard = Collector(backend.follower.bus, self.output / 'motor-diagnostics')
        self.program_path = self.output / ("poses_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
        self.closing = False
        self.cleanup_fault = False
        self.state = "DISCONNECTED"
        self.latest = None
        self.space_down = False
        self.space_release_job = None
        # Prepend our tag so a focused ttk button cannot also consume Space.
        self.shortcut_tag = "TeachingSpace" + str(id(self))
        root.bind_class(self.shortcut_tag, "<KeyPress-space>", self.space_press)
        root.bind_class(self.shortcut_tag, "<KeyRelease-space>", self.space_release)
        title = backend.identity.get("pair_name", "로봇팔 티칭")
        root.title("SO-101 OpenCV | " + title + (" | 데모" if backend.identity["demo"] else " | 실물"))
        configure_ui(root)
        import re
        geometry = re.match(r"(\d+)x(\d+)", GUI_GEOMETRY)
        width, height = map(int, geometry.groups()) if geometry else (1680, 900)
        root.geometry(f"{min(width, root.winfo_screenwidth()-80)}x{min(height, root.winfo_screenheight()-160)}")
        root.minsize(1080, 680)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)
        self.follower_only = getattr(backend, "follower_only", False)
        self.episode_identity = backend.identity
        self.preview_only = False
        self.buttons = {}

        top = ttk.Frame(root, style="Chrome.TFrame", padding=(14, 8))
        top.grid(row=0, column=0, sticky="ew")
        ttk.Label(top, text="SO-101", style="Model.TLabel").pack(side="left")
        ttk.Label(top, text="로봇팔 티칭", style="Title.TLabel").pack(side="left", padx=16)
        ttk.Label(top, text="데모 · 실물 연결 없음" if backend.identity["demo"] else "실물 · 팔로워 단독" if self.follower_only else f"실물 · {title}",
                  style="Demo.TLabel" if backend.identity["demo"] else "Mode.TLabel").pack(side="right")
        self.utility_toolbar = ttk.Frame(top, style="Chrome.TFrame")
        self.utility_toolbar.pack(side="right", padx=12)
        self.buttons['episode']=ttk.Button(self.utility_toolbar,text='에피소드',command=lambda:self.navigation.home(),style='Nav.TButton')
        self.buttons['episode'].pack(side='left',padx=4)
        self.buttons['manual']=ttk.Button(self.utility_toolbar,text='실물 조정',command=lambda:self.navigation.navigate(self.open_manual),style='Nav.TButton')
        self.buttons['manual'].pack(side='left',padx=4)
        self.buttons["vision"] = ttk.Button(self.utility_toolbar, text="카메라",
                                           command=lambda:self.navigation.navigate(self.open_vision), style="Nav.TButton")
        self.buttons["vision"].pack(side="left")
        self.buttons["settings"]=ttk.Button(self.utility_toolbar,text="장비 설정",command=lambda:self.navigation.navigate(self.open_settings),style="Nav.TButton")
        self.buttons["settings"].pack(side="left",padx=(6,0))
        self.buttons['monitor'] = ttk.Button(self.utility_toolbar, text='장비 모니터링',
                                             command=self.open_monitor, style='Compact.TButton')
        self.buttons['monitor'].pack(side='left',padx=(6,0))

        root.bind("<F1>", lambda _: self.open_help())

        state_strip = ttk.Frame(root, padding=(14, 6))
        state_strip.grid(row=1, column=0, sticky="ew")
        state_strip.columnconfigure(2, weight=1)
        ttk.Label(state_strip, text="로봇 상태", style="Muted.TLabel").grid(row=0, column=0, padx=(0, 10))
        self.status, self.hint = tk.StringVar(), tk.StringVar()
        self.status_label = ttk.Label(state_strip, textvariable=self.status, style="Status.TLabel")
        self.status_label.grid(row=0, column=1, sticky="w")
        self.hint_label = ttk.Label(state_strip, textvariable=self.hint, style="Muted.TLabel", width=1)
        self.hint_label.grid(row=0, column=2, sticky="ew", padx=(18, 0))

        self.workspace = ttk.Frame(root, padding=(14, 0, 14, 8))
        self.workspace.grid(row=2, column=0, sticky="nsew")
        self.workspace.rowconfigure(0, weight=1)
        self.left_workspace = ttk.Frame(self.workspace)
        root._title_editor_host=self.left_workspace
        self.left_workspace.rowconfigure(1, weight=1)
        self.left_workspace.columnconfigure(0, weight=1, minsize=400)
        self.left_workspace.columnconfigure(1, weight=1, minsize=270)
        files = ttk.Frame(self.left_workspace, style="Panel.TFrame", padding=(8, 6))
        files.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        files.columnconfigure(0, weight=1)
        file_header = ttk.Frame(files, style="PanelBody.TFrame")
        file_header.grid(row=0, column=0, sticky="ew")
        file_header.columnconfigure(1, weight=1)
        ttk.Label(file_header, text="에피소드", style="PanelHeading.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.episode_name = tk.StringVar(value="이름 없음")
        self.episode_name_field = ttk.Entry(file_header, textvariable=self.episode_name, state="readonly", width=1, style="Episode.TEntry")
        self.episode_name_field.grid(row=0, column=1, sticky="ew", padx=(0, 14))
        file_actions = ttk.Frame(file_header, style="PanelBody.TFrame")
        file_actions.grid(row=0, column=2, sticky="e")
        for name, label, command in (("rename", "이름 변경", self.rename_episode),
                                     ("new", "새 에피소드", self.new_program),
                                     ("load", "에피소드 열기", self.load),
                                     ("save", "에피소드 저장", self.finish_teaching)):
            button = ttk.Button(file_actions, text=label, command=command, style="Compact.TButton")
            button.pack(side="left", padx=(6, 0))
            self.buttons[name] = button
        file_row = ttk.Frame(files, style="PanelBody.TFrame")
        file_row.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        file_row.columnconfigure(1, weight=1)
        ttk.Label(file_row, text="저장된 에피소드", style="PanelMuted.TLabel").grid(row=0, column=0, padx=(0, 10))
        self.library = AppChoice(file_row, state="readonly", width=1)
        self.library.grid(row=0, column=1, sticky="ew", padx=(0, 14))
        self.library.bind("<<ComboboxSelected>>", self.choose_program)
        self.buttons["play"] = ttk.Button(file_row, text="에피소드 실행", command=self.play, style="Compact.Primary.TButton")
        self.buttons["play"].grid(row=0, column=2, padx=(0, 6))
        self.buttons["simulate"] = ttk.Button(file_row, text="시뮬레이션 실행", command=self.simulate, style="Compact.TButton")
        self.buttons["simulate"].grid(row=0, column=3)
        self.file_status = tk.StringVar()
        self.file_status_label = ttk.Label(files, textvariable=self.file_status, style="PanelMuted.TLabel", width=1)
        self.file_status_label.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        from jig_catalog import install_picker
        install_picker(self,files)

        self.workspace_layout = None
        self.controls = self.steps_panel = ttk.Frame(self.left_workspace, style="Panel.TFrame", padding=1)
        steps = self.steps_panel
        steps.columnconfigure(0, weight=1)
        steps.rowconfigure(1, weight=1)
        step_header = ttk.Frame(steps, style="Band.TFrame", padding=(8, 3))
        step_header.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(step_header, text="스텝 목록", style="Section.TLabel").pack(side="left")
        ttk.Checkbutton(step_header,text="새 스텝 지그 보정",variable=self.capture_follow).pack(side="left",padx=8)
        self.step_count = tk.StringVar(value="0개 스텝")
        ttk.Label(step_header, textvariable=self.step_count, style="Band.TLabel").pack(side="right")
        self.table = ttk.Treeview(steps, columns=("number", "name", "correction"), show="headings", selectmode="browse", height=6)
        self.table.heading("number", text="순서")
        self.table.heading("name", text="스텝 이름")
        self.table.column("number", width=44, minwidth=44, stretch=False, anchor="center")
        self.table.column("name", width=120, minwidth=80)
        self.table.heading("correction", text="지그 보정")
        self.table.column("correction", width=88, minwidth=88, stretch=False, anchor="center")
        self.table.tag_configure("stripe", background=STRIPE)
        self.table.tag_configure("drop-target",background="#a9dde7")
        self.step_drag=None
        self.table.bind("<ButtonPress-1>",self.begin_step_drag)
        self.table.bind("<B1-Motion>",self.drag_step)
        self.table.bind("<ButtonRelease-1>",self.end_step_drag)
        self.table.bind("<Alt-Up>",lambda _:self.reorder(-1))
        self.table.bind("<Alt-Down>",lambda _:self.reorder(1))
        self.table.grid(row=1, column=0, sticky="nsew", padx=(8, 0), pady=(8, 0))
        scrollbar = ttk.Scrollbar(steps, orient="vertical", command=self.scroll_steps)
        scrollbar.grid(row=1, column=1, sticky="ns", padx=(0, 8), pady=(8, 0))
        self.table.configure(yscrollcommand=scrollbar.set)
        self.empty_steps = ttk.Label(self.table, text="저장된 에피소드를 열거나\n리더 따라가기로 자세를 만든 뒤\n스텝을 추가하세요.",
                                     justify="center", background=SURFACE, foreground=MUTED)
        self.table.bind("<<TreeviewSelect>>", self.select)
        for event in ("<ButtonRelease-1>", "<KeyRelease-Up>", "<KeyRelease-Down>"):
            self.table.bind(event, self.preview_selected_step, add="+")
        self.table.bind("<Double-1>", self.begin_step_name_edit)
        self.table.bind("<Configure>", self.cancel_step_name_edit)
        for event in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.table.bind(event, self.cancel_step_name_edit, add="+")
        self.step_correction_detail = tk.StringVar(value="스텝을 선택하면 지그 보정 방식을 표시합니다.")
        self.step_correction_label = ttk.Label(steps, textvariable=self.step_correction_detail,
                                               style="PanelMuted.TLabel", width=1, wraplength=240)
        self.step_correction_label.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(6, 0))
        steps.bind("<Configure>", lambda event: self.step_correction_label.grid_remove()
                   if event.height < 400 else self.step_correction_label.grid())
        self.step_correction_label.bind("<Configure>",
            lambda event: self.step_correction_label.configure(wraplength=max(100, event.width)))
        reference_bar=ttk.Frame(steps,style="PanelBody.TFrame")
        reference_bar.grid(row=3,column=0,columnspan=2,sticky="ew",padx=8,pady=(5,0))
        reference_bar.columnconfigure(0,weight=1)
        self.buttons["move"]=ttk.Button(reference_bar,text="스텝 실행",command=self.move,style="Compact.Primary.TButton")
        self.buttons["move"].grid(row=0,column=0,sticky="ew",pady=(0,5))
        self.buttons["reference"]=ttk.Button(reference_bar,text="스텝 설정 · 고정 / 지그 따라가기",command=self.edit_step_reference,style="Compact.TButton")
        self.buttons["reference"].grid(row=1,column=0,sticky="ew")
        edit = ttk.Frame(steps, style="PanelBody.TFrame")
        edit.grid(row=4, column=0, columnspan=2, sticky="ew", padx=8, pady=3)
        for col in range(2):
            edit.columnconfigure(col, weight=1, uniform="action")
        for index, (name, label, command) in enumerate((
                ("capture", "스텝 추가", lambda: self.send("capture")),
                ("edit_pose", "관절값 편집", self.edit_step_pose),
                ("safe", "안전 자세 추가", self.add_safe_steps),
                ("delete", "스텝 삭제", self.delete))):
            button = ttk.Button(edit, text=label, command=command, style="Compact.TButton")
            button.grid(row=index // 2, column=index % 2, sticky="ew",
                        padx=(0, 4) if index % 2 == 0 else 0, pady=(0, 4) if index < 4 else 0)
            self.buttons[name] = button


        positions = self.joints_panel = ttk.Frame(self.left_workspace, style="Panel.TFrame", padding=1)
        positions.columnconfigure(0, weight=1)
        positions.rowconfigure(1, weight=1)
        joint_header = ttk.Frame(positions, style="Band.TFrame", padding=(10, 7))
        joint_header.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(joint_header, text="관절값", style="Section.TLabel").pack(side="left")
        ttk.Label(joint_header, text="현재값 / 선택 스텝 · 모터 틱", style="Band.TLabel").pack(side="right")
        self.joints = ttk.Treeview(positions, style="Joint.Treeview", columns=("axis", "joint", "unit", "leader", "follower", "step"),
                                   show="headings", height=6, selectmode="none")
        for key, label, width in (("axis", "축", 30), ("joint", "관절", 74), ("unit", "단위", 61),
                                  ("leader", "리더", 65), ("follower", "팔로워", 65), ("step", "스텝", 65)):
            self.joints.heading(key, text=label)
            self.joints.column(key, width=width, minwidth=width, stretch=key not in ("axis", "unit"),
                               anchor="w" if key == "joint" else "center")
        self.joints.tag_configure("stripe", background=STRIPE)
        for index, joint in enumerate(JOINTS):
            self.joints.insert("", "end", iid=joint,
                               values=(f"J{index + 1}", joint_label(joint), ("0~100" if joint=="gripper" else "모터 틱"), "—", "—", "—"),
                               tags=("stripe",) if index % 2 else ())
        self.joints.grid(row=1, column=0, sticky="nsew", padx=(8, 0), pady=(8, 0))
        joint_scroll = ttk.Scrollbar(positions, orient="vertical", command=self.joints.yview)
        joint_scroll.grid(row=1, column=1, sticky="ns", padx=(0, 8), pady=(8, 0))
        self.joints.configure(yscrollcommand=joint_scroll.set)
        self.joints.bind("<Configure>",self.fit_joint_rows)
        self.detail = tk.StringVar(value="스텝을 선택하면 저장된 관절값을 표시합니다.")
        self.detail_label = ttk.Label(positions, textvariable=self.detail, style="PanelMuted.TLabel", width=1)
        self.detail_label.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=8)

        logs = ttk.Frame(root, padding=(14, 2))
        logs.grid(row=4, column=0, sticky="ew")
        logs.columnconfigure(1, weight=1)
        ttk.Label(logs, text="작업 기록").grid(row=0, column=0, sticky="w", padx=(0, 12))
        self.latest_notice = tk.StringVar(value="아직 기록이 없습니다.")
        ttk.Label(logs, textvariable=self.latest_notice, style="Muted.TLabel", width=1,
                  anchor="w").grid(row=0, column=1, sticky="ew")
        self.log_expanded = tk.BooleanVar(value=False)
        ttk.Checkbutton(logs, text="기록 펼치기", variable=self.log_expanded,
                        command=self.toggle_log).grid(row=0, column=2, padx=(12, 0))
        self.log_panel = ttk.Frame(logs)
        self.log_panel.columnconfigure(0, weight=1)
        self.log = tk.Text(self.log_panel, height=3, state="disabled", wrap="word", font="TkDefaultFont",
                           background=SURFACE, foreground=INK, relief="flat", padx=8, pady=6)
        self.log.grid(row=0, column=0, sticky="ew")
        log_scroll = ttk.Scrollbar(self.log_panel, orient="vertical", command=self.log.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=log_scroll.set)
        if backend.identity["demo"] and not self.follower_only:
            demo = ttk.Frame(root, padding=(14, 3))
            demo.grid(row=5, column=0, sticky="ew")
            ttk.Label(demo, text="데모 리더 · 어깨 회전", style="Muted.TLabel").pack(side="left", padx=(0, 12))
            ttk.Scale(demo, from_=-40, to=40, command=lambda v: backend.lead.update({"shoulder_pan.pos": float(v)})).pack(side="left", fill="x", expand=True)

        # One operation bar spans the entire window; STOP is never inside a tab.
        self.operation_bar = toolbar = ttk.Frame(root, style="Chrome.TFrame", padding=(14, 8))
        toolbar.grid(row=6, column=0, sticky="ew")
        for name, label, command in (("connect", "로봇 연결", self.connect),
                                     ("hold_current", "현재 자세 유지 · 토크 켜기", lambda:self.send("hold_current")),
                                     ("release", "손으로 조정 · 토크 끄기", lambda:self.send("release")),
                                     ("sync", "리더와 동기화", self.sync),
                                     ("teach", "리더 따라가기", lambda: self.send("teach")),
                                     ("pause", "동작 중지 · 자세 유지", lambda: self.send("pause")),
                                     ("stop", "정지 / 토크 해제", self.stop)):
            button = ttk.Button(toolbar, text=label, command=command,
                                style="Operation.Stop.TButton" if name == "stop" else
                                      "Run.Primary.TButton" if name == "teach" else "Operation.TButton")
            button.pack(side="right" if name == "stop" else "left", padx=(0, 8) if name != "stop" else 0)
            self.buttons[name] = button
        if self.follower_only:
            self.buttons["teach"].pack_forget()
            self.buttons["sync"].pack_forget()
            self.buttons["connect"].configure(text="팔로워 연결 · 토크 꺼짐")
            self.buttons["capture"].configure(text="현재 자세로 스텝 추가")
            self.joints.configure(displaycolumns=("axis","joint","unit","follower","step"))
            self.empty_steps.configure(text="① 아래에서 팔로워 연결\n② 현재 자세로 스텝 추가\n③ 관절값 편집 → 스텝 실행\n④ 에피소드 저장")
        else:
            self.buttons['hold_current'].pack_forget();self.buttons['release'].pack_forget()
        self.shortcut_hint = ttk.Label(toolbar, text="Space: 스텝 추가 (리더 따라가기 중)\nEsc: 정지 / 토크 해제",
                                      style="Chrome.TLabel", justify="center", anchor="center", width=1)
        if self.follower_only:self.shortcut_hint.configure(text="Space: 현재 자세 저장 (토크 꺼짐도 가능)\nEsc: 정지 / 토크 해제")
        self.shortcut_hint.pack(side="left", fill="both", expand=True, padx=12)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Escape>", lambda _: self.stop())
        root.bind("<Configure>", self.resize_ui, add="+")
        root.bind("<ButtonRelease-1>", self.preview_current_on_outside_click, add="+")
        from app_navigation import Navigation
        self.navigation=Navigation(self)
        self.refresh_library()
        self.layout_workspace()
        self.refresh_buttons()
        self.install_space_shortcut(root)
        self.poll()

    def open_manual(self):
        self.show_preview()
        if self.preview is not None:self.preview.show_simulation()
        view=self.open_vision(show=False)
        if view is not None:view.select_tab(4)

    def open_settings(self):
        view=self.open_vision(show=False)
        if view is not None:view.show_settings()

    def edit_step_reference(self):
        self.show_preview()
        if self.preview is not None:
            self.navigation.home();self.preview.show_step_settings()
            self.preview.step_settings.choice.focus_set()
            self.preview_selected_step()

    def open_help(self):
        from user_help import open_manual
        try:
            open_manual()
        except Exception as exc:
            self.note(f"설명서를 열 수 없습니다: {exc}")

    def open_vision(self, show=True):
        try:
            if self.vision is None:
                from vision_workspace import VisionWorkspace
                self.vision = VisionWorkspace(self)
            if show:self.vision.select_tab(3)
            return self.vision
        except ImportError as exc:
            self.note(f"카메라 기능 설치 필요: bash install_vision.sh ({exc})")
        except Exception as exc:
            self.note(f"카메라 설정을 열 수 없습니다: {exc}")
        return None

    def connect_startup_camera(self):
        """Connect once after startup without changing the active page or robot state."""
        if self.closing or getattr(self, '_startup_camera_attempted', False):
            return
        self._startup_camera_attempted = True
        view = self.open_vision(show=False)
        if view is None or view.closed or view.process is not None:
            return
        try:
            view.start_camera()
        except Exception as exc:
            message = f"상단 카메라 자동 연결 실패: {exc} · 카메라 연결 버튼으로 다시 시도하세요."
            view.status.set(message)
            view.camera_status.set(message)
            self.note(message)

    def open_monitor(self):
        if self.closing:
            return None
        try:
            from motor_dashboard import Dashboard
            import tkinter as tk
            collector = getattr(self.backend, 'motor_dashboard', None)
            if collector is None and not self.backend.identity['demo']:
                raise RuntimeError('현재 연결의 모터 측정기를 사용할 수 없습니다. GUI를 다시 실행하세요.')
            if self.motor_dashboard is None or not self.motor_dashboard.root.winfo_exists():
                from app_navigation import AppPage
                window = AppPage(self,persistent=True)
                self.motor_dashboard = Dashboard(window, collector, stop_motion=self.stop,
                                                 demo=self.backend.identity['demo'], managed=True)
                if collector is not None and collector.latest is not None:
                    self.motor_dashboard.accept(collector.latest)
            self.motor_dashboard.root.deiconify()
            self.motor_dashboard.root.lift()
            return self.motor_dashboard
        except Exception as exc:
            self.note(f'상태 모니터링을 열 수 없습니다: {exc}')
            return None

    def fit_joint_table(self):
        # Tables scroll when space is limited; never grow the window offscreen.
        self.root.update_idletasks()

    def toggle_log(self):
        if self.log_expanded.get():
            self.log_panel.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        else:
            self.log_panel.grid_remove()

    def fit_joint_rows(self, event=None):
        from tkinter import ttk,font
        style=ttk.Style(self.root)
        face=font.Font(self.root,font=style.lookup('Treeview','font'))
        first=self.joints.bbox(JOINTS[0])
        header=first[1] if first else face.metrics('linespace')+12
        height=max(face.metrics('linespace')+2,min(32,(self.joints.winfo_height()-header-2)//len(JOINTS)))
        if int(style.lookup('Joint.Treeview','rowheight') or 32)!=height:
            style.configure('Joint.Treeview',rowheight=height)

    def resize_ui(self, event):
        if event.widget is self.root:
            self.layout_workspace()
            self.hint_label.configure(wraplength=max(300, event.width-330))

    def refresh_guidance(self):
        status = "토크 해제 미확인 · 모터 전원 차단 필요" if self.cleanup_fault else state_label(self.state)
        if self.latest and self.state not in ("DISCONNECTED", "STOPPING"):
            if self.latest.get("limited_joints"):
                status += " · 범위 제한"
            if self.latest.get("lagging_joints"):
                status += " · 추종 지연"
            pending = self.latest.get('temperature_pending', [])
            if pending:
                hottest = max(pending, key=lambda v: v['elapsed_s'])
                status += ' · '+joint_label(hottest['joint'])+' '+pending_label(pending)
        color = DANGER if self.cleanup_fault or self.state == "STOPPING" else ACCENT
        if self.status.get() != status:
            self.status.set(status)
            self.status_label.configure(foreground=color)
        hints = {
            "DISCONNECTED": "에피소드를 열어 편집하거나 시뮬레이션을 실행하세요.",
            "CONNECTING": "로봇과 보정값을 확인하고 있습니다. 두 팔을 움직이지 마세요.",
            "READY": "리더와 동기화를 누른 뒤 리더 따라가기로 티칭을 시작하세요.",
            "SYNCING": "리더와 동기화 중입니다. 리더를 움직이지 마세요.",
            "HOLD": "리더 따라가기로 자세를 기록하거나 스텝·에피소드를 실행하세요.",
            "TEACH": "리더 팔을 움직이면 팔로워가 따라갑니다. 스페이스로 스텝을 추가하세요.",
            "MOVE": "선택한 스텝을 실행 중입니다. 동작 중지 · 자세 유지를 누르면 이동을 멈춥니다.",
            "PLAY": "스텝을 순서대로 실행합니다. 동작 중지 · 자세 유지를 누르면 실행을 취소합니다.",
            "STOPPING": "정지와 토크 해제를 확인하고 있습니다. 로봇팔이 내려오지 않도록 받쳐 주세요.",
        }
        if self.follower_only:
            hints.update({"CONNECTING":"팔로워 연결과 보정값을 확인합니다. 리더암은 사용하지 않습니다.",
                          "READY":"토크 꺼짐 · 손으로 자세를 맞추고 스텝을 저장하세요. 실물 실행 전 토크 켜기를 누르세요.",
                          "HOLD":"현재 자세로 스텝 추가 → 관절값 편집 → 스텝 실행 → 에피소드 저장",
                          "DISCONNECTED":"팔로워를 연결하거나 기존 실물 스텝을 편집하세요."})
        hint = "토크 해제가 확인되지 않았습니다. 모터 전원을 차단하고 작업 기록을 확인하세요." if self.cleanup_fault else hints.get(self.state, "")
        if self.preview_only:
            hint = "다른 로봇의 에피소드입니다. 시뮬레이션으로만 확인할 수 있습니다."
        elif self.finish_pending:
            hint = "움직임을 멈추고 에피소드를 저장할 준비를 하고 있습니다."
        elif self.pose_editor is not None:
            hint = "스텝 관절값을 편집 중입니다. 값을 적용한 뒤 스텝 실행을 누르세요."
        if self.hint.get() != hint:
            self.hint.set(hint)
        episode_name = self.teaching_title.strip() or "이름 없음"
        if self.episode_name.get() != episode_name:
            self.episode_name.set(episode_name)
            self.episode_name_field.xview_moveto(0)
        count = f"{len(self.steps)}개 스텝"
        if self.step_count.get() != count:
            self.step_count.set(count)
        if self.steps and self.empty_steps.winfo_manager():
            self.empty_steps.place_forget()
        elif not self.steps and not self.empty_steps.winfo_manager():
            self.empty_steps.place(relx=.5, rely=.5, anchor="center")
        filename = self.saved_path.name if self.saved_path is not None else ""
        if len(filename) > 28:
            filename = filename[:27] + "…"
        file_status = ("저장하면 티칭을 멈추고 에피소드 이름을 입력합니다." if self.state == "TEACH" else
                       f"저장됨 · {filename}" if self.saved_path is not None else
                       "수정됨 · 에피소드 저장으로 파일에 반영하세요." if self.steps else
                       "팔로워를 연결하고 현재 자세로 스텝을 추가하세요." if self.follower_only else
                       "에피소드를 열거나 리더 따라가기로 스텝을 추가하세요.")
        if self.vision is not None and self.vision_plan['targets']:
            file_status += ' · '+self.teaching_hint.get()
        if self.preview_only:
            file_status = "로봇·보정 정보 불일치 · 읽기 전용 · 시뮬레이션 실행 가능"
        if self.file_status.get() != file_status:
            self.file_status.set(file_status)

    def install_space_shortcut(self, widget):
        widget.bindtags((self.shortcut_tag,) + widget.bindtags())
        for child in widget.winfo_children():
            self.install_space_shortcut(child)

    def space_press(self, event):
        widget = event.widget
        while widget is not None:
            if isinstance(widget, AppChoice):return
            widget = getattr(widget, "master", None)
        # Tk/X11 auto-repeat emits release/press pairs; cancel the pending release.
        if self.space_release_job is not None:
            self.root.after_cancel(self.space_release_job)
            self.space_release_job = None
        if (self.state not in (("READY","HOLD") if getattr(self,"follower_only",False) else ("TEACH",)) or self.closing or self.finish_pending
                or self.name_editor is not None or getattr(self,'reference_editor',None) is not None or self.root.grab_current() is not None
                or event.widget.winfo_toplevel() != self.root
                or event.widget.winfo_class() in ("Entry", "TEntry", "Text", "TCombobox", "Spinbox", "TSpinbox")):
            return
        if not self.space_down:
            self.space_down = True
            self.send("capture" if getattr(self,"follower_only",False) else "quick_capture")
        return "break"

    def space_release(self, event):
        if self.space_down:
            self.space_release_job = self.root.after_idle(self.reset_space)
            return "break"

    def reset_space(self):
        self.space_down = False
        self.space_release_job = None

    def note(self, message):
        print(message, flush=True)
        self.log.configure(state="normal")
        localized = message_ko(message)
        self.latest_notice.set(localized if len(localized) <= 70 else localized[:69] + "…")
        self.log.insert("end", f"{datetime.now():%H:%M:%S}  {localized}\n")
        self.log.see("end")
        if int(self.log.index("end-1c").split(".")[0]) > 600:
            self.log.delete("1.0", "100.0")
        self.log.configure(state="disabled")

    def refresh_buttons(self):
        idle = self.state in ("READY", "HOLD", "DISCONNECTED")
        if getattr(self, "name_editor", None) is not None and (not idle or self.finish_pending or self.closing):
            self.cancel_step_name_edit()
        if self.pose_editor is not None and (not idle or self.finish_pending or self.closing):
            self.pose_editor.close()
        selected = self.selected()
        if self.preview is not None and hasattr(self.preview,'step_settings'):self.preview.step_settings.refresh()
        if hasattr(self,'jig_picker'):
            self.jig_picker.configure(state='readonly' if idle and not self.vision_busy and not self.preview_only and not self.closing else 'disabled')
        simulating = self.preview is not None and self.preview.simulation_active
        for name, button in self.buttons.items():
            enabled = {"connect": self.state == "DISCONNECTED" and not self.cleanup_fault and not getattr(self,"calibration_busy",False) and not getattr(self,"calibration_reload_required",False), "teach": self.state == "HOLD",
                       "hold_current": self.follower_only and self.state=="READY",
                       "release": self.follower_only and self.state=="HOLD",
                       "sync": self.state in ("READY", "HOLD"),
                       "pause": self.vision_busy or self.state in ("TEACH", "MOVE", "SYNCING", "PLAY"),
                       "stop": self.state != "DISCONNECTED" or (self.preview is not None and self.preview.player.running),
                       "simulate": idle and bool(self.steps) and self.preview is not None and self.preview.photo is not None and not self.preview.render_error,
                       "edit_pose": idle and selected is not None,
                       "reference": idle and selected is not None,
                       "delete": idle and selected is not None,
                       "safe": idle and bool(self.steps),
                       "capture": self.state == "HOLD" or (self.follower_only and self.state=="READY"), "move": self.state == "HOLD" and selected is not None,
                       "play": self.state == "HOLD" and self.saved_path is not None,
                       "save": self.state == "TEACH" or (idle and bool(self.steps))}.get(name, idle)
            if self.follower_only and name in ("teach","sync"):enabled = False
            if self.preview_only and name not in ("new", "load", "simulate", "stop"):
                enabled = False
            if self.finish_pending and name != "stop":
                enabled = False
            if (getattr(self, "name_editor", None) is not None or self.pose_editor is not None or self.reference_editor is not None) and name not in ("stop","monitor"):
                enabled = False
            if name == "edit_pose":button.configure(text="관절값 편집")
            if name == "simulate":
                label = "실행 중지" if simulating else "시뮬레이션 실행"
                if button.cget("text") != label:
                    button.configure(text=label)
                if simulating:
                    enabled = True
            if self.vision_busy and name not in ("stop", "pause"):
                enabled = False
            if name in ('episode','monitor','manual','vision','settings'):
                enabled = True  # viewing telemetry is allowed during motion and editing
            desired = "normal" if enabled and not self.closing else "disabled"
            if str(button.cget("state")) != desired:
                button.configure(state=desired)
        library_state = ("readonly" if idle and not self.closing and not self.finish_pending
                         and self.name_editor is None and self.pose_editor is None and not self.vision_busy else "disabled")
        if str(self.library.cget("state")) != library_state:
            self.library.configure(state=library_state)
        self.refresh_guidance()

    def connect(self, *, auto_sync=True):
        from tkinter import messagebox
        if getattr(self, 'calibration_busy', False):
            self.note('모터 캘리브레이션을 마친 뒤 연결하세요.'); return
        if getattr(self, 'calibration_reload_required', False):
            self.note('모터 보정이 바뀌었습니다. 앱을 다시 실행하세요.'); return
        if self.state != "DISCONNECTED" or self.cleanup_fault or self.closing or self.preview_only:
            return
        if auto_sync and not self.follower_only and not self.backend.identity["demo"] and not messagebox.askokcancel(
                "리더 자세로 맞추기", "연결 후 팔로워가 현재 리더 자세로 이동합니다. 시작할까요?", parent=self.root):
            return
        self.session = Session(self.backend, self.output, auto_sync=auto_sync)
        self.state = "CONNECTING"
        self.latest = None
        self.refresh_buttons()
        self.session.start()

    def sync(self):
        if self.follower_only:return
        from tkinter import messagebox
        if self.state not in ("READY", "HOLD") or self.latest is None:
            return
        if time.monotonic() - self.latest["sample_time"] > DISPLAY_STALE_SECONDS:
            self.note("Position display is stale; wait for a fresh sample.")
            return
        approved = {role: self.latest[role].copy() for role in ("leader", "follower")}
        try:
            goal = sync_goal(self.backend, approved, approved["leader"], approved["follower"])
        except ValueError as exc:
            self.note(str(exc))
            return
        lines = [f"{joint_label(joint)}: {approved['follower'][key]:.2f} → {goal[key]:.2f} "
                 f"({goal[key] - approved['follower'][key]:+.2f} {unit_label(key)})"
                 for joint, key in zip(JOINTS, KEYS)]
        if difference(goal, approved["leader"]) > .001:
            lines.append("팔로워의 보정 범위로 제한되어 리더 자세와 정확히 일치하지 않을 수 있습니다.")
        if messagebox.askokcancel("리더와 동기화", "\n".join(lines) + "\n\n이 자세로 이동할까요?", parent=self.root):
            if self.state in ("READY", "HOLD") and not self.session.stop.is_set():
                self.send("sync", approved)

    def send(self, name, value=None):
        if name in ("pause", "finish") and self.vision is not None:
            self.vision.cancel_execution("사용자가 실행을 중지했습니다.")
        if self.follower_only and name in ("teach","sync","quick_capture"):return
        if self.preview_only and name in ("capture", "quick_capture", "teach", "move", "play", "jog", "measure_target"):
            return
        if name in ("capture", "quick_capture"):
            if len(self.steps) >= MAX_STEPS:
                self.note(f"Maximum {MAX_STEPS} steps.")
                return
            try:context=self.new_capture_context()
            except ValueError as exc:self.note(str(exc));return False
            value = next_step_name(self.steps)
            if context is not None:value={'name':value,'teaching_context':context}
        try:
            self.session.command(name, value)
            return True
        except queue.Full:
            self.note("Command queue full")
            return False

    def new_capture_context(self):
        if not self.capture_follow.get():return None
        if not self.vision_plan.get('targets'):raise ValueError('티칭에 사용할 지그를 먼저 선택하세요.')
        if self.vision is None:raise ValueError('카메라에서 티칭 기준을 먼저 읽어 주세요.')
        view=self.vision;profile=view.read_target()
        if profile.get('mode')!='rectangle':raise ValueError('지그를 따라갈 목표를 선택하세요.')
        from copy import deepcopy
        from vision_teaching import reference_key,calibration_signature
        observation=view.teaching_observations({'targets':{'capture':profile}})[reference_key(profile)]
        return {'profile':deepcopy(profile),'observation':observation,
                'reference_id':view.teaching_reference.value['id'],'calibration':calibration_signature(view.station)}

    def store_captured_step(self,value,context=None):
        from copy import deepcopy
        value=deepcopy(value);value.pop('_teaching_context',None)
        value['name']=next_step_name(self.steps)
        steps=[*self.steps,value];plan=self.vision_plan
        if context is not None:
            from vision_teaching import calibration_signature
            view=self.vision
            if view is None or view.teaching_reference.value is None or view.teaching_reference.value['id']!=context['reference_id']:
                raise ValueError('저장 중 에피소드의 티칭 기준이 바뀌었습니다. 다시 저장하세요.')
            if context['calibration']!=calibration_signature(view.station):raise ValueError('저장 중 보정값이 바뀌었습니다.')
            from step_reference import change_reference
            steps,plan=change_reference(steps,plan,[len(steps)-1],'지그 따라가기',view.station,
                profile=context['profile'],observation=context['observation'])
        self.steps,self.vision_plan=steps,plan
        self.refresh_steps(len(steps)-1);self.edited()
        if self.vision:self.vision.refresh_target_list()

    def stop(self):
        calibration = getattr(self.vision, 'calibration_window', None) if self.vision else None
        if calibration and calibration.pending_start:
            calibration.close()
        elif calibration and calibration.worker and not calibration.worker.finished.is_set():
            calibration.worker.cancel.set()
        if self.vision is not None:
            self.vision.cancel_execution("정지 요청으로 실행을 취소했습니다.")
        if self.preview is not None:
            self.preview.pause_simulation()
        self.finish_pending = False
        self.session.stop.set()
        self.session.wakeup.set()
        if self.state != "DISCONNECTED":
            self.state = "STOPPING"
            self.status.set("정지 처리 중 · 로봇팔을 받쳐 주세요")
        self.refresh_buttons()

    def selected(self):
        selection = self.table.selection()
        return int(selection[0]) if selection and selection[0].isdecimal() else None

    def move(self):
        index = self.selected()
        if index is None or self.preview_only or self.state != "HOLD":
            return
        step = self.steps[index]
        if step.get("motion"):
            view = self.open_vision(show=False)
            if view is not None:
                view.guarded(lambda: view.execute(index))
            return
        target=step["pose"].copy()
        if step.get("gripper_only"):
            if not self.latest or time.monotonic()-self.latest["sample_time"]>.6:return
            target={**self.latest['follower'],'gripper.pos':target['gripper.pos']}
        self.send("move", target)

    def refresh_step_corrections(self):
        for index, step in enumerate(self.steps):
            if self.table.exists(str(index)):
                self.table.set(str(index), "correction", step_correction(step, self.vision_plan)[0])
        index = self.selected()
        self.step_correction_detail.set(step_correction(self.steps[index], self.vision_plan)[1]
                                        if index is not None else "스텝을 선택하면 지그 보정 방식을 표시합니다.")

    def select(self, event=None):
        self.refresh_step_corrections()
        index = self.selected()
        if index is not None:
            step = self.steps[index]
            name = step["name"]
            display_name = name if len(name) <= 20 else name[:19] + "…"
            self.detail.set(f"선택 스텝: {display_name}")
        else:
            self.detail.set("스텝을 선택하면 저장된 관절값을 표시합니다.")
        for key in KEYS:
            value = self.steps[index]["pose"][key] if index is not None else None
            from motor_positions import to_ticks
            display='—' if value is None else f"{value:.4f}".rstrip('0').rstrip('.')
            if value is not None and key!='gripper.pos':
                display=f'{value:.2f}°' if self.preview_only else str(to_ticks(self.backend,key,value))
            self.joints.set(key.removesuffix(".pos"),"step",display)
        if self.table.selection()==('measure',):
            self.step_correction_detail.set("시작 안전 자세 도착 후 최대 3초간 새 지그 위치를 읽습니다. 티칭 기준은 그대로 유지합니다.")
        self.refresh_buttons()

    def preview_selected_step(self, event=None):
        if self.preview is not None and self.selected() is not None:
            from robot_preview import TARGET
            self.preview.pause_simulation()
            self.preview.mode.set(TARGET)

    def preview_current_on_outside_click(self, event):
        from robot_preview import CURRENT, TARGET
        view = self.preview
        if (view is None or self.closing or view.mode.get() != TARGET
                or self.selected() is None or self.pose_editor is not None
                or self.name_editor is not None):
            return
        widget = event.widget
        # Page navigation is not a request to deselect the working episode/step.
        ancestor=widget
        while ancestor is not None:
            if ancestor in (self.utility_toolbar,self.navigation.host,getattr(getattr(view,"step_settings",None),"window",None)):return
            ancestor=getattr(ancestor,'master',None)
        if widget is self.table and self.table.identify_row(event.y):
            return
        # Camera and reorder controls operate on the selected step.
        if widget in (view.canvas, view.camera_reset):
            return
        if not widget.winfo_exists() or widget.winfo_toplevel() is not self.root:
            return
        self.clear_step_selection()
        view.mode.set(CURRENT)

    def clear_step_selection(self):
        selection = self.table.selection()
        if selection:
            self.table.selection_remove(*selection)
        self.table.focus("")
        self.select()

    def edit_step_pose(self):
        if (self.state not in ("READY", "HOLD", "DISCONNECTED") or self.closing or self.preview_only
                or self.finish_pending or self.name_editor is not None):
            return
        if self.pose_editor is not None:
            self.pose_editor.window.lift()
            self.preview_selected_step()
            return
        index = self.selected()
        if index is None:
            return
        self.show_preview()
        self.preview.show_simulation()
        self.pose_editor = StepPoseEditor(self, self.steps[index])
        self.preview_selected_step()
        self.refresh_buttons()

    def update_step_pose(self, step, values):
        if self.state not in ("READY", "HOLD", "DISCONNECTED") or self.closing or self.finish_pending or self.preview_only:
            raise ValueError("Hold or stop teaching/playback before editing a step.")
        index = next((i for i, candidate in enumerate(self.steps) if candidate is step), None)
        if index is None:
            raise ValueError("This step is no longer in the teaching. Select a step again.")
        target = self.backend.validate_target(pose(values))
        from step_reference import editable_servos,edit_calculated_gripper
        if any(target[key]!=step['pose'][key] for key in KEYS if key not in editable_servos(step)):
            raise ValueError('자동 계산되는 관절은 직접 편집할 수 없습니다. 스텝의 위치 기준을 확인하세요.')
        if target == step["pose"]:
            return
        if step.get('motion'):
            steps,plan=edit_calculated_gripper(self.steps,self.vision_plan,index,target['gripper.pos'])
            document={**self.document(),'steps':steps,'vision_plan':plan}
            validate_program(document,self.episode_identity);atomic_json(self.program_path,document)
            self.steps,self.vision_plan=steps,plan
            if self.preview is not None:self.preview.invalidate_episode()
            self.saved_path=None;self.library.set('');self.refresh_steps(index)
            if self.vision is not None:self.vision.refresh_target_list();self.vision.load_target()
            self.note('집게 값을 반영했습니다. 지그 경로는 실행 시 다시 계산합니다. 에피소드를 저장하세요.')
            return
        steps = self.steps.copy()
        steps[index] = {**step, "pose": target}
        if "fixed_pose" in step:
            steps[index]["fixed_pose"]=target.copy()
            steps[index].pop("follow_motion",None);steps[index].pop("teaching_reference",None)
        if step.get("safe_boundary"):
            steps = [{**item,"pose":target.copy()} if item.get("safe_boundary") else item for item in steps]
        # Commit the draft first. Failed writes keep both the old pose and saved-file state.
        atomic_json(self.program_path, {**self.document(), "steps": steps})
        step["pose"] = target
        if "fixed_pose" in step:
            step["fixed_pose"]=target.copy()
            step.pop("follow_motion",None);step.pop("teaching_reference",None)
        if step.get("safe_boundary"):
            for item in self.steps:
                if item.get("safe_boundary"):item["pose"] = target.copy()
        if self.preview is not None:
            self.preview.invalidate_episode()
        self.saved_path = None
        self.library.set("")
        self.table.selection_set(str(index))
        self.select()
        self.note(f"{step['name']} 관절값을 초안에 반영했습니다. 에피소드 저장을 눌러 저장하세요.")

    def cancel_step_name_edit(self, event=None):
        editor = getattr(self, "name_editor", None)
        if editor is not None:
            self.name_editor = self.name_edit_step = None
            editor.destroy()
            self.refresh_buttons()

    def scroll_steps(self, *args):
        self.cancel_step_name_edit()
        self.table.yview(*args)

    def begin_step_name_edit(self, event):
        from tkinter import ttk
        if (self.state not in ("READY", "HOLD", "DISCONNECTED") or self.closing or self.preview_only
                or self.finish_pending or self.pose_editor is not None):
            return
        if self.table.identify_region(event.x, event.y) != "cell" or self.table.identify_column(event.x) != "#2":
            return
        row = self.table.identify_row(event.y)
        if not row or not row.isdecimal():
            return
        bounds = self.table.bbox(row, "name")
        if not bounds:
            return
        self.cancel_step_name_edit()
        self.table.selection_set(row)
        self.name_edit_step = self.steps[int(row)]
        editor = self.name_editor = ttk.Entry(self.table)
        editor.insert(0, self.name_edit_step["name"])
        editor.place(x=bounds[0], y=bounds[1], width=bounds[2], height=bounds[3])
        editor.select_range(0, "end")
        editor.focus_set()
        editor.bind("<Return>", lambda event: self.finish_step_name_edit(editor))
        editor.bind("<Escape>", lambda event: (self.cancel_step_name_edit(), "break")[-1])
        editor.bind("<FocusOut>", lambda event: self.finish_step_name_edit(editor, focus_out=True))
        self.refresh_buttons()
        return "break"

    def rename_step(self, index, name):
        if self.state not in ("READY", "HOLD", "DISCONNECTED") or self.closing or self.finish_pending or self.preview_only:
            return False
        name = name.strip()
        if not 1 <= len(name) <= MAX_NAME_LENGTH or any(ord(char) < 32 or ord(char) == 127 for char in name):
            raise ValueError(f"Use a single-line step name of 1-{MAX_NAME_LENGTH} characters.")
        if self.steps[index]["name"] != name:
            self.steps[index]["name"] = name
            self.table.set(str(index), "name", name)
            self.edited()
            self.refresh_buttons()
        return True

    def finish_step_name_edit(self, editor, *, focus_out=False):
        if editor is not getattr(self, "name_editor", None):
            return "break"
        index = next((i for i, step in enumerate(self.steps) if step is self.name_edit_step), None)
        try:
            if index is not None:
                self.rename_step(index, editor.get())
        except ValueError as exc:
            self.note(str(exc))
            if not focus_out:
                editor.focus_set()
                return "break"
        self.cancel_step_name_edit()
        return "break"

    def refresh_steps(self, select=None):
        if self.preview is not None:
            self.preview.invalidate_episode()
        self.cancel_step_name_edit()
        self.table.delete(*self.table.get_children())
        for index, step in enumerate(self.steps):
            from step_reference import needs_measurement
            if index==1 and self.steps[0].get('safe_boundary') and needs_measurement(self.steps,self.vision_plan):
                self.table.insert("","end",iid="measure",values=("자동","지그 위치 읽기 · 최대 3초","측정"))
            self.table.insert("", "end", iid=str(index), values=(index + 1, step["name"], step_correction(step, self.vision_plan)[0]),
                              tags=("stripe",) if index % 2 else ())
        if select is not None and self.steps:
            self.table.selection_set(str(min(select, len(self.steps) - 1)))
        self.select()
        from jig_catalog import refresh_picker
        refresh_picker(self)

    def document(self):
        document = {"schema": 2 if self.vision_plan["targets"] else 1,
                    "identity": self.episode_identity, "units": UNITS,
                    "title": self.teaching_title, "updated_at": stamp(), "steps": self.steps}
        if document["schema"] == 2:
            document["vision_plan"] = self.vision_plan
        from robot_preview import preview_tool
        model_tool=preview_tool(self)
        if model_tool is not None:
            from copy import deepcopy
            document["model_tool"]=deepcopy(model_tool)
        return document

    def persist(self):
        if self.preview_only:
            return
        try:
            atomic_json(self.program_path, self.document())
            self.note(f"Saved: {self.program_path}")
        except Exception as exc:
            self.stop()
            self.note(f"SAVE FAILED: {exc}. Poses remain in memory; use Finish & Save.")

    def edited(self):
        self.refresh_step_corrections()
        if self.preview is not None:
            self.preview.invalidate_episode()
        self.saved_path = None
        self.library.set("")
        self.persist()

    def delete(self):
        if self.preview_only:
            return
        from tkinter import messagebox
        index = self.selected()
        if index is not None and messagebox.askyesno("스텝 삭제", f"{self.steps[index]['name']} 스텝을 삭제할까요?", parent=self.root):
            del self.steps[index]
            self.refresh_steps(index)
            self.edited()

    def add_safe_steps(self):
        existing=next((i for i,s in enumerate(self.steps) if s.get('safe_boundary')),None)
        if existing is not None:
            self.refresh_steps(existing);self.preview_selected_step()
            self.note('안전 자세가 이미 있습니다. 관절값 편집에서 조정하세요.');return
        view=self.open_vision(show=False)
        if view is not None:
            view.guarded(view.add_safe_endpoints)
            self.note(view.status.get())

    def can_reorder(self):
        return (self.state in ('DISCONNECTED','READY','HOLD') and not self.preview_only
                and not self.vision_busy and not self.finish_pending and not self.closing
                and self.pose_editor is None and self.reference_editor is None and self.name_editor is None)

    def begin_step_drag(self,event):
        self.step_drag=None
        row=self.table.identify_row(event.y)
        if self.can_reorder() and row.isdecimal():
            self.step_drag=(int(row),event.x,event.y,tuple(id(s) for s in self.steps),None)

    def drag_step(self,event):
        if self.step_drag is None:return
        source,x,y,signature,_=self.step_drag
        if abs(event.x-x)+abs(event.y-y)<6:return
        if event.y<28:self.table.yview_scroll(-1,'units')
        elif event.y>self.table.winfo_height()-20:self.table.yview_scroll(1,'units')
        row=self.table.identify_row(event.y)
        if row.isdecimal():
            self.clear_drop_mark();self.table.item(row,tags=(*self.table.item(row,'tags'),'drop-target'))
            self.table.configure(cursor='fleur');self.step_drag=(source,x,y,signature,int(row))
        return 'break'

    def clear_drop_mark(self):
        for row in self.table.get_children():
            self.table.item(row,tags=tuple(tag for tag in self.table.item(row,'tags') if tag!='drop-target'))
        self.table.configure(cursor='')

    def end_step_drag(self,event):
        drag=self.step_drag;self.step_drag=None;self.clear_drop_mark()
        if drag is None:return
        source,_,_,signature,target=drag
        row=self.table.identify_row(event.y)
        if not row.isdecimal() or int(row)!=target or not 0<=event.x<=self.table.winfo_width():return
        if target is not None and signature==tuple(id(s) for s in self.steps):
            self.reorder_to(source,target)

    def reorder_to(self,source,target):
        if not self.can_reorder() or not(0<=source<len(self.steps) and 0<=target<len(self.steps)) or source==target:return
        if self.steps[source].get('safe_boundary') or self.steps[target].get('safe_boundary'):
            self.note('시작·종료 안전 자세는 처음과 끝에 유지합니다.');return
        self.steps.insert(target,self.steps.pop(source))
        self.refresh_steps(target);self.edited();self.preview_selected_step()

    def reorder(self, direction):
        index=self.selected()
        if index is not None:self.reorder_to(index,index+direction)

    def new_program(self):
        from tkinter import messagebox
        if self.state not in ("READY", "HOLD", "DISCONNECTED"):
            return
        if self.steps and not messagebox.askokcancel("새 에피소드", "새 에피소드를 만들까요? 현재 스텝 목록은 비워지며, 초안은 자동 저장 파일에 남습니다.", parent=self.root):
            return
        if self.vision is not None:
            self.vision.cancel_execution('새 에피소드');self.vision.teaching_reference.clear()
        self.steps = []
        self.vision_plan = {"targets": {}, "calibration_signature": ""}
        self.episode_identity = self.backend.identity
        self.episode_model_tool = None
        self.preview_only = False
        self.teaching_title = ""
        self.saved_path = None
        self.program_path = self.output / ("poses_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
        self.library.set("")
        self.detail.set("")
        self.refresh_steps()
        self.refresh_buttons()

    def refresh_library(self):
        entries = []
        for path in self.output.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                identity = data.get("identity")
                if not isinstance(identity, dict) or type(identity.get("demo")) is not bool:
                    continue
                validate_program(data, identity)
                if data.get("kind") != "teaching" and not path.name.startswith("teaching_"):
                    continue
                title = data.get("title")
                if isinstance(title, str) and title.strip():
                    updated = str(data.get("updated_at", ""))
                    entries.append((path, title[:MAX_NAME_LENGTH] + " | " + updated[:19], updated))
            except (ValueError, OSError, TypeError):
                continue
        entries.sort(key=lambda item: (item[2], item[0].name), reverse=True)
        self.library_paths = [path for path, _, _ in entries]
        self.library["values"] = [label for _, label, _ in entries]
        if self.saved_path in self.library_paths:
            self.library.current(self.library_paths.index(self.saved_path))

    def choose_program(self, event=None):
        index = self.library.current()
        if 0 <= index < len(self.library_paths):
            self.load_path(self.library_paths[index])

    def load(self):
        from app_choices import askopenfilename
        path = askopenfilename(parent=self.root, initialdir=self.output, title="에피소드 열기", filetypes=[("에피소드 파일 (JSON)", "*.json")])
        if not path:
            return
        self.load_path(Path(path))

    def load_path(self, path):
        if self.state not in ("READY", "HOLD", "DISCONNECTED"):
            return
        try:
            from episode_preview import read_episode
            data = read_episode(path)
            identity = data.get("identity") if isinstance(data, dict) else None
            if not isinstance(identity, dict) or type(identity.get("demo")) is not bool:
                raise ValueError("에피소드의 로봇 정보가 올바르지 않습니다.")
            preview_only = identity != self.backend.identity
            if preview_only and self.state != "DISCONNECTED":
                raise ValueError("로봇 연결을 종료한 뒤 다른 로봇의 에피소드를 열어 주세요.")
            steps = validate_program(data, identity)
            if not preview_only:
                for step in steps:
                    self.backend.validate_target(step["pose"])
            self.steps = steps
            self.episode_identity, self.preview_only = identity, preview_only
            self.episode_model_tool = data.get("model_tool")
            self.vision_plan = data.get("vision_plan", {"targets": {}, "calibration_signature": ""})
            if self.vision is not None:
                from vision_teaching import reference_key
                targets=self.vision_plan.get('targets',{})
                active=self.vision_plan.get('teaching_jig_key')
                self.vision.target_id=next((k for k,v in targets.items() if v.get('mode')=='rectangle' and reference_key(v)==active),next(iter(targets),None))
                self.vision.refresh_target_list()
                if self.vision.target_id is not None:
                    self.vision.load_target()
                self.vision.tracker.clear()
            title = data.get("title", "")
            self.teaching_title = title if isinstance(title, str) else ""
            self.saved_path = Path(path) if self.teaching_title.strip() and steps else None
            self.program_path = self.output / ("poses_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
            self.refresh_steps(0)
            # Foreign episodes are read-only; never relabel them as this robot.
            if not self.preview_only:
                self.persist()
            self.refresh_buttons()
            if not self.preview_only and any(t.get('mode')=='rectangle' for t in self.vision_plan['targets'].values()):
                view=self.open_vision(show=False)
                if view is not None:view.schedule_teaching_reference()
            elif self.vision is not None:
                self.vision.teaching_auto_pending=False;self.vision.teaching_reference.clear()
        except Exception as exc:
            self.note(f"Open failed: {exc}")
            self.saved_path = None
            self.library.set("")
            self.refresh_library()
            self.refresh_buttons()

    def finish_teaching(self):
        if self.finish_pending or self.closing or self.preview_only:
            return
        if self.state in ("TEACH", "HOLD"):
            self.finish_pending = True
            self.refresh_buttons()
            try:
                self.session.command("finish")
            except queue.Full:
                self.finish_pending = False
                self.refresh_buttons()
                self.note("Command queue full; teaching has not stopped. Try again or use STOP.")
        elif self.state in ("READY", "DISCONNECTED"):
            self.save_as()

    def finish_save(self):
        if not self.finish_pending:
            return
        try:
            if self.state == "HOLD" and not self.closing and not self.session.stop.is_set():
                if self.steps:
                    self.save_as()
                else:
                    self.note("Teaching paused. No steps to save; use Add step first.")
        finally:
            self.finish_pending = False
            self.refresh_buttons()

    def rename_episode(self):
        if not self.can_reorder():return
        title=ask_teaching_title(self.root,initialvalue=self.teaching_title)
        if title is None:return
        title=title.strip()
        if not 1<=len(title)<=MAX_NAME_LENGTH:
            self.note(f'에피소드 이름은 1~{MAX_NAME_LENGTH}자로 입력하세요.');return
        if title==self.teaching_title:return
        try:
            if self.saved_path is not None:
                data=json.loads(self.saved_path.read_text(encoding='utf-8'))
                validate_program(data,self.episode_identity)
                atomic_json(self.saved_path,{**data,'title':title,'updated_at':stamp()})
            self.teaching_title=title
            self.persist();self.refresh_library();self.refresh_buttons()
            self.note('에피소드 이름 변경: '+title)
        except Exception as exc:self.note('이름 변경 실패: '+str(exc))

    def save_as(self):
        if self.preview_only or not self.steps or self.state not in ("READY", "HOLD", "DISCONNECTED"):
            return
        title = ask_teaching_title(self.root, initialvalue=self.teaching_title)
        if title is None:
            return
        if not 1 <= len(title.strip()) <= MAX_NAME_LENGTH:
            self.note(f"Use a 1-{MAX_NAME_LENGTH} character teaching title.")
            return
        try:
            data = {**self.document(), "title": title.strip()}
            validate_program(data, self.backend.identity)
            path = save_named_teaching(self.output, data)
        except Exception as exc:
            self.note(f"SAVE FAILED: {exc}")
            return
        self.teaching_title = title.strip()
        self.saved_path = path
        self.persist()
        self.refresh_library()
        self.refresh_buttons()
        self.note(f"Teaching saved: {path}")

    def play(self):
        from tkinter import messagebox
        if self.preview_only or self.state != "HOLD" or self.saved_path is None or self.latest is None:
            return
        if any(step.get("motion") for step in self.steps):
            view = self.open_vision(show=False)
            if view is not None:
                if not view.frame_live:view.select_tab(3)
                view.guarded(view.execute)
            return
        if time.monotonic() - self.latest["sample_time"] > DISPLAY_STALE_SECONDS:
            self.note("Position display is stale; wait for a fresh sample.")
            return
        try:
            data = json.loads(self.saved_path.read_text(encoding="utf-8"))
            steps = validate_program(data, self.backend.identity)
            current = self.latest["follower"].copy()
            validate_sequence(self.backend, steps, current)
        except (ValueError, OSError, TypeError, KeyError) as exc:
            self.note(f"Playback blocked: {exc}")
            return
        if messagebox.askokcancel("에피소드 실행", f"{data.get('title', '')}\n{len(steps)}개 스텝을 한 번 실행할까요?", parent=self.root):
            if self.state == "HOLD" and not self.session.stop.is_set():
                self.send("play", {"steps": steps, "title": data.get("title", ""), "follower": current})

    def poll(self):
        self.session.heartbeat = time.monotonic()
        finish_ready = False
        while True:
            try:
                kind, value = self.session.events.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                self.latest = value
                if self.state != "STOPPING":
                    self.state = value["state"]
                for joint in JOINTS:
                    key = joint + ".pos"
                    from motor_positions import to_ticks
                    for role in ('leader','follower'):
                        actual=value.get(role)
                        display='—'
                        if actual is not None:
                            raw=value.get('motor_raw',{}).get(role,{}).get(key)
                            if key!='gripper.pos' and raw is None:raw=to_ticks(self.backend,key,actual[key],role)
                            display=f"{actual[key]:.2f}" if key=='gripper.pos' else str(raw)
                        self.joints.set(joint,role,display)
            elif kind == "play_result":
                if self.vision is not None and not self.vision.closed:
                    self.vision.safe_execution.completed(value)
            elif kind == 'jog_result':
                if self.vision is not None and not self.vision.closed:
                    self.vision.manual.jog_completed(value)
                self.note(value['message'])
            elif kind == "manual_result":
                if self.vision is not None and not self.vision.closed:
                    self.vision.manual.completed(value)
            elif kind == "captured":
                if len(self.steps) >= MAX_STEPS:
                    self.note(f"Maximum {MAX_STEPS} steps.")
                    continue
                try:self.store_captured_step(value,value.pop('_teaching_context',None))
                except ValueError as exc:self.note('스텝 저장 실패 · '+str(exc));continue
                self.note(f"{value['name']} 스텝 추가 완료" + (" · 티칭 계속 중, 스페이스로 다음 스텝 추가" if self.state == "TEACH" else ""))
            elif kind == "closed":
                self.finish_pending = False
                self.state = "DISCONNECTED"
                self.cleanup_fault = bool(value)
                for joint in JOINTS:
                    self.joints.set(joint, "leader", "—")
                    self.joints.set(joint, "follower", "—")
                self.refresh_guidance()
                if value:
                    self.note("CUT MOTOR POWER: " + "; ".join(value))
                elif self.backend.identity["demo"]:
                    self.note("DEMO disconnected; no hardware accessed")
                elif self.backend.motion_attempted:
                    self.note("Torque OFF verified / ports closed")
                else:
                    self.note("Ports closed after read-only checks; motor torque was not changed")
            elif kind == "finish_ready":
                finish_ready = self.finish_pending
            else:
                if kind == "error":
                    self.finish_pending = False
                self.note(str(value))
        self.refresh_buttons()
        if self.closing and not (self.session.thread and self.session.thread.is_alive()):
            self.root.destroy()
            return
        self.root.after(GUI_POLL_MS, self.poll)
        if finish_ready and self.finish_pending:
            # Keep heartbeat polling alive inside the title dialog's nested event loop.
            self.root.after_idle(self.finish_save)

    def show_preview(self):
        if self.closing:
            return
        if self.preview is None:
            from robot_preview import PreviewPanel
            self.preview = PreviewPanel(self, self.workspace)
        self.layout_workspace()

    def simulate(self):
        self.show_preview()
        if self.preview is not None:
            if self.preview.simulation_active:
                self.preview.stop_simulation()
            else:
                self.preview.play_simulation()
                if self.preview.simulation_active:
                    self.clear_step_selection()
        self.refresh_buttons()

    def layout_workspace(self):
        # Reference layout: episode controls above the two left-hand tables;
        # the 3D preview spans the entire right side. No tabs hide live values.
        self.left_workspace.grid_propagate(self.preview is None)
        left_width = max(690, (self.root.winfo_width()-28)//2)
        self.left_workspace.columnconfigure(0,minsize=400)
        self.left_workspace.columnconfigure(1,minsize=270)
        self.workspace.columnconfigure(0, weight=0 if self.preview is not None else 1, minsize=left_width if self.preview is not None else 0)
        self.workspace.columnconfigure(1, weight=1 if self.preview is not None else 0, minsize=340 if self.preview is not None else 0)
        layout = "integrated" if self.preview is not None else "controls"
        if layout == self.workspace_layout:
            return
        self.workspace_layout = layout
        self.left_workspace.grid(row=0, column=0, sticky="nsew", padx=(0, 10) if self.preview is not None else 0)
        self.joints_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        self.steps_panel.grid(row=1, column=1, sticky="nsew")
        if self.preview is not None:
            self.preview.frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.workspace.columnconfigure(0, weight=1)

    def close(self):
        calibration = getattr(self.vision, 'calibration_window', None) if self.vision else None
        if calibration and calibration.worker and not calibration.worker.finished.is_set():
            calibration.close()
            self.root.after(100, self.close)
            return
        self.closing = True
        self.stop()
        if self.motor_dashboard is not None:
            self.motor_dashboard.dispose()
        if self.vision is not None:
            self.vision.close()
        if self.preview is not None:
            self.preview.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--follower-only", action="store_true", help="Real SO-101 follower only; no leader is constructed")
    parser.add_argument("--setup", action="store_true", help="Open follower registration and calibration-file selection")
    parser.add_argument("--demo", action="store_true", help="Simulated joints; never open serial ports")
    parser.add_argument("--preview", action="store_true", help="Compatibility option; the integrated 3D preview is always shown")
    parser.add_argument("--profile", type=Path, help="Robot-pair JSON for another SO-101 pair")
    parser.add_argument("--check-only", action="store_true", help="Validate files and port ownership without opening ports")
    parser.add_argument("--diagnose-wrist-temperature", action="store_true",
                        help="Record ID 4 block/single temperature during normal teaching; keep safety stops")
    parser.add_argument("--diagnose-gripper", choices=("empty", "object_present"),
                        help="Record ID 6 telemetry; context label only, not grasp control or force limiting")
    parser.add_argument("--motor-dashboard", action="store_true", help="Open all-motor diagnostics on the existing teaching bus")
    parser.add_argument("--output", type=Path, default=ROOT / Path(OUTPUT_DIRECTORY).expanduser())
    parser.add_argument('--choose-mode',action='store_true',help='실행 모드와 로봇 프로필 선택')
    args = parser.parse_args(argv)
    if args.choose_mode and not args.demo and not args.check_only:
        from launch_mode import choose_mode
        selected=choose_mode(ROOT)
        if selected is None:return 0
        args.follower_only,args.profile=selected
    if args.demo and (args.diagnose_wrist_temperature or args.diagnose_gripper):
        parser.error("Motor telemetry diagnostics require hardware; omit --demo")
    if args.follower_only and args.profile is None:
        args.profile = ROOT / "follower.json"
    if args.setup or (args.follower_only and not args.demo and not args.check_only and not args.profile.exists()):
        from follower_setup import choose_follower_profile
        args.profile = choose_follower_profile(args.profile or ROOT / "follower.json")
        if args.profile is None:return 0
        args.follower_only = True
    if args.demo:
        backend = Demo()
        backend.follower_only = args.follower_only
    else:
        try:
            backend = Hardware(args.profile, follower_only=args.follower_only)
        except (ValueError, OSError) as exc:
            if not args.follower_only or args.check_only:raise
            from follower_setup import choose_follower_profile
            args.profile = choose_follower_profile(args.profile, error=str(exc))
            if args.profile is None:return 0
            backend = Hardware(args.profile, follower_only=True)
    if args.check_only:
        print(json.dumps({"identity": backend.identity, "ports": backend.preflight()}, indent=2))
        print("Preflight OK. No serial ports opened; no motor writes.")
        return 0
    if args.diagnose_wrist_temperature:
        from wrist_temperature_diagnostic import WristTemperatureDiagnostic
        backend.temperature_diagnostic = WristTemperatureDiagnostic(
            backend.follower.bus, args.output / "temperature-checks")
    if args.diagnose_gripper:
        from gripper_grasp_diagnostic import GripperGraspDiagnostic
        backend.gripper_diagnostic = GripperGraspDiagnostic(
            backend.follower.bus, args.output / "gripper-checks",
            sample_kind=args.diagnose_gripper, identity=backend.identity)
    import tkinter as tk
    root = tk.Tk(className="OpenCV-teach")
    app = App(root, backend, args.output)
    app.show_preview()
    root.after_idle(app.connect_startup_camera)
    if args.motor_dashboard:
        app.open_monitor()
    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, lambda *_: app.close())
    try:
        root.mainloop()
    finally:
        if app.preview is not None and app.preview.client is not None:
            app.preview.client.close()
        app.session.stop.set()
        if app.session.thread:
            app.session.thread.join()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
