"""Validated settings shared by all Ubuntu teaching tools; edit settings.toml."""
from pathlib import Path
import math
import re
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent
SCHEMA = {'CONTROL_HZ': ('response', 'int', 10, 200),
 'GUI_POLL_MS': ('response', 'int', 5, 200),
 'HEALTH_INTERVAL': ('response', 'float', 0.05, 0.5),
 'GRIPPER_HEALTH_INTERVAL': ('response', 'float', 0.05, 0.5),
 'GUI_HEARTBEAT_TIMEOUT': ('response', 'float', 0.5, 2.0),
 'DISPLAY_STALE_SECONDS': ('response', 'float', 0.05, 1.0),
 'COMMAND_QUEUE_SIZE': ('response', 'int', 1, 100),
 'COMMAND_RATE': ('motion', 'float', 0.1, 30.0),
 'GRIPPER_COMMAND_RATE': ('motion', 'float', 0.1, 40.0),
 'SYNC_RATE': ('motion', 'float', 0.1, 5.0),
 'MAX_COMMAND_DT': ('motion', 'float', 0.001, 0.05),
 'RELATIVE_TARGET_LIMIT': ('motion', 'float', 0.1, 30.0),
 'SYNC_TIMEOUT': ('motion', 'float', 1, 300),
 'MOVE_TIMEOUT': ('motion', 'float', 1, 300),
 'ARRIVAL_SETTLE_SECONDS': ('motion', 'float', 0.1, 5),
 'SYNC_ARRIVAL_TOLERANCE': ('motion', 'float', 0.1, 2),
 'PLAY_GRIPPER_ARRIVAL_TOLERANCE': ('motion', 'float', 0.1, 10),
 'PREVIEW_POSE_TOLERANCE': ('motion', 'float', 0.1, 2),
 'WRIST_MOVE_LIMIT': ('motion', 'float', 1, 90),
 'TEACH_START_GAP': ('motion', 'float', 0.1, 10),
 'SYNC_LEADER_DRIFT': ('motion', 'float', 0.1, 5),
 'ENTRY_MARGIN_DEG': ('safety', 'float', 0, 5),
 'GRIPPER_ENTRY_MAX_RAW': ('safety', 'int', 0, 10),
 'GRIPPER_ENTRY_MARGIN_UNITS': ('safety', 'float', 0, 2),
 'ACTIVATION_READ_RETRIES': ('safety', 'int', 0, 2),
 'STALL_SECONDS': ('safety', 'float', 0.1, 4),
 'TRACKING_ERROR': ('safety', 'float', 0.1, 8),
 'TRACKING_SENT_ERROR': ('safety', 'float', 0.1, 2),
 'TRACKING_PROGRESS_TICKS': ('safety', 'int', 1, 10),
 'TEMPERATURE_LIMIT': ('safety', 'int', 1, 65),
 'TEMPERATURE_DURATION_S': ('safety', 'float', 0.1, 5),
 'TEMPERATURE_MAX_GAP_S': ('safety', 'float', 0.5, 1.5),
 'COOL_TEMPERATURE_LIMIT': ('safety', 'int', 1, 40),
 'TEMPERATURE_DIFFERENCE_WARNING': ('safety', 'int', 0, 10),
 'CAPTURE_HISTORY_SECONDS': ('capture', 'float', 0.1, 5),
 'CAPTURE_STABLE_SECONDS': ('capture', 'float', 0.1, 5),
 'CAPTURE_POSITION_TOLERANCE': ('capture', 'float', 0.01, 1),
 'CAPTURE_TARGET_TOLERANCE': ('capture', 'float', 0.01, 2),
 'MAX_STEPS': ('capture', 'int', 1, 5000),
 'MAX_NAME_LENGTH': ('capture', 'int', 1, 200),
 'TEMPERATURE_LOG_INTERVAL': ('logging', 'float', 0.1, 60),
 'TELEMETRY_LOG_INTERVAL': ('logging', 'float', 0.01, 5),
 'LOG_FLUSH_INTERVAL': ('logging', 'float', 0.05, 5),
 'PLAYBACK_LOG_INTERVAL': ('logging', 'float', 0.1, 10),
 'MOTOR_LOG_INTERVAL': ('logging', 'float', 0.5, 10),
 'CAPTURE_SECONDS': ('logging', 'float', 0.1, 30),
 'CAPTURE_ROUNDS': ('logging', 'int', 1, 100),
 'CAPTURE_INTERVAL': ('logging', 'float', 0.01, 5),
 'DASHBOARD_POLL_MS': ('dashboard', 'int', 16, 1000),
 'DASHBOARD_HISTORY_SIZE': ('dashboard', 'int', 10, 5000),
 'DASHBOARD_EVENT_SIZE': ('dashboard', 'int', 10, 1000),
 'DASHBOARD_QUEUE_SIZE': ('dashboard', 'int', 1, 1000),
 'DASHBOARD_STALE_SECONDS': ('dashboard', 'float', 0.5, 10),
 'READONLY_INTERVAL': ('dashboard', 'float', 0.1, 5),
 'DIAGNOSTIC_SECONDS': ('diagnostic_cli', 'float', 0.1, 3600),
 'DIAGNOSTIC_FPS': ('diagnostic_cli', 'float', 1, 100),
 'DIAGNOSTIC_TELEMETRY_HZ': ('diagnostic_cli', 'float', 0.1, 20),
 'DIAGNOSTIC_MAX_ERRORS': ('diagnostic_cli', 'int', 1, 3),
 'DIAGNOSTIC_RELATIVE_TARGET': ('diagnostic_cli', 'float', 5, 7),
 'MONITOR_SECONDS': ('diagnostic_cli', 'float', 0.1, 3600),
 'MONITOR_HZ': ('diagnostic_cli', 'float', 0.1, 20),
 'MONITOR_MOTOR_ID': ('diagnostic_cli', 'int', 1, 6),
 'ACTIVATION_MOVEMENT_LIMIT': ('safety', 'float', 0.1, 5),
 'DEMO_RATE': ('response', 'float', 0.1, 100),
 'HEALTH_HISTORY_SIZE': ('logging', 'int', 1, 1000)}

def load_settings(path=ROOT / "settings.toml"):
    with Path(path).open("rb") as stream:
        data = tomllib.load(stream)
    allowed = {}
    result = {}
    for name, (section, kind, low, high) in SCHEMA.items():
        allowed.setdefault(section, set()).add(name)
        value = data.get(section, {}).get(name)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or (kind == "int" and not isinstance(value, int))
                or not math.isfinite(value) or not low <= value <= high):
            raise ValueError(f"settings.toml [{section}] {name}: required {kind} {low}..{high}; got {value!r}")
        result[name] = value
    tolerances = data.get("arrival_tolerances", {})
    maxima = dict(zip((j + ".pos" for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")), (2,4,3,2,2,4)))
    allowed["arrival_tolerances"] = set(maxima)
    for key, maximum in maxima.items():
        value = tolerances.get(key)
        if isinstance(value, bool) or not isinstance(value, (int,float)) or not 0 < value <= maximum:
            raise ValueError(f"settings.toml arrival_tolerances {key}: required 0 < value <= {maximum}")
    result["REPLAY_ARRIVAL_TOLERANCES"] = dict(tolerances)
    for section, names in {"paths": ("OUTPUT_DIRECTORY", "DIAGNOSTIC_LOG_DIRECTORY"),
                           "appearance": ("GUI_GEOMETRY", "DASHBOARD_GEOMETRY")}.items():
        allowed[section] = set(names)
        for name in names:
            value = data.get(section, {}).get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"settings.toml {name}: nonempty string required")
            if section == "appearance" and not re.fullmatch(r"[1-9][0-9]{2,3}x[1-9][0-9]{2,3}", value):
                raise ValueError(f"settings.toml {name}: use WIDTHxHEIGHT")
            result[name] = value
    for section, values in data.items():
        if section not in allowed or not isinstance(values, dict) or set(values) - allowed[section]:
            raise ValueError(f"settings.toml: unknown section/key in {section}")
    if result["CAPTURE_STABLE_SECONDS"] > result["CAPTURE_HISTORY_SECONDS"]:
        raise ValueError("CAPTURE_STABLE_SECONDS must be <= CAPTURE_HISTORY_SECONDS")
    if result["GUI_POLL_MS"] / 1000 >= result["GUI_HEARTBEAT_TIMEOUT"]:
        raise ValueError("GUI_POLL_MS must be shorter than GUI_HEARTBEAT_TIMEOUT")
    if result["DIAGNOSTIC_RELATIVE_TARGET"] not in (5, 7):
        raise ValueError("DIAGNOSTIC_RELATIVE_TARGET must be 5 or 7")
    if result["TEMPERATURE_MAX_GAP_S"] <= max(result["HEALTH_INTERVAL"], result["GRIPPER_HEALTH_INTERVAL"]):
        raise ValueError("TEMPERATURE_MAX_GAP_S must exceed health intervals")
    if result["COOL_TEMPERATURE_LIMIT"] > result["TEMPERATURE_LIMIT"]:
        raise ValueError("COOL_TEMPERATURE_LIMIT must not exceed TEMPERATURE_LIMIT")
    return result

try:
    globals().update(load_settings())
except (OSError, ValueError) as exc:
    raise ValueError(f"Cannot load {ROOT / 'settings.toml'}: {exc}") from exc
