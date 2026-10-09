"""Portable robot-pair configuration, without opening hardware."""

import json
from pathlib import Path
import re
from types import SimpleNamespace
from teaching_settings import DIAGNOSTIC_RELATIVE_TARGET


def load_profile(path, follower_only=False):
    path = Path(path).expanduser().resolve(strict=True)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != 1 or data.get("robot_type") != "so101":
        raise ValueError("Profile requires schema=1 and robot_type=so101")
    name = data.get("pair_name", "")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise ValueError("pair_name must be 1-64 ASCII letters, digits, underscores or hyphens")
    follower_only = follower_only or data.get("mode") == "follower_only"
    ports = {}
    for role in (("follower",) if follower_only else ("leader", "follower")):
        value = data.get(role + "_port", "")
        if not isinstance(value, str) or Path(value).parent != Path("/dev/serial/by-id") or not value.strip():
            raise ValueError(f"{role}_port must be a persistent /dev/serial/by-id/... path")
        ports[role] = value
    if not follower_only and ports["leader"] == ports["follower"]:
        raise ValueError("Leader and follower ports must be different")
    follower_id = data.get("follower_id", "follower_left")
    if not isinstance(follower_id,str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}",follower_id):
        raise ValueError("Invalid follower calibration ID")
    side = data.get("side", "left")
    if side not in ("left","right"):raise ValueError("side must be left or right")
    calibration = path.parent / "calibration" / name
    args = SimpleNamespace(follower_id=follower_id, side=side, leader_port=ports.get("leader"), follower_port=ports["follower"],
                           calibration_dir=calibration, shoulder_max_relative_target=DIAGNOSTIC_RELATIVE_TARGET)
    identity = {"side": side, "pair_name": name, "robot_type": "so101",
                "serials": {role: Path(port).name for role, port in ports.items()},
                "demo": False, "calibration_sha256": {}}
    if follower_only:
        identity["control_mode"] = "follower_only"
        identity["follower_id"] = follower_id
    return args, identity
