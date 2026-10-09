"""Register and calibrate another SO-101 pair for the portable teaching GUI."""

import argparse
import json
from pathlib import Path
import sys

from teaching_profile import load_profile

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=ROOT / "robot.json")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ports", action="store_true", help="List USB port names without opening them")
    mode.add_argument("--register", action="store_true", help="Create a new profile; never overwrite")
    mode.add_argument("--calibrate", choices=("leader", "follower"))
    mode.add_argument("--setup-motors", choices=("leader", "follower"), help="Assign IDs with one motor connected at a time")
    mode.add_argument("--compare", action="store_true", help="Read and compare both calibrations")
    parser.add_argument("--pair-name")
    parser.add_argument("--follower-only", action="store_true")
    parser.add_argument("--follower-id", default="follower_left")
    parser.add_argument("--side", choices=("left","right"), default="left")
    parser.add_argument("--leader-port")
    parser.add_argument("--follower-port")
    options = parser.parse_args(argv)
    if options.ports:
        for port in sorted(Path("/dev/serial/by-id").glob("*")):
            print(f"{port} -> {port.resolve()}")
        return 0
    profile = options.profile.expanduser().resolve()
    if options.register:
        data = dict(schema=1, robot_type="so101", pair_name=options.pair_name,
                    leader_port=options.leader_port, follower_port=options.follower_port)
        if options.follower_only:
            data.update(mode="follower_only",follower_id=options.follower_id,side=options.side)
            data.pop("leader_port",None)
        # Validate before creating the persistent profile.
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            check = Path(directory) / "robot.json"
            check.write_text(json.dumps(data), encoding="utf-8")
            load_profile(check)
        with profile.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")
        print(f"Registered: {profile}. No serial ports opened.")
        return 0
    args, identity = load_profile(profile)
    from importlib.metadata import version
    if version("lerobot") not in ("0.4.4","0.6.1"):
        raise RuntimeError("LeRobot 0.4.4 / 0.6.1 is required")
    from diagnose_shoulder_lift import ensure_ports_available, make_arms, compare_calibration, protect_port
    if options.compare:
        ensure_ports_available([args.follower_port] if args.leader_port is None else [args.leader_port,args.follower_port])
        from diagnose_shoulder_lift import make_follower
        arms = (make_follower(args,side=args.side),) if args.leader_port is None else make_arms(args, side=args.side)
        matched = True
        try:
            for arm in arms:
                arm.bus.connect()
                matched = compare_calibration(arm)["matches"] and matched
        finally:
            for arm in arms:
                arm.bus.port_handler.closePort()
        return 0 if matched else 1
    role = options.calibrate or options.setup_motors
    if role=="leader" and args.leader_port is None:raise ValueError("팔로워 단독 프로필에는 리더암이 없습니다.")
    port = getattr(args, role + "_port")
    ensure_ports_available([port])
    args.calibration_dir.mkdir(parents=True, exist_ok=True)
    if role == "leader":
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
        arm = SO101Leader(SO101LeaderConfig(port=port, id="leader_left",
                          calibration_dir=args.calibration_dir, use_degrees=True))
    else:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        arm = SO101Follower(SO101FollowerConfig(port=port, id=args.follower_id,
                            calibration_dir=args.calibration_dir, use_degrees=True, cameras={}))
    protect_port(arm.bus)
    if options.setup_motors:
        print("ID setup writes motor IDs/baud rates. Connect only the motor requested by each prompt.")
        try:
            arm.setup_motors()
        finally:
            arm.bus.port_handler.closePort()
        return 0
    print(f"Calibration: {identity['pair_name']} / {role} / {port}")
    print("Support the arm. Calibration disables torque and writes offsets/ranges.")
    # Enter calibration directly, without configure() or any goal-position writes.
    try:
        arm.bus.connect()
        arm.calibrate()
    finally:
        try:
            if arm.bus.is_connected:
                arm.bus.disconnect()
        finally:
            arm.bus.port_handler.closePort()
    print(f"Calibration saved: {arm.calibration_fpath}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
