"""Compare block and single-register temperature READs; no motor writes."""

import argparse
import csv
from datetime import datetime
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

from diagnose_shoulder_lift import ensure_ports_available, now_iso, protect_port


from teaching_settings import (TEMPERATURE_DIFFERENCE_WARNING, MONITOR_SECONDS, MONITOR_HZ,
                               MONITOR_MOTOR_ID, DIAGNOSTIC_LOG_DIRECTORY)

FIELDS = (
    "timestamp_iso", "elapsed_s", "read_order", "temperature_block_raw",
    "temperature_single_raw", "temperature_raw_difference", "voltage_raw",
    "status_raw", "torque_enable_raw", "current_raw", "block_comm", "single_comm",
    "block_packet_error_raw", "single_packet_error_raw", "block_hex", "single_hex",
    "block_ms", "single_ms", "sample_ms", "read_error", "block_rx_hex", "single_rx_hex",
)
ALLOWED_READS = {(3, 2), (40, 31), (63, 1)}


def annotate_temperature_difference(row, first_key, second_key, motor_id):
    """Keep discrepant readings visible without treating the difference as an alarm."""
    row.pop("temperature_warning", None)
    first, second = row.get(first_key), row.get(second_key)
    if isinstance(first, int) and isinstance(second, int) and abs(first - second) > TEMPERATURE_DIFFERENCE_WARNING:
        row["temperature_warning"] = (
            f"ID{motor_id} temperature reads disagree: [{first}, {second}]; "
            "difference is warning-only; individual temperature limits remain active")


def restrict_packets(bus, motor_id=2):
    original = bus.port_handler.writePort

    def read_only(packet):
        if (len(packet) != 8 or packet[2] != motor_id or packet[4] != 2
                or (packet[5], packet[6]) not in ALLOWED_READS):
            raise RuntimeError(f"Temperature monitor permits only whitelisted ID {motor_id} READ packets")
        return original(packet)

    bus.port_handler.writePort = read_only


def validate_responses(bus):
    original = bus.packet_handler.txRxPacket
    if getattr(original, "_temperature_validated", False):
        return

    def checked(port, request):
        response, comm, error = original(port, request)
        # Broadcast writes legitimately return no status packet in this SDK.
        bus.temperature_rx_hex = "" if response is None else bytes(response).hex()
        if comm == 0 and request[4] == 2:
            length = request[6]
            if (response is None or len(response) != length + 6 or response[:2] != [255, 255]
                    or response[2] != request[2] or response[3] != length + 2
                    or (sum(response[2:]) & 255) != 255):
                raise ConnectionError("Invalid READ response framing, ID, length or checksum")
        return response, comm, error

    checked._temperature_validated = True
    bus.packet_handler.txRxPacket = checked


def read_packet(bus, address, length, motor_id=2):
    start = time.monotonic()
    result = {}
    bus.temperature_rx_hex = ""
    try:
        data, comm, error = bus.packet_handler.readTxRx(bus.port_handler, motor_id, address, length)
        result.update(comm=comm, packet_error_raw=error, hex=bytes(data).hex())
        if comm != 0 or len(data) != length:
            raise ConnectionError(f"comm={comm}, payload={len(data)}/{length}")
        result["data"] = data
        if error:
            result["error"] = f"motor packet error={error}; valid payload retained"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["rx_hex"] = bus.temperature_rx_hex
    result["ms"] = (time.monotonic() - start) * 1000
    return result


def read_pair(bus, index, motor_id=2):
    # Alternate order to avoid always reading one method later than the other.
    order = ("block", "single") if index % 2 == 0 else ("single", "block")
    row = {"read_order": ",".join(order)}
    errors = []
    for mode in order:
        address, length = (40, 31) if mode == "block" else (63, 1)
        result = read_packet(bus, address, length, motor_id)
        for key in ("comm", "packet_error_raw", "hex", "ms", "rx_hex"):
            if key in result:
                row[f"{mode}_{key}"] = result[key]
        if "error" in result:
            errors.append(f"{mode}: {result['error']}")
        if "data" in result:
            data = result["data"]
            row[f"temperature_{mode}_raw"] = data[63 - address]
            if mode == "block":
                row.update(voltage_raw=data[62 - address], status_raw=data[65 - address],
                           torque_enable_raw=data[0], current_raw=int.from_bytes(bytes(data[29:31]), "little"))
    if "temperature_block_raw" in row and "temperature_single_raw" in row:
        row["temperature_raw_difference"] = row["temperature_block_raw"] - row["temperature_single_raw"]
    row["read_error"] = "; ".join(errors)
    return row


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=os.environ.get("SO101_FOLLOWER_RIGHT_PORT"))
    parser.add_argument("--motor-id", type=int, choices=range(1, 7), default=MONITOR_MOTOR_ID)
    parser.add_argument("--seconds", type=float, default=MONITOR_SECONDS)
    parser.add_argument("--hz", type=float, default=MONITOR_HZ)
    parser.add_argument("--log-dir", type=Path, default=Path(DIAGNOSTIC_LOG_DIRECTORY).expanduser())
    args = parser.parse_args()
    if not args.port or not args.port.strip():
        parser.error("Port is empty; source right_arm_env.sh first")
    if any(not math.isfinite(x) or x <= 0 for x in (args.seconds, args.hz)) or args.hz > 10:
        parser.error("Require finite seconds > 0 and 0 < hz <= 10")
    args.log_dir = args.log_dir.expanduser()
    return args


def run(bus, args, resolved_port):
    motor_id = getattr(args, "motor_id", 2)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    prefix = "shoulder_temperature_" if motor_id == 2 else f"id{motor_id}_temperature_"
    stem = prefix + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    csv_path = args.log_dir / (stem + ".csv")
    json_path = args.log_dir / (stem + ".json")
    report = {"started_at": now_iso(), "port": args.port, "resolved_port": resolved_port,
              "seconds": args.seconds, "hz": args.hz, "mode": "read_only_temperature_comparison",
              "units": "raw; physical temperature scale unverified", "samples": 0,
              "cleanup_errors": [], "motor_id": motor_id}
    status = 0
    stream = None

    def stop(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        stream = csv_path.open("x", newline="", encoding="utf-8")
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        stream.flush()
        bus.connect(handshake=False)
        model = read_packet(bus, 3, 2, motor_id)
        report["model_read"] = model
        if "data" not in model or int.from_bytes(bytes(model["data"]), "little") != 777:
            raise RuntimeError(f"ID {motor_id} did not report expected STS3215 model 777")
        print(f"CSV: {csv_path}\nNo goal, configure, torque or calibration writes.", flush=True)
        print("  time_s  block_raw single_raw difference voltage_raw status_raw torque_raw", flush=True)
        start = time.monotonic()
        while time.monotonic() - start < args.seconds:
            sample_start = time.monotonic()
            row = {"timestamp_iso": now_iso(), "elapsed_s": sample_start - start}
            row.update(read_pair(bus, report["samples"], motor_id))
            row["sample_ms"] = (time.monotonic() - sample_start) * 1000
            writer.writerow(row)
            stream.flush()
            report["samples"] += 1
            keys = ("temperature_block_raw", "temperature_single_raw", "temperature_raw_difference",
                    "voltage_raw", "status_raw", "torque_enable_raw")
            print(f"{row['elapsed_s']:8.2f} " + " ".join(f"{str(row.get(k, '-')):>10}" for k in keys), flush=True)
            if row["read_error"]:
                print(row["read_error"], flush=True)
            if row.get("torque_enable_raw") not in (None, 0):
                raise RuntimeError("Torque is not OFF; stop this inspection and cut motor power")
            remaining = 1 / args.hz - (time.monotonic() - sample_start)
            if remaining > 0:
                time.sleep(remaining)
        report["outcome"] = "time_limit"
    except KeyboardInterrupt:
        report["outcome"] = "interrupted"
        status = 130
    except Exception as exc:
        report["outcome"] = f"{type(exc).__name__}: {exc}"
        print(report["outcome"], file=sys.stderr)
        status = 1
    finally:
        try:
            if stream is not None:
                try:
                    stream.flush()
                finally:
                    stream.close()
        except BaseException as exc:
            report["cleanup_errors"].append(f"CSV close: {exc}")
        finally:
            try:
                # bus.disconnect() disables torque; close only the host serial port.
                bus.port_handler.closePort()
            except BaseException as exc:
                report["cleanup_errors"].append(f"Serial close: {exc}")
            signal.signal(signal.SIGTERM, previous)
        report["finished_at"] = now_iso()
        json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if report["cleanup_errors"]:
            status = 1
            print(report["cleanup_errors"], file=sys.stderr)
    print(f"Saved: {csv_path}\nSettings/outcome: {json_path}", flush=True)
    return status


def main():
    args = parse_args()
    if version("lerobot") != "0.6.1":
        raise RuntimeError("This tool requires installed LeRobot 0.6.1")
    resolved = ensure_ports_available([args.port])[0]
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    bus = FeetechMotorsBus(args.port, {names[args.motor_id - 1]: Motor(args.motor_id, "sts3215", MotorNormMode.DEGREES)})
    protect_port(bus)
    restrict_packets(bus, args.motor_id)
    validate_responses(bus)
    return run(bus, args, resolved)


if __name__ == "__main__":
    sys.exit(main())
