#!/usr/bin/env python3
"""Read-only Pi inventory. Never opens robot/camera devices or sends motor commands."""
import grp
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import shutil
import sys


def read(path):
    try:
        return Path(path).read_text().strip().strip('\x00')
    except (OSError, UnicodeError):
        return None


def main():
    memory = {}
    for line in (read('/proc/meminfo') or '').splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable', 'SwapTotal'):
            memory[key + '_MiB'] = round(int(value.split()[0]) / 1024)
    packages = {}
    for name in ('torch', 'torchvision', 'lerobot', 'pyserial', 'grpcio', 'pyrealsense2'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    serial = []
    for path in sorted(Path('/dev/serial/by-id').glob('*')):
        serial.append({'path': str(path), 'device': str(path.resolve()),
                       'read_write_permission': os.access(path, os.R_OK | os.W_OK)})
    usb = []
    for device in sorted(Path('/sys/bus/usb/devices').glob('*')):
        vendor, product = read(device / 'idVendor'), read(device / 'idProduct')
        if vendor and product:
            usb.append({'vendor_id': vendor, 'product_id': product,
                        'name': read(device / 'product'), 'speed_Mbps': read(device / 'speed')})
    thermal = []
    for zone in sorted(Path('/sys/class/thermal').glob('thermal_zone*')):
        value = read(zone / 'temp')
        try:
            temperature = float(value) / 1000
        except (TypeError, ValueError):
            continue
        thermal.append({'sensor': read(zone / 'type'), 'temperature_C': temperature})
    groups = []
    for gid in set(os.getgroups() + [os.getgid()]):
        try:
            groups.append(grp.getgrgid(gid).gr_name)
        except KeyError:
            groups.append(str(gid))
    report = {
        'scope': 'Inventory only; no motor communication, camera capture or inference performed',
        'board': read('/proc/device-tree/model'),
        'architecture': platform.machine(), 'os_release': read('/etc/os-release'),
        'python': platform.python_version(), 'python_executable': sys.executable,
        'cpu_count': os.cpu_count(), 'memory': memory,
        'free_disk_GiB': round(shutil.disk_usage(Path.cwd()).free / 1024**3, 1),
        'groups': sorted(groups), 'serial_devices': serial,
        'video_devices': [{'path': str(p), 'name': read(p / 'name')}
                          for p in sorted(Path('/sys/class/video4linux').glob('video*'))],
        'usb_devices': usb, 'thermal': thermal, 'installed_packages': packages,
        'notes': ['Package metadata does not prove imports or inference work.',
                  'USB serial enumeration does not identify leader versus follower.',
                  'Missing packages/devices are expected before setup; this is not a pass/fail report.'],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
