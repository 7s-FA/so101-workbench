#!/usr/bin/env python3
"""Stop this project's OMX USB workers, then release and verify all six motors."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from teach_io.bus import Bus

IO_PYTHON = ROOT.parent / 'Sim_real/.venv-io/bin/python'
WORKERS = {'teach_io.worker', 'teach_io.observe_worker'}


def matches_worker(pid, port, proc_root=Path('/proc')):
    """Restrict interruption to our user, project, interpreter, module and USB."""
    entry = proc_root / str(pid)
    try:
        if entry.stat().st_uid != os.getuid() or (entry / 'cwd').resolve() != ROOT:
            return False
        argv = (entry / 'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        if (len(argv) < 5 or Path(argv[0]).resolve() != IO_PYTHON.resolve()
                or argv[1] != '-m' or argv[2] not in WORKERS
                or argv.count('--port') != 1):
            return False
        device = argv[argv.index('--port') + 1]
        return Path(device).resolve() == Path(port).resolve()
    except (OSError, ValueError, IndexError, UnicodeError):
        return False


def stop_workers(port):
    # pidfd prevents accidentally signalling a different process after PID reuse.
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('This command requires Linux pidfd support; no motor changes made.')
    stopped = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if not matches_worker(pid, port):
            continue
        try:
            descriptor = os.pidfd_open(pid)
        except ProcessLookupError:
            continue
        try:
            if not matches_worker(pid, port):
                continue
            try:
                # Worker cleanup holds the measured pose until we own the USB.
                signal.pidfd_send_signal(descriptor, signal.SIGTERM)
            except ProcessLookupError:
                continue
            if not select.select([descriptor], [], [], 8)[0]:
                raise RuntimeError(f'USB worker {pid} did not exit. Torque OFF is NOT confirmed. No forced kill was sent.')
            stopped.append(pid)
        finally:
            os.close(descriptor)
    return stopped


def release(port):
    stopped = stop_workers(port)
    if stopped:
        print('OMX USB control stopped: ' + ', '.join(map(str, stopped)), flush=True)
    bus = Bus(port)  # Exclusive serial lock; never steal from an unknown owner.
    try:
        bus.inspect()  # Verify model IDs before motor writes; no calibration needed.
        bus.release()  # Attempts all IDs 11–16 and verifies torque/watchdog readback.
    finally:
        bus.close()
    print('Torque OFF confirmed: IDs 11–16. Watchdog cleared. EEPROM unchanged.')
    if stopped:
        print('시뮬의 실물 연결은 중단되었습니다. 다시 구동하려면 모드를 재선택하고 Enable follower를 누르세요.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', help='OMX USB port; default: config/devices.json')
    parser.add_argument('--supported', action='store_true',
                        help='Confirm that the arm and any held object are supported; skip the prompt')
    args = parser.parse_args(argv)
    try:
        port = args.port or json.loads((ROOT / 'config/devices.json').read_text()).get('follower_port')
        if not port:
            parser.error('No follower_port configured; specify --port')
        if not args.supported:
            print('팔과 잡고 있는 물체를 손이나 받침대로 지지하세요. 토크 해제 시 팔이 내려갈 수 있습니다.')
            if input('지지를 완료했으면 SUPPORTED 입력 (그 외 취소): ').strip() != 'SUPPORTED':
                print('취소됨. 모터와 실행 중인 앱을 변경하지 않았습니다.')
                return 1
        # Serialize terminal release commands, without taking the app session lock.
        lock_dir = ROOT / 'outputs/follower'
        lock_dir.mkdir(parents=True, exist_ok=True)
        with (lock_dir / '.torque_off.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            release(port)
        return 0
    except (EOFError, KeyboardInterrupt):
        print('\n명령이 취소되었습니다. 토크 해제 완료는 확인되지 않았습니다.', file=sys.stderr)
        return 130
    except Exception as error:
        print(f'Torque OFF NOT confirmed: {error}', file=sys.stderr)
        print('팔을 계속 지지하세요. USB·전원·다른 제어 프로그램을 확인한 뒤 다시 실행하세요.', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
