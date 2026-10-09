"""Terminal release tests: fake motors only, never open a hardware device."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('torque_off', ROOT / 'scripts/torque_off.py')
torque_off = importlib.util.module_from_spec(spec)
spec.loader.exec_module(torque_off)


class TorqueOffTests(unittest.TestCase):
    def test_cancel_never_touches_workers_or_usb(self):
        with patch.object(torque_off, 'release') as release, patch('builtins.input', return_value='no'):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(torque_off.main(['--port', '/dev/fake']), 1)
            release.assert_not_called()

    def test_worker_matching_is_project_module_and_device_specific(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp); entry = proc / '123'; entry.mkdir()
            (entry / 'cwd').symlink_to(ROOT, target_is_directory=True)
            good = [str(torque_off.IO_PYTHON), '-m', 'teach_io.worker', '--port', '/dev/fake']
            def match(argv):
                (entry / 'cmdline').write_bytes(b'\0'.join(s.encode() for s in argv) + b'\0')
                return torque_off.matches_worker(123, '/dev/fake', proc)
            self.assertTrue(match(good))
            self.assertFalse(match(good[:-1] + ['/dev/another']))
            self.assertFalse(match([good[0], '-m', 'another.worker'] + good[3:]))
            self.assertFalse(match([good[0], '-c', 'pass'] + good[3:]))
            (entry / 'cwd').unlink(); (entry / 'cwd').symlink_to(tmp, target_is_directory=True)
            self.assertFalse(match(good))

    def test_release_waits_for_owned_process_exit_before_usb_open(self):
        # This child only sleeps: no application or real motor process is signalled.
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        bus = Mock()
        def open_bus(port):
            self.assertIsNotNone(child.poll())
            return bus
        try:
            with patch.object(torque_off, 'matches_worker', side_effect=lambda pid, port: pid == child.pid):
                with patch.object(torque_off, 'Bus', side_effect=open_bus), contextlib.redirect_stdout(io.StringIO()):
                    torque_off.release('/dev/fake')
            bus.inspect.assert_called_once(); bus.release.assert_called_once(); bus.close.assert_called_once()
            self.assertEqual(child.returncode, -15)
        finally:
            if child.poll() is None: child.kill()
            child.wait()

    def test_worker_timeout_never_opens_usb_or_forces_kill(self):
        with patch.object(torque_off, 'stop_workers', side_effect=RuntimeError('did not exit')):
            with patch.object(torque_off, 'Bus') as bus:
                with self.assertRaisesRegex(RuntimeError, 'did not exit'):
                    torque_off.release('/dev/fake')
                bus.assert_not_called()

    def test_failed_readback_does_not_print_success_and_closes_port(self):
        bus = Mock(); bus.release.side_effect = RuntimeError('ID 13: release not confirmed')
        output = io.StringIO()
        with patch.object(torque_off, 'stop_workers', return_value=[]), patch.object(torque_off, 'Bus', return_value=bus):
            with contextlib.redirect_stdout(output), self.assertRaisesRegex(RuntimeError, 'ID 13'):
                torque_off.release('/dev/fake')
        bus.close.assert_called_once()
        self.assertNotIn('Torque OFF confirmed', output.getvalue())

    def test_model_mismatch_blocks_writes(self):
        bus = Mock(); bus.inspect.side_effect = RuntimeError('wrong motor model')
        with patch.object(torque_off, 'stop_workers', return_value=[]), patch.object(torque_off, 'Bus', return_value=bus):
            with self.assertRaisesRegex(RuntimeError, 'wrong motor'):
                torque_off.release('/dev/fake')
        bus.release.assert_not_called(); bus.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
