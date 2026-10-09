"""Subprocess tests: never import Isaac or open a motor/camera device."""
import ast
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


class ShutdownTests(unittest.TestCase):
    def test_real_window_callback_notifies_before_native_shutdown(self):
        tree=ast.parse((ROOT/'scripts/sim.py').read_text())
        callback=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='window_close_requested')
        notify=Mock();namespace=dict(intentional_exit=False,notify_launcher=notify)
        exec(compile(ast.Module(body=[callback],type_ignores=[]),'window_close_callback','exec'),namespace)
        namespace['window_close_requested'](None)
        self.assertTrue(namespace['intentional_exit'])
        notify.assert_called_once_with('exit',reason='Isaac window close')

    def run_case(self, body, *, interrupt=False, twice=False, grace=.25, release_ok=True):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            child=folder/'fake_isaac.py'
            child.write_text('import os,signal,time,subprocess,sys\n'
                'from pathlib import Path\n'
                'from teach_io.lifecycle import notify\n'
                f'folder=Path({str(folder)!r})\n'+body)
            launcher='\n'.join([
                'import sys', 'from pathlib import Path',
                'from teach_io.supervisor import supervise',
                f'folder=Path({str(folder)!r})',
                'def release(port):',
                "    (folder/'released').write_text(port)",
                f'    return {release_ok!r}',
                f'sys.exit(supervise([sys.executable,{str(child)!r}],grace={grace!r},release=release))'])
            env=dict(os.environ,PYTHONPATH=str(ROOT))
            proc=subprocess.Popen([sys.executable,'-c',launcher],cwd=ROOT,env=env,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
            try:
                if interrupt:
                    deadline=time.monotonic()+4
                    while not (folder/'ready').exists() and proc.poll() is None and time.monotonic()<deadline:time.sleep(.01)
                    self.assertTrue((folder/'ready').exists())
                    # Terminal Ctrl+C targets the foreground launcher group.
                    os.killpg(proc.pid,signal.SIGINT)
                    if twice:
                        time.sleep(.1);os.killpg(proc.pid,signal.SIGINT)
                stdout,stderr=proc.communicate(timeout=4)
                released=(folder/'released').read_text() if (folder/'released').exists() else None
                if (folder/'camera_pid').exists():
                    camera=int((folder/'camera_pid').read_text())
                    deadline=time.monotonic()+1
                    while time.monotonic()<deadline:
                        stat=Path(f'/proc/{camera}/stat')
                        if not stat.exists() or stat.read_text().split()[2]=='Z':break
                        time.sleep(.01)
                    else:self.fail('Camera child survived launcher shutdown')
                return proc.returncode,released,stdout,stderr
            finally:
                if proc.poll() is None:proc.kill();proc.wait()
                for stream in (proc.stdout,proc.stderr):stream.close()

    def test_window_close_kills_stuck_kit_and_child_then_releases_registered_port(self):
        code,port,out,_=self.run_case('''
signal.signal(signal.SIGINT,signal.SIG_IGN)
camera=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
(folder/'camera_pid').write_text(str(camera.pid))
notify('motor_port',port='/dev/fake')
notify('exit',reason='Isaac window close')
while True:time.sleep(.05)
''')
        self.assertEqual(code,0);self.assertEqual(port,'/dev/fake')
        self.assertIn('deadline reached',out)

    def test_ctrl_c_remains_effective_when_isaac_ignores_signals(self):
        code,port,_,_=self.run_case('''
signal.signal(signal.SIGINT,signal.SIG_IGN)
notify('motor_port',port='/dev/fake')
(folder/'ready').touch()
while True:time.sleep(.05)
''',interrupt=True)
        self.assertEqual(code,130);self.assertEqual(port,'/dev/fake')

    def test_second_ctrl_c_shortens_wait_without_skipping_release(self):
        started=time.monotonic()
        code,port,_,_=self.run_case('''
signal.signal(signal.SIGINT,signal.SIG_IGN)
notify('motor_port',port='/dev/fake')
(folder/'ready').touch()
while True:time.sleep(.05)
''',interrupt=True,twice=True,grace=20)
        self.assertLess(time.monotonic()-started,3)
        self.assertEqual(code,130);self.assertEqual(port,'/dev/fake')

    def test_unexpected_crash_does_not_drop_arm(self):
        code,port,_,_=self.run_case("notify('motor_port',port='/dev/fake')\nsys.exit(7)\n")
        self.assertEqual(code,7);self.assertIsNone(port)

    def test_unconfirmed_release_is_reported_as_failure(self):
        code,port,_,_=self.run_case("notify('motor_port',port='/dev/fake')\nnotify('exit')\nsys.exit(0)\n",release_ok=False)
        self.assertEqual(code,2);self.assertEqual(port,'/dev/fake')

    def test_confirmed_release_is_not_repeated(self):
        code,port,_,_=self.run_case("notify('motor_port',port='/dev/fake')\nnotify('exit')\nnotify('released',port='/dev/fake')\nsys.exit(0)\n")
        self.assertEqual(code,0);self.assertIsNone(port)

    def test_new_enable_after_release_requires_release_again(self):
        code,port,_,_=self.run_case("notify('motor_port',port='/dev/fake')\nnotify('released',port='/dev/fake')\nnotify('motor_port',port='/dev/fake')\nnotify('exit')\nsys.exit(0)\n")
        self.assertEqual(code,0);self.assertEqual(port,'/dev/fake')

    def test_sim_only_shutdown_never_releases_hardware(self):
        code,port,_,_=self.run_case("notify('exit')\nsys.exit(0)\n")
        self.assertEqual(code,0);self.assertIsNone(port)

    def test_unintentional_cleanup_timeout_does_not_release(self):
        code,port,_,_=self.run_case("notify('motor_port',port='/dev/fake')\nnotify('cleanup',intentional=False)\ntime.sleep(60)\n")
        self.assertEqual(code,1);self.assertIsNone(port)

    def test_other_application_is_not_killed(self):
        other=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
        try:
            self.run_case("notify('exit')\ntime.sleep(60)\n")
            self.assertIsNone(other.poll())
        finally:other.terminate();other.wait(timeout=2)


if __name__=='__main__':unittest.main()
