"""Exercise real child lifecycle using dummy programs, never hardware."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

class LauncherTests(unittest.TestCase):
    def test_one_session_lock_and_interrupt_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            worker = "import os,time;from pathlib import Path;Path('pid'+str(os.getpid())).write_text(str(os.getpid()));time.sleep(60)"
            commands = [(name,[sys.executable,'-c',worker]) for name in ('so','omx','sim')]
            setup = f"from scripts import run_all as r;from pathlib import Path;r.ROOT=Path({d!r});r.COMMANDS={commands!r};raise SystemExit(r.main())"
            parent = subprocess.Popen([sys.executable,'-c',setup],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            try:
                deadline=time.monotonic()+5
                while len(list(Path(d).glob('pid*')))<3 and time.monotonic()<deadline:time.sleep(.05)
                pids=[int(p.read_text()) for p in Path(d).glob('pid*')]
                self.assertEqual(len(pids),3)
                duplicate=subprocess.run([sys.executable,'-c',setup],capture_output=True,text=True,timeout=5)
                self.assertEqual(duplicate.returncode,1)
                parent.send_signal(signal.SIGINT)
                out=parent.communicate(timeout=15)[0]
                self.assertEqual(parent.returncode,0,out)
                for pid in pids:
                    with self.assertRaises(ProcessLookupError):os.kill(pid,0)
            finally:
                if parent.poll() is None:
                    parent.send_signal(signal.SIGTERM)
                    parent.communicate(timeout=15)
