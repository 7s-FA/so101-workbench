import contextlib,io,sys,tempfile,unittest,subprocess,signal,time,os
from pathlib import Path
from scripts.run_real_sim import run

class RealConsoleTests(unittest.TestCase):
    def test_warning_is_logged_but_error_is_visible_and_exit_code_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            log=Path(d)/'console.log';output=io.StringIO()
            with contextlib.redirect_stdout(output),contextlib.redirect_stderr(output):
                code=run([sys.executable,'-c',"print('[Warning] noisy'); print('[Error] important'); raise RuntimeError('failed')"],log)
            self.assertEqual(code,1)
            self.assertIn('[Warning] noisy',log.read_text());self.assertNotIn('[Warning] noisy',output.getvalue())
            self.assertIn('[Error] important',output.getvalue());self.assertIn('RuntimeError: failed',output.getvalue())
    def test_verbose_preserves_warning(self):
        with tempfile.TemporaryDirectory() as d:
            output=io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(run([sys.executable,'-c',"print('[Warning] visible')"],Path(d)/'console.log',True),0)
            self.assertIn('[Warning] visible',output.getvalue())

    def test_interrupt_reaches_child_and_leaves_no_running_simulator(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);log=folder/'console.log';script=folder/'child.py'
            script.write_text("import signal,sys\ndef stop(*_):\n print('child_stopped',flush=True);sys.exit(0)\nsignal.signal(signal.SIGINT,stop)\nprint('child_ready',flush=True)\nsignal.pause()\n")
            command="from scripts.run_real_sim import run; import sys; raise SystemExit(run([sys.executable,sys.argv[1]],sys.argv[2]))"
            process=subprocess.Popen([sys.executable,'-c',command,str(script),str(log)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            try:
                deadline=time.monotonic()+5
                while time.monotonic()<deadline:
                    if log.exists() and 'child_ready' in log.read_text():break
                    time.sleep(.02)
                else:self.fail('child did not become ready')
                process.send_signal(signal.SIGINT);process.communicate(timeout=5)
                self.assertEqual(process.returncode,0);self.assertIn('child_stopped',log.read_text())
            finally:
                if process.poll() is None:process.terminate();process.communicate(timeout=5)
