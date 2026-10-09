"""Exercise process boundaries and watchdog behavior without opening a USB device."""
import json
import os
import select
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from teach_io.mapping import IDS, MODELS, TICKS_PER_RAD

ROOT=Path(__file__).resolve().parents[1]
FAKE='''
import sys, os
from teach_io import worker
class FakeBus:
    def __init__(self, port): self.q=[2048]*6;self.owned=False
    def inspect(self):return [dict(id=i,model=m,mode=4 if i<16 else 5,drive=0,offset=0,torque=0,watchdog=0) for i,m in zip(range(11,17),[1060]*3+[1200]*3)]
    def sample(self,detailed=False,packed=False,temperature_guard=None,active=False):
        if temperature_guard:
            temperature=65 if active and os.environ.get('TEST_HIGH_TEMPERATURE')=='True' else 25
            for i in range(11,17):temperature_guard.observe(i,'packed',temperature,active=active)
        return [dict(id=i,ticks=t,trajectory_ticks=t,velocity_rad_s=0,pwm_percent=0) for i,t in zip(range(11,17),self.q)]
    def read(self,motor,address,size):return 0
    def enable(self,info,ticks,reference=False):self.owned=True;self.power_profile={'name':'fake_verified_profile'}
    def positions(self,ticks):self.q=ticks
    def release(self):self.owned=False;print("FAKE_TORQUE_OFF",file=sys.stderr,flush=True)
    def hold(self):pass
    def close(self):pass
worker.Bus=FakeBus
sys.exit(worker.main())
'''


class ProtocolTests(unittest.TestCase):
    def test_il_high_temperature_publishes_pending_then_latched_fault(self):
        self.exercise('temperature',reference_control=True,sustained_temperature=True)

    def test_il_temperature_policy_crosses_rpc_without_weakening_watchdog(self):
        self.exercise('watchdog',reference_control=True,sustained_temperature=True)

    def test_display_only_scene_accepts_nominal_overlap_and_keeps_watchdog(self):
        self.exercise('entry',reference_control=True,display_only_scene=True)

    def test_reference_entry_accepts_faster_speed_and_keeps_watchdog(self):
        self.exercise('entry',reference_control=True,entry_speed=.5)

    def test_reference_switches_after_entry_and_retains_watchdog(self):
        self.exercise('watchdog', reference_control=True)

    def test_fast_feedback_preserves_watchdog(self):
        self.exercise('watchdog', fast_feedback=True)

    def test_read_only_then_enable_and_latched_watchdog(self):
        self.exercise('watchdog')

    def test_ctrl_c_releases_owned_motors(self):
        self.exercise('interrupt')

    def test_compensation_can_be_disabled_and_invalid_context_is_latched(self):
        self.exercise('bad_context')

    def test_invalid_compensation_options_rejected_before_enable(self):
        self.exercise('bad_option')

    def test_current_pose_entry_enables_without_manual_start_matching(self):
        self.exercise('entry')

    def test_changed_pose_invalidates_entry_before_torque_enable(self):
        self.exercise('stale_entry')

    def exercise(self, ending, fast_feedback=False, reference_control=False, display_only_scene=False, entry_speed=.1, sustained_temperature=False):
        with tempfile.TemporaryDirectory() as tmp:
            config=dict(kind='omx_follower_urdf',verified=True,port='/dev/fake',ids=IDS,models=MODELS,
                zero_ticks=[2048]*6,ticks_per_rad=[TICKS_PER_RAD]*6,
                limits=[[-2,2]]*5+[[0,1.1]],tick_limits=[[0,4095]]*6,
                motor_config=[dict(id=i,model=m,mode=4 if i<16 else 5,drive=0,offset=0) for i,m in zip(IDS,MODELS)])
            path=Path(tmp)/'calibration.json';path.write_text(json.dumps(config))
            proc=subprocess.Popen([sys.executable,'-c',FAKE,'--port','/dev/fake','--calibration',str(path)]+(['--fast-feedback'] if fast_feedback else [])+(['--reference-control'] if reference_control else [])+(['--sustained-temperature'] if sustained_temperature else [])+(['--display-only-scene'] if display_only_scene else [])+['--entry-speed',str(entry_speed)],
                cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1,
                env={**os.environ,'TEST_HIGH_TEMPERATURE':str(ending=='temperature')})
            def receive():
                self.assertTrue(select.select([proc.stdout],[],[],3)[0])
                return json.loads(proc.stdout.readline())
            def send(op,**data):
                proc.stdin.write(json.dumps(dict(op=op,sent=time.monotonic(),**data))+'\n');proc.stdin.flush()
                return receive()
            try:
                self.assertEqual(receive()['state'],'connected_read_only')
                first=send('read');self.assertEqual(first['state'],'read_only')
                if sustained_temperature:
                    self.assertEqual(first['temperature']['limit_c'],65)
                    self.assertEqual(first['temperature']['duration_s'],5.)
                    self.assertEqual(first['temperature']['pending'],[])
                home=[.008]*6
                entry=None
                if ending in ('entry','stale_entry'):
                    home=[.2]*6
                    entry=dict(start=[.05 if ending=='stale_entry' else 0.]*6,goal=home,collision_checked=True)
                    if display_only_scene:entry.update(collision_checked=False,pairs=[['link7','table'],['link6','table']])
                if ending=='stale_entry':
                    reply=send('arm',target=home,path=[home],entry=entry)
                    self.assertFalse(reply['ok']);self.assertIn('moved',reply['error'])
                    self.assertEqual(reply['code'],'entry_pose_changed')
                    retry=send('read')
                    self.assertEqual(retry['state'],'read_only')
                    self.assertFalse(retry['torque_on'])
                    return
                if ending=='bad_option':
                    reply=send('arm',target=home,path=[home],compensation_enabled='true')
                    self.assertFalse(reply['ok']);self.assertIn('boolean',reply['error'])
                    self.assertEqual(proc.wait(timeout=3),2)
                    return
                enabled=send('arm',target=home,path=[home],compensation_enabled=ending!='bad_context',entry=entry)
                self.assertEqual(enabled['state'],'armed')
                self.assertEqual(enabled['power_profile']['name'],'fake_verified_profile')
                self.assertFalse(enabled['aligned'])
                self.assertTrue(enabled['torque_on'])
                self.assertFalse(enabled['release_required'])
                for _ in range(550 if ending=='entry' else 90):
                    time.sleep(.01)
                    reply=send('target',target=home)
                    self.assertTrue(reply['ok'])
                    if reply['aligned']:break
                self.assertTrue(reply['aligned'])
                self.assertEqual(reply['motion_error']['samples'],0)
                self.assertEqual(reply['tracking']['profile'],'lerobot_reference' if reference_control else 'evaluation')
                # Once aligned, a slow command produces measured motion on the fake bus.
                for i in range(1,6):
                    time.sleep(.01)
                    moved=send('target',target=[v+i*.001 for v in home],context='empty',phase='free',trial_active=True)
                    self.assertTrue(moved['ok'])
                self.assertGreater(moved['angles'][0],.009)
                self.assertEqual(moved['motion_error']['samples'],5)
                self.assertIn('tracking',moved)
                self.assertEqual(moved['tracking']['compensation']['context'],'empty')
                self.assertEqual(moved['tracking']['compensation']['phase'],'free')
                self.assertEqual(len(moved['tracking']['desired_angles']),6)
                if reference_control:
                    time.sleep(.02)
                    goal=[.5]*6
                    direct=send('target',target=goal)
                    self.assertTrue(direct['ok'])
                    self.assertEqual(direct['tracking']['profile'],'lerobot_reference')
                    self.assertEqual(direct['tracking']['command_angles'],goal)
                    self.assertEqual(direct['tracking']['pace'],1.)
                if ending=='temperature':
                    self.assertEqual(moved['temperature']['state'],'high')
                    self.assertEqual(len(moved['temperature']['pending']),6)
                    for _ in range(600):
                        time.sleep(.01)
                        fault=send('target',target=home)
                        if not fault['ok']:break
                    self.assertFalse(fault['ok'])
                    self.assertEqual(fault['temperature']['fault']['reason'],'sustained_high')
                    self.assertGreaterEqual(fault['temperature']['fault']['elapsed_s'],5)
                    self.assertIn('65°C',fault['error'])
                    self.assertEqual(proc.wait(timeout=3),2)
                elif ending=='bad_context':
                    self.assertFalse(moved['tracking']['compensation']['enabled'])
                    self.assertEqual(moved['tracking']['bias_deg'],[0]*6)
                    reply=send('target',target=home,context='invalid',phase='free')
                    self.assertFalse(reply['ok']);self.assertIn('context',reply['error'])
                    self.assertEqual(proc.wait(timeout=3),2)
                elif ending=='interrupt':
                    proc.send_signal(signal.SIGINT)
                    self.assertEqual(proc.wait(timeout=3),0)
                    self.assertIn('FAKE_TORQUE_OFF',proc.stderr.read())
                else:
                    fault=receive()
                    self.assertFalse(fault['ok']);self.assertIn('heartbeat',fault['error'])
                    self.assertEqual(proc.wait(timeout=3),2)
                    self.assertNotIn('FAKE_TORQUE_OFF',proc.stderr.read())
            finally:
                if proc.poll() is None:proc.kill();proc.wait()
                for f in (proc.stdin,proc.stdout,proc.stderr):f.close()


if __name__=='__main__':unittest.main()
