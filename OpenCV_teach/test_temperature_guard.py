"""Thermal policy and worker integration; all motor packets are simulated."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest.mock import Mock, patch

import teach_left_arm as app
import test_optimization as baseline
import motor_diagnostics as motors
from temperature_guard import TemperatureGuard, TemperatureFault, check_readings


class Clock:
    def __init__(self): self.now = 0.
    def __call__(self): return self.now


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.guard = TemperatureGuard(clock=self.clock)

    def observe(self, raw, joint='wrist_flex', channel='health/block', active=True):
        return self.guard.observe(joint, channel, raw, active=active)

    def test_spike_recovers_and_repeated_spikes_do_not_accumulate(self):
        self.observe(31); self.clock.now=.2; self.observe(75)
        self.clock.now=.3; self.observe(31)
        self.assertEqual(self.guard.pending(), [])
        self.assertEqual([v['event'] for v in self.guard.take_transitions()], ['started','recovered'])
        for n in range(10):
            self.clock.now=1+n; self.assertEqual(self.observe(65)['elapsed_s'],0)
            self.clock.now+=.1; self.observe(64)

    def test_exact_five_seconds_inclusive_threshold_and_fault_latches(self):
        for n in range(10):
            self.clock.now=n*.5
            state=self.observe(65)
            self.assertEqual(state['state'],'high')
        self.clock.now=5
        with self.assertRaises(TemperatureFault) as raised:self.observe(65)
        self.assertEqual(raised.exception.detail['reason'],'sustained_high')
        with self.assertRaises(TemperatureFault):self.observe(31)
        self.assertEqual(self.guard.take_transitions()[-1]['event'],'fault')

    def test_joint_and_block_single_paths_are_independent(self):
        self.observe(75)
        for n in range(1,9):
            self.clock.now=n*.5
            self.observe(75)
            self.observe(31,channel='health/single')
            self.observe(31,joint='gripper')
        self.clock.now=4.5
        self.assertEqual(self.observe(75,joint='gripper')['elapsed_s'],0)
        self.assertEqual(self.observe(75,channel='wrist_diagnostic/block')['elapsed_s'],0)
        self.clock.now=5
        with self.assertRaises(TemperatureFault):self.observe(75)

    def test_pending_gap_is_checked_without_new_data_and_before_low_reset(self):
        for low_next in (False,True):
            with self.subTest(low_next=low_next):
                clock=Clock();guard=TemperatureGuard(clock=clock)
                guard.observe('gripper','block',75,active=True)
                clock.now=1.5;guard.check_timeouts()
                clock.now=1.501
                with self.assertRaises(TemperatureFault) as raised:
                    if low_next:guard.observe('gripper','block',31,active=True)
                    else:guard.check_timeouts()
                self.assertEqual(raised.exception.detail['reason'],'sample_gap')

    def test_invalid_data_and_preactivation_are_immediate(self):
        for raw in (None,True,False,65.,-1,151,'31'):
            with self.subTest(raw=raw),self.assertRaises(TemperatureFault):
                TemperatureGuard().observe('gripper','block',raw,active=True)
        with self.assertRaises(TemperatureFault):self.observe(65,active=False)
        guard=TemperatureGuard()
        with self.assertRaises(TemperatureFault):
            guard.observe('gripper','diagnostic',40,active=False,inactive_limit=40)
        self.assertEqual(TemperatureGuard().observe('gripper','block',150,active=True)['state'],'high')

    def test_bad_clock_and_unsafe_configuration_rejected(self):
        self.observe(75);self.clock.now=-1
        with self.assertRaises(TemperatureFault):self.guard.check_timeouts()
        for args in ({'duration':6},{'duration':0},{'max_gap':2},{'limit':120},{'duration':float('nan')}):
            with self.subTest(args=args),self.assertRaises(ValueError):TemperatureGuard(**args)
        for key,value in [('TEMPERATURE_DURATION_S','6.0'),('TEMPERATURE_MAX_GAP_S','2.0')]:
            with self.assertRaises(ValueError):
                baseline.SettingsTests().changed(f'{key} = '+('5.0' if 'DURATION' in key else '1.5'),f'{key} = {value}')


class IntegrationTests(unittest.TestCase):
    def hardware(self):
        case=baseline.HealthTests();case.setUp()
        self.clock=Clock();case.hardware.temperature_guard=TemperatureGuard(clock=self.clock)
        return case

    def test_real_health_packet_path_spike_recovery_and_sustained_stop(self):
        case=self.hardware();case.hardware.health(True)
        case.hot=True;self.clock.now=.5;sample=case.hardware.health(True)
        self.assertTrue(sample['complete']);self.assertEqual(len(sample['temperature_pending']),12)
        case.hot=False;self.clock.now=1;sample=case.hardware.health(True)
        self.assertFalse(sample['temperature_pending'])
        self.assertEqual(len(sample['temperature_transition']),12)
        case.hot=True
        for n in range(10):self.clock.now=2+n*.5;case.hardware.health(True)
        self.clock.now=7
        with self.assertRaises(TemperatureFault):case.hardware.health(True)
        self.assertEqual(case.hardware.temperature_sample['temperature_fault']['reason'],'sustained_high')
        self.assertFalse(case.hardware.temperature_sample['complete'])

    def test_active_mode_changes_do_not_clear_timer(self):
        case=self.hardware();case.hot=True
        for n,mode in enumerate(('HOLD','MOVE','PLAY','HOLD','TEACH','PLAY','HOLD','MOVE','PLAY','HOLD')):
            self.clock.now=n*.5
            case.hardware.dashboard_context={'state':mode}
            case.hardware.health(True)
        self.clock.now=5
        with self.assertRaises(TemperatureFault):case.hardware.health(True)

    def test_watchdog_reports_gap_and_normal_read_cannot_hide_it(self):
        case=self.hardware();case.hot=True;case.hardware.health(True)
        case.hot=False;self.clock.now=1.6
        with self.assertRaises(TemperatureFault):case.hardware.check_temperature_watchdog()
        self.assertEqual(case.hardware.temperature_sample['temperature_fault']['reason'],'sample_gap')
        self.assertEqual(case.hardware.temperature_sample['temperature_transition'][-1]['event'],'fault')

    def test_new_connection_resets_old_latched_fault_without_writes(self):
        case=self.hardware();hardware=case.hardware;case.hot=True
        hardware.health(True);self.clock.now=2
        with self.assertRaises(TemperatureFault):hardware.check_temperature_watchdog()
        hardware.preflight=Mock()
        hardware.follower.id='simulated'
        hardware.follower.bus.connect=Mock()
        hardware.follower.bus.read=Mock(return_value=0)
        hardware.follower.bus.write=Mock(side_effect=AssertionError('unexpected write'))
        hardware.read=Mock(return_value=(None,dict.fromkeys(app.KEYS,0.)))
        with patch('diagnose_shoulder_lift.compare_calibration',return_value={'matches':True}):
            hardware.connect()
        self.assertIsNone(hardware.temperature_guard.fault)
        self.assertEqual(hardware.temperature_guard.pending(),[])
        hardware.follower.bus.write.assert_not_called()

    def test_dashboard_collector_uses_same_policy_and_emits_fault_evidence(self):
        from lerobot.motors.feetech.tables import STS_SMS_SERIES_CONTROL_TABLE as table
        clock=Clock();guard=TemperatureGuard(clock=clock)
        bus=types.SimpleNamespace(motors={j:types.SimpleNamespace(model='sts3215') for j in app.JOINTS},
             model_ctrl_table={'sts3215':table},_decode_sign=lambda k,v:v,_normalize=lambda v:v)
        block=bytearray(31);block[0]=1;block[23]=75
        pair={'block_hex':block.hex(),'block_comm':0,'temperature_single_raw':31}
        with tempfile.TemporaryDirectory() as d,patch.object(motors.temperature,'validate_responses'),patch.object(motors.temperature,'read_pair',return_value=pair):
            collector=motors.Collector(bus,d);collector.temperature_guard=guard
            collector.settings={j:{'Model_Number':777} for j in app.JOINTS}
            for n in range(10):
                clock.now=n*.5;frame=collector.capture(True)
                motors.validate_live(frame)
                self.assertEqual(frame['rows'][0]['issues'][0][0],'warning')
            clock.now=5;frame=collector.capture(True)
            with self.assertRaises(TemperatureFault):motors.validate_live(frame)
            written=json.loads(collector.path.read_text().splitlines()[-1])
            self.assertEqual(written['rows'][0]['temperature_fault']['reason'],'sustained_high')

    def test_diagnostics_log_spike_and_fault_with_separate_paths(self):
        import wrist_temperature_diagnostic as wrist
        import gripper_grasp_diagnostic as grip
        for which in ('wrist','gripper'):
            with self.subTest(which=which),tempfile.TemporaryDirectory() as folder:
                bus=types.SimpleNamespace(read=lambda *a,**k:0)
                clock=Clock();guard=TemperatureGuard(clock=clock)
                if which=='wrist':
                    sample={'temperature_block_raw':75,'temperature_single_raw':31,'status_raw':0,'torque_enable_raw':1}
                    with patch.object(wrist.temperature,'validate_responses'):
                        diag=wrist.WristTemperatureDiagnostic(bus,folder)
                    reader=patch.object(wrist.temperature,'read_pair',side_effect=lambda *a,**k:deepcopy(sample))
                else:
                    sample=dict.fromkeys(('telemetry_comm_result','telemetry_packet_error_raw','temperature_single_comm','temperature_single_packet_error_raw','status_raw'),0)
                    sample.update(present_temperature=75,temperature_single_raw=31,torque_enable=1)
                    with patch.object(grip,'validate_responses'):
                        diag=grip.GripperGraspDiagnostic(bus,folder,sample_kind='empty',identity={})
                    reader=patch.object(grip,'read_telemetry',side_effect=lambda *a,**k:deepcopy(sample))
                diag.temperature_guard=guard
                with reader,patch('builtins.print'):
                    for n in range(10):clock.now=n*.5;diag(True)
                    clock.now=5
                    with self.assertRaises(TemperatureFault):diag(True)
                last=json.loads(diag.path.read_text().splitlines()[-1])
                self.assertEqual(last['temperature_fault']['reason'],'sustained_high')
                self.assertEqual(last['temperature_fault']['joint'],'wrist_flex' if which=='wrist' else 'gripper')

    def test_session_fault_attempts_off_for_all_six_and_retains_log(self):
        fixture=self.hardware();fixture.hot=True
        hardware=fixture.hardware
        hardware.follower.id='simulated';hardware.motion_attempted=True
        hardware.follower.bus.is_connected=True
        writes=[]
        hardware.follower.bus.write=lambda *a,**k:writes.append(a)
        hardware.follower.bus.read=lambda *a,**k:0
        hardware.follower.bus.port_handler=types.SimpleNamespace(closePort=Mock())
        class Backend(app.Demo):
            follower_only=True
            def health(self,active=True):
                if active:return app.Hardware.health(self,True)
                return app.Demo.health(self,False)
            check_temperature_watchdog=app.Hardware.check_temperature_watchdog
            def close(self):return hardware.close()
        backend=Backend();backend.follower=hardware.follower
        backend.temperature_guard=TemperatureGuard(duration=.08)
        with tempfile.TemporaryDirectory() as folder,patch.object(app,'HEALTH_INTERVAL',.02):
            session=app.Session(backend,folder,auto_sync=True);session.start()
            deadline=time.monotonic()+3
            while session.thread.is_alive() and time.monotonic()<deadline:
                session.heartbeat=time.monotonic()
                if session.state=='READY':session.command('hold_current')
                time.sleep(.005)
            if session.thread.is_alive():session.stop.set();session.wakeup.set();session.thread.join(2)
            report=json.loads(next(Path(folder).glob('session_*.json')).read_text())
            self.assertIn('TemperatureFault',report['outcome'])
            self.assertFalse(report['cleanup_errors'])
            self.assertEqual(writes,[('Torque_Enable',j,0) for j in app.JOINTS])
            self.assertTrue(any(s.get('temperature_transition') for s in report['temperature_samples']))
            self.assertTrue(report['temperature_samples'][-1]['fault'])
            events=[json.loads(line) for line in next(Path(folder).glob('*_playback.jsonl')).read_text().splitlines()]
            self.assertTrue(any(e.get('event')=='temperature_transition' for e in events))

    def test_transition_logging_is_not_lost_in_ten_second_throttle(self):
        session=app.Session(app.Demo(),Path('/unused'))
        report={'temperature_samples':[]}
        sample={'time':app.stamp(),'temperatures_raw':{'gripper':75},'complete':True}
        session.log_temperatures(report,sample,0)
        session.log_temperatures(report,{**sample,'temperature_transition':[{'event':'started'}]},.1)
        session.log_temperatures(report,{**sample,'temperature_transition':[{'event':'recovered'}]},.2)
        self.assertEqual(len(report['temperature_samples']),3)

    def test_worker_watchdog_stops_before_next_send_during_monitor_gap(self):
        clock=Clock()
        class Backend(app.Demo):
            follower_only=True
            check_temperature_watchdog=app.Hardware.check_temperature_watchdog
            def __init__(self):
                super().__init__();self.temperature_guard=TemperatureGuard(clock=clock)
                self.closed=False;self.sent=[];self.triggered=False
            def health(self,active=True):
                sample=super().health(active)
                if active and not self.triggered:
                    check_readings(sample,self.temperature_guard,'gripper',{'block':75},active=True,source='health')
                    self.triggered=True
                    clock.now=1.6  # no next temperature observation arrives
                return sample
            def send(self,target):
                self.sent.append(clock.now)
                return super().send(target)
            def close(self):self.closed=True;return []
        backend=Backend()
        with tempfile.TemporaryDirectory() as folder:
            session=app.Session(backend,folder,auto_sync=True);session.start()
            deadline=time.monotonic()+3
            while session.thread.is_alive() and time.monotonic()<deadline:
                session.heartbeat=time.monotonic()
                if session.state=='READY':session.command('hold_current')
                time.sleep(.005)
            if session.thread.is_alive():session.stop.set();session.wakeup.set();session.thread.join(2)
            self.assertTrue(backend.closed)
            self.assertTrue(backend.sent)
            self.assertTrue(all(t==0 for t in backend.sent))
            report=json.loads(next(Path(folder).glob('session_*.json')).read_text())
            self.assertIn('감시 공백',report['outcome'])
            self.assertEqual(report['temperature_samples'][-1]['temperature_fault']['reason'],'sample_gap')


if __name__=='__main__':unittest.main()
