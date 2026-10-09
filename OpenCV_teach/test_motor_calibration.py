"""EEPROM lifecycle and encoder UI tests; serial hardware is never opened."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import motor_calibration as c
import motor_positions as p
import manual_teaching as m
import teach_left_arm as t


class FakeBus:
    def __init__(self):
        from lerobot.motors import MotorCalibration
        self.motors={n:SimpleNamespace(id=i) for i,n in enumerate(c.LABELS,1)}
        self.calibration={n:MotorCalibration(id=i,drive_mode=0,homing_offset=i,
            range_min=300,range_max=3500) for i,n in enumerate(c.LABELS,1)}
        self.previous=deepcopy(self.calibration);self.positions=dict.fromkeys(c.LABELS,2047)
        self.is_connected=False;self.calls=[];self.port_handler=SimpleNamespace(closePort=self.close)
    def connect(self):self.is_connected=True
    def close(self):self.is_connected=False
    def read_calibration(self):return deepcopy(self.calibration)
    def read(self,register,name,normalize=False):return 0
    def disable_torque(self):self.calls.append(('torque_off',))
    def write(self,register,name,value):
        assert register=='Operating_Mode',(register,name,value)
        self.calls.append((register,name,value))
    def sync_read(self,register,normalize=False):
        assert register=='Present_Position' and normalize is False
        return self.positions.copy()
    def set_half_turn_homings(self):
        self.calls.append(('middle',));self.calibration={};self.positions=dict.fromkeys(c.LABELS,c.MIDPOINT_TICK);return dict.fromkeys(c.LABELS,123)
    def write_calibration(self,values):
        self.calls.append(('save',deepcopy(values)));self.calibration=deepcopy(values)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        import gc
        gc.collect()  # Finalize earlier Tk test objects before starting this worker.
        self.temp=tempfile.TemporaryDirectory();self.folder=Path(self.temp.name)
        self.bus=FakeBus();self.path=self.folder/'follower_test.json';self.path.write_text('{"original":true}')
        self.arm=SimpleNamespace(id='follower_test',bus=self.bus,calibration_fpath=self.path,calibration=self.bus.previous)
        self.worker=c.CalibrationWorker(lambda:self.arm,self.folder/'recovery.json')
        self.worker.thread.start();self.wait(lambda:self.worker.state=='MIDPOINT')
    def tearDown(self):
        self.worker.cancel.set();self.worker.thread.join(3);self.assertFalse(self.worker.thread.is_alive());self.temp.cleanup()
    def wait(self,fn):
        end=time.monotonic()+3
        while time.monotonic()<end:
            if fn():return
            time.sleep(.01)
        self.fail(str((self.worker.state,list(self.worker.events.queue))))
    def ranges(self):
        self.worker.commands.put('middle');self.wait(lambda:self.worker.state=='RANGE')
        self.bus.positions=dict.fromkeys(c.LABELS,1000);time.sleep(.15)
        self.bus.positions=dict.fromkeys(c.LABELS,3000);time.sleep(.15)
        self.worker.commands.put('review');self.wait(lambda:self.worker.state=='REVIEW')
    def test_six_motor_midpoint_and_measured_ranges_save_only_on_save(self):
        self.ranges();self.assertEqual(self.path.read_text(),'{"original":true}')
        self.worker.commands.put('save');self.assertTrue(self.worker.finished.wait(3))
        self.assertTrue(self.worker.saved);self.assertFalse(self.bus.is_connected)
        saved=json.loads(self.path.read_text());self.assertEqual(set(saved),set(c.LABELS))
        self.assertEqual((saved['gripper']['range_min'],saved['gripper']['range_max']),(1000,3000))
        self.assertEqual((saved['wrist_roll']['range_min'],saved['wrist_roll']['range_max']),(1000,3000))
        self.assertEqual(saved['gripper']['homing_offset'],123)
        self.assertEqual(saved,c.serializable(self.bus.calibration))
        record=json.loads((self.folder/'recovery.json').read_text())
        self.assertEqual(record['midpoint_after'],dict.fromkeys(c.LABELS,2047))
        self.assertEqual(record['calibration'],saved)
    def test_cancel_after_homing_restores_previous_eeprom_without_touching_json(self):
        self.ranges();self.worker.cancel.set();self.assertTrue(self.worker.finished.wait(3))
        self.assertEqual(self.bus.calibration,self.bus.previous)
        self.assertEqual(self.path.read_text(),'{"original":true}')
        self.assertFalse(self.worker.saved);self.assertIsNone(self.worker.rollback_error)
    def test_unmoved_gripper_rejects_range_review(self):
        self.worker.commands.put('middle');self.wait(lambda:self.worker.state=='RANGE');time.sleep(.1)
        self.bus.positions={n:1000 if n!='gripper' else 2047 for n in c.LABELS};time.sleep(.15)
        self.bus.positions={n:3000 if n!='gripper' else 2047 for n in c.LABELS};time.sleep(.15)
        self.worker.commands.put('review');time.sleep(.15)
        self.assertEqual(self.worker.state,'RANGE');self.assertFalse(self.worker.saved)
        self.assertTrue(any(k=='notice' and '집게 벌림' in str(v) for k,v in self.worker.events.queue))
    def test_unmoved_wrist_rotation_cannot_silently_use_full_turn_range(self):
        self.worker.commands.put('middle');self.wait(lambda:self.worker.state=='RANGE')
        self.bus.positions={n:1000 if n!='wrist_roll' else 2047 for n in c.LABELS};time.sleep(.15)
        self.bus.positions={n:3000 if n!='wrist_roll' else 2047 for n in c.LABELS};time.sleep(.15)
        self.worker.commands.put('review');time.sleep(.15)
        self.assertEqual(self.worker.state,'RANGE');self.assertFalse(self.worker.saved)
        self.assertTrue(any(k=='notice' and '집게 회전' in str(v) for k,v in self.worker.events.queue))

    def test_midpoint_is_recorded_before_range_and_never_replaced_by_range_average(self):
        self.bus.positions=dict.fromkeys(c.LABELS,1600)
        time.sleep(.1);self.worker.commands.put('middle');self.wait(lambda:self.worker.state=='RANGE')
        self.bus.positions=dict.fromkeys(c.LABELS,1400);time.sleep(.15)
        self.bus.positions=dict.fromkeys(c.LABELS,3100);time.sleep(.15)
        self.worker.commands.put('review');self.wait(lambda:self.worker.state=='REVIEW')
        self.assertEqual(self.worker.midpoints,dict.fromkeys(c.LABELS,2047))
        self.assertEqual(self.worker.middle_before,dict.fromkeys(c.LABELS,1600))
        self.worker.commands.put('save');self.assertTrue(self.worker.finished.wait(3))
        self.assertEqual(json.loads((self.folder/'recovery.json').read_text())['midpoint_after']['gripper'],2047)

    def test_failed_middle_readback_rolls_back_without_saving(self):
        with patch.object(self.bus,'set_half_turn_homings',return_value=dict.fromkeys(c.LABELS,123)):
            self.bus.positions=dict.fromkeys(c.LABELS,1600)
            self.worker.commands.put('middle');self.assertTrue(self.worker.finished.wait(3))
        self.assertFalse(self.worker.saved);self.assertEqual(self.bus.calibration,self.bus.previous)
        self.assertEqual(self.path.read_text(),'{'+'"original":true}')
        self.assertTrue(any(k=='error' and '적용 확인 실패' in str(v) for k,v in self.worker.events.queue))

    def test_encoder_wrap_during_range_is_rejected_and_restored(self):
        self.worker.commands.put('middle');self.wait(lambda:self.worker.state=='RANGE')
        self.bus.positions=dict.fromkeys(c.LABELS,10);time.sleep(.15)
        self.bus.positions=dict.fromkeys(c.LABELS,4090)
        self.assertTrue(self.worker.finished.wait(3));self.assertFalse(self.worker.saved)
        self.assertEqual(self.bus.calibration,self.bus.previous)
        self.assertTrue(any(k=='error' and '경계를' in str(v) for k,v in self.worker.events.queue))

    def test_file_save_failure_restores_motor_calibration(self):
        self.ranges();real=t.atomic_json
        def fail_file(path,value,**kwargs):
            if Path(path)==self.path:raise OSError('disk write failed')
            return real(path,value,**kwargs)
        with patch.object(t,'atomic_json',side_effect=fail_file):
            self.worker.commands.put('save');self.assertTrue(self.worker.finished.wait(3))
        self.assertEqual(self.bus.calibration,self.bus.previous);self.assertFalse(self.worker.saved)
        self.assertEqual(self.path.read_text(),'{"original":true}')
    def test_restore_failure_is_reported_and_keeps_recovery_evidence(self):
        self.ranges()
        with patch.object(self.bus,'write_calibration',side_effect=OSError('USB lost')):
            self.worker.cancel.set();self.assertTrue(self.worker.finished.wait(3))
        self.assertIn('USB lost',self.worker.rollback_error)
        evidence=json.loads((self.folder/'recovery.json').read_text())
        self.assertEqual(evidence['previous_eeprom'],c.serializable(self.bus.previous))


class EncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.hardware=t.Hardware(Path(__file__).parent/'follower.json',follower_only=True)
    def test_every_calibrated_integer_roundtrips_without_truncation_and_gripper_is_unchanged(self):
        bus=self.hardware.follower.bus
        for name,entry in self.hardware.follower.calibration.items():
            if name=='gripper':continue
            for raw in range(entry.range_min,entry.range_max+1):
                self.assertEqual(bus._unnormalize(bus._normalize({entry.id:raw}))[entry.id],raw)
        with self.assertRaises(ValueError):p.to_ticks(self.hardware,'gripper.pos',10)
    def test_read_uses_one_raw_sample_for_display_and_angles_without_extra_serial_reads(self):
        h=self.hardware;bus=h.follower.bus
        raw={n:(e.range_min+e.range_max)//2 for n,e in h.follower.calibration.items()}
        with patch.object(bus,'sync_read',return_value=raw) as read:
            lead,actual=h.read()
        read.assert_called_once_with('Present_Position',normalize=False);self.assertIsNone(lead)
        for name in c.LABELS:
            self.assertEqual(h.raw_positions['follower'][name+'.pos'],raw[name])
            if name!='gripper':self.assertEqual(p.to_ticks(h,name+'.pos',actual[name+'.pos']),raw[name])
    def test_one_tick_motion_is_not_arrival_without_encoder_change_and_repeats_accumulate(self):
        h=self.hardware;current={k:(30 if k=='gripper.pos' else p.from_ticks(h,k,2047)) for k in t.KEYS}
        goal=p.tick_goal(h,current,'shoulder_pan.pos',1)
        self.assertEqual(p.to_ticks(h,'shoulder_pan.pos',goal['shoulder_pan.pos']),2048)
        request={'axis':'shoulder_pan.pos','unit':'ticks','increment':1,'start':current}
        self.assertFalse(m.jog_arrived(request,goal,current));self.assertTrue(m.jog_arrived(request,goal,goal))
        outcome=m.jog_outcome(request,goal,current,False,10)
        second=p.tick_goal(h,current,'shoulder_pan.pos',1,outcome['reference'],10.1)
        self.assertEqual(p.to_ticks(h,'shoulder_pan.pos',second['shoulder_pan.pos']),2049)
        self.assertEqual(goal['gripper.pos'],30)
        with self.assertRaises(ValueError):p.tick_goal(h,current,'shoulder_pan.pos',.5)
        with self.assertRaises(ValueError):p.tick_goal(h,current,'gripper.pos',1)


class SavedCalibrationTests(unittest.TestCase):
    def test_imported_file_uses_exact_offsets_and_ranges_without_rehoming(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);bus=FakeBus();values=c.serializable(bus.previous)
            values['wrist_roll']['homing_offset']=-1650
            values['wrist_roll']['range_min']=0;values['wrist_roll']['range_max']=4095
            source=folder/'selected.json';source.write_text(json.dumps(values))
            target=folder/'follower.json';target.write_text('{"old":true}')
            arm=SimpleNamespace(id='follower',bus=bus,calibration_fpath=target,calibration=bus.previous)
            worker=c.CalibrationWorker(lambda:arm,folder/'recovery.json',preset=c.read_calibration_file(source))
            worker.thread.start();self.assertTrue(worker.finished.wait(3));worker.thread.join(1)
            self.assertTrue(worker.saved);self.assertEqual(json.loads(target.read_text()),values)
            self.assertEqual(c.serializable(bus.calibration),values)
            self.assertNotIn(('middle',),bus.calls)
            self.assertFalse(any(call[0]=='Operating_Mode' for call in bus.calls))
    def test_malformed_file_is_rejected_before_any_serial_access(self):
        bus=FakeBus();values=c.serializable(bus.previous);values['gripper']['range_max']=values['gripper']['range_min']
        with tempfile.TemporaryDirectory() as folder,patch('serial.Serial.open',side_effect=AssertionError('port opened')):
            factory=patch('builtins.print')
            with patch.object(c,'make_arm',side_effect=AssertionError('constructed arm')) as make:
                worker=c.CalibrationWorker(make,Path(folder)/'recover.json',preset=values)
                worker.thread.start();self.assertTrue(worker.finished.wait(3));worker.thread.join(1)
                make.assert_not_called();self.assertFalse(worker.saved)


class CalibrationWindowTests(unittest.TestCase):
    def test_file_picker_previews_six_motor_values_without_writes(self):
        import tkinter as tk
        root=tk.Tk()
        with tempfile.TemporaryDirectory() as folder:
            app=t.App(root,t.Demo(),folder);view=app.open_vision(show=False);view.open_motor_calibration();window=view.calibration_window
            file=Path(folder)/'saved.json';file.write_text(json.dumps(c.serializable(FakeBus().previous)))
            try:
                with patch('app_choices.askopenfilename',return_value=str(file)),patch('serial.Serial.open',side_effect=AssertionError('port opened')):
                    window.choose_button.invoke();root.update()
                self.assertEqual(window.table.set('gripper','min'),'300')
                self.assertEqual(window.table.set('wrist_roll','offset'),'5')
                self.assertIsNone(window.worker);self.assertFalse(window.apply_file_button.instate(['disabled']))
            finally:
                app.close()
                from tk_test_support import cancel_jobs
                cancel_jobs(root)
                root.destroy()

    def test_window_has_six_live_rows_and_waits_for_explicit_stage_buttons(self):
        import tkinter as tk
        root=tk.Tk();errors=[];root.report_callback_exception=lambda _t,e,_b:errors.append(str(e))
        with tempfile.TemporaryDirectory() as folder:
            backend=t.Demo();backend.identity={**backend.identity,'demo':False}
            app=t.App(root,backend,folder);view=app.open_vision(show=False)
            bus=FakeBus();path=Path(folder)/'arm.json'
            arm=SimpleNamespace(id='arm',bus=bus,calibration_fpath=path,calibration=bus.previous)
            view.open_motor_calibration();window=view.calibration_window
            def spin(predicate):
                end=time.monotonic()+4
                while time.monotonic()<end:
                    root.update()
                    if predicate():return
                    time.sleep(.01)
                self.fail(str((window.status.get(),errors)))
            try:
                self.assertIsNone(window.worker);self.assertFalse(bus.is_connected)
                window.port.set('/dev/serial/by-id/test');window.arm_id.set('arm')
                with patch.object(c,'make_arm',return_value=arm) as factory,patch.object(c,'carry_station_after_calibration'):
                    window.start();spin(lambda:window.worker and window.worker.state=='MIDPOINT')
                    with patch.object(t.Session,'start') as connect:
                        app.connect();connect.assert_not_called()
                    spin(lambda:window.table.set('gripper','current')=='2047')
                    self.assertFalse(path.exists());self.assertEqual(len(window.table.get_children()),6)
                    window.next();spin(lambda:window.worker.state=='RANGE')
                    spin(lambda:window.table.set('gripper','middle')=='2047')
                    self.assertEqual(window.table.set('wrist_roll','middle'),'2047')
                    bus.positions=dict.fromkeys(c.LABELS,1000);spin(lambda:window.table.set('gripper','min')=='1000')
                    bus.positions=dict.fromkeys(c.LABELS,3000);spin(lambda:window.table.set('gripper','max')=='3000')
                    window.next();spin(lambda:window.worker.state=='REVIEW')
                    self.assertFalse(path.exists());window.next()
                    spin(lambda:getattr(app,'calibration_reload_required',False) and window.worker.finished.is_set())
                    self.assertTrue(path.exists());self.assertFalse(bus.is_connected)
                    factory.assert_called_once()
            finally:
                if window.worker:window.worker.cancel.set();window.worker.thread.join(3)
                app.close()
                from tk_test_support import cancel_jobs
                cancel_jobs(root)
                root.destroy()
            self.assertEqual(errors,[])


if __name__=='__main__':unittest.main()
