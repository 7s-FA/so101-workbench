from copy import deepcopy
import json
from pathlib import Path
import queue
import tempfile
import time
import unittest
from unittest.mock import patch

import teach_left_arm as t
from follower_setup import register_follower


class FollowerTests(unittest.TestCase):
    def test_hardware_initialization_never_constructs_leader_or_opens_usb(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'follower_right.json'
            data={name:dict(id=i,drive_mode=0,homing_offset=0,range_min=1000,range_max=3000)
                  for i,name in enumerate(t.JOINTS,1)}
            source.write_text(json.dumps(data))
            profile=register_follower(Path(tmp)/'follower.json','/dev/serial/by-id/test-follower',source,'test_arm')
            with patch('serial.Serial.open',side_effect=AssertionError('USB opened')),patch('diagnose_shoulder_lift.make_arms',side_effect=AssertionError('Leader created')):
                backend=t.Hardware(profile,follower_only=True)
            self.assertIsNone(backend.leader)
            self.assertFalse(backend.identity['demo'])
            self.assertEqual(backend.identity['side'],'right')
            self.assertEqual(set(backend.identity['serials']),{'follower'})
            self.assertEqual(set(backend.identity['calibration_sha256']),{'follower_right'})
            self.assertEqual(backend.follower.calibration_fpath.read_bytes(),source.read_bytes())
            with patch('diagnose_shoulder_lift.ensure_ports_available') as check:
                backend.preflight();check.assert_called_once_with(['/dev/serial/by-id/test-follower'])

    def test_follower_session_starts_at_measured_pose_and_blocks_leader_commands(self):
        backend=t.Demo();backend.follower_only=True
        backend.current=dict.fromkeys(t.KEYS,12.);backend.target=backend.current.copy()
        backend.lead=dict.fromkeys(t.KEYS,75.)
        with tempfile.TemporaryDirectory() as tmp:
            session=t.Session(backend,tmp,auto_sync=True);session.start();events=[]
            def wait(condition,timeout=4):
                deadline=time.monotonic()+timeout
                while time.monotonic()<deadline:
                    session.heartbeat=time.monotonic()
                    try:
                        while True:events.append(session.events.get_nowait())
                    except queue.Empty:pass
                    if condition():return
                    time.sleep(.005)
                self.fail(f'{session.state}: {events[-4:]}')
            try:
                wait(lambda:session.state=='READY')
                session.command('hold_current')
                wait(lambda:session.state=='HOLD')
                self.assertEqual(backend.target,dict.fromkeys(t.KEYS,12.))
                session.command('teach');session.command('sync',{'leader':backend.lead,'follower':backend.current})
                wait(lambda:any(kind=='notice' and '리더 동작' in value for kind,value in events))
                self.assertEqual(session.state,'HOLD')
                wait(lambda:len([v for k,v in events if k=='status'])>5)
                self.assertTrue(all(value['leader'] is None for kind,value in events if kind=='status'))
                # The same stable measured capture contract applies without a leader.
                end=time.monotonic()+.7
                while time.monotonic()<end:session.heartbeat=time.monotonic();time.sleep(.01)
                session.command('capture','측정 자세')
                wait(lambda:any(kind=='captured' for kind,_ in events))
                captured=next(value for kind,value in events if kind=='captured')
                self.assertEqual(captured['pose'],dict.fromkeys(t.KEYS,12.))
                goal=backend.current.copy();goal['shoulder_pan.pos']=15.
                session.command('move',goal)
                wait(lambda:session.state=='MOVE')
                wait(lambda:session.state=='HOLD')
                self.assertAlmostEqual(backend.current['shoulder_pan.pos'],15.,delta=.5)
            finally:
                session.stop.set();session.wakeup.set();session.thread.join(3)
            self.assertFalse(session.thread.is_alive())
            report=json.loads(next(Path(tmp).glob('session_*.json')).read_text())
            self.assertTrue(any(e.get('command')=='activate_at_current_pose' for e in report['events']))


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        self.root=tk.Tk();self.root.geometry('1080x720')
        self.temp=tempfile.TemporaryDirectory()
        self.backend=t.Demo();self.backend.follower_only=True
        # Test-only hardware identity proves serialization, without pretending to use a device.
        self.backend.identity={'demo':False,'control_mode':'follower_only','side':'right',
            'serials':{'follower':'TEST-ONLY'},'calibration_sha256':{'follower_right':'TEST-ONLY'}}
        self.app=t.App(self.root,self.backend,self.temp.name);self.root.update()

    def tearDown(self):
        if self.app.vision:self.app.vision.close()
        from tk_test_support import cancel_jobs
        cancel_jobs(self.root)
        self.root.destroy();self.temp.cleanup()

    def test_manual_edit_and_save_keep_real_follower_identity(self):
        self.assertNotIn('leader',self.app.joints['displaycolumns'])
        self.assertEqual(str(self.app.buttons['teach']['state']),'disabled')
        self.app.steps=[{'name':'수동 자세','pose':dict.fromkeys(t.KEYS,0.),'captured_at':''}]
        self.app.refresh_steps(0);self.app.edit_step_pose()
        editor=self.app.pose_editor
        editor.values['shoulder_pan.pos'].set('15')
        with patch.object(self.app.session,'command',side_effect=AssertionError('Editing must not move hardware')):
            self.app.update_step_pose(self.app.steps[0],dict.fromkeys(t.KEYS,15.))
        editor.close()
        with patch('teach_left_arm.ask_teaching_title',return_value='팔로워 수동 테스트'):self.app.save_as()
        document=json.loads(self.app.saved_path.read_text())
        self.assertFalse(document['identity']['demo'])
        self.assertEqual(document['identity']['control_mode'],'follower_only')
        self.assertEqual(document['steps'][0]['pose']['shoulder_pan.pos'],15.)

    def test_in_app_choice_keyboard_selection_and_space_does_not_capture(self):
        from app_choices import AppChoice
        choice=self.app.library;choice['values']=('첫 번째','두 번째');choice.current(0)
        self.assertIsInstance(choice,AppChoice)
        changed=[];choice.bind('<<ComboboxSelected>>',lambda e:changed.append(choice.get()))
        self.app.state='HOLD'
        import types
        with patch.object(self.app,'send') as send:
            self.app.space_press(types.SimpleNamespace(widget=choice.button));send.assert_not_called()
        choice.open();self.root.update()
        self.assertEqual(choice.panel.winfo_toplevel(),self.root)
        self.assertGreaterEqual(choice.panel.winfo_x(),0)
        self.assertLessEqual(choice.panel.winfo_x()+choice.panel.winfo_width(),self.root.winfo_width())
        choice.listbox.selection_clear(0,'end');choice.listbox.selection_set(1);choice.choose();self.root.update()
        self.assertEqual(changed,['두 번째']);self.assertIsNone(choice.panel)
        choice.configure(state='disabled');choice.open();self.assertIsNone(choice.panel)

    def test_file_picker_uses_in_window_list_and_filters(self):
        from app_choices import askopenfilename
        root=Path(self.temp.name);(root/'target.json').write_text('{}');(root/'ignore.txt').write_text('')
        def select():
            def walk(w):
                for child in w.winfo_children():
                    yield child;yield from walk(child)
            tree=next(w for w in walk(self.root) if w.winfo_class()=='Treeview' and w is not self.app.table and w is not self.app.joints)
            rows=tree.get_children();self.assertEqual(len(rows),1)
            tree.selection_set(rows[0]);tree.event_generate('<Return>')
        self.root.after(60,select)
        chosen=askopenfilename(parent=self.root,initialdir=root,filetypes=[('JSON','*.json')])
        self.assertEqual(chosen,str(root/'target.json'))


if __name__=='__main__':unittest.main()
