"""The monitoring button never opens another port or queues motor commands."""
import json
from pathlib import Path
import tempfile
import time
import tkinter as tk
from tkinter import ttk
import unittest
from unittest.mock import patch

import teach_left_arm as t
from follower_setup import register_follower
from motor_diagnostics import demo_frame


class StatusMonitorTests(unittest.TestCase):
    def setUp(self):
        try:self.root=tk.Tk()
        except tk.TclError as exc:self.skipTest(str(exc))
        self.errors=[]
        self.root.report_callback_exception=lambda _t,error,_trace:self.errors.append(error)
        self.temp=tempfile.TemporaryDirectory()
        self.app=None

    def tearDown(self):
        if not hasattr(self,'temp'):return
        if self.app:self.app.close()
        def widgets(w):
            for child in w.winfo_children():yield child;yield from widgets(child)
        owners=[self.root,*widgets(self.root)]
        for job in self.root.tk.call('after','info'):
            script=self.root.tk.call('after','info',job)[0]
            owner=next((w for w in owners if script in (w._tclCommands or [])),self.root)
            owner.after_cancel(job)
        self.root.destroy();self.temp.cleanup()
        self.assertEqual(self.errors,[])

    def wait(self,condition):
        until=time.monotonic()+2
        while time.monotonic()<until:
            self.root.update()
            if condition():return
            time.sleep(.01)
        self.fail('monitor update timed out')

    def test_demo_button_available_during_motion_and_reuses_hidden_window(self):
        self.app=t.App(self.root,t.Demo(),self.temp.name)
        button=self.app.buttons['monitor']
        self.assertEqual(button.cget('text'),'장비 모니터링')
        with patch('serial.Serial.open',side_effect=AssertionError('opened port')),patch.object(self.app,'send',side_effect=AssertionError('motor command')):
            button.invoke();self.root.update();view=self.app.motor_dashboard
            self.assertTrue(view.embedded);self.assertTrue(view.demo)
            self.assertEqual(len(view.table.get_children()),6)
            for state in ('DISCONNECTED','READY','HOLD','PLAY','MOVE','TEACH'):
                self.app.state=state;self.app.refresh_buttons()
                self.assertFalse(button.instate(['disabled']),state)
                view.close();self.root.update()
                self.assertEqual(view.root.state(),'withdrawn')
                button.invoke();self.root.update()
                self.assertIs(self.app.motor_dashboard,view)
                self.assertEqual(view.root.state(),'normal')
            # Inspection stays available while a foreign episode is being edited.
            self.app.preview_only=True;self.app.vision_busy=True;self.app.refresh_buttons()
            self.assertFalse(button.instate(['disabled']))

    def test_camera_window_button_opens_the_same_monitor(self):
        self.app=t.App(self.root,t.Demo(),self.temp.name)
        monitor=self.app.open_monitor();monitor.close()
        vision=self.app.open_vision();self.root.update()
        def widgets(w):
            for child in w.winfo_children():yield child;yield from widgets(child)
        self.app.open_settings()
        button=vision.monitor_button
        with patch.object(self.app,'send',side_effect=AssertionError('motor command')):
            button.invoke();self.root.update()
        self.assertIs(self.app.motor_dashboard,monitor)
        self.assertEqual(monitor.root.state(),'normal')

    def test_default_real_setup_shares_follower_bus_without_io_and_accepts_updates(self):
        base=Path(self.temp.name);source=base/'follower_right.json'
        source.write_text(json.dumps({j:dict(id=i,drive_mode=0,homing_offset=0,range_min=1000,range_max=3000)
                                    for i,j in enumerate(t.JOINTS,1)}))
        profile=register_follower(base/'follower.json','/dev/serial/by-id/test',source,'test')
        with patch('serial.Serial.open',side_effect=AssertionError('opened port')):
            backend=t.Hardware(profile,follower_only=True)
            bus=backend.follower.bus
            with patch.object(bus,'read',side_effect=AssertionError('UI read')),patch.object(bus,'write',side_effect=AssertionError('UI write')),patch.object(bus,'connect',side_effect=AssertionError('UI connected')):
                self.app=t.App(self.root,backend,base)
                collector=backend.motor_dashboard
                self.assertIs(collector.bus,bus)
                self.assertFalse(collector.path.exists())
                view=self.app.open_monitor();self.root.update()
                self.assertIs(view.collector,collector);self.assertIsNone(view.thread)
                self.assertIn('측정 대기',view.status.get())
                self.assertFalse(view.connect_button.winfo_ismapped())
                frame=demo_frame();frame['demo']=False;frame['context']['state']='PLAY'
                collector.emit(frame);self.wait(lambda:view.table.exists('1'))
                self.assertEqual(view.table.set('1','voltage'),'123')
                self.assertEqual(view.table.set('1','temp'),'31')
                view.metric.set('전압');view.draw()
                labels=[view.canvas.itemcget(i,'text') for i in view.canvas.find_all() if view.canvas.type(i)=='text']
                self.assertIn('전압',labels)
                self.assertTrue(any(label.isdigit() and int(label)>=123 for label in labels),labels)
                view.close();self.assertIs(backend.motor_dashboard,collector)
                collector.finish('stopped',torque_off_verified=True)
                self.app.open_monitor();self.wait(lambda:'연결 종료' in view.status.get())
                self.assertIn('연결 종료',view.status.get())
                self.assertEqual(view.table.set('1','voltage'),'123')

    def test_monitor_button_fits_compact_header_and_disposes_timer(self):
        self.app=t.App(self.root,t.Demo(),self.temp.name)
        self.root.geometry('1080x720');self.root.update()
        self.app.open_settings();self.root.update()
        button=self.app.vision.monitor_button
        self.assertTrue(button.winfo_ismapped())
        self.assertLessEqual(button.winfo_rootx()-self.root.winfo_rootx()+button.winfo_width(),1080)
        view=self.app.open_monitor();self.root.update();job=view.poll_job
        self.assertIn(job,self.root.tk.call('after','info'))
        self.app.close()
        self.assertNotIn(job,self.root.tk.call('after','info'))
        self.assertFalse(view.root.winfo_exists())


if __name__=='__main__':unittest.main()
