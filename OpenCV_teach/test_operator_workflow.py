"""Direct operator actions without repeated prompts; controller limits retained."""
from copy import deepcopy
import tempfile
import time
import unittest
from unittest.mock import patch

import manual_teaching as m
import teach_left_arm as t


class DirectOperationTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        self.root=tk.Tk();self.folder=tempfile.TemporaryDirectory()
        self.backend=t.Demo();self.backend.follower_only=True
        self.backend.identity={**t.Demo.identity,'demo':False}
        self.app=t.App(self.root,self.backend,self.folder.name)

    def tearDown(self):
        self.app.session.stop.set();self.app.session.wakeup.set()
        if self.app.session.thread:self.app.session.thread.join(3)
        if self.app.vision:self.app.vision.close()
        for job in self.root.tk.call('after','info'):self.root.after_cancel(job)
        self.root.destroy();self.folder.cleanup()

    def test_follower_hold_and_readonly_connect_do_not_open_confirmation(self):
        with patch('tkinter.messagebox.askokcancel',side_effect=AssertionError('Repeated popup')),patch.object(t.Session,'start') as start:
            self.app.connect();self.assertEqual(start.call_count,1)
            self.assertFalse(self.app.session.auto_sync)
            self.app.state='DISCONNECTED';self.app.connect(auto_sync=False)
            self.assertEqual(start.call_count,2);self.assertFalse(self.app.session.auto_sync)

    def test_fixed_step_dispatches_once_without_popup_but_busy_and_foreign_modes_still_block(self):
        self.app.steps=[{'name':'자세','pose':dict.fromkeys(t.KEYS,0.)}]
        self.app.refresh_steps(0);self.app.state='HOLD'
        with patch('tkinter.messagebox.askokcancel',side_effect=AssertionError('Repeated popup')),patch.object(self.app,'send') as send:
            self.app.move();send.assert_called_once_with('move',self.app.steps[0]['pose'])
            send.reset_mock();self.app.state='MOVE';self.app.move();send.assert_not_called()
            self.app.state='HOLD';self.app.preview_only=True;self.app.move();send.assert_not_called()

    def test_pair_connection_retains_one_short_confirmation_and_cancel_sends_nothing(self):
        self.app.follower_only=False;self.backend.follower_only=False
        with patch('tkinter.messagebox.askokcancel',return_value=False) as ask,patch.object(t.Session,'start') as start:
            self.app.connect();ask.assert_called_once();start.assert_not_called()
        with patch('tkinter.messagebox.askokcancel',return_value=True) as ask,patch.object(t.Session,'start') as start:
            self.app.connect();ask.assert_called_once();start.assert_called_once()

    def test_whole_episode_retains_one_confirmation(self):
        self.app.steps=[{'name':'자세','pose':dict.fromkeys(t.KEYS,0.)}]
        self.app.teaching_title='시험';self.app.state='HOLD'
        self.app.latest={'sample_time':time.monotonic(),'follower':dict.fromkeys(t.KEYS,0.)}
        self.app.saved_path=t.save_named_teaching(self.folder.name,self.app.document())
        with patch('tkinter.messagebox.askokcancel',return_value=True) as ask,patch.object(self.app,'send') as send:
            self.app.play();ask.assert_called_once();self.assertEqual(send.call_args.args[0],'play')


class CoarseAdjustmentTests(unittest.TestCase):
    def test_five_degree_adjustment_starts_at_measured_pose_without_adding_pending_residual(self):
        current=dict.fromkeys(t.KEYS,0.);held={**current,'shoulder_pan.pos':.3,'elbow_flex.pos':.2}
        reference={'axis':'shoulder_pan.pos','requested':4.,'measured':0.,'direction':1,'at':10.}
        goal=m.jog_goal(current,None,'shoulder_pan.pos',5.,reference,10.1,held)
        self.assertEqual(goal['shoulder_pan.pos'],5.)
        self.assertEqual(goal['elbow_flex.pos'],.2)
        self.assertEqual(m.checked_jog({'start':current,'goal':goal,'sample_time':10.},current,10.1,t.pose),goal)
        with self.assertRaises(ValueError):m.jog_goal(current,None,'shoulder_pan.pos',5.1)
        with self.assertRaises(ValueError):m.checked_jog({'start':current,'goal':goal,'sample_time':9.},current,10.1,t.pose)
        hardware=t.Hardware.__new__(t.Hardware);hardware.bounds={k:(-2.,2.) for k in t.KEYS}
        with self.assertRaisesRegex(ValueError,'calibration range'):
            m.checked_jog({'start':current,'goal':goal,'sample_time':10.},current,10.1,hardware.validate_target)


if __name__=='__main__':unittest.main()
