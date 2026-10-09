"""Bounded follower adjustment and measured teaching, using no hardware."""
from copy import deepcopy
import json
from pathlib import Path
import queue
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

import teach_left_arm as t
import vision_teaching as v
import manual_teaching as m
from test_vision_teaching import target


class GeometryTests(unittest.TestCase):
    def test_fixed_and_camera_relative_capture_preserve_xyz_and_full_orientation(self):
        tool=v.default_station({})['tool'];q=dict(zip(t.KEYS,[15,-30,45,10,50,35]))
        expected=v.Kinematics(tool).fk(q)
        for mode in ('fixed','rectangle'):
            original=target(mode=mode);ref=dict(x_mm=130.,y_mm=-20.,yaw_deg=37.)
            captured=m.measured_target(original,q,tool,ref if mode=='rectangle' else None)
            xyz,rot=v.target_pose(captured,{v.reference_key(captured):ref})
            np.testing.assert_allclose(xyz,expected[:3,3],atol=1e-9)
            np.testing.assert_allclose(rot,expected[:3,:3],atol=1e-9)
            self.assertEqual(original['x_mm'],0)
            self.assertEqual(captured['geometry_verified'],original['geometry_verified'])
        with self.assertRaisesRegex(ValueError,'지그 위치'):m.measured_target(target(),q,tool)

    def test_small_joint_and_cartesian_moves_and_stale_drift_bounds(self):
        tool=v.default_station({})['tool'];seed=dict(zip(t.KEYS,[0,-30,50,60,0,50]))
        goal=m.jog_goal(seed,tool,'elbow_flex.pos',.5)
        self.assertEqual(goal['elbow_flex.pos'],50.5);self.assertEqual(seed['elbow_flex.pos'],50)
        with self.assertRaises(ValueError):m.jog_goal(seed,tool,'elbow_flex.pos',6)
        with self.assertRaises(ValueError):m.jog_goal(seed,tool,'gripper.pos',6)
        q=v.Kinematics(tool).ik(np.array([200.,0,30]),np.diag([-1.,1.,-1.]),seed,50)
        moved=m.jog_goal(q,tool,'Z',1)
        np.testing.assert_allclose(v.Kinematics(tool).fk(moved)[:3,3],v.Kinematics(tool).fk(q)[:3,3]+[0,0,1],atol=.5)
        request={'start':seed,'goal':goal,'sample_time':10.}
        self.assertEqual(m.checked_jog(request,seed,10.1,t.pose),goal)
        for current,now,goal_change in [(seed,11,{}),({**seed,'shoulder_pan.pos':2},10.1,{}),(seed,10.1,{'elbow_flex.pos':70})]:
            with self.assertRaises(ValueError):m.checked_jog({**request,'goal':{**goal,**goal_change}},current,now,t.pose)


class SessionTests(unittest.TestCase):
    def test_jog_rejects_expired_or_large_moves_and_measures_actual_follower(self):
        backend=t.Demo();backend.follower_only=True
        with tempfile.TemporaryDirectory() as folder:
            session=t.Session(backend,folder,auto_sync=True);events=[];session.start()
            def wait(predicate,timeout=5):
                end=time.monotonic()+timeout
                while time.monotonic()<end:
                    session.heartbeat=time.monotonic()
                    try:
                        while True:events.append(session.events.get_nowait())
                    except queue.Empty:pass
                    if predicate():return
                    time.sleep(.005)
                self.fail(str((session.state,events[-6:])))
            def request(token,goal,age=0):
                session.command('jog',dict(token=token,start=backend.current.copy(),goal=goal,sample_time=time.monotonic()-age))
            try:
                wait(lambda:session.state=='READY');session.command('hold_current')
                wait(lambda:session.state=='HOLD')
                request('expired',{**backend.current,'shoulder_pan.pos':1},age=2)
                wait(lambda:any(k=='manual_result' and e['token']=='expired' for k,e in events))
                self.assertFalse(next(e for k,e in events if k=='manual_result' and e['token']=='expired')['ok'])
                request('too_far',{**backend.current,'shoulder_pan.pos':20})
                wait(lambda:any(k=='manual_result' and e['token']=='too_far' for k,e in events))
                self.assertEqual(session.state,'HOLD')
                request('good',{**backend.current,'shoulder_pan.pos':2})
                wait(lambda:session.state=='MOVE')
                request('busy',{**backend.current,'shoulder_pan.pos':3})
                wait(lambda:any(k=='manual_result' and e['token']=='busy' for k,e in events))
                self.assertFalse(next(e for k,e in events if k=='manual_result' and e['token']=='busy')['ok'])
                wait(lambda:session.state=='HOLD')
                self.assertAlmostEqual(backend.current['shoulder_pan.pos'],2,delta=.35)
                stable_until=time.monotonic()+1
                wait(lambda:time.monotonic()>stable_until)
                session.command('measure_target',dict(token='measure',start=backend.current.copy(),sample_time=time.monotonic()))
                wait(lambda:any(k=='manual_result' and e['token']=='measure' for k,e in events))
                measured=next(e for k,e in events if k=='manual_result' and e['token']=='measure')
                self.assertTrue(measured['ok'],measured)
                self.assertAlmostEqual(measured['pose']['shoulder_pan.pos'],backend.current['shoulder_pan.pos'])
            finally:
                session.stop.set();session.wakeup.set();session.thread.join(3)
            self.assertFalse(session.thread.is_alive())


class InterfaceTests(unittest.TestCase):
    def test_real_adjustment_workflow_without_leader_or_camera(self):
        import tkinter as tk
        root=tk.Tk();errors=[];root.report_callback_exception=lambda _t,e,_b:errors.append(e)
        with tempfile.TemporaryDirectory() as folder:
            backend=t.Demo();backend.follower_only=True;app=t.App(root,backend,folder)
            try:
                view=app.open_vision();manual=view.manual;view.select_tab(4)
                self.assertIsNone(view.process)
                manual.refresh();self.assertTrue(all(str(b['state'])=='disabled' for b in manual.buttons))
                app.connect()
                def wait(predicate,timeout=6):
                    end=time.monotonic()+timeout
                    while time.monotonic()<end:
                        root.update()
                        if predicate():return
                        time.sleep(.01)
                    self.fail(str((app.state,manual.status.get(),view.status.get(),errors)))
                wait(lambda:app.state=='READY' and app.latest is not None)
                app.send('hold_current');wait(lambda:app.state=='HOLD')
                manual.jog('shoulder_pan.pos',1)
                wait(lambda:app.state=='MOVE')
                wait(lambda:app.state=='HOLD' and not manual.pending)
                self.assertAlmostEqual(app.latest['follower']['shoulder_pan.pos'],.5,delta=.35)
                end=time.monotonic()+1;wait(lambda:time.monotonic()>end)
                station_before=deepcopy(view.station['tool'])
                view.vars['t_mode'].set('이 자리에 고정');manual.measure('target')
                wait(lambda:manual.pending is None)
                self.assertIsNotNone(view.target_id,manual.status.get())
                self.assertEqual(app.vision_plan['targets'][view.target_id]['mode'],'fixed')
                self.assertEqual(view.station['tool'],station_before)
                manual.measure('step');wait(lambda:manual.pending is None)
                self.assertEqual(len(app.steps),1,manual.status.get())
                self.assertNotIn('motion',app.steps[0]);self.assertTrue(app.steps[0]['captured_at'])
                saved=json.loads(app.program_path.read_text())
                self.assertEqual(saved['steps'][0]['pose'],app.steps[0]['pose'])
                self.assertIsNone(view.process)
                manual.pause();wait(lambda:app.state=='HOLD')
                jobs=[]
                with patch.object(view,'run_job',side_effect=lambda f,d:jobs.append((f,d))):
                    manual.jog('shoulder_pan.pos',1)
                manual.pause()
                with patch.object(app,'send') as send:
                    jobs[0][1](jobs[0][0]())
                send.assert_not_called()  # Stop wins over delayed IK completion.
                self.assertFalse(errors,errors)
            finally:
                app.session.stop.set();app.session.wakeup.set()
                if app.session.thread:app.session.thread.join(3)
                if app.vision:app.vision.close()
                if app.preview:app.preview.close()
                for job in root.tk.call('after','info'):root.after_cancel(job)
                root.destroy()


if __name__=='__main__':unittest.main()
