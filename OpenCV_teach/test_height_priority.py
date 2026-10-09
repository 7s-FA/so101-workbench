from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import json,unittest
import numpy as np
import vision_teaching as v
from height_planner import height_pose

FIXTURE=json.loads((Path(__file__).parent/'tests/fixtures/height_priority.json').read_text())

class HeightPriorityTests(unittest.TestCase):
    def test_cycle_prioritizes_height_and_aims_at_left_contact_in_upper_pose(self):
        d=deepcopy(FIXTURE['document']);s=FIXTURE['station'];target=next(iter(d['vision_plan']['targets'].values()))
        obs={v.reference_key(target):FIXTURE['observation']};before=deepcopy(d)
        c=v.compile_steps(d['steps'],d['vision_plan'],s,obs,d['steps'][0]['pose'])
        k=v.Kinematics(s['tool']);wrist=c[0]['pose']['wrist_roll.pos']
        grasp=c[3]['pose']['wrist_roll.pos']
        self.assertLess(abs(c[1]['pose']['wrist_roll.pos']-grasp),5.)
        self.assertLess(abs(c[4]['pose']['wrist_roll.pos']-grasp),5.)
        self.assertGreater(abs(c[1]['pose']['wrist_roll.pos']-wrist),5.)
        self.assertAlmostEqual(k.fk(c[1]['pose'])[2,3],target['z_mm']+target['hover_mm'],delta=.5)
        self.assertAlmostEqual(k.fk(c[4]['pose'])[2,3],target['z_mm']+target['retreat_mm'],delta=.5)
        descent=[c[1]['pose']]+[n['pose'] for n in c[2]['path']]
        zs=[k.fk(p)[2,3] for p in descent]
        self.assertTrue(all(b<=a+.05 for a,b in zip(zs,zs[1:])))
        self.assertLess(max(abs(a['wrist_roll.pos']-b['wrist_roll.pos']) for a,b in zip(descent,descent[1:])),3)
        self.assertEqual(d,before)
        self.assertTrue(all(not x['precision'] for x in v.expand_steps(c) if x.get('cartesian')))
        # Both ends retain precisely the original fixed safe joints.
        self.assertEqual(c[0]['pose'],c[-1]['pose'])
    def test_impossible_height_is_not_reported_as_reached(self):
        d=FIXTURE['document'];k=v.Kinematics(FIXTURE['station']['tool']);seed=d['steps'][0]['pose']
        with self.assertRaises(ValueError):height_pose(k,[200,0,1000],seed,50)
    def test_preview_recomputes_with_current_jig_without_motor_commands_or_saved_edits(self):
        from robot_preview import PreviewPanel
        d=deepcopy(FIXTURE['document']);target=next(iter(d['vision_plan']['targets'].values()));key=v.reference_key(target)
        view=SimpleNamespace(station=FIXTURE['station'],job=None,frame_live=True,
            reference_acquisition=SimpleNamespace(pending=None),status=Mock())
        view.reference_acquisition.begin=Mock(side_effect=lambda plan,cb,**kw:cb())
        current={key:deepcopy(FIXTURE['observation'])}
        view.reference_observations=lambda plan:deepcopy(current)
        view.run_job=lambda work,done,**kw:done(work())
        app=SimpleNamespace(steps=d['steps'],vision_plan=d['vision_plan'],closing=False,
            pose_editor=None,name_editor=None,preview_only=False,state='DISCONNECTED',
            document=lambda:deepcopy(d),open_vision=lambda **kw:view,note=Mock(),send=Mock())
        panel=PreviewPanel.__new__(PreviewPanel);panel.app=app;panel.closed=False;panel.render_error=None;panel.status=Mock()
        panel.start_simulation_document=Mock()
        before=deepcopy(d);panel.play_simulation()
        first=panel.start_simulation_document.call_args.args[0]
        current[key]['x_mm']+=2.;panel.play_simulation()
        second,label,scene=panel.start_simulation_document.call_args.args
        self.assertEqual(view.reference_acquisition.begin.call_count,2)
        self.assertNotEqual(first['steps'][2]['pose'],second['steps'][2]['pose'])
        self.assertIn('지그 측정 보정 적용',label);self.assertEqual(scene[0]['x'],current[key]['x_mm'])
        self.assertEqual(d,before);app.send.assert_not_called();app.note.assert_not_called()

class PlanningResponsivenessTests(unittest.TestCase):
    def test_isolated_height_calculation_keeps_tk_heartbeat_responsive(self):
        import tkinter as tk,time,threading,queue
        from planning_worker import compile_steps_isolated
        root=tk.Tk();root.withdraw();ticks=[];results=queue.Queue();started=time.monotonic();finished=False
        d=deepcopy(FIXTURE['document']);target=next(iter(d['vision_plan']['targets'].values()))
        obs={v.reference_key(target):FIXTURE['observation']}
        def tick():
            ticks.append(time.monotonic())
            if not finished:root.after(20,tick)
        def solve():
            try:results.put(compile_steps_isolated(d['steps'],d['vision_plan'],FIXTURE['station'],obs,d['steps'][0]['pose']))
            except Exception as exc:results.put(exc)
        thread=threading.Thread(target=solve,daemon=True);thread.start();tick()
        try:
            while results.empty() and time.monotonic()-started<30:root.update();time.sleep(.005)
            self.assertFalse(results.empty(),'Planning did not finish')
            value=results.get_nowait()
            if isinstance(value,Exception):raise value
            self.assertEqual(len(value),9);self.assertGreater(len(ticks),20)
            self.assertLess(max(b-a for a,b in zip(ticks,ticks[1:])),1.,'Tk heartbeat stalled')
        finally:
            finished=True
            for job in root.tk.call('after','info'):root.after_cancel(job)
            root.destroy()
