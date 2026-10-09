from copy import deepcopy
import queue
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

import teach_left_arm as teaching
import vision_teaching as v
from safe_episode import model_safe_pose,with_safe_endpoints,validate_boundaries
from jig_detection import profile_signature
from test_vision_teaching import target


class BoundaryTests(unittest.TestCase):
    def test_offsets_and_persisted_endpoints_preserve_fixed_robot_pose(self):
        tool=v.default_station({})['tool'];tool['joint_offsets_deg']=[1,2,3,4,5]
        config={'home_deg':[0]*6,'safe_deg':[0,-55,15,85,90,15],
                'tcp':{'joint_offsets_deg':[0]*5}}
        pose=model_safe_pose(config,tool)
        self.assertEqual([pose[n+'.pos'] for n in v.JOINTS],[-1,-57,12,81,85])
        t=target(return_to_start=True);station=v.default_station(teaching.Demo.identity)
        steps=with_safe_endpoints(v.make_steps('t',t,pose),pose)
        self.assertEqual(len(with_safe_endpoints(steps,pose)),9)
        doc={'schema':2,'identity':teaching.Demo.identity,'units':teaching.UNITS,'steps':steps,
             'vision_plan':{'targets':{'t':t},'calibration_signature':v.calibration_signature(station)}}
        clean=teaching.validate_program(doc,doc['identity'])
        self.assertEqual(clean[0]['safe_boundary'],'start');self.assertEqual(clean[-1]['safe_boundary'],'end')
        bad=deepcopy(steps);bad[-1]['pose']['shoulder_pan.pos']+=1
        with self.assertRaises(ValueError):validate_boundaries(bad)
        with self.assertRaises(ValueError):validate_boundaries(steps[1:])


class SafeFlowTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        self.root=tk.Tk();self.folder=tempfile.TemporaryDirectory()
        self.app=teaching.App(self.root,teaching.Demo(),self.folder.name)
        self.view=self.app.open_vision();self.view.vars['t_return_to_start'].set('집고 제자리에 놓기');self.view.add_steps()
        self.target=self.view.read_target();self.key=v.reference_key(self.target)
        self.safe=dict.fromkeys(teaching.KEYS,0.)
        self.app.steps=with_safe_endpoints(self.app.steps,self.safe);self.app.refresh_steps(0)
        self.app.state='HOLD';self.now=100.
        self.app.latest={'sample_time':self.now,'follower':deepcopy(self.safe)}
        self.view.frame_live=True;self.view.frame_at=self.now
        self.value={'x_mm':200.,'y_mm':0.,'yaw_deg':0.,'quality':'precise','profile_signature':profile_signature(self.target)}
        self.clock=patch('time.monotonic',side_effect=lambda:self.now);self.clock.start()
        self.commands=patch.object(self.app.session,'command');self.sent=self.commands.start()
        self.confirm=patch('tkinter.messagebox.askokcancel',return_value=True);self.confirm.start()
        self.camera=patch.object(self.view,'process_frame');self.camera.start()

    def tearDown(self):
        self.camera.stop();self.confirm.stop();self.commands.stop();self.clock.stop()
        self.view.close()
        if self.app.preview:self.app.preview.close()
        for job in self.root.tk.call('after','info'):self.root.after_cancel(job)
        self.root.destroy();self.folder.cleanup()

    def arrive(self):
        self.view.frame_live=True;self.view.frame_at=self.now
        self.view.execute();name,payload=self.sent.call_args.args
        self.assertEqual(name,'play');self.assertEqual(len(payload['steps']),1)
        self.assertEqual(payload['steps'][0]['safe_boundary'],'start')
        self.now=101.;self.app.latest['sample_time']=self.now
        self.view.safe_execution.completed({'token':payload['request_id'],'ok':True,'sample_time':self.now,'pose':self.safe})
        self.view.safe_execution.poll()

    def test_measurement_starts_after_tagged_safe_arrival_and_then_runs_remaining_steps(self):
        for i in range(31):self.view.tracker.observe(self.key,self.value,97+i*.1,i)
        with patch.object(self.view,'execute_motion') as resume:
            self.arrive()
            self.assertFalse(self.view.tracker.history)
            for i in range(31):
                self.now=101+i*.1;self.app.latest['sample_time']=self.now
                self.view.frame_live=True;self.view.frame_at=self.now
                self.view.tracker.observe(self.key,{**self.value,'quality':'precise' if i in (0,10,20) else 'medium'},self.now,100+i)
                self.view.safe_execution.poll()
                if i<1:resume.assert_not_called()
            resume.assert_called_once_with(prepared=True)
            self.assertIsNone(self.app.navigation.current)
            self.assertIsNone(self.view.safe_execution.pending)

    def test_episode_opens_camera_only_when_safe_arrival_starts_measurement(self):
        self.app.saved_path=self.app.output/'selected.json'
        self.app.navigation.home();self.assertIsNone(self.app.navigation.current)
        self.app.play()
        self.assertIsNone(self.app.navigation.current)
        name,payload=self.sent.call_args.args
        self.assertEqual(name,'play');self.assertEqual(len(payload['steps']),1)
        self.assertEqual(self.view.safe_execution.pending['phase'],'entry')
        self.view.safe_execution.poll();self.assertIsNone(self.app.navigation.current)
        self.now=101.;self.app.latest['sample_time']=self.now;self.view.frame_at=self.now
        self.view.safe_execution.completed({'token':payload['request_id'],'ok':True,'sample_time':self.now,'pose':self.safe})
        self.view.safe_execution.poll()
        self.assertIs(self.app.navigation.current,self.view.window)
        self.assertTrue(self.view.checking())
        self.assertEqual(self.view.safe_execution.pending['phase'],'measuring')

    def test_stop_camera_pause_and_revision_change_never_resume(self):
        for cancellation in ('pause','camera','revision'):
            with self.subTest(cancellation=cancellation),patch.object(self.view,'execute_motion') as resume:
                self.now=100.;self.app.latest['sample_time']=self.now;self.arrive()
                if cancellation=='pause':self.app.send('pause')
                elif cancellation=='camera':self.view.stop_camera()
                else:
                    self.app.steps[1]['name']+='changed';self.view.safe_execution.poll()
                self.assertIsNone(self.view.safe_execution.pending);resume.assert_not_called()
                self.assertFalse(self.app.vision_busy)

    def test_missing_measurement_times_out_at_safe_pose_and_duplicate_result_is_ignored(self):
        self.arrive();old=self.view.safe_execution.pending['token']
        self.now=132.;self.app.latest['sample_time']=self.now
        with patch.object(self.view,'execute_motion') as resume:
            self.view.safe_execution.poll();resume.assert_not_called()
            self.assertIsNone(self.view.safe_execution.pending)
            self.view.safe_execution.completed({'token':old,'ok':True,'sample_time':self.now})
            self.assertIsNone(self.view.safe_execution.pending)
        self.assertIn('3초',self.view.status.get())

    def test_camera_off_blocks_safe_entry_before_any_motor_command(self):
        self.view.frame_live=False
        with self.assertRaisesRegex(ValueError,'카메라'):self.view.execute()
        self.sent.assert_not_called()

    def test_step_correction_display_follows_saved_target_mode_and_not_step_name(self):
        self.assertEqual(self.app.table.set('0','correction'),'미적용')
        self.assertIn('안전 자세',self.app.step_correction_detail.get())
        self.assertEqual(self.app.table.set('8','correction'),'미적용')
        self.assertTrue(all(self.app.table.set(str(i),'correction').startswith('적용 · ') for i in range(1,8)))
        self.app.rename_step(1,'안전 자세라는 이름만 붙인 집기 목표')
        self.assertTrue(self.app.table.set('1','correction').startswith('적용 · '))
        self.app.table.selection_set('1');self.app.select()
        self.assertIn('실행 시 자동 측정',self.app.step_correction_detail.get())
        self.view.vars['t_mode'].set('이 자리에 고정');self.view.save_target()
        self.assertTrue(all(self.app.table.set(str(i),'correction')=='미적용' for i in range(9)))
        self.assertIn('로봇 기준 목표',self.app.step_correction_detail.get())
        self.view.vars['t_mode'].set('지그를 따라가기');self.view.save_target()
        self.assertTrue(self.app.table.set('1','correction').startswith('적용 · '))
        self.assertEqual(self.app.table.set('0','correction'),'미적용')
        self.sent.assert_not_called()

    def test_editing_one_safe_step_updates_both_and_preserves_roundtrip(self):
        other=deepcopy(self.app.steps[1:-1]);pose={**self.safe,'shoulder_lift.pos':-40}
        self.app.update_step_pose(self.app.steps[0],pose)
        self.assertEqual(self.app.steps[0]['pose'],pose);self.assertEqual(self.app.steps[-1]['pose'],pose)
        self.assertEqual(self.app.steps[1:-1],other)
        self.assertTrue(validate_boundaries(self.app.steps))

    def test_prepared_dispatch_skips_entry_but_keeps_end_safe_pose(self):
        self.arrive()
        for i in range(31):
            self.now=101+i*.1;self.app.latest['sample_time']=self.now
            self.view.frame_live=True;self.view.frame_at=self.now
            self.view.tracker.observe(self.key,self.value,self.now,100+i)
        self.view.safe_execution.poll()
        for _ in range(400):
            self.root.update()
            if not self.app.vision_busy:break
            time.sleep(.01)
        self.assertFalse(self.app.vision_busy,self.view.status.get())
        name,payload=self.sent.call_args.args
        self.assertEqual(name,'play');self.assertEqual(len(payload['steps']),8,self.view.status.get())
        self.assertEqual(payload['steps'][0]['motion']['phase'],'approach')
        self.assertEqual(payload['steps'][-1]['safe_boundary'],'end')
        self.assertEqual(payload['steps'][-1]['pose'],self.safe)

    def test_selected_camera_step_recomputes_from_saved_xy_angle_reference(self):
        current={**self.value,'x_mm':210.,'y_mm':10.,'yaw_deg':15.}
        self.view.jig_reference.capture(self.target,current,self.view.station)
        self.app.table.selection_set('1')
        with patch.object(self.view,'observations_for',return_value={self.key:current}):
            self.app.move()
            for _ in range(400):
                self.root.update()
                if not self.app.vision_busy:break
                time.sleep(.01)
        self.assertFalse(self.app.vision_busy,self.view.status.get())
        name,payload=self.sent.call_args.args;self.assertEqual(name,'play');self.assertEqual(len(payload['steps']),1)
        expected,rotation=v.target_pose(self.target,{self.key:current});expected[2]+=self.target['hover_mm']
        actual=v.Kinematics(self.view.station['tool']).fk(payload['steps'][0]['pose'])
        np.testing.assert_allclose(actual[:3,3],expected,atol=.5)
        np.testing.assert_allclose(actual[:3,:3],rotation,atol=.02)


class ControllerTokenTests(unittest.TestCase):
    def test_demo_controller_emits_completion_token_only_after_reaching_safe_joint_pose(self):
        backend=teaching.Demo();backend.follower_only=True
        with tempfile.TemporaryDirectory() as folder:
            session=teaching.Session(backend,folder,auto_sync=True);session.start()
            try:
                end=time.monotonic()+6
                while session.state!='READY' and time.monotonic()<end:
                    session.heartbeat=time.monotonic();time.sleep(.01)
                session.command('hold_current')
                while session.state!='HOLD' and time.monotonic()<end:
                    session.heartbeat=time.monotonic();time.sleep(.01)
                self.assertEqual(session.state,'HOLD')
                current=deepcopy(backend.current);goal={**current,'shoulder_pan.pos':current['shoulder_pan.pos']+1}
                session.command('play',{'steps':[{'name':'시작 안전','pose':goal,'safe_boundary':'start'}],
                    'follower':current,'request_id':'entry-token','title':'safe test'})
                result=None;end=time.monotonic()+8
                while time.monotonic()<end and result is None:
                    session.heartbeat=time.monotonic()
                    try:
                        kind,value=session.events.get_nowait()
                        if kind=='play_result':result=value
                    except queue.Empty:time.sleep(.01)
                self.assertIsNotNone(result);self.assertTrue(result['ok']);self.assertEqual(result['token'],'entry-token')
                self.assertTrue(teaching.within_arrival_tolerance(goal,result['pose'],'MOVE'))
            finally:
                session.stop.set();session.wakeup.set();session.thread.join(3)
            self.assertFalse(session.thread.is_alive())


if __name__=='__main__':unittest.main()
