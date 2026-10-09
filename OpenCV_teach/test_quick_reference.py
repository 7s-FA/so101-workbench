import unittest,time,tempfile
from pathlib import Path
from unittest.mock import patch
import tkinter as tk
from quick_reference import QuickJigs
import teach_left_arm as t
from vision_teaching import reference_key
from jig_detection import profile_signature

class QuickTests(unittest.TestCase):
    def value(self,x=200.,quality='precise'):
        return dict(x_mm=x,y_mm=0.,yaw_deg=179.8,quality=quality,profile_signature='p')
    def test_two_measurements_finish_early_and_sparse_fallback_at_deadline(self):
        tracker=QuickJigs(max_gap=1.2)
        tracker.observe('j',self.value(),10.,1)
        with self.assertRaises(ValueError):tracker.snapshot(['j'],10.)
        self.assertEqual(tracker.snapshot(['j'],10.,final=True)['j']['sample_count'],1)
        tracker.observe('j',self.value(202.,'medium'),10.3,2)
        self.assertTrue(tracker.assessment('j',10.3)['ready'])
        tracker.clear()
        for i in range(2):tracker.observe('j',self.value(quality='medium'),10+i*.2,i)
        with self.assertRaises(ValueError):tracker.snapshot(['j'],10.2)
        self.assertEqual(tracker.snapshot(['j'],10.2,final=True)['j']['quality_counts']['medium'],2)
    def test_missing_ambiguous_stale_and_outlying_latest_frame_do_not_finalize(self):
        for case in ('missing','ambiguous','stale','outlier'):
            tracker=QuickJigs(max_gap=1.2)
            for i in range(3):tracker.observe('j',self.value(),10+i*.2,i)
            now=10.4
            if case=='missing':tracker.missing('j',10.5,4);now=10.5
            if case=='ambiguous':tracker.missing('j',hard=True)
            if case=='stale':now=12.
            if case=='outlier':tracker.observe('j',self.value(260.),10.6,4);now=10.6
            with self.subTest(case=case),self.assertRaises(ValueError):tracker.snapshot(['j'],now,final=True)

class TeachingSessionTests(unittest.TestCase):
    def setUp(self):
        self.root=tk.Tk();self.tmp=tempfile.TemporaryDirectory();self.app=t.App(self.root,t.Demo(),self.tmp.name)
        self.view=self.app.open_vision();self.view.add_steps();self.target=self.view.read_target()
        self.key=reference_key(self.target);self.now=100.
        self.clock=patch('time.monotonic',side_effect=lambda:self.now);self.clock.start()
        self.view.frame_live=True;self.view.frame_at=self.now
        self.obs=dict(x_mm=200.,y_mm=0.,yaw_deg=0.,quality='precise',profile_signature=profile_signature(self.target))
    def tearDown(self):
        self.clock.stop();self.app.close()
        for job in self.root.tk.call('after','info'):self.root.after_cancel(job)
        self.root.destroy();self.tmp.cleanup()
    def feed(self):
        for i in range(2):
            self.now+=.2;self.view.frame_at=self.now
            self.view.tracker.observe(self.key,self.obs,self.now,self.now)
            self.view.reference_acquisition.poll()
    def test_episode_load_auto_captures_without_page_change_and_capture_toggle_binds_offset(self):
        path=t.save_named_teaching(self.app.output,{**self.app.document(),'title':'auto teaching'})
        self.app.navigation.home();self.app.load_path(path)
        self.assertTrue(self.view.teaching_auto_pending)
        self.view.poll_teaching_reference();self.feed()
        self.assertIsNone(self.app.navigation.current)
        anchor=self.view.teaching_reference.value['id']
        self.app.capture_follow.set(True);context=self.app.new_capture_context()
        self.app.store_captured_step({'pose':dict.fromkeys(t.KEYS,0.),'captured_at':'now'},context)
        step=self.app.steps[-1];self.assertEqual(step['teaching_reference']['x_mm'],200.)
        self.assertEqual(step['motion']['phase'],'waypoint')
        self.view.frame_live=False;self.view.tracker.clear()
        self.assertEqual(self.app.new_capture_context()['reference_id'],anchor)
        self.app.capture_follow.set(False)
        self.app.store_captured_step({'pose':dict.fromkeys(t.KEYS,0.),'captured_at':'now'},self.app.new_capture_context())
        self.assertNotIn('motion',self.app.steps[-1])
        self.view.end_teaching_reference()
        self.assertIsNone(self.view.teaching_reference.value)
        self.assertIsNone(self.view.jig_reference.value)
    def test_execution_remeasures_and_keeps_teaching_anchor_unchanged(self):
        self.view.teaching_reference.capture(self.target,self.obs,self.view.station)
        anchor=self.view.teaching_reference.value['id'];self.app.state='HOLD'
        self.app.latest={'sample_time':self.now,'follower':dict.fromkeys(t.KEYS,0.)}
        with patch.object(self.view,'execute_motion') as move:
            self.view.jig_reference.capture(self.target,self.obs,self.view.station)
            self.view.execute(index=0);move.assert_not_called()
            self.obs={**self.obs,'x_mm':220.};self.feed()
            move.assert_called_once_with(0,prepared=True)
        self.assertEqual(self.view.jig_reference.value['observation']['x_mm'],220.)
        self.assertEqual(self.view.teaching_reference.value['observation']['x_mm'],200.)
        self.assertEqual(self.view.teaching_reference.value['id'],anchor)
    def test_idle_detection_drift_keeps_confirmed_reference_and_active_motion_still_checks_it(self):
        self.view.teaching_reference.capture(self.target,self.obs,self.view.station)
        self.view.jig_reference.capture(self.target,self.obs,self.view.station)
        ident=self.view.jig_reference.value['id']
        for i in range(2):self.view.tracker.observe(self.key,{**self.obs,'x_mm':210.},100.+i*.1,i)
        self.now=100.1;self.view.frame_at=self.now
        self.view.update_jig_reference()
        self.assertEqual(self.view.jig_reference.value['id'],ident)
        self.assertIn('자동 확정 완료',self.view.reference_status.get())
        self.assertNotIn('저장을 누르세요',self.view.reference_status.get())
        self.app.state='PLAY'
        with patch.object(self.app,'send') as send:
            self.view.update_jig_reference();send.assert_called_with('pause')
        self.assertIsNone(self.view.jig_reference.value)
        self.assertIsNotNone(self.view.teaching_reference.value)

    def test_post_measurement_setup_error_is_distinct_from_capture_failure(self):
        self.view.teaching_reference.capture(self.target,self.obs,self.view.station)
        self.view.jig_reference.capture(self.target,self.obs,self.view.station)
        self.app.state='HOLD';self.app.latest={'sample_time':self.now,'follower':dict.fromkeys(t.KEYS,0.)}
        with patch('vision_teaching.compile_steps',side_effect=ValueError('지그 실측 크기·높이 확인')),patch.object(self.app,'send') as send:
            self.view.execute_motion(index=0,prepared=True)
            # The worker only computes; consume its reported error on the Tk thread.
            item=self.view.completed.get(timeout=2);self.view.completed.put(item)
            with patch.object(self.view,'process_frame'),patch.object(self.view,'finish_detection'):
                self.view.poll()
            self.view.update_jig_reference()
            self.assertIn('실행 중단 · 지그 측정은 완료됨',self.view.reference_status.get())
            self.assertIn('지그 실측',self.view.status.get())
            self.assertNotIn('저장을 누르세요',self.view.reference_status.get())
            send.assert_not_called()

    def test_confirmed_execution_survives_missed_contour_after_calculation(self):
        self.view.jig_reference.capture(self.target,self.obs,self.view.station)
        self.app.state='HOLD';self.app.latest={'sample_time':self.now,'follower':dict.fromkeys(t.KEYS,0.)}
        completed=[]
        def job(work,done,**kw):completed.append(done)
        with patch.object(self.view,'run_job',side_effect=job),patch('teach_left_arm.validate_sequence'),patch('tkinter.messagebox.askokcancel',return_value=True),patch.object(self.app,'send') as send,patch.object(self.view,'observations_for',side_effect=AssertionError('Must not remeasure confirmed reference')):
            self.view.execute_motion(prepared=True)
            completed[0]([{'name':'compiled','pose':self.app.latest['follower']}])
            self.assertEqual(send.call_args.args[0],'play')
        self.assertIsNotNone(self.view.jig_reference.value)

    def test_three_second_fallback_and_deadline_failure(self):
        self.view.capture_jig_reference();self.now=102.9;self.view.frame_at=self.now
        self.view.tracker.observe(self.key,self.obs,self.now,1);self.view.reference_acquisition.poll()
        self.assertIsNotNone(self.view.reference_acquisition.pending)
        self.now=103.;self.view.reference_acquisition.poll()
        self.assertIsNotNone(self.view.teaching_reference.value)
        self.view.capture_jig_reference();self.now=106.;self.view.frame_at=self.now;self.view.reference_acquisition.poll()
        self.assertIsNone(self.view.reference_acquisition.pending);self.assertIn('3초',self.view.status.get())

if __name__=='__main__':unittest.main()
