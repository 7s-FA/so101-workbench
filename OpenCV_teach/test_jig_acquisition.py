import unittest,time,tempfile,json
from pathlib import Path
from unittest.mock import patch
import tkinter as tk
import numpy as np
import teach_left_arm as t
import vision_teaching as v
from jig_detection import profile_signature

class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.root=tk.Tk();self.folder=tempfile.TemporaryDirectory()
        self.app=t.App(self.root,t.Demo(),self.folder.name);self.view=self.app.open_vision();self.view.add_steps()
        self.target=self.view.read_target();self.key=v.reference_key(self.target)
        self.app.state='HOLD';self.view.frame_live=True;self.now=100.;self.view.frame_at=self.now
        self.app.latest={'sample_time':self.now,'follower':dict.fromkeys(t.KEYS,0.)}
        self.value={'x_mm':200.,'y_mm':0.,'yaw_deg':0.,'quality':'precise','profile_signature':profile_signature(self.target)}
        self.clock=patch('time.monotonic',side_effect=lambda:self.now);self.clock.start()
    def tearDown(self):
        self.clock.stop();self.app.close()
        for job in self.root.tk.call('after','info'):self.root.after_cancel(job)
        self.root.destroy();self.folder.cleanup()
    def feed(self):
        # A slower detector still has >=3 precise samples over an actual 3 s span.
        for i in range(5):
            self.now=100+i*.8;self.view.frame_at=self.now;self.app.latest['sample_time']=self.now
            self.view.tracker.observe(self.key,self.value,self.now,i)
            self.view.reference_acquisition.poll()
    def test_single_step_waits_and_each_run_requires_fresh_measurement(self):
        with patch.object(self.view,'execute_motion') as execute:
            self.view.execute(index=0);execute.assert_not_called();self.assertTrue(self.app.vision_busy)
            self.feed();execute.assert_called_once_with(0,prepared=True)
            self.assertIsNotNone(self.view.jig_reference.value)
            self.assertGreaterEqual(self.view.jig_reference.value['observation']['quality_counts']['precise'],2)
            self.view.tracker.clear();self.view.execute(index=0);self.assertEqual(execute.call_count,1)
            self.assertIsNotNone(self.view.reference_acquisition.pending)
    def test_stop_and_revision_changes_never_dispatch_waiting_step(self):
        for cause in ('stop','revision'):
            with patch.object(self.view,'execute_motion') as execute:
                self.view.execute(index=0)
                if cause=='stop':self.view.cancel_execution('stop')
                else:self.app.steps[0]['name']+=' changed';self.view.reference_acquisition.poll()
                self.feed();execute.assert_not_called();self.assertIsNone(self.view.reference_acquisition.pending)
            self.view.tracker.clear();self.view.clear_jig_reference();self.now=100.;self.view.frame_at=100.
    def test_timeout_remains_visible_and_late_readiness_needs_a_new_request(self):
        self.view.capture_jig_reference()
        self.now=131.;self.view.reference_acquisition.poll()
        self.assertIsNone(self.view.reference_acquisition.pending)
        for i in range(31):
            self.now=131+i*.1;self.view.frame_at=self.now
            self.view.tracker.observe(self.key,self.value,self.now,i)
        self.view.reference_acquisition.poll();self.view.update_jig_reference()
        self.assertIsNone(self.view.jig_reference.value)
        self.assertIn('자동 측정 종료',self.view.reference_status.get())
        self.assertIn('다시 실행',self.view.reference_status.get())
        self.view.capture_jig_reference()
        for i in range(2):
            self.now+=.2;self.view.frame_at=self.now
            self.view.tracker.observe(self.key,self.value,self.now,100+i)
            self.view.reference_acquisition.poll()
        self.view.update_jig_reference()
        self.assertIsNotNone(self.view.jig_reference.value)
        self.assertIn('저장된 기준',self.view.reference_status.get())
        self.assertEqual(self.view.measurement_failure,'')

    def test_manual_capture_does_not_require_robot_motion_or_tcp_verified(self):
        self.app.state='DISCONNECTED';self.view.station['tool']['verified']=False
        with patch.object(self.app,'send',side_effect=AssertionError('Capture moved robot')):
            self.view.capture_jig_reference();self.feed()
        self.assertIsNotNone(self.view.jig_reference.value)
    def test_processing_age_is_separate_from_live_camera_expiry(self):
        self.view.frame=np.zeros((720,1280,3),np.uint8);self.view.frame_id=10
        signature,_=self.view.detection_request()
        with patch.object(self.view,'process_frame_results') as consume:
            for age,expected in ((.9,1),(1.3,1)):
                token=(self.view.detection_generation,1,self.now-age,signature)
                self.view.detection_completed.put((token,self.view.frame.copy(),{},None,age*1000))
                self.view.finish_detection();self.assertEqual(consume.call_count,expected)

class CameraFixtureTests(unittest.TestCase):
    def test_current_roi_keeps_precise_outer_rim_and_corrected_heading(self):
        import cv2
        from jig_detection import inspection_candidates,detect_rectangle
        root=Path(__file__).parent/'verification/20260922_reference_recovery'
        fixture=json.loads((root/'measurement-fixture.json').read_text())
        frame=cv2.imread(str(root/'live-raw.png'))
        profile,station=fixture['profile'],fixture['station']
        candidates=inspection_candidates(frame,profile,station)
        self.assertEqual(len(candidates),1);self.assertEqual(candidates[0].quality,'precise')
        _,pose=detect_rectangle(frame,profile,station,candidates=candidates)
        self.assertLess(pose['size_error_pct'],2)
        self.assertAlmostEqual(pose['yaw_deg'],-172.785,delta=.1)
        with self.assertRaisesRegex(ValueError,'예상 방향'):
            detect_rectangle(frame,{**profile,'expected_yaw_deg':-90},station,candidates=candidates)

if __name__=='__main__':unittest.main()
