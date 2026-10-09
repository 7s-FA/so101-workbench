"""Coverage for retired UI paths and shared detection work; no real device access."""
import tempfile,unittest,tkinter as tk
from unittest.mock import patch
import numpy as np
import teach_left_arm as t
from carrier_detection import Candidate
from test_vision_teaching import target,optical_station,project,rectangle_world
from jig_detection import profile_signature

class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.root=tk.Tk();self.tmp=tempfile.TemporaryDirectory()
        self.app=t.App(self.root,t.Demo(),self.tmp.name);self.view=self.app.open_vision()
    def tearDown(self):
        self.app.close()
        for job in self.root.tk.call('after','info'):self.root.after_cancel(job)
        self.root.destroy();self.tmp.cleanup()
    def widgets(self,w):
        for child in w.winfo_children():yield child;yield from self.widgets(child)
    def test_retired_controls_are_absent_and_main_safe_action_still_works(self):
        labels=[str(w.cget('text')) for w in self.widgets(self.root) if 'text' in w.keys()]
        for old in ('고급 · 모델 미세 보정','모델 보정 열기','이 지그의 저장된 작업 찾기','현재 에피소드 앞뒤에 안전 자세 추가','시작·종료 안전 자세 추가'):
            self.assertNotIn(old,labels)
        self.assertIn('SO-101 모터 캘리브레이션 · 영점·집게',labels)
        with patch.object(self.app,'send',side_effect=AssertionError('configuration moved robot')):
            self.view.add_steps();self.app.add_safe_steps()
        self.assertEqual(self.app.steps[0]['safe_boundary'],'start')
        self.assertEqual(self.app.steps[-1]['safe_boundary'],'end')
        for name in ('open_joint_zero','image_tracker','find_episodes','focus_detection_roi','geometry_verified','verified'):
            self.assertFalse(hasattr(self.view,name),name)
    def test_metric_projection_reused_and_invalidated_by_frame_profile_and_calibration(self):
        import jig_detection
        self.view.station=optical_station();self.view.frame=np.zeros((720,1280,3),np.uint8)
        profile=target();candidate=Candidate(project(rectangle_world(),self.view.station).astype(np.float32),.9,True)
        with patch('jig_detection.detect_rectangle',wraps=jig_detection.detect_rectangle) as detect:
            self.view.metric_pose_for(profile,candidates=[candidate]);self.view.metric_pose_for(profile,candidates=[candidate])
            self.assertEqual(detect.call_count,1)
            self.view.frame_id+=1;self.view.metric_pose_for(profile,candidates=[candidate])
            self.assertEqual(detect.call_count,2)
            self.view.metric_pose_for({**profile,'roi_x_pct':.1,'roi_w_pct':99.9},candidates=[candidate])
            self.assertEqual(detect.call_count,3)
            self.view.station['tool']['xyz_mm'][0]+=.1
            self.view.metric_pose_for(profile,candidates=[candidate]);self.assertEqual(detect.call_count,4)
            self.view.reset_detection();self.view.metric_pose_for(profile,candidates=[candidate]);self.assertEqual(detect.call_count,5)
    def test_hidden_camera_does_not_allocate_images_and_reopens_with_latest_frame(self):
        self.view.frame=np.zeros((720,1280,3),np.uint8)
        self.app.navigation.home()
        with patch('PIL.ImageTk.PhotoImage',side_effect=AssertionError('hidden image allocation')):
            self.view.draw_frame()
        self.view.select_tab(3)
        self.assertIsNotNone(self.view.photo)
    def test_idle_saved_reference_does_not_recompute_consensus(self):
        profile=self.view.read_target();obs={'x_mm':200.,'y_mm':0.,'yaw_deg':0.,'profile_signature':profile_signature(profile)}
        self.view.jig_reference.capture(profile,obs,self.view.station)
        with patch.object(self.view.tracker,'snapshot',side_effect=AssertionError('idle recomputation')):
            self.view.update_jig_reference()
        self.assertIsNotNone(self.view.jig_reference.value)
    def test_legacy_verification_metadata_roundtrips_without_live_controls(self):
        self.view.add_steps();key=self.view.target_id
        self.app.vision_plan['targets'][key]['geometry_verified']=True
        self.assertTrue(self.view.read_target()['geometry_verified'])
        self.view.vars['t_long_mm'].set('222');self.view.save_target()
        self.assertFalse(self.app.vision_plan['targets'][key]['geometry_verified'])
        self.view.station['tool']['verified']=True;self.view.save_tool()
        self.assertTrue(self.view.station['tool']['verified'])
        self.view.vars['tool_xyz_mm0'].set('1');self.view.save_tool()
        self.assertFalse(self.view.station['tool']['verified'])

if __name__=='__main__':unittest.main()
