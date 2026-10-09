"""Rectangle ambiguity, clipping, profile migration, and measurement freshness."""
from copy import deepcopy
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import jig_detection as j
import vision_teaching as v
from test_vision_teaching import optical_station, project, rectangle_world, target


class RectangleTests(unittest.TestCase):
    def test_cyclic_corner_order_and_heading_prior_are_consistent(self):
        station=optical_station();p=project(rectangle_world(yaw=-175),station)
        t=target(expected_yaw_deg=179)
        for corners in (p,np.roll(p,1,axis=0),p[::-1]):
            result=j.rectangle_pose(corners,t,station,[1280,720])
            self.assertAlmostEqual(v.angle_delta(result['yaw_deg'],-175),0,places=4)
        # A 180 degree reversal is physically indistinguishable: report the configured branch.
        opposite=j.rectangle_pose(p,target(expected_yaw_deg=0),station,[1280,720])
        self.assertAlmostEqual(opposite['yaw_deg'],5,places=4)
        with self.assertRaisesRegex(ValueError,'방향'):
            j.rectangle_pose(p,target(expected_yaw_deg=90),station,[1280,720])

    def test_invalid_dimensions_roi_and_non_rectangular_shape_are_rejected(self):
        for changes in ({'long_mm':140,'short_mm':148},{'roi_x_pct':80,'roi_w_pct':30},
                        {'yaw_tolerance_deg':90},{'geometry_verified':'yes'}):
            with self.assertRaises(ValueError):target(**changes)
        s=optical_station();points=rectangle_world();points[0,0]+=35
        with self.assertRaises(ValueError):j.rectangle_pose(project(points,s),target(),s,[1280,720])

    def test_loaded_frame_holes_and_rotations_keep_outer_reference(self):
        s=optical_station()
        for yaw in (-20,0,20):
            image=np.full((720,1280,3),45,np.uint8)
            cv2.fillConvexPoly(image,project(rectangle_world(yaw=yaw),s).astype(np.int32),(225,225,225))
            cv2.fillConvexPoly(image,project(rectangle_world(yaw=yaw,long=170,short=100),s).astype(np.int32),(45,45,45))
            cv2.circle(image,(620,360),30,(60,80,160),-1)
            _,result=j.detect_rectangle(image,target(),s)
            self.assertAlmostEqual(result['x_mm'],200,delta=1.2)
            self.assertAlmostEqual(result['yaw_deg'],yaw,delta=1)

    def test_cropped_and_missing_outer_boundary_not_used(self):
        s=optical_station();image=np.full((720,1280,3),40,np.uint8)
        points=project(rectangle_world(),s).astype(np.int32)
        cv2.fillConvexPoly(image,points,(220,220,220))
        with self.assertRaises(ValueError):j.detect_rectangle(image,target(roi_x_pct=50,roi_w_pct=50),s)
        cv2.rectangle(image,(480,255),(550,465),(40,40,40),-1)
        with self.assertRaises(ValueError):j.detect_rectangle(image,target(),s)
        with self.assertRaises(ValueError):j.detect_rectangle(np.zeros_like(image),target(),s)

    def test_legacy_targets_load_but_never_execute(self):
        import teach_left_arm as teaching
        t={**target(), 'mode':'marker', 'dictionary':'DICT_5X5_100','marker_id':10,
           'marker_size_mm':40,'marker_z_mm':30,'x_mm':17,'y_mm':-9}
        s=v.default_station(teaching.Demo.identity)
        plan=v.validate_plan({'targets':{'old':t},'calibration_signature':v.calibration_signature(s)})
        self.assertEqual(plan['targets']['old']['x_mm'],17)
        seed=dict.fromkeys(teaching.KEYS,0.)
        with self.assertRaisesRegex(ValueError,'기존 마커'):
            v.compile_steps(v.make_steps('old',t,seed),plan,s,{},seed)

    def test_live_execution_uses_registered_jig_dimensions(self):
        import teach_left_arm as teaching
        s=v.default_station(teaching.Demo.identity);s['tool']['verified']=True;t=target()
        plan={'targets':{'s':t},'calibration_signature':v.calibration_signature(s)}
        v.validate_execution_setup(plan,s,live=True)
        with self.assertRaisesRegex(ValueError,'관측|측정|인식|기준'):
            v.compile_steps(v.make_steps('s',t,dict.fromkeys(teaching.KEYS,0.)),plan,s,{},dict.fromkeys(teaching.KEYS,0.),live=True)

    def test_changed_profile_cannot_reuse_stable_observations(self):
        tracker=v.StableJigs();value=dict(x_mm=200,y_mm=0,yaw_deg=0,profile_signature='before')
        for i in range(31):tracker.observe('j',value,i*.1,i)
        self.assertEqual(tracker.snapshot(['j'],now=3.01)['j']['profile_signature'],'before')
        tracker.observe('j',{**value,'profile_signature':'after'},3.1,31)
        with self.assertRaises(ValueError):tracker.snapshot(['j'],now=3.11)


class IntegrationTests(unittest.TestCase):
    def test_detection_tab_reports_pixels_without_robot_calibration_and_clears_stale(self):
        import tkinter as tk
        import teach_left_arm as teaching
        with tempfile.TemporaryDirectory() as folder:
            root=tk.Tk();app=teaching.App(root,teaching.Demo(),folder)
            try:
                view=app.open_vision();view.vars['t_shape_model'].set('일반 사각형');view.select_tab(3);root.update()
                frame=np.full((720,1280,3),35,np.uint8)
                cv2.rectangle(frame,(400,220),(840,516),(230,230,230),-1)
                view.frame=frame;view.frame_live=False
                view.process_frame()
                self.assertIn('620',view.check_center.get())
                self.assertIn('°',view.check_angle.get())
                self.assertFalse(view.observations)
                self.assertFalse(view.tracker.history)
                self.assertIn('보정',view.check_world.get())
                view.frame_live=True;view.frame_at=time.monotonic()-2
                view.detection_overlay(frame.copy())
                self.assertEqual(view.check_center.get(),'—')
                self.assertIn('갱신 중단',view.check_status.get())
            finally:
                if app.vision:app.vision.close()
                for job in root.tk.call('after','info'):root.after_cancel(job)
                root.destroy()

    def test_shared_profile_updates_slots_and_photo_cannot_authorize_execution(self):
        import tkinter as tk
        import teach_left_arm as teaching
        with tempfile.TemporaryDirectory() as folder:
            root=tk.Tk();app=teaching.App(root,teaching.Demo(),folder)
            try:
                view=app.open_vision();view.save_target();first=view.target_id
                view.new_target();view.vars['t_name'].set('2번 칸');view.vars['t_x_mm'].set('20');view.save_target()
                self.assertFalse(app.vision_plan['targets'][first]['geometry_verified'])
                view.vars['t_long_mm'].set('225');view.save_target()
                self.assertEqual(app.vision_plan['targets'][first]['long_mm'],225)
                self.assertFalse(app.vision_plan['targets'][first]['geometry_verified'])
                self.assertEqual(app.vision_plan['targets'][first]['x_mm'],0)
                self.assertEqual(app.vision_plan['targets'][view.target_id]['x_mm'],20)
                view.frame_live=False
                with self.assertRaisesRegex(ValueError,'현재 상단'):view.observations_for(app.vision_plan)
            finally:
                if app.vision:app.vision.close()
                for job in root.tk.call('after','info'):root.after_cancel(job)
                root.destroy()

    def test_real_detector_keeps_short_loss_but_blocks_it_and_clears_ambiguity(self):
        import tkinter as tk
        import teach_left_arm as teaching
        with tempfile.TemporaryDirectory() as folder:
            root=tk.Tk();app=teaching.App(root,teaching.Demo(),folder)
            try:
                view=app.open_vision();view.vars['t_shape_model'].set('일반 사각형');view.save_target();t=app.vision_plan['targets'][view.target_id]
                t['surface_z_mm']=30;view.station=optical_station();view.frame_live=True
                frame=np.full((720,1280,3),40,np.uint8)
                cv2.fillConvexPoly(frame,project(rectangle_world(),view.station).astype(np.int32),(220,220,220))
                base=time.monotonic()-3.
                for i in range(31):
                    view.frame=frame;view.frame_at=base+i*.1;view.frame_id=i;view.process_frame()
                snapshot=view.observations_for(app.vision_plan)
                self.assertEqual(set(snapshot),{v.reference_key(t)})
                view.frame=np.zeros_like(frame);view.frame_id+=1;view.process_frame()
                self.assertTrue(view.tracker.history)
                with self.assertRaises(ValueError):view.observations_for(app.vision_plan)
                self.assertFalse(view.observations)
                app.vision_plan['targets']['other']={**t,'jig_name':'같은 모양의 제품 B','product':'제품 B'}
                view.frame=frame;view.frame_id+=1;view.process_frame()
                self.assertFalse(view.observations)
                self.assertIn('겹칩니다',str([view.detected.item(i)['values'] for i in view.detected.get_children()]))
                self.assertFalse(view.tracker.history)
                with patch.object(view,'process_frame',side_effect=RuntimeError('detector failed')):
                    view.tracker.history['old']=[1];view.reset_detection_if_error()
                    self.assertFalse(view.tracker.history)
            finally:
                if app.vision:app.vision.close()
                for job in root.tk.call('after','info'):root.after_cancel(job)
                root.destroy()

    def test_legacy_file_is_read_only_and_new_opencv_target_is_separate(self):
        import tkinter as tk
        import teach_left_arm as teaching
        with tempfile.TemporaryDirectory() as folder:
            root=tk.Tk();app=teaching.App(root,teaching.Demo(),folder)
            try:
                view=app.open_vision()
                legacy={**target(),'mode':'marker','dictionary':'DICT_5X5_100','marker_id':10,
                        'marker_size_mm':40,'marker_z_mm':30,'x_mm':17,'y_mm':-9}
                app.vision_plan['targets']['old']=legacy
                view.target_id='old';view.refresh_target_list();view.load_target()
                with self.assertRaises(ValueError):view.save_target()
                self.assertEqual(app.vision_plan['targets']['old']['mode'],'marker')
                self.assertEqual(app.vision_plan['targets']['old']['x_mm'],17)
                view.new_target();view.vars['t_x_mm'].set('0');view.vars['t_y_mm'].set('0');view.save_target()
                self.assertNotEqual(view.target_id,'old')
                self.assertEqual(app.vision_plan['targets'][view.target_id]['mode'],'rectangle')
                self.assertEqual(app.vision_plan['targets']['old']['mode'],'marker')
            finally:
                if app.vision:app.vision.close()
                for job in root.tk.call('after','info'):root.after_cancel(job)
                root.destroy()


if __name__=='__main__':unittest.main()
