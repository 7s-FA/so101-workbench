"""Image regressions and geometry/async contracts; never open physical devices."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from carrier_detection import Candidate, detect_carriers, image_pose
import jig_detection as j
from test_vision_teaching import target,optical_station,project,rectangle_world

FIXTURES=Path(__file__).parent/'tests/fixtures/carrier'
REFERENCE=json.loads((FIXTURES/'reference.json').read_text())


def overlap(a,b):
    a=np.array(a,np.float32);b=np.array(b,np.float32)
    intersection,_=cv2.intersectConvexConvex(a,b)
    return intersection/(cv2.contourArea(a)+cv2.contourArea(b)-intersection)


class CarrierImageTests(unittest.TestCase):
    def test_seven_user_views_select_carrier_not_large_white_board(self):
        for sample in REFERENCE['samples']:
            with self.subTest(image=sample['file']):
                image=cv2.imread(str(FIXTURES/sample['file']))
                candidates=detect_carriers(image,intrinsics=REFERENCE['intrinsics'])
                self.assertEqual(len(candidates),1)
                c=candidates[0]
                self.assertGreater(overlap(c.corners,sample['approximate_outer_corners_px']),.9)
                if sample['occluded']:
                    self.assertFalse(c.precision_ok)
                    self.assertEqual(c.quality,'low')

    def test_plain_boards_and_empty_images_are_not_carriers(self):
        for background in (45,180):
            image=np.full((720,1280,3),background,np.uint8)
            cv2.rectangle(image,(350,180),(950,584),(235,235,235),-1)
            self.assertEqual(detect_carriers(image),[])
        self.assertEqual(detect_carriers(np.zeros((720,1280,3),np.uint8)),[])

    def test_two_carriers_remain_ambiguous_and_roi_can_isolate_one(self):
        src=cv2.imread(str(FIXTURES/'15-51-55.png'))[185:455,425:830]
        image=np.full((720,1280,3),175,np.uint8)
        image[220:490,60:465]=src;image[220:490,760:1165]=src
        candidates=detect_carriers(image)
        self.assertEqual(len(candidates),2)
        single=detect_carriers(image,(25,150,520,550))
        self.assertEqual(len(single),1)
        self.assertLess(single[0].corners[:,0].mean(),520)

    def test_weak_outline_cannot_be_used_for_robot_coordinates(self):
        station=optical_station();corners=project(rectangle_world(),station).astype(np.float32)
        image=np.zeros((720,1280,3),np.uint8)
        weak=Candidate(corners,.95,False,reason='occluded')
        with self.assertRaises(ValueError):
            j.detect_rectangle(image,target(shape_model='carrier_6jigs'),station,candidates=[weak])

    def test_projective_centre_with_lens_distortion_is_not_corner_average(self):
        k=np.array([[950.,0,640],[0,950.,360],[0,0,1]])
        d=np.array([-.35,.15,.001,-.001,0])
        rvec=np.array([.65,.12,.25]);tvec=np.array([-60.,-80.,550.])
        points=np.array([[-110,-74,0],[110,-74,0],[110,74,0],[-110,74,0]],float)
        corners=cv2.projectPoints(points,rvec,tvec,k,d)[0].reshape(4,2)
        expected=cv2.projectPoints(np.array([[0.,0,0]]),rvec,tvec,k,d)[0].reshape(2)
        center,angle,axis=image_pose(corners,{'K':k.tolist(),'D':d.tolist()})
        self.assertLess(np.linalg.norm(center-expected),.02)
        self.assertGreater(np.linalg.norm(corners.mean(0)-expected),2.)
        for variant in (np.roll(corners,1,axis=0),corners[::-1]):
            other,other_angle,_=image_pose(variant,{'K':k.tolist(),'D':d.tolist()})
            self.assertLess(np.linalg.norm(other-center),.02)
            self.assertLess(abs((other_angle-angle+90)%180-90),.02)

    def test_legacy_carrier_and_generic_profiles_remain_distinct(self):
        carrier=target(shape_model='carrier_6jigs');carrier.pop('shape_model')
        self.assertEqual(j.validate_profile(carrier)['shape_model'],'ring')
        carrier['jig_name']='다른 고정 지그'
        self.assertEqual(j.validate_profile(carrier)['shape_model'],'rectangle')
        with self.assertRaises(ValueError):j.validate_profile({**carrier,'shape_model':'invalid'})


class DetectionWorkerTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        import teach_left_arm as teaching
        self.temp=tempfile.TemporaryDirectory();self.root=tk.Tk()
        self.app=teaching.App(self.root,teaching.Demo(),self.temp.name)
        self.view=self.app.open_vision();self.errors=[]
        self.root.report_callback_exception=lambda *a:self.errors.append(a)
        self.view.frame=np.zeros((720,1280,3),np.uint8)
        self.view.frame_id=1;self.view.frame_at=time.monotonic();self.view.frame_live=True

    def tearDown(self):
        self.view.close()
        for job in self.root.tk.call('after','info'):self.root.after_cancel(job)
        self.root.destroy();self.temp.cleanup()
        self.assertEqual(self.errors,[])

    def test_image_worker_does_not_block_tk_and_discards_old_profile(self):
        view=self.view;calls=[]
        def slow(*_):time.sleep(.12);return []
        with patch('jig_detection.inspection_candidates',side_effect=slow):
            start=time.monotonic();view.process_frame(background=True)
            self.assertLess(time.monotonic()-start,.08)
            self.root.after(10,lambda:calls.append('heartbeat'))
            view.vars['t_roi_x_pct'].set('5');view.vars['t_roi_w_pct'].set('95')
            end=time.monotonic()+.25
            while time.monotonic()<end:self.root.update();time.sleep(.005)
            self.assertEqual(calls,['heartbeat'])
            self.assertFalse(view.observations)
            self.assertFalse(view.tracker.history)

    def test_stop_clears_pending_detection_and_never_moves_robot(self):
        view=self.view
        def slow(*_):time.sleep(.06);return []
        with patch('jig_detection.inspection_candidates',side_effect=slow),patch.object(self.app.session,'command') as command:
            view.process_frame(background=True);view.stop_camera();time.sleep(.1);view.finish_detection()
            self.assertFalse(view.frame_live)
            self.assertEqual(view.check_center.get(),'—')
            command.assert_not_called()

    def test_calibration_change_invalidates_photo_candidate_cache(self):
        view=self.view;view.frame_live=False;profile=view.read_target()
        with patch('jig_detection.inspection_candidates',return_value=[]) as detector:
            view.candidates_for(profile);view.candidates_for(profile)
            self.assertEqual(detector.call_count,1)
            view.station['intrinsics']={'size':[1280,720],'K':[[950,0,640],[0,950,360],[0,0,1]],'D':[0,0,0,0,0]}
            view.candidates_for(profile)
            self.assertEqual(detector.call_count,2)


if __name__=='__main__':unittest.main()
