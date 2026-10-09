"""Drag-to-ROI stores image coordinates and restricts actual detector candidates."""
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tkinter as tk
import cv2,numpy as np
import teach_left_arm as t
from jig_detection import roi_pixels

class ImageROITests(unittest.TestCase):
    def setUp(self):
        self.root=tk.Tk();self.temp=tempfile.TemporaryDirectory()
        self.app=t.App(self.root,t.Demo(),self.temp.name);self.view=self.app.open_vision()
        self.root.geometry('1280x850');self.root.update()
        self.view.frame=np.zeros((720,1280,3),np.uint8);self.view.draw_frame()
        self.roi=self.view.image_roi
    def tearDown(self):
        self.app.close()
        for job in self.root.tk.call('after','info'):self.root.after_cancel(job)
        self.root.destroy();self.temp.cleanup()
    def point(self,a,b):
        x,y,w,h=self.roi.bounds;return SimpleNamespace(x=x+w*a,y=y+h*b)
    def drag(self,start,end):
        self.roi.toggle();self.roi.press(self.point(*start));self.roi.motion(self.point(*end));self.roi.release(self.point(*end))
    def test_reverse_drag_uses_letterboxed_image_and_persists_without_motion(self):
        with patch.object(self.app,'send',side_effect=AssertionError('ROI moved robot')):
            self.drag((.7,.8),(.2,.1))
        self.assertFalse(self.roi.active)
        self.assertEqual([float(self.view.vars['t_'+k].get()) for k in self.roi.fields],[20,10,50,70])
        saved=json.loads(self.view.profile_path.read_text())
        self.assertEqual(roi_pixels(next(iter(saved['jigs'].values())),(1280,720)),(256,72,896,576))
        self.assertTrue(self.view.canvas.find_withtag('detection-area'))
        self.assertIsNone(self.view.jig_reference.value)
    def test_saved_episode_keeps_latest_station_roi_on_reload_and_reset(self):
        self.view.add_steps()
        path=t.save_named_teaching(self.app.output,{**self.app.document(),'title':'ROI persistence'})
        self.app.refresh_library();self.app.load_path(path);self.app.refresh_library()
        label=self.app.library.get()
        self.app.steps[0]['path']=['nominal-preview-placeholder']
        self.drag((.2,.1),(.7,.8))
        self.assertEqual(self.app.saved_path,path);self.assertEqual(self.app.library.get(),label)
        self.assertEqual(self.app.steps[0]['path'],['nominal-preview-placeholder'])
        self.app.load_path(path)
        self.assertEqual(roi_pixels(self.view.read_target(),(1280,720)),(256,72,896,576))
        # Re-opening the camera page must also restore station settings.
        self.view.close();self.app.vision=None
        self.view=self.app.open_vision();self.roi=self.view.image_roi
        self.assertEqual(roi_pixels(self.view.read_target(),(1280,720)),(256,72,896,576))
        self.view.reset_detection_roi();self.app.load_path(path)
        self.assertEqual(roi_pixels(self.view.read_target(),(1280,720)),(0,0,1280,720))
        self.assertEqual(self.app.saved_path,path)

    def test_station_override_is_per_jig_and_foreign_preview_is_unchanged(self):
        from copy import deepcopy
        self.view.add_steps();original=deepcopy(self.app.vision_plan)
        self.drag((.2,.1),(.7,.8))
        self.app.vision_plan=deepcopy(original);self.app.preview_only=True
        self.assertFalse(self.view.apply_saved_detection_settings())
        self.assertEqual(self.app.vision_plan,original)
        self.app.preview_only=False
        self.assertTrue(self.view.apply_saved_detection_settings())
        self.app.vision_plan=deepcopy(original)
        target=next(iter(self.app.vision_plan['targets'].values()));target['jig_name']='Another jig'
        self.assertFalse(self.view.apply_saved_detection_settings())

    def test_dragged_area_filters_out_the_other_carrier(self):
        from carrier_detection import detect_carriers
        source=cv2.imread(str(Path(__file__).parent/'tests/fixtures/carrier/15-51-55.png'))[185:455,425:830]
        image=np.full((720,1280,3),175,np.uint8);image[220:490,60:465]=source;image[220:490,760:1165]=source
        self.view.frame=image;self.view.draw_frame()
        self.assertEqual(len(detect_carriers(image)),2)
        self.drag((0,.15),(.45,.85))
        result=detect_carriers(image,roi_pixels(self.view.read_target(),(1280,720)))
        self.assertEqual(len(result),1);self.assertLess(result[0].corners[:,0].mean(),576)
    def test_cancel_tiny_drag_busy_release_and_reset_do_not_leave_partial_area(self):
        self.roi.toggle();x,y,w,h=self.roi.bounds
        self.roi.press(SimpleNamespace(x=x-5,y=y-5));self.assertIsNone(self.roi.start)
        self.roi.press(self.point(.2,.2));self.roi.release(self.point(.201,.201));self.assertTrue(self.roi.active)
        self.roi.press(self.point(.1,.1));self.app.state='MOVE';self.roi.release(self.point(.8,.8))
        self.assertEqual(float(self.view.vars['t_roi_w_pct'].get()),100)
        self.app.state='DISCONNECTED';self.roi.cancel();self.drag((.1,.1),(.8,.8));self.view.reset_detection_roi()
        self.assertEqual(roi_pixels(self.view.read_target(),(1280,720)),(0,0,1280,720))
        self.assertFalse(self.view.canvas.find_withtag('detection-area'))

if __name__=='__main__':unittest.main()
