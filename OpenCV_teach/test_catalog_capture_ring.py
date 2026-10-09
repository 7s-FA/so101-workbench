import unittest,time,tempfile
from copy import deepcopy
from unittest.mock import patch
import numpy as np
import cv2
import jig_detection as j
import jig_stl as stl
from test_jig_catalog import jig
from test_jig_stl import SOURCE,FakeRenderer
from test_vision_teaching import optical_station,project,rectangle_world,target

class RingTests(unittest.TestCase):
    def test_stl_holes_and_rectangular_ring_without_grid(self):
        holes=stl.projected_holes(stl.triangles(SOURCE))
        self.assertEqual(len(holes),8)
        image=np.full((600,800,3),40,np.uint8)
        cv2.rectangle(image,(150,150),(650,480),(235,235,235),-1)
        plain=j.inspection_candidates(image,jig(shape_model='ring'))
        self.assertEqual(plain,[])
        cv2.rectangle(image,(185,185),(615,445),(40,40,40),-1)
        found=j.inspection_candidates(image,jig(shape_model='ring'))
        self.assertTrue(found)
        self.assertTrue(any(np.linalg.norm(c.corners.mean(0)-[400,315])<5 for c in found))
        migrated=j.validate_profile(jig(shape_model='carrier_6jigs'))
        self.assertEqual(migrated['shape_model'],'ring');self.assertEqual(len(migrated['hole_layout']),8)
    def test_photo_candidates_do_not_prefer_support_board(self):
        from test_carrier_detection import REFERENCE,FIXTURES,overlap
        results=[]
        for sample in REFERENCE['samples']:
            frame=cv2.imread(str(FIXTURES/sample['file']))
            found=j.inspection_candidates(frame,jig(shape_model='ring',hole_layout=stl.legacy_holes()))
            results.append((sample['file'],[(c.quality,round(overlap(c.corners,sample['approximate_outer_corners_px']),2)) for c in found]))
            self.assertTrue(all(overlap(c.corners,sample['approximate_outer_corners_px'])>.85 for c in found),results)
        print('Ring photo evidence:',results)

    def test_actual_square_pallet_rim_opening_and_heading(self):
        model=stl.import_stl(stl.ASSETS/'625e7099050134ab1961174838b22cf248839fd13402d7132dd6160edf09c1ca.stl')
        self.assertEqual(model['dimensions_mm'],[70.,70.,25.])
        self.assertEqual(model['rim_height_mm'],20.)
        self.assertEqual(len(model['hole_layout']),1)
        p=jig(shape_model='ring',long_mm=70,short_mm=70,hole_layout=model['hole_layout'],surface_z_mm=30,expected_yaw_deg=80)
        j.validate_profile(p)
        station=optical_station();c=project(rectangle_world(long=70,short=70,yaw=7),station)
        a=j.rectangle_pose(c,p,station,[1280,720])
        for corners in (np.roll(c,1,axis=0),c[::-1]):
            b=j.rectangle_pose(corners,p,station,[1280,720])
            self.assertAlmostEqual(a['yaw_deg'],b['yaw_deg'],places=5)
        self.assertAlmostEqual(a['yaw_deg'],97.,places=5)
        from ring_detection import evidence
        image=np.full((600,600,3),40,np.uint8)
        cv2.rectangle(image,(80,80),(520,520),(235,235,235),-1)
        self.assertLess(evidence(image,np.float32([[80,80],[520,80],[520,520],[80,520]]),p)[0],.1)
        cv2.rectangle(image,(212,212),(388,388),(40,40,40),-1)
        self.assertGreater(evidence(image,np.float32([[80,80],[520,80],[520,520],[80,520]]),p)[0],.8)

    def test_heading_capture_can_measure_outside_prior_without_loosening_execution(self):
        station=optical_station();p=target(expected_yaw_deg=80,yaw_tolerance_deg=5)
        corners=project(rectangle_world(),station)
        with self.assertRaisesRegex(ValueError,'예상 방향'):j.rectangle_pose(corners,p,station,[1280,720])
        measured=j.rectangle_pose(corners,p,station,[1280,720],enforce_heading=False)
        self.assertTrue(np.isfinite(measured['yaw_deg']))

class CaptureTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        import teach_left_arm as t
        self.tmp=tempfile.TemporaryDirectory();self.render=patch('robot_preview.RenderClient',FakeRenderer);self.render.start()
        self.root=tk.Tk();self.app=t.App(self.root,t.Demo(),self.tmp.name)
        self.view=self.app.open_vision(show=False);self.panel=self.view.catalog;self.view.select_tab(1)
        self.view.station.update({k:v for k,v in optical_station().items() if k in ('camera','intrinsics','extrinsics')})
        self.now=100.;self.clock=patch('time.monotonic',side_effect=lambda:self.now);self.clock.start()
        self.view.frame_live=True;self.view.frame_at=self.now;self.view.teaching_auto_pending=False
    def tearDown(self):
        self.clock.stop();self.app.close()
        for job in self.root.tk.call('after','info'):self.root.after_cancel(job)
        self.root.destroy();self.render.stop();self.tmp.cleanup()
    def test_selected_draft_measured_returns_applies_only_angle(self):
        from vision_teaching import reference_key
        before=deepcopy(self.app.vision_plan)
        with patch.object(self.app,'send',side_effect=AssertionError('motor command')):
            self.panel.capture_heading();self.root.update()
            self.assertIs(self.app.navigation.current,self.view.window)
            pending=self.view.reference_acquisition.pending;self.assertTrue(pending['heading_only'])
            t=pending['target'];key=reference_key(t)
            _,profiles=self.view.detection_request();self.assertEqual(len(profiles),1)
            self.assertEqual(next(iter(profiles.values()))['jig_name'],self.panel.vars['jig_name'].get())
            for i in range(2):
                self.now+=.2;self.view.frame_at=self.now
                self.view.tracker.observe(key,dict(x_mm=200.,y_mm=0.,yaw_deg=25.,quality='precise',profile_signature=j.profile_signature(t)),self.now,i)
                self.view.reference_acquisition.poll()
            self.assertIsNone(self.view.reference_acquisition.pending)
            self.assertIs(self.app.navigation.current,self.view.settings_window)
            self.assertAlmostEqual(float(self.panel.vars['expected_yaw_deg'].get()),25.)
            self.assertEqual(self.app.vision_plan,before)
            self.assertIsNone(self.view.jig_reference.value)
    def test_timeout_and_cancel_keep_prior_and_return(self):
        for cancel in (False,True):
            self.view.frame_at=self.now;self.panel.vars['expected_yaw_deg'].set('17')
            self.panel.capture_heading()
            if cancel:self.view.reference_acquisition.cancel('사용자 취소')
            else:self.now+=3.1;self.view.reference_acquisition.poll()
            self.assertIsNone(self.view.reference_acquisition.pending)
            self.assertIs(self.app.navigation.current,self.view.settings_window)
            self.assertEqual(self.panel.vars['expected_yaw_deg'].get(),'17')
            self.assertIn('유지',self.panel.status.get())
