"""Docking latch and teaching with an occluded jig; no devices are accessed."""
from copy import deepcopy
import tempfile
import time
import unittest
from unittest.mock import patch

import teach_left_arm as teaching
import vision_teaching as v
from jig_detection import profile_signature
from jig_reference import JigReference
from test_vision_teaching import target


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.target=target();self.station=v.default_station(teaching.Demo.identity)
        self.observation=dict(x_mm=200.,y_mm=20.,yaw_deg=37.,at=time.monotonic(),profile_signature=profile_signature(self.target))
        self.ref=JigReference();self.ref.capture(self.target,self.observation,self.station)

    def test_capture_is_immutable_shared_by_slots_and_never_restored(self):
        original=self.ref.check(self.target,self.station)
        self.observation['x_mm']=900
        copy=self.ref.check({**self.target,'name':'다른 칸','x_mm':30},self.station)
        self.assertEqual(copy['id'],original['id']);self.assertEqual(copy['observation']['x_mm'],200)
        copy['observation']['x_mm']=600
        self.assertEqual(self.ref.check(self.target,self.station)['observation']['x_mm'],200)
        new=JigReference()
        with self.assertRaisesRegex(ValueError,'먼저'):new.check(self.target,self.station)

    def test_calibration_and_physical_profile_change_clear_reference(self):
        altered=deepcopy(self.station);altered['tool']['xyz_mm'][0]=1
        with self.assertRaisesRegex(ValueError,'보정'):self.ref.check(self.target,altered)
        self.assertIsNone(self.ref.value)
        self.ref.capture(self.target,self.observation,self.station)
        with self.assertRaisesRegex(ValueError,'설정'):self.ref.check({**self.target,'surface_z_mm':31},self.station)
        self.assertIsNone(self.ref.value)

    def test_movement_clears_reference_instead_of_following_it(self):
        self.assertFalse(self.ref.observe_current({**self.observation,'x_mm':200.4,'yaw_deg':37.4}))
        self.assertEqual(self.ref.value['observation']['x_mm'],200)
        self.assertTrue(self.ref.observe_current({**self.observation,'yaw_deg':42}))
        self.assertIsNone(self.ref.value)


class InterfaceTests(unittest.TestCase):
    def test_capture_then_occlude_teach_preview_execute_and_release(self):
        import tkinter as tk
        root=tk.Tk();errors=[];root.report_callback_exception=lambda _t,e,_b:errors.append(e)
        with tempfile.TemporaryDirectory() as folder:
            app=teaching.App(root,teaching.Demo(),folder)
            try:
                view=app.open_vision();view.vars['t_return_to_start'].set('집고 제자리에 놓기');view.add_steps()
                t=view.read_target();key=v.reference_key(t)
                obs=dict(x_mm=200.,y_mm=0.,yaw_deg=0.,at=time.monotonic(),profile_signature=profile_signature(t))
                view.frame_live=True;view.frame_at=time.monotonic()
                with patch.object(view,'observations_for',return_value={key:obs}):view.capture_jig_reference()
                captured=view.teaching_reference.value['id']
                q=v.Kinematics(view.station['tool']).ik(*v.target_pose(t,{key:obs}),dict.fromkeys(teaching.KEYS,0.),80)
                app.state='HOLD';app.latest=dict(sample_time=time.monotonic(),follower=q)
                # The image still streams but all jig detections disappear.
                view.tracker.clear();view.observations={};view.frame_at=time.monotonic()
                with patch.object(view,'observations_for',side_effect=AssertionError('occluded jig must not be re-read')),patch.object(app,'send') as send:
                    view.manual.measure('target')
                    token=view.manual.pending['token']
                    view.manual.completed(dict(token=token,ok=True,kind='measure',pose=q,captured_at=teaching.stamp(),sample_time=time.monotonic()))
                    saved=app.vision_plan['targets'][view.target_id]
                    self.assertEqual(saved['teaching_measurement']['jig_reference']['id'],captured)
                    self.assertAlmostEqual(saved['x_mm'],t['x_mm'],delta=.5)
                    self.assertEqual(view.teaching_reference.value['id'],captured)
                # Teaching saves tolerate occlusion; execution still needs a fresh view.
                view.frame_live=False
                app.latest['sample_time']=time.monotonic()
                with patch.object(app,'send') as send:
                    with self.assertRaisesRegex(ValueError,'카메라'):view.execute(index=2)
                    send.assert_not_called()
                self.assertEqual(view.teaching_reference.value['id'],captured)
                view.clear_jig_reference()
                with self.assertRaisesRegex(ValueError,'先|먼저'):view.reference_observations(app.vision_plan)
                self.assertFalse(errors,errors)
            finally:
                if app.vision:app.vision.close()
                if app.preview:app.preview.close()
                for job in root.tk.call('after','info'):root.after_cancel(job)
                root.destroy()

    def test_failed_read_camera_stop_and_replacement_invalidate_pending_save(self):
        import tkinter as tk
        root=tk.Tk()
        with tempfile.TemporaryDirectory() as folder:
            app=teaching.App(root,teaching.Demo(),folder)
            try:
                view=app.open_vision();view.add_steps();t=view.read_target();key=v.reference_key(t)
                obs=dict(x_mm=200.,y_mm=0.,yaw_deg=0.,profile_signature=profile_signature(t))
                view.frame_live=True;view.frame_at=time.monotonic()
                with patch.object(view,'observations_for',return_value={key:obs}):view.capture_jig_reference()
                with patch.object(view,'observations_for',side_effect=ValueError('가림')):
                    view.capture_jig_reference()
                    self.assertIsNotNone(view.reference_acquisition.pending)
                    self.assertIn('가림',view.status.get())
                self.assertIsNone(view.jig_reference.value)
                view.jig_reference.capture(t,obs,view.station);view.stop_camera();self.assertIsNone(view.jig_reference.value)
                view.teaching_reference.capture(t,obs,view.station)
                q=dict.fromkeys(teaching.KEYS,0.);app.state='HOLD';app.latest=dict(sample_time=time.monotonic(),follower=q)
                with patch.object(app,'send'):
                    view.manual.measure('target');token=view.manual.pending['token']
                view.teaching_reference.capture(t,obs,view.station)
                with patch.object(view,'save_target') as save:
                    view.manual.completed(dict(token=token,ok=True,kind='measure',pose=q,captured_at=teaching.stamp(),sample_time=time.monotonic()))
                    save.assert_not_called();self.assertIn('기준이 바뀌',view.status.get())
            finally:
                if app.vision:app.vision.close()
                if app.preview:app.preview.close()
                for job in root.tk.call('after','info'):root.after_cancel(job)
                root.destroy()


if __name__=='__main__':unittest.main()
