"""Temporal observation and execution-time capture without physical devices."""
from copy import deepcopy
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

import vision_teaching as v
from carrier_detection import Candidate
from jig_detection import detect_rectangle, profile_signature
from test_vision_teaching import target, optical_station, project, rectangle_world


class ConsensusTests(unittest.TestCase):
    def test_short_missing_frames_keep_window_but_cannot_count_or_finalize_until_redetected(self):
        tracker=v.StableJigs();self.fill(tracker)
        tracker.missing('j',3.1,31);tracker.missing('j',3.1,31)
        self.assertEqual(tracker.progress('j',3.1),3.)
        state=tracker.assessment('j',3.1)
        self.assertFalse(state['ready']);self.assertEqual(state['missing'],1)
        tracker.missing('j',3.2,32)
        tracker.observe('j',self.value(33),3.3,33)
        state=tracker.assessment('j',3.3)
        self.assertTrue(state['ready']);self.assertEqual(state['missing'],2)
        self.assertEqual(state['value']['missing_count'],2)
        tracker.missing('j',4.,40)
        self.assertNotIn('j',tracker.history)

    def test_frequent_missing_and_ambiguous_detection_cannot_be_accepted(self):
        tracker=v.StableJigs()
        for i in range(31):
            if i%2:tracker.missing('j',i*.1,i)
            else:tracker.observe('j',self.value(),i*.1,i)
        self.assertEqual(tracker.progress('j',3),3.)
        self.assertFalse(tracker.assessment('j',3)['ready'])
        tracker.missing('j',3.1,31,hard=True)
        self.assertNotIn('j',tracker.history)

    def value(self, i=0, **changes):
        return dict(x_mm=200.+.2*(i%2),y_mm=0.,yaw_deg=.2*(i%2),
                    profile_signature='same',quality='precise' if i%3==0 else 'medium',**changes)

    def fill(self, tracker, end=30):
        for i in range(end+1):tracker.observe('j',self.value(i),i*.1,i)

    def test_mixed_precision_needs_full_three_seconds_and_records_mean(self):
        tracker=v.StableJigs();self.fill(tracker,29)
        with self.assertRaisesRegex(ValueError,'2.9/3.0'):tracker.snapshot(['j'],2.9)
        tracker.observe('j',self.value(30),3.,30)
        value=tracker.snapshot(['j'],3.01)['j']
        self.assertEqual(value['sample_count'],31)
        self.assertEqual(value['quality_counts'],{'precise':11,'medium':20})
        self.assertAlmostEqual(value['x_mm'],200.+5./53.)
        self.assertEqual(value['quality_weight_sums'],{'precise':33.,'medium':20.})
        with self.assertRaises(ValueError):tracker.snapshot(['j'],3.7)

    def test_missing_low_gap_profile_change_and_duplicates_reset_or_do_not_count(self):
        for event in ('missing','low','gap','profile','backwards'):
            with self.subTest(event=event):
                tracker=v.StableJigs();self.fill(tracker)
                if event=='missing':tracker.missing('j')
                else:
                    value=self.value();stamp=3.1
                    if event=='low':value['quality']='low'
                    if event=='profile':value['profile_signature']='new'
                    if event=='gap':stamp=4.
                    if event=='backwards':stamp=2.
                    tracker.observe('j',value,stamp,31)
                with self.assertRaises(ValueError):tracker.snapshot(['j'],4.01)
        tracker=v.StableJigs()
        for i in range(40):tracker.observe('j',self.value(),i*.1,1)
        self.assertEqual(tracker.progress('j',3.9),0)
        with self.assertRaises(ValueError):tracker.snapshot(['j'],3.9)

    def test_slow_drift_and_angular_jump_restart_window(self):
        for axis in ('x_mm','yaw_deg'):
            tracker=v.StableJigs()
            for i in range(41):
                value=self.value();value[axis]=i*.15
                tracker.observe('j',value,i*.1,i)
            self.assertLess(tracker.progress('j',4.),3.)
            with self.assertRaises(ValueError):tracker.snapshot(['j'],4.)

    def test_medium_only_is_insufficient_and_pixel_half_turn_wrap_is_supported(self):
        tracker=v.StableJigs(position_tolerance=2,angle_period=180)
        for i in range(31):
            value=self.value();value.update(quality='medium',yaw_deg=179.8 if i%2 else .2)
            tracker.observe('j',value,i*.1,i)
        with self.assertRaisesRegex(ValueError,'정밀 0/2'):tracker.snapshot(['j'],3.01)
        tracker.clear()
        for i in range(31):
            value=self.value();value.update(quality='precise' if i in (1,15,30) else 'medium',yaw_deg=179.8 if i%2 else .2)
            tracker.observe('j',value,i*.1,i)
        value=tracker.snapshot(['j'],3.01)['j']
        self.assertEqual(value['quality_counts']['precise'],3)
        self.assertLess(abs((value['yaw_deg']+90)%180-90),.21)

    def test_two_precise_frames_still_require_three_seconds_of_stability(self):
        tracker=v.StableJigs()
        for i in range(31):
            value=self.value(i);value['quality']='precise' if i in (5,15) else 'medium'
            tracker.observe('j',value,i*.1,i)
            if i<30:self.assertFalse(tracker.assessment('j',i*.1)['ready'])
        self.assertEqual(tracker.snapshot(['j'],3.01)['j']['quality_counts']['precise'],2)
        self.assertIn('정밀 2/2회',tracker.status_text('j',3.01))

    def test_precision_and_confidence_control_weight_without_changing_outlier_filter(self):
        def measure(confidence):
            tracker=v.StableJigs()
            for i in range(31):
                precise=i in (5,15)
                value=self.value();value.update(x_mm=200.6 if precise else 200.,
                    yaw_deg=-179.8 if precise else 179.8,quality='precise' if precise else 'medium',
                    detection_confidence=confidence if precise else 1.)
                tracker.observe('j',value,i*.1,i)
            return tracker.snapshot(['j'],3.01)['j']
        strong,weak=measure(1.),measure(.5)
        self.assertAlmostEqual(strong['x_mm'],200.+.6*6/35)
        self.assertAlmostEqual(weak['x_mm'],200.+.6*1.5/30.5)
        self.assertGreater(strong['x_mm'],weak['x_mm'])
        self.assertGreater(abs(strong['yaw_deg']),179.)
        self.assertEqual(strong['quality_counts']['precise'],2)
        self.assertEqual(strong['estimator'],'quality_weighted_inlier_mean')

    def test_precise_count_alone_does_not_hide_insufficient_valid_measurements(self):
        tracker=v.StableJigs()
        for i in range(31):
            if i%3==1:tracker.missing('j',i*.1,i)
            else:tracker.observe('j',{**self.value(i),'quality':'precise'},i*.1,i)
        state=tracker.assessment('j',3.)
        self.assertGreaterEqual(state['precise'],2)
        self.assertFalse(state['ready'])
        self.assertTrue(any('유효 측정' in reason for reason in state['reasons']))
        with self.assertRaisesRegex(ValueError,'필요 70%'):tracker.snapshot(['j'],3.)

    def test_medium_still_requires_calibrated_geometry_and_rejects_low(self):
        station=optical_station();corners=project(rectangle_world(),station).astype(np.float32)
        frame=np.zeros((720,1280,3),np.uint8);profile=target(shape_model='carrier_6jigs')
        medium=Candidate(corners,.9,False,tracking_ok=True)
        _,value=detect_rectangle(frame,profile,station,candidates=[medium])
        self.assertEqual(value['quality'],'medium')
        self.assertAlmostEqual(value['detection_confidence'],.9)
        for hypotheses,t in (([Candidate(corners,.9,False)],profile),([medium],{**profile,'long_mm':300}),([medium,medium],profile)):
            with self.assertRaises(ValueError):detect_rectangle(frame,t,station,candidates=hypotheses)
        with self.assertRaises(ValueError):detect_rectangle(frame,profile,{**station,'extrinsics':None},candidates=[medium])

    def test_outliers_do_not_drag_centre_or_count_as_precise_evidence(self):
        tracker=v.StableJigs()
        for i in range(31):
            value=self.value(i);value['quality']='precise' if i in (2,12,22) else 'medium'
            if i in (5,18):value.update(x_mm=260.,yaw_deg=80.,quality='precise')
            tracker.observe('j',value,i*.1,i)
        result=tracker.snapshot(['j'],3.01)['j']
        self.assertEqual(result['rejected_count'],2)
        self.assertEqual(result['quality_counts']['precise'],3)
        self.assertAlmostEqual(result['x_mm'],200.,delta=.2)
        self.assertAlmostEqual(result['yaw_deg'],0.,delta=.2)
        # Two valid precise samples suffice; the rejected spikes never count.
        for item in tracker.history['j']:
            if item['frame_id']==22:item['quality']='medium'
        self.assertEqual(tracker.snapshot(['j'],3.01)['j']['quality_counts']['precise'],2)
        for item in tracker.history['j']:
            if item['frame_id']==12:item['quality']='medium'
        with self.assertRaisesRegex(ValueError,'정밀 1/2'):tracker.snapshot(['j'],3.01)

    def test_old_precise_samples_expire_and_two_displacements_restart(self):
        tracker=v.StableJigs()
        for i in range(35):
            value=self.value(i);value['quality']='precise' if i<3 else 'medium'
            tracker.observe('j',value,i*.1,i)
        with self.assertRaisesRegex(ValueError,'정밀 0/2'):tracker.snapshot(['j'],3.41)
        tracker.clear();self.fill(tracker)
        tracker.observe('j',{**self.value(),'x_mm':250},3.1,31)
        self.assertTrue(tracker.assessment('j',3.1)['ready'])  # One isolated spike.
        tracker.observe('j',{**self.value(),'x_mm':250},3.2,32)
        self.assertLess(tracker.progress('j',3.2),.2)
        with self.assertRaises(ValueError):tracker.snapshot(['j'],3.2)

    def test_two_inconsistent_clusters_cannot_be_cherry_picked(self):
        tracker=v.StableJigs()
        for i in range(31):
            value=self.value();value.update(x_mm=200 if i%2 else 220,quality='precise')
            tracker.observe('j',value,i*.1,i)
        with self.assertRaises(ValueError):tracker.snapshot(['j'],3.01)

    def test_consecutive_scattered_spikes_wait_for_clean_sample_without_erasing_window(self):
        tracker=v.StableJigs();self.fill(tracker)
        tracker.observe('j',{**self.value(),'x_mm':250},3.1,31)
        tracker.observe('j',{**self.value(),'x_mm':150},3.2,32)
        self.assertEqual(tracker.progress('j',3.2),3)
        self.assertFalse(tracker.assessment('j',3.2)['ready'])
        tracker.observe('j',self.value(),3.3,33)
        self.assertTrue(tracker.assessment('j',3.3)['ready'])


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        import teach_left_arm as teaching
        self.folder=tempfile.TemporaryDirectory();self.root=tk.Tk()
        self.app=teaching.App(self.root,teaching.Demo(),self.folder.name)
        self.view=self.app.open_vision();self.view.vars['t_return_to_start'].set('집고 제자리에 놓기');self.view.add_steps()
        self.target=self.view.read_target();self.key=v.reference_key(self.target)
        self.value=dict(x_mm=200.,y_mm=0.,yaw_deg=0.,quality='medium',profile_signature=profile_signature(self.target))
        self.app.state='HOLD';self.app.latest={'sample_time':time.monotonic(),'follower':self.view.seed()}
        self.camera=patch.object(self.view,'process_frame');self.camera.start()

    def tearDown(self):
        self.camera.stop();self.view.close()
        if self.app.preview:self.app.preview.close()
        for job in self.root.tk.call('after','info'):self.root.after_cancel(job)
        self.root.destroy();self.folder.cleanup()

    def feed(self, seconds=3.):
        now=time.monotonic();self.view.tracker.clear()
        for i in range(round(seconds*10)+1):
            self.view.tracker.observe(self.key,{**self.value,'quality':'precise' if i%3==0 else 'medium'},now-seconds+i*.1,i)
        self.view.frame_live=True;self.view.frame_at=now;self.app.latest['sample_time']=now

    def test_full_episode_replaces_old_reference_with_current_three_second_mean(self):
        self.view.jig_reference.capture(self.target,{**self.value,'x_mm':180},self.view.station)
        old=self.view.jig_reference.value['id'];self.feed()
        with patch('tkinter.messagebox.askokcancel',return_value=True),patch.object(self.app,'send') as send:
            self.view.execute();frame=100
            end=time.monotonic()+8
            while self.app.vision_busy and time.monotonic()<end:
                now=time.monotonic();self.view.frame_at=now;self.app.latest['sample_time']=now
                self.view.tracker.observe(self.key,self.value,now,frame);frame+=1
                self.root.update();time.sleep(.01)
            self.assertFalse(self.app.vision_busy,self.view.status.get())
            self.assertEqual(send.call_count,1,self.view.status.get())
            command,payload=send.call_args.args;self.assertEqual(command,'play')
            self.assertEqual(len(payload['steps']),7)
            evidence=payload['vision_evidence']['jigs'][self.key]
            self.assertNotEqual(evidence['reference_id'],old);self.assertEqual(evidence['x_mm'],200)
            self.assertGreaterEqual(evidence['sample_count'],2)
            self.assertEqual(evidence['estimator'],'quick_quality_weighted_mean')

    def test_full_episode_cannot_fall_back_to_old_latch_when_unstable_or_occluded(self):
        for visible in (True,False):
            self.feed(2.9);self.view.frame_live=visible
            self.view.jig_reference.capture(self.target,self.value,self.view.station)
            with patch.object(self.app,'send') as send:
                if visible:
                    self.view.execute();self.assertIsNotNone(self.view.reference_acquisition.pending)
                    self.view.reference_acquisition.cancel('test cancellation')
                else:
                    with self.assertRaises(ValueError):self.view.execute()
                send.assert_not_called()

    def test_loss_of_consensus_during_execution_confirmation_blocks_play(self):
        self.feed()
        def confirm(*_a,**_k):self.view.tracker.clear();return True
        with patch('tkinter.messagebox.askokcancel',side_effect=confirm),patch.object(self.app,'send') as send:
            self.view.execute();end=time.monotonic()+8
            while self.app.vision_busy and time.monotonic()<end:self.root.update();time.sleep(.01)
            send.assert_not_called();self.assertIn('3초 측정 종료',self.view.status.get())

    def test_loading_episode_selects_its_target_in_existing_and_new_camera_window(self):
        import json
        from pathlib import Path
        import teach_left_arm as teaching
        t={**self.target,'jig_name':'현재 작업 지그','name':'부품 목표'}
        doc={'schema':2,'kind':'teaching','identity':self.app.backend.identity,'units':teaching.UNITS,
             'title':'선택 확인','steps':v.make_steps('loaded',t,self.view.seed()),
             'vision_plan':{'targets':{'loaded':t},'calibration_signature':v.calibration_signature(self.view.station)}}
        path=Path(self.folder.name)/'selection.json';path.write_text(json.dumps(doc))
        self.app.load_path(path)
        self.assertEqual(self.view.target_id,'loaded');self.assertEqual(self.view.read_target()['jig_name'],'현재 작업 지그')
        self.camera.stop();self.view.close();self.app.vision=None
        self.view=self.app.open_vision()
        self.camera=patch.object(self.view,'process_frame');self.camera.start()
        self.assertEqual(self.view.target_id,'loaded');self.assertEqual(self.view.read_target()['jig_name'],'현재 작업 지그')


if __name__=='__main__':unittest.main()
