from copy import deepcopy
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from test_jig_catalog import CatalogUITests
from test_catalog_capture_ring import CaptureTests
from test_vision_teaching import optical_station
from jig_catalog import choose
from camera_profile_recovery import recover_camera
from model_camera import model_extrinsics

class HeadingApplyTests(CatalogUITests):
    def test_saved_heading_reaches_existing_episode_only_for_same_jig(self):
        a=self.save_jig('A');b=self.save_jig('B');choose(self.app,a);choose(self.app,b)
        old=deepcopy(self.app.vision_plan);steps=deepcopy(self.app.steps)
        panel=self.view.catalog;panel.list.current(panel.keys.index(a));panel.load()
        panel.vars['expected_yaw_deg'].set('42.5');panel.vars['yaw_tolerance_deg'].set('12')
        self.assertTrue(panel.save());self.assertEqual(self.app.steps,steps)
        from vision_teaching import reference_key
        for key,t in self.app.vision_plan['targets'].items():
            if reference_key(t)==a:
                self.assertEqual(t['expected_yaw_deg'],42.5);self.assertEqual(t['yaw_tolerance_deg'],12.)
                for name,value in old['targets'][key].items():
                    if name not in ('expected_yaw_deg','yaw_tolerance_deg'):self.assertEqual(t[name],value)
            else:self.assertEqual(t,old['targets'][key])
        self.app.vision_plan=old;self.view.load_target()
        self.assertTrue(any(t['expected_yaw_deg']==42.5 for t in self.app.vision_plan['targets'].values()))

class MissingCalibrationTests(CaptureTests):
    def test_missing_calibration_explained_before_measurement(self):
        self.view.station['intrinsics']=None;self.view.station['extrinsics']=None
        self.panel.vars['expected_yaw_deg'].set('17');self.panel.capture_heading()
        self.assertIsNone(self.view.reference_acquisition.pending)
        self.assertEqual(self.panel.vars['expected_yaw_deg'].get(),'17')
        self.assertIn('적용하지 못했습니다',self.view.status.get())
        self.assertEqual(self.view.settings_notebook.select(),str(self.view.pages[0]))

class RecoveryTests(unittest.TestCase):
    def test_motor_zero_change_restores_camera_not_tool_or_jigs(self):
        with tempfile.TemporaryDirectory() as folder:
            old=optical_station();old['identity']={'demo':False,'serials':{'follower':'arm1'},'calibration_sha256':{'follower':'old'}}
            old['extrinsics']=model_extrinsics(old)
            current=deepcopy(old);current['identity']['calibration_sha256']['follower']='new'
            current['intrinsics']=current['extrinsics']=None;current['tool']['xyz_mm']=[1,2,3];current['jigs']={'kept':{}}
            Path(folder,'old.json').write_text(json.dumps(old))
            with patch('camera_profile_recovery.device_path',return_value=Path('/dev/same')):
                recovered,source=recover_camera(current,folder)
            self.assertEqual(source,'old.json')
            for key in ('camera','intrinsics','extrinsics'):self.assertEqual(recovered[key],old[key])
            for key in ('identity','tool','jigs','board'):self.assertEqual(recovered[key],current[key])
            self.assertIsNone(current['intrinsics'])
    def test_foreign_camera_robot_existing_or_conflicting_calibration_not_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            old=optical_station();old['identity']={'demo':False,'serials':{'follower':'arm1'}};old['extrinsics']=model_extrinsics(old)
            current=deepcopy(old);current['intrinsics']=current['extrinsics']=None
            p=Path(folder,'old.json');p.write_text(json.dumps(old))
            with patch('camera_profile_recovery.device_path',return_value=Path('/dev/same')):
                self.assertIsNone(recover_camera(old,folder)[1])
                other=deepcopy(current);other['identity']['serials']['follower']='arm2'
                self.assertIsNone(recover_camera(other,folder)[1])
                alternate=deepcopy(old);alternate['intrinsics']['K'][0][0]+=10
                alternate['extrinsics']=model_extrinsics(alternate)
                Path(folder,'other.json').write_text(json.dumps(alternate))
                self.assertIsNone(recover_camera(current,folder)[1])
            with patch('camera_profile_recovery.device_path',side_effect=[Path('/dev/one'),Path('/dev/two'),Path('/dev/two')]):
                self.assertIsNone(recover_camera(current,folder)[1])
