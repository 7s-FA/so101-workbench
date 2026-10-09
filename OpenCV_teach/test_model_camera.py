from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from model_camera import model_extrinsics,calibration_label
from test_vision_teaching import optical_station,project
from vision_teaching import pixels_on_plane


class ModelCameraTests(unittest.TestCase):
    def test_offset_and_right_handed_axes_with_explicit_estimate_metadata(self):
        s=optical_station();a=model_extrinsics(s,0);b=model_extrinsics(s,20)
        tf=np.array(b['base_from_camera']);r=tf[:3,:3]
        np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-8)
        self.assertAlmostEqual(np.linalg.det(r),1)
        np.testing.assert_allclose(tf[:3,3]-np.array(a['base_from_camera'])[:3,3],r[:,2]*20)
        np.testing.assert_allclose(tf[:3,3],[34.43956,-116.04772,527.54564],atol=.0001)
        self.assertFalse(b['verified_for_robot_motion']);self.assertNotIn('origin_mm',b)
        self.assertIn('모델 추정',calibration_label({**s,'extrinsics':b}))

    def test_projected_plane_roundtrip_and_stale_intrinsics_rejected(self):
        s=optical_station();s['extrinsics']=model_extrinsics(s)
        points=np.array([[200,0,2.4],[240,-70,2.4],[170,-100,2.4]])
        np.testing.assert_allclose(pixels_on_plane(project(points,s),2.4,s,[1280,720]),points,atol=.001)
        s['intrinsics']['D'][0]+=.01
        with self.assertRaises(ValueError):pixels_on_plane([[640,360]],2.4,s,[1280,720])

    def test_gui_saves_estimate_without_changing_tool_or_intrinsics(self):
        import tkinter as tk
        import teach_left_arm as teaching
        root=tk.Tk()
        with tempfile.TemporaryDirectory() as folder:
            app=teaching.App(root,teaching.Demo(),folder);view=app.open_vision()
            try:
                view.station=optical_station();before=deepcopy(view.station)
                with patch.object(app,'send',side_effect=AssertionError('motor command')):
                    view.apply_model_camera()
                saved=json.loads(view.profile_path.read_text())
                self.assertEqual(saved['tool'],before['tool']);self.assertEqual(saved['intrinsics'],before['intrinsics'])
                self.assertEqual(saved['extrinsics']['source'],'cad_estimate')
                view.close();app.vision=None
                view=app.open_vision()  # No measured checkerboard origin required for CAD profiles.
                self.assertIn('모델 추정',view.setup_status.get())
            finally:
                view.close()
                for job in root.tk.call('after','info'):root.after_cancel(job)
                root.destroy()


if __name__=='__main__':unittest.main()
