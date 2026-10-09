import tempfile,unittest,tkinter as tk
from unittest.mock import patch
import teach_left_arm as t
from test_jig_stl import FakeRenderer

class RetryTests(unittest.TestCase):
    def test_dropped_last_frame_is_retried_without_resubmitting_acknowledged_pose(self):
        root=tk.Tk();tmp=tempfile.TemporaryDirectory()
        with patch('robot_preview.RenderClient',FakeRenderer):
            app=t.App(root,t.Demo(),tmp.name);app.steps=[{'name':'pose','pose':dict.fromkeys(t.KEYS,0.)}];app.refresh_steps(0);app.show_preview();preview=app.preview
            try:
                preview.last_request=None;preview.last_frame_angles=None
                with patch.object(preview.client,'submit',return_value=True) as submit,patch('robot_preview.time.monotonic',return_value=10):preview.tick()
                self.assertEqual(submit.call_count,1);request=preview.last_request
                with patch.object(preview.client,'submit',return_value=True) as submit,patch('robot_preview.time.monotonic',return_value=10.7):preview.tick()
                self.assertEqual(submit.call_count,1);self.assertEqual(preview.last_request,request)
                preview.last_frame_angles=request[1]
                with patch.object(preview.client,'submit',return_value=True) as submit,patch('robot_preview.time.monotonic',return_value=11.4):preview.tick()
                submit.assert_not_called()
            finally:
                app.close()
                from tk_test_support import cancel_jobs
                cancel_jobs(root);root.destroy();tmp.cleanup()
