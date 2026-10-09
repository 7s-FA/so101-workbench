import json,queue,tempfile,tkinter as tk,unittest
from pathlib import Path
from unittest.mock import patch,MagicMock
import teach_left_arm as t
from vision_teaching import default_station,digest
from test_jig_stl import FakeRenderer

class StartupTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=tk.Tk()
        self.renderer=patch('robot_preview.RenderClient',FakeRenderer);self.renderer.start()
        station=default_station(t.Demo.identity);station['camera']={'source':'/dev/video-test','width':960,'height':540}
        path=Path(self.tmp.name)/'vision'/(digest(t.Demo.identity)[:16]+'.json');path.parent.mkdir();path.write_text(json.dumps(station))
        self.app=t.App(self.root,t.Demo(),self.tmp.name)
    def tearDown(self):
        self.app.close()
        for job in self.root.tk.call('after','info'):self.root.after_cancel(job)
        self.root.destroy();self.renderer.stop();self.tmp.cleanup()
    def test_startup_saved_camera_in_background_once_without_motor_commands(self):
        context=MagicMock();context.Queue.return_value=queue.Queue();context.Queue.return_value.close=lambda:None
        context.Process.return_value.is_alive.return_value=True
        page=self.app.navigation.current
        with patch('vision_workspace.mp.get_context',return_value=context),patch.object(self.app,'send',side_effect=AssertionError('motor command')):
            self.root.after_idle(self.app.connect_startup_camera);self.root.update()
            self.app.connect_startup_camera()
        context.Process.assert_called_once()
        self.assertEqual(context.Process.call_args.kwargs['args'][:3],('/dev/video-test',960,540))
        context.Process.return_value.start.assert_called_once()
        self.assertIs(self.app.navigation.current,page);self.assertEqual(self.app.state,'DISCONNECTED')
    def test_error_visible_and_manual_reconnect_available(self):
        context=MagicMock();q=queue.Queue();q.close=lambda:None;context.Queue.return_value=q
        context.Process.return_value.is_alive.return_value=True
        with patch('vision_workspace.mp.get_context',return_value=context):
            self.app.connect_startup_camera();view=self.app.vision
            q.put(('error','카메라 없음'));view.poll()
            self.assertIsNone(view.process);self.assertIn('카메라 없음',view.camera_status.get())
            self.assertIn('다시 시도',view.status.get())
            view.start_camera();self.assertEqual(context.Process.call_count,2)
    def test_closing_or_existing_camera_never_reopens(self):
        self.app.closing=True
        with patch.object(self.app,'open_vision') as open_view:self.app.connect_startup_camera();open_view.assert_not_called()
        self.app.closing=False;view=self.app.open_vision(show=False);view.process=MagicMock()
        with patch.object(view,'start_camera') as start:self.app.connect_startup_camera();start.assert_not_called()
        view.process=None
    def test_synchronous_start_failure_keeps_app_usable(self):
        view=self.app.open_vision(show=False)
        with patch.object(view,'start_camera',side_effect=ValueError('잘못된 설정')):self.app.connect_startup_camera()
        self.assertIn('자동 연결 실패',view.camera_status.get());self.assertEqual(self.app.state,'DISCONNECTED')

if __name__=='__main__':unittest.main()
