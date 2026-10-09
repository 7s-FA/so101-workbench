"""Task pages share one OS window; back never moves or disconnects motors."""
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch
import teach_left_arm as t

class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.root=tk.Tk();self.temp=tempfile.TemporaryDirectory()
        self.app=t.App(self.root,t.Demo(),self.temp.name)
    def tearDown(self):
        self.app.close()
        for job in self.root.tk.call('after','info'):self.root.after_cancel(job)
        self.root.destroy();self.temp.cleanup()
    def test_navigation_reuses_main_window_for_all_operational_pages(self):
        with patch('tkinter.Toplevel',side_effect=AssertionError('new OS window')):
            self.app.open_manual();view=self.app.vision;self.root.update()
            self.assertIs(self.app.navigation.current,view.window)
            self.app.open_settings();self.assertIs(self.app.navigation.current,view.settings_window)
            view.open_motor_calibration();page=view.calibration_window.window
            self.assertIs(page.winfo_toplevel(),self.root)
            self.app.navigation.back();self.assertIsNone(view.calibration_window)
            self.assertIs(self.app.navigation.current,view.settings_window)
            monitor=self.app.open_monitor();self.assertIs(monitor.root.winfo_toplevel(),self.root)
            self.app.navigation.home();self.assertIsNone(self.app.navigation.current)
    def test_active_menu_tracks_direct_tab_changes_and_back_navigation(self):
        def selected(key):
            self.assertEqual(self.app.navigation.active_section,key)
            for name in ('episode','manual','vision','settings'):
                self.assertEqual(str(self.app.buttons[name].cget('style')),'Active.Nav.TButton' if name==key else 'Nav.TButton')
        selected('episode')
        self.app.open_manual();self.root.update();selected('manual')
        view=self.app.vision;view.notebook.select(view.pages[3]);self.root.update();selected('vision')
        self.app.open_settings();self.root.update();selected('settings')
        self.app.navigation.back();self.root.update();selected('vision')
        self.app.navigation.home();selected('episode')

    def test_navigation_clicks_preserve_saved_episode_and_selected_step(self):
        self.app.show_preview()
        self.app.steps=[{'name':name,'pose':dict.fromkeys(t.KEYS,0.)} for name in ('first','second')]
        path=t.save_named_teaching(self.app.output,{**self.app.document(),'title':'Keep selection'})
        self.app.refresh_library();self.app.load_path(path);self.app.refresh_library()
        self.app.table.selection_set('1');self.app.select();self.app.preview_selected_step()
        self.assertIsNotNone(self.app.preview)
        from robot_preview import TARGET
        self.assertEqual(self.app.preview.mode.get(),TARGET)
        label=self.app.library.get()
        self.root.geometry('1280x850');self.root.update()
        for name in ('manual','vision','settings','episode'):
            button=self.app.buttons[name]
            button.event_generate('<ButtonPress-1>',x=10,y=10)
            button.event_generate('<ButtonRelease-1>',x=10,y=10);self.root.update()
            self.assertEqual(self.app.navigation.active_section,name)
            self.assertEqual(self.app.preview.mode.get(),TARGET)
            self.assertEqual(self.app.selected(),1,name)
            self.assertEqual(self.app.saved_path,path,name)
            self.assertEqual(self.app.library.get(),label,name)
        self.app.open_settings();self.root.update()
        back=self.app.navigation.host.winfo_children()[0].winfo_children()[0]
        back.event_generate('<ButtonPress-1>',x=10,y=10)
        back.event_generate('<ButtonRelease-1>',x=10,y=10);self.root.update()
        self.assertEqual(self.app.selected(),1)

    def test_back_cancels_unsaved_step_edit_without_a_motor_command(self):
        self.app.steps=[{'name':'one','pose':dict.fromkeys(t.KEYS,0.)}];self.app.refresh_steps(0)
        self.app.edit_step_pose();editor=self.app.pose_editor
        editor.motor_values['shoulder_pan.pos'].set('2200')
        with patch.object(self.app,'stop',side_effect=AssertionError('back stopped robot')),patch.object(self.app,'send',side_effect=AssertionError('back moved robot')):
            self.app.navigation.back()
        self.assertIsNone(self.app.pose_editor);self.assertEqual(self.app.steps[0]['pose']['shoulder_pan.pos'],0)
    def test_global_navigation_clears_editor_before_entering_manual_page(self):
        self.app.steps=[{'name':'one','pose':dict.fromkeys(t.KEYS,0.)}];self.app.refresh_steps(0);self.app.edit_step_pose()
        self.app.buttons['manual'].invoke();self.root.update()
        self.assertIsNone(self.app.pose_editor)
        self.assertIs(self.app.navigation.current,self.app.vision.window)

    def test_stop_cancels_calibration_waiting_for_old_connection_to_close(self):
        self.app.open_settings();self.app.vision.open_motor_calibration()
        cal=self.app.vision.calibration_window
        self.app.backend.identity={**self.app.backend.identity,'demo':False}
        cal.port.set('/dev/test');cal.arm_id.set('follower_test')
        with patch('motor_calibration.make_arm',side_effect=AssertionError('cancelled calibration opened USB')):
            cal.start();self.assertIsNotNone(cal.pending_start)
            self.app.stop();self.root.update()
            self.assertIsNone(self.app.vision.calibration_window)
            self.assertIsNone(cal.worker)
            self.assertFalse(self.app.calibration_busy)

if __name__=='__main__':unittest.main()
