from copy import deepcopy
from pathlib import Path
import json,queue,tempfile,time,unittest
from unittest.mock import patch
import tkinter as tk
import numpy as np
import teach_left_arm as t
import vision_teaching as v
from jig_catalog import choose,register,teaching_target
from jig_detection import profile_signature
from jig_reference import JigReferences
from test_jig_catalog import jig
from step_reference import change_reference,StepReferenceEditor


def observation(target,x=200,y=0):
    return dict(x_mm=x,y_mm=y,yaw_deg=0.,at=time.monotonic(),profile_signature=profile_signature(target))

class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=tk.Tk();self.app=t.App(self.root,t.Demo(),self.tmp.name);self.view=self.app.open_vision(show=False)
        for name,x in [('A',0),('B',50)]:
            self.view.station,key=register(self.view.station,jig(name,roi_x_pct=x,roi_w_pct=50))
        self.keys=list(self.view.station['jigs']);self.view.save_station()
    def tearDown(self):
        self.app.close()
        for job in self.root.tk.call('after','info'):self.root.after_cancel(job)
        self.root.destroy();self.tmp.cleanup()
    def capture(self,key):
        choose(self.app,key);p=self.view.read_target();self.view.teaching_auto_pending=False
        self.view.teaching_reference.capture(p,observation(p),self.view.station)
        self.app.capture_follow.set(True);self.app.store_captured_step({'pose':dict.fromkeys(t.KEYS,0.)},self.app.new_capture_context())
    def test_two_jigs_steps_roi_reload_and_rebind(self):
        a,b=self.keys
        with patch.object(self.app,'send',side_effect=AssertionError('unexpected motor command')):
            self.capture(a);a_binding=deepcopy(self.app.steps[0]['motion']);self.capture(b)
            self.assertEqual(self.app.steps[0]['motion'],a_binding)
            self.assertEqual(len(self.view.teaching_reference.values),2)
            self.view.roi_jig_picker.current(self.view.roi_jig_ids.index(next(k for k,p in self.app.vision_plan['targets'].items() if v.reference_key(p)==a)))
            self.view.select_roi_jig();self.view.vars['t_roi_w_pct'].set('40');self.view.save_detection_method()
            self.assertEqual(self.view.station['jigs'][a]['roi_w_pct'],40)
            self.assertEqual(self.view.station['jigs'][b]['roi_w_pct'],50)
            self.assertEqual(self.view.station['jigs'][b]['roi_x_pct'],50)
            path=t.save_named_teaching(self.app.output,{**self.app.document(),'title':'두 지그'})
            self.app.load_path(path)
            keys=[v.reference_key(self.app.vision_plan['targets'][s['motion']['target']]) for s in self.app.steps]
            self.assertEqual(keys,[a,b])
            p=next(p for p in self.app.vision_plan['targets'].values() if v.reference_key(p)==b)
            self.view.teaching_reference.capture(p,observation(p,300),self.view.station)
            editor=StepReferenceEditor(self.app,0);self.app.reference_editor=editor
            editor.jig_choice.current(editor.jig_keys.index(b));editor.apply()
            self.assertIsNone(self.app.reference_editor,editor.message.get())
            changed=next(s for s in self.app.steps if s.get('motion'))
            self.assertEqual(v.reference_key(self.app.vision_plan['targets'][changed['motion']['target']]),b)
    def test_acquisition_captures_both_before_execution_callback(self):
        self.capture(self.keys[0]);self.capture(self.keys[1]);self.view.teaching_auto_pending=False
        plan=self.app.vision_plan;self.view.frame_live=True;self.view.frame_at=time.monotonic()
        observations={v.reference_key(p):observation(p,200+i*100) for i,p in enumerate(plan['targets'].values())}
        with patch.object(self.view,'observations_for',return_value=observations),patch.object(self.view,'process_frame'),patch.object(self.view,'update_jig_reference'):
            calls=[];self.view.reference_acquisition.begin(plan,lambda:calls.append(self.view.reference_observations(plan)),show=False)
        self.assertEqual(set(calls[0]),set(self.keys));self.assertEqual(len(self.view.jig_reference.values),2)
    def test_step_settings_replace_only_preview_region(self):
        self.app.show_preview();self.capture(self.keys[0]);self.app.navigation.home();self.root.geometry('1440x900');self.root.update()
        self.app.edit_step_reference();self.root.update()
        panel=self.app.preview.step_settings
        self.assertIsNone(self.app.navigation.current);self.assertIsNone(self.app.reference_editor)
        self.assertTrue(self.app.table.winfo_viewable());self.assertTrue(self.app.preview.frame.winfo_viewable())
        self.assertTrue(self.app.preview.settings_visible);self.assertFalse(self.app.preview.canvas.winfo_ismapped())
        self.assertTrue(panel.window.winfo_viewable())
        self.app.preview.show_simulation();self.root.update();self.assertTrue(self.app.preview.canvas.winfo_viewable())
        self.app.preview.show_step_settings();self.root.update()
        self.assertEqual(panel.title.get(),self.app.steps[self.app.selected()]['name'])
        panel.mode.set('고정 위치');panel.apply()
        self.assertNotIn('motion',self.app.steps[self.app.selected()])
        panel.mode.set('지그 따라가기');panel.apply();self.assertIn('motion',self.app.steps[self.app.selected()])
        self.app.open_manual();self.root.update()
        self.assertTrue(self.view.window.dock_preview);self.assertFalse(self.view.camera_pane.winfo_ismapped())
        self.assertTrue(self.app.preview.frame.winfo_viewable())
        self.view.select_tab(3);self.root.update();self.assertTrue(self.view.camera_pane.winfo_ismapped())
        self.app.navigation.home();self.root.update();self.assertTrue(self.app.preview.frame.winfo_viewable())

    def test_rename_preserves_path_steps_and_jig_data(self):
        self.capture(self.keys[0]);path=t.save_named_teaching(self.app.output,{**self.app.document(),'title':'before'})
        self.app.saved_path=path;before=json.loads(path.read_text())
        with patch('teach_left_arm.ask_teaching_title',return_value='바뀐 이름'):self.app.rename_episode()
        after=json.loads(path.read_text());self.assertEqual(after['title'],'바뀐 이름');self.assertEqual(after['steps'],before['steps']);self.assertEqual(after['vision_plan'],before['vision_plan'])
        self.assertEqual(self.app.saved_path,path);self.assertEqual(len(list(self.app.output.glob('before*.json'))),1)
        with patch('teach_left_arm.ask_teaching_title',return_value='   '):self.app.rename_episode()
        self.assertEqual(json.loads(path.read_text())['title'],'바뀐 이름')

class ContractTests(unittest.TestCase):
    def test_reference_bank_keeps_others_and_capture_is_atomic(self):
        station=v.default_station({});a=teaching_target(jig('A'));b=teaching_target(jig('B'))
        ref=JigReferences();obs={v.reference_key(a):observation(a),v.reference_key(b):observation(b,300)}
        ref.capture_many([a,b],obs,station);before=deepcopy(ref.values)
        with self.assertRaises(ValueError):ref.capture_many([a,b],{**obs,v.reference_key(b):{**obs[v.reference_key(b)],'profile_signature':'bad'}},station)
        self.assertEqual(ref.values,before)
        ref.check(a,station);ref.check(b,station);self.assertEqual(ref.values,before)
        self.assertTrue(ref.observe_key(v.reference_key(a),observation(a,250)))
        self.assertFalse(ref.values)
    def test_follower_connect_capture_activate_release_never_implicitly_powers(self):
        backend=t.Demo();backend.follower_only=True
        with tempfile.TemporaryDirectory() as folder,patch.object(backend,'activate',wraps=backend.activate) as activate,patch.object(backend,'send',wraps=backend.send) as send:
            session=t.Session(backend,folder,auto_sync=True);session.start();events=[]
            def wait(predicate):
                deadline=time.monotonic()+4
                while time.monotonic()<deadline:
                    session.heartbeat=time.monotonic()
                    try:
                        while True:events.append(session.events.get_nowait())
                    except queue.Empty:pass
                    if predicate():return
                    time.sleep(.005)
                self.fail((session.state,events[-3:]))
            try:
                wait(lambda:len([e for e in events if e[0]=='status'])>=8)
                self.assertEqual(session.state,'READY');activate.assert_not_called();send.assert_not_called()
                backend.current=dict.fromkeys(t.KEYS,13.);backend.target=backend.current.copy()
                changed_at=time.monotonic()
                wait(lambda:any(k=='status' and e['sample_time']>=changed_at+t.CAPTURE_HISTORY_SECONDS+.1 for k,e in events))
                session.command('capture','freehand');wait(lambda:any(k=='captured' for k,_ in events))
                self.assertEqual(next(e for k,e in events if k=='captured')['pose'],backend.current)
                activate.assert_not_called();send.assert_not_called()
                session.command('hold_current');wait(lambda:session.state=='HOLD');activate.assert_called_once()
                session.command('release');wait(lambda:session.state=='READY');count=send.call_count
                n=len(events);wait(lambda:len(events)>=n+5);self.assertEqual(send.call_count,count)
            finally:session.stop.set();session.wakeup.set();session.thread.join(3)
            self.assertFalse(session.thread.is_alive())
    def test_pair_registration_and_mode_validation_without_opening_ports(self):
        from launch_mode import register_pair,validate_selection
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);data={name:dict(id=i,drive_mode=0,homing_offset=0,range_min=1000,range_max=3000) for i,name in enumerate(t.JOINTS,1)}
            a=p/'leader.json';b=p/'follower.json';a.write_text(json.dumps(data));b.write_text(json.dumps(data))
            profile=register_pair(p,'/dev/serial/by-id/test-leader',a,'/dev/serial/by-id/test-follower',b)
            with patch('serial.Serial.open',side_effect=AssertionError('port opened')):
                pair,path=validate_selection('pair',profile);self.assertFalse(pair)
                single,_=validate_selection('follower_only',profile);self.assertTrue(single)
            document=json.loads(profile.read_text());copies=p/'calibration'/document['pair_name']
            self.assertEqual((copies/'leader_left.json').read_bytes(),a.read_bytes());self.assertEqual((copies/'follower_left.json').read_bytes(),b.read_bytes())
    def test_mount_matches_camera_and_raised_table_geometry(self):
        import mujoco
        from mount_scene import camera_pose,base_height_mm
        from robot_preview import SCENE
        from model_camera import model_extrinsics
        from test_vision_teaching import optical_station
        model=mujoco.MjModel.from_xml_path(str(SCENE));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        p,r=camera_pose();np.testing.assert_allclose(data.site('overhead_optical_center').xpos,p,atol=1e-9)
        tf=np.array(model_extrinsics(optical_station(),0)['base_from_camera']);np.testing.assert_allclose(tf[:3,3]/1000,p);np.testing.assert_allclose(tf[:3,:3],r)
        self.assertEqual(base_height_mm(),7.4)
        c=json.loads((Path(__file__).resolve().parents[1]/'Test_teach/config/so101_scene.json').read_text())
        self.assertAlmostEqual(c['base_mm'][2]-c['table']['height_mm'],7.4)
        self.assertAlmostEqual(c['base_mm'][2]+c['carrier']['center_base_mm'][2],c['table']['height_mm'])
