from copy import deepcopy
from pathlib import Path
import json,tempfile,unittest
from unittest.mock import patch
import tkinter as tk
import cv2,numpy as np
import teach_left_arm as t
import vision_teaching as v
from jig_detection import PROFILE_DEFAULTS,profile_signature,detect_rectangle
from jig_catalog import register,profiles,select_plan,teaching_target,choose,refresh_picker
from test_vision_teaching import optical_station,project,rectangle_world

def jig(name='지그 A',**kw):
    return {'mode':'rectangle',**PROFILE_DEFAULTS,'product':'공통 지그','jig_name':name,'shape_model':'rectangle',**kw}

class ProfileTests(unittest.TestCase):
    def test_multiple_profiles_have_unique_identity_and_no_part_recipe(self):
        station=v.default_station(t.Demo.identity);old=deepcopy(station)
        station,a=register(station,{**jig(),'x_mm':123,'closed':12})
        station,b=register(station,jig('지그 B',long_mm=160,short_mm=80))
        self.assertNotEqual(a,b);self.assertEqual(len(station['jigs']),2)
        self.assertNotIn('x_mm',station['jigs'][a]);self.assertNotIn('closed',station['jigs'][a])
        self.assertEqual(old['jigs'],{})
        with self.assertRaisesRegex(ValueError,'같은 이름'):register(station,jig())
        edited={**station['jigs'][a],'long_mm':240}
        updated,key=register(station,edited,a);self.assertEqual(key,a)
        self.assertEqual(updated['jigs'][b],station['jigs'][b])
    def test_legacy_keys_and_signatures_remain_readable(self):
        p=jig();signature=profile_signature(p);s={'jigs':{v.reference_key(p):p}}
        self.assertIn('rectangle:지그 A',profiles(s))
        self.assertEqual(profile_signature({**p,'jig_id':'immutable-id'}),signature)
        self.assertNotEqual(v.reference_key(p),v.reference_key({**p,'jig_id':'immutable-id'}))
    def test_adding_jig_preserves_bound_and_temporarily_fixed_steps(self):
        station=v.default_station(t.Demo.identity);station,a=register(station,jig());station,b=register(station,jig('지그 B'))
        plan,ident,_=select_plan([],{'targets':{}},station['jigs'][a],station)
        pose=dict.fromkeys(t.KEYS,0.);steps=[{'name':'잡기','pose':pose,'motion':{'target':ident,'phase':'waypoint'}}]
        before=deepcopy(plan)
        both,bid,added=select_plan(steps,plan,station['jigs'][b],station)
        self.assertTrue(added);self.assertEqual(both['targets'][ident],before['targets'][ident]);self.assertNotEqual(ident,bid)
        steps[0]['follow_motion']=steps[0].pop('motion')
        self.assertEqual(select_plan(steps,plan,station['jigs'][b],station)[0]['targets'][ident],before['targets'][ident])
        changed={**station['jigs'][a],'long_mm':500}
        retained,_,dirty=select_plan(steps,plan,changed,station)
        self.assertFalse(dirty);self.assertEqual(retained,before);self.assertEqual(plan,before)
    def test_generic_rectangles_ignore_different_internal_structure_and_match_size(self):
        station=optical_station();frame=np.full((720,1280,3),45,np.uint8)
        for center,length,width in [((120,-120,0),120,60),((280,120,0),200,100)]:
            corners=project(rectangle_world(center=center,long=length,short=width),station).round().astype(np.int32)
            cv2.fillConvexPoly(frame,corners,(235,235,235))
            centre=project([center],station)[0].round().astype(int)
            cv2.circle(frame,tuple(centre),13,(45,45,45),-1)
        for name,center,length,width in [('small',(120,-120),120,60),('large',(280,120),200,100)]:
            _,value=detect_rectangle(frame,jig(name,long_mm=length,short_mm=width),station)
            np.testing.assert_allclose([value['x_mm'],value['y_mm']],center,atol=2.)

class CatalogUITests(unittest.TestCase):
    def setUp(self):
        self.root=tk.Tk();self.temp=tempfile.TemporaryDirectory();self.app=t.App(self.root,t.Demo(),self.temp.name);self.view=self.app.open_vision(show=False)
    def tearDown(self):
        self.app.close()
        for job in self.root.tk.call('after','info'):self.root.after_cancel(job)
        self.root.destroy();self.temp.cleanup()
    def save_jig(self,name,length=220,width=148):
        panel=self.view.catalog;panel.new();panel.vars['jig_name'].set(name);panel.vars['long_mm'].set(str(length));panel.vars['short_mm'].set(str(width));self.assertTrue(panel.save(),panel.status.get());return panel.key
    def test_registration_selection_capture_and_reload_keep_selected_profile(self):
        with patch.object(self.app,'send',side_effect=AssertionError('Selection moved robot')):
            a=self.save_jig('흰색 큰 지그');b=self.save_jig('작은 지그',160,80)
            choose(self.app,b)
            self.assertEqual(self.app.jig_picker.get(),'작은 지그')
            target=self.view.read_target();self.assertEqual(v.reference_key(target),b)
            self.assertEqual(target['long_mm'],160)
            self.assertEqual(self.app.steps,[])
            obs={'x_mm':200.,'y_mm':0.,'yaw_deg':0.,'profile_signature':profile_signature(target)}
            self.view.teaching_reference.capture(target,obs,self.view.station)
            self.app.capture_follow.set(True);context=self.app.new_capture_context()
            self.app.store_captured_step({'pose':dict.fromkeys(t.KEYS,0.),'captured_at':''},context)
            binding=self.app.steps[0]['motion'];self.assertEqual(v.reference_key(self.app.vision_plan['targets'][binding['target']]),b)
            original=deepcopy(self.app.vision_plan)
            choose(self.app,a);self.assertEqual(len(self.app.vision_plan['targets']),len(original['targets'])+1)
            self.assertEqual(self.app.vision_plan['targets'][binding['target']],original['targets'][binding['target']])
            choose(self.app,b)
            path=t.save_named_teaching(self.app.output,{**self.app.document(),'title':'catalog selected'})
            self.app.load_path(path);self.assertEqual(self.app.jig_picker.get(),'작은 지그')
            self.assertEqual(v.reference_key(self.view.read_target()),b)
    def test_profile_edit_keeps_episode_snapshot_and_other_roi(self):
        a=self.save_jig('A');b=self.save_jig('B',180,90);choose(self.app,a)
        p=self.view.read_target();self.view.teaching_reference.capture(p,{'x_mm':200,'y_mm':0,'yaw_deg':0,'profile_signature':profile_signature(p)},self.view.station)
        self.app.capture_follow.set(True);self.app.store_captured_step({'pose':dict.fromkeys(t.KEYS,0.)},self.app.new_capture_context())
        old=deepcopy(self.app.vision_plan);other=deepcopy(self.view.station['jigs'][b])
        panel=self.view.catalog;panel.list.current(panel.keys.index(a));panel.load();panel.vars['long_mm'].set('250');self.assertTrue(panel.save())
        self.assertEqual(self.app.vision_plan,old);self.assertEqual(self.view.station['jigs'][b],other)
        self.assertEqual(json.loads(self.view.profile_path.read_text())['jigs'][a]['long_mm'],250)
        self.view.vars['t_roi_x_pct'].set('10');self.view.vars['t_roi_w_pct'].set('80')
        self.view.save_detection_method()
        self.assertEqual(self.view.station['jigs'][a]['long_mm'],250)
        self.assertEqual(self.view.station['jigs'][a]['roi_x_pct'],10)
        self.assertEqual(self.view.station['jigs'][b],other)
        self.assertEqual(self.view.read_target()['long_mm'],220)
        self.view.save_target()
        self.assertEqual(self.view.station['jigs'][a]['long_mm'],250)
