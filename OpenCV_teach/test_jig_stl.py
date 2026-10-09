from copy import deepcopy
from pathlib import Path
import json,queue,tempfile,time,unittest
from unittest.mock import patch
import numpy as np
import jig_stl as stl
from jig_catalog import register,profiles
from jig_detection import profile_signature
from test_jig_catalog import jig
import vision_teaching as v
import teach_left_arm as t

SOURCE=Path(__file__).parent/'assets/jigs/carrier_6jigs_220x148_M3_flat_lightweight.stl'

class STLTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.asset_patch=patch.object(stl,'ASSETS',self.root/'assets');self.asset_patch.start()
    def tearDown(self):self.asset_patch.stop();self.temp.cleanup()
    def test_import_canonical_origin_units_and_persistence(self):
        tri=stl.triangles(SOURCE);tri=tri@np.array([[0,-1,0],[1,0,0],[0,0,1.]]).T+[100,200,37]
        source=self.root/'offset.stl';source.write_bytes(stl.binary_stl(tri))
        model=stl.import_stl(source);source.unlink();vertices=stl.triangles(stl.resolve_asset(model))
        lo=vertices.min(axis=(0,1));hi=vertices.max(axis=(0,1));np.testing.assert_allclose((lo+hi)[:2],0,atol=1e-5);self.assertAlmostEqual(lo[2],0)
        np.testing.assert_allclose(hi-lo,model['dimensions_mm'],atol=1e-4);self.assertGreater(hi[0]-lo[0],hi[1]-lo[1])
        self.assertEqual(model['bottom_z_mm'],-7.4);self.assertAlmostEqual(model['rim_height_mm'],4.8);self.assertNotIn('pose',model)
        station,key=register(v.default_station({}),{**jig(),'preview_model':model})
        loaded=profiles(json.loads(json.dumps(station)))[key]['preview_model'];self.assertEqual(loaded,model)
        self.assertEqual(profile_signature(station['jigs'][key]),profile_signature({**jig(),"surface_z_mm":stl.detection_height(model)}))
    def test_ascii_unit_and_invalid_or_tampered_input(self):
        tri=stl.triangles(SOURCE)/1000
        text='solid jig\n'+''.join('facet normal 0 0 0\nouter loop\n'+''.join('vertex %g %g %g\n'%tuple(p) for p in face)+'endloop\nendfacet\n' for face in tri)+'endsolid jig\n'
        path=self.root/'ascii.stl';path.write_text(text);model=stl.import_stl(path,'m');self.assertAlmostEqual(model['dimensions_mm'][0],220,places=2)
        stl.resolve_asset(model).write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError,'변경'):stl.resolve_asset(model)
        path.write_text('bad')
        with self.assertRaises(ValueError):stl.import_stl(path)
        with self.assertRaises(ValueError):stl.validate_model({**model,'asset':'../../foo.stl'})
        with self.assertRaises(ValueError):stl.validate_model({**model,'bottom_z_mm':float('nan')})
    def test_model_placement_keeps_robot_geometry_and_supports_holes(self):
        import mujoco
        from robot_preview import SCENE
        m=stl.import_stl(SOURCE);profile={**jig(),'preview_model':m};item=stl.observed_scenes([profile],{'jigs':{}},{v.reference_key(profile):dict(x_mm=210,y_mm=0,yaw_deg=0)})[0];model=stl.model_with_meshes(SCENE,[item]);data=mujoco.MjData(model)
        joints=model.nq;stl.place_meshes(model,[{**item,'x':280,'y':-50,'z':25,'yaw':90}]);mujoco.mj_forward(model,data)
        np.testing.assert_allclose(data.body('jig_visual_0').xpos,[.28,-.05,.025],atol=1e-9)
        np.testing.assert_allclose(data.body('jig_visual_0').xmat.reshape(3,3)[:,0],[0,1,0],atol=1e-9)
        self.assertEqual(joints,mujoco.MjModel.from_xml_path(str(SCENE)).nq)
        geom=model.geom_bodyid==model.body('jig_visual_0').id
        self.assertTrue(np.all(model.geom_contype[geom]==0))

class RimTests(unittest.TestCase):
    def test_internal_high_face_does_not_replace_perimeter(self):
        def top(x,y,z):
            return np.array([[[-x,-y,z],[x,-y,z],[x,y,z]],[[-x,-y,z],[x,y,z],[-x,y,z]]],float)
        faces=np.concatenate([top(100,60,0),top(100,60,3),top(40,20,20)])
        self.assertEqual(stl.outer_rim_height(faces),3.)
        self.assertEqual(stl.outer_rim_height(np.concatenate([faces,*[top(40,20,20) for _ in range(20)]])),3.)

class FakeRenderer:
    def __init__(self,*_):self.inbox=queue.Queue()
    def submit(self,item):return True
    def close(self):pass

class STLPanelTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        self.temp=tempfile.TemporaryDirectory();self.ap=patch.object(stl,'ASSETS',Path(self.temp.name)/'assets');self.ap.start()
        self.renderer=patch('robot_preview.RenderClient',FakeRenderer);self.renderer.start()
        self.root=tk.Tk();self.app=t.App(self.root,t.Demo(),self.temp.name);self.view=self.app.open_vision(show=False);self.panel=self.view.catalog
    def tearDown(self):
        self.app.close()
        for job in self.root.tk.call('after','info'):self.root.after_cancel(job)
        self.root.destroy();self.renderer.stop();self.ap.stop();self.temp.cleanup()
    def test_registration_hides_jig_until_detection_and_derives_outer_rim(self):
        self.root.geometry('1080x760');self.view.select_tab(1);self.root.update()
        self.assertTrue(self.panel.save_button.winfo_viewable())
        before=deepcopy(self.app.vision_plan)
        with patch.object(self.app,'send',side_effect=AssertionError('motor command')):
            self.panel.stl.load(stl.import_stl(SOURCE))
            self.assertFalse(hasattr(self.panel.stl,'vars'))
            self.assertAlmostEqual(float(self.panel.vars['surface_z_mm'].get()),-2.6)
            self.assertEqual(str(self.panel.fields['surface_z_mm'].cget('state')),'readonly')
            self.assertEqual(self.app.preview.confirmed_jig_scenes(),[])
            self.app.preview.tick();self.assertEqual(self.app.preview.last_request[4],[])
            self.panel.stl.use_dimensions();self.assertTrue(self.panel.save(),self.panel.status.get())
            key=self.panel.key;saved=json.loads(self.view.profile_path.read_text())['jigs'][key]
            self.assertNotIn('pose',saved['preview_model']);self.assertAlmostEqual(saved['surface_z_mm'],-2.6)
            self.assertEqual(self.app.vision_plan,before)
            self.panel.new();self.panel.list.current(self.panel.keys.index(key));self.panel.load()
            self.assertEqual(self.panel.stl.read(),saved['preview_model'])
            self.assertEqual(self.app.preview.confirmed_jig_scenes(),[])

    def test_old_virtual_xy_angle_are_ignored_and_saved_rim_applies_to_detector(self):
        model=stl.import_stl(SOURCE);model.pop('rim_height_mm')
        model['pose']=dict(x_mm=999,y_mm=-999,z_mm=99,yaw_deg=170)
        cleaned=stl.with_rim_metadata(model);self.assertNotIn('pose',cleaned)
        self.assertAlmostEqual(cleaned['rim_height_mm'],4.8)
        self.panel.stl.load(cleaned);self.assertTrue(self.panel.save())
        from jig_catalog import choose
        choose(self.app,self.panel.key)
        for t0 in self.app.vision_plan['targets'].values():t0['surface_z_mm']=100
        self.view.apply_saved_detection_settings()
        self.assertTrue(all(abs(t0['surface_z_mm']+2.6)<1e-6 for t0 in self.app.vision_plan['targets'].values()))
        self.assertEqual(self.app.preview.confirmed_jig_scenes() if self.app.preview else [],[])

    def test_confirmed_camera_pose_updates_mesh_and_survives_occlusion(self):
        from jig_catalog import choose
        self.panel.stl.load(stl.import_stl(SOURCE))
        self.assertTrue(self.panel.save());key=self.panel.key;choose(self.app,key)
        self.app.show_preview();self.app.preview.show_step_settings()
        target=self.view.read_target();obs=dict(x_mm=290.,y_mm=-80.,yaw_deg=-15.,profile_signature=profile_signature(target))
        self.view.jig_reference.capture(target,obs,self.view.station)
        scene=self.app.preview.confirmed_jig_scenes()
        self.assertEqual(len(scene),1);item=scene[0]
        self.assertEqual([item[k] for k in ('x','y','yaw','z')],[290.,-80.,-15.,-7.4]);self.assertIn('mesh_asset',item)
        self.assertFalse(item['nominal']);self.assertEqual(item['height_source'],'stl_outer_rim_at_detection_plane')
        self.view.observations={key:{**obs,'x_mm':999}}  # Unconfirmed drift cannot move the simulation.
        self.assertEqual(self.app.preview.confirmed_jig_scenes(),scene)
        self.view.observations.clear();self.assertEqual(self.app.preview.confirmed_jig_scenes(),scene)
        self.app.latest={'sample_time':time.monotonic(),'follower':dict.fromkeys(t.KEYS,0.)};self.app.state='PLAY'
        self.app.preview.tick();self.assertFalse(self.app.preview.settings_visible)
        from robot_preview import CURRENT
        self.assertEqual(self.app.preview.mode.get(),CURRENT)
        self.view.jig_reference.clear();self.assertEqual(self.app.preview.confirmed_jig_scenes(),[])
        from robot_preview import SIMULATION
        self.app.state='DISCONNECTED';self.app.preview.simulation_scene=scene;self.app.preview.mode.set(SIMULATION)
        self.app.preview.tick();self.assertEqual(self.app.preview.last_request[4],[])
        self.app.state='DISCONNECTED'

    def test_missing_mesh_falls_back_to_measured_outline_without_nominal_xy(self):
        model=stl.import_stl(SOURCE);self.panel.stl.load(model);self.assertTrue(self.panel.save());profile=self.panel.read()
        key=v.reference_key(profile);stl.resolve_asset(model).unlink()
        scenes=stl.observed_scenes([profile],self.view.station,{key:dict(x_mm=333,y_mm=22,yaw_deg=19)})
        self.assertNotIn('mesh_asset',scenes[0]);self.assertEqual(scenes[0]['x'],333)
        self.assertEqual(scenes[0]['z'],float(profile['surface_z_mm']));self.assertIn('model_warning',scenes[0])
