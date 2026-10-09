"""Camera/simulation coordinate transfer without opening cameras or motors."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

import simulation_bridge as bridge
import vision_teaching as v
import teach_left_arm as teaching


class BridgeTests(unittest.TestCase):
    def setUp(self):
        # Operator-edited simulation settings are live data, not a test fixture.
        # Keep the known reachable, noncentral seating case independent of them.
        fixture = {'schema':1, 'robot':'so101', 'part_order':list(bridge.PARTS),
            'carrier':{'center_base_mm':[210,0,-2.4], 'yaw_deg':-90, 'size_mm':[220,148,6]},
            'home_deg':[0,-30,50,60,0,50], 'parts':{},
            'tcp':{**v.default_station({})['tool'], 'xyz_mm':[1.773409675938293,.6026567343814877,6.549653899876567]},
            'contact':{'static_friction':1., 'dynamic_friction':.8}, 'mass_source':'test fixture'}
        for kind,center,seat,grasp,opening in (
                ('cap',[-72,36,4.85],[0,4,8.05],[25,0,2.35],55),
                ('ball',[0,-36,4.85],[0,4,1.60209134817],[15,0,11],45),
                ('housing',[72,36,4.85],[0,4,3.55],[25,0,5],55)):
            fixture['parts'][kind] = {'jig_center_carrier_mm':center, 'part_offset_jig_mm':seat,
                'grasp_tcp_part_mm':grasp, 'jig_yaw_deg':0, 'part_yaw_deg':0,
                'grasp_yaw_deg':0, 'grasp_pitch_deg':180, 'hover_mm':40, 'lift_mm':40,
                'open_percent':opening, 'close_percent':0}
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup)
        source=Path(directory.name)/'so101_test_scene.json';source.write_text(json.dumps(fixture))
        read_source=bridge.read_source
        patcher=patch.object(bridge,'read_source',side_effect=lambda path=bridge.SOURCE:read_source(source))
        patcher.start();self.addCleanup(patcher.stop)
        self.config=bridge.read_source()
        self.station=v.default_station(teaching.Demo.identity)
        self.station['tool']=deepcopy(self.config['tcp'])

    def test_transforms_jig_and_part_offsets_including_noncentral_seating(self):
        c=self.config
        for kind in bridge.PARTS:
            p=c['parts'][kind];t=bridge.convert_target(c,kind)
            local=np.array(p['jig_center_carrier_mm'])+np.array(p['part_offset_jig_mm'])+np.array([-abs(p['grasp_tcp_part_mm'][0]),*p['grasp_tcp_part_mm'][1:]])
            np.testing.assert_allclose([t['x_mm'],t['y_mm']],local[:2],atol=1e-8)
            self.assertAlmostEqual(t['z_mm'],local[2]+c['carrier']['center_base_mm'][2])
            self.assertAlmostEqual(t['surface_z_mm'],c['carrier']['center_base_mm'][2]+4.8)
            self.assertFalse(t['geometry_verified'])
        c['parts']['ball']['jig_yaw_deg']=23;c['parts']['ball']['part_yaw_deg']=-41
        t=bridge.convert_target(c,'ball');p=c['parts']['ball']
        expected=np.array(p['jig_center_carrier_mm'])+Rotation.from_euler('z',23,degrees=True).apply(
            np.array(p['part_offset_jig_mm'])+Rotation.from_euler('z',-41,degrees=True).apply([-abs(p['grasp_tcp_part_mm'][0]),*p['grasp_tcp_part_mm'][1:]]))
        np.testing.assert_allclose([t['x_mm'],t['y_mm']],expected[:2]);self.assertAlmostEqual(t['yaw_deg'],162)

    def test_measured_installation_height_changes_rim_and_target_together(self):
        a=bridge.convert_target(self.config,'cap')
        b=bridge.convert_target(self.config,'cap',carrier_bottom_z_mm=27.6,carrier_yaw_deg=25)
        self.assertAlmostEqual(b['z_mm']-a['z_mm'],30)
        self.assertAlmostEqual(b['surface_z_mm']-a['surface_z_mm'],30)
        self.assertEqual(b['expected_yaw_deg'],25)
        self.assertEqual(b['x_mm'],a['x_mm'])

    def test_candidates_are_separate_complete_camera_plans_and_preserve_calibration(self):
        before=deepcopy(self.station)
        for kind in bridge.PARTS:
            doc=bridge.candidate(self.config,kind,self.station)
            self.assertEqual(len(doc['steps']),9)
            self.assertEqual([s['motion']['phase'] for s in doc['steps'] if s.get('motion')],list(v.PHASES+v.RETURN_PHASES))
            self.assertTrue(all(s['path'] for s in doc['steps'] if s.get('motion')))
            teaching.validate_program(doc,doc['identity'])
            from episode_preview import EpisodePlayer
            player=EpisodePlayer();player.load(doc);self.assertGreater(player.duration,0)
            plan=doc['vision_plan'];t=next(iter(plan['targets'].values()))
            obs={v.reference_key(t):dict(x_mm=210,y_mm=0,yaw_deg=-90)}
            compiled=v.compile_steps(doc['steps'],plan,self.station,obs,doc['steps'][0]['pose'],live=True)
            self.assertEqual(len(compiled),9)
            self.assertEqual(compiled[0]['pose'],compiled[-1]['pose'])
        self.assertEqual(self.station,before)
        confirmed=deepcopy(self.station);confirmed['tool']['verified']=True
        doc=bridge.candidate(self.config,'ball',confirmed);t=next(iter(doc['vision_plan']['targets'].values()))
        obs={v.reference_key(t):dict(x_mm=210,y_mm=0,yaw_deg=-90)}
        self.assertEqual(len(v.compile_steps(doc['steps'],doc['vision_plan'],confirmed,obs,doc['steps'][0]['pose'],live=True)),9)

    def test_pitch_and_separate_retreat_survive_compile_and_old_targets_default(self):
        doc=bridge.candidate(self.config,'ball',self.station);plan=doc['vision_plan']
        t=next(iter(plan['targets'].values()));t['pitch_deg']=175;t['retreat_mm']=30
        obs={v.reference_key(t):dict(x_mm=210,y_mm=0,yaw_deg=-90)}
        _,tilted=v.target_pose(t,obs)
        np.testing.assert_allclose(tilted,Rotation.from_euler('xyz',[180,-5,90],degrees=True).as_matrix(),atol=1e-12)
        # Five arm axes do not permit every tilted yaw at an arbitrary XYZ.
        # Use the known reachable downward pose for the vertical-path check.
        t['pitch_deg']=180
        full=v.compile_steps(doc['steps'],plan,self.station,obs,doc['steps'][0]['pose'])
        self.assertEqual(full[0]['pose'],full[-1]['pose'])
        self.assertEqual(full[0]['pose'],doc['steps'][0]['pose'])
        steps=full[1:-1]
        original=doc['steps'][1:-1]
        kin=v.Kinematics(self.station['tool']);low=kin.fk(steps[1]['pose']);high=kin.fk(steps[3]['pose'])
        self.assertAlmostEqual(high[2,3]-low[2,3],30,delta=.5)
        _,expected=v.target_pose(t,obs);np.testing.assert_allclose(low[:3,:3],expected,atol=.02)
        t.pop('pitch_deg');t.pop('retreat_mm');old=v.validate_target(t)
        self.assertEqual(old['pitch_deg'],180);self.assertEqual(old['retreat_mm'],old['hover_mm'])

    def test_saving_candidates_never_replaces_previous_teaching(self):
        with tempfile.TemporaryDirectory() as directory:
            first=bridge.save_candidates(directory,self.station)
            before={k:p.read_bytes() for k,p in first.items()}
            second=bridge.save_candidates(directory,self.station)
            for k in first:
                self.assertNotEqual(first[k],second[k]);self.assertEqual(first[k].read_bytes(),before[k])

    def test_gui_import_preserves_existing_draft_station_and_no_hardware(self):
        import tkinter as tk
        try:root=tk.Tk()
        except tk.TclError as exc:self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as directory:
            app=teaching.App(root,teaching.Demo(),directory)
            try:
                view=app.open_vision();view.station['tool']=deepcopy(self.config['tcp'])
                before=deepcopy(view.station)
                app.steps=[{'name':'기존 초안','pose':dict.fromkeys(teaching.KEYS,0.),'captured_at':''}]
                original=app.program_path
                view.sim_part.current(1)
                with patch.object(app,'send',side_effect=AssertionError('motor command')),patch('vision_workspace.mp.get_context',side_effect=AssertionError('camera')),patch.object(bridge,'read_source',return_value=self.modern_config()):
                    view.import_simulation()
                    deadline=time.monotonic()+15
                    while app.vision_busy and time.monotonic()<deadline:root.update();time.sleep(.01)
                self.assertFalse(app.vision_busy,view.status.get())
                self.assertEqual(len(app.steps),9,view.status.get())
                self.assertEqual(view.target_id,'sim_A_ball')
                self.assertEqual(view.station,before)
                self.assertTrue(view.read_target()['return_to_start'])
                self.assertEqual(view.read_target()['lift_tilt_deg'],25)
                self.assertEqual(view.read_target()['approach_x_mm'],-3)
                self.assertEqual(view.read_target()['retreat_wait_s'],3)
                view.save_target()
                self.assertEqual(app.vision_plan['targets'][view.target_id]['lift_tilt_deg'],25)
                self.assertEqual(json.loads(original.read_text())['steps'][0]['name'],'기존 초안')
                named=[p for p in Path(directory).glob('*.json') if json.loads(p.read_text()).get('kind')=='teaching']
                self.assertEqual(len(named),3)
            finally:
                if app.vision:app.vision.close()
                if app.preview:app.preview.close()
                for job in root.tk.call('after','info'):root.after_cancel(job)
                root.destroy()

    def modern_config(self):
        c=deepcopy(self.config)
        c['home_deg']=[0,-55,15,85,90,15]
        for kind,p in c['parts'].items():
            p['grasp_tcp_part_mm'][0]+=.7
            p.update(lift_mm=100, lift_clearance_mm=40, lift_tilt_deg=25, approach_clearance_mm=3)
        c['parts']['ball'].update(jig_yaw_deg=90,part_yaw_deg=-90)
        return c

    def test_latest_geometry_roundtrip_returns_same_pose_and_reverses_lift(self):
        c=self.modern_config();kin=v.Kinematics(self.station['tool'])
        for kind in bridge.PARTS:
            with self.subTest(part=kind):
                doc=bridge.candidate(c,kind,self.station);steps=doc['steps'][1:-1]
                t=next(iter(doc['vision_plan']['targets'].values()))
                observation={v.reference_key(t):dict(x_mm=210,y_mm=0,yaw_deg=-90)}
                goal,rotation=v.target_pose(t,observation)
                entry=v.approach_point(t,observation,goal)
                for node in steps[1]['path']:
                    tf=kin.fk(node['pose'])
                    self.assertGreaterEqual(tf[2,3],goal[2]- .5)
                np.testing.assert_allclose(kin.fk(steps[1]['pose'])[:3,3],goal,atol=.5)
                self.assertEqual(steps[4]['pose'],steps[2]['pose'])
                ascent=[steps[2]['pose']]+[n['pose'] for n in steps[3]['path']]
                self.assertEqual([n['pose'] for n in steps[4]['path']],list(reversed(ascent[:-1])))
                for node in steps[3]['path']:
                    tf=kin.fk(node['pose'])
                    self.assertTrue(np.isfinite(tf).all())
                    tilt=Rotation.from_matrix(tf[:3,:3]@rotation.T).magnitude()*180/np.pi
                    self.assertLessEqual(tilt,90)  # Upper posture aims at contact while height takes priority.
                    self.assertEqual(node['pose']['gripper.pos'],t['closed'])
                heights=[kin.fk(q)[2,3] for q in ascent]
                self.assertTrue(all(b>=a-.05 for a,b in zip(heights,heights[1:])))
                self.assertAlmostEqual(heights[-1]-goal[2],100,delta=.5)
                self.assertEqual(steps[3]['path'][-1]['hold_s'],3)
                self.assertEqual(steps[5]['pose']['gripper.pos'],t['open'])
                np.testing.assert_allclose(kin.fk(steps[6]['pose']),kin.fk(steps[0]['pose']),atol=.02)
                self.assertEqual(steps[6]['pose']['gripper.pos'],t['open'])
                teaching.validate_program(doc,doc['identity'])

    def test_relocated_docking_recomputes_roundtrip_and_does_not_use_saved_joints(self):
        c=self.modern_config();doc=bridge.candidate(c,'ball',self.station)
        plan=doc['vision_plan'];t=next(iter(plan['targets'].values()))
        obs={v.reference_key(t):dict(x_mm=214,y_mm=3,yaw_deg=-85)}
        full=v.compile_steps(doc['steps'],plan,self.station,obs,doc['steps'][0]['pose'])
        self.assertEqual(full[0]['pose'],full[-1]['pose'])
        self.assertEqual(full[0]['pose'],doc['steps'][0]['pose'])
        steps=full[1:-1]
        original=doc['steps'][1:-1]
        kin=v.Kinematics(self.station['tool']);goal,rotation=v.target_pose(t,obs)
        self.assertGreater(np.linalg.norm(kin.fk(steps[2]['pose'])[:3,3]-kin.fk(original[2]['pose'])[:3,3]),2)
        np.testing.assert_allclose(kin.fk(steps[2]['pose'])[:3,3],goal,atol=.5)
        self.assertEqual(steps[4]['pose'],steps[2]['pose'])
        self.assertEqual(v.compile_steps([steps[4]],plan,self.station,obs,steps[3]['pose'],single=True)[0]['pose']['gripper.pos'],t['closed'])
        with self.assertRaisesRegex(ValueError,'앞 단계'):
            v.compile_steps([steps[4]],plan,self.station,obs,steps[0]['pose'],single=True)
        with self.assertRaises(ValueError):
            v.compile_steps([steps[5],steps[4]],plan,self.station,obs,steps[3]['pose'])

    def test_approach_offset_rotates_with_part_even_when_grasp_yaw_differs(self):
        c=self.modern_config();p=c['parts']['ball'];p.update(jig_yaw_deg=20,part_yaw_deg=10,grasp_yaw_deg=40)
        t=bridge.convert_target(c,'ball')
        np.testing.assert_allclose([t['approach_x_mm'],t['approach_y_mm']],Rotation.from_euler('z',30,degrees=True).apply([-3,0,0])[:2])
        for changes in ({'lift_tilt_deg':31},{'approach_x_mm':11},{'return_to_start':'yes'},
                        {'return_to_start':True,'operation':'place'},{'retreat_wait_s':float('nan')}):
            with self.assertRaises(ValueError):v.validate_target({**t,**changes})


if __name__=='__main__':unittest.main()
