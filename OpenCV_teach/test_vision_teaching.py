"""Synthetic optical and robot regressions; no physical devices are opened."""
from copy import deepcopy
import math
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import vision_teaching as v
import teach_left_arm as teaching


def target(**changes):
    from jig_detection import PROFILE_DEFAULTS
    return v.validate_target({**PROFILE_DEFAULTS, 'name':'1번 칸', 'product':'제품 A',
        'mode':'rectangle', 'shape_model':'rectangle', 'operation':'pick', 'surface_z_mm':30,
        'x_mm':0,'y_mm':0,'z_mm':30,'yaw_deg':180,'hover_mm':20,'open':80,'closed':20,
        'speed_mm_s':10,'wait_s':.1, **changes})


def rectangle_world(center=(200,0,30), yaw=0, long=220, short=148):
    local=np.array([[-long/2,short/2,0],[long/2,short/2,0],[long/2,-short/2,0],[-long/2,-short/2,0]],float)
    return local@Rotation.from_euler('z',yaw,degrees=True).as_matrix().T+center


def optical_station():
    s=v.default_station(teaching.Demo.identity)
    s['camera']={'source':'synthetic','width':1280,'height':720}
    s['intrinsics']={'K':[[900.,0,640],[0,900.,360],[0,0,1]],'D':[.04,-.02,.001,.001,0],
                     'size':[1280,720],'camera':deepcopy(s['camera'])}
    tf=np.eye(4);tf[:3,:3]=Rotation.from_euler('x',180,degrees=True).as_matrix();tf[:3,3]=[200,0,700]
    s['extrinsics']={'base_from_camera':tf.tolist(),'intrinsics_signature':v.digest(s['intrinsics'])}
    return s


def project(points,s):
    camera_from_base=np.linalg.inv(s['extrinsics']['base_from_camera'])
    return cv2.projectPoints(np.asarray(points,float),cv2.Rodrigues(camera_from_base[:3,:3])[0],
                             camera_from_base[:3,3],np.array(s['intrinsics']['K']),np.array(s['intrinsics']['D']))[0].reshape(-1,2)


class OpticalTests(unittest.TestCase):
    def test_low_view_angle_diversity_cannot_enable_metric_coordinates(self):
        s=optical_station();obj=v.board_points(s['board']);samples=[]
        for i in range(12):
            points,_=cv2.projectPoints(obj,np.array([.03,.02,i*.03]),np.array([-70.+i*4,-60.,650.]),
                                      np.array(s['intrinsics']['K']),np.array(s['intrinsics']['D']))
            samples.append(points.astype(np.float32))
        candidate=v.calibrate_intrinsics(samples,[1280,720],s['board'],s['camera'])
        self.assertFalse(candidate['usable_for_metric'])
        s['intrinsics']=candidate
        with self.assertRaisesRegex(ValueError,'기울인'):v.require_intrinsics(s,[1280,720])

    def test_detect_rectangle_pixels_and_reject_duplicate_geometry(self):
        from jig_detection import detect_rectangle
        station=optical_station();t=target()
        image=np.full((720,1280,3),40,np.uint8)
        cv2.fillConvexPoly(image,project(rectangle_world(),station).astype(np.int32),(220,220,220))
        corners,pose=detect_rectangle(image,t,station)
        self.assertAlmostEqual(pose['x_mm'],200,delta=1.)
        self.assertAlmostEqual(pose['yaw_deg'],0,delta=1.)
        cv2.fillConvexPoly(image,project(rectangle_world((450,0,30)),station).astype(np.int32),(220,220,220))
        with self.assertRaisesRegex(ValueError,'여러 개'):detect_rectangle(image,t,station)

    def test_distortion_height_perspective_and_rotated_slot_offset(self):
        from jig_detection import rectangle_pose
        s=optical_station();t=target(x_mm=15,y_mm=-12,expected_yaw_deg=37)
        # Oblique camera: the image angle is not the robot angle.
        tf=np.array(s['extrinsics']['base_from_camera']);tf[:3,:3]=Rotation.from_euler('xyz',[166,12,24],degrees=True).as_matrix()
        s['extrinsics']['base_from_camera']=tf.tolist()
        world=rectangle_world((210,25,30),37)
        p=rectangle_pose(project(world,s),t,s,[1280,720])
        self.assertAlmostEqual(p['x_mm'],210,places=4);self.assertAlmostEqual(p['yaw_deg'],37,places=4)
        xyz,rot=v.target_pose(t,{v.reference_key(t):p})
        expected=Rotation.from_euler('z',37,degrees=True).apply([15,-12,0])+[210,25,30]
        np.testing.assert_allclose(xyz,expected,atol=1e-4)
        np.testing.assert_allclose(rot[:,2],[0,0,-1],atol=1e-6)

    def test_resolution_revision_and_wrong_jig_size_blocked(self):
        from jig_detection import rectangle_pose
        s=optical_station();pixels=project(rectangle_world(),s)
        with self.assertRaises(ValueError):rectangle_pose(pixels,target(),s,[640,480])
        with self.assertRaises(ValueError):rectangle_pose(pixels,target(long_mm=300),s,[1280,720])
        s['intrinsics']['D'][0]+=.01
        with self.assertRaises(ValueError):rectangle_pose(pixels,target(),s,[1280,720])

    def test_extrinsic_calibration_recovers_measured_board(self):
        s=optical_station();origin=[120,-60,0];yaw=20
        points=v.board_points(s['board'])@Rotation.from_euler('z',yaw,degrees=True).as_matrix().T+origin
        result=v.calibrate_extrinsics(project(points,s).reshape(-1,1,2),[1280,720],s,origin,yaw,row_sign=1)
        np.testing.assert_allclose(result['base_from_camera'],s['extrinsics']['base_from_camera'],atol=1e-4)
        with self.assertRaisesRegex(ValueError,'아래'):
            v.calibrate_extrinsics(project(points,s).reshape(-1,1,2),[1280,720],s,origin,yaw,row_sign=-1)

    def test_intrinsics_from_diverse_synthetic_views(self):
        s=optical_station();obj=v.board_points(s['board']);samples=[]
        for i in range(15):
            r=Rotation.from_euler('xyz',[(i%5-2)*10,(i%3-1)*15,i*7],degrees=True).as_rotvec()
            p,_=cv2.projectPoints(obj,r,np.array([-80+(i%5)*20.,-60.,500+i*15]),np.array(s['intrinsics']['K']),np.array(s['intrinsics']['D']))
            samples.append(p.astype(np.float32))
        result=v.calibrate_intrinsics(samples,[1280,720],s['board'],s['camera'])
        self.assertLess(result['rms_px'],.001)
        np.testing.assert_allclose(result['K'],s['intrinsics']['K'],atol=.01)
        with self.assertRaises(ValueError):v.calibrate_intrinsics(samples[:2],[1280,720],s['board'],s['camera'])


class StabilityTests(unittest.TestCase):
    def test_wraparound_stale_and_distinct_frames(self):
        tracker=v.StableJigs()
        for i in range(31):tracker.observe('j',{'x_mm':200.,'y_mm':0.,'yaw_deg':179.8 if i%2 else -179.8},i*.1,i)
        self.assertAlmostEqual(abs(tracker.snapshot(['j'],now=3.01)['j']['yaw_deg']),180.,delta=.21)
        with self.assertRaises(ValueError):tracker.snapshot(['j'],now=4.)
        tracker.clear()
        for i in range(8):tracker.observe('j',{'x_mm':200.,'y_mm':0.,'yaw_deg':0},i*.06,1)
        with self.assertRaises(ValueError):tracker.snapshot(['j'],now=.5)

    def test_motion_resets_stability(self):
        tracker=v.StableJigs()
        for i in range(8):tracker.observe('j',{'x_mm':200.,'y_mm':0.,'yaw_deg':0},i*.06,i)
        tracker.observe('j',{'x_mm':210.,'y_mm':0.,'yaw_deg':0},.5,9)
        with self.assertRaises(ValueError):tracker.snapshot(['j'],now=.51)


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.station=v.default_station(teaching.Demo.identity)
        self.t=target()
        self.plan={'targets':{'slot':self.t},'calibration_signature':v.calibration_signature(self.station)}
        self.seed=dict.fromkeys(teaching.KEYS,0.)
        self.obs={v.reference_key(self.t):{'x_mm':200,'y_mm':0,'yaw_deg':0}}
        self.steps=v.make_steps('slot',self.t,self.seed)

    def test_pick_path_holds_xy_orientation_and_independent_grip(self):
        compiled=v.compile_steps(self.steps,self.plan,self.station,self.obs,self.seed)
        k=v.Kinematics()
        for step in compiled:
            for node in step['path']:
                tf=k.fk(node['pose'])
                np.testing.assert_allclose(tf[:2,3],[200,0],atol=.5)
                np.testing.assert_allclose(tf[:3,2],[0,0,-1],atol=.02)
        self.assertEqual(compiled[1]['pose']['gripper.pos'],80)
        self.assertEqual(compiled[2]['pose']['gripper.pos'],20)
        self.assertEqual(compiled[3]['pose']['gripper.pos'],20)
        self.assertEqual({k:val for k,val in compiled[1]['pose'].items() if k!='gripper.pos'},
                         {k:val for k,val in compiled[2]['pose'].items() if k!='gripper.pos'})
        self.assertGreater(len(compiled[1]['path']),5)
        doc={'schema':2,'identity':teaching.Demo.identity,'units':teaching.UNITS,'steps':compiled,'vision_plan':self.plan}
        self.assertEqual(teaching.validate_program(doc,teaching.Demo.identity),compiled)
        from episode_preview import EpisodePlayer
        player=EpisodePlayer();player.load(doc)
        self.assertGreater(player.duration,4)
        self.assertGreater(player.step_time(3),player.step_time(1))

    def test_bad_order_unreachable_and_changed_calibration_rejected(self):
        with self.assertRaises(ValueError):v.compile_steps(self.steps[::-1],self.plan,self.station,self.obs,self.seed)
        self.assertEqual(len(v.compile_steps(self.steps,self.plan,self.station,self.obs,self.seed,live=True)),len(self.steps))
        other=deepcopy(self.station);other['tool']['xyz_mm'][0]=5
        with self.assertRaises(ValueError):v.compile_steps(self.steps,self.plan,other,self.obs,self.seed)
        self.plan['targets']['slot']['x_mm']=1000
        with self.assertRaises(ValueError):v.compile_steps(self.steps,self.plan,self.station,self.obs,self.seed)

    def test_unreachable_error_identifies_step_and_preserves_original_plan(self):
        self.plan['targets']['slot']['x_mm']=1000
        original=deepcopy(self.steps)
        with self.assertRaises(ValueError) as caught:
            v.compile_steps(self.steps,self.plan,self.station,self.obs,self.seed)
        self.assertIn('실행 불가 스텝: '+self.steps[0]['name'],str(caught.exception))
        self.assertIn('목표 X ',str(caught.exception))
        self.assertIn('집게 방향',str(caught.exception))
        self.assertEqual(self.steps,original)

    def test_single_descent_requires_actual_hover(self):
        with self.assertRaises(ValueError):v.compile_steps([self.steps[1]],self.plan,self.station,self.obs,self.seed,single=True)
        hover=v.compile_steps([self.steps[0]],self.plan,self.station,self.obs,self.seed)[0]['pose']
        self.assertEqual(len(v.compile_steps([self.steps[1]],self.plan,self.station,self.obs,hover,single=True)),1)

    def test_joint_interpolation_is_proportional(self):
        start=dict.fromkeys(teaching.KEYS,0.)
        goal={k:(i+1)*2. for i,k in enumerate(teaching.KEYS)}
        result=v.synchronized_approach(start,goal,start,.01,1.)
        self.assertLess(np.ptp([result[k]/goal[k] for k in goal]),1e-12)

    def test_same_jig_different_geometry_and_invalid_values_rejected(self):
        self.plan['targets']['other']=target(long_mm=250)
        with self.assertRaises(ValueError):v.validate_plan(self.plan)
        for change in ({'x_mm':float('nan')},{'long_mm':float('inf')},{'hover_mm':-1},{'open':0}):
            with self.assertRaises(ValueError):target(**change)

    def test_fk_matches_mujoco(self):
        import mujoco
        from robot_preview import SCENE,pose_to_radians
        m=mujoco.MjModel.from_xml_path(str(SCENE));data=mujoco.MjData(m)
        p=dict(zip(teaching.KEYS,[20,-30,40,15,80,30]))
        for name,q in zip(teaching.JOINTS,pose_to_radians(p)):data.qpos[m.joint(name).qposadr[0]]=q
        mujoco.mj_forward(m,data)
        tf=v.Kinematics().fk(p)
        np.testing.assert_allclose(tf[:3,3],data.body('gripper_frame_link').xpos*1000,atol=.002)
        np.testing.assert_allclose(tf[:3,:3],data.body('gripper_frame_link').xmat.reshape(3,3),atol=2e-5)


class UITests(unittest.TestCase):
    def test_async_execute_rechecks_jig_and_sends_computed_path(self):
        import tkinter as tk
        with tempfile.TemporaryDirectory() as temp:
            root=tk.Tk();app=teaching.App(root,teaching.Demo(),temp)
            try:
                view=app.open_vision();view.add_steps();app.state='HOLD'
                app.latest={'sample_time':time.monotonic(),'follower':view.seed()}
                obs={'rectangle:하위 지그 220×148':{'x_mm':200,'y_mm':0,'yaw_deg':0}}
                from jig_detection import profile_signature
                selected=next(iter(app.vision_plan['targets'].values()))
                observation={**next(iter(obs.values())),'profile_signature':profile_signature(selected)}
                view.jig_reference.capture(selected,observation,view.station)
                with patch.object(view,'observations_for',return_value={v.reference_key(selected):observation}),patch('tkinter.messagebox.askokcancel',return_value=True),patch.object(app,'send') as send,patch.object(view,'process_frame'):
                    view.frame_live=True;view.frame_at=time.monotonic()
                    view.execute()
                    deadline=time.monotonic()+5
                    while app.vision_busy and time.monotonic()<deadline:root.update();time.sleep(.01)
                    self.assertEqual(send.call_count,1,view.status.get())
                    name,payload=send.call_args.args
                    self.assertEqual(name,'play');self.assertIn('vision_checked_at',payload)
                    self.assertTrue(payload['steps'][1]['path'])
                app.latest['sample_time']=time.monotonic()
                def measured_with_movement(*args,**kwargs):
                    return {v.reference_key(selected):{**observation,'x_mm':220 if kwargs.get('final') else 200}}
                with patch.object(view,'observations_for',return_value={v.reference_key(selected):observation}),patch.object(view.tracker,'assessment',return_value={'ready':True,'value':{**observation,'x_mm':220}}),patch('tkinter.messagebox.askokcancel',return_value=True),patch.object(app,'send') as send,patch.object(view,'process_frame'):
                    view.frame_live=True;view.frame_at=time.monotonic()
                    view.execute()
                    deadline=time.monotonic()+5
                    while app.vision_busy and time.monotonic()<deadline:root.update();time.sleep(.01)
                    send.assert_not_called();self.assertRegex(view.status.get(),'측정 위치|지그 이동')
            finally:
                if app.vision:app.vision.close()
                from tk_test_support import cancel_jobs
                cancel_jobs(root)
                root.destroy()

    def test_settings_scroll_and_focus_reaches_bottom_at_small_size(self):
        import tkinter as tk
        with tempfile.TemporaryDirectory() as temp:
            root=tk.Tk();app=teaching.App(root,teaching.Demo(),temp)
            try:
                view=app.open_vision();root.geometry('1000x720');root.update()
                for i,canvas in enumerate(view.scroll_canvases):
                    view.select_tab(i);root.update()
                    canvas.yview_moveto(1);root.update()
                    self.assertAlmostEqual(canvas.yview()[1],1.,places=2)
                    self.assertGreater(canvas.winfo_width(),350)
                    self.assertGreater(canvas.winfo_height(),150)  # shared navigation and STOP remain visible
                view.select_tab(2);root.update()
                canvas=view.scroll_canvases[2];canvas.yview_moveto(0)
                def descend(w):
                    for c in w.winfo_children():
                        yield c
                        yield from descend(c)
                for w in descend(view.tabs[2]):
                    if w.winfo_class()=='TButton' and 'TCP와 관절 영점 수정' in str(w.cget('text')):w.invoke();break
                root.update()
                last=[w for w in descend(view.tabs[2]) if w.winfo_class()=='TSpinbox' and w.winfo_ismapped()][-1]
                last.focus_force();root.update()
                self.assertGreaterEqual(last.winfo_rooty(),canvas.winfo_rooty())
                self.assertLessEqual(last.winfo_rooty()+last.winfo_height(),canvas.winfo_rooty()+canvas.winfo_height())
            finally:
                if app.vision:app.vision.close()
                from tk_test_support import cancel_jobs
                cancel_jobs(root)
                root.destroy()

    def test_target_save_generate_compile_roundtrip_and_camera_free_start(self):
        import tkinter as tk
        from vision_workspace import VisionWorkspace
        with tempfile.TemporaryDirectory() as temp:
            root=tk.Tk();app=teaching.App(root,teaching.Demo(),temp)
            try:
                with patch('vision_workspace.mp.get_context',side_effect=AssertionError('camera opened')):
                    view=app.open_vision()
                self.assertIsNotNone(view)
                view.add_steps();self.assertEqual(len(app.steps),4)
                view.preview_plan(offline=True)
                deadline=time.monotonic()+10
                while app.vision_busy and time.monotonic()<deadline:root.update();time.sleep(.01)
                self.assertFalse(app.vision_busy)
                self.assertTrue(all(s.get('path') for s in app.steps),view.status.get())
                path=Path(temp)/'plan.json';teaching.atomic_json(path,app.document())
                app.load_path(path);self.assertEqual(len(app.steps),4)
                self.assertEqual(app.steps[0]['motion']['phase'],'approach')
                view.vars['t_x_mm'].set('5');view.save_target()
                self.assertTrue(all('path' not in s for s in app.steps))
            finally:
                if app.vision:app.vision.close()
                from tk_test_support import cancel_jobs
                cancel_jobs(root)
                root.destroy()


class SessionTests(unittest.TestCase):
    def test_computed_cartesian_program_finishes_in_demo(self):
        backend=teaching.Demo()
        s=v.default_station(backend.identity);t=target(mode='fixed',x_mm=200,return_to_start=True)
        plan={'targets':{'s':t},'calibration_signature':v.calibration_signature(s)}
        seed=dict.fromkeys(teaching.KEYS,0.)
        compiled=v.compile_steps(v.make_steps('s',t,seed),plan,s,{},seed)
        self.assertEqual(len(compiled),7)
        # Begin at the already reached upper waypoint; no real serial ports exist.
        backend.lead=deepcopy(compiled[0]['pose'])
        backend.current=deepcopy(backend.lead);backend.target=deepcopy(backend.lead)
        with tempfile.TemporaryDirectory() as temp:
            session=teaching.Session(backend,temp,auto_sync=True);session.start()
            try:
                deadline=time.monotonic()+10
                while session.state!='HOLD' and time.monotonic()<deadline:
                    session.heartbeat=time.monotonic();time.sleep(.01)
                self.assertEqual(session.state,'HOLD')
                session.command('play',{'steps':compiled,'follower':deepcopy(backend.current),
                                       'vision_checked_at':time.monotonic()})
                started=False;deadline=time.monotonic()+20
                while time.monotonic()<deadline:
                    session.heartbeat=time.monotonic()
                    started=started or session.state=='PLAY'
                    if started and session.state=='HOLD':break
                    if session.state=='DISCONNECTED':break
                    time.sleep(.01)
                self.assertTrue(started);self.assertEqual(session.state,'HOLD')
                self.assertLess(teaching.difference(backend.current,compiled[-1]['pose']),1.)
            finally:
                session.stop.set();session.wakeup.set();session.thread.join(3)
            self.assertFalse(session.thread.is_alive())


if __name__=='__main__':unittest.main()
