import sys
from copy import copy
from pathlib import Path
import time
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import Reference, JOINTS, JointPath, ROOT, read_json
from teaching_executor import TeachingExecutor, estimated_load_raw


class TeachingExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.original=Reference()

    def make_executor(self):
        ref=copy(self.original)
        start=ref.steps[0]['ticks'].copy()
        close={**start,'gripper':start['gripper']-200}
        lift={**close,'shoulder_pan':start['shoulder_pan']+100}
        opened={**lift,'gripper':start['gripper']+20}
        ref.steps=[dict(name=n,ticks=t) for n,t in zip(('잡기','들기','열기'),(close,lift,opened))]
        ref.joint_path=JointPath(start,[s['ticks'] for s in ref.steps],rate=ref.rate)
        ref.ends=np.cumsum([s.duration for s in ref.joint_path.segments]);ref.duration=float(ref.ends[-1])
        return TeachingExecutor(ref,ref.mapping.angles(start)),start

    def test_uses_current_worker_and_arm2_contact_setting(self):
        before=time.monotonic
        executor,_=self.make_executor()
        self.assertEqual(executor.session.grip_contact.threshold,self.original.profile['grip_contact_load'])
        self.assertEqual(Path(executor.session.on_snapshot.__code__.co_filename).resolve(),
                         Path('SO101_teach/so101_teach/motion.py').resolve())
        self.assertIs(time.monotonic,before)
        self.assertIsNone(executor.session.thread)
        self.assertFalse(executor.session.requires_port_check)

    def test_contact_holds_position_before_lift_and_releases_for_open(self):
        executor,start=self.make_executor();actual=start.copy();barrier=start['gripper']-50
        seen=False
        for _ in range(650):
            goal=executor.session.last_goals
            actual={**goal,'gripper':max(barrier,goal['gripper'])}
            raw=[0]*6;raw[5]=1024+80 if goal['gripper']<barrier else 0
            executor.advance(.02,executor.ref.mapping.angles(actual),np.zeros(6),raw)
            if executor.hold_tick is not None:
                seen=True;self.assertEqual(executor.hold_tick,barrier)
                self.assertGreaterEqual(executor.session.last_goals['gripper'],barrier)
            if executor.index>=1:self.assertTrue(seen)
            if executor.finished or executor.failure:break
        self.assertFalse(executor.failure)
        self.assertTrue(executor.finished)
        self.assertIsNone(executor.hold_tick)
        event=executor.contact_events[0]
        self.assertGreaterEqual(event['confirmed_seconds'],.12)
        self.assertGreaterEqual(event['confirmed_samples'],3)
        self.assertLessEqual(event['position_spread_ticks'],3)
        self.assertEqual(event['threshold'],70)
        self.assertEqual(executor.arrivals[0]['target']['gripper'],barrier)

    def test_below_threshold_stall_cannot_lift(self):
        executor,start=self.make_executor();actual=start.copy()
        for _ in range(450):
            goal=executor.session.last_goals
            actual={**goal,'gripper':max(start['gripper']-50,goal['gripper'])}
            executor.advance(.02,executor.ref.mapping.angles(actual),np.zeros(6),[0]*5+[69])
            self.assertEqual(executor.index,0)
            if executor.failure:break
        self.assertTrue(executor.failure)
        self.assertFalse(executor.contact_events)

    def test_moving_load_spike_does_not_confirm_contact(self):
        executor,start=self.make_executor();actual=start.copy()
        for i in range(18):
            actual['gripper']=start['gripper']-i*6
            executor.advance(.02,executor.ref.mapping.angles(actual),np.zeros(6),[0]*5+[80])
        self.assertFalse(executor.contact_events)
        self.assertEqual(executor.index,0)

    def test_contact_does_not_bypass_body_arrival(self):
        executor,start=self.make_executor()
        for _ in range(450):
            actual={**executor.session.last_goals,'shoulder_pan':start['shoulder_pan']+40}
            actual['gripper']=max(start['gripper']-50,actual['gripper'])
            executor.advance(.02,executor.ref.mapping.angles(actual),np.zeros(6),[0]*5+[80])
            self.assertEqual(executor.index,0)
            if executor.failure:break
        self.assertTrue(executor.contact_events)
        self.assertTrue(executor.failure)

    def test_no_contact_requires_original_arrival_dwell(self):
        executor,_=self.make_executor()
        for _ in range(400):
            executor.advance(.02,executor.nominal,np.zeros(6),[0]*6)
            if executor.finished or executor.failure:break
        self.assertTrue(executor.finished)
        self.assertFalse(executor.contact_events)
        self.assertTrue(all(a['arrival_type']=='settled' for a in executor.arrivals))
        self.assertEqual(executor.arrivals[0]['tolerance_ticks']['gripper'],30)
        self.assertEqual(executor.arrivals[0]['tolerance_ticks']['shoulder_pan'],20)

    def test_effort_proxy_is_explicit_bounded_and_sign_magnitude(self):
        scale=np.ones(6)
        raw=estimated_load_raw([0,1,-1,2,-2,.5],scale)
        np.testing.assert_array_equal(raw,[0,1023,2047,1023,2047,512])
        with self.assertRaises(ValueError):estimated_load_raw([np.nan]*6,scale)

    def test_recorded_gripper_load_straddles_current_threshold(self):
        model=read_json(ROOT/'config/gripper_load_model.json')
        self.assertEqual(model['calibration_sha256'],self.original.calibration.sha256)
        estimate=lambda error:estimated_load_raw([0]*6,[1]*6,gripper_error_ticks=error,gripper_model=model)[5]
        self.assertEqual(estimate(14),68)
        self.assertEqual(estimate(15),80)
        self.assertLess(estimate(14),self.original.profile['grip_contact_load'])
        self.assertGreaterEqual(estimate(15),self.original.profile['grip_contact_load'])
        self.assertEqual(estimate(400),model['load_magnitude'][-1])
        self.assertEqual(estimate(-20),0)

    def test_approach_deflection_does_not_fabricate_closing_load(self):
        model=read_json(ROOT/'config/gripper_load_model.json')
        raw=estimated_load_raw([0]*5+[-.15],[1]*6,gripper_error_ticks=110,gripper_model=model,gripper_closing=False)
        self.assertEqual(raw[5],0)
        executor=TeachingExecutor(self.original,self.original.sample(0)[0])
        self.assertFalse(any(executor.closing_steps[:4]))
        self.assertTrue(executor.closing_steps[4])
        self.assertFalse(executor.closing_steps[5])
        self.assertFalse(executor.closing_steps[8])

    def test_complete_b_preserves_all_open_and_close_targets(self):
        executor=TeachingExecutor(self.original,self.original.sample(0)[0])
        for _ in range(8000):
            executor.advance(.02,executor.nominal,np.zeros(6),[0]*6)
            if executor.finished or executor.failure:break
        self.assertFalse(executor.failure)
        self.assertTrue(executor.finished)
        self.assertEqual(len(executor.arrivals),len(self.original.steps))
        self.assertEqual([a['target']['gripper'] for a in executor.arrivals],
                         [s['ticks']['gripper'] for s in self.original.document['steps']])
        self.assertFalse(executor.contact_events)

    def test_recorded_arm_load_uses_evidence_range_and_falls_back_outside(self):
        model=read_json(ROOT/'config/arm_load_model.json')
        self.assertEqual(model['calibration_sha256'],self.original.calibration.sha256)
        errors=[0,22,16,7,4,0]
        raw=estimated_load_raw([1]*6,[1]*6,joint_error_ticks=errors,arm_model=model,joint_velocity_ticks=[0]*6)
        self.assertEqual(raw[1],112)
        self.assertTrue(all(v<409 for v in raw[:5]))
        raw=estimated_load_raw([-1]*6,[1]*6,joint_error_ticks=[100]*6,arm_model=model,joint_velocity_ticks=[0]*6)
        np.testing.assert_array_equal(raw,[2047]*6)
        raw=estimated_load_raw([1]*6,[1]*6,joint_error_ticks=errors,arm_model=model,joint_velocity_ticks=[20]*6)
        np.testing.assert_array_equal(raw,[1023]*6)

    def test_slow_feedback_velocity_is_not_quantized_to_ten_tick_jumps(self):
        executor,_=self.make_executor();angles=executor.nominal
        values=executor.tick_values(angles)
        later=[np.radians(executor.ref.calibration.angle_mapping.degrees(n,values[n]+.4))+executor.ref.mapping.radians[n] for n in JOINTS]
        speed=executor.velocity_ticks(angles,(np.asarray(later)-angles)/.1)
        self.assertEqual(list(speed.values()),[4]*6)


if __name__=='__main__':unittest.main()
