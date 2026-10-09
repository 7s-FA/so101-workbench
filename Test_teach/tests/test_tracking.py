import math
import unittest
from teach_io.tracking import Tracking
from teach_io.mapping import Mapping,HARD_LIMITS
from teach_io.worker import StartAlignment
from test_follower import calibration

D=math.radians

class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.mapping=Mapping(calibration())
        self.q=[0.,.3,.3,.3,0.,.5]

    def motors(self,command,pwm=10,velocity=0):
        return [dict(trajectory_ticks=t,velocity_rad_s=velocity,pwm_percent=pwm)
                for t in self.mapping.to_ticks(command)]

    def test_three_point_five_degree_lag_slows_without_freezing(self):
        measured=self.q.copy();measured[1]-=D(3.5)
        tracker=Tracking(self.mapping,self.q,0)
        _,s=tracker.update(self.q,measured,self.motors(self.q),.02)
        self.assertGreater(s['pace'],0);self.assertLess(s['pace'],1)
        self.assertEqual(s['state'],'slowing');self.assertEqual(s['worst_id'],12)

    def test_evaluation_keeps_speed_for_small_error_and_preserves_recovery(self):
        tracker=Tracking(self.mapping,self.q,0,evaluation=True)
        measured=self.q.copy();measured[1]-=D(5.5)
        _,s=tracker.update(self.q,measured,self.motors(self.q),.02)
        self.assertEqual(s['pace'],1.)
        self.assertEqual(s['profile'],'evaluation')
        measured[1]=self.q[1]-D(7)
        _,s=tracker.update(self.q,measured,self.motors(self.q),.04)
        self.assertEqual(s['pace'],0.)
        measured[1]=self.q[1]-D(2.9)
        _,s=tracker.update(self.q,measured,self.motors(self.q),.06)
        self.assertEqual(s['pace'],1.)
        measured[1]=self.q[1]-D(11)
        with self.assertRaisesRegex(ValueError,'ID 12'):
            tracker.update(self.q,measured,self.motors(self.q),.08)

    def test_evaluation_does_not_ignore_high_effort_or_stale_feedback(self):
        tracker=Tracking(self.mapping,self.q,0,evaluation=True)
        with self.assertRaisesRegex(ValueError,'output effort'):
            for i in range(1,100):tracker.update(self.q,self.q,self.motors(self.q,pwm=95),i*.02)
        tracker=Tracking(self.mapping,self.q,0,evaluation=True)
        with self.assertRaisesRegex(ValueError,'Stale'):
            tracker.update(self.q,self.q,self.motors(self.q),.31)

    def test_evaluation_recovers_from_observed_four_degree_static_error(self):
        tracker=Tracking(self.mapping,self.q,0,evaluation=True)
        measured=self.q.copy();measured[1]-=D(6.1)
        _,status=tracker.update(self.q,measured,self.motors(self.q,pwm=58),.02,phase='contact')
        self.assertEqual(status['pace'],0)
        measured[1]=self.q[1]-D(4.1122507519148295)
        for i in range(2,502):
            _,status=tracker.update(self.q,measured,self.motors(self.q,pwm=25.99),i*.02,phase='contact')
            self.assertEqual(status['pace'],1.)
        self.assertEqual(status['bias_deg'],[0]*6)

    def test_evaluation_recovery_still_waits_above_five_or_high_output(self):
        tracker=Tracking(self.mapping,self.q,0,evaluation=True)
        for i,(error,pwm) in enumerate([(6.1,20),(5.1,20),(4.1,75),(5,69)],1):
            measured=self.q.copy();measured[1]-=D(error)
            _,status=tracker.update(self.q,measured,self.motors(self.q,pwm=pwm),i*.02)
            self.assertEqual(status['pace'],1 if i==4 else 0)

    def test_motion_metrics_only_include_explicit_active_trial_samples(self):
        tracker=Tracking(self.mapping,self.q,0,evaluation=True)
        measured=self.q.copy();measured[1]-=D(2)
        tracker.update(self.q,measured,self.motors(self.q),.02)
        self.assertEqual(tracker.error_summary()['samples'],0)
        tracker.record_motion_error()
        measured[1]=self.q[1]+D(4)
        tracker.update(self.q,measured,self.motors(self.q),.04);tracker.record_motion_error()
        summary=tracker.error_summary()
        self.assertEqual(summary['samples'],2)
        self.assertAlmostEqual(summary['mean_absolute_deg'][1],3)
        self.assertAlmostEqual(summary['max_absolute_deg'][1],4)

    def test_recovery_hysteresis_and_persistent_stall(self):
        tracker=Tracking(self.mapping,self.q,0);measured=self.q.copy();measured[2]-=D(7)
        _,s=tracker.update(self.q,measured,self.motors(self.q),.02)
        self.assertEqual(s['pace'],0)
        measured[2]=self.q[2]-D(4)
        _,s=tracker.update(self.q,measured,self.motors(self.q),.04)
        self.assertEqual(s['pace'],0)
        measured[2]=self.q[2]-D(2.9)
        _,s=tracker.update(self.q,measured,self.motors(self.q),.06)
        self.assertGreater(s['pace'],0)
        measured[2]=self.q[2]-D(7)
        with self.assertRaisesRegex(ValueError,'no tracking recovery'):
            for i in range(4,400):tracker.update(self.q,measured,self.motors(self.q),i*.02)

    def test_static_sag_model_is_reduced_without_gain_changes(self):
        tracker=Tracking(self.mapping,self.q,0);command=self.q.copy()
        for i in range(1,801):
            measured=command.copy();measured[1]-=D(1.5)
            previous=command;command,s=tracker.update(self.q,measured,self.motors(command),i*.02)
            self.assertLessEqual(max(abs(a-b) for a,b in zip(command,previous)),.35*.02+1e-9)
        self.assertLess(abs(s['error_deg'][1]),.4)
        self.assertLessEqual(max(abs(v) for v in s['bias_deg']),2.)
        self.assertEqual(s['bias_deg'][5],0)

    def test_profile_lag_is_not_mistaken_for_static_sag(self):
        tracker=Tracking(self.mapping,self.q,0)
        measured=self.q.copy();measured[1]-=D(3.5)
        trajectory=self.q.copy();trajectory[1]-=D(2)
        for i in range(1,201):
            _,s=tracker.update(self.q,measured,self.motors(trajectory),i*.02)
        self.assertEqual(s['bias_deg'],[0]*6)

    def test_missing_telemetry_does_not_enable_correction(self):
        tracker=Tracking(self.mapping,self.q,0);measured=self.q.copy();measured[1]-=D(3)
        for i in range(1,101):_,s=tracker.update(self.q,measured,[{}]*6,i*.02)
        self.assertEqual(s['bias_deg'],[0]*6)

    def test_effort_and_large_lag_protection_remain(self):
        tracker=Tracking(self.mapping,self.q,0);measured=self.q.copy();measured[1]-=D(11)
        with self.assertRaisesRegex(ValueError,'ID 12'):tracker.update(self.q,measured,self.motors(self.q),.02)
        tracker=Tracking(self.mapping,self.q,0)
        with self.assertRaisesRegex(ValueError,'output effort'):
            for i in range(1,100):tracker.update(self.q,self.q,self.motors(self.q,pwm=95),i*.02)
        self.assertEqual(tracker.bias,[0]*6)

    def test_correction_never_crosses_joint_limit(self):
        cfg=calibration();cfg['limits']=[list(v) for v in HARD_LIMITS]
        m=Mapping(cfg);q=self.q.copy();q[3]=D(99.9)
        tracker=Tracking(m,q,0);command=q.copy()
        for i in range(1,401):
            measured=command.copy();measured[3]-=D(1.5)
            motors=[dict(trajectory_ticks=t,velocity_rad_s=0,pwm_percent=10) for t in m.to_ticks(command)]
            command,s=tracker.update(q,measured,motors,i*.02)
            m.to_ticks(command)
            self.assertLessEqual(command[3],D(100))

    def test_start_alignment_can_settle_despite_small_static_sag(self):
        home=self.q.copy();measured=home.copy();measured[1]-=D(3)
        alignment=StartAlignment(measured,home,0)
        tracker=Tracking(self.mapping,measured,0);command=measured.copy();reached=False
        for i in range(1,1000):
            now=i*.02
            measured=command.copy();measured[1]-=D(1.5)
            requested=alignment.advance(home,now)
            command,status=tracker.update(requested,measured,self.motors(command),now,
                correction=max(abs(a-b) for a,b in zip(requested,home))<1e-6,phase='alignment')
            if alignment.reached(measured,stationary=status['compensation']['stationary'],
                    effort=status['pwm_percent']):reached=True;break
        self.assertTrue(reached)
        self.assertLess(now,20)

    def test_start_alignment_timeout_does_not_allow_endless_loading(self):
        alignment=StartAlignment(self.q,self.q,0)
        with self.assertRaisesRegex(ValueError,'20 s'):alignment.advance(self.q,20.1)
