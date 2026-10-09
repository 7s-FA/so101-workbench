import json
import math
import tempfile
import unittest
from pathlib import Path
from teach_io.compensation import PoseMap,CAP,RATE,stage_context
from teach_io.mapping import Mapping
from teach_io.tracking import Tracking
from test_follower import calibration

D=math.radians

class CompensationTests(unittest.TestCase):
    def setUp(self):
        self.mapping=Mapping(calibration());self.q=[0.,.3,.3,.3,0.,.5]

    def motors(self,command,velocity=0,pwm=10):
        return [dict(trajectory_ticks=t,velocity_rad_s=velocity,pwm_percent=pwm)
                for t in self.mapping.to_ticks(command)]

    def run_sag(self,enabled=True,pose_map=None,seconds=18,phase='free',context='empty',velocity=0,pwm=10):
        tracker=Tracking(self.mapping,self.q,0,enabled,pose_map);command=self.q.copy();history=[]
        for step in range(1,int(seconds/.02)+1):
            measured=command.copy()
            for i,sag in zip((1,2,3),(2.5,-1.5,.8)):measured[i]-=D(sag)
            previous=command
            command,status=tracker.update(self.q,measured,self.motors(command,velocity,pwm),step*.02,context=context,phase=phase)
            self.assertLessEqual(max(abs(v) for v in tracker.bias),CAP+1e-12)
            self.assertLessEqual(max(abs(a-b) for a,b in zip(command,previous)),RATE*.02+1e-10)
            self.assertEqual([tracker.bias[i] for i in (0,4,5)],[0,0,0])
            history.append(status)
        return tracker,history

    def test_static_sag_reduced_without_feedforward_double_count(self):
        tracker,history=self.run_sag()
        _,off=self.run_sag(enabled=False)
        self.assertLess(max(abs(history[-1]['error_deg'][i]) for i in (1,2,3)),.4)
        self.assertAlmostEqual(off[-1]['error_deg'][1],2.5)
        self.assertLessEqual(max(h['bias_deg'][1] for h in history),2.8)
        self.assertEqual(len(tracker.compensation.map.points),1)

    def test_reuse_nearby_pose_improves_early_error(self):
        learned,_=self.run_sag();pose_map=learned.compensation.map
        self.q[0]+=D(1)
        _,fresh=self.run_sag(seconds=4)
        _,reuse=self.run_sag(pose_map=pose_map,seconds=4)
        self.assertLess(abs(reuse[-1]['error_deg'][1]),abs(fresh[-1]['error_deg'][1]))

    def test_no_learning_during_contact_movement_unknown_or_high_effort(self):
        for kwargs in (dict(phase='contact'),dict(velocity=D(3)),dict(context='unknown'),dict(pwm=65),dict(enabled=False)):
            with self.subTest(kwargs=kwargs):
                tracker,_=self.run_sag(seconds=4,**kwargs)
                self.assertFalse(tracker.compensation.map.points)
                self.assertEqual(tracker.bias,[0]*6)

    def test_alignment_corrects_residual_but_does_not_train_or_use_map(self):
        learned,_=self.run_sag()
        tracker,h=self.run_sag(pose_map=learned.compensation.map,phase='alignment')
        self.assertLess(abs(h[-1]['error_deg'][1]),.4)
        self.assertFalse(h[-1]['compensation']['learning'])
        self.assertEqual(h[-1]['compensation']['map_confidence'],0)

    def test_contact_decays_cached_bias_without_learning(self):
        tracker,_=self.run_sag();count=tracker.compensation.map.points[0]['seconds']
        command=tracker.last
        for k in range(1,401):
            measured=command.copy();measured[1]-=D(2.5)
            command,status=tracker.update(self.q,measured,self.motors(command),18+k*.02,phase='contact')
        self.assertEqual(count,tracker.compensation.map.points[0]['seconds'])
        self.assertAlmostEqual(status['bias_deg'][1],0)

    def test_map_rejects_unobserved_payload_distant_pose_or_unstable_samples(self):
        p=PoseMap()
        for _ in range(60):p.observe(self.q,[D(2),0,0],'empty',.02)
        self.assertGreater(p.predict(self.q,'empty')[1],.99)
        self.assertEqual(p.predict(self.q,'holding:ball'),([0]*3,0))
        distant=self.q.copy();distant[1]+=D(20)
        self.assertEqual(p.predict(distant,'empty'),([0]*3,0))
        p.points[0]['variance'][0]=D(1)**2
        self.assertEqual(p.predict(self.q,'empty'),([0]*3,0))

    def test_map_persistence_checks_identity_and_limits(self):
        p=PoseMap()
        for _ in range(60):p.observe(self.q,[D(2),0,0],'empty',.02)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'map.json';p.save(path,'test-fingerprint')
            other=PoseMap();other.load(path,'test-fingerprint')
            self.assertEqual(other.predict(self.q,'empty'),p.predict(self.q,'empty'))
            other.load(path,'different-calibration');self.assertFalse(other.points)
            raw=json.loads(path.read_text());raw['points'][0]['bias'][0]=D(20)
            path.write_text(json.dumps(raw));other.load(path,'test-fingerprint');self.assertFalse(other.points)
            path.write_text('[]');other.load(path,'test-fingerprint');self.assertFalse(other.points)

    def test_one_velocity_unit_without_position_change_can_learn(self):
        tracker=Tracking(self.mapping,self.q,0);command=self.q.copy()
        for k in range(1,501):
            measured=command.copy();measured[1]-=D(2)
            motors=self.motors(command)
            motors[3]['velocity_rad_s']=D(1.374)
            motors[5]['velocity_rad_s']=D(1.374)
            command,status=tracker.update(self.q,measured,motors,k*.02)
        self.assertTrue(tracker.compensation.map.points)
        self.assertLess(abs(status['error_deg'][1]),.5)

    def test_actual_motion_blocks_learning_even_with_zero_reported_speed(self):
        tracker=Tracking(self.mapping,self.q,0)
        for k in range(1,251):
            measured=self.q.copy();measured[3]+=D(2)*math.sin(k*.02*2)
            tracker.update(self.q,measured,self.motors(self.q),k*.02)
        self.assertFalse(tracker.compensation.map.points)
        self.assertEqual(tracker.bias,[0]*6)

    def test_phase_mapping_is_explicit_and_contact_conservative(self):
        for kind in ('housing','ball','cap'):
            self.assertEqual(stage_context(dict(kind=kind,label=kind+': transfer')),('holding:'+kind,'free'))
            for action in ('descend','close','lift','lower','release'):
                self.assertEqual(stage_context(dict(kind=kind,label=kind+': '+action))[1],'contact')
            self.assertEqual(stage_context(dict(kind=kind,label=kind+': unknown')),('unknown','unknown'))
        self.assertEqual(stage_context(dict(kind=None,label='Park')),('empty','free'))

if __name__=='__main__':unittest.main()
