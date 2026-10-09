"""Replay the measured no-response micro-jog without touching serial hardware."""
import json
from pathlib import Path
import queue
import tempfile
import time
import unittest
from unittest.mock import patch

import manual_teaching as m
import teach_left_arm as t


class QuietMotor(t.Demo):
    """Encoder does not respond to a goal (deadband/stiction or blocked motion)."""
    follower_only=True
    def read(self):return None,self.current.copy()


class AccumulationTests(unittest.TestCase):
    def test_small_repeats_accumulate_but_reversal_drift_expiry_and_axis_change_reset(self):
        p=dict.fromkeys(t.KEYS,0.)
        ref={'axis':'shoulder_pan.pos','requested':.5,'measured':0.,'direction':1,'at':10.}
        self.assertEqual(m.jog_goal(p,None,'shoulder_pan.pos',.5,ref,10.1)['shoulder_pan.pos'],1.)
        self.assertEqual(m.jog_goal(p,None,'shoulder_pan.pos',-.5,ref,10.1)['shoulder_pan.pos'],-.5)
        self.assertEqual(m.jog_goal(p,None,'shoulder_pan.pos',.5,ref,21.)['shoulder_pan.pos'],.5)
        shifted={**p,'shoulder_pan.pos':2.}
        self.assertEqual(m.jog_goal(shifted,None,'shoulder_pan.pos',.5,ref,10.1)['shoulder_pan.pos'],2.5)
        self.assertEqual(m.jog_goal(p,None,'wrist_roll.pos',.5,ref,10.1)['wrist_roll.pos'],.5)
        ref['requested']=5.
        goal=m.jog_goal(p,None,'shoulder_pan.pos',.5,ref,10.1)
        with self.assertRaisesRegex(ValueError,'관절 변화'):
            m.checked_jog({'start':p,'goal':goal,'sample_time':10.},p,10.1,t.pose)


    def test_untouched_joints_keep_hold_commands_despite_measured_load_error(self):
        p=dict.fromkeys(t.KEYS,0.);p['elbow_flex.pos']=1.2
        held={**p,'elbow_flex.pos':.9}
        goal=m.jog_goal(p,None,'shoulder_pan.pos',.5,hold_target=held)
        self.assertEqual(goal['elbow_flex.pos'],.9)
        self.assertEqual(goal['shoulder_pan.pos'],.5)
        result=m.jog_hold_pose({'axis':'shoulder_pan.pos'},goal,p,t.pose)
        self.assertEqual(result['elbow_flex.pos'],.9)
        self.assertEqual(result['shoulder_pan.pos'],0.)
        legacy=m.jog_hold_pose({},goal,p,lambda q:{**q,'elbow_flex.pos':min(1.,q['elbow_flex.pos'])})
        self.assertEqual(legacy['elbow_flex.pos'],1.)


class JogRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory();self.backend=QuietMotor()
        self.session=t.Session(self.backend,self.folder.name,auto_sync=True)
        self.events=[];self.last=None
        self.timeout=patch.object(m,'JOG_TIMEOUT_SECONDS',.25);self.timeout.start()
        self.session.start();self.wait(lambda:self.last and self.last['state']=='READY')
        self.session.command('hold_current');self.wait(lambda:self.last and self.last['state']=='HOLD')

    def tearDown(self):
        self.session.stop.set();self.session.wakeup.set();self.session.thread.join(3)
        self.timeout.stop();self.folder.cleanup()

    def wait(self,predicate,timeout=4):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            self.session.heartbeat=time.monotonic()
            try:
                while True:
                    kind,value=self.session.events.get_nowait();self.events.append((kind,value))
                    if kind=='status':self.last=value
            except queue.Empty:pass
            if predicate():return
            time.sleep(.005)
        self.fail(str((self.session.state,self.events[-5:])))

    def jog(self,token,delta):
        snapshot=self.last.copy();goal=m.jog_goal(snapshot['follower'],None,'shoulder_pan.pos',delta,snapshot.get('jog_reference'),hold_target=snapshot.get('hold_target'))
        self.session.command('jog',{'token':token,'start':snapshot['follower'],'goal':goal,
                                    'sample_time':snapshot['sample_time'],'axis':'shoulder_pan.pos','increment':delta})
        self.wait(lambda:any(k=='jog_result' and e['token']==token for k,e in self.events))
        self.wait(lambda:self.last['state']=='HOLD')
        return next(e for k,e in self.events if k=='jog_result' and e['token']==token)

    def test_no_response_returns_to_hold_without_claiming_arrival_and_next_jog_accumulates(self):
        first=self.jog('first',.5)
        self.assertFalse(first['arrived']);self.assertEqual(first['pose']['shoulder_pan.pos'],0)
        self.assertEqual(first['goal']['shoulder_pan.pos'],.5)
        self.assertEqual(self.backend.target,self.backend.current)  # unmet command no longer pursued
        second=self.jog('second',.5)
        self.assertEqual(second['goal']['shoulder_pan.pos'],1.)
        self.assertTrue(self.session.thread.is_alive())
        self.assertFalse(any(k in ('closed','error') for k,_ in self.events))
        reverse=self.jog('reverse',-.5)
        self.assertEqual(reverse['goal']['shoulder_pan.pos'],-.5)
        self.session.command('pause')
        self.wait(lambda:self.last['jog_reference'] is None)

    def test_unmet_jog_projects_slight_boundary_drift_to_legal_hold_without_relaxing_limits(self):
        old_send=self.backend.send
        def bounded_send(goal):
            if goal['elbow_flex.pos']>1.:raise ValueError('Outward motion blocked')
            if goal['shoulder_pan.pos']>.1:self.backend.current['elbow_flex.pos']=1.2
            return old_send(goal)
        def project(goal):return {**goal,'elbow_flex.pos':min(1.,goal['elbow_flex.pos'])}
        with patch.object(self.backend,'send',side_effect=bounded_send),patch.object(self.backend,'project_target',side_effect=project):
            result=self.jog('boundary',.5)
            self.assertFalse(result['arrived'])
            self.assertEqual(self.backend.target['elbow_flex.pos'],0.)
            self.assertEqual(self.session.state,'HOLD')
            self.assertFalse(any(k in ('closed','error') for k,_ in self.events))
            # Exercise the held command after the boundary recovery as well.
            until=time.monotonic()+.1;self.wait(lambda:time.monotonic()>until)

    def test_read_or_health_fault_still_disconnects_instead_of_becoming_hold(self):
        with patch.object(self.backend,'read',side_effect=ConnectionError('test packet loss')):
            self.wait(lambda:any(k=='closed' for k,_ in self.events))
        self.assertTrue(any(k=='error' and 'test packet loss' in e for k,e in self.events))

    def test_ordinary_move_still_uses_its_timeout_and_does_not_get_micro_jog_recovery(self):
        # Error 6° exceeds ordinary move arrival tolerance; no response is simulated.
        with patch.object(t,'MOVE_TIMEOUT',.6):
            self.session.command('move',{**self.backend.current,'shoulder_pan.pos':6.})
            self.wait(lambda:any(k=='closed' for k,_ in self.events))
        self.assertTrue(any(k=='error' and 'MOVE timeout' in e for k,e in self.events))
        self.assertFalse(any(k=='jog_result' for k,_ in self.events))


if __name__=='__main__':unittest.main()
