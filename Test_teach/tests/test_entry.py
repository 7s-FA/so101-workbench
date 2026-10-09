import math
import ast
from pathlib import Path
from types import SimpleNamespace,ModuleType
import unittest
from unittest.mock import Mock,patch
from teach_io.entry import CurrentPoseEntry, entry_samples, validate_entry, ENTRY_SPEED, EntryPoseChanged
from teach_io.mapping import Mapping
from test_follower import calibration


class EntryTests(unittest.TestCase):
    def test_large_entry_is_slow_synchronized_and_has_no_initial_jump(self):
        start=[0.,-.8,.8,.8,0.,.2];goal=[0.,-.3,-.5,1.,0.,.6]
        motion=CurrentPoseEntry(start,goal,0);previous=start;peak=0.
        for i in range(1,1600):
            q=motion.advance(goal,i*.02)
            speed=max(abs(a-b)/.02 for a,b in zip(q,previous));peak=max(peak,speed)
            self.assertLessEqual(speed,ENTRY_SPEED+1e-9)
            fractions=[(q[j]-start[j])/(goal[j]-start[j]) for j in (1,2,3,5)]
            self.assertLess(max(fractions)-min(fractions),1e-9)
            if i==1:self.assertLess(speed,.001)
            previous=q
            if motion.elapsed>=motion.duration:break
        self.assertGreater(peak,.099)
        for actual,expected in zip(q,goal):self.assertAlmostEqual(actual,expected)
        self.assertFalse(motion.reached(goal))
        self.assertTrue(motion.reached(goal,stationary=True,effort=10))
        self.assertLess(speed,.001)

    def test_entry_pauses_on_lag_and_never_catches_up_after_stall(self):
        start=[0.]*6;goal=[.5]*6;motion=CurrentPoseEntry(start,goal,0)
        first=motion.advance(goal,.02)
        self.assertEqual(first,motion.advance(goal,.04,pace=0.))
        with self.assertRaisesRegex(ValueError,'Stale'):motion.advance(goal,.35)

    def test_checked_path_and_fresh_pose_required_before_large_entry(self):
        mapping=Mapping(calibration());start=[0.]*5+[.2];goal=[.6]*5+[.3]
        plan=dict(start=start,goal=goal,collision_checked=True)
        self.assertEqual(validate_entry(plan,start,goal,mapping),start)
        moved=start.copy();moved[0]+=.02
        with self.assertRaisesRegex(ValueError,'moved'):validate_entry(plan,moved,goal,mapping)
        with self.assertRaisesRegex(ValueError,'collision-checked'):
            validate_entry(dict(plan,collision_checked=False),start,goal,mapping)
        with self.assertRaisesRegex(ValueError,'goal differs'):
            validate_entry(plan,start,[0]*6,mapping)
        outside=[0.]*5+[1.5]
        with self.assertRaises(ValueError):
            validate_entry(dict(start=start,goal=outside,collision_checked=True),start,outside,mapping)

    def test_audit_samples_include_endpoints_at_half_degree_resolution(self):
        start=[0.]*6;end=[.1,.5,-1.,.3,0.,.2]
        samples=entry_samples(start,end)
        self.assertEqual(samples[0],start);self.assertEqual(samples[-1],end)
        for a,b in zip(samples,samples[1:]):
            self.assertLessEqual(max(abs(x-y) for x,y in zip(a,b)),math.radians(.5)+1e-12)

    def test_entry_blocks_early_trial_and_unsettled_completion(self):
        motion=CurrentPoseEntry([0.]*6,[.4]*6,0)
        with self.assertRaisesRegex(ValueError,'Wait'):motion.advance([.5]*6,.02)
        self.assertFalse(motion.reached([.4]*6,stationary=True,effort=0))

    def test_gui_passes_checked_large_entry_and_blocks_collision_before_arm(self):
        # Exercise the actual callback without launching Isaac or accessing USB.
        tree=ast.parse((Path(__file__).resolve().parents[1]/'scripts/sim.py').read_text())
        callback=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='enable_follower')
        class Home(list):
            def __getitem__(self,key):
                value=super().__getitem__(key)
                return Home(value) if isinstance(key,slice) else value
            def tolist(self):return list(self)
        for clear,retries in ((True,0),(True,1),(True,3),(False,0)):
            follower=Mock();follower.armed=False
            follower.arm.side_effect=[EntryPoseChanged('moved')]*retries+[None]
            measured=[0.,-.8,.8,.8,0.,.2]
            follower.request.return_value={'angles':measured,'release_required':False}
            entry=dict(start=measured,goal=[0.]*6,collision_checked=clear,pairs=[['link3','table']])
            audit=Mock();audit.check.return_value=entry
            module=ModuleType('teach_io.entry_collision');module.EntryCollisionAudit=Mock()
            namespace=dict(follower=follower,exit_release_port=None,entry_audit=audit,writes_allowed=True,notify_launcher=Mock(),
                observe_panel=None,hardware_state={'fault':False,'pose_dirty':False},
                state=dict(reset=False,done=False,playing=False),control_mode={'real':True},
                home=Home([0.]*6),control_profile=Mock(return_value=(None,None,Mapping(calibration()))),
                args=SimpleNamespace(calibration=None,visual_confirmed=True),report={},save=Mock(),port='/dev/fake',
                hardware_path=[[0.]*6],compensation_model=SimpleNamespace(as_bool=True),
                reuse_map_model=SimpleNamespace(as_bool=False),evaluation_model=SimpleNamespace(as_bool=True),
                compensation_checkbox=Mock(),reuse_map_checkbox=Mock(),evaluation_checkbox=Mock(),support_model=Mock())
            with patch.dict('sys.modules',{'teach_io.entry_collision':module}):
                exec(compile(ast.Module(body=[callback],type_ignores=[]),'actual_enable_callback','exec'),namespace)
                namespace['enable_follower']()
            if clear:
                self.assertEqual(follower.arm.call_count,min(retries+1,3))
                self.assertEqual(follower.arm.call_args.kwargs['entry'],entry)
                if retries==3:
                    self.assertIn('press Enable again',namespace['hardware_state']['text'])
                    self.assertFalse(namespace['hardware_state']['fault'])
            else:
                follower.arm.assert_not_called()
                self.assertIn('BLOCKED',namespace['hardware_state']['text'])

    def test_hold_stops_automatic_entry_but_keeps_learning_at_ready_pose(self):
        tree=ast.parse((Path(__file__).resolve().parents[1]/'scripts/sim.py').read_text())
        callback=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='hold_action')
        for aligned in (True,False):
            stop=Mock();namespace=dict(state={'playing':True},stop_follower=stop,
                follower=SimpleNamespace(armed=True,latest={'aligned':aligned}))
            exec(compile(ast.Module(body=[callback],type_ignores=[]),'actual_hold_callback','exec'),namespace)
            namespace['hold_action']()
            self.assertFalse(namespace['state']['playing'])
            self.assertEqual(stop.call_count,0 if aligned else 1)


if __name__=='__main__':unittest.main()
