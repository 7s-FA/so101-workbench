import importlib.util
import unittest
from pathlib import Path
from arm_sim.common import load_models
spec=importlib.util.spec_from_file_location('leader',Path(__file__).resolve().parents[1]/'scripts/leader.py')
leader=importlib.util.module_from_spec(spec)
spec.loader.exec_module(leader)

class Mapping(unittest.TestCase):
    def setUp(self):
        self.model=load_models()['omx']
        self.cfg=dict(model='omx',neutral_ticks=[4090]*5+[200],home=self.model['home'],signs=[1]*5,gripper_closed_ticks=200,gripper_open_ticks=900)
        self.mapper=leader.Mapper(self.cfg,self.model)
    def test_wrap_is_small_continuous_motion(self):
        q=self.mapper.convert([5]*5+[200])
        self.assertAlmostEqual(q[0],11*2*3.141592653589793/4096)
    def test_fingers_follow_opposite_angles(self):
        q=self.mapper.convert([4090]*5+[900])
        self.assertEqual(q[-2:],[1.1,-1.1])
    def test_reversed_trigger_range(self):
        self.cfg['gripper_closed_ticks']=900; self.cfg['gripper_open_ticks']=200
        m=leader.Mapper(self.cfg,self.model)
        self.assertEqual(m.convert([4090]*5+[200])[-2:],[1.1,-1.1])
    def test_one_joint_at_limit_does_not_freeze_other_joints(self):
        model=load_models()['so101']
        cfg=dict(model='so101',neutral_ticks=[2000]*6,home=[0.]*6,signs=[1]*5,gripper_closed_ticks=1900,gripper_open_ticks=2400)
        mapper=leader.Mapper(cfg,model)
        q=mapper.convert([2000,700,2100,2000,2000,2100])
        self.assertEqual(q[1],model['limits'][1][0])
        self.assertGreater(q[2],0)
        self.assertEqual(mapper.limited,['shoulder_lift'])
        q2=mapper.convert([2000,710,2200,2000,2000,2100])
        self.assertEqual(q2[1],q[1]);self.assertGreater(q2[2],q[2])
        q3=mapper.convert([2000,1800,2200,2000,2000,2100])
        self.assertGreater(q3[1],q[1]);self.assertEqual(mapper.limited,[])
