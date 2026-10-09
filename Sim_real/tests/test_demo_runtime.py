"""CPU adapter checks using native USD; no Kit app, GPU, serial I/O or server."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
try:
    import torch
    from pxr import Usd,UsdGeom,Gf,PhysxSchema
    from arm_sim.demo_runtime import DemoRuntime
except ImportError:
    DemoRuntime=None


@unittest.skipIf(DemoRuntime is None,'Native USD/PhysxSchema and torch required')
class DemoRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from arm_sim.factory import load_layout
        from arm_sim.factory_demo import DemoPlan,demo_layout
        cls.plan=DemoPlan(demo_layout(load_layout()))

    def runtime(self,enabled=False):
        from arm_sim.factory import FactoryCell
        p=self.plan;stage=Usd.Stage.CreateInMemory()
        cell=FactoryCell.__new__(FactoryCell)
        cell.stage=stage;cell.cfg=p.cfg;cell.G=UsdGeom;cell.F=Gf
        cell.materials={n:None for n in ('metal','orange','finished','tray')};cell.wood=[None]*6
        cell.bind=lambda *args:None;cell.collision=lambda *args:None
        for name in p.vehicles:UsdGeom.Xform.Define(stage,'/World/'+name)
        cell.attach_forklift('/World/waffle_1')
        names=list(p.parts)
        view=SimpleNamespace(prim_paths=['/World/Parts/'+n for n in names],_device='cpu')
        def set_poses(positions,quaternions):view.positions=positions.numpy().copy()
        view.set_world_poses=set_poses;cell.view=view
        cell.snapshot=lambda:dict(zip(view.prim_paths,view.positions.tolist()))
        robots={}
        for name,k in p.ik.items():
            q=torch.tensor([k.model['home']],dtype=torch.float32)
            data=SimpleNamespace(default_joint_pos=q.clone(),joint_pos=q.clone(),body_pos_w=torch.zeros((1,1,3)))
            robot=SimpleNamespace(data=data,body_names=['gripper_frame_link' if k.so else 'end_effector_link'])
            robot.find_joints=lambda joints,preserve_order:(list(range(len(joints))),joints)
            def write(q,v,data=data,k=k):
                data.joint_pos=q.clone();data.body_pos_w[0,0]=torch.tensor(k.position(q[0].numpy()),dtype=torch.float32)
            robot.write_joint_state_to_sim=write;robot.set_joint_position_target=lambda q:None
            robots[name]=robot
        return DemoRuntime(stage,cell,robots,p,runtime_geometry=enabled)

    def test_disabled_geometry_is_not_constructed_but_target_limits_still_apply(self):
        with patch('arm_sim.demo_geometry.GeometryGuard',side_effect=AssertionError('heavy guard constructed')):
            r=self.runtime(False)
        self.assertEqual(r.geometry_status(),{'geometry_enabled':False,'geometry_checks':0,'traffic_checks':0})
        state=self.plan.sample(0);r.apply(state,reset=True)
        arm=next(iter(self.plan.relative_limits));state['joints'][arm][0]+=3.
        with self.assertRaisesRegex(RuntimeError,'relative target'):r.apply(state)

    def test_enabled_geometry_rejects_a_frame_before_positions_are_written(self):
        with patch('arm_sim.demo_geometry.GeometryGuard') as factory:
            factory.return_value.check.side_effect=RuntimeError('geometry rejection')
            r=self.runtime(True)
            with self.assertRaisesRegex(RuntimeError,'geometry rejection'):r.apply(self.plan.sample(0),reset=True)
            self.assertTrue(r.geometry_status()['geometry_enabled'])
            self.assertIsNone(r.last)

    def test_original_empty_trays_follow_world_lift_world_in_real_usd(self):
        r=self.runtime(False);p=self.plan
        times={0.,p.duration}
        for history in p.trays:
            for t,_,_ in history:times.update((max(0,t-1e-6),t,min(p.duration,t+1e-6)))
        for t in sorted(times):
            r.apply(p.sample(t),reset=True);r.verify()
        self.assertLess(r.max_tray_error,1e-5)
        self.assertEqual(r.geometry_status()['geometry_checks'],0)

    def test_switch_modes_reuses_stage_and_restores_both_initial_states(self):
        from arm_sim.factory_demo import DemoPlan
        r=self.runtime(False);original=self.plan;full=DemoPlan(original.cfg,start_mode='load')
        r.apply(original.sample(original.duration),reset=True)
        prim_count=len(list(r.stage.Traverse()))
        for plan,cargo in ((full,0),(original,9),(full,0)):
            r.select_plan(plan)
            self.assertIsNone(r.last)
            self.assertFalse(r.gate.completed or r.traffic_gate.completed)
            r.apply(plan.sample(0.),reset=True);r.verify()
            self.assertEqual(r.last['cargo']['burger_1'],cargo)
            self.assertEqual(len(list(r.stage.Traverse())),prim_count)
