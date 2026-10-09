import pickle,unittest
import numpy as np
from arm_sim.common import ROOT
from scripts.real_workflow import Workflow

class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=pickle.loads((ROOT/'outputs/sim_real_demo/plan.pkl').read_bytes());cls.flow=Workflow(cls.plan)
    def snapshot(self,t,**overrides):
        args=dict(started=t>0,paused=False,done=t>=self.plan.duration,
                  arm_acks={s['id'] for s in self.plan.tasks if s['end']<=t},
                  traffic_acks={s['id'] for s in self.plan.traffic_jobs if s['end']<=t})
        args.update(overrides)
        return self.flow.snapshot(t,self.plan.sample(t),**args)
    def test_initial_inventory_and_all_seven_devices_are_visible(self):
        flow=self.snapshot(0)
        self.assertEqual(flow['complete'],0)
        self.assertEqual({s['device'] for s in flow['steps']},set(self.plan.arms)|set(self.plan.vehicles))
        self.assertEqual(flow['vehicles'][0]['cargo'],'재료 9 / 9개')
        self.assertEqual([s['location'] for s in flow['trays']],['창고 1','창고 2'])
        self.assertEqual([s['start'] for s in flow['steps']],sorted(s['start'] for s in flow['steps']))
    def test_forklift_returns_to_starting_bay_after_storage(self):
        for t in (0,self.plan.duration):
            fork=next(v for v in self.snapshot(t)['vehicles'] if v['id']=='waffle_1')
            self.assertEqual(fork['location'],'지게차 전용 대기 위치')
            self.assertEqual(fork['cargo'],'용기 없음')
        last=self.flow.steps[-1]
        self.assertIn('전용 대기 위치 복귀',last['label'])
        self.assertGreater(last['end'],max(self.plan.storage_times))
    def test_both_carriers_finish_centred_in_their_parking_marks(self):
        end=self.snapshot(self.plan.duration)
        for index,name in enumerate(('burger_1','burger_2')):
            pose=self.plan.vehicles[name].at(self.plan.duration)
            target=self.plan.cfg['burger_parking_poses'][name]
            self.assertTrue(np.allclose(pose[:2],target[:2]))
            self.assertLess(abs(np.sin((pose[2]-target[2])/2)),1e-8)
            deck=self.plan.mobile_point(name,[-.032,0,0],self.plan.duration)[:2]
            self.assertTrue(np.allclose(deck,self.plan.cfg['traffic']['waiting_bays'][index]['center']))
            vehicle=next(v for v in end['vehicles'] if v['id']==name)
            self.assertNotEqual(vehicle['location'],'재료 하차 위치')
        job=next(j for j in self.plan.traffic_jobs if j['vehicle']=='burger_2' and '대기 위치 복귀' in j['label'])
        state=self.snapshot(job['end']+1e-6,traffic_acks=set())
        self.assertEqual(next(s for s in state['steps'] if s['id']==job['id'])['status'],'checking')
    def test_reverse_trip_shows_left_heading_empty_tray_and_actual_action(self):
        event=next(e for e in self.plan.vehicle_moves if e['vehicle']=='waffle_1' and self.plan.vehicles['waffle_1'].at(e['end'])[0]-self.plan.vehicles['waffle_1'].at(e['start'])[0]>.1)
        flow=self.snapshot((event['start']+event['end'])/2)
        fork=next(s for s in flow['vehicles'] if s['id']=='waffle_1')
        self.assertIn('후진',fork['phase']);self.assertEqual(fork['heading'],'왼쪽');self.assertIn('빈 용기',fork['cargo'])
        self.assertTrue(any(a['kind']=='reverse_transit' and a['status']=='active' for s in flow['steps'] for a in s['children']))
    def test_completion_requires_arrival_and_arm_acknowledgements(self):
        end=self.snapshot(self.plan.duration,arm_acks=set(),traffic_acks=set())
        self.assertTrue(any(s['status']=='checking' for s in end['steps']))
        end=self.snapshot(self.plan.duration)
        self.assertEqual(end['complete'],end['total'])
        self.assertEqual([s['products'] for s in end['trays']],[3,3])
        self.assertTrue(all(v['phase']=='작업 완료 · 정지' for v in end['vehicles']))
    def test_paused_steps_and_lift_actions_are_reported(self):
        action=next(a for a in self.flow.actions if a['kind']=='lift')
        flow=self.snapshot((action['start']+action['end'])/2,paused=True)
        self.assertTrue(any(s['status']=='paused' for s in flow['steps']))
        self.assertIn('일시정지',flow['vehicles'][-1]['phase'])
        self.assertIn('포크 올리기',flow['vehicles'][-1]['phase'])
        self.assertTrue(any(a['kind']=='tray' for s in flow['steps'] for a in s['children']))
