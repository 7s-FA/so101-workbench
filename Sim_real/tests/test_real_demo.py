import json,pickle,unittest
from types import SimpleNamespace
from threading import Lock
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import numpy as np
from arm_sim.common import ROOT
from arm_sim.real_monitor import RealMonitor
from arm_sim.demo_coordination import WorkcellGate
from arm_sim.traffic import TrafficGate

class RealDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.plan=pickle.loads((ROOT/'outputs/sim_real_demo/plan.pkl').read_bytes())
    def test_initial_inventory_and_final_product_accounting(self):
        p=self.plan;s=p.sample(0);end=p.sample(p.duration)
        self.assertEqual(s['cargo']['burger_1'],9);self.assertEqual(s['cargo']['burger_2'],0)
        self.assertEqual(sum(owner=='world' for owner in s['owners'].values()),9)
        self.assertEqual(end['stored'],6);self.assertEqual(end['assembled'],6)
        self.assertTrue(all(o=='world' for o in end['owners'].values()))
        with self.assertRaisesRegex(ValueError,'stroke'):p.raise_lift(.101)
    def test_vehicles_stop_before_turning_and_reverse_to_aisle(self):
        p=self.plan
        for event in p.vehicle_moves:
            tr=p.vehicles[event['vehicle']]
            q=np.array([tr.at(t) for t in np.linspace(event['start'],event['end'],31)])
            if event['mode']=='turn':
                self.assertLess(np.max(np.linalg.norm(q[:,:2]-q[0,:2],axis=1)),1e-8)
                self.assertLessEqual(abs(q[-1,2]-q[0,2]),np.pi/2+1e-7)
            else:
                self.assertLess(np.ptp(q[:,2]),1e-8,'Moving turns are forbidden')
                if event['vehicle']=='waffle_1' and abs(q[-1,0]-q[0,0])>1e-8:
                    self.assertTrue(np.allclose(np.cos(q[:,2]),-1.,atol=1e-8),'Forklift must face left throughout the cross aisle')
        first=[e for e in p.vehicle_moves if e['vehicle']=='burger_1']
        turn=next(e for e in first if e['mode']=='turn')
        self.assertAlmostEqual(p.vehicles['burger_1'].at(turn['start'])[1],-.42)
    def test_route_revision_preserves_map_and_workcell_geometry(self):
        before=pickle.loads((ROOT/'outputs/sim_real_demo/motion_revision/before_inplace.pkl').read_bytes())
        for key in ('map','arms','tables','warehouses','shipping_dock','shipping_vehicle_pose','shipping_height','shipping_support_size','finished_container','preparation_slots','handoff','jig','loading_dock','unload_dock','source_banks'):
            self.assertEqual(self.plan.cfg[key],before.cfg[key],key)
        self.assertEqual(self.plan.cfg['waffle'],before.cfg['waffle'])
        self.assertEqual(self.plan.cfg['burger'],before.cfg['burger'])
    def test_shared_zone_cannot_advance_without_pose_feedback(self):
        p=self.plan;gate=WorkcellGate(p);t=min(task['end'] for task in p.tasks)
        self.assertEqual(gate.advance(t,t+.1,{}),t);self.assertTrue(gate.hold_reason)
        measured={n:tr.at(t) for n,tr in p.arms.items()}
        self.assertGreater(gate.advance(t,t+.1,measured),t);self.assertEqual(gate.acknowledgements,1)
        gate=TrafficGate(p);job=min(p.traffic_jobs,key=lambda j:j['end']);t=job['end']
        self.assertEqual(gate.advance(t,t+.1,{}),t)
        self.assertGreater(gate.advance(t,t+.1,{job['vehicle']:np.array(job['end_pose'])}),t)
    def test_simulation_only_http_contract(self):
        stream=SimpleNamespace(lock=Lock(),names=[],frames={},metadata={},commands=[]);monitor=RealMonitor(stream,port=0)
        base='http://127.0.0.1:'+str(monitor.server.server_address[1])
        try:
            for endpoint in ['/diagnostics','/api/voltage']:
                with self.assertRaises(HTTPError) as e:urlopen(base+endpoint)
                self.assertEqual(e.exception.code,404)
            for payload in [{'command':'start_mode','mode':'load'},{'command':'start_mode','mode':'preloaded'},{'command':'speed','speed':17},{'command':'motor_on'}]:
                with self.assertRaises(HTTPError) as e:urlopen(Request(base+'/api/demo/control',data=json.dumps(payload).encode()))
                self.assertEqual(e.exception.code,400)
            self.assertEqual(stream.commands,[])
            for command in ['resume','pause','restart']:
                with urlopen(Request(base+'/api/demo/control',data=json.dumps({'command':command}).encode())) as r:self.assertEqual(r.status,200)
            self.assertEqual(stream.commands,['resume','pause','restart'])
        finally:monitor.close()
