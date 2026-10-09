import unittest
import numpy as np
from arm_sim.factory import load_layout
from arm_sim.factory_demo import DemoPlan, demo_layout, KINDS, ASSEMBLED_Z
from arm_sim.demo_validation import route_check, rectangle, overlaps


class FactoryDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=DemoPlan(demo_layout(load_layout()))
        from scripts.audit_motion_quality import audit
        cls.motion_audit=audit(cls.plan)

    def test_all_eighteen_parts_reach_warehouse_as_six_complete_units(self):
        p=self.plan;final=p.sample(p.duration)
        self.assertEqual(final['stored'],6)
        self.assertEqual(set(final['parts']),{f'{kind}_{i}' for kind in KINDS for i in range(1,7)})
        for i in range(1,7):
            housing=final['parts'][f'housing_{i}']
            self.assertGreater(housing[2],p.cfg['warehouse']['levels'][0 if i<=3 else 1])
            self.assertLess(abs(housing[0]-p.cfg['warehouse']['center'][0]),.11)
            for kind in KINDS:
                np.testing.assert_allclose(final['parts'][f'{kind}_{i}']-housing,[0,0,ASSEMBLED_Z[kind]],atol=1e-6)

    def test_both_batches_complete_all_stages_without_respawning(self):
        for name,history in self.plan.parts.items():
            owners=[entry[1] for entry in history]
            vehicle='burger_1' if int(name.rsplit('_',1)[1])<=3 else 'burger_2'
            for required in (vehicle,'arm_b','arm_c','arm_d','lift'):
                self.assertIn(required,owners,name)
            if vehicle=='burger_1':
                self.assertEqual(owners[0],vehicle)
                self.assertNotIn('arm_a',owners,name)
            else:self.assertIn('arm_a',owners,name)
            self.assertEqual(owners[-1],'world')
        self.assertEqual(self.plan.sample(self.plan.duration)['cargo'],{'burger_1':0,'burger_2':0})
        self.assertEqual(len(self.plan.assembly_times),6)
        self.assertEqual(len(self.plan.storage_times),2)

    def test_cargo_departure_requires_nine_parts_and_a_safe(self):
        p=self.plan;t=p.milestones['first_loaded'];s=p.sample(t)
        self.assertEqual(s['cargo']['burger_1'],9)
        np.testing.assert_allclose(s['joints']['arm_a'][:5],p.ik['arm_a'].model['home'][:5])
        for t in np.linspace(0,t,100):
            np.testing.assert_allclose(p.vehicles['burger_1'].at(t),p.cfg['burger']['poses'][0])

    def test_dependency_zone_exclusion_and_safe_returns(self):
        p=self.plan;tasks={t['id']:t for t in p.tasks}
        for task in p.tasks:
            for dep in task['deps']:self.assertLessEqual(tasks[dep]['end'],task['start']+1e-8)
            np.testing.assert_allclose(p.arms[task['arm']].at(task['end'])[:5],task['safe_joints'][:5],atol=1e-7)
        for i,a in enumerate(p.tasks):
            for b in p.tasks[i+1:]:
                if set(a['zones'])&set(b['zones']):
                    self.assertLessEqual(min(a['end'],b['end'])-max(a['start'],b['start']),1e-8)
        for i in range(6):
            self.assertGreaterEqual(tasks[f'c_assemble_kit_{i}']['start'],p.milestones['kits_ready'][i])

    def test_a_remains_concurrent_while_b_waits_empty_during_c(self):
        p=self.plan;a=p.milestones['a_second_window']
        for windows in (p.milestones['b_windows'],p.milestones['c_windows']):
            self.assertGreater(max(min(a[1],w[1])-max(a[0],w[0]) for w in windows),1.)
        for start,end in p.milestones['c_windows']:
            for t in np.linspace(start,end,51):
                self.assertFalse(any(p.entry(h,t)[1]=='arm_b' for h in p.parts.values()))
                np.testing.assert_allclose(p.arms['arm_b'].at(t)[:5],p.ik['arm_b'].model['home'][:5],atol=1e-7)

    def test_feedback_gate_holds_when_safe_ack_is_missing(self):
        from arm_sim.demo_coordination import WorkcellGate
        p=self.plan;gate=WorkcellGate(p);task=min(p.tasks,key=lambda t:t['end']);t=task['end']
        measured={n:tr.at(t) for n,tr in p.arms.items()};measured[task['arm']][0]+=.1
        self.assertEqual(gate.advance(t,t+.1,measured),t)
        self.assertNotIn(task['id'],gate.completed)
        measured[task['arm']]=np.asarray(task['safe_joints'])
        self.assertGreater(gate.advance(t,t+.1,measured),t)
        self.assertIn(task['id'],gate.completed)

    def test_b_rotation_and_nominal_velocity_are_bounded(self):
        p=self.plan;track=p.arms['arm_b'];deltas=np.diff(track.values,axis=0);dt=np.diff(track.times)
        self.assertLess(float(np.max(abs(deltas[:,:5]))),np.pi)
        self.assertLessEqual(self.motion_audit['arms']['arm_b']['velocity_limit_ratio'],1.00001)

    def test_preloaded_nominal_cycle_finishes_within_ten_minutes(self):
        self.assertLessEqual(self.plan.duration,600.)

    def test_loaded_and_empty_profiles_follow_payload_ownership(self):
        p=self.plan
        for name,track in p.arms.items():
            for span in {id(s):s for s in track.spans.values()}.values():
                held=any(p.entry(h,span.start+span.duration/2)[1]==name for h in p.parts.values())
                if span.phase!='precision':self.assertEqual(span.phase,'loaded' if held else 'empty')
                self.assertTrue(np.all(span.velocity_limits[5:]<=p.motion_profiles['gripper']['velocity_rad_s']))

    def test_b_transfer_does_not_repeatedly_flex_at_arc_chords(self):
        for span in {id(s):s for s in self.plan.arms['arm_b'].spans.values()}.values():
            if span.phase=='precision':continue
            v=span.spline(np.linspace(0,span.duration,1000)[1:-1],1)[:,2]
            v=v[abs(v)>.002]
            reversals=np.count_nonzero(np.sign(v[1:])!=np.sign(v[:-1]))
            self.assertLessEqual(reversals,8)

    def test_payload_continuity_at_grab_release_and_pallet_delivery(self):
        p=self.plan
        for name,history in p.parts.items():
            for t,_,_ in history[1:]:
                jump=np.linalg.norm(p.part_at(name,t)-p.part_at(name,t-1e-7))
                self.assertLess(jump,.003,name)

    def test_reset_restores_sources_vehicles_and_joints(self):
        p=self.plan;p.sample(p.duration)
        start=p.sample(0.)
        self.assertEqual(start['assembled'],0);self.assertEqual(start['stored'],0)
        for name,q in start['joints'].items():np.testing.assert_allclose(q,p.ik[name].model['home'])
        self.assertEqual(start['cargo'],{'burger_1':9,'burger_2':0})
        self.assertEqual(sum(o=='world' for o in start['owners'].values()),9)
        for n,h in p.parts.items():
            expected=p.mobile_point('burger_1',h[0][2],0.) if h[0][1]=='burger_1' else h[0][2]
            np.testing.assert_allclose(start['parts'][n],expected)

    def test_preloaded_batch_leaves_without_waiting_for_a(self):
        p=self.plan;job=p.traffic_jobs[0]
        self.assertEqual(job['vehicle'],'burger_1');self.assertLessEqual(job['start'],1.)
        self.assertLess(p.milestones['first_arrived'],45.)
        self.assertEqual(len(p.parts),18)
        for kind in KINDS:
            self.assertEqual(sum(n.startswith(kind+'_') and o=='burger_1' for n,o in p.sample(0.)['owners'].items()),3)
        for t in np.linspace(0,p.milestones['first_arrived'],20):
            np.testing.assert_allclose(p.arms['arm_a'].at(t)[:5],p.ik['arm_a'].model['home'][:5])

    def test_preflight_routes_and_joint_limits(self):
        self.assertEqual(route_check(self.plan)['result'],'clear')
        self.assertLess(max(self.plan.validate()['max_ik_error_m'].values()),.003)

    def test_route_checker_detects_colliding_layout(self):
        original=self.plan.cfg['tables'][0]['center'][:]
        try:
            self.plan.cfg['tables'][0]['center']=[-.9,.195]
            with self.assertRaisesRegex(ValueError,'route overlaps'):route_check(self.plan)
        finally:self.plan.cfg['tables'][0]['center']=original

    def test_footprints_touching_is_not_penetration(self):
        self.assertFalse(overlaps(rectangle(0,0,1,1),rectangle(2,0,1,1)))
        self.assertTrue(overlaps(rectangle(0,0,1,1),rectangle(1.5,0,1,1)))

    def test_demo_layout_does_not_mutate_preview(self):
        preview=load_layout();old=preview['arms']['arm_a']['position'][:]
        demo=demo_layout(preview)
        self.assertEqual(preview['arms']['arm_a']['position'],old)
        self.assertNotEqual(demo['arms']['arm_a']['position'],old)

    def test_compact_shared_table_corner_mounts_and_low_shelves(self):
        from arm_sim.demo_geometry import clamp_shapes
        cfg=self.plan.cfg
        self.assertEqual(cfg['map']['size'],[3.,2.])
        table=next(t for t in cfg['tables'] if t['id']=='assembly')
        self.assertNotIn('output',[t['id'] for t in cfg['tables']])
        low=np.asarray(table['center'])-np.asarray(table['size'])/2
        high=np.asarray(table['center'])+np.asarray(table['size'])/2
        for name,side in [('arm_b',0),('arm_d',1)]:
            plate=clamp_shapes(cfg)[name][0]
            self.assertTrue(np.all(plate[4][:2]>=low-1e-8))
            self.assertTrue(np.all(plate[5][:2]<=high+1e-8))
            if side==0:self.assertAlmostEqual(plate[4][0],low[0])
            else:self.assertAlmostEqual(plate[5][0],high[0])
            self.assertLess(abs(plate[4][1]-low[1]),.011)
            tip=self.plan.ik[name].position(self.plan.arms[name].at(0))
            self.assertGreater(tip[2],cfg['table_height'])
        self.assertAlmostEqual(cfg['warehouse']['levels'][1],cfg['table_height'])
        self.assertAlmostEqual(cfg['warehouse']['yaw'],0.)
        self.assertEqual(cfg['warehouse']['entry_side'],'bottom')
        self.assertLess(abs(cfg['warehouse']['center'][0]),.3)
        self.assertGreater(cfg['warehouse']['center'][1],.5)
        self.assertLessEqual(high[0],cfg['map']['size'][0]/2-.04)
        self.assertEqual(cfg['clamps']['arm_d']['edge'],'right')
        self.assertAlmostEqual(cfg['clamps']['arm_d']['edge_value'],high[0])
        self.assertLess(table['size'][0],.85)
        self.assertGreater(cfg['waffle']['fork_top']-.018,0.)
        self.assertLessEqual(cfg['waffle']['fork_top']-.018,.011)

    def test_preparation_targets_match_container_slots(self):
        p=self.plan;c=p.cfg['preparation']
        for kind in KINDS:
            for i in range(1,4):
                positions=[entry[2] for entry in p.parts[f'{kind}_{i}'] if entry[1]=='world']
                target=np.array([c['columns'][kind],c['center_y']+((i-1)%2-.5)*c['pitch'],p.cfg['table_height']+.005])
                self.assertTrue(any(np.linalg.norm(pos-target)<1e-7 for pos in positions))

    def test_forks_withdraw_through_shelf_channels_without_body_penetration(self):
        import copy
        from arm_sim.demo_geometry import forklift_shapes,moved_shapes,fixture_shapes,intersections
        p=self.plan;pose=p.tray[-1][2][:3];height=p.tray[-1][2][3]
        fixed,moving=forklift_shapes(p.cfg)
        shapes=moved_shapes(fixed,pose)+moved_shapes(moving,pose,height)
        rack=[s for s in fixture_shapes(p.cfg) if s[0].startswith('warehouse/')]
        self.assertEqual(intersections(shapes,rack,margin=-.00001),[])
        solid=copy.deepcopy(p.cfg);solid['warehouse'].pop('fork_slots')
        shelves=[s for s in fixture_shapes(solid) if s[0]=='warehouse/shelf']
        self.assertTrue(intersections(shapes,shelves,margin=-.00001))

    def test_full_process_retiming_preserves_measured_target_limits(self):
        p=self.plan;t=0.;retimed=0
        measured={name:track.at(0) for name,track in p.arms.items()}
        for _ in range(20000):
            next_t=p.advance(t,16/15,measured)
            if next_t<min(p.duration,t+16/15)-1e-7:retimed+=1
            self.assertGreater(next_t,t)
            command={name:track.at(next_t) for name,track in p.arms.items()}
            for name,limits in p.relative_limits.items():
                self.assertTrue(np.all(np.abs(command[name]-measured[name])<=limits))
            measured=command;t=next_t
            if t>=p.duration:break
        self.assertEqual(t,p.duration)
        self.assertGreater(retimed,0)
        self.assertEqual(p.sample(t)['stored'],6)

    def test_retiming_rejects_invalid_feedback(self):
        with self.assertRaises(ValueError):self.plan.advance(0,.1,{})

    def test_process_metadata_is_json_serializable(self):
        import json
        s=self.plan.sample(220.)
        json.dumps({k:s[k] for k in ('time','coordination','assembled','stored','cargo','progress','label','traffic','goal')},allow_nan=False)

    def test_first_ready_kit_starts_c_without_waiting_for_another_kit(self):
        p=self.plan;task=next(t for t in p.tasks if t['id']=='c_assemble_kit_0')
        self.assertAlmostEqual(task['start'],p.milestones['kits_ready'][0])
        by_id={t['id']:t for t in p.tasks}
        self.assertLessEqual(by_id['c_assemble_kit_0']['end'],by_id['b_unload_2']['start'])

    def test_b_keeps_zone_and_only_returns_after_three_parts(self):
        p=self.plan;tasks=[t for t in p.tasks if t['arm']=='arm_b']
        self.assertEqual(len(tasks),6)
        for kit,task in enumerate(sorted(tasks,key=lambda t:t['start'])):
            grabs=[e for e in p.grasp_events if e['arm']=='arm_b' and task['start']<=e['time']<task['end']]
            self.assertEqual([e['parts'][0] for e in grabs],[f'{kind}_{kit+1}' for kind in KINDS])
            first=grabs[0]['time'];last=grabs[-1]['time']
            for time,q in zip(p.arms['arm_b'].times,p.arms['arm_b'].values):
                if first<=time<=last:
                    self.assertFalse(np.allclose(q[:5],p.ik['arm_b'].model['home'][:5],atol=1e-5))
                    self.assertFalse(np.allclose(q[:5],p.ready['arm_b'][:5],atol=1e-5))
            self.assertEqual(p.sample((first+last)/2)['coordination']['zones']['bc_transfer'],'arm_b')
            np.testing.assert_allclose(p.arms['arm_b'].at(task['end'])[:5],p.ik['arm_b'].model['home'][:5],atol=1e-7)

    def test_c_finishes_assembly_and_handoff_before_returning(self):
        p=self.plan;tasks=[t for t in p.tasks if t['arm']=='arm_c']
        self.assertEqual(len(tasks),6)
        for task in tasks:
            grabs=[e for e in p.grasp_events if e['arm']=='arm_c' and task['start']<=e['time']<task['end']]
            self.assertEqual([e['profile'] for e in grabs],['housing','ball','cap','product'])
            for time,q in zip(p.arms['arm_c'].times,p.arms['arm_c'].values):
                if grabs[0]['time']<=time<=grabs[-1]['time']:
                    self.assertFalse(np.allclose(q[:5],p.ik['arm_c'].model['home'][:5],atol=1e-5))
                    self.assertFalse(np.allclose(q[:5],p.ready['arm_c'][:5],atol=1e-5))
            np.testing.assert_allclose(p.arms['arm_c'].at(task['end'])[:5],p.ik['arm_c'].model['home'][:5],atol=1e-7)

    def test_d_loads_directly_and_overlaps_b_in_separate_zones(self):
        from arm_sim.demo_geometry import fixture_shapes
        p=self.plan
        self.assertFalse(p.cfg['finished']['enabled'])
        self.assertFalse(any(s[0].startswith('finished') for s in fixture_shapes(p.cfg)))
        b=[t for t in p.tasks if t['arm']=='arm_b'];d=[t for t in p.tasks if t['arm']=='arm_d']
        self.assertEqual(len(d),6)
        self.assertGreater(max(min(x['end'],y['end'])-max(x['start'],y['start']) for x in b for y in d),1.)
        for kind in KINDS:
            for kit in range(1,7):
                owners=[entry[1] for entry in p.parts[f'{kind}_{kit}']]
                self.assertEqual(owners.count('arm_d'),1)
                self.assertEqual(owners[owners.index('arm_d')+1],'lift')

    def test_d_unfolds_toward_pickup_without_visiting_forklift_ready(self):
        p=self.plan;track=p.arms['arm_d'];home=np.asarray(p.ik['arm_d'].model['home'])
        first=next(i for i,q in enumerate(track.values) if not np.allclose(q[:5],home[:5],atol=1e-6))
        target=p.ik['arm_d'].position(track.values[first])
        grab=next(e for e in p.grasp_events if e['arm']=='arm_d')
        grasp_tcp=p.ik['arm_d'].position(track.at(grab['time']))
        np.testing.assert_allclose(target,grasp_tcp+[0,0,.035],atol=.0001)
        self.assertFalse(np.allclose(track.values[first][:5],p.ready['arm_d'][:5]))
        self.assertLessEqual(np.max(abs(np.asarray(track.values)[:,4])),np.pi+.001)

    def test_two_shipping_pallets_are_continuous_and_first_delivery_stays_put(self):
        p=self.plan
        def world(entry,time):
            from arm_sim.demo_geometry import moved_shapes,cargo_shapes
            pose=p.vehicles['waffle_1'].at(time) if entry[1]=='lift' else entry[2][:3]
            height=float(p.lift.at(time)[0]) if entry[1]=='lift' else entry[2][3]
            return moved_shapes(cargo_shapes(p.cfg)[1],pose,height)[0][2]
        for history in p.trays:
            for time,_,_ in history[1:]:
                np.testing.assert_allclose(world(p.entry(history,time),time),world(p.entry(history,time-1e-7),time-1e-7),atol=1e-6)
        first=p.storage_times[0]
        for name in [f'{k}_{i}' for k in KINDS for i in range(1,4)]:
            np.testing.assert_allclose(p.part_at(name,first),p.part_at(name,p.duration),atol=1e-7)
        initial_floor=p.cfg['waffle']['fork_top']+.001+p.trays[1][0][2][3]
        self.assertAlmostEqual(initial_floor,p.cfg['warehouse']['levels'][1])
        self.assertAlmostEqual(p.cfg['waffle']['fork_top']+.001+p.trays[0][-1][2][3],p.cfg['warehouse']['levels'][0])
        self.assertAlmostEqual(p.cfg['waffle']['fork_top']+.001+p.trays[1][-1][2][3],p.cfg['warehouse']['levels'][1])
        self.assertEqual([e[1] for e in p.trays[1]],['world','lift','world'])
        self.assertGreater(p.storage_times[1],p.milestones['empty_tray_retrieved'])

    def test_both_empty_trays_start_in_rack_and_forklift_waits_for_material_arrival(self):
        p=self.plan;arrived=p.milestones['first_arrived']
        self.assertAlmostEqual(p.milestones['first_empty_pickup_start'],arrived)
        for i,h in enumerate(p.trays):
            self.assertEqual([e[1] for e in h],['world','lift','world'])
            self.assertAlmostEqual(h[0][2][0],p.cfg['warehouse']['center'][0])
            self.assertAlmostEqual(p.cfg['waffle']['fork_top']+.001+h[0][2][3],p.cfg['warehouse']['levels'][i])
        for t in np.linspace(0,arrived,50):
            np.testing.assert_allclose(p.vehicles['waffle_1'].at(t),p.cfg['waffle']['pose'])
            self.assertEqual(float(p.lift.at(t)[0]),0.)
        first_d=next(t for t in p.tasks if t['id']=='d_load_0')
        self.assertGreaterEqual(first_d['start'],p.milestones['first_tray_ready'])
        first_b=next(t for t in p.tasks if t['id']=='b_unload_0')
        self.assertLess(first_b['start'],p.milestones['first_tray_ready'])

    def test_shared_lane_is_reserved_until_vehicle_finishes(self):
        from arm_sim.traffic import TrafficGate
        p=self.plan;TrafficGate(p)
        for a,b in zip(p.traffic_jobs,p.traffic_jobs[1:]):
            self.assertLessEqual(a['end'],b['start'])
        for job in p.traffic_jobs:
            self.assertGreater(job['end'],job['start'])
            np.testing.assert_allclose(p.vehicles[job['vehicle']].at(job['end']),job['end_pose'])
        state=p.sample((p.traffic_jobs[0]['start']+p.traffic_jobs[0]['end'])/2)
        self.assertEqual(state['traffic']['owner'],'burger_1')

    def test_all_arm_profiles_limit_nominal_acceleration(self):
        for name,track in self.plan.arms.items():
            self.assertLessEqual(self.motion_audit['arms'][name]['acceleration_limit_ratio'],1.0001,name)
        self.assertEqual(self.motion_audit['status'],'clear')

    def test_empty_wait_has_less_gravity_load_than_the_old_extended_home(self):
        from arm_sim.common import load_models
        from arm_sim.static_load import StaticLoad
        models=load_models()
        for name,k in self.plan.ik.items():
            load=StaticLoad(k);old=models[self.plan.cfg['arms'][name]['model']]['home']
            self.assertLess(np.max(abs(load.torques(k.model['home']))),.6*np.max(abs(load.torques(old))),name)
