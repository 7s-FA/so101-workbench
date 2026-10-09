"""Geometry constraints that matter before connecting a motion controller."""
import json
import tempfile
import unittest
from pathlib import Path
from arm_sim.factory import load_layout, rows
from arm_sim.workcell import vertices


class FactoryLayout(unittest.TestCase):
    def test_corner_clamps_have_clearance_from_table_legs(self):
        from arm_sim.factory_demo import demo_layout
        from arm_sim.demo_geometry import clamp_shapes,fixture_shapes,intersections
        cfg=demo_layout(load_layout())
        mounts=[s for n,v in clamp_shapes(cfg).items() for s in v]
        legs=[s for s in fixture_shapes(cfg) if s[0].startswith('table_leg/')]
        self.assertEqual(intersections(mounts,legs,margin=.005),[])
        # Default corner legs obstruct a lower jaw even with side margins.
        next(t for t in cfg['tables'] if t['id']=='assembly').pop('leg_inset')
        legs=[s for s in fixture_shapes(cfg) if s[0].startswith('table_leg/')]
        hits=intersections(mounts,legs,margin=-.00001)
        self.assertTrue(any(a.startswith('arm_b/') for a,b in hits))

    def test_source_banks_are_square_grids_supported_by_their_tables(self):
        from arm_sim.factory_demo import demo_layout
        cfg=demo_layout(load_layout())
        for index,table_id in enumerate(('supply_left','supply_right')):
            table=next(t for t in cfg['tables'] if t['id']==table_id)
            batches=[b for b in cfg['source_batches'] if b['start_index']==index*3]
            self.assertEqual(len(batches),3)
            self.assertEqual(len({b['center_y'] for b in batches}),1)
            xs=sorted(x for b in batches for x in b['columns'].values())
            for a,b in zip(xs,xs[1:]):self.assertAlmostEqual(b-a,batches[0]['pitch'])
            for b in batches:
                self.assertEqual(b['count'],3)
                for x in b['columns'].values():
                    self.assertLessEqual(abs(x-table['center'][0])+b['carrier_width']/2,table['size'][0]/2)
                half_depth=((b['count']-1)*b['pitch']+b['carrier_width'])/2
                self.assertLessEqual(abs(b['center_y']-table['center'][1])+half_depth,table['size'][1]/2)

    def test_follower_arms_keep_original_height_without_leader_risers(self):
        from arm_sim.common import load_models
        from arm_sim.factory import arm_base_height
        from arm_sim.factory_demo import demo_layout
        from arm_sim.demo_geometry import clamp_shapes
        from arm_sim.demo_kinematics import ArmKinematics
        cfg=demo_layout(load_layout());mounts=clamp_shapes(cfg)
        for name in ('arm_a','arm_c'):
            pose=cfg['arms'][name]
            self.assertEqual(pose['mount_height'],0.)
            self.assertEqual(pose['mount_height_status'],'follower_original_height')
            k=ArmKinematics(load_models()['so101'],pose,cfg['table_height'])
            self.assertFalse(any(s[0].endswith('/Riser') for s in mounts[name]))
            self.assertAlmostEqual(k.base[2,3],cfg['table_height']+.004)
            self.assertAlmostEqual(k.base[2,3],arm_base_height(pose,cfg['table_height']))
        for name in ('arm_b','arm_d'):
            self.assertFalse(any(s[0].endswith('/Riser') for s in mounts[name]))

    def test_real_parts_fit_pitch_and_source_table(self):
        cfg=load_layout()
        table=cfg['tables'][0]
        cx,cy=table['center'];sx,sy=table['size']
        for kind,x in cfg['source']['columns'].items():
            pts=vertices(kind)
            width=max(p[1] for p in pts)-min(p[1] for p in pts)
            self.assertGreaterEqual(cfg['source']['pitch']-width,.0399)
            for y in rows(cfg['source']['center_y'],3,cfg['source']['pitch']):
                self.assertLessEqual(abs(x-cx)+.04,sx/2)
                self.assertLessEqual(abs(y-cy)+.04,sy/2)

    def test_mobile_carrier_fits_deck(self):
        cfg=load_layout()
        self.assertGreaterEqual(cfg['burger']['deck_size'][0],.08)
        self.assertGreaterEqual(cfg['burger']['deck_size'][1],2*.09+.08)

    def test_invalid_clearance_rejected(self):
        cfg=load_layout();cfg['source']['pitch']=.06
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bad.json';path.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):load_layout(path)

    def test_tables_are_separated(self):
        tables=load_layout()['tables']
        for left,right in zip(tables,tables[1:]):
            self.assertGreater(right['center'][0]-right['size'][0]/2,
                               left['center'][0]+left['size'][0]/2)

    def test_map_contains_routes_and_workbenches(self):
        cfg=load_layout();half=[v/2 for v in cfg['map']['size']]
        for table in cfg['tables']:
            for i in (0,1):
                self.assertLessEqual(abs(table['center'][i])+table['size'][i]/2,half[i])
        for route in cfg['traffic']['routes']:
            for point in route['points']:
                for i in (0,1):
                    self.assertLessEqual(abs(point[i])+cfg['traffic']['lane_width']/2,half[i])

    def test_lidar_height_matches_source_urdf(self):
        import xml.etree.ElementTree as ET
        from arm_sim.common import ROOT
        cfg=load_layout()
        for kind in ('burger','waffle'):
            tree=ET.parse(ROOT/'assets/factory/turtlebot3_description/urdf'/f'turtlebot3_{kind}.urdf')
            joints={j.get('name').removeprefix('${namespace}'):j for j in tree.findall('joint')}
            z=sum(float(joints[name].find('origin').get('xyz').split()[2]) for name in ('base_joint','scan_joint'))
            self.assertAlmostEqual(z,cfg['lidar_clearance'][kind+'_scan_z'])

    def test_plates_cannot_be_lowered_into_lidar_clearance(self):
        from arm_sim.factory import check_lidar_clearance
        for kind,key in (('burger','deck_top'),('waffle','fork_top')):
            cfg=load_layout();cfg[kind][key]-=.005
            with self.assertRaises(ValueError):check_lidar_clearance(cfg)
