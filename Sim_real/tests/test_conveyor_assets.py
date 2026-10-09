import hashlib
from pathlib import Path
import struct
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile

from scripts.prepare_conveyor_assets import ROOT, import_parts, mesh_summary, prepare


def triangle(value=1.):
    return b'\0'*80 + struct.pack('<I', 1) + struct.pack('<12fH', 0, 0, 1, 0, 0, 0, value, 0, 0, 0, 1, 1, 0)


class ConveyorAssets(unittest.TestCase):
    def test_printing_units_and_corrupt_files(self):
        self.assertEqual(mesh_summary(triangle())['extent_m'], [.001, .001, .001])
        for data in (b'', triangle()[:-1], triangle(float('nan'))):
            with self.assertRaises(ValueError):
                mesh_summary(data)

    def test_archive_import_uses_only_expected_basenames(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root/'parts.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('../../expected.stl', triangle())
                z.writestr('run_me.py', 'raise RuntimeError()')
            self.assertEqual(import_parts(archive, root/'parts', {'expected.stl'}), ['expected.stl'])
            self.assertEqual(list((root/'parts').iterdir()), [root/'parts/expected.stl'])
            self.assertFalse((root/'expected.stl').exists())
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('a/expected.stl', triangle())
                z.writestr('b/expected.stl', triangle())
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                import_parts(archive, root/'parts', {'expected.stl'})

    def test_xlerobot_parts_and_legacy_assets(self):
        source = ROOT/'assets/generated/so101.urdf'
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        original = ET.parse(source).getroot()
        with tempfile.TemporaryDirectory() as folder:
            report = prepare(Path(folder))
            revised_omx=ET.parse(Path(folder)/'omx_no_marker.urdf').getroot()
            stock_omx=ET.parse(ROOT/'assets/generated/omx.urdf').getroot()
            self.assertEqual(revised_omx.find("link[@name='end_effector_link']").findall('visual'),[])
            self.assertEqual(ET.tostring(revised_omx.find("joint[@name='end_effector_joint']")),
                             ET.tostring(stock_omx.find("joint[@name='end_effector_joint']")))
            revised = ET.parse(Path(folder)/'so101_xlerobot.urdf').getroot()
            self.assertFalse(report['runnable_demo'])
            self.assertEqual(len(report['missing_parts']), 6)
            self.assertEqual([ET.tostring(j) for j in original.findall('joint') if j.get('name') not in ('wrist_camera_joint','wrist_camera_mount_joint')],
                             [ET.tostring(j) for j in revised.findall('joint')])
            for name in ('gripper_link','moving_jaw_so101_v1_link'):
                link=revised.find(f"link[@name='{name}']")
                for tag in ('visual','collision'):
                    parts=[c for c in link.findall(tag) if c.get('name','').startswith('xlerobot_')]
                    self.assertEqual(len(parts),2)
                    for part in parts:
                        mesh=part.find('geometry/mesh')
                        self.assertEqual(mesh.get('scale'),'0.001 0.001 0.001')
                        info=mesh_summary(Path(mesh.get('filename')).read_bytes())
                        self.assertLess(max(info['extent_m']),.09)
                        self.assertGreater(min(info['extent_m']),.01)
            files=report['gripper_assembly']['files']
            self.assertEqual(mesh_summary(Path(files['fixed_finger']).read_bytes())['triangles'],3480)
            self.assertEqual(mesh_summary(Path(files['moving_finger']).read_bytes())['triangles'],3480)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),before)

    def test_four_arm_scene_has_fixed_kit_stations_without_conveyor(self):
        from arm_sim.conveyor_scene import preview_layout
        cfg=preview_layout()
        self.assertEqual({n:p['model'] for n,p in cfg['arms'].items()},
                         {'arm_a':'so101_xlerobot','arm_b':'omx','arm_c':'so101_xlerobot','arm_d':'omx'})
        self.assertFalse(cfg['conveyor']['enabled'])
        from arm_sim.conveyor_hardware import mount_plate
        for arm in ('arm_b','arm_d'):
            self.assertIsNone(mount_plate(cfg,arm))
            self.assertEqual(cfg['arms'][arm]['base_plate_thickness'],0.)
            self.assertEqual(cfg['arms'][arm]['base_origin_offset'],0.)
        for route in cfg['traffic']['routes']:
            for a,b in zip(route['points'],route['points'][1:]):self.assertNotEqual(a,b)
        stock=cfg['initial_materials']
        self.assertEqual(stock['source_loaded_banks'],[0])
        self.assertEqual(stock['burger_loaded_indices'],[0])
        self.assertEqual(9*(len(stock['source_loaded_banks'])+len(stock['burger_loaded_indices'])),stock['total_parts'])
        self.assertEqual(next(t['size'] for t in cfg['tables'] if t['id']=='assembly'),[.9,.4])
        for kit in (0,):
            self.assertCountEqual([s['kind'] for s in cfg['preparation_slots'] if s['kit']==kit],['housing','ball','cap'])
        self.assertNotEqual(cfg['handoff'],cfg['assembly_jig'])
        self.assertIn(['arm_b','arm_d'],cfg['process_reference']['parallel_pairs'])

    def test_shared_camera_tracks_c_base_but_not_other_arms(self):
        import copy
        import numpy as np
        from arm_sim.conveyor_scene import preview_layout
        from arm_sim.overhead_camera import assembly
        cfg=preview_layout();before,eye,direction=assembly(cfg)
        self.assertEqual(cfg['assembly_camera']['count'],1)
        self.assertEqual([m[0] for m in before],['ArmBase','Bottom','Middle','Top','Module'])
        moved=copy.deepcopy(cfg)
        for name,pose in moved['arms'].items():
            if name=='arm_c':continue
            pose['position']=[0.,0.];pose['yaw']=0.
        after,new_eye,new_direction=assembly(moved)
        for a,b in zip(before,after):np.testing.assert_allclose(a[3],b[3])
        np.testing.assert_allclose(eye,new_eye)
        np.testing.assert_allclose(direction,new_direction)
        moved['arms']['arm_c']['position'][0]+=.1
        shifted,shifted_eye,_=assembly(moved)
        for old,new in zip(before,shifted):np.testing.assert_allclose(new[3][:3,3]-old[3][:3,3],[.1,0,0],atol=1e-8)
        np.testing.assert_allclose(shifted_eye-eye,[.1,0,0],atol=1e-8)
        # CAD cross-sections of all three shafts must share a centreline.
        positions=[]
        for entry,center in zip(before[1:4],([.0187,.100,.0365125],[0.,.100,0.],[0.,.250,0.])):
            tf=entry[3];positions.append((tf[:3,:3]@center+tf[:3,3])[:2])
        np.testing.assert_allclose(positions,np.tile(positions[0],(3,1)),atol=1e-8)
        self.assertLess(direction[2],-.8)


    def test_camera_foot_fits_table_and_low_shelves_fit_lift(self):
        import numpy as np
        from arm_sim.conveyor_scene import preview_layout
        from arm_sim.overhead_camera import support_boxes,assembly
        from arm_sim.grasp_geometry import stl_triangles
        cfg=preview_layout();t=next(t for t in cfg['tables'] if t['id']=='assembly')
        lo=np.array(t['center'])-np.array(t['size'])/2;hi=np.array(t['center'])+np.array(t['size'])/2
        self.assertEqual(support_boxes(cfg),[])
        for name,file,scale,tf in assembly(cfg)[0]:
            if name not in ('ArmBase','Bottom'):continue
            vertices=stl_triangles(file)*scale;world=vertices@tf[:3,:3].T+tf[:3,3]
            foot=world[world[:,2]<cfg['table_height']+.012]
            self.assertTrue((foot[:,:2]>=lo-1e-6).all())
            self.assertTrue((foot[:,:2]<=hi+1e-6).all())
            if name=='ArmBase':self.assertAlmostEqual(cfg['arms']['arm_c']['mount_height'],float(np.ptp(vertices[:,1])),places=6)
        limit=cfg['waffle']['fork_max_top'];clear=cfg['waffle']['handling_clearance']
        for height in [cfg['shipping_height']]+[h for r in cfg['warehouses'] for h in r['levels']]:
            self.assertLessEqual(height+clear,limit)
            self.assertGreaterEqual(height-(cfg['waffle']['fork_top']+.001),.009)
        self.assertLessEqual(cfg['waffle']['fork_max_top'],.15)
        for arm in ('arm_a','arm_c'):
            self.assertEqual(cfg['arms'][arm]['base_plate_thickness'],0.)
            self.assertIn(cfg['arm_fixtures'][arm]['side'],('front','back','left'))

    def test_supply_grid_and_loaded_cargo_clear_tabletops(self):
        from arm_sim.conveyor_scene import preview_layout
        cfg=preview_layout();tables={t['id']:t for t in cfg['tables']}
        for (x,y),table in [(cfg['loading_dock'],tables['supply']), (cfg['unload_dock'],tables['assembly'])]:
            rear=y+.032+cfg['burger']['deck_size'][0]/2
            self.assertGreaterEqual(table['center'][1]-table['size'][1]/2-rear,.020-1e-9)
        table=tables[cfg['source_table']]
        for cx,cy in cfg['source_banks']:
            for pos,center,width in zip((cx,cy),table['center'],table['size']):
                self.assertLessEqual(abs(pos-center)+.075+.034,width/2)

    def test_purchased_boards_are_uncut_inside_map_and_do_not_overlap(self):
        import itertools
        from arm_sim.conveyor_scene import preview_layout
        cfg=preview_layout();tables=cfg['tables']
        self.assertEqual(cfg['map']['size'],[3.,2.])
        self.assertCountEqual([t['size'] for t in tables],[[.4,.4],[.9,.4]])
        for t in tables:
            for center,size,bound in zip(t['center'],t['size'],cfg['map']['size']):
                self.assertLessEqual(abs(center)+size/2,bound/2+1e-9)
        for bay in cfg['traffic']['waiting_bays']:
            for center,size,bound in zip(bay['center'],bay['size'],cfg['map']['size']):
                self.assertLessEqual(abs(center)+size/2,bound/2)
            for table in tables:
                overlap=[(sa+sb)/2-abs(x-y) for x,y,sa,sb in zip(bay['center'],table['center'],bay['size'],table['size'])]
                self.assertLessEqual(min(overlap),0.)
        for a,b in itertools.combinations(tables,2):
            overlap=[(sa+sb)/2-abs(x-y) for x,y,sa,sb in zip(a['center'],b['center'],a['size'],b['size'])]
            self.assertLessEqual(min(overlap),1e-9)

    def test_actual_base_mesh_meets_configured_support_edges(self):
        import json
        import numpy as np
        from scipy.spatial.transform import Rotation
        from arm_sim.conveyor_scene import preview_layout
        from arm_sim.grasp_geometry import stl_triangles
        from arm_sim.demo_geometry import origin
        from arm_sim.conveyor_hardware import mount_plate
        cfg=preview_layout();models=json.loads((ROOT/'assets/generated/conveyor_demo/models.json').read_text())
        for name,pose in cfg['arms'].items():
            fixture=cfg['arm_fixtures'][name]
            table=next(t for t in cfg['tables'] if t['id']==fixture['table'])
            lo=np.array(table['center'])-np.array(table['size'])/2;hi=np.array(table['center'])+np.array(table['size'])/2
            root=ET.parse(models[pose['model']]['urdf']).getroot() if Path(models[pose['model']]['urdf']).is_absolute() else ET.parse(ROOT/models[pose['model']]['urdf']).getroot()
            query="link[@name='base_link']/visual[2]" if pose['model']=='so101_xlerobot' else "link[@name='link0']/visual"
            visual=root.find(query);mesh=visual.find('geometry/mesh');tf=origin(visual.find('origin'))
            v=stl_triangles(mesh.get('filename'))*np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
            v=v@tf[:3,:3].T+tf[:3,3];v=v[v[:,2]<.007]
            xy=(v@Rotation.from_euler('z',pose['yaw']).as_matrix().T)[:,:2]+pose['position']
            gaps=np.r_[xy.min(0)-lo,hi-xy.max(0)]
            self.assertGreaterEqual(min(gaps),-1e-5)
            index={'left':0,'front':1,'right':2,'back':3}[fixture['side']]
            self.assertLess(abs(gaps[index]-fixture.get('edge_inset',0.)),1e-5)
            if 'corner' in fixture:self.assertLess(abs(gaps[1]),1e-5)
            plate=mount_plate(cfg,name)
            if plate:
                center,size=plate
                self.assertTrue((np.array(center[:2])-np.array(size[:2])/2>=lo-1e-9).all())
                self.assertTrue((np.array(center[:2])+np.array(size[:2])/2<=hi+1e-9).all())



if __name__ == '__main__':
    unittest.main()
