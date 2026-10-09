import importlib.util,json,math,tempfile,unittest,zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prepare',ROOT/'scripts/prepare_test.py')
prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)

class AssetsTest(unittest.TestCase):
    def sample_mass(self,ratio):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'sample.gcode.3mf'
            with zipfile.ZipFile(path,'w') as z:
                z.writestr('Metadata/plate_1.gcode','M83\n; FEATURE: Custom\nG1 X0 E100\n; FEATURE: Outer wall\nG1 X1 E.06578\nG1 Y1 E1\nG1 E2\nG1 X2 E-.1\n; FEATURE: Sparse infill\nG1 X3 E.4\n')
            return prepare.mass_estimate(path,ratio)
    def test_extrusion_mass_uses_printed_features_only(self):
        # Includes Bambu's leading-dot extrusion values such as E.06578.
        mass,features=self.sample_mass(1.)
        self.assertAlmostEqual(mass,1.46578*math.pi*(1.75/2)**2*1.32/1e6)
        self.assertAlmostEqual(features['Outer wall'],1.06578)
    def test_changed_infill_is_not_linear_scaling_of_entire_mass(self):
        five,_=self.sample_mass(1/3);fifteen,_=self.sample_mass(1)
        self.assertGreater(five,fifteen*.6);self.assertLess(five,fifteen)
    def test_display_links_do_not_import_as_one_kg_payloads(self):
        tree=ET.parse(ROOT/'assets/omx_teach.urdf')
        self.assertIsNone(tree.find("link[@name='world']"))
        self.assertIsNone(tree.find("link[@name='end_effector_link']"))
        mimic=tree.find("joint[@name='gripper_joint_2']/mimic")
        self.assertEqual(mimic.get('joint'),'gripper_joint_1')
        self.assertEqual(float(mimic.get('multiplier')),-1.)
    def test_latest_assets_exist_and_actual_infill_preserved(self):
        cfg=json.loads((ROOT/'config/scene.json').read_text());materials=json.loads((ROOT/'config/materials.json').read_text())
        self.assertEqual(set(cfg['parts']),{'housing','ball','cap'})
        for k,p in cfg['parts'].items():
            self.assertTrue(Path(p['mesh']).is_file());self.assertTrue(Path(p['jig']).is_file())
            self.assertEqual(materials['parts'][k]['infill_fraction'],.05)
        self.assertEqual(materials['parts']['ball']['fuzzy_skin']['thickness_mm'],.4)
        self.assertEqual(materials['parts']['ball']['fuzzy_skin']['point_distance_mm'],.6)
if __name__=='__main__':unittest.main()
