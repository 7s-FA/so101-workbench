import unittest
from calibration.link_mass.gcode_mass import extrusion

class ExtrusionTests(unittest.TestCase):
    def test_relative_extrusion_retraction_recovery_and_features(self):
        result=extrusion('''M83
; start printing object, unique label id: 42
; FEATURE: Outer wall
G1 X1 Y0 E2
G1 E-0.8
G1 X2 E0.8
G1 X3 E1
; FEATURE: Brim
G1 X4 E0.2
; stop printing object, unique label id: 42
G1 E5
'''.splitlines())['extrusion_mm_by_object_feature']
        self.assertAlmostEqual(result['42']['Outer wall'],3.)
        self.assertAlmostEqual(result['42']['Brim'],.2)
        self.assertAlmostEqual(result['unassigned']['stationary_extrusion'],5.)

    def test_absolute_e_reset_relative_xy_and_full_circle(self):
        result=extrusion('''M82
G92 E10
G90
G1 X5 Y5 Z0.2
; start printing object, unique label id: 7
; FEATURE: Inner wall
G1 X6 E12
G92 E0
G91
G1 X1 Y0 Z0.1 E1
G90
G2 X7 Y5 I1 J0 E2
'''.splitlines())
        self.assertAlmostEqual(result['extrusion_mm_by_object_feature']['7']['Inner wall'],4.)
        self.assertAlmostEqual(result['path_bounds_mm']['7'][4],.3)

    def test_cancel_ids_are_data_and_do_not_share_mass(self):
        result=extrusion('''M83
; start printing object, unique label id: 1
; FEATURE: Sparse infill
G1 X1 E2
; stop printing object, unique label id: 1
; start printing object, unique label id: 2
G1 X2 E3
'''.splitlines())
        self.assertEqual(result['extrusion_mm_by_object_feature']['1']['Sparse infill'],2)
        self.assertEqual(result['extrusion_mm_by_object_feature']['2']['Sparse infill'],3)

if __name__=='__main__':unittest.main()
