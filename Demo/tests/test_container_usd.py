"""USD hierarchy regression; run with Isaac's USD libraries on PYTHONPATH."""
import unittest
from arm_sim.factory import FactoryCell
try:
    from pxr import Usd,UsdGeom,Gf
except ImportError:
    Usd=None


@unittest.skipIf(Usd is None,'Isaac USD libraries required')
class ContainerUsdTests(unittest.TestCase):
    def setUp(self):
        self.stage=Usd.Stage.CreateInMemory()
        self.cell=FactoryCell.__new__(FactoryCell)
        self.cell.stage=self.stage;self.cell.G=UsdGeom;self.cell.F=Gf
        self.cell.cfg={'table_height':.23};self.cell.materials={'tray':None}
        # Appearance and contact authoring are independent of USD transform inheritance.
        self.cell.bind=lambda *args:None;self.cell.collision=lambda *args:None
        self.cell.container('/World/TestTray',(.245,0),['housing']*3,.09,.032,None)

    def test_moving_container_moves_its_carrier_and_all_nests(self):
        root=self.stage.GetPrimAtPath('/World/TestTray')
        self.assertTrue(root.IsA(UsdGeom.Xform))
        UsdGeom.Xformable(root).AddTranslateOp().Set(Gf.Vec3d(1,2,3))
        cache=UsdGeom.XformCache()
        for leaf,expected in [('Carrier',(1.245,2,3.033)),('Slot1/Base',(1.245,1.91,3.034)),('Slot3/Base',(1.245,2.09,3.034))]:
            actual=cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath('/World/TestTray/'+leaf)).ExtractTranslation()
            for a,b in zip(actual,expected):self.assertAlmostEqual(a,b,places=10)

    def test_hidden_preview_container_hides_all_its_geometry(self):
        UsdGeom.Imageable(self.stage.GetPrimAtPath('/World/TestTray')).MakeInvisible()
        for prim in self.stage.Traverse():
            if prim.IsA(UsdGeom.Cube):
                self.assertEqual(UsdGeom.Imageable(prim).ComputeVisibility(),UsdGeom.Tokens.invisible)

if __name__=='__main__':unittest.main()
