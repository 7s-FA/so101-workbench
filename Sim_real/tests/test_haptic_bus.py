import unittest
from unittest.mock import patch
from scripts.omx_haptics import Gripper

class Packet:
    def __init__(self):
        self.reg={0:1190,11:5,64:0,98:0,102:90,116:2300,38:100,144:50,146:25,70:0,126:0};self.writes=[]
    def __getattr__(self,name):
        if name.startswith('read'):
            return lambda port,id,a:(self.reg[a],0,0)
        def write(port,id,a,v):
            self.writes.append((id,a,v));self.reg[a]=v;return 0,0
        return write
class Bus:
    def __init__(self):self.packet=Packet();self.port=None
class BusTests(unittest.TestCase):
    def test_dry_run_never_writes(self):
        b=Bus();g=Gripper(b,False);g.apply((2300,40));g.close();self.assertEqual(b.packet.writes,[])
    def test_gripper_only_current_cap_release_restore(self):
        b=Bus();g=Gripper(b,True);g.apply((2200,200))
        self.assertEqual(b.packet.reg[102],100);self.assertEqual(b.packet.reg[64],1)
        g.apply(None);self.assertEqual(b.packet.reg[64],0)
        g.close();self.assertEqual(b.packet.reg[102],90);self.assertEqual(b.packet.reg[116],2300)
        self.assertTrue(all(id==6 for id,a,v in b.packet.writes))
        self.assertTrue(all(a>=64 for id,a,v in b.packet.writes))
    def test_low_voltage_start_keeps_reader_alive_and_blocks_force(self):
        b=Bus();b.packet.reg[144]=36
        g=Gripper(b,True);g.apply((2200,100))
        self.assertTrue(g.voltage_blocked)
        self.assertEqual(b.packet.writes,[])
        self.assertEqual(g.health(),(3.6,25))

    def test_drop_releases_and_requires_stable_recovery(self):
        b=Bus();g=Gripper(b,True);g.apply((2200,100))
        b.packet.reg[144]=36
        with patch('scripts.omx_haptics.time.monotonic',return_value=0):g.health()
        self.assertEqual(b.packet.reg[64],0)
        b.packet.reg[144]=40
        with patch('scripts.omx_haptics.time.monotonic',return_value=1):g.health()
        with patch('scripts.omx_haptics.time.monotonic',return_value=1.9):g.health()
        g.apply((2200,100));self.assertEqual(b.packet.reg[64],0)
        b.packet.reg[144]=39
        with patch('scripts.omx_haptics.time.monotonic',return_value=2):g.health()
        b.packet.reg[144]=40
        with patch('scripts.omx_haptics.time.monotonic',return_value=3):g.health()
        self.assertTrue(g.voltage_blocked)
        with patch('scripts.omx_haptics.time.monotonic',return_value=4):g.health()
        self.assertFalse(g.voltage_blocked)
        g.apply((2200,3));self.assertEqual(b.packet.reg[64],1)

    def test_other_health_fault_still_rejected(self):
        for address,value in ((144,61),(146,55),(70,1)):
            b=Bus();b.packet.reg[address]=value
            with self.assertRaises(RuntimeError):Gripper(b,True)
            self.assertEqual(b.packet.writes,[])
