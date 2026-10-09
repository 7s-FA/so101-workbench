"""Temperature/voltage tests use fake readings only; never open a USB port."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from teach_io.temperature import TemperatureGuard, TemperatureFault
from teach_io.bus import Bus


class TemperatureTests(unittest.TestCase):
    def setUp(self):
        self.now=0.;self.guard=TemperatureGuard(lambda:self.now)

    def observe(self, value, motor=12, channel='packed', active=True):
        self.guard.observe(motor,channel,value,active=active)

    def test_high_requires_five_continuous_seconds_and_latches(self):
        self.observe(65)
        for n in range(1,50):self.now=n/10;self.observe(67)
        self.assertAlmostEqual(self.guard.status()['pending'][0]['elapsed_s'],4.9)
        self.now=5
        with self.assertRaisesRegex(TemperatureFault,'5.0초'):self.observe(65)
        self.now=5.1
        with self.assertRaises(TemperatureFault):self.observe(25)
        self.assertEqual(self.guard.status()['fault']['reason'],'sustained_high')

    def test_cool_reading_resets_only_its_own_timer(self):
        self.observe(70)
        for n in range(1,5):self.now=n;self.observe(70)
        self.now=4.1;self.observe(64)
        self.assertFalse(self.guard.status()['pending'])
        self.now=4.2;self.observe(65)
        self.now=5.2;self.observe(65)
        self.assertAlmostEqual(self.guard.status()['pending'][0]['elapsed_s'],1.)
        self.assertEqual([v['event'] for v in self.guard.status()['transitions']],['started','recovered','started'])

    def test_joints_and_read_paths_do_not_clear_each_other(self):
        self.observe(65);self.observe(66,13);self.observe(67,12,'single')
        self.now=1;self.observe(25,12)
        pending=self.guard.status()['pending']
        self.assertEqual({(p['id'],p['channel']) for p in pending},{(13,'packed'),(12,'single')})
        self.guard.clear_transitions()
        self.assertEqual(len(self.guard.status()['pending']),2)

    def test_monitoring_gap_cannot_be_erased_by_late_cool_reading(self):
        self.observe(65);self.now=1.50001
        with self.assertRaisesRegex(TemperatureFault,'감시'):self.observe(25)
        self.assertEqual(self.guard.status()['fault']['reason'],'sample_gap')

    def test_pre_activation_is_immediate_and_new_connection_resets_fault(self):
        self.observe(64,active=False)
        with self.assertRaisesRegex(TemperatureFault,'시작 전'):self.observe(65,active=False)
        guard=TemperatureGuard(lambda:self.now)
        guard.observe(12,'packed',25,active=False)
        self.assertEqual(guard.status()['state'],'normal')

    def test_bad_temperature_or_clock_cannot_continue(self):
        for value in (None,True,65.,'65',-1,151):
            with self.subTest(value=value):
                guard=TemperatureGuard(lambda:0.)
                with self.assertRaises(TemperatureFault):guard.observe(12,'packed',value,active=True)
        self.observe(65);self.now=-1
        with self.assertRaises(TemperatureFault):self.observe(25)

    def bus(self, temperature=25, voltage=None, error=0):
        bus=Bus.__new__(Bus)
        def read(motor,address,size):
            return {132:2048,144:voltage if voltage is not None else (120 if motor<=13 else 50),
                    146:temperature,70:error,126:0}[address]
        bus.read=read
        return bus

    def test_legacy_teaching_keeps_immediate_55_limit(self):
        self.bus(54).sample()
        with self.assertRaises(RuntimeError):self.bus(55).sample()

    def test_bus_active_temperature_policy_and_immediate_start(self):
        bus=self.bus(65)
        rows=bus.sample(temperature_guard=self.guard,active=True)
        self.assertEqual(len(rows),6)
        self.assertEqual(len(self.guard.status()['pending']),6)
        with self.assertRaisesRegex(TemperatureFault,'시작 전'):
            bus.sample(temperature_guard=TemperatureGuard(),active=False)

    def test_voltage_and_hardware_error_are_never_delayed(self):
        for voltage,error in [(64,0),(141,0),(120,4)]:
            with self.subTest(voltage=voltage,error=error):
                with self.assertRaises(RuntimeError):
                    self.bus(65,voltage,error).sample(temperature_guard=self.guard,active=True)
        for voltage in (36,61):
            bus=self.bus(25)
            old=bus.read
            bus.read=lambda motor,address,size:voltage if motor>=14 and address==144 else old(motor,address,size)
            with self.assertRaises(RuntimeError):bus.sample(temperature_guard=self.guard,active=True)


if __name__=='__main__':unittest.main()
