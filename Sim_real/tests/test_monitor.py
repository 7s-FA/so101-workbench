import json
import tempfile
import time
import unittest
from pathlib import Path
from urllib.request import urlopen
from arm_sim.monitor import MonitorServer,normalize_so,read_omx
from arm_sim.voltage import VoltageState

class MonitorTests(unittest.TestCase):
    def test_so_stale_values_are_cleared_but_history_is_retained(self):
        raw={'motors':[{'id':1,'age':.1,'stale':False,'voltage':5.,'load_raw':42,'temperature':30,'history':[[1,5,5]]}]}
        self.assertEqual(normalize_so(raw)['motors'][0]['load_raw'],42)
        old=normalize_so(raw,1.)['motors'][0]
        self.assertTrue(old['stale'])
        for key in ('voltage','load_raw','temperature'):self.assertIsNone(old[key])
        self.assertEqual(old['history'],[[1,5,5]])

    def test_zero_reading_is_not_missing(self):
        raw={'motors':[{'id':1,'age':0.,'stale':False,'voltage':0.,'load_raw':0,'temperature':0}]}
        self.assertEqual(normalize_so(raw)['motors'][0]['voltage'],0.)
        self.assertFalse(normalize_so(None)['connected'])

    def test_omx_file_freshness_and_missing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'state.json'
            self.assertFalse(read_omx(path,100)['connected'])
            path.write_text(json.dumps({'timestamp':99,'voltage':5.,'motor':{'present_current_mA':40}}))
            self.assertEqual(read_omx(path,100)['motor']['present_current_mA'],40)
            self.assertIsNone(read_omx(path,104)['voltage'])
            self.assertFalse(read_omx(path,90)['connected'])
            path.write_text('invalid json')
            self.assertFalse(read_omx(path,100)['connected'])

    def test_load_sign_and_timeout_from_existing_read_packet(self):
        clock=[0.];s=VoltageState(lambda:clock[0]);data=[0]*15;data[6]=50
        # Negative magnitude 73 encoded with sign bit 10; not two's complement.
        raw=1024+73;data[4]=raw&255;data[5]=raw>>8
        s.update(1,data,0,0);self.assertEqual(s.snapshot()['motors'][0]['load_raw'],-73)
        clock[0]=.6;self.assertIsNone(s.snapshot()['motors'][0]['load_raw'])
        s.update(1,[],-6,0);self.assertIsNone(s.motors[1]['load_raw'])

    def test_http_monitor_works_with_all_sources_offline(self):
        server=MonitorServer(port=0,camera_url='http://127.0.0.1:1',voltage_url='http://127.0.0.1:1',omx_path='/nonexistent/status.json')
        try:
            base=f'http://127.0.0.1:{server.port}'
            with urlopen(base+'/api/state') as response:s=json.load(response)
            self.assertFalse(s['simulation_available']);self.assertFalse(s['so101']['connected']);self.assertFalse(s['omx']['connected'])
            with urlopen(base) as response:self.assertIn('로봇팔 통합 모니터',response.read().decode())
            with urlopen(base+'/api/health') as response:self.assertEqual(json.load(response)['application'],'final-arm-monitor')
        finally:server.close()
