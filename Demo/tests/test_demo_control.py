import json
import time
import unittest
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import numpy as np
from arm_sim.camera_stream import CameraStream
from arm_sim.monitor import MonitorServer


class DemoControlTests(unittest.TestCase):
    def test_commands_only_available_in_demo_and_reject_foreign_origin(self):
        source=CameraStream(['top'],port=0,demo_control=True)
        monitor=None
        try:
            source.update('top',np.zeros((8,8,3),dtype=np.uint8))
            source.set_metadata(mode='demo',scene='factory')
            monitor=MonitorServer(port=0,camera_url=f'http://127.0.0.1:{source.server.server_address[1]}',voltage_url='http://127.0.0.1:1')
            deadline=time.monotonic()+2
            while not monitor.snapshot()['simulation_available'] and time.monotonic()<deadline:time.sleep(.02)
            url=f'http://127.0.0.1:{monitor.port}/api/demo/control'
            for command in ('pause','resume','restart'):
                with urlopen(Request(url,data=json.dumps({'command':command}).encode(),headers={'Content-Type':'application/json'})) as r:
                    self.assertEqual(r.status,200)
            self.assertEqual(source.pop_commands(),['pause','resume','restart'])
            for mode in ('load','preloaded'):
                command={'command':'start_mode','mode':mode}
                with urlopen(Request(url,data=json.dumps(command).encode())) as r:self.assertEqual(r.status,200)
                self.assertEqual(source.pop_commands(),[command])
            for mode in ('unknown',None,True):
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(url,data=json.dumps({'command':'start_mode','mode':mode}).encode()))
                self.assertEqual(error.exception.code,400)
            for speed in (.25,1,4,16):
                with urlopen(Request(url,data=json.dumps({'command':'speed','speed':speed}).encode())) as r:
                    self.assertEqual(r.status,200)
                self.assertEqual(source.pop_commands(),[{'command':'speed','speed':speed}])
            for speed in (0,17,True,'4',None,float('nan'),float('inf')):
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(url,data=json.dumps({'command':'speed','speed':speed}).encode()))
                self.assertEqual(error.exception.code,400)
            self.assertEqual(source.pop_commands(),[])
            for data,headers,status in [(b'{"command":"motor_on"}',{},400),(b'{"command":"restart"}',{'Origin':'https://foreign.example'},403)]:
                with self.assertRaises(HTTPError) as error:urlopen(Request(url,data=data,headers=headers))
                self.assertEqual(error.exception.code,status)
            self.assertEqual(source.pop_commands(),[])
            source.demo_control=False
            with self.assertRaises(HTTPError):urlopen(Request(url,data=b'{"command":"restart"}'))
            self.assertEqual(source.pop_commands(),[])
        finally:
            if monitor:monitor.close()
            source.close()
