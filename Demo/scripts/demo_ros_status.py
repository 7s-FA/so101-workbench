#!/usr/bin/env python3
"""Publish simulated workcell status only. Never issues hardware commands."""
import json
import signal
import time
from urllib.request import urlopen
import rclpy
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy
from std_msgs.msg import String


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node=rclpy.create_node('factory_demo_status')
    qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
    topics=['workcell']+[f'arm_{c}' for c in 'abcd']
    pubs={name:node.create_publisher(String,f'/factory_demo/{name}/state',qos) for name in topics}
    stop=False;sequence=0
    def end(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGINT,end);signal.signal(signal.SIGTERM,end)
    try:
        while not stop:
            now=time.time();sequence+=1
            try:
                with urlopen('http://127.0.0.1:8767/api/state',timeout=.5) as response:raw=json.load(response)
                demo=raw.get('simulation',{}).get('demo') or {}
                live=raw.get('simulation_available',False) and bool(demo)
            except (OSError,ValueError):demo={};live=False
            base={'timestamp':now,'sequence':sequence,'source':'isaac_assisted_demo','hardware_control':False,'available':live,'valid_for_s':1.0}
            data={**base,**(demo if live else {})}
            pubs['workcell'].publish(String(data=json.dumps(data,ensure_ascii=False,allow_nan=False)))
            for arm in topics[1:]:
                detail=demo.get('coordination',{}).get('arms',{}).get(arm,{}) if live else {}
                pubs[arm].publish(String(data=json.dumps({**base,**detail},ensure_ascii=False,allow_nan=False)))
            rclpy.spin_once(node,timeout_sec=0.);time.sleep(.2)
    finally:
        for pub in pubs.values():pub.publish(String(data=json.dumps({'available':False,'timestamp':time.time(),'hardware_control':False})))
        node.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
