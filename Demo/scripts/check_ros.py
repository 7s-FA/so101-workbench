#!/usr/bin/env python3
"""Observe actual physics feedback while an external demo publisher runs."""
import json
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from arm_sim.common import ROOT,load_models

rclpy.init()
node=Node('dual_arm_feedback_check')
models=load_models()
samples={k:[] for k in models}
def callback(kind,msg):
    data=dict(zip(msg.name,msg.position))
    if set(models[kind]['joints']).issubset(data):
        samples[kind].append([data[n] for n in models[kind]['joints']])
subscriptions=[node.create_subscription(JointState,f'/{k}/joint_states',lambda msg,k=k:callback(k,msg),10) for k in models]
end=time.monotonic()+35
while time.monotonic()<end:
    rclpy.spin_once(node,timeout_sec=.1)
result={}
for k,rows in samples.items():
    if len(rows)<10:
        raise RuntimeError(f'{k}: only {len(rows)} feedback samples')
    ranges=[max(col)-min(col) for col in zip(*rows)]
    if not any(x>.05 for x in ranges):
        raise RuntimeError(f'{k}: no observed motion')
    result[k]={'feedback_samples':len(rows),'motion_range_rad':ranges}
(ROOT/'outputs/ros_feedback_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
node.destroy_node()
rclpy.shutdown()
