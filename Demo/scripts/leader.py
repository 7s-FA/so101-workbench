#!/usr/bin/env python3
"""Leader-only readout. Never sends torque, calibration, or position writes."""
import argparse
import json
import math
import time
from pathlib import Path
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from arm_sim.common import ROOT, load_models, demo_positions

class ReadOnlyBus:
    def __init__(self, kind, port, baud):
        if kind=='so101':
            import scservo_sdk as sdk
            self.packet = sdk.PacketHandler(0)
        else:
            import dynamixel_sdk as sdk
            self.packet = sdk.PacketHandler(2.0)
        self.sdk, self.kind = sdk, kind
        self.port = sdk.PortHandler(port)
        if not self.port.openPort() or not self.port.setBaudRate(baud):
            self.port.closePort()
            raise RuntimeError(f'Cannot open {port} at {baud}')
        if kind=='so101':
            # SDK 1.0 timeout workaround, as documented in LeRobot.
            def timeout(packet_length):
                self.port.packet_start_time = self.port.getCurrentTime()
                self.port.packet_timeout = self.port.tx_time_per_byte*(packet_length+3)+50
            self.port.setPacketTimeout = timeout

    def read(self):
        values=[]
        for motor_id in range(1,7):
            if self.kind=='so101':
                raw, result, error = self.packet.read2ByteTxRx(self.port,motor_id,56)
                # STS3215 sign-magnitude position encoding (bit 15).
                raw = -(raw & 0x7fff) if raw & 0x8000 else raw
            else:
                raw, result, error = self.packet.read4ByteTxRx(self.port,motor_id,132)
                raw = raw-(1<<32) if raw & (1<<31) else raw
            if result != self.sdk.COMM_SUCCESS or error:
                raise RuntimeError(f'Motor {motor_id}: {self.packet.getTxRxResult(result)}, error={error}')
            values.append(raw)
        return values

    def close(self):
        self.port.closePort()

class Mapper:
    def __init__(self, calibration, model):
        self.c=calibration
        self.model=model
        self.previous=list(calibration['neutral_ticks'])
        self.unwrapped=[0.]*5
        self.limited=[]
        if len(self.previous)!=6 or len(calibration['signs'])!=5:
            raise ValueError('Invalid neutral/sign configuration')
        if abs(calibration['gripper_open_ticks']-calibration['gripper_closed_ticks'])<20:
            raise ValueError('Gripper range too small')

    def convert(self, raw):
        q=[]
        for i in range(5):
            # First measurement is relative to the stored physical reference pose.
            delta=(raw[i]-self.previous[i]+2048)%4096-2048
            self.unwrapped[i]+=delta
            self.previous[i]=raw[i]
            q.append(self.c['home'][i]+self.c['signs'][i]*self.unwrapped[i]*2*math.pi/4096)
        fraction=(raw[5]-self.c['gripper_closed_ticks'])/(self.c['gripper_open_ticks']-self.c['gripper_closed_ticks'])
        fraction=max(0.,min(1.,fraction))
        q.append(fraction*(1.74533 if self.c['model']=='so101' else 1.1))
        if self.model['mimic']:
            q.append(-q[-1])
        # Clamp each joint independently; malformed input remains an error.
        self.limited=[]
        for i,(name,val,(lo,hi)) in enumerate(zip(self.model['joints'],q,self.model['limits'])):
            if not math.isfinite(val):raise ValueError(f'{name}: non-finite input')
            if not lo<=val<=hi:
                self.limited.append(name)
                q[i]=max(lo,min(hi,val))
        return q


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', choices=['both','so101','omx'],default='both')
    parser.add_argument('--demo',action='store_true')
    parser.add_argument('--port')
    parser.add_argument('--baud',type=int,default=1000000)
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--capture-neutral',action='store_true')
    parser.add_argument('--rate',type=float,default=60.)
    parser.add_argument('--seconds',type=float,default=0.)
    args=parser.parse_args()
    if args.rate<=0 or args.seconds<0:
        parser.error('rate must be positive; seconds must be nonnegative')
    models=load_models()
    selected=list(models) if args.model=='both' else [args.model]
    bus=None
    if not args.demo:
        if args.model=='both' or not args.port:
            parser.error('Hardware: select one --model and its --port; run two terminals for both leaders')
        args.calibration=args.calibration or ROOT/'config'/f'{args.model}_neutral.json'
        if not args.capture_neutral and not args.calibration.exists():
            parser.error(f'Missing {args.calibration}; first use --capture-neutral')
        bus=ReadOnlyBus(args.model,args.port,args.baud)
    try:
        if args.capture_neutral:
            if args.demo:
                parser.error('capture-neutral requires hardware')
            if args.calibration.exists():
                raise FileExistsError(f'{args.calibration}: choose another --calibration path to preserve it')
            print('Simulation-only reference. Motor EEPROM and torque settings are unchanged.')
            print('Open the simulator without --demo. Align the physical leader with its virtual arm home pose.')
            print('Home joint angles:',models[args.model]['home'][:5])
            input('Press Enter when the five arm joints match the virtual arm: ')
            neutral=bus.read()
            input('Close the leader gripper/trigger, then press Enter: ')
            closed=bus.read()[5]
            input('Open the leader gripper/trigger fully, then press Enter: ')
            opened=bus.read()[5]
            if abs(opened-closed)<20:
                raise ValueError('Gripper samples too close; check the correct input device')
            calibration=dict(model=args.model,port=args.port,neutral_ticks=neutral,home=models[args.model]['home'],
                             signs=[1]*5,gripper_closed_ticks=closed,gripper_open_ticks=opened,
                             note='Simulation reference only; validate each joint direction. Does not modify LeRobot calibration.')
            args.calibration.parent.mkdir(exist_ok=True,parents=True)
            args.calibration.write_text(json.dumps(calibration,indent=2)+'\n')
            print('Saved',args.calibration)
            return
        mapper=None
        if bus:
            calibration=json.loads(args.calibration.read_text())
            if calibration['model']!=args.model or calibration['port']!=args.port:
                raise ValueError('Calibration model/port does not match; use the same stable device path')
            mapper=Mapper(calibration,models[args.model])
        rclpy.init()
        node=Node('leader_demo' if args.demo else f'{args.model}_leader_reader')
        pubs={k:node.create_publisher(JointState,f'/{k}/joint_command',1) for k in selected}
        start=time.monotonic()
        count=0
        print('Publishing to', [f'/{k}/joint_command' for k in selected],flush=True)
        while rclpy.ok() and (not args.seconds or time.monotonic()-start<args.seconds):
            tick=time.monotonic()
            raw=bus.read() if bus else None
            for kind in selected:
                values=mapper.convert(raw) if mapper else demo_positions(models[kind],tick-start)
                msg=JointState()
                # System time for real device freshness; do not use paused simulation time.
                stamp=time.time_ns()
                msg.header.stamp.sec=stamp//1_000_000_000
                msg.header.stamp.nanosec=stamp%1_000_000_000
                msg.name=models[kind]['joints']
                msg.position=values
                pubs[kind].publish(msg)
            rclpy.spin_once(node,timeout_sec=0.)
            count+=1
            time.sleep(max(0.,1/args.rate-(time.monotonic()-tick)))
        print(f'Published {count} samples per model',flush=True)
        node.destroy_node()
        rclpy.shutdown()
    finally:
        if bus:
            bus.close()

if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
