"""One serial owner supplies voltage diagnostics and validated ROS simulation input."""
import copy
import json
import time
import rclpy
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from arm_sim.common import ROOT,load_models
from arm_sim.recovery import RecoveryGate
from scripts.leader import Mapper

class TelemetryBridge:
    def __init__(self,port):
        model=load_models()['so101'];calibration=json.loads((ROOT/'config/so101_neutral.json').read_text())
        if calibration['model']!='so101' or calibration['port']!=port:
            raise ValueError('SO-101 calibration model/port mismatch')
        self.mapper=Mapper(calibration,model);self.model=model;self.gate=RecoveryGate(3)
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO);self.node=rclpy.create_node('so101_monitored_leader')
        self.command=self.node.create_publisher(JointState,'/so101/joint_command',1)
        # Positive wall timestamp = fresh HOLD request; zero = released.
        self.hold=self.node.create_publisher(Float64,'/so101/input_hold',1)
        self.reason='정상 응답 확인 중';self.published=0

    def cycle(self,frames,limits):
        raw=[];reason=''
        for i in range(1,7):
            data,comm,error=frames.get(i,([],None,0));lo,hi=limits[i]
            if comm!=0 or len(data)!=15:reason=f'{i}번 통신 응답 없음';break
            if error or data[9]:reason=f'{i}번 모터 오류 0x{(error|data[9]):x}';break
            if lo is None or hi is None:reason=f'{i}번 전압 한계 확인 중';break
            if not lo<=data[6]*.1<=hi:reason=f'{i}번 전압 범위 이탈';break
            tick=data[0] | (data[1]<<8)
            raw.append(-(tick&0x7fff) if tick&0x8000 else tick)
        values=None
        if not reason:
            try:
                # Invalid conversion must not partially change the unwrap history.
                mapper=copy.deepcopy(self.mapper);values=mapper.convert(raw);self.mapper=mapper
            except ValueError as exc:reason=f'기준 자세/범위 확인: {exc}'
        tracking=self.gate.update(not reason)
        self.hold.publish(Float64(data=0. if tracking else time.time()))
        if tracking:
            msg=JointState();stamp=time.time_ns();msg.header.stamp.sec=stamp//1_000_000_000
            msg.header.stamp.nanosec=stamp%1_000_000_000;msg.name=self.model['joints'];msg.position=values
            self.command.publish(msg);self.published+=1;self.reason='추종 중' + (' · 한계 유지: '+', '.join(self.mapper.limited) if self.mapper.limited else '')
        else:self.reason=reason or f'정상 응답 확인 중 {self.gate.good}/3'
        rclpy.spin_once(self.node,timeout_sec=0.)
        return {'tracking':tracking,'reason':self.reason,'healthy_streak':self.gate.good,
            'recoveries':max(0,self.gate.recoveries-1),'faults':self.gate.faults,'published':self.published}

    def close(self):
        self.hold.publish(Float64(data=time.time()))
        self.node.destroy_node();rclpy.shutdown()
