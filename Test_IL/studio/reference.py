"""LeRobot conversions. Legacy OMX metadata stays readable without controlling OMX."""
import json
from .config import ROOT
from .robot import NAMES,reference_config

class ReferenceCodec:
    def __init__(self, config=None):
        from lerobot.motors import Motor, MotorCalibration, MotorNormMode
        self.cfg=reference_config() if config is None else config
        self.buses={}
        so=self.cfg.get('robot_type')=='so101'
        if so:
            from lerobot.motors.feetech import FeetechMotorsBus as Bus
        else:
            from lerobot.motors.dynamixel import DynamixelMotorsBus as Bus
        for role in ('leader','follower'):
            cal=self.cfg.get(role)
            if not cal:continue
            models=['sts3215']*6 if so else (['xl330-m288']*5+['xl330-m077'] if role=='leader' else ['xl430-w250']*3+['xl330-m288']*3)
            body=MotorNormMode.DEGREES if so else MotorNormMode.RANGE_M100_100
            self.buses[role]=Bus(port='offline-codec',motors={n:Motor(cal[n]['id'],model,MotorNormMode.RANGE_0_100 if i==5 else body) for i,(n,model) in enumerate(zip(NAMES,models))},calibration={n:MotorCalibration(**cal[n]) for n in NAMES})

    def normalize(self,ticks,role='follower'):
        if role not in self.buses:raise ValueError('SO-101 리더 보정 파일을 먼저 등록하세요.')
        ids=[self.cfg[role][n]['id'] for n in NAMES]
        values=self.buses[role]._normalize(dict(zip(ids,ticks)))
        return [values[i] for i in ids]

    def follower_ticks(self,values):
        ids=[self.cfg['follower'][n]['id'] for n in NAMES]
        ticks=self.buses['follower']._unnormalize(dict(zip(ids,values)))
        return [ticks[i] for i in ids]

    def leader_target(self,ticks,follower_mapping):
        return follower_mapping.clamp_target(follower_mapping.to_angles(self.follower_ticks(self.normalize(ticks,'leader'))))
