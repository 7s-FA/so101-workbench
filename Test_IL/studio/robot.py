"""SO-101 identity and coordinates shared by USB, recordings and ACT (no IO)."""
import hashlib
import json
import math
from pathlib import Path
from .config import ROOT, TEACH

OCV = ROOT.parent/'OpenCV_teach'
NAMES = ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']
SIM_JOINTS = NAMES[:5]+['gripper']
ROBOT_TYPE = 'so101'
GRIPPER_MODEL = 'xlerobot_tpu'
GRIPPER_RAD = 1.74533


def vector(values):
    if len(values)!=6 or any(type(x) not in (int,float) or not math.isfinite(x) for x in values):
        raise ValueError('유한한 관절값 6개가 필요합니다.')
    return list(values)


def check_calibration(data):
    if not isinstance(data,dict) or set(data)!=set(NAMES):
        raise ValueError('SO-101 관절 6개의 보정 JSON이 필요합니다.')
    for i,name in enumerate(NAMES,1):
        row=data[name]
        if any(type(row.get(k)) is not int for k in ('id','drive_mode','homing_offset','range_min','range_max')):
            raise ValueError(f'{name}: 보정값은 정수여야 합니다.')
        if row['id']!=i or row['drive_mode'] not in (0,1) or not -2047<=row['homing_offset']<=2047 or not 0<=row['range_min']<row['range_max']<=4095:
            raise ValueError(f'{name}: SO-101 ID·범위·영점 보정을 확인하세요.')
    return data


def profile():
    return json.loads((ROOT/'config/robot.json').read_text())


def reference_config():
    cfg=profile();follower=check_calibration(json.loads(Path(cfg['follower_calibration']).read_text()))
    leader=check_calibration(json.loads(Path(cfg['leader_calibration']).read_text())) if cfg.get('leader_calibration') else None
    return dict(robot_type=ROBOT_TYPE,gripper_model=GRIPPER_MODEL,body_units='degrees',
        port=cfg.get('leader_port',''),follower_port=cfg['follower_port'],follower=follower,leader=leader)


def identity(meta):
    return meta.get('robot_type') or meta.get('lerobot_calibration',{}).get('robot_type') or ('so101' if meta.get('calibration',{}).get('kind')=='so101_follower_urdf' else 'omx')


def pose_to_angles(values):
    if isinstance(values,dict):values=[values[n+'.pos'] for n in NAMES]
    v=vector(values)
    return [math.radians(x) for x in v[:5]]+[v[5]*GRIPPER_RAD/100]


def angles_to_pose(q):
    q=vector(q)
    return dict(zip((n+'.pos' for n in NAMES),[math.degrees(x) for x in q[:5]]+[q[5]*100/GRIPPER_RAD]))


class Mapping:
    """Same midrange degrees and gripper fraction as latest OpenCV teaching."""
    def __init__(self, config=None):
        ref=reference_config() if config is None else config
        cal=check_calibration(ref['follower'])
        self.zero=[(cal[n]['range_min']+cal[n]['range_max'])/2 for n in NAMES[:5]]+[cal['gripper']['range_min']]
        self.scale=[4095/math.tau]*5+[(cal['gripper']['range_max']-cal['gripper']['range_min'])/GRIPPER_RAD]
        ticks=[[cal[n]['range_min'],cal[n]['range_max']] for n in NAMES]
        limits=[[(a-z)/s,(b-z)/s] for (a,b),z,s in zip(ticks,self.zero,self.scale)]
        self.cfg=dict(kind='so101_follower_urdf',robot_type=ROBOT_TYPE,gripper_model=GRIPPER_MODEL,
            ids=list(range(1,7)),models=['sts3215']*6,zero_ticks=self.zero,ticks_per_rad=self.scale,
            limits=limits,tick_limits=ticks,calibration=cal)

    def to_angles(self,ticks):return [(v-z)/s for v,z,s in zip(vector(ticks),self.zero,self.scale)]
    def clamp_target(self,q):
        result=[]
        for v,(lo,hi),(a,b),z,s in zip(vector(q),self.cfg['limits'],self.cfg['tick_limits'],self.zero,self.scale):
            lo=max(lo,(a-z)/s);hi=min(hi,(b-z)/s)
            if lo>hi:raise ValueError('서로 맞지 않는 SO-101 관절 범위입니다.')
            result.append(max(lo,min(hi,v)))
        return result
    def to_ticks(self,q):
        q=vector(q)
        if any(abs(a-b)>1e-9 for a,b in zip(q,self.clamp_target(q))):raise ValueError('SO-101 목표가 보정 범위를 벗어났습니다.')
        return [max(lo,min(hi,round(z+v*s))) for v,z,s,(lo,hi) in zip(q,self.zero,self.scale,self.cfg['tick_limits'])]
    def validate_observation(self,q):
        q=vector(q)
        # Same entry allowance as OpenCV, without extending command bounds.
        import sys
        sys.path.insert(0,str(OCV))
        from teaching_settings import ENTRY_MARGIN_DEG,GRIPPER_ENTRY_MAX_RAW,GRIPPER_ENTRY_MARGIN_UNITS
        margin=[math.radians(ENTRY_MARGIN_DEG)]*5+[min(GRIPPER_ENTRY_MAX_RAW/self.scale[5],GRIPPER_ENTRY_MARGIN_UNITS*GRIPPER_RAD/100)]
        if any(abs(a-b)>m+1e-9 for a,b,m in zip(q,self.clamp_target(q),margin)):
            raise ValueError('SO-101 현재 자세가 보정 범위의 시작 허용치를 벗어났습니다.')
        return q


def register_leader(port, calibration):
    if not isinstance(port,str) or Path(port).parent!=Path('/dev/serial/by-id') or not port.startswith('/dev/serial/by-id/usb-1a86_'):
        raise ValueError('SO-101 리더의 USB 포트를 선택하세요.')
    cfg=profile()
    if Path(port).resolve()==Path(cfg['follower_port']).resolve():raise ValueError('리더와 팔로워는 서로 다른 USB여야 합니다.')
    check_calibration(calibration)
    from .store import atomic_json
    target=ROOT/'config/so101_leader.json'
    atomic_json(target,calibration)
    cfg.update(leader_port=port,leader_calibration=str(target))
    atomic_json(ROOT/'config/robot.json',cfg)
    atomic_json(ROOT/'config/lerobot_calibration.json',reference_config())
    return dict(ok=True,message='리더 보정 파일을 등록했습니다. 연결할 때 모터의 보정값과 대조합니다.')
