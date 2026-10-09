"""Re-express stored SO-101 positions after an explicit motor recalibration.

No hardware I/O. Present_Position = encoder - Homing_Offset (Feetech).
The encoder wraps at 4096; LeRobot converts angles using a 4095 divisor.
"""
from copy import deepcopy
import hashlib
import json
import math


def calibration_digest(calibration):
    return hashlib.sha256(json.dumps(calibration,sort_keys=True).encode()).hexdigest()


def raw_position(entry, value, gripper=False):
    if entry['drive_mode'] != 0:
        raise ValueError('이 변환은 drive_mode=0인 SO-101 보정에 사용합니다.')
    return (entry['range_min']+value/100*(entry['range_max']-entry['range_min']) if gripper
            else (entry['range_min']+entry['range_max'])/2+value*4095/360)


def convert_value(name, value, old, new, *, endpoints=False):
    a,b=old[name],new[name];grip=name=='gripper'
    if a['id']!=b['id'] or a['drive_mode']!=b['drive_mode']:
        raise ValueError('모터 ID 또는 방향이 바뀌어 자세를 자동 변환할 수 없습니다.')
    # A fully closed/open command names a newly measured endpoint, not an
    # out-of-range old encoder position. Other values preserve physical angle.
    if grip and endpoints and value in (0.,100.):return float(value)
    raw=(raw_position(a,float(value),grip)+a['homing_offset']-b['homing_offset'])%4096
    if not b['range_min']-1e-7 <= raw <= b['range_max']+1e-7:
        raise ValueError(f'{name}: 이전 자세가 새 보정 범위를 벗어납니다 ({raw:.3f} 틱).')
    return ((raw-b['range_min'])/(b['range_max']-b['range_min'])*100 if grip
            else (raw-(b['range_min']+b['range_max'])/2)*360/4095)


def convert_pose(pose, old, new, *, endpoints=False):
    return {key:convert_value(key.removesuffix('.pos'),value,old,new,endpoints=endpoints)
            for key,value in pose.items()}


def motor_limits(calibration):
    from vision_teaching import JOINTS
    return [[-(calibration[n]['range_max']-calibration[n]['range_min'])/2*360/4095,
               (calibration[n]['range_max']-calibration[n]['range_min'])/2*360/4095] for n in JOINTS]


def rebase_scene(scene, station, old, new, safe_pose, calibration_path):
    from teach_left_arm import JOINTS,stamp
    result=deepcopy(scene)
    result['tcp']=deepcopy(station['tool'])
    result['tcp']['motor_limits_deg']=motor_limits(new)
    result['home_deg']=result['safe_deg']=[safe_pose[n+'.pos'] for n in JOINTS]
    for part in result['parts'].values():
        for key in ('open_percent','close_percent'):
            part[key]=convert_value('gripper',part[key],old,new,endpoints=True)
    result['home_note']=result['safe_note']='Same physical saved safe pose, re-expressed in the user-confirmed SO-101 calibration; wrist opens vertically. Not a new encoder measurement.'
    result['motor_reference']={'calibration_path':str(calibration_path),'calibration_sha256':calibration_digest(new),
        'calibration':deepcopy(new),'accepted_by_user':True,'accepted_at':stamp(),
        'source_calibration_sha256':calibration_digest(old),
        'safe_pose_method':'Preserve encoder angle modulo 4096; calibrated positions use 4095 divisor',
        'gripper_endpoint_policy':'0 and 100 use newly measured endpoints; intermediate commands preserve encoder angle',
        'hardware_motion_verified':False}
    return result
