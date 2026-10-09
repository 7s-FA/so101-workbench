#!/usr/bin/env python3
"""Inventory, read-only inspection and manual measurement-based calibration."""
import argparse
import glob
import json
import math
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from teach_io.bus import Bus
from teach_io.mapping import IDS, MODELS, HARD_LIMITS, Mapping, fit_arm_pose, vector


def angles(prompt,n):
    return vector([float(x.strip()) for x in input(prompt).split(',')],n)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['devices','inspect','calibrate','release'])
    devices=json.loads((ROOT/'config/devices.json').read_text()) if (ROOT/'config/devices.json').exists() else {}
    p.add_argument('--port',default=devices.get('follower_port'))
    p.add_argument('--output',type=Path,default=ROOT/'config/follower.json')
    args=p.parse_args()
    if args.action=='devices':
        print(json.dumps({'serial':glob.glob('/dev/serial/by-id/*'),
            'cameras':glob.glob('/dev/v4l/by-id/*'),'video_nodes':glob.glob('/dev/video*')},indent=2));return
    if not args.port:p.error('--port is required; do not reuse the leader port')
    if args.action=='calibrate' and args.output.exists():
        p.error('Calibration already exists; choose --output to preserve it')
    bus=Bus(args.port)
    try:
        info=bus.inspect()
        if args.action=='release':
            print('팔과 잡고 있는 물체를 손/받침으로 지지하세요. 토크를 끄면 팔이 내려갈 수 있습니다.')
            if input('지지를 완료했으면 SUPPORTED 입력: ').strip()!='SUPPORTED':return
            bus.release()
            print('Torque OFF; watchdog cleared. No EEPROM changes.');return
        sample=bus.sample()
        if args.action=='inspect':
            print(json.dumps({'motors':info,'sample':sample},indent=2));return
        if any(m['torque'] for m in info):
            raise RuntimeError('수동 보정은 모든 관절 토크 OFF 상태에서만 진행합니다. 프로그램은 토크를 해제하지 않습니다.')
        print('모터 설정을 쓰지 않습니다. 팔을 지지하면서 URDF와 일치하는 관절 각도를 측정하세요.')
        print('단순히 편한 자세를 0도로 입력하면 안 됩니다. joint1~5 순서, 단위 rad.')
        print('두 자세는 각 관절이 0.15 rad 이상 달라야 하며, 엔코더 0/4095 경계를 넘지 않아야 합니다.')
        q1=angles('첫 자세의 실측 URDF 각도 5개 (쉼표 구분): ',5)
        input('첫 자세를 유지하고 Enter: ');t1=[m['ticks'] for m in bus.sample()]
        q2=angles('두 번째 자세의 실측 URDF 각도 5개: ',5)
        input('두 번째 자세를 유지하고 Enter: ');t2=[m['ticks'] for m in bus.sample()]
        zero,scale=fit_arm_pose(q1,t1[:5],q2,t2[:5])
        input('그리퍼를 무리한 힘 없이 닫아 URDF 0 rad에 맞추고 Enter: ')
        closed=bus.sample()[5]['ticks']
        opened_angle=angles('그리퍼를 벌린 뒤 측정한 gripper_joint_1 각도(rad): ',1)[0]
        if not .3<=opened_angle<=1.1:raise ValueError('Gripper measured angle must be 0.3–1.1 rad')
        input('그 자세를 유지하고 Enter: ');opened=bus.sample()[5]['ticks']
        if abs(opened-closed)<100:raise ValueError('Gripper encoder span too small')
        zero.append(closed);scale.append((opened-closed)/opened_angle)
        models=json.loads((ROOT.parent/'Sim_real/assets/generated/conveyor_demo/models.json').read_text())['omx']
        limits=[list(x) for x in HARD_LIMITS];limits[5]=[0,opened_angle]
        data=dict(kind='omx_follower_urdf',verified=True,port=str(Path(args.port).absolute()),ids=IDS,models=MODELS,
            zero_ticks=zero,ticks_per_rad=scale,limits=limits,
            tick_limits=[[max(0,e['min_ticks']),min(4095,e['max_ticks'])] if e['mode']==3 else [0,4095] for e in info],
            motor_config=[{k:e[k] for k in ('id','model','mode','drive','offset')} for e in info],
            measured_poses=[{'angles':q1,'ticks':t1},{'angles':q2,'ticks':t2}],
            note='Manual URDF pose measurements; no EEPROM changes. Validate unloaded motion before manipulation.')
        Mapping(data)
        if any(not 0<=t<=4095 for t in t1+t2+[closed,opened]):
            raise ValueError('Wrapped/multi-turn calibration is not supported')
        print(json.dumps(data,indent=2))
        if input('각도·방향·부품 기준을 실제로 확인했으면 VERIFIED 입력: ').strip()!='VERIFIED':
            raise RuntimeError('보정 파일을 저장하지 않았습니다.')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as f:json.dump(data,f,indent=2)
        print('Saved',args.output)
    finally:bus.close()


if __name__=='__main__':main()
