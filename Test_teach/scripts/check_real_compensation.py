#!/usr/bin/env python3
"""Operator-supervised, bounded OMX trial. Default is read-only; --execute enables motors.
Ends with support torque held. The operator must support the arm before releasing it.
"""
import argparse,fcntl,json,math,sys,time,select
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from teach_io.mapping import Mapping
from teach_io.tracking import Tracking
from teach_io.worker import Guard
from teach_io import bus as bus_module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    lock=(ROOT/'.session.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    mapping=Mapping(json.loads((ROOT/'config/follower_visual_checked.json').read_text()),allow_visual_confirmation=True)
    bus=bus_module.Bus(json.loads((ROOT/'config/devices.json').read_text())['follower_port'])
    result=dict(status='read_only',physical_motion=False,calibration='user_visual_confirmation_only')
    samples=[];log=None
    try:
        info=bus.inspect();sample=bus.sample(detailed=True)
        q=mapping.to_angles([m['ticks'] for m in sample])
        print('CURRENT_DEG',json.dumps([math.degrees(v) for v in q]),flush=True)
        if not args.execute:return 0
        mapping.to_ticks(q)
        expected=json.loads((ROOT/'reports/short_test_before.json').read_text())
        audit=json.loads((ROOT/'reports/short_test_mesh_audit.json').read_text())
        if audit['status']!='clear' or time.time()-expected['time']>900:raise ValueError('A fresh clear offline pose-box audit is required')
        reference=expected['angles']
        if max(abs(a-b) for a,b in zip(q,reference))>math.radians(.4):raise ValueError('Pose changed since the mesh audit; inspect again')
        if [{k:e[k] for k in ('id','model','mode','drive','offset')} for e in info]!=mapping.cfg['motor_config']:
            raise ValueError('Motor configuration changed')
        # Confirm the hand-set arm has settled before enabling the measured pose.
        for _ in range(15):
            time.sleep(.02);sample=bus.sample(detailed=True)
            now_q=mapping.to_angles([m['ticks'] for m in sample]);mapping.to_ticks(now_q)
            if max(abs(a-b) for a,b in zip(now_q,q))>math.radians(.25):raise ValueError('Arm is moving before enable')
        q=now_q
        # Stay in the already audited pitch-joint box. No gripper closing or base rotation.
        for offset in (0.,1.):
            target=q.copy();target[1]+=math.radians(offset);mapping.to_ticks(target)
        bus_module.motor_profile=lambda index:(1,2)  # temporary RAM: 2.748 deg/s maximum
        log=(ROOT/'outputs/follower/supervised_short_trial.jsonl').open('w')
        result['physical_activation_attempted']=True
        bus.enable(info,[m['ticks'] for m in sample])
        result.update(physical_motion=True,status='running',start_deg=[math.degrees(v) for v in q])
        print('SUPPORT_ON: measured pose held. Remove the support and hands, then enter CLEAR. Timeout 120 s.',flush=True)
        waiting=time.monotonic()
        while True:
            sample=bus.sample(detailed=True);measured=mapping.to_angles([m['ticks'] for m in sample])
            if max(abs(a-b) for a,b in zip(q[:5],measured[:5]))>math.radians(4):raise ValueError('Support hold lag exceeds 4 degrees')
            if max(abs(m['pwm_percent']) for m in sample[:5])>=60:raise ValueError('Support hold PWM reached 60 percent')
            bus.positions(mapping.to_ticks(q))
            if select.select([sys.stdin],[],[],0)[0]:
                if sys.stdin.readline().strip()!='CLEAR':raise ValueError('Short test cancelled before motion')
                break
            if time.monotonic()-waiting>120:raise ValueError('Workspace confirmation timeout; holding torque')
            time.sleep(.02)
        now=time.monotonic();tracker=Tracking(mapping,q,now,False);guard=Guard(mapping)
        guard.last=q.copy();guard.last_time=now
        phases=[('baseline',3.),('corrected_hold',12.),('shoulder_plus_1deg',4.),('offset_hold',5.),('return',4.),('final_hold',3.)]
        for phase,duration in phases:
            began=time.monotonic();last_print=0.
            if phase=='corrected_hold':tracker.compensation.enabled=True
            while True:
                time.sleep(.02)
                sample=bus.sample(detailed=True);measured=mapping.to_angles([m['ticks'] for m in sample]);now=time.monotonic()
                u=min(1.,(now-began)/duration);smooth=u*u*u*(10+u*(-15+6*u))
                offset=smooth if phase=='shoulder_plus_1deg' else 1. if phase=='offset_hold' else 1.-smooth if phase=='return' else 0.
                target=q.copy();target[1]+=math.radians(offset)
                if max(abs(a-b) for a,b in zip(measured[:5],target[:5]))>math.radians(4):raise ValueError('Short test lag exceeds 4 degrees')
                if max(abs(m['pwm_percent']) for m in sample[:5])>=60:raise ValueError('Short test arm PWM reached 60 percent')
                command,status=tracker.update(target,measured,sample,now,context='empty',phase='free')
                if max(abs(command[i]-reference[i]) for i in (1,2,3))>math.radians(2):raise ValueError('Command reached audited +/-2 degree box')
                ticks=guard.check(command,measured,now);bus.positions(ticks)
                row=dict(time=time.time(),phase=phase,angles=measured,motors=sample,tracking=status)
                samples.append(row);log.write(json.dumps(row)+'\n');log.flush()
                if now-last_print>=1:
                    print(phase,'lag_deg',round(status['lag_deg'],2),'bias_deg',[round(v,2) for v in status['bias_deg'][1:4]],'learning',status['compensation']['learning'],flush=True);last_print=now
                if u>=1:break
        result['status']='bounded_trial_complete'
    except BaseException as e:
        result.update(status='stopped',error=str(e))
        print('STOPPED',str(e),flush=True)
    finally:
        bus.close()  # hold measured position; never release a possibly unsupported arm
        if log:log.close()
        if args.execute:
            result['samples']=len(samples)
            result['support_torque_retained']=bool(bus.owned)
            if samples:
                result['learned_pose_count']=max(r['tracking']['compensation']['learned_poses'] for r in samples)
                result['learning_samples']=sum(r['tracking']['compensation']['learning'] for r in samples)
                result['max_bias_deg']=[max(abs(r['tracking']['bias_deg'][i]) for r in samples) for i in range(6)]
            (ROOT/'reports/supervised_short_trial.json').write_text(json.dumps(result,indent=2)+'\n')
            print('RESULT',json.dumps(result),flush=True)
    return 0 if result['status']=='bounded_trial_complete' else 2

if __name__=='__main__':sys.exit(main())
