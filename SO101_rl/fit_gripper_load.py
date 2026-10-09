"""Summarize arm2 stationary closing-load evidence; never edits source logs."""
from collections import defaultdict
import numpy as np
from core import DATA, ROOT, Reference, JOINTS, read_json, atomic_json, sha


def fit_arm(reference):
    groups={n:defaultdict(list) for n in JOINTS[:5]};sources={};seen=set()
    for path in sorted((DATA/'diagnostics').glob('session-*.json')):
        document=read_json(path)
        if document.get('calibration_sha256')!=reference.calibration.sha256 or document.get('role')!='follower':continue
        count=0
        for row in [*document.get('telemetry_history',[]),*document.get('recent_samples',[])]:
            for name in groups:
                health=row.get('telemetry',{}).get(name,{})
                actual=row.get('ticks',{}).get(name);goal=health.get('internal_goal_ticks')
                raw=health.get('load_raw');velocity=health.get('velocity_signed_raw')
                key=(row.get('time'),name,actual,goal,raw)
                if key in seen:continue
                seen.add(key)
                if (actual is None or goal is None or type(raw) is not int or velocity is None
                        or health.get('torque')!=1 or health.get('status')!=0 or abs(velocity)>5
                        or not 0<=raw<=2047):continue
                groups[name][abs(actual-goal)].append(raw&1023);count+=1
        if count:sources[str(path)]={'sha256':sha(path),'samples':count}
    joints={}
    for name,group in groups.items():
        if not group:raise ValueError(name+': 같은 보정의 부하 기록이 없습니다.')
        errors=sorted(group);loads=np.maximum.accumulate([np.median(group[e]) for e in errors])
        joints[name]=dict(error_ticks=errors,load_magnitude=loads.tolist(),sample_counts=[len(group[e]) for e in errors])
    return dict(schema=1,calibration_sha256=reference.calibration.sha256,sources=sources,joints=joints,
        hardware_torque_calibrated=False,kind='recorded_stationary_position_error',
        filter='Same calibration, follower, torque on, status 0, |velocity raw| <= 5; timestamps deduplicated',
        limitation='Static error/load approximation; within recorded error range only. Outside range uses conservative effort proxy. No torque calibration or dynamic validation.')


def fit(reference):
    groups=defaultdict(list);sources={};seen=set()
    for path in sorted((DATA/'diagnostics').glob('session-*.json')):
        document=read_json(path)
        if document.get('calibration_sha256')!=reference.calibration.sha256 or document.get('role')!='follower':continue
        count=0
        for row in [*document.get('telemetry_history',[]),*document.get('recent_samples',[])]:
            health=row.get('telemetry',{}).get('gripper',{})
            actual=row.get('ticks',{}).get('gripper');goal=health.get('internal_goal_ticks')
            raw=health.get('load_raw');velocity=health.get('velocity_signed_raw')
            key=(row.get('time'),actual,goal,raw)
            if key in seen:continue
            seen.add(key)
            if (actual is None or goal is None or type(raw) is not int or velocity is None
                    or health.get('torque')!=1 or health.get('status')!=0 or abs(velocity)>5
                    or not 0<=raw<=2047 or actual<goal):continue
            groups[actual-goal].append(raw&1023);count+=1
        if count:sources[str(path)]={'sha256':sha(path),'samples':count}
    if not groups:raise ValueError('같은 보정의 집게 부하 기록이 없습니다.')
    errors=sorted(groups);medians=np.array([np.median(groups[e]) for e in errors])
    monotone=np.maximum.accumulate(medians)
    residual=[abs(load-estimate) for e,estimate in zip(errors,monotone) for load in groups[e]]
    return dict(schema=1,robot_id='arm2',calibration_sha256=reference.calibration.sha256,
        kind='recorded_stationary_closing_position_error',hardware_torque_calibrated=False,
        sample_count=sum(map(len,groups.values())),source_count=len(sources),sources=sources,
        filter='Same calibration, follower, torque on, status 0, |velocity raw| <= 5, actual >= internal goal; timestamps deduplicated',
        estimator='Per-tick-error median load magnitude, monotonically nondecreasing; linear interpolation; clamp at recorded endpoints',
        error_ticks=errors,load_magnitude=monotone.tolist(),sample_counts=[len(groups[e]) for e in errors],
        in_sample_mean_absolute_error_raw=float(np.mean(residual)),
        limitation='Static closing-load approximation; no measured torque conversion or dynamic servo validation. Endpoint clamp is not evidence outside the recorded error range.')


if __name__=='__main__':
    path=ROOT/'config/gripper_load_model.json'
    model=fit(Reference());atomic_json(path,model)
    print(path,model['sample_count'],'samples from',model['source_count'],'sessions')
    path=ROOT/'config/arm_load_model.json';atomic_json(path,fit_arm(Reference()));print(path)
