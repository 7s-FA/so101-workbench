"""Bounded position-bias correction and measured-progress pacing, not torque control."""
import math
from collections import deque
from .mapping import vector
from .motion_timing import VELOCITY_RAD_S
from .compensation import Compensation

D=math.radians


class Tracking:
    def __init__(self, mapping, measured, now, compensation_enabled=True, pose_map=None, evaluation=False, clamp_commands=False):
        self.mapping=mapping
        self.compensation=Compensation(mapping,now,compensation_enabled,pose_map)
        self.bias=[0.]*6
        self.last=vector(measured)
        self.last_requested=self.last.copy()
        self.time=now
        self.recovering=False
        self.progress_at=now
        self.best_error=math.inf
        self.high_effort_since=None
        self.status={}
        self.evaluation=evaluation
        self.clamp_commands=clamp_commands
        self.error_samples=0
        self.error_sum=[0.]*6
        self.error_max=[0.]*6

    def record_motion_error(self):
        """Nominal target vs calibrated encoder, only while the trial is playing."""
        errors=[abs(v) for v in self.status['error_deg']]
        self.error_samples+=1
        self.error_sum=[a+b for a,b in zip(self.error_sum,errors)]
        self.error_max=[max(a,b) for a,b in zip(self.error_max,errors)]

    def error_summary(self):
        return dict(samples=self.error_samples,joint_ids=list(range(11,17)),
            mean_absolute_deg=[v/self.error_samples for v in self.error_sum] if self.error_samples else None,
            max_absolute_deg=list(self.error_max) if self.error_samples else None,
            basis='Nominal target minus calibrated encoder during active trial; gripper contact may create intentional error. Not measured TCP or grasp accuracy.')

    def update(self, requested, measured, motors, now, correction=True, context='empty', phase='free'):
        requested=vector(requested);measured=vector(measured)
        dt=now-self.time
        if not 0<dt<=.3:raise ValueError('Stale tracking feedback')
        bounded_dt=min(dt,.05)
        errors=[a-b for a,b in zip(requested,measured)]
        worst=max(range(5),key=lambda i:abs(errors[i]));error=abs(errors[worst])
        if error>.18:raise ValueError(f'ID {11+worst}: target lag {math.degrees(error):.1f} deg exceeds 10.3 deg')
        # PWM is output effort, not a force measurement. Exclude intentional gripper contact.
        effort=max(abs(m.get('pwm_percent',0)) for m in motors[:5])
        if effort>=90:
            if self.high_effort_since is None:self.high_effort_since=now
            if now-self.high_effort_since>1.5:raise ValueError('Arm output effort remains >=90% for 1.5 s; inspect load or obstruction')
        else:self.high_effort_since=None
        if error>=D(6) or effort>=80:
            if not self.recovering:self.progress_at=now;self.best_error=error
            self.recovering=True
        elif self.recovering and error<=D(5 if self.evaluation else 3)+1e-12 and effort<70:
            self.recovering=False
        if self.recovering:
            if error<self.best_error-D(.25):self.progress_at=now;self.best_error=error
            if now-self.progress_at>6:
                raise ValueError(f'ID {11+worst}: no tracking recovery for 6 s (lag {math.degrees(error):.1f} deg); inspect load or obstruction')
            scale=0.
        else:
            scale=1. if self.evaluation else max(.2,min(1.,1.-max(0,error-D(2))/D(4)*.8))
        self.bias,compensation_status=self.compensation.update(requested,measured,motors,
            self.last,self.last_requested,dt,now,effort,correction,context,phase)
        command=[a+b for a,b in zip(requested,self.bias)]
        # Limit corrections and nominal transitions together; no queued catch-up.
        command=[a+max(-limit*bounded_dt,min(limit*bounded_dt,b-a))
                 for a,b,limit in zip(self.last,command,VELOCITY_RAD_S)]
        if self.clamp_commands:command=self.mapping.clamp_target(command)
        self.mapping.to_ticks(command)
        self.last=command.copy();self.last_requested=requested.copy();self.time=now
        self.status=dict(state='catching_up' if self.recovering else 'following' if scale>.99 else 'slowing',
            profile='evaluation' if self.evaluation else 'adaptive',
            pace=scale,lag_deg=math.degrees(error),worst_id=11+worst,
            bias_deg=[math.degrees(v) for v in self.bias],pwm_percent=effort,
            desired_angles=requested,command_angles=command,compensation=compensation_status,
            error_deg=[math.degrees(v) for v in errors])
        return command,self.status


class ReferenceTracking(Tracking):
    """LeRobot direct goals, including interpolated entry; checks don't pace paths."""
    def __init__(self, mapping, measured, now):
        super().__init__(mapping,measured,now,compensation_enabled=False,evaluation=True)
        self.progress_pose=list(measured)
        self.progress_times=[now]*5
        self.position_history=deque()

    def update(self, requested, measured, motors, now, **kwargs):
        requested=self.mapping.clamp_target(requested);measured=vector(measured)
        if not 0<now-self.time<=.3:raise ValueError('Stale tracking feedback')
        self.mapping.to_ticks(requested)
        errors=[a-b for a,b in zip(requested,measured)]
        effort=max(abs(m.get('pwm_percent',0)) for m in motors[:5])
        if effort>=90:
            if self.high_effort_since is None:self.high_effort_since=now
            if now-self.high_effort_since>1.5:raise ValueError('Arm output effort remains >=90% for 1.5 s; inspect load or obstruction')
        else:self.high_effort_since=None
        for i in range(5):
            if abs(errors[i])<=D(6) or abs(measured[i]-self.progress_pose[i])>=D(.25):
                self.progress_times[i]=now;self.progress_pose[i]=measured[i]
            if now-self.progress_times[i]>6:
                raise ValueError(f'ID {11+i}: no measured movement for 6 s; inspect load or obstruction')
        worst=max(range(5),key=lambda i:abs(errors[i]))
        self.position_history.append((now,list(measured)))
        while len(self.position_history)>1 and now-self.position_history[1][0]>=.5:
            self.position_history.popleft()
        position_stable=(now-self.position_history[0][0]>=.5-1e-9 and all(
            max(q[i] for _,q in self.position_history)-min(q[i] for _,q in self.position_history)<=D(.4)
            for i in range(5)))
        stationary=position_stable and len(motors)>=5 and all(
            abs(m.get('velocity_rad_s',math.inf))<=D(1.5) for m in motors[:5])
        self.last=list(requested);self.last_requested=list(requested);self.time=now
        self.status=dict(state='following',profile='lerobot_reference',pace=1.,
            lag_deg=math.degrees(abs(errors[worst])),worst_id=11+worst,
            bias_deg=[0.]*6,pwm_percent=effort,desired_angles=list(requested),
            command_angles=list(requested),compensation={'enabled':False,'kind':'none',
                'stationary':bool(stationary),'position_stable':bool(position_stable),
                'context':kwargs.get('context','unknown'),'phase':kwargs.get('phase','unknown')},
            error_deg=[math.degrees(v) for v in errors])
        return list(requested),self.status
