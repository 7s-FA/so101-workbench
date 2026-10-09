"""Shared, deterministic current-pose entry path; no motor IO or geometry imports."""
import math
from .mapping import vector

ENTRY_SPEED = .1  # rad/s peak, including the gripper
ENTRY_SAMPLE_STEP = math.radians(.5)


def entry_duration(start, goal, peak_speed=ENTRY_SPEED):
    if not math.isfinite(peak_speed) or not 0 < peak_speed <= .75:
        raise ValueError('Entry speed must be within (0, 0.75] rad/s')
    return max(2.,1.875*max(abs(b-a) for a,b in zip(vector(start),vector(goal)))/peak_speed)


def reference_entry_ready(measured, goal, *, stationary=False, effort=math.inf):
    """Accept bounded arm load residuals; gripper contact is not an arm fault."""
    measured=vector(measured);goal=vector(goal)
    return (stationary and effort<90 and
            max(abs(a-b) for a,b in zip(measured[:5],goal[:5]))<=math.radians(6)+1e-12)


class EntryPoseChanged(ValueError):
    """Pre-enable snapshot changed; refresh and audit again without a motor write."""


def entry_samples(start, goal):
    start=vector(start);goal=vector(goal)
    count=max(1,math.ceil(max(abs(b-a) for a,b in zip(start,goal))/ENTRY_SAMPLE_STEP))
    return [[a+(b-a)*i/count for a,b in zip(start,goal)] for i in range(count+1)]


def validate_entry(plan, measured, goal, mapping, allow_start_residual=False, require_collision_check=True):
    if not isinstance(plan,dict):
        raise ValueError('A current-pose entry path is required')
    if require_collision_check and plan.get('collision_checked') is not True:
        raise ValueError('A collision-checked current-pose entry path is required')
    start=vector(plan['start']);end=vector(plan['goal'])
    if max(abs(a-b) for a,b in zip(end,vector(goal)))>1e-9:
        raise ValueError('Entry goal differs from the trial start')
    if max(abs(a-b) for a,b in zip(start,vector(measured)))>math.radians(.5):
        raise EntryPoseChanged('Arm moved during entry check; refresh the pose and check again')
    mapping.to_ticks(end)
    if allow_start_residual:
        mapping.validate_observation(start);mapping.validate_observation(measured)
    for q in entry_samples(start,end):
        mapping.to_ticks(mapping.clamp_target(q) if allow_start_residual else q)
    return start


class CurrentPoseEntry:
    """Synchronized quintic interpolation of the checked straight joint path."""
    def __init__(self, start, goal, now, reference=False, peak_speed=ENTRY_SPEED):
        self.reference=reference
        self.start=vector(start);self.home=vector(goal);self.value=list(start)
        self.time=now;self.started=now;self.elapsed=0.;self.last_error=None
        self.duration=entry_duration(start,goal,peak_speed)
        self.timeout=self.duration+20.

    def advance(self, requested, now, pace=1.):
        if max(abs(a-b) for a,b in zip(vector(requested),self.home))>1e-6:
            raise ValueError('Wait for current-pose entry before starting the trial')
        dt=now-self.time
        if not 0<dt<=.3:raise ValueError('Stale current-pose entry feedback')
        if not 0<=pace<=1:raise ValueError('Invalid entry pace')
        if now-self.started>self.timeout:
            raise ValueError('Current-pose entry did not settle; inspect load or obstruction')
        self.elapsed=min(self.duration,self.elapsed+min(dt,.05)*pace);self.time=now
        u=self.elapsed/self.duration;s=u*u*u*(10+u*(-15+6*u))
        self.value=[a+(b-a)*s for a,b in zip(self.start,self.home)]
        return self.value

    def reached(self, measured, *, stationary=False, effort=math.inf):
        if self.reference:
            return self.elapsed>=self.duration and reference_entry_ready(
                measured,self.home,stationary=stationary,effort=effort)
        return (self.elapsed>=self.duration and stationary and effort<60 and
                max(abs(a-b) for a,b in zip(vector(measured),self.home))<=math.radians(3)+1e-12)
