"""Empirical pose-dependent position offsets for OMX position servos, not torque control."""
import hashlib
import json
import math
from collections import deque
from pathlib import Path
from .mapping import vector

PITCH=(1,2,3)
CAP=math.radians(3)
RATE=math.radians(.5)
CONTEXTS={'empty','holding:housing','holding:ball','holding:cap','unknown'}
PHASES={'free','contact','alignment','unknown'}


def identity(mapping,urdf):
    return hashlib.sha256(json.dumps(mapping.cfg,sort_keys=True).encode()+Path(urdf).read_bytes()).hexdigest()


def bounded(v,limit=CAP):return max(-limit,min(limit,v))


class PoseMap:
    def __init__(self):
        self.points=[];self.dirty=False;self.load_status='New session'

    def observe(self,q,offset,context,dt):
        if context not in CONTEXTS or context=='unknown':return
        nearby=[p for p in self.points if p['context']==context and max(abs(a-b) for a,b in zip(p['pose'],q[:5]))<math.radians(2)]
        if nearby:
            p=min(nearby,key=lambda p:sum((a-b)**2 for a,b in zip(p['pose'],q[:5])))
            alpha=1-math.exp(-dt/.5)
            for i in range(3):
                difference=offset[i]-p['bias'][i]
                p['variance'][i]=(1-alpha)*(p['variance'][i]+alpha*difference*difference)
                p['bias'][i]+=alpha*difference
            p['seconds']=min(60,p['seconds']+dt)
        else:
            if len(self.points)>=128:self.points.pop(0)
            self.points.append(dict(pose=list(q[:5]),bias=list(offset),context=context,seconds=dt,variance=[0.]*3))
        self.dirty=True

    def predict(self,q,context):
        candidates=[]
        for p in self.points:
            if p['context']!=context or p['seconds']<1 or max(p['variance'])>math.radians(.75)**2:continue
            distance=max(abs(a-b) for a,b in zip(q[:5],p['pose']))
            if distance>=math.radians(12):continue
            candidates.append((distance,p))
        if not candidates:return [0.]*3,0.
        distance,p=min(candidates,key=lambda item:item[0])
        # Full support within 2 degrees; smoothly fade to zero at 12, no extrapolation.
        x=max(0.,(distance-math.radians(2))/math.radians(10))
        confidence=.5*(1+math.cos(math.pi*x))
        return [bounded(v)*confidence for v in p['bias']],confidence

    def save(self,path,fingerprint):
        if not self.dirty or not self.points:return
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(version=1,fingerprint=fingerprint,points=self.points),indent=2)+'\n')
        tmp.replace(path);self.dirty=False

    def load(self,path,fingerprint):
        try:
            d=json.loads(Path(path).read_text())
            if d.get('version')!=1 or d.get('fingerprint')!=fingerprint:raise ValueError('hardware/model identity mismatch')
            points=d['points']
            if not isinstance(points,list) or len(points)>128:raise ValueError('invalid point count')
            for p in points:
                vector(p['pose'],5);vector(p['bias'],3);vector(p['variance'],3)
                if p['context'] not in CONTEXTS-{'unknown'} or not 0<float(p['seconds'])<=60:raise ValueError('invalid point metadata')
                if max(abs(v) for v in p['bias'])>CAP+1e-12 or min(p['variance'])<0:raise ValueError('invalid offset')
            self.points=points;self.load_status=f'Loaded {len(points)} poses'
        except (OSError,ValueError,KeyError,TypeError,AttributeError) as e:
            self.points=[];self.load_status='Saved map ignored: '+str(e)


class Compensation:
    def __init__(self,mapping,now,enabled=True,pose_map=None):
        self.mapping=mapping;self.enabled=enabled;self.map=pose_map if pose_map is not None else PoseMap()
        self.bias=[0.]*6;self.residual=[0.]*3;self.stable_since=now;self.anchor=None
        self.context=None;self.status={};self.previous_feed=[0.]*3
        self.position_history=deque()

    def update(self,requested,measured,motors,previous,previous_requested,dt,now,effort,allowed,context,phase):
        if context not in CONTEXTS or phase not in PHASES:raise ValueError('Unknown compensation phase/context')
        # Reject invalid nominal targets even when an offset would put them inside bounds.
        self.mapping.to_ticks(requested)
        delta=min(dt,.05)
        errors=[a-b for a,b in zip(requested,measured)]
        telemetry=(len(motors)==6 and all('velocity_rad_s' in m and 'trajectory_ticks' in m and 'pwm_percent' in m for m in motors))
        # Encoder motion over time must agree with the velocity register. A single
        # reported velocity unit (1.374 deg/s) can persist with unchanged positions.
        self.position_history.append((now,list(measured)))
        while len(self.position_history)>1 and now-self.position_history[1][0]>=.5:
            self.position_history.popleft()
        history_ready=now-self.position_history[0][0]>=.5-1e-9
        position_stable=history_ready and all(
            max(q[i] for _,q in self.position_history)-min(q[i] for _,q in self.position_history)<=math.radians(.4)
            for i in range(6))
        slow=telemetry and position_stable and all(
            abs(requested[i]-previous_requested[i])/dt<math.radians(1) and
            abs(motors[i]['velocity_rad_s'])<=math.radians(1.5) for i in range(6))
        settled=telemetry and all(abs((motors[i]['trajectory_ticks']-self.mapping.zero[i])/self.mapping.scale[i]-previous[i])<math.radians(.5) for i in PITCH)
        anchor_ok=self.anchor is not None and max(abs(a-b) for a,b in zip(requested[:5],self.anchor))<math.radians(.3)
        eligible=(self.enabled and allowed and phase in ('free','alignment') and context!='unknown' and slow and settled and effort<60
                  and max(abs(errors[i]) for i in PITCH)<math.radians(6))
        if self.context!=context:
            self.residual=[0.]*3;self.context=context;eligible=False;self.anchor=None
        if not eligible or not anchor_ok:
            self.stable_since=now;self.anchor=list(requested[:5])
        integrating=eligible and now-self.stable_since>=.8
        learning=integrating and phase=='free'
        if learning:
            # At rest command-measurement estimates the offset needed, including prior correction.
            observation=[bounded(previous[i]-measured[i]) for i in PITCH]
            self.map.observe(requested,observation,context,delta)
        feed,confidence=self.map.predict(requested,context) if self.enabled and allowed and telemetry and effort<60 and phase=='free' else ([0.]*3,0.)
        for j,i in enumerate(PITCH):
            # Transfer a newly learned offset from feedback to feedforward without doubling it.
            if learning:self.residual[j]-=feed[j]-self.previous_feed[j]
            if integrating and abs(errors[i])>math.radians(.3):
                # Anti-windup: constrain residual + feedforward to the combined 3-degree budget.
                self.residual[j]=bounded(self.residual[j]+bounded(errors[i]*.6,RATE)*delta)
                self.residual[j]=max(-CAP-feed[j],min(CAP-feed[j],self.residual[j]))
            elif not slow or not allowed or phase not in ('free','alignment') or effort>=60 or not self.enabled:
                self.residual[j]-=bounded(self.residual[j],RATE*delta)
            desired=bounded(feed[j]+self.residual[j]) if self.enabled and allowed and telemetry and phase in ('free','alignment') and effort<60 else 0.
            candidate=self.bias[i]+bounded(desired-self.bias[i],RATE*delta)
            trial=list(requested);trial[i]+=candidate
            try:self.mapping.to_ticks(trial)
            except ValueError:
                # Find the feasible offset without relaxing either angular or encoder limits.
                lo,hi=0.,1.
                for _ in range(16):
                    mid=(lo+hi)/2;trial[i]=requested[i]+candidate*mid
                    try:self.mapping.to_ticks(trial);lo=mid
                    except ValueError:hi=mid
                candidate*=lo;self.residual[j]=0.
            self.bias[i]=candidate
        self.previous_feed=feed.copy()
        self.status=dict(enabled=self.enabled,kind='empirical_position_offsets',phase=phase,context=context,
            learning=learning,learned_poses=len(self.map.points),map_confidence=confidence,map_status=self.map.load_status,
            feedforward_deg=[math.degrees(v) for v in feed],residual_deg=[math.degrees(v) for v in self.residual],
            correction_deg=[math.degrees(v) for v in self.bias],stationary=bool(slow and settled),position_stable=bool(position_stable),
            position_window_s=now-self.position_history[0][0])
        return list(self.bias),self.status


def stage_context(stage):
    """Nominal task context; this does not certify physical grasp or payload."""
    kind=stage.get('kind')
    if not kind:return 'empty','free'
    action=stage['label'].rsplit(': ',1)[-1]
    if action in ('close','lift','transfer','lower'):
        return 'holding:'+kind,'free' if action=='transfer' else 'contact'
    if action in ('descend','release'):return 'empty','contact'
    if action in ('approach','retreat'):return 'empty','free'
    return 'unknown','unknown'
