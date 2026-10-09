"""ACT execution settings and wall-clock scheduling, independent of hardware."""
import math
from collections import deque


def options(values=None, params=None):
    params=params or {}
    defaults=dict(mode='original',action_steps=int(params.get('n_action_steps',20)),
        prefetch='auto',prefetch_fraction=.5,blend_ms=0.,filter_ms=0.,gripper_filter=False,
        ensemble_coeff=.01)
    values={} if values is None else values
    legacy_limits={'velocity_scale':(.1,1),'acceleration_scale':(0,1)}
    if not isinstance(values,dict) or set(values)-set(defaults)-set(legacy_limits):raise ValueError('알 수 없는 실행 설정입니다.')
    values=dict(values)
    # Old saved settings remain readable, but cannot reactivate the former demo
    # motion ceilings. The workshop sends absolute targets directly.
    for key,(lo,hi) in legacy_limits.items():
        if key in values:
            value=values.pop(key)
            if type(value) not in (int,float) or not math.isfinite(value) or not lo<=value<=hi:
                raise ValueError(f'{key}: {lo}~{hi} 범위로 입력하세요.')
    result=dict(defaults,**values)
    if result['mode'] not in ('original','prefetch','ensemble'):raise ValueError('실행 방식을 선택하세요.')
    if result['prefetch'] not in ('auto','fixed'):raise ValueError('미리 계산할 시점을 선택하세요.')
    n=result['action_steps'];maximum=int(params.get('chunk_size',100))
    if type(n) is not int or not 1<=n<=maximum:raise ValueError(f'실행 길이는 1~{maximum} 사이 정수로 입력하세요.')
    for key,lo,hi in [('prefetch_fraction',.1,.9),('blend_ms',0,200),('filter_ms',0,200),
                       ('ensemble_coeff',-.1,.1)]:
        value=result[key]
        if type(value) not in (int,float) or not math.isfinite(value) or not lo<=value<=hi:
            raise ValueError(f'{key}: {lo}~{hi} 범위로 입력하세요.')
    if type(result['gripper_filter']) is not bool:raise ValueError('그리퍼 필터 사용 여부를 선택하세요.')
    if result['mode']=='prefetch' and n<2:raise ValueError('미리 계산 방식의 실행 길이는 2 이상이어야 합니다.')
    if result['mode']=='ensemble':
        result['action_steps']=1
        result['blend_ms']=0.  # Native temporal averaging already connects overlapping predictions.
    return result


def action_vector(value):
    if not isinstance(value,(list,tuple)) or len(value)!=6 or any(type(v) not in (int,float) or not math.isfinite(v) for v in value):
        raise ValueError('모델이 올바른 6개 관절 목표를 내지 않았습니다.')
    return [float(v) for v in value]


class ChunkTimeline:
    """A chunk's sample i belongs to observation time + i/fps, not receipt time."""
    def __init__(self, fps, config, warm_ms=0.):
        if not math.isfinite(fps) or fps<=0:raise ValueError('Invalid recorded frame rate')
        self.fps=fps;self.config=config;self.queue=deque();self.last_origin=None
        self.last_tag=None;self.waiting=False;self.waits=0;self.dropped=0
        self.latencies=deque([max(0.,warm_ms/1000)],maxlen=30)
        self.peak_remaining=0

    def accept(self, result, now):
        origin=float(result['acquired'])
        if not math.isfinite(origin) or origin>now+.001:raise ValueError('잘못된 예측 시각입니다.')
        if self.last_origin is not None and origin<=self.last_origin:return False
        self.last_origin=origin
        age=now-origin
        self.latencies.append(age)
        if age>.5:
            self.dropped+=1;return False
        actions=result.get('actions')
        if not isinstance(actions,list) or not actions:raise ValueError('예측한 동작 묶음이 비어 있습니다.')
        vectors=[action_vector(a) for a in actions[:self.config['action_steps']]]
        first=max(0,math.ceil(age*self.fps-1e-7))
        new=[(origin+i/self.fps,vectors[i],origin) for i in range(first,len(vectors))]
        if not new:
            self.dropped+=1;return False
        self.queue=deque([item for item in self.queue if item[0]<new[0][0]]+new)
        self.peak_remaining=max(self.peak_remaining,len(self.queue))
        return True

    def take(self, now):
        chosen=None
        while self.queue and self.queue[0][0]<=now+1e-7:chosen=self.queue.popleft()
        # A delayed control loop cannot replay all missed setpoints in a burst.
        if chosen and now-chosen[0]>1.5/self.fps:chosen=None
        if chosen:
            changed=chosen[2]!=self.last_tag;self.last_tag=chosen[2];self.waiting=False
            return chosen[1],changed
        if not self.queue and not self.waiting:self.waits+=1;self.waiting=True
        return None,False

    def should_request(self):
        n=self.config['action_steps']
        if self.config['prefetch']=='auto':
            sorted_delays=sorted(self.latencies)
            delay=sorted_delays[min(len(sorted_delays)-1,math.ceil(.95*len(sorted_delays))-1)]
            threshold=min(n-1,max(1,math.ceil(delay*self.fps)+2))
        else:threshold=max(1,math.ceil(n*self.config['prefetch_fraction']))
        return len(self.queue)<=threshold

    def metrics(self):
        return dict(remaining=len(self.queue),buffer_waits=self.waits,dropped_chunks=self.dropped)


class GoalFilter:
    """Optional smoothing only; disabled filters pass through the exact target."""
    def __init__(self, initial, config):
        self.config=config;self.position=list(initial);self.filtered=list(initial)
        self.blend_from=list(initial);self.blend_start=None

    def apply(self, target, now, dt, new_chunk=False):
        target=action_vector(target);dt=max(0.,min(float(dt),.05))
        if new_chunk:self.blend_start=now;self.blend_from=self.filtered.copy()
        blend=self.config['blend_ms']/1000
        tau=self.config['filter_ms']/1000
        for i in range(6):
            smooth=i!=5 or self.config['gripper_filter']
            value=target[i]
            if smooth and blend and self.blend_start is not None:
                u=min(1.,max(0.,(now-self.blend_start)/blend))
                value=self.blend_from[i]+u*(value-self.blend_from[i])
            if smooth and tau:
                alpha=1.-math.exp(-dt/tau)
                self.filtered[i]+=alpha*(value-self.filtered[i])
            else:self.filtered[i]=value
        self.position=self.filtered.copy()
        return self.position.copy()
