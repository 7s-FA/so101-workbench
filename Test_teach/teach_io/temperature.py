"""OMX software temperature persistence; no motor/register writes.

Opted into by Test_IL only. DYNAMIXEL Present Temperature(146) is 1 degree C.
Timers belong to a motor/read channel and persist through motion mode changes.
"""
import math
import time
from copy import deepcopy


class TemperatureFault(RuntimeError):
    def __init__(self, message, detail):
        super().__init__(message)
        self.detail=deepcopy(detail)


class TemperatureGuard:
    LIMIT_C=65
    DURATION_S=5.
    MAX_GAP_S=1.5

    def __init__(self, clock=None):
        self.clock=clock or time.monotonic
        self.high={};self.transitions=[];self.fault=None;self.last_time=None

    def fail(self, message, detail):
        self.fault=TemperatureFault(message,dict(detail,state='fault'))
        self.transitions.append(dict(self.fault.detail,event='fault'))
        raise self.fault

    def check_timeouts(self):
        if self.fault:raise self.fault
        now=self.clock()
        if not math.isfinite(now) or self.last_time is not None and now<self.last_time:
            self.fail('온도 감시 시각이 올바르지 않아 중단했습니다.',{'reason':'clock'})
        self.last_time=now
        for value in self.high.values():
            gap=now-value['observed_at']
            if gap>self.MAX_GAP_S:
                self.fail(f"ID {value['id']}: 고온 감시가 {gap:.2f}초 끊겨 중단했습니다.",
                          dict(value,reason='sample_gap',gap_s=gap))
        return now

    def observe(self, motor, channel, temperature, *, active):
        now=self.check_timeouts()
        key=(motor,channel)
        detail=dict(id=motor,channel=channel,temperature_c=temperature,observed_at=now,
                    limit_c=self.LIMIT_C,duration_s=self.DURATION_S,elapsed_s=0.)
        if type(temperature) is not int or not 0<=temperature<=150:
            self.fail(f'ID {motor}: 온도 측정값이 올바르지 않아 중단했습니다.',dict(detail,reason='invalid_data'))
        if not active and temperature>=self.LIMIT_C:
            self.fail(f'ID {motor}: 시작 전 온도 {temperature}°C가 65°C 이상이어서 구동을 차단했습니다.',
                      dict(detail,reason='preactivation'))
        if temperature<self.LIMIT_C:
            previous=self.high.pop(key,None)
            if previous:self.transitions.append(dict(detail,event='recovered',high_span_s=now-previous['started_at']))
            return
        previous=self.high.get(key)
        detail.update(started_at=previous['started_at'] if previous else now,state='high')
        detail['elapsed_s']=now-detail['started_at'];self.high[key]=detail
        if previous is None:self.transitions.append(dict(detail,event='started'))
        if detail['elapsed_s']>=self.DURATION_S:
            self.fail(f'ID {motor}: 온도 {temperature}°C, 65°C 이상이 {detail["elapsed_s"]:.1f}초 지속되어 중단했습니다.',
                      dict(detail,reason='sustained_high'))

    def status(self):
        return dict(limit_c=self.LIMIT_C,duration_s=self.DURATION_S,max_gap_s=self.MAX_GAP_S,
                    state='fault' if self.fault else 'high' if self.high else 'normal',
                    pending=deepcopy(list(self.high.values())),
                    fault=deepcopy(self.fault.detail) if self.fault else None,
                    transitions=deepcopy(self.transitions))

    def clear_transitions(self):
        self.transitions.clear()
