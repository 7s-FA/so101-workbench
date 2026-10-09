"""Continuous high-temperature detection; no filtering, IO or motor writes.

Each joint/read path has its own monotonic timer. A fresh low reading resets
only that timer. Modes and steps never reset it; a new connection does.
"""
from copy import deepcopy
import math
import time

from teaching_settings import TEMPERATURE_LIMIT, TEMPERATURE_DURATION_S, TEMPERATURE_MAX_GAP_S


class TemperatureFault(RuntimeError):
    def __init__(self, message, detail):
        super().__init__(message)
        self.detail = deepcopy(detail)


class TemperatureGuard:
    def __init__(self, *, limit=TEMPERATURE_LIMIT, duration=TEMPERATURE_DURATION_S,
                 max_gap=TEMPERATURE_MAX_GAP_S, clock=None):
        if type(limit) is not int or not 1 <= limit <= 65:
            raise ValueError('Temperature limit must be an integer in 1..65 raw')
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
               for v in (duration, max_gap)) or duration > 5 or max_gap > 1.5:
            raise ValueError('Temperature duration/gap must be positive, at most 5/1.5 seconds')
        self.limit, self.duration, self.max_gap = limit, duration, max_gap
        self.clock = clock or time.monotonic
        self.high = {}
        self.transitions = []
        self.fault = None
        self.last_time = None

    def _now(self, now=None):
        now = self.clock() if now is None else now
        if not math.isfinite(now) or (self.last_time is not None and now < self.last_time):
            self._fail('온도 감시 시간 오류', {'reason': 'clock', 'observed_at': now})
        self.last_time = now
        return now

    def _fail(self, message, detail):
        self.fault = TemperatureFault(message, {**detail, 'state': 'fault'})
        self.transitions.append({**self.fault.detail, 'event': 'fault'})
        raise self.fault

    def check_timeouts(self, now=None):
        if self.fault is not None:
            raise self.fault
        now = self._now(now)
        for value in self.high.values():
            gap = now - value['observed_at']
            if gap > self.max_gap:
                self._fail(f"{value['joint']}: 고온 감시 공백 {gap:.2f}초; 즉시 중단",
                           {**value, 'reason': 'sample_gap', 'gap_s': gap})

    def observe(self, joint, channel, raw, *, active, inactive_limit=None):
        self.check_timeouts()
        now = self.last_time
        key = (joint, channel)
        detail = {'joint': joint, 'channel': channel, 'raw': raw, 'observed_at': now,
                  'limit_raw': self.limit, 'duration_s': self.duration, 'elapsed_s': 0.}
        if type(raw) is not int or not 0 <= raw <= 150:
            self._fail(f'{joint}: 잘못된 온도 데이터 {raw!r}; 정수 raw 0..150 필요',
                       {**detail, 'reason': 'invalid_data'})
        if not active and raw >= (self.limit if inactive_limit is None else inactive_limit):
            self._fail(f'{joint}: 시작 전 온도 {raw} raw; 즉시 중단',
                       {**detail, 'reason': 'preactivation'})
        if raw < self.limit:
            old = self.high.pop(key, None)
            if old is not None:
                self.transitions.append({**detail, 'event': 'recovered',
                                         'high_span_s': now-old['started_at']})
            return {**detail, 'state': 'normal'}
        old = self.high.get(key)
        detail.update(started_at=old['started_at'] if old else now, state='high')
        detail['elapsed_s'] = now-detail['started_at']
        self.high[key] = detail
        if old is None:
            self.transitions.append({**detail, 'event': 'started'})
        if detail['elapsed_s'] >= self.duration:
            self._fail(f'{joint}: 온도 {raw} raw, {self.limit} 이상 '
                       f'{detail["elapsed_s"]:.2f}초 지속; 중단',
                       {**detail, 'reason': 'sustained_high'})
        return deepcopy(detail)

    def pending(self):
        return deepcopy(list(self.high.values()))

    def take_transitions(self):
        result, self.transitions = self.transitions, []
        return result


def check_readings(record, guard, joint, readings, *, active, source, inactive_limit=None):
    """Attach evidence before raising, including the reading that caused a fault."""
    start = len(guard.transitions)
    channels = {source+'/'+name for name in readings}
    try:
        for name, raw in readings.items():
            guard.observe(joint, source+'/'+name, raw, active=active, inactive_limit=inactive_limit)
    except TemperatureFault as exc:
        record['temperature_error'] = str(exc)
        record['temperature_fault'] = exc.detail
        raise
    finally:
        record['temperature_pending'] = [v for v in guard.pending()
                                         if v['joint'] == joint and v['channel'] in channels]
        record['temperature_transition'] = deepcopy(guard.transitions[start:])


def pending_label(pending):
    if not pending:
        return ''
    value = max(pending, key=lambda v: v['elapsed_s'])
    return f"온도 높음 {value['elapsed_s']:.1f}/{value['duration_s']:g}초"
