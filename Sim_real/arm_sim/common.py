"""Dependency-free command validation and rate limiting; angles are radians."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load_models():
    return json.loads((ROOT / 'config/models.json').read_text())

def relative_target_limits(model):
    """Measured-position target gap in radians, not calibration or speed.

    Grip points use the existing simulation mapping from 0 to 100.
    """
    profile = model.get('relative_target')
    if profile is None:
        return None
    closed, opened = profile['gripper_radians_at_0_100']
    limits = [math.radians(profile['body_degrees'])] * 5
    limits.append(abs(opened-closed) * profile['gripper_points'] / 100.)
    if len(model['joints']) != 6 or any(not math.isfinite(x) or x <= 0 for x in limits):
        raise ValueError('Invalid measured-position relative target profile')
    return limits

def limit_relative_target(target, measured, limits):
    if len(target) != len(measured) or len(target) != len(limits):
        raise ValueError('Relative target requires every measured joint')
    if not all(math.isfinite(float(x)) for x in [*target, *measured]):
        raise ValueError('Relative target requires finite measured joints and targets')
    return [float(q) + max(-gap, min(gap, float(t)-float(q)))
            for t,q,gap in zip(target, measured, limits)]

class CommandGate:
    def __init__(self, model, timeout=0.5):
        self.names = model['joints']
        self.limits = model['limits']
        self.target = list(model['home'])
        self.command = list(self.target)
        self.speed = model.get('max_command_speed', 0.8)
        self.relative_limits = relative_target_limits(model)
        self.timeout = timeout
        self.received = None
        self.stale = True

    def receive(self, names, values, now):
        if len(names) != len(values) or len(set(names)) != len(names) or set(names) != set(self.names):
            raise ValueError('A complete, unique set of joint names is required')
        by_name = dict(zip(names, values))
        ordered = [float(by_name[n]) for n in self.names]
        for name, val, (lo, hi) in zip(self.names, ordered, self.limits):
            if not math.isfinite(val) or not lo <= val <= hi:
                raise ValueError(f'{name}: invalid/out-of-range command {val}')
        self.target = ordered
        self.received = now
        self.stale = False

    def hold(self, measured):
        self.command=[max(lo,min(hi,float(v))) for v,(lo,hi) in zip(measured,self.limits)]
        self.target=list(self.command);self.received=None;self.stale=True

    def step(self, now, dt, measured):
        if self.received is None or now - self.received > self.timeout:
            if not self.stale:
                self.command = [max(lo, min(hi, float(v))) for v, (lo, hi) in zip(measured, self.limits)]
                self.target = list(self.command)
            self.stale = True
        if not self.stale:
            step = self.speed * dt
            self.command = [c + max(-step, min(step, t-c)) for c,t in zip(self.command,self.target)]
            if self.relative_limits is not None:
                # Re-anchor to fresh follower feedback on every step. A blocked
                # joint cannot accumulate targets from the previous command.
                self.command = limit_relative_target(self.command, measured, self.relative_limits)
        return list(self.command)

def demo_positions(model, t):
    q = list(model['home'])
    # One joint at a time, returning to home between tests; includes gripper.
    index = int(t // 4) % len(q)
    phase = (t % 4) / 4 * 2 * math.pi
    q[index] += 0.18 * math.sin(phase)
    if model['mimic']:
        q[-1] = -q[-2]
    return q
