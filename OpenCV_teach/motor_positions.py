"""SO-101 encoder units for the operator UI; stored poses remain calibrated."""
import math

TICK_DEGREES = 360 / 4095
TICK_STEPS = ('1', '5', '10', '20', '50')


def midpoint(backend, key, role='follower'):
    arm = getattr(backend, role, None)
    if arm is None:
        if backend.identity.get('demo') or getattr(backend,'simulated_encoders',False):
            return 2047.
        raise ValueError('해당 팔의 모터 보정값이 없습니다.')
    entry = arm.calibration[key.removesuffix('.pos')]
    return (entry.range_min + entry.range_max) / 2


def to_ticks(backend, key, position, role='follower'):
    if key == 'gripper.pos':
        raise ValueError('집게 벌림은 0~100 값을 사용합니다.')
    return round(float(position) / TICK_DEGREES + midpoint(backend, key, role))


def from_ticks(backend, key, ticks, role='follower'):
    if key == 'gripper.pos':
        raise ValueError('집게 벌림은 0~100 값을 사용합니다.')
    return (ticks - midpoint(backend, key, role)) * TICK_DEGREES


def tick_goal(backend, current, axis, amount, reference=None, now=None, hold_target=None):
    from manual_teaching import jog_goal
    if axis == 'gripper.pos' or not math.isfinite(amount) or amount != int(amount) or not 1 <= abs(amount) <= 50:
        raise ValueError('관절 이동은 1~50의 정수 모터값을 선택하세요.')
    # Coarse requests start at the measured position; small repeats accumulate.
    held = dict(hold_target or current)
    if abs(amount) == 50:
        held[axis] = current[axis]
        reference = None
    goal = jog_goal(current, None, axis, amount * TICK_DEGREES, reference, now, held)
    goal[axis] = from_ticks(backend, axis, to_ticks(backend, axis, goal[axis]))
    return goal


def preserve_integer_roundtrip(bus):
    """Avoid SDK int() turning an exact encoder position into the previous tick.

    Leave genuinely fractional goals and gripper normalization unchanged.
    """
    original = bus._unnormalize
    def unnormalize(values):
        result = original(values)
        for name, entry in bus.calibration.items():
            if name == 'gripper' or entry.id not in values:
                continue
            if bus.motors[name].norm_mode.name != 'DEGREES':
                continue
            resolution = bus.model_resolution_table[bus.motors[name].model] - 1
            raw = values[entry.id] * resolution / 360 + (entry.range_min + entry.range_max) / 2
            if abs(raw - round(raw)) < 1e-8:
                result[entry.id] = round(raw)
        return result
    bus._unnormalize = unnormalize
