"""Bounded SO-101 hold activation; no calibration/EEPROM edits.

Slow startup and fail-fast checks mitigate an observed wrist departure even
when Goal_Position equals Present_Position. Acceleration=0 alone did NOT cure
it at normal torque; these guards do not claim to repair servo firmware.
"""
import math
import time

ACTIVATION_SPEED_RAW = 1
ACTIVATION_TOLERANCE_TICKS = 8
SETTLE_SECONDS = .5
POLL_SECONDS = .01
HOLD_BODY_DEGREES = 2.
HOLD_GRIPPER_PERCENT = 5.


def check_hold(target, measured):
    """Fail before another goal is sent, rather than chasing a departing servo."""
    for key, wanted in target.items():
        limit = HOLD_GRIPPER_PERCENT if key == 'gripper.pos' else HOLD_BODY_DEGREES
        actual = measured[key]
        if not math.isfinite(actual) or abs(actual-wanted) > limit:
            raise RuntimeError(f'자세 유지 이탈: {key} 목표 {wanted:.3f}, 현재 {actual:.3f}; 토크를 해제합니다.')


def configure_preload(bus, name):
    for register, value in (('Acceleration', 0), ('Goal_Time', 0),
                            ('Goal_Velocity', ACTIVATION_SPEED_RAW)):
        bus.write(register, name, value, normalize=False, num_retry=0)
        if bus.read(register, name, normalize=False, num_retry=2) != value:
            raise RuntimeError(f'{name}: {register} 초기화 확인 실패')
        if bus.read('Torque_Enable', name, normalize=False, num_retry=2) != 0:
            raise RuntimeError(f'{name}: 목표 초기화 전에 토크가 켜졌습니다.')


def settle(bus, targets, cancelled=None, *, clock=time.monotonic, sleep=time.sleep):
    """Observe every already enabled motor against its fixed raw goal."""
    started = clock()
    extrema = {name: [goal, goal] for name, goal in targets.items()}
    while True:
        if cancelled is not None and cancelled.is_set():
            raise RuntimeError('Connection cancelled')
        for name, goal in targets.items():
            raw = bus.read('Present_Position', name, normalize=False, num_retry=2)
            extrema[name][0] = min(extrema[name][0], raw)
            extrema[name][1] = max(extrema[name][1], raw)
            if not 0 <= raw <= 4095 or abs(raw-goal) > ACTIVATION_TOLERANCE_TICKS:
                raise RuntimeError(f'{name}: 토크 시작 중 위치 이탈 (목표 {goal}, 현재 {raw} 틱)')
            if (bus.read('Goal_Position', name, normalize=False, num_retry=2) != goal
                    or bus.read('Torque_Enable', name, normalize=False, num_retry=2) != 1
                    or bus.read('Status', name, normalize=False, num_retry=2) != 0):
                raise RuntimeError(f'{name}: 토크 시작 중 목표·토크·상태 확인 실패')
        if clock()-started >= SETTLE_SECONDS:
            return extrema
        sleep(POLL_SECONDS)


def running_speed_raw(name, calibration, body_rate, gripper_rate):
    # Conservative cap for STS firmware using 50 encoder ticks/s per speed unit.
    # Firmware using 1 tick/s per unit runs slower, never faster than this cap.
    ticks_per_second = (gripper_rate*(calibration.range_max-calibration.range_min)/100
                        if name == 'gripper' else body_rate*4095/360)
    return max(1, min(60, int(ticks_per_second/50)))
