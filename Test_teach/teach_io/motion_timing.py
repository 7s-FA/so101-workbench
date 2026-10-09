"""Path-derived software ceilings and bounded wall-clock scheduling."""
import math

# 1x path peaks (rad/s): 2.232, 1.151, 1.233, .622, 2.232, 1.407.
# Motor acceleration profiles bound successive streamed goals; torque is unchanged.
VELOCITY_RAD_S=(2.4,1.3,1.4,.75,2.4,1.5)
ACCELERATION_RAD_S2=(4.5,6.0,5.3,1.6,4.5,4.6)
VELOCITY_UNIT=.229*math.tau/60
ACCELERATION_UNIT=214.577*math.tau/3600


def motor_profile(index):
    # Round DOWN so the motor profile never exceeds the stated ceiling.
    return (max(1,math.floor(ACCELERATION_RAD_S2[index]/ACCELERATION_UNIT)),
            max(1,math.floor(VELOCITY_RAD_S[index]/VELOCITY_UNIT)))


class MotionClock:
    """Advance only fresh elapsed time; discard pauses/stalls, never replay a backlog."""
    def __init__(self,dt):
        self.dt=dt;self.last=None;self.remainder=0.

    def steps(self,now,active,scale=1.,pace=1.):
        delta=0. if self.last is None else now-self.last
        self.last=now
        if not active or not 0<delta<=.15:
            self.remainder=0.;return 0
        if not 0<scale<=1 or not 0<=pace<=1:raise ValueError('Invalid real motion timing')
        self.remainder+=min(delta,.05)*scale*pace
        count=int((self.remainder+1e-12)/self.dt)
        self.remainder=max(0.,self.remainder-count*self.dt)
        return count
