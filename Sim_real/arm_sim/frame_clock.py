"""Wall-clock pacing without catch-up bursts after a slow frame."""
import math
import time


class FrameClock:
    def __init__(self, fps, *, clock=time.monotonic, sleep=time.sleep):
        if not math.isfinite(fps) or fps<=0:raise ValueError('Invalid frame rate')
        self.period=1/fps;self.clock=clock;self.sleep=sleep;self.last=None

    def due(self):
        now=self.clock()
        if self.last is not None and now-self.last<self.period-1e-9:return False
        self.last=now
        return True

    def wait(self):
        now=self.clock()
        if self.last is not None:
            delay=self.last+self.period-now
            if delay>0:self.sleep(delay)
        self.last=self.clock()


class PlaybackClock:
    """Follow wall time while bounding catch-up after rendering or a hold.

    The caller still passes every request through joint and workcell gates.
    Rendering latency must not turn a requested 8x into 3x playback.
    """
    def __init__(self, *, clock=time.monotonic):
        self.clock=clock
        self.reset()

    def reset(self, position=0.):
        self.last=self.clock();self.target=float(position);self.elapsed_s=0.

    def request(self, position, speed, running):
        now=self.clock();elapsed=max(0.,now-self.last);self.last=now
        if not running:
            self.target=float(position)
            return 0.
        self.elapsed_s+=elapsed
        # Discard unbounded backlog (e.g. debugging/suspend or a safety hold).
        self.target=min(max(self.target,position)+elapsed*speed,position+.25*speed)
        return min(max(0.,self.target-position),speed/15.)
