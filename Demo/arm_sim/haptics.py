"""Opening spring plus a latched contact stop; current is not fingertip force."""
import math

class ContactCue:
    def __init__(self, closed, opened, maximum=100):
        if abs(opened-closed)<20 or not 40<=maximum<=100:raise ValueError('Invalid haptic limits')
        self.closed,self.opened,self.maximum=closed,opened,maximum
        self.previous=None;self.wall=None;self.current=0.;self.last_time=None;self.state='released'

    def release(self):
        self.wall=None;self.current=0.;self.state='released'
        return None

    def step(self, raw, sample, now):
        dt=0. if self.last_time is None else max(0.,min(.05,now-self.last_time))
        self.last_time=now
        if not math.isfinite(raw):return self.release()
        fraction=(raw-self.closed)/(self.opened-self.closed)
        opening=self.previous is not None and fraction-self.previous>.003
        self.previous=fraction
        if sample is None or len(sample)!=4 or not all(math.isfinite(v) for v in sample):return self.release()
        stamp,angle,force,enabled=sample
        if not -.03<=now-stamp<=.15 or enabled!=1 or not 0<=angle<=1.1 or force<0 or not 0<=fraction<=1:return self.release()
        # Opening releases the contact stop; the gentle opening spring may assist.
        if opening or force<.15 or (self.wall is not None and fraction>=self.wall-.005):self.wall=None
        if not opening and self.wall is None and force>=.15 and angle/1.1-fraction>.015:
            self.wall=angle/1.1
        if self.wall is not None:
            target=self.closed+(self.opened-self.closed)*self.wall
            desired=min(self.maximum,60+max(0,self.wall-fraction)*400)
            self.state='contact_stop'
        elif fraction<.985:
            target=self.opened;desired=40.;self.state='opening_spring'
        else:return self.release()
        # Fixed target, ramped current. Do not move the contact stop along with the hand.
        self.current=min(desired,self.current+200*dt)
        if self.current<1:return None
        return round(target),round(self.current)
