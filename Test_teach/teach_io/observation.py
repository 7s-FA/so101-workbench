"""Read-only pose conversion. Candidate zeros never authorize motor commands."""
import math
from .mapping import IDS, MODELS, vector


class PoseObservation:
    def __init__(self, cfg):
        if cfg.get('kind')!='omx_follower_urdf' or cfg.get('ids')!=IDS or cfg.get('models')!=MODELS:
            raise ValueError('Expected an OMX-F pose mapping')
        self.cfg=cfg
        self.zero=vector(cfg['zero_ticks']);self.scale=vector(cfg['ticks_per_rad'])
        if any(abs(s)<100 for s in self.scale):
            raise ValueError('Invalid encoder scale')

    def decode(self, ticks, limits):
        ticks=vector(ticks)
        if len(limits)!=6:raise ValueError('Six model joint ranges are required')
        raw=[(t-z)/s for t,z,s in zip(ticks,self.zero,self.scale)]
        shown=[];limited=[]
        for i,(angle,(lo,hi)) in enumerate(zip(raw,limits)):
            if not math.isfinite(lo) or not math.isfinite(hi) or lo>=hi:
                raise ValueError('Invalid visual joint range')
            value=min(hi,max(lo,angle));shown.append(value)
            limited.append(abs(value-angle)>1e-6)
        return dict(ticks=ticks,motor_angles_rad=raw,display_angles_rad=shown,
            limited=limited,display_error_rad=[s-r for s,r in zip(shown,raw)],
            mapping_verified=self.cfg.get('verified') is True)
