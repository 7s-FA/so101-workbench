"""Shared-lane reservations and fail-closed vehicle geometry checks."""
import math
from itertools import combinations
import numpy as np
from arm_sim.demo_validation import rectangle,overlaps,burger_footprint
from arm_sim.factory import warehouse_world_size


def vehicle_footprint(cfg,name,pose,margin=0.):
    if name.startswith('burger'):return burger_footprint(cfg,pose,margin=margin)
    x,y,yaw=pose
    return rectangle(x+.072*math.cos(yaw),y+.072*math.sin(yaw),.273+margin,.16+margin,yaw)


class TrafficGuard:
    def __init__(self,plan):
        self.plan=plan;cfg=plan.cfg;self.checked=0;self.last_key=None
        self.clearance=float(cfg['traffic'].get('clearance_m',.02))
        self.obstacles={t['id']:rectangle(*t['center'],*(np.asarray(t['size'])/2)) for t in cfg['tables']}
        for panel in cfg.get('docking_panels',[]):
            self.obstacles['docking_panel/'+panel['id']]=rectangle(*panel['center'][:2],*(np.asarray(panel['size'][:2])/2))
        for i,o in enumerate(cfg['traffic']['obstacles']):
            self.obstacles['crate'+str(i)]=rectangle(*o['center'],*(np.asarray(o['size'][:2])/2+.015))
        rx,ry=cfg['warehouse']['center'];rw,rd=warehouse_world_size(cfg)
        self.posts=[rectangle(rx+dx*(rw/2-.012),ry+dy*(rd/2-.012),.009,.009) for dx,dy in ((-1,-1),(-1,1),(1,-1),(1,1))]

    def check(self,time,poses=None):
        if poses is None:poses={n:tr.at(time) for n,tr in self.plan.vehicles.items()}
        if set(poses)!=set(self.plan.vehicles) or any(np.shape(p)!=(3,) or not np.isfinite(p).all() for p in poses.values()):
            raise RuntimeError('Traffic feedback missing or invalid')
        key=tuple((n,*p) for n,p in poses.items())
        if key==self.last_key:return
        cfg=self.plan.cfg;foot={};expanded={}
        for name,p in poses.items():
            foot[name]=vehicle_footprint(cfg,name,p,.002)
            expanded[name]=vehicle_footprint(cfg,name,p,self.clearance/2)
            if np.any(np.abs(foot[name])>np.asarray(cfg['map']['size'])/2):
                raise RuntimeError(f'Demo leaves map: {name} at {time:.2f} s')
            for other,shape in self.obstacles.items():
                if overlaps(foot[name],shape):raise RuntimeError(f'Demo route overlaps {other}: {name} at {time:.2f} s')
            x,y,yaw=p
            body=rectangle(x-.064*math.cos(yaw),y-.064*math.sin(yaw),.135,.162,yaw) if name=='waffle_1' else foot[name]
            if any(overlaps(body,post) for post in self.posts):raise RuntimeError(f'Vehicle body hits rack post: {name} at {time:.2f} s')
        for a,b in combinations(foot,2):
            if overlaps(expanded[a],expanded[b]):raise RuntimeError(f'Vehicle clearance below {self.clearance*1000:g} mm: {a}/{b} at {time:.2f} s')
        self.last_key=key;self.checked+=1


class TrafficGate:
    def __init__(self,plan):
        self.plan=plan;self.reset()
        jobs=sorted(getattr(plan,'traffic_jobs',[]),key=lambda j:j['start'])
        for a,b in zip(jobs,jobs[1:]):
            if a['end']>b['start']+1e-8:raise ValueError('Shared traffic reservations overlap')
        self.boundaries=sorted({job[k] for job in getattr(plan,'traffic_jobs',[]) for k in ('start','end')})

    def reset(self):
        self.completed=set();self.hold_reason=None

    def advance(self,t,end,measured):
        self.hold_reason=None
        for job in getattr(self.plan,'traffic_jobs',[]):
            if job['end']>t+1e-8 or job['id'] in self.completed:continue
            p=measured.get(job['vehicle']);expected=np.asarray(job['end_pose'])
            if p is None or np.shape(p)!=(3,) or not np.isfinite(p).all() or np.linalg.norm(p[:2]-expected[:2])>.005 or abs(math.atan2(math.sin(p[2]-expected[2]),math.cos(p[2]-expected[2])))>.02:
                self.hold_reason=f'{job["vehicle"]} 통로 이탈·도킹 확인 대기'
                return t
            self.completed.add(job['id'])
        for boundary in self.boundaries:
            if t+1e-8<boundary<end:return boundary
        return end
