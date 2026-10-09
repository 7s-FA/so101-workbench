"""Conservative 2-D footprint checks for scripted factory routes.

These checks cover the floor boundary, workbenches, pallet islands, rack posts,
and other vehicles; they do not certify full 3-D robot-arm swept volumes.
"""
import math
import numpy as np
from arm_sim.factory import warehouse_world_size


def rectangle(x,y,hx,hy,yaw=0.):
    c,s=math.cos(yaw),math.sin(yaw)
    return np.array([[x+c*dx-s*dy,y+s*dx+c*dy] for dx,dy in [(-hx,-hy),(hx,-hy),(hx,hy),(-hx,hy)]])


def overlaps(a,b):
    for poly in (a,b):
        for i in range(2):
            v=poly[i+1]-poly[i];axis=np.array([-v[1],v[0]])
            aa=a@axis;bb=b@axis
            if aa.max()<=bb.min()+1e-7 or bb.max()<=aa.min()+1e-7:return False
    return True


def burger_footprint(cfg,pose,margin=.002):
    """Union bounds of the bundled Burger collision geometry and cargo deck.

    URDF: base box 140 x 140 mm centred at x=-32 mm, wheels at y=+/-80
    mm with 18 mm thickness and 33 mm radius. The deck shares the base offset.
    """
    x,y,yaw=pose
    deck=np.asarray(cfg['burger']['deck_size'])/2
    low=np.minimum([-.102,-.089],[-.032-deck[0],-deck[1]])-margin
    high=np.maximum([.038,.089],[-.032+deck[0],deck[1]])+margin
    center=(low+high)/2;half=(high-low)/2
    c,s=math.cos(yaw),math.sin(yaw)
    return rectangle(x+c*center[0]-s*center[1],y+s*center[0]+c*center[1],*half,yaw)


def route_check(plan, interval=.1):
    cfg=plan.cfg
    obstacles={t['id']:rectangle(*t['center'],*(np.array(t['size'])/2)) for t in cfg['tables']}
    for panel in cfg.get('docking_panels',[]):
        obstacles['docking_panel/'+panel['id']]=rectangle(*panel['center'][:2],*(np.asarray(panel['size'][:2])/2))
    for i,o in enumerate(cfg['traffic']['obstacles']):
        obstacles['crate'+str(i)]=rectangle(*o['center'],*(np.array(o['size'][:2])/2+.015))
    # Check the vehicle bodies against rack legs. Forks enter between these legs.
    rack=cfg['warehouse'];rx,ry=rack['center'];rw,rd=warehouse_world_size(cfg)
    posts=[rectangle(rx+dx*(rw/2-.012),ry+dy*(rd/2-.012),.009,.009) for dx,dy in ((-1,-1),(-1,1),(1,-1),(1,1))]
    minimum=float('inf');samples=0
    for t in np.arange(0,plan.duration+interval,interval):
        poses={n:tr.at(t) for n,tr in plan.vehicles.items()};foot={}
        for name,(x,y,yaw) in poses.items():
            if name.startswith('burger'):
                foot[name]=burger_footprint(cfg,(x,y,yaw))
            else:
                foot[name]=rectangle(x+.072*math.cos(yaw),y+.072*math.sin(yaw),.273,.16,yaw)
            if np.any(np.abs(foot[name])>np.asarray(cfg['map']['size'])/2):raise ValueError(f'Demo leaves map: {name} at {t:.2f} s')
            for other,shape in obstacles.items():
                if overlaps(foot[name],shape):raise ValueError(f'Demo route overlaps {other}: {name} at {t:.2f} s')
            body=(rectangle(x-.064*math.cos(yaw),y-.064*math.sin(yaw),.133,.16,yaw)
                  if name=='waffle_1' else foot[name])
            if any(overlaps(body,post) for post in posts):raise ValueError(f'Vehicle body hits rack post: {name} at {t:.2f} s')
        names=list(foot)
        for i,name in enumerate(names):
            for other in names[i+1:]:
                if overlaps(foot[name],foot[other]):raise ValueError(f'Demo vehicles overlap: {name}, {other}, {t:.2f} s')
        minimum=min(minimum,float(np.linalg.norm(poses['burger_1'][:2]-poses['burger_2'][:2])))
        samples+=1
    return {'samples':samples,'interval_s':interval,'burger_minimum_center_distance_m':minimum,
            'result':'clear','scope':'2D conservative vehicle footprints; not arm swept-volume or sensor navigation validation'}
