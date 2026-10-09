"""Height-first free-space paths. No hardware I/O or changes to saved poses."""
from copy import deepcopy
import math
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from vision_teaching import JOINTS


def height_pose(kin, xyz, seed, grip, *, xy_limit=40., posture_goal=None, rotation=None):
    """Solve height as an equality; aim at the grasp direction and minimize XY drift.

    Free-space only: orientation is deliberately not a contact constraint here.
    Contact poses must be solved by the ordinary IK before making this path.
    """
    xyz=np.asarray(xyz,float);qseed=np.array([seed[n+'.pos'] for n in JOINTS])
    if rotation is None:rotation=kin.fk(seed)[:3,:3]
    count=5
    low=kin.low[:count].copy();high=kin.high[:count].copy();high[3]-=2.
    def pose(q):return {**{n+'.pos':float(v) for n,v in zip(JOINTS,q)},'gripper.pos':float(grip)}
    def position(q):return kin.fk(pose(q))[:3,3]
    def cost(q):
        delta=position(q)[:2]-xyz[:2]
        regular=qseed[:count] if posture_goal is None else np.array([posture_goal[n+".pos"] for n in JOINTS[:count]])
        angular=0. if rotation is None else np.linalg.norm(Rotation.from_matrix(rotation.T@kin.fk(pose(q))[:3,:3]).as_rotvec())**2*80**2
        return float(delta@delta+angular+.002*np.sum((q-regular)**2))
    best=None
    starts=[np.clip(qseed[:count],low+1e-5,high-1e-5)]
    for elbow in (-35.,35.):
        guess=starts[0].copy();guess[2]=np.clip(elbow,low[2],high[2]);starts.append(guess)
    for start in starts:
        sol=minimize(cost,start,method='SLSQP',bounds=list(zip(low,high)),
            constraints=[{'type':'eq','fun':lambda q:position(q)[2]-xyz[2]}],
            options={'maxiter':120,'ftol':1e-7})
        point=position(sol.x);error_xy=float(np.linalg.norm(point[:2]-xyz[:2]));error_z=abs(point[2]-xyz[2])
        if np.isfinite(sol.x).all() and error_z<=.5 and error_xy<=xy_limit:
            value=(cost(sol.x),pose(sol.x))
            if best is None or value[0]<best[0]:best=value
            if value[0]<.01:break
    if best is None:raise ValueError(f'상단 높이 {xyz[2]:.1f}mm를 관절 범위와 허용 XY 이동 안에서 확보하지 못했습니다.')
    return best[1]


def transition(kin, upper, contact, *, gripper, speed, hold=0.):
    """Converge XY while descending, aiming at the contact orientation throughout.

    Reversing these nodes gives an ascent which prioritizes the requested height. Every node retains the requested height within 0.5 mm.
    """
    start=kin.fk(upper)[:3,3];end=kin.fk(contact)[:3,3]
    height=start[2]-end[2]
    if height<=0:raise ValueError('상단 높이는 집기 높이보다 높아야 합니다.')
    count=max(1,math.ceil(height/2.),math.ceil(max(abs(contact[n+'.pos']-upper[n+'.pos']) for n in JOINTS)/3.))
    nodes=[];previous=deepcopy(upper)
    for i in range(1,count+1):
        fraction=i/count;point=start+(end-start)*fraction
        if i==count:p={**contact,'gripper.pos':gripper}
        else:
            guide={key:upper[key]+(contact[key]-upper[key])*fraction for key in upper}
            p=height_pose(kin,point,previous,gripper,xy_limit=max(.5,40.*(1-fraction)),posture_goal=guide,rotation=kin.fk(contact)[:3,:3])
        duration=float(np.linalg.norm(kin.fk(p)[:3,3]-kin.fk(previous)[:3,3])/speed)
        nodes.append({'pose':p,'duration_s':duration,'hold_s':hold if i==count else 0.})
        previous=p
    return nodes


def compile_height_cycle(kin,target,xyz,rot,entry,previous):
    """Prepare complete approach/contact/lift/return paths before any motion."""
    opened,closed=(target['open'],target['closed']) if target['operation']=='pick' else (target['closed'],target['open'])
    contact=kin.ik(xyz,rot,previous,opened)
    # Side entry is the grasp orientation and stays strict near the part.
    seated=kin.ik(entry,rot,contact,opened) if np.linalg.norm(entry-xyz)>1e-8 else contact
    hover=height_pose(kin,entry+[0.,0.,target['hover_mm']],seated,opened,rotation=rot)
    descending=transition(kin,hover,seated,gripper=opened,speed=target['speed_mm_s'])
    current=descending[-1]['pose']
    for i in range(1,max(1,math.ceil(np.linalg.norm(entry-xyz)/2.))+1):
        if np.linalg.norm(entry-xyz)<1e-8:break
        point=entry+(xyz-entry)*i/max(1,math.ceil(np.linalg.norm(entry-xyz)/2.))
        p=kin.ik(point,rot,current,opened)
        descending.append({'pose':p,'duration_s':float(np.linalg.norm(kin.fk(p)[:3,3]-kin.fk(current)[:3,3])/target['speed_mm_s']),'hold_s':0.})
        current=p
    descending[-1]['hold_s']=target['wait_s']
    contact=descending[-1]['pose']
    held={**contact,'gripper.pos':closed}
    high=height_pose(kin,xyz+[0.,0.,target['retreat_mm']],held,closed,rotation=rot)
    # Generate high -> contact, then reverse exactly for the loaded ascent.
    down=transition(kin,high,held,gripper=closed,speed=target['speed_mm_s'])
    points=[high]+[n['pose'] for n in down]
    ascent=[];current=held
    for p in reversed(points[:-1]):
        ascent.append({'pose':deepcopy(p),'duration_s':float(np.linalg.norm(kin.fk(p)[:3,3]-kin.fk(current)[:3,3])/target['speed_mm_s']),'hold_s':0.});current=p
    ascent[-1]['hold_s']=target['retreat_wait_s'];down[-1]['hold_s']=target['wait_s']
    withdraw=[];current=contact
    for p in reversed([hover]+[n['pose'] for n in descending][:-1]):
        p={**p,'gripper.pos':opened}
        withdraw.append({'pose':p,'duration_s':float(np.linalg.norm(kin.fk(p)[:3,3]-kin.fk(current)[:3,3])/target['speed_mm_s']),'hold_s':0.});current=p
    withdraw[-1]['hold_s']=target['wait_s']
    return {'approach':[{'pose':hover,'duration_s':0.,'hold_s':target['wait_s']}],
            'descend':descending,'grip':[{'pose':held,'duration_s':0.,'hold_s':target['wait_s']}],
            'retreat':ascent,'return':down,
            'release':[{'pose':{**contact,'gripper.pos':opened},'duration_s':0.,'hold_s':target['wait_s']}],
            'withdraw':withdraw}
