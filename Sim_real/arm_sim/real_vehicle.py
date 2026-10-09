"""Forward U routes with smooth corners and bounded acceleration."""
import math
import numpy as np
from arm_sim.trajectory_timing import retime_tail

def rounded_path(points,radius=.15):
    points=np.asarray(points,dtype=float);out=[points[0]]
    def line(end):
        a=out[-1].copy()
        for u in np.linspace(0,1,max(2,math.ceil(np.linalg.norm(end-a)/.025)+1))[1:]:out.append(a+(end-a)*u)
    for i in range(1,len(points)-1):
        prev,corner,nxt=points[i-1:i+2];a=corner-prev;b=nxt-corner;la=np.linalg.norm(a);lb=np.linalg.norm(b)
        if min(la,lb)<1e-8:continue
        a/=la;b/=lb
        if np.dot(a,b)>.999:continue
        r=min(radius,.4*la,.4*lb);start=corner-r*a;end=corner+r*b
        line(start);delta=end-start;ta=2*r*a;tb=2*r*b
        for u in np.linspace(0,1,33)[1:]:
            out.append(start+ta*u+(10*delta-6*ta-4*tb)*u**3+(-15*delta+8*ta+7*tb)*u**4+(6*delta-3*ta-3*tb)*u**5)
    line(points[-1]);out=np.asarray(out)
    return out[np.r_[True,np.linalg.norm(np.diff(out,axis=0),axis=1)>1e-8]]

def forward_route(plan,name,points):
    track=plan.vehicles[name];start=track.at(plan.time);xy=rounded_path([start[:2],*points])
    if len(xy)<2:return
    # Dense geometric tangents retain differential-drive heading through corners.
    tangents=np.gradient(xy,axis=0);yaw=np.unwrap(np.arctan2(tangents[:,1],tangents[:,0]))
    plan.face(name,float(yaw[0]));start=track.at(plan.time);yaw+=round((start[2]-yaw[0])/(2*math.pi))*2*math.pi
    track.add(plan.time,start);first=len(track.times)-1
    for point,angle in zip(xy[1:],yaw[1:]):plan.time+=.1;track.add(plan.time,[*point,angle])
    v=plan.cfg['vehicle_motion'];retime_tail(track,first,len(track.times)-1,max_velocity=[v['linear_velocity']/math.sqrt(2)]*2+[v['angular_velocity']],max_acceleration=[v['linear_acceleration']/math.sqrt(2)]*2+[v['angular_acceleration']],phase='vehicle_forward')
    plan.time=track.times[-1]
    plan.vehicle_moves.append({'vehicle':name,'start':track.times[first],'end':plan.time,'mode':'forward_route'})

def rest_duration(delta,v,a):
    return max(.08,1.875*abs(delta)/v,math.sqrt(5.774*abs(delta)/a))
