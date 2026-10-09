"""Continuous joint trajectories with certified polynomial derivative limits."""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import PPoly


def bounds(poly, derivative=0):
    p=poly.derivative(derivative)
    left=p.c[-1];right=np.zeros_like(left)
    for coefficient in p.c:right=right*np.diff(p.x)[:,None]+coefficient
    low=np.minimum(left,right);high=np.maximum(left,right)
    for j in range(p.c.shape[2]):
        scalar=PPoly(p.c[:,:,j],p.x)
        roots=scalar.derivative().roots(extrapolate=False)
        roots=roots[np.isfinite(roots)]
        for x in roots:
            i=min(np.searchsorted(p.x,x,side='right')-1,len(p.x)-2)
            value=scalar(x);low[i,j]=min(low[i,j],value);high[i,j]=max(high[i,j],value)
    return low,high


@dataclass
class MotionSpan:
    spline: object
    start: float
    duration: float
    velocity_bound: float
    acceleration_bound: float
    lower: object
    upper: object

    def at(self,t):
        return self.spline(np.clip(t-self.start,0.,self.duration))


def install(track,first,last,spline,knot_times,vmax,amax,phase,scale=1.):
    vl,vh=bounds(spline,1);al,ah=bounds(spline,2);lo,hi=bounds(spline)
    span=MotionSpan(spline,track.times[first],float(spline.x[-1]),float(np.max(np.maximum(abs(vl),abs(vh)))),float(np.max(np.maximum(abs(al),abs(ah)))),np.min(lo,axis=0),np.max(hi,axis=0))
    span.time_scale=scale;span.phase=phase;span.velocity_limits=vmax.copy();span.acceleration_limits=amax.copy()
    if not hasattr(track,'spans'):track.spans={}
    track.times[first+1:last+1]=(span.start+knot_times[1:]).tolist()
    for i in range(first,last):track.spans[i]=span
    for i in range(first,last+1):track.tangents.pop(i,None)


def retime_tail(track,first,last,max_velocity=.8,max_acceleration=2.,phase='transit'):
    import toppra as ta
    import toppra.constraint as constraint
    import toppra.algorithm as algorithm
    if last<=first:return
    if last!=len(track.times)-1:raise ValueError('Only an unscheduled tail can be retimed')
    q=np.asarray(track.values[first:last+1]);d=np.diff(q,axis=0)
    vmax=np.broadcast_to(max_velocity,q.shape[1]);amax=np.broadcast_to(max_acceleration,q.shape[1])
    if not np.isfinite(q).all() or not np.isfinite(vmax).all() or not np.isfinite(amax).all() or np.any(vmax<=0) or np.any(amax<=0):
        raise ValueError('Motion limits and waypoints must be finite; limits must be positive')
    if len(q)==2 and np.max(abs(d))<1e-5:
        # Tiny IK corrections make path-speed variables ill-conditioned.
        # Preserve the endpoint with an analytically bounded short quintic.
        delta=d[0];h=max(.015,1.875*np.max(abs(delta)/vmax),np.sqrt(5.774*np.max(abs(delta)/amax)))
        zero=np.zeros_like(delta)
        spline=PPoly(np.array([6*delta/h**5,-15*delta/h**4,10*delta/h**3,zero,zero,q[0]])[:,None,:],[0.,h])
        install(track,first,last,spline,np.array([0.,h]),vmax,amax,phase)
        return
    length=np.maximum(np.linalg.norm(d,axis=1),1e-6)
    s=np.r_[0.,np.cumsum(length)];s/=s[-1]
    path=ta.SplineInterpolator(s,q,bc_type='natural')
    extra=np.r_[np.linspace(0.,1.,max(80,len(s)*3)),(s[:-1,None]+np.diff(s)[:,None]*np.linspace(0,1,9)[None,1:-1]).ravel()]
    extra=extra[np.min(abs(extra[:,None]-s[None,:]),axis=1)>1e-8]
    extra=np.unique(extra)
    extra=extra[np.r_[True,np.diff(extra)>1e-8]]
    grid=np.sort(np.r_[s,extra])
    instance=algorithm.TOPPRA([constraint.JointVelocityConstraint(vmax),constraint.JointAccelerationConstraint(amax)],path,gridpoints=grid,parametrizer='ParametrizeConstAccel')
    trajectory=instance.compute_trajectory(0.,0.)
    if trajectory is None:raise ValueError('No feasible continuous motion timing')
    speed=instance.problem_data.sd_vec
    h=2*np.diff(grid)/(speed[:-1]+speed[1:]);t=np.r_[0.,np.cumsum(h)]
    acceleration=np.diff(speed)/h
    spatial=path.cspl
    indices=np.minimum(np.searchsorted(spatial.x,grid[:-1],side='right')-1,len(spatial.x)-2)
    local=grid[:-1]-spatial.x[indices]
    # Exact composition q(s(t)): cubic spatial path and constant path
    # acceleration give degree-six joint polynomials. No time-domain refit.
    powers=[np.vstack([np.ones(len(h)),np.zeros((6,len(h)))])]
    base=np.array([local,speed[:-1],acceleration/2])
    for degree in range(1,4):
        value=np.zeros((7,len(h)))
        for i in range(7):
            for j in range(3):
                if i+j<7:value[i+j]+=powers[-1][i]*base[j]
        powers.append(value)
    coefficients=np.zeros((7,len(h),q.shape[1]))
    for degree in range(4):coefficients+=powers[degree][:,:,None]*spatial.c[3-degree,indices][None,:,:]
    spline=PPoly(coefficients[::-1],t)
    vl,vh=bounds(spline,1);al,ah=bounds(spline,2)
    scale=max(1.,np.max(np.maximum(abs(vl),abs(vh))/vmax),np.sqrt(np.max(np.maximum(abs(al),abs(ah))/amax)))*1.002
    spline=PPoly(spline.c/scale**np.arange(6,-1,-1)[:,None,None],t*scale)
    knot_times=t[np.searchsorted(grid,s)]*scale
    install(track,first,last,spline,knot_times,vmax,amax,phase,scale)
