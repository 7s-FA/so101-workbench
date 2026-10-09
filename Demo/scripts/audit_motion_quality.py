#!/usr/bin/env python3
"""Audit polynomial motion, not physical grip success. Local trusted plans only."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from numpy.polynomial import Polynomial


def extrema(poly):
    roots=poly.deriv().roots()
    points=[0.,1.,*[float(x.real) for x in roots if abs(x.imag)<1e-8 and 0<x.real<1]]
    values=poly(np.asarray(points))
    return float(np.min(values)),float(np.max(values))


def audit(plan):
    result={'status':'clear','duration_s':plan.duration,'arms':{}}
    for name,tr in plan.arms.items():
        vmax=amax=overshoot=0.;vratio=aratio=0.;transit_stops=aligned=0
        checked=set();phases={};limits=np.asarray(plan.ik[name].model['limits'])
        for i,(start,end) in enumerate(zip(tr.times[:-1],tr.times[1:])):
            span=getattr(tr,'spans',{}).get(i)
            if span is not None:
                if id(span) in checked:continue
                checked.add(id(span))
                phases[span.phase]=phases.get(span.phase,0.)+span.duration
                from scipy.interpolate import PPoly
                for j in range(span.spline.c.shape[2]):
                    poly=PPoly(span.spline.c[:,:,j],span.spline.x)
                    for order in (0,1,2):
                        p=poly.derivative(order);roots=p.derivative().roots(extrapolate=False)
                        points=np.r_[p.x,np.nextafter(p.x[1:],-np.inf),roots[np.isfinite(roots)]];values=p(points)
                        lo=float(min(values));hi=float(max(values));peak=max(abs(lo),abs(hi))
                        if order==0:
                            if lo<limits[j,0]-1e-6 or hi>limits[j,1]+1e-6:result['status']='blocked'
                        elif order==1:vmax=max(vmax,peak);vratio=max(vratio,peak/span.velocity_limits[j])
                        else:amax=max(amax,peak);aratio=max(aratio,peak/span.acceleration_limits[j])
                    if abs(poly(0.,1))>1e-6 or abs(poly(span.duration,1))>1e-6:result['status']='blocked'
                continue
            h=end-start;q0=tr.values[i];q1=tr.values[i+1];d=q1-q0
            a=h*getattr(tr,'tangents',{}).get(i,np.zeros_like(q0))
            b=h*getattr(tr,'tangents',{}).get(i+1,np.zeros_like(q0))
            coefficients=np.array([q0,a,np.zeros_like(q0),10*d-6*a-4*b,-15*d+8*a+7*b,6*d-3*a-3*b])
            held=any(plan.entry(hist,(start+end)/2)[1]==name for hist in plan.parts.values())
            profile=plan.motion_profiles['loaded' if held else 'empty']
            for joint in range(len(q0)):
                poly=Polynomial(coefficients[:,joint]);v=poly.deriv()/h;acc=v.deriv()/h
                vp=max(map(abs,extrema(v)));ap=max(map(abs,extrema(acc)))
                vmax=max(vmax,vp);amax=max(amax,ap)
                cap=plan.motion_profiles['gripper'] if joint>=5 else profile
                vratio=max(vratio,vp/cap['velocity_rad_s']);aratio=max(aratio,ap/cap['acceleration_rad_s2'])
                lo,hi=extrema(poly)
                overshoot=max(overshoot,min(q0[joint],q1[joint])-lo,hi-max(q0[joint],q1[joint]))
            if i>0:
                before=tr.values[i][:5]-tr.values[i-1][:5];after=d[:5]
                if np.linalg.norm(before)>1e-5 and np.linalg.norm(after)>1e-5 and np.dot(before,after)/(np.linalg.norm(before)*np.linalg.norm(after))>.98:
                    aligned+=1;transit_stops+=int(np.linalg.norm(getattr(tr,'tangents',{}).get(i,np.zeros_like(q0))[:5])<1e-6)
        result['arms'][name]={'peak_velocity_rad_s':vmax,'peak_acceleration_rad_s2':amax,'joint_endpoint_overshoot_rad':overshoot,
                              'aligned_knots':aligned,'aligned_rest_stops':transit_stops,'velocity_limit_ratio':vratio,'acceleration_limit_ratio':aratio,'phase_seconds':phases}
        if vratio>1.00001 or aratio>1.0001 or overshoot>1e-8:result['status']='blocked'
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with args.plan.open('rb') as source:plan=pickle.load(source)
    result=audit(plan);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['status']=='clear' else 1)
