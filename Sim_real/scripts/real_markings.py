"""Render-only centre-line intervals; no changes to vehicle routes or timing."""
import math
from collections import defaultdict

def centre_dashes(cfg, dock_bars, exclusions):
    groups=defaultdict(list)
    for route in cfg['traffic']['routes']:
        family='material' if route['name']=='MaterialU' else 'forklift'
        for a,b in zip(route['points'],route['points'][1:]):
            axis=0 if abs(b[0]-a[0])>1e-8 else 1
            if abs(b[1-axis]-a[1-axis])>1e-8:raise ValueError('Floor markings require orthogonal routes')
            fixed=round(a[1-axis],8);lo,hi=sorted((a[axis],b[axis]))
            if axis==1:
                stops=[y-.03 for x,y in dock_bars if abs(x-fixed)<1e-7]
                if stops:hi=min(hi,min(stops))
            if hi>lo:groups[axis,fixed].append((lo,hi,family))
    result=[]
    for (axis,fixed),segments in groups.items():
        edges=sorted({v for a,b,_ in segments for v in (a,b)});runs=[]
        for lo,hi in zip(edges,edges[1:]):
            mid=(lo+hi)/2;families={name for a,b,name in segments if a<=mid<=b}
            if not families:continue
            family='shared' if len(families)>1 else next(iter(families))
            if runs and runs[-1][2]==family and abs(runs[-1][1]-lo)<1e-8:runs[-1]=(runs[-1][0],hi,family)
            else:runs.append((lo,hi,family))
        anchor=cfg['loading_dock'][0] if axis==0 else cfg['vehicle_lane_y']
        for lo,hi,family in runs:
            for index in range(math.floor((lo-anchor)/.12)-1,math.ceil((hi-anchor)/.12)+1):
                start=max(lo,anchor+index*.12+.025);end=min(hi,anchor+index*.12+.070)
                if end-start<.010:continue
                a=[fixed,fixed];b=[fixed,fixed];a[axis]=start;b[axis]=end
                # Omit a complete dash near paint boundaries rather than leave fragments.
                low=[min(a[i],b[i])-.003 for i in (0,1)];high=[max(a[i],b[i])+.003 for i in (0,1)]
                if any(all(high[i]>=left[i] and right[i]>=low[i] for i in (0,1)) for left,right in exclusions):continue
                result.append((a,b,family))
    return result
