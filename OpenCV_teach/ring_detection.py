"""Rectangular frames with visible internal holes; optional STL hole evidence."""
import cv2
import numpy as np
from carrier_detection import Candidate,order_quad,refine_straight_edges,border_stats,_iou,propose_quads,line_options,variants_fast


def evidence(frame,corners,profile):
    # Warp with padding so the outside boundary remains measurable.
    w=440;h=max(60,int(w*profile['short_mm']/profile['long_mm']));pad=8
    dst=np.float32([[pad,pad],[pad+w,pad],[pad+w,pad+h],[pad,pad+h]])
    warp=cv2.warpPerspective(frame,cv2.getPerspectiveTransform(order_quad(corners),dst),(w+2*pad+1,h+2*pad+1))
    gray=cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(cv2.GaussianBlur(gray,(3,3),0),10,35)
    near=cv2.dilate(edges,np.ones((5,5),np.uint8))>0
    outer=[near[pad-2:pad+3,pad+8:pad+w-8].any(0).mean(),
           near[pad+h-2:pad+h+3,pad+8:pad+w-8].any(0).mean(),
           near[pad+8:pad+h-8,pad-2:pad+3].any(1).mean(),
           near[pad+8:pad+h-8,pad+w-2:pad+w+3].any(1).mean()]
    holes=profile.get('hole_layout',[])
    scores=[]
    if holes:
        # Four vertex orderings can differ by 180 degrees, never infer front/back.
        rotations=range(4) if profile["long_mm"]/profile["short_mm"]<1.15 else (0,2)
        for rotation in rotations:
            per_hole=[]
            for hole in holes:
                pts=np.asarray(hole,float)
                if abs(cv2.contourArea(pts.astype(np.float32)))<.025:continue
                for _ in range(rotation):pts=np.column_stack((1-pts[:,1],pts[:,0]))
                mask=np.zeros_like(edges)
                cv2.polylines(mask,[np.rint(pts*[w,h]+pad).astype(np.int32)],True,255,1)
                per_hole.append(float(near[mask>0].mean()))
            # A second visible hole prevents a part on a larger white board
            # from matching a single accidental interior contour.
            ordered=sorted(per_hole,reverse=True)
            scores.append(ordered[min(1,len(ordered)-1)] if ordered else 0.)
        inner=max(scores,default=0.)
    else:
        contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        inner=0.
        for c in contours:
            x,y,cw,ch=cv2.boundingRect(c);area=abs(cv2.contourArea(c))
            if x>pad+4 and y>pad+4 and x+cw<pad+w-4 and y+ch<pad+h-4 and .01*w*h<area<.85*w*h:
                inner=1.;break
    return float(inner),tuple(float(x) for x in outer)


def detect_rings(frame,profile,roi,intrinsics=None,station=None):
    from jig_detection import outline_candidates
    appearance={'edge':'contrast','white':'white_frame'}.get(profile['detection_method'],profile['appearance'])
    found=[];x,y,x2,y2=roi
    proposals=outline_candidates(frame,roi,appearance)+propose_quads(frame,profile["detection_method"])
    ranked=[]
    for c in proposals:
        c=order_quad(c)
        if not ((c[:,0]>x+2).all() and (c[:,0]<x2-2).all() and (c[:,1]>y+2).all() and (c[:,1]<y2-2).all()):continue
        inner,outer=evidence(frame,c,profile)
        ranked.append((inner+min(outer),c))
    seeds=[]
    for score,c in sorted(ranked,key=lambda item:item[0],reverse=True):
        if sum(_iou(c,old)>.7 for old in seeds)>=2:continue
        seeds.append(c)
        if len(seeds)>=8:break
    lines=line_options(frame)
    proposals=[q for c in seeds for q in variants_fast(c,lines)]
    for c in proposals:
        c=order_quad(c)
        if not ((c[:,0]>=x).all() and (c[:,0]<x2).all() and (c[:,1]>=y).all() and (c[:,1]<y2).all()):continue
        c=order_quad(c);length=np.linalg.norm(np.roll(c,-1,axis=0)-c,axis=1)
        ratio=(length[0]+length[2])/(length[1]+length[3])
        if abs(ratio-profile['long_mm']/profile['short_mm'])/(profile['long_mm']/profile['short_mm'])>.3:continue
        inner,outer=evidence(frame,c,profile)
        if inner<.55 or min(outer)<.55:continue
        refined,ok=refine_straight_edges(frame,c,intrinsics)
        if ok and (refined[:,0]>=x).all() and (refined[:,0]<x2).all() and (refined[:,1]>=y).all() and (refined[:,1]<y2).all():
            score2,outer2=evidence(frame,refined,profile)
            if score2>=inner-.05 and min(outer2)>=min(outer)-.05:c,inner,outer=refined,score2,outer2
        corners_visible,_=border_stats(frame,c)
        if min(corners_visible)<.25:continue
        precise=min(outer)>=.85 and inner>=.75 and min(corners_visible)>=.65
        score=(inner+min(outer))/2
        if station and station.get('intrinsics') and station.get('extrinsics'):
            from jig_detection import rectangle_pose
            try:metric=rectangle_pose(c,profile,station,list(frame.shape[1::-1]),enforce_heading=False)
            except ValueError:continue
            score*=1-.3*metric['size_error_pct']/profile['size_tolerance_pct']
        found.append(Candidate(c,score,precise,edge_support=outer,tracking_ok=True,
                               reason='STL 내부 구멍·외곽 일치' if profile.get('hole_layout') else '내부 구멍·외곽 확인'))
    result=[]
    for candidate in sorted(found,key=lambda c:c.score,reverse=True):
        if any(_iou(candidate.corners,old.corners)>.72 for old in result):continue
        result.append(candidate)
    return result
