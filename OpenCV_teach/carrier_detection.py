"""Appearance proposals and 3x2 carrier validation, separate from robot coordinates.

No biggest-rectangle fallback: plain white support boards lack the carrier's
separator layout. Weak/occluded outlines are inspection hypotheses only.
"""
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class Candidate:
    corners: np.ndarray
    score: float
    precision_ok: bool
    grid_support: tuple = ()
    edge_support: tuple = ()
    reason: str = ''
    tracking_ok: bool = False

    @property
    def quality(self):
        return 'precise' if self.precision_ok else 'medium' if self.tracking_ok else 'low'

def order_quad(c):
 c=np.array(c,np.float32).reshape(4,2)
 if cv2.contourArea(c,oriented=True)<0:c=c[::-1]
 lengths=np.linalg.norm(np.roll(c,-1,axis=0)-c,axis=1)
 if lengths[0]+lengths[2]<lengths[1]+lengths[3]:c=np.roll(c,1,axis=0)
 return c

def propose_quads(frame, method="hybrid"):
 gray=cv2.GaussianBlur(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(3,3),0)
 lab=cv2.cvtColor(frame,cv2.COLOR_BGR2LAB);sat=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)[:,:,1]
 masks=[]
 for b in (range(122,135) if method != "edge" else ()):
  mask=cv2.inRange(lab,np.array([125,112,95]),np.array([255,145,b]));mask[sat>70]=0
  mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
  masks.append(cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8)))
 for lo,hi in ([(8,24),(15,45),(25,75)] if method != "white" else []):
  edge=cv2.Canny(gray,lo,hi)
  for size in [3,7,11]:masks.append(cv2.morphologyEx(edge,cv2.MORPH_CLOSE,np.ones((size,size),np.uint8)))
 cand=[]
 for mask in masks:
  contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
  for contour in sorted(contours,key=cv2.contourArea,reverse=True)[:120]:
   area=cv2.contourArea(contour)
   if not 5000<area<frame.shape[0]*frame.shape[1]*.65:continue
   for c in [contour,cv2.convexHull(contour)]:
    for eps in [.01,.015,.02,.025,.035]:
     poly=cv2.approxPolyDP(c,eps*cv2.arcLength(c,True),True)
     if len(poly)!=4 or not cv2.isContourConvex(poly):continue
     pts=order_quad(poly)
     if np.min(pts)<3 or np.max(pts[:,0])>frame.shape[1]-4 or np.max(pts[:,1])>frame.shape[0]-4:continue
     length=np.linalg.norm(np.roll(pts,-1,axis=0)-pts,axis=1);ratio=(length[0]+length[2])/(length[1]+length[3])
     if not 1.1<ratio<2.2 or min(length)<55:continue
     if any(np.min([np.max(np.linalg.norm(pts-np.roll(old,k,axis=0),axis=1)) for k in range(4)])<4 for old in cand):continue
     cand.append(pts)
 return cand

def evidence(frame,c):
 W,H,P=264,178,10
 dst=np.float32([[P,P],[P+W,P],[P+W,P+H],[P,P+H]])
 warp=cv2.warpPerspective(frame,cv2.getPerspectiveTransform(c,dst),(W+2*P+1,H+2*P+1))
 g=cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)
 edges=cv2.Canny(cv2.GaussianBlur(g,(3,3),0),10,30)
 near=cv2.dilate(edges,np.ones((3,3),np.uint8))>0
 v=[]
 for x in [P+W/3,P+W*2/3]:
  region=near[P+8:P+H-8,int(x-W*.025):int(x+W*.025)+1]
  v.append(float(region.mean(0).max()))
 y=P+H/2;region=near[int(y-H*.035):int(y+H*.035)+1,P+8:P+W-8]
 v.append(float(region.mean(1).max()))
 outer=[float(near[P-2:P+3,P+5:P+W-5].any(0).mean()),float(near[P+H-2:P+H+3,P+5:P+W-5].any(0).mean()),float(near[P+5:P+H-5,P-2:P+3].any(1).mean()),float(near[P+5:P+H-5,P+W-2:P+W+3].any(1).mean())]
 # Look for the inner edge of a narrow frame, excluding the boundary itself.
 rim=[float(near[P+4:P+13,P+10:P+W-10].mean(1).max()),float(near[P+H-12:P+H-3,P+10:P+W-10].mean(1).max()),float(near[P+10:P+H-10,P+4:P+13].mean(0).max()),float(near[P+10:P+H-10,P+W-12:P+W-3].mean(0).max())]
 score=min(v)*.6+np.mean(v)*.2+np.mean(outer)*.1+np.mean(rim)*.1
 return dict(grid=v,outer=outer,rim=rim,score=float(score)),warp

def line_options(frame):
 gray=cv2.GaussianBlur(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(3,3),0)
 edges=cv2.Canny(gray,10,30)
 lines=cv2.HoughLinesP(edges,1,np.pi/360,30,minLineLength=40,maxLineGap=18)
 if lines is None:return np.empty((0,4))
 return lines[:,0].astype(float)

def variants_fast(c,lines):
 vectors=np.roll(c,-1,axis=0)-c;lens=np.linalg.norm(vectors,axis=1)
 opts=[]
 a,b=lines[:,:2],lines[:,2:];d=b-a;sizes=np.linalg.norm(d,axis=1);directions=d/np.maximum(sizes[:,None],1e-8);mids=(a+b)/2
 normals=np.column_stack((-directions[:,1],directions[:,0]));rhos=(normals*a).sum(1)
 for i,(start,vec,length) in enumerate(zip(c,vectors,lens)):
  v=vec/length;normal=np.array([-v[1],v[0]]);rho=normal@start
  choices=[(normal,rho)]
  dist=abs(mids@normal-rho)
  s,e=(a-start)@v,(b-start)@v;lo=np.minimum(s,e);hi=np.maximum(s,e)
  overlap=np.maximum(0,np.minimum(length,hi)-np.maximum(0,lo))
  valid=(sizes>=.22*length)&(abs(directions@v)>=.96)&(dist<=min(lens)*.15)&(overlap>=length*.35)
  scores=np.where(valid,overlap/length-.2*dist/(min(lens)*.15),-np.inf)
  if valid.any():j=int(np.argmax(scores));choices.append((normals[j],rhos[j]))
  opts.append(choices)
 configs={(0,0,0,0),*[tuple(int(i==j and len(opts[i])>1) for i in range(4)) for j in range(4)],tuple(len(o)-1 for o in opts)}
 result=[]
 for config in configs:
  quad=[]
  for i in range(4):
   l1,l2=opts[(i-1)%4][config[(i-1)%4]],opts[i][config[i]]
   mat=np.stack((l1[0],l2[0]))
   if abs(np.linalg.det(mat))<.15:break
   quad.append(np.linalg.solve(mat,[l1[1],l2[1]]))
  if len(quad)!=4:continue
  quad=np.array(quad,np.float32)
  if cv2.isContourConvex(quad) and np.max(np.linalg.norm(quad-c,axis=1))<min(lens)*.35:result.append(quad)
 return result

def border_stats(frame,c):
 W,H,P=264,178,10;dst=np.float32([[P,P],[P+W,P],[P+W,P+H],[P,P+H]])
 warp=cv2.warpPerspective(frame,cv2.getPerspectiveTransform(c,dst),(W+2*P+1,H+2*P+1));g=cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)
 near=cv2.dilate(cv2.Canny(cv2.GaussianBlur(g,(3,3),0),10,30),np.ones((3,3),np.uint8))>0
 profiles=[near[P-2:P+3,P:P+W].any(0),near[P:P+H,P+W-2:P+W+3].any(1),near[P+H-2:P+H+3,P:P+W].any(0)[::-1],near[P:P+H,P-2:P+3].any(1)[::-1]]
 corners=[]
 for i in range(4):
  a,b=profiles[i-1],profiles[i]
  corners.append(min(float(a[int(len(a)*.85):int(len(a)*.97)].mean()),float(b[int(len(b)*.03):int(len(b)*.15)].mean())))
 hsv=cv2.cvtColor(warp,cv2.COLOR_BGR2HSV);white=(hsv[:,:,1]<85)&(hsv[:,:,2]>120)
 wc=[]
 for x,y in [(P+5,P+5),(P+W-5,P+5),(P+W-5,P+H-5),(P+5,P+H-5)]:wc.append(float(white[y-2:y+3,x-2:x+3].mean()))
 return corners,wc


def _iou(a, b):
    inter, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    return float(inter/max(cv2.contourArea(a)+cv2.contourArea(b)-inter,1))


def image_pose(corners, intrinsics=None):
    """Project the rectangle's physical centre, not the mean of image corners."""
    c=order_quad(corners)
    k=d=None
    if intrinsics:
        k=np.asarray(intrinsics['K'],float);d=np.asarray(intrinsics['D'],float)
        c=cv2.undistortPoints(c.reshape(-1,1,2),k,d,P=k).reshape(4,2).astype(np.float32)
    h=cv2.getPerspectiveTransform(np.float32([[0,0],[1,0],[1,1],[0,1]]),c)
    p=cv2.perspectiveTransform(np.float32([[[.5,.5],[.3,.5],[.7,.5]]]),h)[0]
    if k is not None:
        rays=np.column_stack(((p[:,0]-k[0,2])/k[0,0],(p[:,1]-k[1,2])/k[1,1],np.ones(len(p))))
        p=cv2.projectPoints(rays,np.zeros(3),np.zeros(3),k,d)[0].reshape(-1,2)
    axis=(p[2]-p[1])*.5
    angle=float(np.degrees(np.arctan2(axis[1],axis[0]))%180)
    axis=np.array([np.cos(np.radians(angle)),np.sin(np.radians(angle))])*np.linalg.norm(axis)
    return p[0],angle,axis


def refine_straight_edges(frame, corners, intrinsics=None):
    """Fit measured nearby edge pixels; apply lens correction before line fitting."""
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    edge=cv2.Canny(cv2.GaussianBlur(gray,(3,3),0),10,30)
    yy,xx=np.nonzero(edge);points=np.column_stack((xx,yy)).astype(np.float32)
    k=d=None
    if intrinsics:
        k=np.asarray(intrinsics['K'],float);d=np.asarray(intrinsics['D'],float)
    lines=[]
    for start,end in zip(corners,np.roll(corners,-1,axis=0)):
        v=end-start;length=np.linalg.norm(v);v=v/max(length,1);n=np.array([-v[1],v[0]])
        offset=(points-start)@n;t=(points-start)@v
        mask=(abs(offset)<4)&(t>length*.08)&(t<length*.92)
        selected=points[mask]
        if len(selected)<max(15,length*.2):return corners,False
        # Keep the strongest narrow ridge instead of mixing adjacent parallel rims.
        offsets=offset[mask];bins=np.arange(-4,4.5,.5)
        hist,_=np.histogram(offsets,bins)
        peak=(bins[int(np.argmax(hist))]+bins[int(np.argmax(hist))+1])/2
        selected=selected[abs(offsets-peak)<1.25]
        if k is not None:selected=cv2.undistortPoints(selected.reshape(-1,1,2),k,d,P=k).reshape(-1,2)
        vx,vy,x,y=cv2.fitLine(selected,cv2.DIST_HUBER,0,.01,.01).ravel()
        normal=np.array([-vy,vx]);lines.append((normal,float(normal@[x,y])))
    output=[]
    for i in range(4):
        a,b=lines[i-1],lines[i];mat=np.stack((a[0],b[0]))
        if abs(np.linalg.det(mat))<.15:return corners,False
        output.append(np.linalg.solve(mat,[a[1],b[1]]))
    refined=np.array(output,np.float32)
    if k is not None:
        rays=np.column_stack(((refined[:,0]-k[0,2])/k[0,0],(refined[:,1]-k[1,2])/k[1,1],np.ones(4)))
        refined=cv2.projectPoints(rays,np.zeros(3),np.zeros(3),k,d)[0].reshape(4,2).astype(np.float32)
    if not cv2.isContourConvex(refined) or np.max(np.linalg.norm(refined-corners,axis=1))>8:
        return corners,False
    return order_quad(refined),True


def detect_carriers(frame, roi=None, intrinsics=None, method="hybrid"):
    """Return distinct carrier hypotheses, ranked by structure and edge evidence."""
    x,y,x2,y2=roi or (0,0,frame.shape[1],frame.shape[0])
    # Keep camera context for edge/contour proposals: cropping changes contour
    # area ranking and can replace the outside rim with an inside parallel edge.
    # ROI still strictly limits every accepted corner, including after refinement.
    crop=frame
    proposals=[c for c in propose_quads(crop, method)
               if (c[:,0]>x+3).all() and (c[:,0]<x2-4).all()
               and (c[:,1]>y+3).all() and (c[:,1]<y2-4).all()]
    if not proposals:return []
    ranked=sorted([(evidence(crop,c)[0]['score'],c) for c in proposals],key=lambda item:item[0],reverse=True)
    # Refine each spatial group, including multiple carriers. A single large board
    # must not consume the entire refinement budget or win simply by its area.
    seeds=[]
    for score,c in ranked:
        if score<.48:continue
        if sum(_iou(c,old)>.72 for old in seeds)>=3:continue
        seeds.append(c)
        if len(seeds)>=12:break
    lines=line_options(crop);found=[]
    for c in seeds:
        for quad in variants_fast(c,lines):
            ev,_=evidence(crop,quad)
            if min(ev['grid'])<.70 or ev['score']<.74:continue
            found.append((ev['score'],quad,ev))
    found.sort(key=lambda item:item[0],reverse=True)
    groups=[]
    for score,c,ev in found:
        if any(_iou(c,old[1])>.70 for old in groups):continue
        groups.append((score,c,ev))
    result=[]
    for score,c,ev in groups:
        raw=c.copy()
        refined,fit_ok=refine_straight_edges(frame,raw,intrinsics)
        # Do not accept a visually stronger fit if it breaks the six-cell geometry.
        new_ev,_=evidence(frame,refined)
        if min(new_ev['grid'])>=.70 and new_ev['score']>=score-.035:
            raw=refined;ev=new_ev;score=new_ev['score']
        else:fit_ok=False
        corner_edges,corner_white=border_stats(frame,raw)
        h,w=frame.shape[:2]
        in_roi=bool((raw[:,0]>x+3).all() and (raw[:,0]<x2-4).all() and (raw[:,1]>y+3).all() and (raw[:,1]<y2-4).all())
        clear=fit_ok and in_roi and min(corner_white)>=.8 and min(corner_edges)>=.65 and min(ev['outer'])>=.7
        # A supported but imperfect rim can contribute to temporal consensus.
        # Missing/occluded corners remain inspection-only even if stationary.
        tracking=in_roi and min(corner_white)>=.65 and min(corner_edges)>=.25 and min(ev['outer'])>=.55
        reason='' if clear else '테두리 일부가 가려지거나 경계가 약합니다. 팔·조명·배경을 확인하세요.'
        result.append(Candidate(raw,float(score),bool(clear),tuple(ev['grid']),tuple(ev['outer']),reason,bool(tracking)))
    return result
