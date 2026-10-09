"""Short, request-scoped measurement for a stationary teaching workcell."""
import time
import numpy as np
from vision_teaching import StableJigs


class QuickJigs(StableJigs):
    """Keep at most 3 s; no continuous-window or precise-count prerequisite."""
    def __init__(self, **kwargs):
        super().__init__(position_tolerance=3., **kwargs)

    def assessment(self, key, now=None, *, final=False):
        now=time.monotonic() if now is None else now
        hist=list(self.history.get(key,()))
        h=[v for v in hist if now-v['at']<=self.duration]
        result={'seconds':self.progress(key,now),'ready':False,'precise':0,'rejected':0,
                'missing':len(self.failures.get(key,())),'total':len(h),'value':None,
                'reasons':['새 지그 좌표 대기'],'inlier_fraction':0.}
        if not h or now-h[-1]['at']>self.max_gap:return result
        failures=self.failures.get(key,())
        if failures and failures[-1][0]>=h[-1]['at']:
            result['reasons']=['현재 지그 미검출'];return result
        xy,angles=self._statistics(h)
        mask=(np.linalg.norm(xy-np.median(xy,axis=0),axis=1)<=3.)&(np.abs(angles-np.median(angles))<=2.5)
        accepted=[v for v,keep in zip(h,mask) if keep]
        counts={q:sum(v.get('quality','precise')==q for v in accepted) for q in ('precise','medium')}
        result.update(precise=counts['precise'],rejected=len(h)-len(accepted),inlier_fraction=len(accepted)/len(h))
        ready=(len(accepted)>=2 and counts['precise']>=1) or (final and (counts['precise']>=1 or len(accepted)>=2))
        if not ready or not mask[-1]:
            result['reasons']=['유효 측정 추가 수집 중' if mask[-1] else '현재 위치·각도가 다른 측정과 다름'];return result
        weights=np.array([(3. if v.get('quality','precise')=='precise' else 1.)*
                          float(np.clip(v.get('detection_confidence',1.),.1,1.))**2 for v in accepted])
        centre=np.average(xy[mask],axis=0,weights=weights)
        angle=float(np.average(angles[mask],weights=weights))
        result.update(ready=True,reasons=[],value={
            'x_mm':float(centre[0]),'y_mm':float(centre[1]),'yaw_deg':float(self._delta(angle,0.)),
            'at':accepted[-1]['at'],'profile_signature':h[-1].get('profile_signature'),
            'stability_seconds':result['seconds'],'sample_count':len(accepted),'quality_counts':counts,
            'rejected_count':result['rejected'],'total_samples':len(h),'missing_count':result['missing'],
            'estimator':'quick_quality_weighted_mean','deadline_fallback':bool(final),
            'quality_weight_sums':{q:float(sum(w for v,w in zip(accepted,weights) if v.get('quality','precise')==q)) for q in counts}})
        return result

    def snapshot(self, keys, now=None, *, final=False):
        result={}
        for key in keys:
            a=self.assessment(key,now,final=final)
            if not a['ready']:raise ValueError(key+': '+' / '.join(a['reasons']))
            result[key]=a['value']
        return result

    def status_text(self,key,now=None):
        a=self.assessment(key,now)
        return f"빠른 측정 · 정밀 {a['precise']}회 · 유효 {a['total']-a['rejected']}개 · 최대 3초"
