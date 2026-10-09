"""Feedback-gated shared-zone handover for the simulated workcell.

The central scheduler is authoritative. A phase boundary cannot be skipped and
its completion is acknowledged only from fresh simulator joint feedback.
"""
import numpy as np


class WorkcellGate:
    def __init__(self,plan):
        self.plan=plan
        self.boundaries=sorted({t[k] for t in plan.tasks for k in ('start','end')})
        self.reset()

    def reset(self):
        self.completed=set();self.hold_reason=None;self.acknowledgements=0

    def advance(self,t,end,measured):
        self.hold_reason=None
        for task in self.plan.tasks:
            if task['end']>t+1e-8 or task['id'] in self.completed:continue
            name=task['arm'];q=measured.get(name)
            home=np.asarray(task.get('safe_joints',self.plan.ik[name].model['home']))
            if q is None or np.shape(q)!=home.shape or not np.isfinite(q).all() or np.max(np.abs(q[:5]-home[:5]))>.01:
                self.hold_reason=f'{name.upper()} 안전 복귀 응답 대기 · {task["id"]}'
                return t
            self.completed.add(task['id']);self.acknowledgements+=1
        for boundary in self.boundaries:
            if t+1e-8<boundary<end:return boundary
        return end
