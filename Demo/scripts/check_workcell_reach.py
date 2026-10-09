#!/usr/bin/env python3
"""Position-only URDF IK sanity check, NOT collision-free grasp feasibility."""
import json
import math
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from arm_sim.common import ROOT, load_models
from arm_sim.workcell import config, slot_centers

def chain(model,tip):
    root=ET.parse(ROOT/model['urdf']).getroot()
    joints={j.find('child').get('link'):j for j in root.findall('joint')}
    result=[]
    while tip in joints:
        j=joints[tip];result.append(j);tip=j.find('parent').get('link')
    return list(reversed(result))

def main():
    cfg=config(); models=load_models(); results=[]; rng=np.random.default_rng(7)
    for kind,model in models.items():
        js=chain(model,'gripper_frame_link' if kind=='so101' else 'end_effector_link')
        terms=[]
        for j in js:
            o=j.find('origin'); mat=np.eye(4)
            if o is not None:
                mat[:3,3]=[float(v) for v in o.get('xyz','0 0 0').split()]
                mat[:3,:3]=Rotation.from_euler('xyz',[float(v) for v in o.get('rpy','0 0 0').split()]).as_matrix()
            axis=j.find('axis')
            terms.append((mat,model['joints'].index(j.get('name')) if j.get('type')!='fixed' else None,np.array([float(v) for v in axis.get('xyz').split()]) if axis is not None else None))
        base=np.eye(4);base[:3,3]=cfg['robots'][kind]['position'];base[:3,:3]=Rotation.from_euler('z',cfg['robots'][kind]['yaw']).as_matrix()
        def fk(q):
            T=base.copy()
            for mat,index,axis in terms:
                T=T@mat
                if index is not None:
                    r=np.eye(4);r[:3,:3]=Rotation.from_rotvec(axis*q[index]).as_matrix();T=T@r
            return T[:3,3]
        prep=cfg['preparation']; x,y=prep['center']
        targets=[(f'prep_{p}',[x,y+(i-1)*prep['pitch'],.735]) for i,p in enumerate(prep['parts'])]
        if kind=='omx':
            targets += [(f"{t['part']}_{i+1}",[*xy,.735]) for t in cfg['trays'] for i,xy in enumerate(slot_centers(t))]
        else:targets += [('jig',[*cfg['jig'],.735]),('finished',[*cfg['finished'],.735])]
        lo,hi=np.array(model['limits']).T
        for name,target in targets:
            best=None
            for _ in range(8):
                fit=least_squares(lambda q:fk(q)-target,rng.uniform(lo,hi),bounds=(lo,hi),max_nfev=150)
                error=float(np.linalg.norm(fit.fun))
                if best is None or error<best[0]:best=(error,fit.x)
                if error<.001:break
            results.append({'robot':kind,'target':name,'position':target,'position_error_m':best[0],'joints':best[1].tolist()})
    out={'scope':'Position only; orientation, approach path, joint self-collision and gripper fit NOT validated','targets':results}
    (ROOT/'outputs/reach_report.json').write_text(json.dumps(out,indent=2)+'\n')
    print('targets',len(results),'max position error m',max(r['position_error_m'] for r in results))
if __name__=='__main__':main()
