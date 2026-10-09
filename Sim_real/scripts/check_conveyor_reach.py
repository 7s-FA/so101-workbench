#!/usr/bin/env python3
"""Position-only IK at handling stations; does not certify a grasp or path."""
import json
import math
import numpy as np
from arm_sim.common import ROOT
from arm_sim.conveyor_scene import preview_layout
from arm_sim.demo_kinematics import ArmKinematics


def targets(cfg):
    z=cfg['table_height'];result={name:[] for name in cfg['arms']}
    for bank,(cx,cy) in enumerate(cfg['source_banks']):
        if bank not in cfg['initial_materials']['source_loaded_banks']:continue
        for col in range(3):
            for row in range(3):result['arm_a'].append((f'source_{bank}_{col}_{row}',[cx+(col-1)*.075,cy+(row-1)*.075,z+.04]))
    for arm,(x,y,yaw) in [('arm_a',[*cfg['loading_dock'],-math.pi/2]),('arm_b',[*cfg['unload_dock'],-math.pi/2])]:
        r=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
        for col in range(3):
            for row in range(3):
                xy=np.array([x,y])+r@[-.032+(col-1)*.075,(row-1)*.075]
                result[arm].append((f'cargo_{col}_{row}',[*xy,cfg['burger']['deck_top']+.04]))
    for slot in cfg['preparation_slots']:
        for arm in ('arm_b','arm_c'):
            result[arm].append((f"prep_{slot['kit']}_{slot['kind']}",[*slot['position'],z+.04]))
    result['arm_c'].append(('assembly',[*cfg['assembly_jig'],z+.04]))
    for arm in ('arm_c','arm_d'):result[arm].append(('handoff',[*cfg['handoff'],z+.04]))
    for i in range(3):result['arm_d'].append((f'finished_{i}',[cfg['shipping_dock'][0]+(i-1)*.078,cfg['shipping_dock'][1],cfg['shipping_height']+.04]))
    return result


def check():
    cfg=preview_layout();models=json.loads((ROOT/'assets/generated/conveyor_demo/models.json').read_text());failures=[];report={}
    for arm,points in targets(cfg).items():
        kin=ArmKinematics(models[cfg['arms'][arm]['model']],cfg['arms'][arm],cfg['table_height']);rows=[]
        for name,point in points:
            for dz in (0,.04):
                target=np.array(point)+[0,0,dz]
                try:
                    q=kin.solve(target,cfg['rest_joints'][arm]);error=float(np.linalg.norm(kin.position(q)-target))
                    rows.append({'station':name,'approach_m':dz,'error_m':error,'joints':q.tolist(),'horizontal_reach_m':float(np.linalg.norm(target[:2]-cfg['arms'][arm]['position']))})
                except ValueError as exc:failures.append({'arm':arm,'station':name,'approach_m':dz,'error':str(exc)})
        report[arm]=rows
    return {'status':'pass' if not failures else 'blocked','failures':failures,'samples':sum(map(len,report.values())),
            'scope':'position-only IK at part-level and 40mm-above station proxies',
            'exclusions':['tool orientation','collision along paths','part-specific grasp transform','motor torque or temperature'],'arms':report}

if __name__=='__main__':
    r=check();(ROOT/'outputs/conveyor_preview_latest/reach_check.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:r[k] for k in ('status','samples','failures')},ensure_ascii=False));raise SystemExit(r['status']!='pass')
