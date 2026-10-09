#!/usr/bin/env python3
"""Build an authenticated local cache; cached data is never accepted from HTTP."""
import hashlib,json,pickle
import xml.etree.ElementTree as ET
from arm_sim.common import ROOT
from arm_sim.real_demo import RealPlan

def fingerprint():
    files=sorted([* (ROOT/'arm_sim').glob('*.py'),*(ROOT/'config').glob('*.json'),*(ROOT/'assets/generated/conveyor_demo').glob('*.urdf'),ROOT/'assets/generated/conveyor_demo/models.json'])
    files+=list((ROOT/'assets/generated/conveyor_demo/parts').glob('*.stl'))+[ROOT/'assets/product/02_ball_d30.stl',ROOT/'assets/factory/realsense/d435.dae']
    models=json.loads((ROOT/'assets/generated/conveyor_demo/models.json').read_text())
    for model in models.values():
        path=ROOT/model['urdf'];files.append(path)
        for mesh in ET.parse(path).iter('mesh'):
            from pathlib import Path
            files.append(Path(mesh.get('filename')))
    files+=list((ROOT/'vendor/so_arm100/Optional').glob('**/*.stl'))
    h=hashlib.sha256()
    for p in sorted(set(files)):h.update(str(p.relative_to(ROOT)).encode());h.update(p.read_bytes())
    return h.hexdigest()
if __name__=='__main__':
    out=ROOT/'outputs/sim_real_demo';out.mkdir(parents=True,exist_ok=True);key=fingerprint();meta=out/'plan_cache.json'
    if meta.exists() and json.loads(meta.read_text()).get('sha256')==key and (out/'plan.pkl').exists():print('Current plan cache ready')
    else:
        p=RealPlan()
        for task in p.tasks:task['label']=task['label'].replace('지게차 직접 상차','작업대 용기에 적재')
        p.activities=[(a,b,l.replace('지게차 직접 상차','작업대 용기에 적재')) for a,b,l in p.activities]
        (out/'plan.pkl').write_bytes(pickle.dumps(p));meta.write_text(json.dumps({'sha256':key,'summary':p.validate()},indent=2,ensure_ascii=False));print(json.dumps(p.validate(),ensure_ascii=False))
