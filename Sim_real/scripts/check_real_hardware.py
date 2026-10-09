import json,pickle,hashlib
from arm_sim.common import ROOT
from scripts.check_conveyor_hardware import check
blob=(ROOT/'outputs/sim_real_demo/plan.pkl').read_bytes();plan=pickle.loads(blob)
r=check(plan.cfg,plan=plan);r['plan_sha256']=hashlib.sha256(blob).hexdigest()
(ROOT/'outputs/sim_real_demo/hardware_check.json').write_text(json.dumps(r,indent=2));print(json.dumps(r));raise SystemExit(r['status']!='pass')
