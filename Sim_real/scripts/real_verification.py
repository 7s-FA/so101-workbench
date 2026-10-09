"""Reject stale or incomplete geometry/native playback verification."""
import hashlib,json,sys
from arm_sim.common import ROOT
from scripts.build_real_plan import fingerprint
OUT=ROOT/'outputs/sim_real_demo'
REPORTS=['path_check.json','manipulation_check.json','payload_check.json','traffic_check.json','support_check.json','hardware_check.json','motion_quality_check.json','docking_check.json','rest_check.json']
def verifier_hash():
    files=[ROOT/'scripts/real_sim.py',ROOT/'scripts/real_workflow.py',ROOT/'scripts/real_markings.py',ROOT/'scripts/run_real_sim.py',ROOT/'scripts/real_demo.sh',ROOT/'scripts/check_conveyor_hardware.py',ROOT/'scripts/real_verification.py',*sorted((ROOT/'scripts').glob('check_real_*.py'))]
    h=hashlib.sha256()
    for p in files:h.update(p.read_bytes())
    return h.hexdigest()
def verify():
    key=hashlib.sha256((OUT/'plan.pkl').read_bytes()).hexdigest()
    if json.loads((OUT/'plan_cache.json').read_text())['sha256']!=fingerprint():raise ValueError('동작 계획이 현재 모델과 다릅니다.')
    reports={name:json.loads((OUT/name).read_text()) for name in REPORTS}
    for name,r in reports.items():
        if r.get('status')!='pass' or r.get('plan_sha256')!=key:raise ValueError(name+' 검사 미통과 또는 이전 모델의 결과입니다.')
    native=json.loads((OUT/'verified/runtime_report.json').read_text())
    if native.get('status')!='complete' or not native.get('completed') or native.get('plan_sha256')!=key:raise ValueError('현재 모델의 전체 Isaac Sim 재생 확인이 필요합니다.')
    return {'status':'pass','plan_sha256':key,'verifier_sha256':verifier_hash(),'native_completed':True,'duration_s':native['duration_s'],'physical_grasp_verified':False,'reports':REPORTS}
if __name__=='__main__':
    try:
        result=verify()
        if '--write' in sys.argv:
            from datetime import datetime,timezone
            result['verified_at']=datetime.now(timezone.utc).isoformat();(OUT/'verification.json').write_text(json.dumps(result,indent=2))
        else:
            saved=json.loads((OUT/'verification.json').read_text())
            if any(saved.get(k)!=result[k] for k in ('plan_sha256','verifier_sha256')):raise ValueError('검사 코드가 변경되어 다시 확인해야 합니다.')
        print('현재 모델의 동작·간섭·전체 재생 검사 확인 완료')
    except (ValueError,KeyError,OSError) as error:
        print(str(error)+'\nfinal 실행 후 ./Sim_real/scripts/verify_real_demo.sh 로 검사하세요.',file=sys.stderr);raise SystemExit(1)
