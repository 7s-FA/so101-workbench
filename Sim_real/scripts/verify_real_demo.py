"""Finite validation, leaves no local monitor running."""
import concurrent.futures,os,subprocess,sys,tempfile
from pathlib import Path
from arm_sim.common import ROOT
out=ROOT/'outputs/sim_real_demo';out.mkdir(parents=True,exist_ok=True)
subprocess.run([sys.executable,'scripts/build_real_plan.py'],cwd=ROOT,check=True)
with (out/'unit_tests.log').open('w') as log:
    subprocess.run([sys.executable,'-m','unittest','tests.test_real_demo','tests.test_motion_profiles','tests.test_commands','tests.test_traffic','tests.test_real_console','tests.test_camera_resolution','tests.test_real_workflow'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
names=['paths','manipulation','traffic','payloads','supports','hardware','motion_quality','docking','rest']
def run(name):
    with (out/f'check_real_{name}.log').open('w') as log:
        result=subprocess.run([sys.executable,f'scripts/check_real_{name}.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    print(name+(' 통과' if result.returncode==0 else ' 미통과'),flush=True)
    return result.returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:codes=list(pool.map(run,names))
if any(codes):raise SystemExit('간섭 검사 미통과. 검사 로그를 확인하세요.')
with tempfile.TemporaryDirectory(prefix='sim-real-verification-') as temp:
    marker=Path(temp)/'complete';env={**os.environ,'REAL_STATUS_FILE':str(marker)}
    with (out/'native_verify.log').open('w') as log:
        native=subprocess.run(['/path/to/isaac/env_isaaclab/bin/python','scripts/real_sim.py','--headless','--verify','--output',str(out/'verified')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if native.returncode or not marker.exists() or marker.read_text()!='ok':raise SystemExit('Isaac Sim 전체 재생 확인 실패')
subprocess.run([sys.executable,'scripts/real_verification.py','--write'],cwd=ROOT,check=True)
