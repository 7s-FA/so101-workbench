#!/usr/bin/env python3
"""Train a short ACT validation job on saved observations; never opens hardware.

The source episode labels/categories are preserved. The job is published to the
same local model list as a web-created job, with a clearly named test run.
"""
import argparse
import hashlib
import fcntl
import json
import math
import subprocess
import sys
import time
import uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from studio.config import IL_PYTHON
from studio.learning import Learning,parameters
from studio.policy import PolicyWorker
from studio.reference import ReferenceCodec
from studio.store import Store,atomic_json


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def wait_for(worker,predicate,seconds):
    deadline=time.monotonic()+seconds
    while not predicate():
        if worker.error:raise RuntimeError(worker.error)
        if time.monotonic()>=deadline:raise TimeoutError('Model loading or inference timed out')
        time.sleep(.005)


def run(root, ids, steps=100):
    store=Store(root)
    # Check before constructing Learning, whose startup recovery marks old jobs interrupted.
    if any(json.loads(p.read_text())['state'] in ('running','starting') for p in (store.root/'training').glob('*/job.json')):
        raise ValueError('Another training job is active')
    learning=Learning(store)
    params=parameters(dict(steps=steps,save_freq=max(1,steps//2),batch_size=8,device='cuda'))
    with store.edit_saved():
        records=[store.episode(i) for i in ids]
        first,rows=records[0]
        if any(m.get('synthetic') or not m.get('included') for m,_ in records):raise ValueError('Choose included real episodes')
        if any(m.get('fingerprint')!=first.get('fingerprint') or m.get('calibration')!=first.get('calibration') or m.get('lerobot_calibration')!=first.get('lerobot_calibration') for m,_ in records):raise ValueError('Calibration mismatch')
        if any(m.get('task_id')!=first.get('task_id') for m,_ in records):raise ValueError('Choose episodes from one task folder')
        dataset=(store.task_folder(first.get('task_id'))/'lerobot').resolve(strict=True)
        manifest=json.loads((dataset/'studio_manifest.json').read_text())
        indices={m['studio_id']:m for m in manifest['episodes']}
        for number in ids:
            if indices[number]['source_fingerprint']!=first['fingerprint']:raise ValueError('Dataset generation does not match source')
        # Hash actual source frames, metadata and timestamps before and after testing.
        hashes={}
        from PIL import Image
        for meta,samples in records:
            folder=store.path(meta['id'])
            for filename in ('metadata.json','samples.jsonl'):hashes[str(folder/filename)]=digest(folder/filename)
            for row in samples:
                for name,frame in row['cameras'].items():
                    path=folder/name/Path(frame['file']).name
                    with Image.open(path) as image:
                        image.verify()
                    hashes[str(path)]=digest(path)
        task_id=first.get('task_id')
        if any(meta.get('task_id')!=task_id for meta,_ in records):raise ValueError('Choose one task at a time')
        task=learning.tasks.get(task_id) if task_id else dict(id=None,name='미분류 · 학습 저장 시험')
        job_id=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
        directory=learning.root/job_id;directory.mkdir()
        job=dict(id=job_id,name=f"학습·저장 테스트 · 기록 {','.join(f'{n:04d}' for n in ids)} · {steps}회",
            task=task,params=params,state='starting',created_at=time.time(),dataset=str(dataset),
            repo_id=manifest['repo_id'],episodes=[indices[n]['episode_index'] for n in ids],studio_ids=ids,
            fingerprint=first['fingerprint'],calibration=first['calibration'],lerobot_calibration=first['lerobot_calibration'],
            camera_shapes={k:v['shape'] for k,v in rows[0]['cameras'].items()},fps=first.get('requested_fps',30),
            start_pose=rows[0]['action'],synthetic=False,message='저장된 실물 기록으로 학습·저장 시험 준비 중',
            purpose='training_storage_validation',physical_success_validated=False)
        atomic_json(directory/'job.json',job)
        atomic_json(directory/'source_hashes.json',hashes)
    print(json.dumps(dict(job_id=job_id,path=str(directory),episodes=ids,frames=sum(len(r) for _,r in records)),ensure_ascii=False),flush=True)
    with (directory/'train.log').open('w') as log:
        process=subprocess.Popen([str(IL_PYTHON),str(ROOT/'scripts/train_act.py'),str(directory)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while process.poll() is None:
                time.sleep(5)
                progress=directory/'progress.json'
                if progress.exists():print(progress.read_text().replace('\n',' '),flush=True)
            if process.returncode:raise RuntimeError('Training failed; see '+str(directory/'train.log'))
        finally:
            if process.poll() is None:
                import os,signal
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    models=[m for m in learning.models() if m['job_id']==job_id]
    expected={f'{n:06d}' for n in (params['save_freq'],steps)}
    if not expected.issubset({m['step'] for m in models}):raise ValueError('Checkpoint is missing')
    codec=ReferenceCodec(first['lerobot_calibration'])
    checks=[]
    for model in models:
        checkpoint,_=learning.model(model['id']);worker=PolicyWorker(checkpoint,'cuda',directory/f"reload_{model['step']}.log")
        try:
            wait_for(worker,lambda:worker.ready,90)
            latencies=[]
            # Go beyond the 100-action queue boundary, so a second forward pass is tested.
            for index,row in enumerate(rows[:params['n_action_steps']+2]):
                normalized=codec.normalize([round(z+q*s) for z,q,s in zip(first['calibration']['zero_ticks'],row['state'],first['calibration']['ticks_per_rad'])])
                frames={k:dict(jpeg=(store.path(ids[0])/k/Path(v['file']).name).read_bytes()) for k,v in row['cameras'].items()}
                acquired=time.monotonic();wait_for(worker,lambda:not worker.busy,5)
                if not worker.submit(normalized,frames,acquired):raise RuntimeError('Inference did not accept input')
                wait_for(worker,lambda:worker.result is not None and worker.result['acquired']==acquired,10)
                action=worker.result['action']
                if len(action)!=6 or not all(math.isfinite(v) for v in action):raise ValueError('Invalid action')
                latencies.append(worker.result['elapsed']*1000)
            checks.append(dict(model=model['id'],predictions=len(latencies),max_inference_ms=max(latencies),
                first_inference_ms=latencies[0],second_chunk_ms=latencies[params['n_action_steps']] if len(latencies)>params['n_action_steps'] else None,
                checkpoint_bytes=sum(p.stat().st_size for p in checkpoint.rglob('*') if p.is_file()),
                model_sha256=digest(checkpoint/'model.safetensors')))
            print('RELOAD_PASS',model['id'],len(latencies),flush=True)
        finally:worker.close()
    unchanged=all(Path(p).is_file() and digest(Path(p))==h for p,h in hashes.items())
    if not unchanged:raise ValueError('Source data changed during testing')
    report=dict(job_id=job_id,completed_at=time.time(),status='passed',source_episode_ids=ids,
        source_samples=sum(len(r) for _,r in records),source_duration_s=sum(m['duration'] for m,_ in records),
        source_files_verified=len(hashes),source_unchanged=unchanged,parameters=params,
        progress=json.loads((directory/'progress.json').read_text()),checkpoints=checks,
        inference_input='recorded real observations; not a live robot rollout',physical_robot_operated=False,
        physical_success_validated=False)
    atomic_json(directory/'validation_report.json',report)
    print('PASS',str(directory/'validation_report.json'),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data-root',type=Path,default=ROOT/'data')
    parser.add_argument('--episodes',type=int,nargs='+',required=True);parser.add_argument('--steps',type=int,default=100)
    args=parser.parse_args()
    with (ROOT/'.session.lock').open('a') as session:
        try:fcntl.flock(session,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Test_IL 앱을 종료한 뒤 학습·저장 검증을 실행하세요.')
        run(args.data_root,args.episodes,args.steps)
