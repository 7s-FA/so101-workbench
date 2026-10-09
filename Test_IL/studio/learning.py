"""Local task catalog and supervised LeRobot ACT training processes."""
import json
import math
import os
import signal
import shutil
import subprocess
import threading
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from .execution import options as execution_options
from .config import ROOT, IL_PYTHON
from .robot import identity,ROBOT_TYPE
from .store import atomic_json, validate_folder_name
from .dataset_retention import training_reference_guard

DEFAULTS = dict(steps=100000, batch_size=8, learning_rate=0.00001, chunk_size=100,
                n_action_steps=100, save_freq=10000, seed=1000, num_workers=2,
                device='cuda', pretrained_backbone=True, keep_latest_only=True)


def parameters(values):
    if not isinstance(values, dict) or set(values)-set(DEFAULTS):
        raise ValueError('알 수 없는 학습 설정입니다.')
    result = dict(DEFAULTS, **values)
    for name, lo, hi in [('steps',1,1000000),('batch_size',1,128),('chunk_size',1,300),
                         ('n_action_steps',1,300),('save_freq',1,1000000),('seed',0,2147483647),('num_workers',0,8)]:
        value=result[name]
        if type(value) is not int or not lo<=value<=hi:raise ValueError(f'{name}: {lo}~{hi} 사이 정수를 입력하세요.')
    if result['n_action_steps']>result['chunk_size']:raise ValueError('연속 실행 길이는 예측 길이보다 클 수 없습니다.')
    lr=result['learning_rate']
    if type(lr) not in (int,float) or not math.isfinite(lr) or not 1e-7<=lr<=.01:raise ValueError('학습률은 0.0000001~0.01 사이로 입력하세요.')
    if result['device'] not in ('cpu','cuda') or type(result['pretrained_backbone']) is not bool:raise ValueError('장치와 사전 학습 설정을 확인하세요.')
    if type(result['keep_latest_only']) is not bool:raise ValueError('최근 모델 보관 여부를 선택하세요.')
    return result


class Tasks:
    def __init__(self, store):self.store=store;self.path=store.root/'tasks.json'
    def list(self):
        tasks = json.loads(self.path.read_text()) if self.path.exists() else []
        return [dict(t, folder=str(self.store.task_folder(t['id']))) for t in tasks]
    def get(self, task_id):
        task=next((t for t in self.list() if t['id']==task_id),None)
        if not task:raise ValueError('작업 폴더를 먼저 만들거나 선택해 주세요.')
        return task
    def save(self, name, task_id=None):
        validate_folder_name(name)
        name=name.strip()
        if name=='미분류':raise ValueError('미분류는 기본 폴더입니다. 다른 이름을 사용하세요.')
        # Empty folder creation does not modify the exporter’s source episodes.
        # Renames still require the export lock; both serialize catalog writes
        # with the short source lock across independent Store instances.
        guard=self.store.edit_saved() if task_id else nullcontext()
        with guard, self.store.lock, self.store.source_snapshot():
            tasks=self.list()
            if any(t['name']==name and t['id']!=task_id for t in tasks):raise ValueError('같은 작업 이름이 있습니다.')
            destination=self.store.root/'tasks'/name
            old=None
            if task_id:
                task=self.get(task_id);old=self.store.task_folder(task_id)
                if old!=destination:
                    if destination.exists():raise ValueError('같은 이름의 폴더가 이미 있습니다.')
                    if old.exists():old.rename(destination)
                    else:destination.mkdir()
                for task in tasks:
                    if task['id']==task_id:task['name']=name
            else:
                if destination.exists():raise ValueError('같은 이름의 폴더가 이미 있습니다.')
                destination.mkdir()
                task_id=uuid.uuid4().hex;tasks.append(dict(id=task_id,name=name))
            try:
                atomic_json(self.path,[{k:v for k,v in t.items() if k!='folder'} for t in tasks])
            except BaseException:
                if old and old!=destination:destination.rename(old)
                elif not old:destination.rmdir()
                raise
        return self.get(task_id)
    def assign(self, number, task_id):
        if task_id:self.get(task_id)
        # Export readers retain open directory handles; moving a record does
        # not invalidate those handles. Publication rechecks its task metadata.
        with self.store.lock, self.store.source_snapshot():
            old=self.store.path(number)
            source_meta=json.loads((old/'metadata.json').read_text())
            self.store.ensure_task_robot(task_id,identity(source_meta))
            parent=self.store.task_folder(task_id)/'episodes';parent.mkdir(parents=True,exist_ok=True)
            destination=parent/old.name
            if destination!=old:
                if destination.exists():raise ValueError('같은 번호의 기록이 이미 있습니다.')
                old.rename(destination)
            meta=json.loads((destination/'metadata.json').read_text())
            meta.update(task_id=task_id or None,updated_at=time.time())
            try:atomic_json(destination/'metadata.json',meta)
            except BaseException:
                if destination!=old:destination.rename(old)
                raise
        return meta


class Learning:
    def __init__(self, store):
        self.store=store;self.tasks=Tasks(store);self.root=store.root/'training';self.root.mkdir(exist_ok=True)
        self.process=None;self.thread=None;self.active=None
        # An earlier host process may have exited mid-training. Never pretend it completed.
        for path in self.root.glob('*/job.json'):
            job=json.loads(path.read_text())
            if job['state'] in ('running','starting','stopping'):
                job.update(state='interrupted',message='프로그램이 종료되어 학습이 중단됐습니다.');atomic_json(path,job)
    def jobs(self):
        result=[]
        for path in sorted(self.root.glob('*/job.json'),reverse=True):
            job=json.loads(path.read_text());progress=path.parent/'progress.json'
            if progress.exists():job['progress']=json.loads(progress.read_text())
            if job['state'] in ('starting','running') and (path.parent/'stop_requested').exists():
                job.update(state='stopping',message='현재 학습 상태를 저장하고 중단하고 있습니다.')
            result.append(job)
        return result
    def models(self):
        models=[]
        jobs=self.jobs()
        active=[j for j in jobs if j['state'] in ('starting','running','stopping') or
                (j['id']==self.active and self.process and self.process.poll() is None)]
        for job in jobs:
            for path in sorted((self.root/job['id']/'output/checkpoints').glob('*/pretrained_model')):
                if not path.parent.name.isdecimal() or path.parent.is_symlink() or not (path/'model.safetensors').exists():continue
                if not (path/'policy_preprocessor.json').exists() or not (path/'policy_postprocessor.json').exists():continue
                # Publish only completely written checkpoints. Training marks this after save returns.
                if not (path.parent/'studio_ready.json').exists():continue
                model_id=job['id']+':'+path.parent.name
                protected=any(j['id']==job['id'] or j.get('resume_model_id')==model_id or
                    (j.get('resume_checkpoint') and Path(j['resume_checkpoint']).resolve()==path.resolve()) for j in active)
                models.append(dict(id=model_id,job_id=job['id'],step=path.parent.name,
                    name=job['name'],task_id=job['task']['id'],task_name=job['task']['name'],state=job['state'],
                    robot_type=identity(job),physical_allowed=not job['synthetic'] and identity(job)==ROBOT_TYPE,created_at=job['created_at'],
                    resumable=self.resumable(path.parent),default_execution_options=execution_options({},job.get('params',{})),execution_options=self.execution_options(path,job),chunk_size=job.get('params',{}).get('chunk_size',100),
                    delete_blocked_reason='학습에 사용 중인 모델입니다. 학습을 저장하고 중단한 뒤 삭제하세요.' if protected else None))
        return models
    @staticmethod
    def execution_options(checkpoint, job):
        path=checkpoint.parent/'execution_options.json'
        return execution_options(json.loads(path.read_text()) if path.exists() else {},job.get('params',{}))

    def save_execution_options(self, model_id, values):
        checkpoint,job=self.model(model_id)
        config=execution_options(values,job.get('params',{}))
        atomic_json(checkpoint.parent/'execution_options.json',config)
        return config

    def delete_model(self, model_id, confirmed=False):
        if confirmed is not True:raise ValueError('삭제할 모델을 확인해 주세요.')
        models=self.models()
        model=next((m for m in models if m['id']==model_id),None)
        if not model:raise ValueError('삭제할 모델이 없습니다. 목록을 다시 확인하세요.')
        if model['delete_blocked_reason']:raise ValueError(model['delete_blocked_reason'])
        job_dir=self.root/model['job_id']
        checkpoint=job_dir/'output/checkpoints'/model['step']
        if not checkpoint.resolve().is_relative_to(self.root.resolve()):raise ValueError('잘못된 모델 경로입니다.')
        history_deleted=not any(m['job_id']==model['job_id'] and m['id']!=model_id for m in models)
        if history_deleted:
            # Remove the whole finished training run from both catalogs atomically.
            # A nested tombstone is outside jobs()'s one-level scan, even if cleanup fails.
            trash=self.root/'.deleted';trash.mkdir(exist_ok=True)
            destination=trash/(model['job_id']+'-'+uuid.uuid4().hex)
            job_dir.rename(destination)
        else:
            destination=job_dir/('.deleted-model-'+model['step']+'-'+uuid.uuid4().hex)
            checkpoint.rename(destination)
            last=checkpoint.parent/'last'
            if last.is_symlink() and last.resolve()==checkpoint.resolve():
                remaining=[p for p in checkpoint.parent.iterdir() if p.name.isdecimal() and
                           not p.is_symlink() and (p/'studio_ready.json').is_file()]
                last.unlink()
                if remaining:last.symlink_to(max(remaining,key=lambda p:int(p.name)).name)
        try:shutil.rmtree(destination)
        except OSError as error:
            raise ValueError('목록에서 제거했지만 파일 정리를 완료하지 못했습니다: '+str(destination)) from error
        return dict(deleted=model_id,name=model['name'],step=model['step'],history_deleted=history_deleted)
    def model(self, model_id):
        model=next((m for m in self.models() if m['id']==model_id),None)
        if not model:raise ValueError('저장이 끝난 모델을 선택하세요.')
        job=json.loads((self.root/model['job_id']/'job.json').read_text())
        return self.root/model['job_id']/'output/checkpoints'/model['step']/'pretrained_model',job
    @staticmethod
    def resumable(checkpoint):
        required=['training_step.json','optimizer_param_groups.json','optimizer_state.safetensors','rng_state.safetensors']
        return all((checkpoint/'training_state'/name).is_file() for name in required)

    def ensure_idle(self):
        if (self.process and self.process.poll() is None) or (self.thread and self.thread.is_alive()):
            raise ValueError('이미 학습 중입니다. 저장하고 중단한 뒤 이어 학습해 주세요.')

    def snapshot(self, task_id):
        with self.store.edit_saved():
            current=self.store.task_folder(task_id)/'lerobot'
            if not current.exists():raise ValueError('먼저 기록을 저장하고 학습 파일 만들기를 완료하세요.')
            dataset=current.resolve()
            manifest=json.loads((dataset/'studio_manifest.json').read_text())
            mapping={e['studio_id']:e for e in manifest['episodes']}
            records=[m for m in self.store.list() if m.get('task_id')==task_id and m.get('included')]
            if not records:raise ValueError('이 작업에 학습에 사용할 기록이 없습니다. 작업 기록에서 폴더와 체크를 확인하세요.')
            if any(m['id'] not in mapping for m in records):raise ValueError('학습 파일 만들기가 끝난 뒤 다시 시작하세요.')
            if len({identity(m) for m in records})!=1:raise ValueError('서로 다른 로봇의 기록은 함께 학습할 수 없습니다.')
            fingerprints={m.get('fingerprint') for m in records}
            if len(fingerprints)!=1 or None in fingerprints:raise ValueError('보정 정보가 같은 기록끼리 학습해 주세요.')
            if any(m.get('calibration')!=records[0].get('calibration') or m.get('lerobot_calibration')!=records[0].get('lerobot_calibration') for m in records):raise ValueError('모터 보정이 서로 다른 기록은 함께 학습할 수 없습니다.')
            if not records[0].get('calibration') or not records[0].get('lerobot_calibration'):raise ValueError('기록에 모터 보정 정보가 없습니다.')
            if any(mapping[m['id']].get('source_fingerprint')!=m['fingerprint'] for m in records):
                raise ValueError('학습 파일의 보정 정보가 원본과 다릅니다. 다시 만들어 주세요.')
            first,rows=self.store.episode(records[0]['id'])
            shapes={k:v['shape'] for k,v in rows[0]['cameras'].items()}
            return dict(robot_type=identity(first),learning_method=first.get('learning_method','legacy_sent_target'),dataset=str(dataset),repo_id=manifest['repo_id'],
                episodes=[mapping[m['id']]['episode_index'] for m in records],
                studio_ids=[m['id'] for m in records],fingerprint=first['fingerprint'],
                calibration=first['calibration'],lerobot_calibration=first['lerobot_calibration'],
                camera_shapes=shapes,fps=first.get('requested_fps',30),start_pose=rows[0]['action'],
                synthetic=any(m.get('synthetic') for m in records))

    def start(self, task_id, values, name):
        self.ensure_idle()
        task=self.tasks.get(task_id);params=parameters(values)
        with training_reference_guard(self.store.root):
            return self.launch(dict(task=task,params=params,**self.snapshot(task_id)),name)

    def resume(self, model_id, additional_steps, include_new=False, name=None, keep_latest_only=True):
        with training_reference_guard(self.store.root):
            return self._resume(model_id,additional_steps,include_new,name,keep_latest_only)

    def _resume(self, model_id, additional_steps, include_new, name, keep_latest_only):
        self.ensure_idle()
        if type(additional_steps) is not int or not 1<=additional_steps<=1000000:
            raise ValueError('추가 학습 횟수는 1~1000000 사이 정수로 입력하세요.')
        if type(include_new) is not bool:raise ValueError('추가 기록 반영 여부를 선택하세요.')
        if type(keep_latest_only) is not bool:raise ValueError('최근 모델 보관 여부를 선택하세요.')
        checkpoint,parent=self.model(model_id)
        if not self.resumable(checkpoint.parent):raise ValueError('학습 재개 상태가 없는 모델입니다.')
        step=json.loads((checkpoint.parent/'training_state/training_step.json').read_text())['step']
        if type(step) is not int or step!=int(checkpoint.parent.name):raise ValueError('저장된 학습 횟수가 일치하지 않습니다.')
        fields=('dataset','repo_id','episodes','studio_ids','fingerprint','calibration','lerobot_calibration',
                'camera_shapes','fps','start_pose','synthetic')
        source={k:parent[k] for k in fields};source['robot_type']=identity(parent)
        source['learning_method']=parent.get('learning_method','legacy_sent_target')
        if include_new:
            source=self.snapshot(parent['task']['id'])
            for key in ('fingerprint','calibration','lerobot_calibration','camera_shapes','fps','synthetic'):
                if source[key]!=parent[key]:raise ValueError('기존 모델과 새 기록의 카메라·보정 설정이 다릅니다.')
        if not (Path(source['dataset'])/'meta/info.json').is_file():raise ValueError('이 모델의 학습 데이터가 없습니다.')
        params=dict(parent['params'],steps=step+additional_steps,keep_latest_only=keep_latest_only)
        return self.launch(dict(task=parent['task'],params=params,**source,
            resume_checkpoint=str(checkpoint),resume_model_id=model_id,start_step=step,
            additional_steps=additional_steps,include_new=include_new,
            normalization='lerobot_native'),name or parent['name'][:72]+' · 이어 학습')

    def launch(self, job, name):
        if not isinstance(name,str) or not 1<=len(name.strip())<=80:raise ValueError('모델 이름을 1~80자로 입력하세요.')
        job_id=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
        directory=self.root/job_id;directory.mkdir()
        job.update(id=job_id,name=name.strip(),state='starting',created_at=time.time(),message='ACT 학습 준비 중')
        atomic_json(directory/'job.json',job)
        log=(directory/'train.log').open('w')
        try:
            self.process=subprocess.Popen([str(IL_PYTHON),str(ROOT/'scripts/train_act.py'),str(directory)],
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        except Exception as error:
            job.update(state='error',message=str(error));atomic_json(directory/'job.json',job);raise
        finally:log.close()
        self.active=job_id
        self.thread=threading.Thread(target=self.wait,args=(self.process,directory),daemon=True);self.thread.start()
        return job
    def wait(self, process, directory):
        code=process.wait()
        job=json.loads((directory/'job.json').read_text())
        if job['state'] not in ('completed','cancelled','error'):
            job.update(state='error',message=f'학습이 종료됐습니다 (종료 코드 {code}). 로그를 확인하세요.')
            atomic_json(directory/'job.json',job)
    def stop(self):
        if self.process and self.process.poll() is None:
            directory=self.root/self.active
            # Worker saves a complete checkpoint between optimizer updates.
            (directory/'stop_requested').touch()
    def close(self):
        self.stop()
        if self.process:
            try:self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(self.process.pid,signal.SIGKILL)
        if self.thread:self.thread.join(timeout=2)
    def status(self):
        records=self.store.list();jobs=self.jobs()
        for job in jobs:
            available={r['id'] for r in records if r.get('task_id')==job['task']['id'] and r.get('included')}
            job['pending_episode_ids']=sorted(available-set(job['studio_ids']))
        return dict(tasks=self.tasks.list(),jobs=jobs,models=self.models(),defaults=DEFAULTS)
