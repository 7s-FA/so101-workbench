"""Single control owner. Recording, preview, preparation and replay are exclusive states."""
import bisect
import concurrent.futures
import copy
import json
import math
import queue
import threading
import time
from pathlib import Path

from .config import ROOT, TEACH, CAMERAS, fingerprint, camera_devices
from .store import Store, atomic_json
from .cameras import Camera, open_camera, camera_status, CameraUnavailable
from .hardware import Hardware, mapping, audit_entry, IL_ENTRY_SPEED
from .dataset import DatasetWriter
from .learning import Learning
from .policy import PolicyWorker
from .reference import ReferenceCodec
from .robot import identity,reference_config,ROBOT_TYPE,GRIPPER_MODEL,register_leader
from .safe_pose import load_safe_pose
from .calibration import CalibrationManager, register as register_calibration, catalog as calibration_catalog
from teach_io.motion_timing import VELOCITY_RAD_S
from .execution import options as execution_options, ChunkTimeline, GoalFilter, action_vector
from teach_io.entry import reference_entry_ready, EntryPoseChanged, entry_duration


def interpolate(samples, t, key):
    times = [s['t'] for s in samples]
    index = max(0, bisect.bisect_right(times, t)-1)
    if index >= len(samples)-1:
        return list(samples[-1][key]), len(samples)-1
    a,b = samples[index:index+2]
    f = max(0., min(1., (t-a['t'])/(b['t']-a['t'])))
    return [x+(y-x)*f for x,y in zip(a[key],b[key])], index


def validate_replay(meta, samples, current_fingerprint, joint_mapping):
    if meta.get('synthetic'):
        raise ValueError('Test fixtures cannot run a physical robot')
    if identity(meta) != ROBOT_TYPE or meta.get('fingerprint') != current_fingerprint or meta.get('units') != 'rad':
        raise ValueError('Robot/scene calibration differs from this episode')
    for s in samples:
        target=s['action']
        if meta.get('learning_method')=='lerobot_native_rgbd_v1':target=joint_mapping.clamp_target(target)
        joint_mapping.to_ticks(target)
    for a,b in zip(samples,samples[1:]):
        dt = b['t']-a['t']
        if dt > .2:
            raise ValueError('Recording has a control gap over 200 ms; physical replay blocked')
        if meta.get('control_profile') != 'lerobot_reference' and any(abs(x-y)>limit*dt+.004 for x,y,limit in zip(a['action'],b['action'],VELOCITY_RAD_S)):
            raise ValueError('Recorded target exceeds the existing controller velocity bounds')


class Recorder:
    def __init__(self, store):
        self.store = store
        self.queue = queue.Queue(maxsize=60)
        self.error = None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                if not self.error:
                    self.store.append(*item)
            except Exception as error:
                self.error = str(error)
            finally:
                self.queue.task_done()

    def append(self, sample, frames):
        if self.error:
            raise RuntimeError('Recording disk error: '+self.error)
        try:
            self.queue.put_nowait((sample, frames))
        except queue.Full:
            raise RuntimeError('Recording writer is too slow; stopped instead of dropping observations')

    def finish(self):
        self.queue.put(None)
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError('Recording writer has not finished; draft preserved')
        self.store.finish()
        if self.error:
            raise RuntimeError(self.error)


class StorageJob:
    """Disk work is independent of the single motor-command owner."""
    def __init__(self, op, future):
        self.op=op
        self.future=future
        self.reply=None


class Engine:
    def __init__(self, settings, data_root=None):
        self.cfg = settings
        self.store = Store(data_root or ROOT/'data')
        self.dataset_writer = DatasetWriter(self.store.root)
        self.learning = Learning(self.store)
        self.task_id = None
        self.policy_worker = None
        self.policy_job = None
        self.model_id = None
        self.model_seconds = 30.
        self.model_clock = None
        self.model_last = None
        self.model_inference_ms = None
        self.model_loading_at = None
        self.model_options=execution_options()
        self.model_timeline=None;self.model_filter=None;self.model_metrics={}
        self.model_last_received=None;self.model_prediction_count=0
        self.model_action_at=None;self.model_command_count=0;self.model_request_count=0
        self.mapping = mapping()
        self.motor_reference=reference_config()
        self.fingerprint = fingerprint()
        self.calibration=CalibrationManager()
        safe = load_safe_pose()
        self.measured = safe['raw_angles'].copy()
        self.reference_pose=self.measured.copy()
        self.reference_model_pose=safe['model_angles']
        self.reference_pose_source=safe['reference']
        self.goal = self.limit_goal(self.measured)
        self.reset_pose = self.limit_goal(self.measured)
        self.hardware = None
        self.cameras = {}
        self.mode = 'idle'
        self.message = 'Connect devices to teach. Saved episodes can be reviewed offline.'
        self.error = None
        self.latest = {}
        self.selected = None
        self.episode_meta = None
        self.samples = []
        self.play_t = 0.
        self.play_index = 0
        self.play_clock = None
        self.previewing = False
        self.record_start = None
        self.recorder = None
        self.storage_job = None
        self.storage_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.last_record_t = None
        self.pending = None
        self.after_prepare = None
        self.entry = None
        self.initial_arm = False
        self.teleop_raw = None
        self.leader_goal = None
        self.eval_stream = None
        self.eval_path = None
        self.eval_start = None
        self.eval_recorder = None
        self.eval_store = None
        self.heartbeat = 0.
        self.commands = queue.Queue()
        self.quit = threading.Event()
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.public = {}
        self.publish()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        if self.dataset_writer.status['state']=='stale':
            self.dataset_writer.request()
        self.thread.start()

    def call(self, op, **kwargs):
        result = concurrent.futures.Future()
        self.commands.put((op, kwargs, result))
        try:
            return result.result(timeout=40)
        except concurrent.futures.TimeoutError:
            result.cancel()
            raise RuntimeError('Command timed out; inspect the current state before retrying')

    def ping(self):
        self.heartbeat = time.monotonic()

    def torque_status(self):
        if not self.hardware:return dict(state='unavailable',release_required=False)
        required = bool(self.hardware.follower.armed or getattr(self.hardware,'owned',False)
                        or self.latest.get('torque_on') or self.latest.get('release_required'))
        return dict(state='unknown' if self.mode=='fault' else 'on' if required else 'off',
                    release_required=required or self.mode=='fault')

    def publish(self):
        now = time.monotonic()
        display = self.measured.copy() if self.hardware else self.reference_model_pose.copy()
        source = 'live_measured' if self.hardware else 'reference_pose'
        display_robot=ROBOT_TYPE
        if self.mode == 'preview' and self.samples:
            display,_ = interpolate(self.samples,self.play_t,'state')
            source = 'recorded_measured'
            display_robot=identity(self.episode_meta)
        table_frame=getattr(self.cameras.get('table'),'frame',None) or {}
        depth=table_frame.get('depth')
        if self.mode=='preview' and self.samples:
            depth=self.samples[self.play_index]['cameras']['table'].get('depth')
        self.public = dict(mode=self.mode, message=self.message, error=self.error,
            display_angles=display,display_source=source,display_robot=display_robot,robot_type=ROBOT_TYPE,gripper_model=GRIPPER_MODEL,
            reference_pose=self.reference_pose_source,
            motor_calibration=self.calibration.snapshot(),
            connected=self.hardware is not None, armed=bool(self.hardware and self.hardware.follower.armed),
            torque=self.torque_status(),
            device_paths=dict(follower=getattr(self.hardware,'port',None),
                leader=getattr(self.hardware,'leader',None),
                **{k:v.device for k,v in self.cameras.items()}),
            leader=bool(self.hardware and self.hardware.leader), leader_goal=self.leader_goal, measured=list(self.measured),
            leader_measured=self.latest.get('leader_angles'),
            goal=list(self.goal), joint_error_deg=[math.degrees(a-b) for a,b in zip(self.goal,self.measured)],
            selected=self.selected, play_t=self.play_t, play_index=self.play_index,
            previewing=self.previewing, duration=self.samples[-1]['t'] if self.samples else 0.,
            storage_busy=self.storage_job is not None,
            draft=self.store.draft.name if self.store.draft else None, draft_count=self.store.count,
            recording_interruption=self.store.metadata.get('interruption') if self.store.draft else None,
            recording_seconds=now-self.record_start if self.mode=='recording' else (self.last_record_t or 0.),
            cameras={name:camera_status(self.cameras.get(name),now) for name in CAMERAS},
            depth_view=dict(available=bool(depth) and (self.mode=='preview' or camera_status(self.cameras.get('table'),now)['connected']),source='recorded' if self.mode=='preview' else 'live',
                            valid_fraction=depth.get('valid_fraction') if depth else None),
            motors=self.latest.get('motors',[]), tracking=self.latest.get('tracking',{}),
            temperature=self.latest.get('temperature',{}),
            task_id=self.task_id, task_folder=str(self.store.task_folder(self.task_id)), model=dict(id=self.model_id,ready=bool(self.policy_worker and self.policy_worker.ready),
                error=self.policy_worker.error if self.policy_worker else None,
                elapsed=now-self.model_clock if self.mode=='model' and self.model_clock else 0,
                seconds=self.model_seconds,inference_ms=self.model_inference_ms,options=self.model_options,metrics=self.model_metrics),
            abandoned_drafts=self.store.abandoned(), dataset=self.dataset_writer.status.copy(), timestamp=now)

    def use_calibration(self,role,port,name,values):
        changed=self.motor_reference.get(role)!=values or self.cfg.get(role)!=port
        result=register_calibration(role,port,name,values)
        result['coordinate_changed']|=changed
        self.unload_policy()
        self.mapping=mapping();self.fingerprint=fingerprint()
        self.motor_reference=reference_config()
        self.cfg.update(follower=reference_config()['follower_port'],leader=reference_config().get('port',''))
        self.goal=self.limit_goal(self.measured);self.reset_pose=self.goal.copy()
        self.message=('팔로워' if role=='follower' else '리더')+' 보정 파일을 등록했습니다. 다시 연결하면 모터값과 대조합니다.'
        if result['coordinate_changed']:self.message+=' 관절 좌표가 바뀌었습니다. 이전 기록·모델의 보정 호환성을 확인하세요.'
        return result

    def finish_recording(self):
        if self.recorder:
            recorder, self.recorder = self.recorder, None
            recorder.finish()

    def close_evaluation(self, outcome):
        if self.eval_stream:
            self.eval_stream.close()
            self.eval_stream = None
            if self.eval_recorder:
                self.eval_recorder.finish()
                self.eval_recorder = None
                self.eval_store.draft.rename(self.eval_path/'capture')
                self.eval_store.draft = None
            atomic_json(self.eval_path/'metadata.json', dict(episode_id=self.selected, outcome=outcome,
                started=self.eval_start, ended=time.time(), original_timing=self.model_id is None, model_id=self.model_id,
                execution_options=self.model_options if self.model_id else None,execution_metrics=self.model_metrics if self.model_id else None,
                physical_grasp_success=None, note='Operator must assess grasp from the real cameras.'))

    def hold(self):
        faulted = self.mode == 'fault'
        if self.mode == 'recording':
            self.mode = 'holding'
            self.finish_recording()
        if self.pending:
            self.pending.cancel()
            self.pending = None
        # Cancel the USB worker's initial interpolation as well as the UI state.
        if self.hardware and self.hardware.follower.armed and callable(getattr(self.hardware,'hold',None)):
            self.latest=self.hardware.hold()
            self.measured=list(self.latest['angles'])
        self.close_evaluation('stopped')
        self.unload_policy()
        self.previewing = False
        self.goal = self.limit_goal(self.measured)
        self.mode = ('fault' if faulted else
                     'holding' if self.hardware and self.hardware.follower.armed else 'idle')
        self.entry = None
        self.after_prepare = None

    def select(self, number):
        if self.mode not in ('idle','holding','preview'):
            raise ValueError('Stop the active operation before selecting another episode')
        self.previewing = False
        self.episode_meta, self.samples = self.store.episode(number)
        self.selected = number
        self.play_t = 0.
        self.play_index = 0

    def prepare(self, goal, next_mode):
        if not self.hardware:
            raise ValueError('Connect the follower first')
        if fingerprint()!=self.fingerprint:
            raise ValueError('Calibration or scene changed while running; restart before physical control')
        self.mapping.validate_observation(self.measured)
        goal=self.limit_goal(goal)
        self.mapping.to_ticks(goal)
        self.after_prepare = next_mode
        self.goal = self.limit_goal(self.measured)
        self.entry = dict(start=self.measured.copy(), goal=list(goal))
        self.pending = self.pool.submit(audit_entry, self.measured.copy(), list(goal))
        self.mode = 'checking'
        self.message = '시작 자세와 관절 범위를 확인하고 있습니다. 시뮬레이션은 확인용입니다.'

    def begin_storage(self, op, data):
        if self.storage_job:raise ValueError('기록을 처리하고 있습니다. 잠시 기다려 주세요.')
        if op=='stop_record' and self.mode!='recording':raise ValueError('Not recording')
        if op in ('discard','reset') and self.mode not in ('idle','holding','teleop','recording','preview','fault'):
            raise ValueError('현재 작업을 멈춘 뒤 녹화를 버려 주세요.')
        if not self.store.draft:raise ValueError('저장하거나 버릴 녹화가 없습니다.')
        if op=='save' and self.store.metadata.get('interruption'):
            raise ValueError('카메라 문제로 중단된 녹화는 저장할 수 없습니다. 녹화를 버리고 다시 시작해 주세요.')
        result=data.get('result')
        if op=='save' and result not in ('Success','Failure'):raise ValueError('Choose Success or Failure')
        if self.mode!='fault':self.error=None
        if self.mode=='recording':self.mode='teleop'
        recorder,self.recorder=self.recorder,None
        def work():
            if recorder:recorder.finish()
            else:self.store.finish()
            if op=='save':return self.store.save(result)
            if op in ('discard','reset'):self.store.discard()
            return dict(ok=True)
        self.storage_job=StorageJob(op,self.storage_pool.submit(work))
        self.message='기록을 처리하고 있습니다. 팔 조작은 계속 유지됩니다.'
        return self.storage_job

    def finish_storage(self, job):
        try:
            result=job.future.result()
            if job.op=='save':
                self.record_start=None;self.last_record_t=None
                self.dataset_writer.request()
                message=f"기록 {result['id']:04d}을 저장했습니다. 다음 동작을 준비한 뒤 녹화 시작을 누르세요."
            elif job.op in ('discard','reset'):
                self.record_start=None;self.last_record_t=None
                message='이번 녹화를 버렸습니다. 저장된 기록은 그대로입니다.'
            elif self.store.metadata.get('interruption'):
                message='카메라 문제로 녹화가 중단되었습니다. 저장할 수 없습니다. 녹화를 버린 뒤 다시 시작해 주세요.'
            else:message='녹화를 끝냈습니다. 저장하거나 버릴 수 있습니다. 팔 조작은 계속 유지됩니다.'
            # Finishing disk work must never resume motion or conceal a motor fault.
            if self.mode!='fault':self.message=message
            return result
        finally:self.storage_job=None

    def poll_storage(self):
        job=self.storage_job
        if job is None or not job.future.done():return
        try:
            result=self.finish_storage(job)
            if job.reply and not job.reply.done():job.reply.set_result(result)
        except Exception as error:
            if self.mode!='fault':self.error=str(error);self.message='기록 처리 실패: '+str(error)
            if job.reply and not job.reply.done():job.reply.set_exception(error)

    def operation(self, op, data, *, deferred=False):
        if self.calibration.status['busy'] and op not in ('calibration_command','stop'):
            raise ValueError('모터 보정을 완료하거나 취소한 뒤 사용하세요.')
        if op=='calibration_command':
            self.calibration.command(data.get('action'));return dict(ok=True)
        if op in ('calibration_start','calibration_select','calibration_apply'):
            if self.mode not in ('idle','holding','preview','fault') or self.store.draft or self.storage_job:
                raise ValueError('현재 동작과 녹화를 끝낸 뒤 보정하세요.')
            if self.learning.process and self.learning.process.poll() is None:
                raise ValueError('학습을 끝내거나 중단한 뒤 모터 보정을 변경하세요.')
            role,port,name=data.get('role'),data.get('port'),data.get('name')
            values=None
            if op in ('calibration_select','calibration_apply'):
                values=data.get('calibration')
                if data.get('file_id'):
                    row=next((r for r in calibration_catalog() if r['id']==data['file_id']),None)
                    if row is None:raise ValueError('선택한 보정 파일을 다시 확인하세요.')
                    values=row['calibration'];name=row['name']
                from .robot import check_calibration
                values=check_calibration(values)
                if op=='calibration_select':
                    if self.hardware:raise ValueError('장치 연결을 해제한 뒤 보정 파일을 선택하세요.')
                    return self.use_calibration(role,port,name,values)
            if data.get('supported') is not True:raise ValueError('팔과 물체를 받친 뒤 보정을 시작하세요.')
            if time.monotonic()-self.heartbeat>2:raise ValueError('화면 연결을 확인한 뒤 시작하세요.')
            # Validation before any release/disconnect; no second USB owner.
            from .calibration import validate_selection
            validate_selection(role,port,name)
            self.disconnect(release=True)
            self.calibration.start(role,port,name,preset=values)
            self.mode='idle';self.error=None;self.message='모터 보정 중 · 손으로 자세를 맞추세요.'
            return dict(ok=True)
        if self.storage_job and op in ('record','save','stop_record','discard','reset','task_select','task_save','task_assign','replay','model_run','delete_episode','change_result','include'):
            raise ValueError('기록을 처리하고 있습니다. 잠시 기다려 주세요.')
        if op in ('save','stop_record','discard','reset'):
            job=self.begin_storage(op,data)
            return job if deferred else self.finish_storage(job)
        if self.mode != 'fault':self.error = None
        if op=='replay' and self.learning.process and self.learning.process.poll() is None:
            raise ValueError('기록의 실물 재실행은 학습을 중단한 뒤 사용하세요. 새 기록 수집은 가능합니다.')
        if op in ('teleop','replay','connect') and self.policy_worker:self.unload_policy()
        if op in ('task_save','task_select','task_assign','train_start','train_resume','train_stop','model_load','model_run','model_delete','model_settings'):
            if op!='train_stop' and self.mode not in ('idle','holding','preview') and not (self.mode=='teleop' and (op=='task_select' or (op=='task_save' and not data.get('id')))):
                raise ValueError('현재 작업을 멈춘 뒤 변경해 주세요.')
            if op=='task_save':
                task=self.learning.tasks.save(data['name'],data.get('id'))
                if data.get('id'):self.dataset_writer.request()
                return task
            if op=='task_select':
                if self.store.draft:raise ValueError('현재 녹화를 저장하거나 버린 뒤 폴더를 바꾸세요.')
                if data.get('id'):self.learning.tasks.get(data['id'])
                if data.get('id'):self.store.ensure_task_robot(data['id'],ROBOT_TYPE)
                self.task_id=data.get('id') or None
            elif op=='task_assign':
                meta=self.learning.tasks.assign(int(data['id']),data.get('task_id'))
                self.dataset_writer.request()
                if self.selected==meta['id']:self.episode_meta=meta
            elif op=='train_start':
                if self.torque_status()['release_required']:raise ValueError('모터 힘을 끄고 학습을 시작하세요. 학습 중에는 GPU를 사용합니다.')
                self.unload_policy()
                self.store.ensure_task_robot(data['task_id'],ROBOT_TYPE)
                return self.learning.start(data['task_id'],data.get('params',{}),data['name'])
            elif op=='train_stop':self.learning.stop()
            elif op=='model_delete':
                if data.get('id')==self.model_id:raise ValueError('준비하거나 실행 중인 모델입니다. 정지 (힘 유지)를 누른 뒤 삭제하세요.')
                result=self.learning.delete_model(data.get('id'),data.get('confirmed',False))
                self.message=f"{result['name']} · {int(result['step']):,}회 모델을 삭제했습니다."
                if result['history_deleted']:self.message+=' 학습 기록과 로그도 삭제했습니다.'
                return result
            elif op=='train_resume':
                if self.torque_status()['release_required']:raise ValueError('모터 힘을 끄고 이어 학습을 시작하세요.')
                _,job=self.learning.model(data['model_id'])
                if identity(job)!=ROBOT_TYPE:raise ValueError('SO-101 기록으로 학습한 모델을 선택하세요.')
                self.unload_policy()
                return self.learning.resume(data['model_id'],data['additional_steps'],data.get('include_new',False),data.get('name'),data.get('keep_latest_only',True))
            elif op=='model_settings':
                return self.learning.save_execution_options(data['id'],data.get('options',{}))
            elif op=='model_load':
                if self.learning.process and self.learning.process.poll() is None:raise ValueError('학습을 끝내거나 중단한 뒤 모델을 준비하세요.')
                if self.torque_status()['release_required']:raise ValueError('모터 힘을 끈 상태에서 모델을 준비하세요.')
                checkpoint,job=self.learning.model(data['id'])
                if identity(job)!=ROBOT_TYPE:raise ValueError('이 모델은 이전 OMX용입니다. SO-101 기록으로 학습한 모델을 선택하세요.')
                device=data.get('device','cuda')
                if device not in ('cpu','cuda'):raise ValueError('모델 실행 장치를 확인하세요.')
                config=self.learning.save_execution_options(data['id'],data['options']) if 'options' in data else self.learning.execution_options(checkpoint,job)
                self.unload_policy();self.model_id=data['id'];self.policy_job=job;self.model_options=config
                self.model_metrics={};self.model_inference_ms=None
                self.policy_worker=PolicyWorker(checkpoint,device,self.store.root/'training'/job['id']/'inference.log',config)
                self.model_loading_at=time.monotonic();self.message='모델을 준비하고 있습니다. 준비 중에는 팔이 움직이지 않습니다.'
            elif op=='model_run':
                if not self.policy_worker or not self.policy_worker.ready:raise ValueError('모델 준비를 먼저 완료하세요.')
                if self.policy_job['synthetic']:raise ValueError('가상 시험 기록으로 학습한 모델은 실물에서 실행할 수 없습니다.')
                if identity(self.policy_job)!=ROBOT_TYPE or self.policy_job['fingerprint']!=self.fingerprint:raise ValueError('모델의 보정 정보와 현재 로봇 설정이 다릅니다.')
                if self.store.draft:raise ValueError('현재 녹화를 저장하거나 버린 뒤 실행하세요.')
                if data.get('workspace_clear') is not True:raise ValueError('실물 작업 공간을 확인하세요.')
                seconds=float(data.get('seconds',30))
                if not math.isfinite(seconds) or not 1<=seconds<=600:raise ValueError('실행 시간은 1~600초로 입력하세요.')
                frames=self.frames()
                if {k:v['shape'] for k,v in frames.items()}!=self.policy_job['camera_shapes']:raise ValueError('카메라 해상도가 학습 기록과 다릅니다.')
                self.model_seconds=seconds;self.model_last=None;self.previewing=False
                self.model_codec=ReferenceCodec(self.policy_job['lerobot_calibration'])
                self.prepare(self.policy_job['start_pose'],'model')
            return dict(ok=True)
        if op == 'select':
            self.select(int(data['id']))
        elif op == 'preview':
            if self.mode not in ('idle','holding','preview') or not self.samples:
                raise ValueError('Stop the robot and select a saved episode first')
            self.previewing = not self.previewing
            if self.play_t >= self.samples[-1]['t']:
                self.play_t = 0.
            self.play_clock = time.monotonic()-self.play_t
            self.mode = 'preview'
            self.message = 'Recorded joint state and camera frames · Original timing'
        elif op == 'seek':
            if self.mode not in ('idle','holding','preview') or not self.samples:
                raise ValueError('Timeline unavailable during robot motion')
            value = float(data['t'])
            if not math.isfinite(value): raise ValueError('Invalid timeline position')
            playing = data.get('playing', self.previewing)
            if type(playing) is not bool:raise ValueError('Invalid playback state')
            self.mode = 'preview'
            self.play_t = max(0.,min(self.samples[-1]['t'],value))
            self.previewing = playing and self.play_t < self.samples[-1]['t']
            self.play_clock = time.monotonic()-self.play_t
        elif op == 'include':
            if self.mode not in ('idle','holding','preview'):raise ValueError('작업을 멈춘 뒤 기록을 수정해 주세요.')
            self.store.include(int(data['id']),data['included'])
            self.dataset_writer.request()
        elif op in ('change_result','delete_episode'):
            if self.mode not in ('idle','holding','preview'):raise ValueError('작업을 멈춘 뒤 기록을 수정해 주세요.')
            number=int(data['id'])
            if op=='change_result':
                meta=self.store.change_result(number,data['result'])
                if self.selected==number:self.episode_meta=meta
                self.message=f'기록 {number:04d}의 결과를 변경했습니다.'
            else:
                if data.get('confirmed') is not True:raise ValueError('삭제할 기록을 확인해 주세요.')
                self.store.delete(number)
                if self.selected==number:
                    self.previewing=False
                    if self.mode=='preview':self.mode='holding' if self.hardware and self.hardware.follower.armed else 'idle'
                    self.selected=None;self.episode_meta=None;self.samples=[];self.play_t=0.;self.play_index=0
                    remaining=self.store.list()
                    if remaining:
                        next_id=next((m['id'] for m in remaining if m['id']>number),remaining[-1]['id'])
                        self.select(next_id)
                self.message=f'기록 {number:04d}을 삭제했습니다. 원본은 로컬 휴지통에 보관했습니다.'
            self.dataset_writer.request()
        elif op == 'build_dataset':
            self.dataset_writer.request()
        elif op == 'register_leader':
            if self.hardware:raise ValueError('팔 연결을 해제한 뒤 보정 파일을 등록하세요.')
            result=register_leader(data.get('port'),data.get('calibration'))
            self.cfg['leader']=data['port']
            return result
        elif op == 'connect':
            if self.hardware or self.mode not in ('idle','preview'):
                raise ValueError('Disconnect existing devices first')
            self.previewing = False
            self.mode = 'idle'
            ports = [data.get('follower'),data.get('leader')]
            allowed = {str(p) for p in Path('/dev/serial/by-id').glob('*')}
            if not ports[0] or any(p and p not in allowed for p in ports):
                raise ValueError('Select connected USB serial devices')
            camera_paths = data.get('cameras',{})
            if camera_paths.get('table') and not camera_paths['table'].startswith('realsense:'):
                raise ValueError('상단 카메라에는 RealSense D435의 RGB 장치를 선택하세요.')
            if camera_paths.get('gripper','').startswith('realsense:'):
                raise ValueError('그리퍼에는 32×32 UVC 카메라 모듈을 선택하세요.')
            allowed_cameras = set(camera_devices())
            chosen = [v for v in camera_paths.values() if v]
            if len(set(chosen)) != len(chosen) or any(p not in allowed_cameras for p in chosen):
                raise ValueError('Select two distinct capture devices')
            try:
                self.hardware = Hardware(*ports)
                for name in CAMERAS:
                    if camera_paths.get(name):
                        self.cameras[name] = open_camera(camera_paths[name],self.cfg['record_fps'])
                self.latest = self.hardware.read()
                self.measured = list(self.latest['angles'])
                self.reset_pose = self.limit_goal(self.measured)
                self.goal = self.limit_goal(self.measured)
                self.error = None
                self.message = 'Connected read-only · No motor enabled'
            except Exception:
                self.disconnect(release=False)
                raise
        elif op == 'teleop':
            if self.mode not in ('idle','holding','preview') or not self.hardware or not self.hardware.leader:
                raise ValueError('Connect both arms and stop the active operation first')
            if data.get('workspace_clear') is not True:
                raise ValueError('Confirm the workspace is clear')
            if time.monotonic()-self.heartbeat>2:raise ValueError('Browser heartbeat missing')
            self.previewing = False
            self.leader_goal, self.teleop_raw = self.hardware.target()
            self.leader_goal = self.limit_goal(self.leader_goal)
            self.prepare(self.leader_goal, 'teleop')
        elif op == 'record':
            if self.mode != 'teleop':
                raise ValueError('Enable leader control before recording')
            self.learning.tasks.get(self.task_id)
            self.store.ensure_task_robot(self.task_id,ROBOT_TYPE)
            record_frames=self.frames()
            self.record_start = time.monotonic()
            self.last_record_t = None
            self.reset_pose = self.limit_goal(self.measured)
            self.store.ensure_task_robot(self.task_id,ROBOT_TYPE)
            self.store.begin(dict(robot_type=ROBOT_TYPE,gripper_model=GRIPPER_MODEL,task_id=self.task_id, task_description=self.learning.tasks.get(self.task_id)['name'] if self.task_id else 'Unclassified', fingerprint=self.fingerprint, synthetic=False, source='real_leader_follower',
                requested_fps=self.cfg['record_fps'], start_pose=list(self.reset_pose),
                calibration=self.mapping.cfg, leader_mapping=self.hardware.leader_cfg,
                lerobot_calibration=reference_config(),
                control_mode='lerobot_absolute', control_profile='lerobot_reference', leader_start_ticks=self.teleop_raw,
                action_semantics='leader_requested_before_robot_limits_rad',learning_method='lerobot_native_rgbd_v1',
                color_storage='lossless_png',
                camera_devices={k:v.device for k,v in self.cameras.items()}, camera_input='rgb',
                depth_cameras=[k for k,v in record_frames.items() if v.get('depth')],
                camera_streams={k:getattr(v,'actual',{}).copy() for k,v in self.cameras.items()},
                camera_models={k:getattr(v,'actual',{}).get('model','UVC') for k,v in self.cameras.items()},controller=self.latest.get('power_profile',{})))
            self.recorder = Recorder(self.store)
            self.mode = 'recording'
            self.message = 'SO-101 동작 · 컬러 영상 2대 · 상단 깊이 영상을 녹화하고 있습니다.'
        elif op == 'replay':
            if self.mode not in ('idle','holding','preview') or not self.samples:
                raise ValueError('Stop leader control and select an episode first')
            if data.get('workspace_clear') is not True:raise ValueError('Place housing at the recorded start and confirm workspace clear')
            if self.store.draft:raise ValueError('Save or discard the draft before replay')
            validate_replay(self.episode_meta,self.samples,self.fingerprint,self.mapping)
            self.previewing = False
            self.frames()  # Require the agreed two real views for evaluation.
            self.prepare(self.samples[0]['action'], 'replay')
        elif op == 'stop':
            if self.calibration.status['busy']:
                self.calibration.command('cancel');return dict(ok=True)
            self.hold()
            self.message = ('팔 제어가 중단된 상태입니다. 장치 연결을 해제한 뒤 다시 연결해 주세요.'
                            if self.mode == 'fault' else 'Stopped · Holding current measured pose')
        elif op == 'release_torque':
            if not self.hardware:raise ValueError('Connect the follower first')
            if data.get('supported') is not True:
                raise ValueError('Support the arm before releasing motors')
            self.hold()
            try:
                self.latest = self.hardware.release_torque()
                self.measured = list(self.latest['angles'])
                self.goal = self.limit_goal(self.measured)
                self.mode = 'idle';self.error = None
                self.message = '모터 힘을 껐습니다. 팔 연결과 카메라는 유지됩니다. 다시 조작하려면 리더 따라가기를 누르세요.'
            except Exception as error:
                self.mode='fault';self.error=str(error)
                self.message='모터 힘 해제를 완료하지 못했습니다. 팔을 받친 상태에서 다시 확인해 주세요.'
                raise
        elif op == 'disconnect':
            if self.torque_status()['release_required'] and data.get('supported') is not True:
                raise ValueError('Support the arm before releasing motors')
            self.disconnect(release=True)
        else:
            raise ValueError('Unknown operation')
        return dict(ok=True)

    def frames(self):
        if set(self.cameras) != set(CAMERAS):
            raise ValueError('Both gripper and table cameras are required for recording/replay')
        frames = {}
        for name in CAMERAS:
            try:
                frames[name] = self.cameras[name].snapshot()
                if self.cameras[name].device.startswith('realsense:') and not frames[name].get('depth'):
                    raise CameraUnavailable('D435 깊이 영상이 없습니다.')
            except CameraUnavailable as error:
                label = '그리퍼 카메라' if name=='gripper' else '작업대 카메라'
                raise CameraUnavailable(f'{label} 영상 수신 중단: {error}') from error
        ts = [v['acquired'] for v in frames.values()]
        if max(ts)-min(ts)>.1:raise CameraUnavailable('Camera timestamps differ by more than 100 ms')
        return frames

    def camera_interrupted(self, error):
        # Camera loss does not invalidate the independently owned motor link.
        # A recording ends at its last complete sample; manual control continues.
        recording = self.mode == 'recording'
        if recording:
            self.store.metadata['interruption'] = dict(reason=str(error), at=time.time(),
                seconds=self.last_record_t or 0.)
            # Drain the writer off the control thread; preserve the reason in the draft.
            self.begin_storage('stop_record', {})
        else:self.mode = 'holding'
        if not recording:self.goal = self.limit_goal(self.measured)
        if not recording:
            self.close_evaluation('camera_interrupted')
            self.unload_policy()
        self.error = None
        self.message = ('카메라 문제로 녹화가 중단되었습니다. 저장할 수 없습니다. 녹화를 버린 뒤 다시 시작해 주세요.'
                        if recording else '영상을 다시 연결하고 있습니다. 팔은 현재 자세를 유지하며 연결은 유지됩니다.')

    def disconnect(self, release):
        self.hold()
        if self.hardware:
            try:self.hardware.close(release=release)
            except Exception as error:
                self.mode='fault';self.error=str(error)
                self.message='연결 해제를 완료하지 못했습니다. 모터 힘 상태를 확인해 주세요.'
                raise
            self.hardware = None
        for c in self.cameras.values():c.close()
        self.cameras.clear()
        self.mode = 'idle'
        self.error = None
        self.latest = {}
        self.message = 'Disconnected'

    def unload_policy(self):
        if self.policy_worker:self.policy_worker.close()
        self.policy_worker=None;self.policy_job=None;self.model_id=None;self.model_clock=None;self.model_loading_at=None
        self.model_timeline=None;self.model_filter=None

    def update_model(self, now, dt):
        worker=self.policy_worker
        if worker.error:raise RuntimeError(worker.error)
        if now-self.model_clock>=self.model_seconds:
            self.close_evaluation('time_limit_completed');self.hold()
            self.message='모델 실행을 끝내고 자세를 유지합니다. 물체 이동 결과는 직접 확인해 주세요.'
            return
        frames=self.frames()
        result=worker.result;boundary=False
        if result and (self.model_last_received is None or result['acquired']>self.model_last_received):
            self.model_last_received=result['acquired']
            age=now-result['acquired']
            self.model_metrics['response_ms']=age*1000
            self.model_metrics['observation_age_ms']=(now-result.get('observation_at',result['acquired']))*1000
            if result.get('computed_chunk',True):
                self.model_inference_ms=result.get('compute_ms',result.get('elapsed',0)*1000)
                self.model_metrics['new_prediction_ms']=self.model_inference_ms
                self.model_prediction_count+=1
            if self.model_options['mode']=='prefetch':
                if self.model_timeline.accept(result,now):self.model_last=result['acquired']
            else:
                # Reference ACT is synchronous: inference latency is not a motor fault.
                # Use receipt freshness; observation age remains visible in metrics.
                received=result.get('received_at',result['acquired'])
                if not math.isfinite(received) or received>now+.01:
                    raise ValueError('잘못된 모델 수신 시각입니다.')
                self.model_last=received
                values=action_vector(result['action'])
                # SO-101 body actions are degrees, not normalized -100..100.
                # The codec bounds gripper %, and limit_goal bounds calibrated joints.
                self.model_target=self.limit_goal(self.mapping.to_angles(self.model_codec.follower_ticks(values)))
                boundary=result.get('computed_chunk',True)
                self.model_metrics['remaining']=result.get('remaining',0)
        if now-(self.model_last or self.model_clock)>2 and not worker.busy:
            raise RuntimeError('모델 응답이 없어 팔을 멈췄습니다.')
        self.model_metrics['waiting_for_inference']=bool(worker.busy and now-(self.model_last or self.model_clock)>.5)
        active=False
        if self.model_options['mode']=='prefetch':
            target,boundary=self.model_timeline.take(now)
            if target is not None:
                values=action_vector(target)
                self.model_target=self.limit_goal(self.mapping.to_angles(self.model_codec.follower_ticks(values)))
                self.model_action_at=now
            active=self.model_action_at is not None and now-self.model_action_at<=1.5/self.policy_job['fps']
            self.model_metrics.update(self.model_timeline.metrics())
        else:active=self.model_last is not None and now-self.model_last<=.5
        if active:
            if self.model_filter is None:self.model_filter=GoalFilter(self.goal,self.model_options)
            candidate=self.model_filter.apply(self.model_target,now,dt,boundary)
            self.goal=self.limit_goal(candidate)
            self.model_filter.position=self.goal.copy()
            self.model_command_count+=1
        # No fresh target means hold the last command; never advance a stale path.
        self.model_metrics['prediction_hz']=self.model_prediction_count/max(.001,now-self.model_clock)
        self.model_metrics['command_hz']=self.model_command_count/max(.001,now-self.model_clock)
        self.model_metrics['request_hz']=self.model_request_count/max(.001,now-self.model_clock)
        need=self.model_timeline.should_request() if self.model_options['mode']=='prefetch' else True
        if need and now-self.model_submit_at>=1/self.policy_job['fps'] and not worker.busy:
            ticks=[round(z+a*s) for z,a,s in zip(self.mapping.zero,self.measured,self.mapping.scale)]
            worker.submit(self.model_codec.normalize(ticks),frames,now);self.model_submit_at=now
            self.model_request_count+=1

    def limit_goal(self, q):
        return self.mapping.clamp_target(q)

    def update(self, now, dt):
        self.calibration.poll()
        calibration=self.calibration.status
        if calibration['busy']:
            if now-self.heartbeat>2:self.calibration.command('cancel')
        elif calibration.get('saved') and not self.calibration.applied:
            self.calibration.applied=True
            if not calibration['error']:
                try:
                    self.use_calibration(calibration['role'],calibration['port'],calibration['name'],calibration['saved']['calibration'])
                    calibration['message']=self.message
                except Exception as error:
                    calibration['error']=str(error);calibration['message']='모터 저장은 끝났지만 파일 등록에 실패했습니다: '+str(error)
        if self.policy_worker and self.policy_worker.ready and self.model_loading_at:
            self.model_loading_at=None;self.message='모델 준비 완료 · 실행 조건은 모델 실행 탭에서 확인하세요.'
        if self.policy_worker and not self.policy_worker.ready and self.model_loading_at and now-self.model_loading_at>120:
            self.unload_policy();self.message='모델 준비 시간이 초과됐습니다. 학습 탭의 로그를 확인하세요.'
        if self.mode=='model':self.update_model(now,dt)
        if self.mode in ('teleop','recording'):
            raw_goal, self.teleop_raw = self.hardware.target()
            self.record_action=list(getattr(self.hardware,'requested_action',raw_goal))
            limited = self.limit_goal(raw_goal)
            self.leader_goal = limited.copy()
            self.goal = limited.copy()  # LeRobot sends the bounded absolute goal directly.
            self.mapping.to_ticks(self.goal)
        if self.mode == 'replay':
            self.frames()
            if dt>.15:raise RuntimeError('Replay scheduling stalled; stopped rather than skipping queued motion')
            self.play_t = min(self.samples[-1]['t'],now-self.play_clock)
            self.goal,self.play_index = interpolate(self.samples,self.play_t,'action')
            self.goal=self.limit_goal(self.goal)
        if self.hardware and self.mode != 'fault':
            if self.mode in ('teleop','recording','replay','model','checking','preparing') and now-self.heartbeat>2:
                raise RuntimeError('Browser connection lost; motion stopped')
            if self.hardware.follower.armed:
                self.latest = self.hardware.update(self.goal, active=self.mode in ('recording','replay','model'))
            else:
                self.latest = self.hardware.read()
            self.measured = list(self.latest['angles'])
            if self.hardware.leader and self.mode in ('idle','holding','preview'):
                leader, self.teleop_raw = self.hardware.target()
                self.leader_goal = self.limit_goal(leader)
        if self.mode in ('checking','preparing') and self.after_prepare == 'teleop':
            leader, self.teleop_raw = self.hardware.target()
            self.leader_goal = self.limit_goal(leader)
        if self.mode == 'checking' and self.pending and self.pending.done():
            audit = self.pending.result()
            self.pending = None
            if max(abs(a-b) for a,b in zip(self.measured,audit['start']))>math.radians(.5):
                self.pending = self.pool.submit(audit_entry,self.measured.copy(),audit['goal'])
                return  # Refresh the unpowered snapshot without latching a fault.
            self.initial_arm = not self.hardware.follower.armed
            if self.initial_arm:
                try:self.hardware.arm(audit['goal'], [audit['goal']], audit)
                except EntryPoseChanged:
                    self.latest = self.hardware.read()
                    self.measured = list(self.latest['angles'])
                    self.goal = self.limit_goal(self.measured)
                    self.pending = self.pool.submit(audit_entry,self.measured.copy(),audit['goal'])
                    return
                self.goal = list(audit['goal'])
            self.entry = dict(audit, began=now, duration=entry_duration(audit['start'],audit['goal'],IL_ENTRY_SPEED))
            self.mode = 'preparing'
            self.message = 'Moving slowly to the start pose; this is not part of the episode'
        if self.mode == 'preparing':
            if self.entry.get('following_leader'):
                ready = True  # Worker finished its fixed entry before catch-up.
            elif not self.initial_arm:
                u = min(1., (now-self.entry['began'])/self.entry['duration'])
                s = u*u*u*(10+u*(-15+6*u))
                self.goal = self.limit_goal([a+(b-a)*s for a,b in zip(self.entry['start'],self.entry['goal'])])
                ready = u>=1
            else:
                ready = self.latest.get('aligned',False)
            if not self.entry.get('following_leader') and now-self.entry['began']>self.entry['duration']+25:
                raise RuntimeError('Start pose did not settle')
            if ready:
                if self.after_prepare == 'teleop':
                    if not self.entry.get('following_leader'):
                        self.goal = self.entry['goal'].copy()
                        self.entry['following_leader'] = True
                    # Leader motion during preparation is expected. Finish the
                    # worker's immutable entry first, then approach its newest
                    # target without a discontinuous first teleop command.
                    step = IL_ENTRY_SPEED*max(0.,min(dt,.05))
                    self.goal = self.limit_goal([a+max(-step,min(step,b-a))
                                                for a,b in zip(self.goal,self.leader_goal)])
                    self.message = '움직인 조작용 팔의 위치를 천천히 따라잡고 있습니다.'
                    if max(abs(a-b) for a,b in zip(self.goal,self.leader_goal))<1e-9:
                        self.mode = 'teleop'
                        self.message = 'Leader control enabled · Absolute workshop joint targets'
                elif self.after_prepare == 'model':
                    self.mode='model';self.model_clock=now;self.model_submit_at=0.;self.model_target=self.goal.copy()
                    self.model_last=None;self.model_last_received=None;self.model_action_at=None
                    self.model_metrics={};self.model_command_count=0;self.model_request_count=0;self.model_prediction_count=0
                    self.model_filter=GoalFilter(self.goal,self.model_options)
                    self.model_timeline=ChunkTimeline(self.policy_job['fps'],self.model_options,getattr(self.policy_worker,'ready_info',{}).get('warm_inference_ms',0.))
                    self.eval_start=time.time();self.eval_path=self.store.root/'evaluations'/f'model_{time.time_ns()}'
                    self.eval_path.mkdir();self.eval_stream=(self.eval_path/'tracking.jsonl').open('w')
                    self.eval_store=Store(self.eval_path);self.eval_store.begin(dict(kind='model_evaluation',model_id=self.model_id,fingerprint=self.fingerprint))
                    self.eval_recorder=Recorder(self.eval_store);self.message='학습한 동작을 실물에서 실행 중입니다. 시뮬레이션은 실제 각도를 보여줍니다.'
                elif self.after_prepare == 'replay':
                    self.mode = 'replay'
                    self.play_clock = now
                    self.play_t = 0.
                    self.eval_start = time.time()
                    self.eval_path = self.store.root/'evaluations'/f'episode_{self.selected:06d}_{time.time_ns()}'
                    self.eval_path.mkdir()
                    self.eval_stream = (self.eval_path/'tracking.jsonl').open('w')
                    self.eval_store = Store(self.eval_path)
                    self.eval_store.begin(dict(kind='evaluation',source_episode=self.selected,fingerprint=self.fingerprint))
                    self.eval_recorder = Recorder(self.eval_store)
                    self.message = 'Physical replay · Original timestamp schedule'
                else:
                    self.mode = 'holding'
                    self.message = 'Arm returned · Reposition the housing by hand'
        if self.mode == 'recording':
            frames = self.frames()
            t = now-self.record_start
            if self.last_record_t is not None and t-self.last_record_t>.2:
                raise RuntimeError('Control capture stalled; recording stopped without filling the gap')
            # Match lerobot_record: action is the requested leader action;
            # command separately records the bounded target actually sent.
            tracking = self.latest.get('tracking',{})
            self.recorder.append(dict(t=t, acquired=now, state=self.measured.copy(),
                action=list(self.record_action),
                command=list(tracking.get('command_angles',self.goal)), leader_ticks=self.teleop_raw,
                motors=self.latest.get('motors',[])), frames)
            self.last_record_t = t
        if self.mode == 'replay':
            self.eval_stream.write(json.dumps(dict(t=self.play_t,measured=self.measured,goal=self.goal,
                tracking=self.latest.get('tracking',{})))+'\n')
            tracking = self.latest.get('tracking',{})
            self.eval_recorder.append(dict(t=self.play_t,acquired=time.monotonic(),state=self.measured.copy(),
                action=self.goal.copy(),command=list(tracking.get('command_angles',self.goal))),self.frames())
            if self.play_t>=self.samples[-1]['t']:
                self.close_evaluation('command_sequence_completed')
                self.mode = 'holding'
                self.message = 'Recorded commands completed · Check physical placement; grasp success is not assumed'
        if self.mode=='model':
            t=now-self.model_clock
            self.eval_stream.write(json.dumps(dict(t=t,measured=self.measured,goal=self.goal,inference_ms=self.model_inference_ms,execution_metrics=self.model_metrics))+'\n')
            self.eval_recorder.append(dict(t=t,acquired=now,state=self.measured.copy(),action=self.goal.copy(),
                command=list(self.latest.get('tracking',{}).get('command_angles',self.goal))),self.frames())
        if self.previewing:
            self.play_t = min(self.samples[-1]['t'],now-self.play_clock)
            if self.play_t>=self.samples[-1]['t']:self.previewing=False
        if self.mode == 'preview' and self.samples:
            _,self.play_index = interpolate(self.samples,self.play_t,'state')

    def run(self):
        last = time.monotonic()
        while not self.quit.is_set():
            began = time.monotonic()
            try:
                self.poll_storage()
                for _ in range(4):
                    try:op,data,future = self.commands.get_nowait()
                    except queue.Empty:break
                    if not future.set_running_or_notify_cancel():continue
                    try:
                        result=self.operation(op,data,deferred=True)
                        if isinstance(result,StorageJob):result.reply=future
                        else:future.set_result(result)
                    except Exception as error:future.set_exception(error)
                # Operations can begin recording after the loop's pacing clock.
                # Sampling again prevents the first observation predating record_start.
                now = time.monotonic()
                self.update(now, now-last)
                last = now
            except CameraUnavailable as error:
                try:self.camera_interrupted(error)
                except Exception as cleanup_error:
                    self.error=str(cleanup_error);self.mode='fault'
                    self.message='Stopped: '+str(cleanup_error)
            except Exception as error:
                if self.hardware and getattr(self.hardware.follower,'latest',{}).get('ok') is False:
                    self.latest=self.hardware.follower.latest
                self.error = str(error)
                self.message = 'Stopped: '+str(error)
                try:self.hold()
                except Exception as cleanup_error:self.message += '; '+str(cleanup_error)
                # Keep the owned session identity for explicit release; do not reconnect
                # or issue more RPCs after a fault, even if the worker has already exited.
                self.mode = 'fault'
            self.publish()
            fps=self.cfg['record_fps'] if self.mode=='recording' else self.cfg.get('teleop_fps',60)
            self.quit.wait(max(0,1/fps-(time.monotonic()-began)))

    def sim_state(self):
        return self.public.copy()

    def close(self):
        self.quit.set()
        if self.thread.is_alive():self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError('Control owner did not stop; supervisor will handle shutdown')
        self.calibration.close()
        if self.storage_job:
            try:self.storage_job.future.result(timeout=6)
            except Exception:pass
            self.poll_storage()
        self.storage_pool.shutdown(wait=False,cancel_futures=False)
        self.finish_recording()
        self.close_evaluation('application_exit')
        self.disconnect(release=True)
        self.pool.shutdown(wait=False,cancel_futures=True)
        self.dataset_writer.close()
        self.learning.close()
