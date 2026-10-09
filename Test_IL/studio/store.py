"""Append-only episode records with atomic commit and stable sequential IDs."""
import fcntl
import json
import math
import os
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .config import CAMERAS, JOINTS
from .robot import identity,ROBOT_TYPE,GRIPPER_MODEL


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def validate_samples(samples):
    if len(samples) < 2:
        raise ValueError('At least two complete observations are required')
    previous = -1.
    for s in samples:
        t = s['t']
        if not isinstance(t, (float, int)) or not math.isfinite(t) or t < 0 or t <= previous:
            raise ValueError('Episode timestamps must be finite and strictly increasing')
        previous = t
        for name in ('state', 'action', 'command'):
            q = s[name]
            if len(q) != 6 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in q):
                raise ValueError(f'Invalid six-joint {name}')
        if set(s['cameras']) != set(CAMERAS):
            raise ValueError('Both camera records are required')
    return samples


def validate_folder_name(name):
    if isinstance(name,str):name=name.strip()
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 80:
        raise ValueError('폴더 이름을 1~80자로 입력하세요.')
    if name in ('.', '..') or name.startswith('.') or any(c in name for c in '/\\') or any(ord(c) < 32 for c in name):
        raise ValueError('폴더 이름에 경로 구분자나 제어 문자를 사용할 수 없습니다.')
    if len(name.encode('utf-8')) > 240:raise ValueError('폴더 이름이 너무 깁니다.')


class Store:
    def __init__(self, root):
        self.root = Path(root)
        for name in ('tasks', 'drafts', 'evaluations'):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.draft = None
        self.stream = None
        self.count = 0
        self.metadata = {}
        self.last_t = -1.
        self.migrate_layout()

    def task_folder(self, task_id=None):
        name = '미분류'
        if task_id:
            catalog = self.root / 'tasks.json'
            tasks = json.loads(catalog.read_text()) if catalog.exists() else []
            task = next((t for t in tasks if t['id'] == task_id), None)
            if task is None:raise ValueError('작업 폴더를 찾을 수 없습니다.')
            name = task['name']
        validate_folder_name(name)
        return self.root / 'tasks' / name

    def migrate_layout(self):
        legacy = sorted((self.root / 'episodes').glob('episode_*/metadata.json'))
        if not legacy:return
        with self.edit_saved():
            for file in legacy:
                meta = json.loads(file.read_text())
                parent = self.task_folder(meta.get('task_id')) / 'episodes'
                parent.mkdir(parents=True, exist_ok=True)
                destination = parent / file.parent.name
                if destination.exists():raise ValueError('같은 번호의 기록이 있어 폴더 이동을 중단했습니다.')
                file.parent.rename(destination)
            old = self.root / 'episodes'
            if old.exists() and not any(old.iterdir()):old.rmdir()

    def ensure_task_robot(self,task_id,robot_type):
        for row in self.list():
            if (row.get('task_id') or None)==(task_id or None) and identity(row)!=robot_type:
                raise ValueError('다른 로봇의 기록이 있는 폴더입니다. SO-101용 작업 폴더를 새로 만들어 주세요.')

    def begin(self, metadata):
        with self.lock:
            if self.draft:
                raise ValueError('Save or discard the current draft first')
            self.draft = self.root / 'drafts' / uuid.uuid4().hex
            self.draft.mkdir()
            for camera in CAMERAS:
                (self.draft / camera).mkdir()
            self.metadata = dict(schema_version=2, robot_type=ROBOT_TYPE,gripper_model=GRIPPER_MODEL,task='unclassified', units='rad',
                                 joints=JOINTS, created_at=time.time())
            self.metadata.update(metadata)
            atomic_json(self.draft / 'metadata.json', self.metadata)
            self.stream = (self.draft / 'samples.jsonl').open('w')
            self.count = 0
            self.last_t = -1.

    def append(self, sample, frames):
        with self.lock:
            if self.stream is None:
                raise ValueError('No active recording')
            t=sample['t']
            if not isinstance(t,(int,float)) or not math.isfinite(t) or t<0 or t <= self.last_t:
                raise ValueError('Non-monotonic observation')
            sample = dict(sample, index=self.count)
            # Frame timestamps are acquisition timestamps, not assumed equal to the control clock.
            sample['cameras'] = {}
            for name in CAMERAS:
                frame = frames[name]
                if name in self.metadata.get('depth_cameras',[]) and not frame.get('depth'):
                    raise ValueError('필수 깊이 영상이 없어 녹화를 저장할 수 없습니다.')
                from .cameras import learning_image
                extension,pixels=learning_image(frame)
                filename = f'{self.count:07d}'+extension
                (self.draft / name / filename).write_bytes(pixels)
                sample['cameras'][name] = dict(file=filename, acquired=frame['acquired'],
                                               sequence=frame['sequence'], shape=frame['shape'])
                for key in ('sensor_timestamp_ms','frame_number','timestamp_domain'):
                    if key in frame:sample['cameras'][name][key]=frame[key]
                if frame.get('depth'):
                    depth=frame['depth'];directory=self.draft/(name+'_depth');directory.mkdir(exist_ok=True)
                    depth_name=f'{self.count:07d}.png'
                    (directory/depth_name).write_bytes(depth['png'])
                    sample['cameras'][name]['depth']=dict(
                        {k:v for k,v in depth.items() if k not in ('png','preview_jpeg')},file=depth_name)
            self.stream.write(json.dumps(sample, allow_nan=False) + '\n')
            self.stream.flush()
            self.count += 1
            self.last_t = sample['t']

    def finish(self):
        with self.lock:
            if self.stream:
                self.stream.flush()
                os.fsync(self.stream.fileno())
                self.stream.close()
                self.stream = None
            if self.draft and self.metadata.get('interruption'):
                atomic_json(self.draft / 'metadata.json', self.metadata)

    def save(self, result):
        if result not in ('Success', 'Failure'):
            raise ValueError('Choose Success or Failure')
        with self.lock:
            if self.metadata.get('interruption'):
                raise ValueError('카메라 문제로 중단된 녹화는 저장할 수 없습니다. 녹화를 버리고 다시 시작해 주세요.')
            self.finish()
            if not self.draft:
                raise ValueError('No draft to save')
            samples = validate_samples(self.read_samples(self.draft))
            with (self.root / '.commit.lock').open('w') as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                # Deleted recordings keep their IDs; a new save never reuses one.
                entries = self.list()+[json.loads(p.read_text()) for p in (self.root/'trash').glob('episode_*/metadata.json')]
                number = max((v['id'] for v in entries), default=0) + 1
                meta = dict(self.metadata, id=number, result=result, included=result == 'Success',
                            samples=len(samples), duration=samples[-1]['t'],
                            max_sample_gap=max(b['t']-a['t'] for a,b in zip(samples,samples[1:])))
                atomic_json(self.draft / 'metadata.json', meta)
                parent = self.task_folder(meta.get('task_id')) / 'episodes'
                parent.mkdir(parents=True, exist_ok=True)
                self.ensure_task_robot(meta.get('task_id'),identity(meta))
                self.draft.rename(parent / f'episode_{number:06d}')
                self.draft = None
                self.count = 0
                return meta

    def discard(self):
        with self.lock:
            self.finish()
            if self.draft:
                shutil.rmtree(self.draft)
            self.draft = None
            self.count = 0

    def path(self, number):
        if type(number) is not int or number < 1:
            raise ValueError('Invalid episode ID')
        paths = list((self.root / 'tasks').glob(f'*/episodes/episode_{number:06d}'))
        if len(paths) != 1:
            raise ValueError('Episode not found' if not paths else 'Duplicate episode ID')
        return paths[0]

    def list(self):
        records = []
        for p in (self.root / 'tasks').glob('*/episodes/episode_*/metadata.json'):
            try:
                meta=json.loads(p.read_text());meta.setdefault('robot_type',identity(meta));records.append(meta)
            except FileNotFoundError:continue  # Individually moved to trash during this read.
        return sorted(records, key=lambda m: m['id'])

    @staticmethod
    def read_samples(path):
        return [json.loads(line) for line in (Path(path) / 'samples.jsonl').read_text().splitlines() if line]

    def episode(self, number):
        path = self.path(number)
        return json.loads((path / 'metadata.json').read_text()), validate_samples(self.read_samples(path))

    def include(self, number, included):
        if type(included) is not bool:
            raise ValueError('included must be boolean')
        with self.lock, self.source_snapshot():
            path = self.path(number) / 'metadata.json'
            meta = json.loads(path.read_text())
            meta['included'] = included
            atomic_json(path, meta)

    @contextmanager
    def edit_saved(self):
        # Do not rename source frames or alter labels halfway through an export.
        with self.lock, (self.root/'.lerobot-write.lock').open('a') as handle:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ValueError('학습 파일을 만드는 중입니다. 완료 후 기록을 수정해 주세요.')
            try:yield
            finally:fcntl.flock(handle,fcntl.LOCK_UN)

    def change_result(self, number, result):
        if result not in ('Success','Failure'):raise ValueError('Choose Success or Failure')
        with self.lock, self.source_snapshot():
            path=self.path(number)/'metadata.json'
            meta=json.loads(path.read_text())
            meta.update(result=result,included=result=='Success',updated_at=time.time())
            atomic_json(path,meta)
            return meta

    @contextmanager
    def source_snapshot(self):
        # Short lock for episode relocation / export publication, not video encoding.
        with (self.root/'.source.lock').open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            try:yield
            finally:fcntl.flock(handle,fcntl.LOCK_UN)

    @contextmanager
    def export_episode(self, number):
        handles={};stream=None
        try:
            with self.source_snapshot():
                try:path=self.path(number)
                except ValueError:
                    path=self.root/'trash'/f'episode_{number:06d}'
                    if not path.is_dir():raise ValueError('Episode not found')
                meta=json.loads((path/'metadata.json').read_text())
                stream=(path/'samples.jsonl').open()
                for name in CAMERAS:handles[name]=os.open(path/name,os.O_RDONLY|os.O_DIRECTORY)
            with stream:rows=validate_samples([json.loads(line) for line in stream if line.strip()])
            def open_frame(name, filename):
                return os.fdopen(os.open(Path(filename).name,os.O_RDONLY,dir_fd=handles[name]),'rb')
            yield meta,rows,open_frame
        finally:
            if stream:stream.close()
            for fd in handles.values():os.close(fd)

    def delete(self, number):
        with self.lock, self.source_snapshot(), (self.root/'.commit.lock').open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            path=self.path(number)
            trash=self.root/'trash';trash.mkdir(exist_ok=True)
            destination=trash/path.name
            if destination.exists():raise ValueError('휴지통에 같은 번호가 있어 삭제하지 않았습니다.')
            path.rename(destination)
            return destination

    def abandoned(self):
        return [p.name for p in (self.root / 'drafts').iterdir() if p.is_dir() and p != self.draft]
